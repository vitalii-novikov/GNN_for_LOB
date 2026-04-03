import argparse
import copy
import html
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


SUPPORTED_MODEL_PIPELINES = {"multigraph", "base_gnn"}


def resolve_model_pipeline_name(config_path: Path, override: Optional[str] = None) -> str:
    if override is not None and str(override).strip():
        model_pipeline = str(override).strip().lower()
        if model_pipeline not in SUPPORTED_MODEL_PIPELINES:
            supported = ", ".join(sorted(SUPPORTED_MODEL_PIPELINES))
            raise ValueError(f"Unsupported model_pipeline override: {model_pipeline}. Expected one of: {supported}")
        return model_pipeline

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_cfg = yaml.safe_load(f) or {}
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")

    if not isinstance(raw_cfg, dict):
        raise ValueError("Config root must be a mapping/object.")

    model_pipeline = str(raw_cfg.get("model_pipeline") or "multigraph").strip().lower()
    if model_pipeline not in SUPPORTED_MODEL_PIPELINES:
        supported = ", ".join(sorted(SUPPORTED_MODEL_PIPELINES))
        raise ValueError(f"Unsupported model_pipeline: {model_pipeline}. Expected one of: {supported}")
    return model_pipeline


def is_gcp_enabled(cfg: Dict[str, object], pipeline: Any) -> bool:
    if pipeline.parse_bool(cfg.get("local_run"), False):
        return False
    return bool(str(cfg.get("gcs_data_prefix") or "").strip() or str(cfg.get("gcs_run_prefix") or "").strip())


def ensure_local_required_files(required_files: List[str], data_dir: Path) -> None:
    missing = [str((data_dir / filename).resolve()) for filename in required_files if not (data_dir / filename).exists()]
    if missing:
        missing_text = "\n".join(missing)
        raise FileNotFoundError(
            "Local run is enabled, but some required input files are missing:\n"
            f"{missing_text}"
        )


def main() -> None:
    bootstrap_parser = argparse.ArgumentParser(add_help=False)
    bootstrap_parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
    )
    bootstrap_parser.add_argument(
        "--model-pipeline",
        type=str,
        default=None,
    )
    bootstrap_args, _ = bootstrap_parser.parse_known_args()
    config_path = Path(bootstrap_args.config).expanduser().resolve()
    model_pipeline = resolve_model_pipeline_name(
        config_path,
        override=bootstrap_args.model_pipeline or os.getenv("MODEL_PIPELINE"),
    )

    if model_pipeline == "multigraph":
        from models import multigraph_pipeline as pipeline
    elif model_pipeline == "base_gnn":
        from models import base_gnn_pipeline as pipeline
    else:
        raise ValueError(
            f"Unsupported model_pipeline: {model_pipeline}. Expected one of: multigraph, base_gnn"
        )

    parser = argparse.ArgumentParser(description="Train graph temporal fusion model.")
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to YAML or JSON config file.",
    )
    parser.add_argument(
        "--model-pipeline",
        type=str,
        default=None,
        help="Override the model pipeline to run. Supported: multigraph, base_gnn.",
    )
    parser.add_argument("--skip-email", action="store_true", help="Skip email sending.")
    parser.add_argument("--skip-gcs-upload", action="store_true", help="Skip artifact upload to GCS.")
    pipeline.add_cli_override_args(parser)
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()

    run_start_dt = pipeline.utc_now()
    run_start_utc = run_start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    run_start_perf = time.perf_counter()
    run_status = "success"
    selected_operator = None
    gcs_uploaded_uris: List[str] = []
    gcs_run_prefix = ""
    report_paths: Dict[str, Path] = {}
    resolved_config_path: Optional[Path] = None
    environment_metadata_path: Optional[Path] = None
    failure_traceback_path: Optional[Path] = None

    cfg = pipeline.load_config(config_path)
    cfg = pipeline.apply_cli_overrides(cfg, args)
    cfg = pipeline.resolve_runtime_overrides(cfg, config_path)
    cfg = pipeline.resolve_extended_config(cfg)
    cfg["model_pipeline"] = model_pipeline

    gcp_enabled = is_gcp_enabled(cfg, pipeline)
    if not gcp_enabled:
        cfg = copy.deepcopy(cfg)
        cfg["gcs_data_prefix"] = ""
        cfg["gcs_run_prefix"] = ""

    run_id = str(cfg["run_id"])
    artifact_root = pipeline.ensure_dir(Path(cfg["artifact_root"]))
    pipeline.configure_logging(artifact_root, run_id)
    pipeline.LOGGER.info("Starting run_id=%s", run_id)
    pipeline.LOGGER.info("Using config: %s", config_path)
    pipeline.LOGGER.info("Using model_pipeline=%s", model_pipeline)
    pipeline.LOGGER.info("Execution mode: %s", "gcp" if gcp_enabled else "local")
    pipeline.LOGGER.info("Resolved gcs_data_prefix=%r", cfg.get("gcs_data_prefix"))
    pipeline.LOGGER.info("Resolved data_dir=%r", cfg.get("data_dir"))
    pipeline.LOGGER.info("Resolved split_file=%r", cfg.get("split_file"))

    try:
        pipeline.initialize_runtime_globals(cfg)

        environment_metadata = pipeline.collect_environment_metadata(
            run_id=run_id,
            run_start_time_utc=run_start_utc,
        )
        environment_metadata_path = pipeline.ARTIFACT_ROOT / "environment_metadata.json"
        pipeline.save_json(environment_metadata_path, environment_metadata)

        resolved_config_path = pipeline.write_resolved_config(cfg, pipeline.ARTIFACT_ROOT)

        required_files = pipeline.expected_required_files(cfg)
        if gcp_enabled:
            from gcp_utils import download_from_gcs

            download_from_gcs(
                required_files=required_files,
                gcs_data_prefix=str(cfg.get("gcs_data_prefix") or ""),
                local_data_dir=Path(cfg["data_dir"]),
            )
        else:
            ensure_local_required_files(required_files, Path(cfg["data_dir"]))

        pipeline.initialize_tensor_state(cfg)

        if bool(pipeline.CFG["run_full_operator_ablation"]):
            operator_runs: Dict[str, Dict[str, object]] = {}
            for operator_name in pipeline.CFG["operator_candidates"]:
                pipeline.LOGGER.info("RUNNING CV FOR OPERATOR: %s", operator_name)
                operator_runs[operator_name] = pipeline.run_cv_for_operator(
                    operator_name=operator_name,
                    base_cfg=pipeline.CFG,
                    is_ablation_context=True,
                )

            operator_selection = pipeline.select_best_operator_from_cv_runs(operator_runs)
            selected_operator = operator_selection["selected_operator_name"]
            operator_post_runs: Dict[str, Dict[str, object]] = {}
            for operator_name in pipeline.CFG["operator_candidates"]:
                pipeline.LOGGER.info("RUNNING FINAL HOLDOUT DIAGNOSTIC FOR OPERATOR: %s", operator_name)
                operator_post_runs[operator_name] = pipeline.run_selected_operator_post_cv_and_production(
                    operator_run=operator_runs[operator_name],
                    cfg=pipeline.CFG,
                )
            final_selected_run = operator_post_runs[selected_operator]

            final_operator_cv_summary_df = pd.concat(
                [run_out["cv_mean_df"] for run_out in operator_runs.values()],
                axis=0,
                ignore_index=True,
            )
            final_operator_cv_summary_df.to_csv(
                pipeline.ARTIFACT_ROOT / "all_operator_cv_mean_summary.csv",
                index=False,
            )
            operator_selection["operator_comparison_df"].to_csv(
                pipeline.ARTIFACT_ROOT / "operator_cv_comparison_summary.csv",
                index=False,
            )
        else:
            default_operator_results = pipeline.run_cv_for_operator(
                operator_name=str(pipeline.CFG["graph_operator"]),
                base_cfg=pipeline.CFG,
                is_ablation_context=False,
            )
            selected_operator = str(pipeline.CFG["graph_operator"])
            final_selected_run = pipeline.run_selected_operator_post_cv_and_production(
                operator_run=default_operator_results,
                cfg=pipeline.CFG,
            )

        preliminary_summary = {
            "run_id": pipeline.RUN_ID,
            "status": "success",
            "selected_operator": selected_operator,
            "artifact_root": str(pipeline.ARTIFACT_ROOT),
            "resolved_config_path": str(resolved_config_path) if resolved_config_path else None,
            "environment_metadata_path": str(environment_metadata_path) if environment_metadata_path else None,
            "gcs_run_prefix": str(pipeline.CFG.get("gcs_run_prefix") or ""),
        }
        pipeline.write_run_summary(pipeline.ARTIFACT_ROOT, preliminary_summary)

        report_paths = pipeline.build_final_report(
            pipeline.ARTIFACT_ROOT,
            gcs_run_prefix=str(pipeline.CFG.get("gcs_run_prefix") or ""),
        )

        if gcp_enabled and not args.skip_gcs_upload and str(pipeline.CFG.get("gcs_run_prefix") or "").strip():
            from gcp_utils import refresh_gcs_artifacts, upload_artifacts_to_gcs

            gcs_run_prefix = str(pipeline.CFG["gcs_run_prefix"])
            gcs_uploaded_uris = upload_artifacts_to_gcs(pipeline.ARTIFACT_ROOT, gcs_run_prefix)
            pipeline.save_json(
                pipeline.ARTIFACT_ROOT / "gcs_upload_summary.json",
                {
                    "run_id": pipeline.RUN_ID,
                    "gcs_run_prefix": gcs_run_prefix,
                    "uploaded_count": len(gcs_uploaded_uris),
                    "uploaded_uris": gcs_uploaded_uris,
                },
            )

            run_end_utc = pipeline.utc_now_iso()
            total_runtime_seconds = float(time.perf_counter() - run_start_perf)
            environment_metadata = pipeline.finalize_environment_metadata(
                metadata=environment_metadata,
                run_end_time_utc=run_end_utc,
                total_runtime_seconds=total_runtime_seconds,
            )
            pipeline.save_json(environment_metadata_path, environment_metadata)

            run_summary = {
                "run_id": pipeline.RUN_ID,
                "status": run_status,
                "selected_operator": selected_operator,
                "artifact_root": str(pipeline.ARTIFACT_ROOT),
                "resolved_config_path": str(resolved_config_path) if resolved_config_path else None,
                "environment_metadata_path": str(environment_metadata_path) if environment_metadata_path else None,
                "final_report_csv": str(report_paths.get("csv")) if report_paths else None,
                "final_report_html": str(report_paths.get("html")) if report_paths else None,
                "gcs_run_prefix": gcs_run_prefix,
                "uploaded_count": len(gcs_uploaded_uris),
                "run_start_time_utc": run_start_utc,
                "run_end_time_utc": run_end_utc,
                "total_runtime_seconds": total_runtime_seconds,
            }
            pipeline.write_run_summary(pipeline.ARTIFACT_ROOT, run_summary)

            report_paths = pipeline.build_final_report(
                pipeline.ARTIFACT_ROOT,
                gcs_run_prefix=gcs_run_prefix or str(pipeline.CFG.get("gcs_run_prefix") or ""),
            )

            refresh_files = [
                report_paths["csv"],
                report_paths["html"],
                environment_metadata_path,
                pipeline.ARTIFACT_ROOT / "run_summary.json",
            ]
            gcs_summary_path = pipeline.ARTIFACT_ROOT / "gcs_upload_summary.json"
            if gcs_summary_path.exists():
                refresh_files.append(gcs_summary_path)
            refresh_gcs_artifacts(pipeline.ARTIFACT_ROOT, gcs_run_prefix, refresh_files)

        run_end_utc = pipeline.utc_now_iso()
        total_runtime_seconds = float(time.perf_counter() - run_start_perf)

        environment_metadata = pipeline.finalize_environment_metadata(
            metadata=environment_metadata,
            run_end_time_utc=run_end_utc,
            total_runtime_seconds=total_runtime_seconds,
        )
        pipeline.save_json(environment_metadata_path, environment_metadata)

        run_summary = {
            "run_id": pipeline.RUN_ID,
            "status": run_status,
            "selected_operator": selected_operator,
            "artifact_root": str(pipeline.ARTIFACT_ROOT),
            "resolved_config_path": str(resolved_config_path) if resolved_config_path else None,
            "environment_metadata_path": str(environment_metadata_path) if environment_metadata_path else None,
            "final_report_csv": str(report_paths.get("csv")) if report_paths else None,
            "final_report_html": str(report_paths.get("html")) if report_paths else None,
            "gcs_run_prefix": gcs_run_prefix,
            "uploaded_count": len(gcs_uploaded_uris),
            "run_start_time_utc": run_start_utc,
            "run_end_time_utc": run_end_utc,
            "total_runtime_seconds": total_runtime_seconds,
        }
        pipeline.write_run_summary(pipeline.ARTIFACT_ROOT, run_summary)

        report_paths = pipeline.build_final_report(
            pipeline.ARTIFACT_ROOT,
            gcs_run_prefix=gcs_run_prefix or str(pipeline.CFG.get("gcs_run_prefix") or ""),
        )

        if not args.skip_email:
            subject = f"[SUCCESS] {pipeline.ARTIFACT_ROOT.name}"
            html_body = (
                report_paths["html"].read_text(encoding="utf-8")
                if report_paths
                else "<p>Training completed.</p>"
            )
            attachments = pipeline.build_success_email_attachments(
                artifact_root=pipeline.ARTIFACT_ROOT,
                report_paths=report_paths,
                resolved_config_path=resolved_config_path,
            )
            pipeline.send_email_report(
                subject=subject,
                html_body=html_body,
                to_email=str(pipeline.CFG.get("email_to") or ""),
                attachments=attachments,
                smtp_settings=pipeline.get_smtp_settings(pipeline.CFG),
            )

        pipeline.LOGGER.info(
            "Training pipeline complete. final_summary_path=%s | operator_dir=%s",
            pipeline.ARTIFACT_ROOT / "run_summary.json",
            final_selected_run.get("artifact_dir"),
        )

    except Exception:
        run_status = "failure"
        run_end_utc = pipeline.utc_now_iso()
        total_runtime_seconds = float(time.perf_counter() - run_start_perf)
        traceback_text = traceback.format_exc()
        pipeline.LOGGER.exception("Training failed.")

        failure_traceback_path = pipeline.ARTIFACT_ROOT / "failure_traceback.txt"
        failure_traceback_path.write_text(traceback_text, encoding="utf-8")

        try:
            if environment_metadata_path is None:
                environment_metadata_path = pipeline.ARTIFACT_ROOT / "environment_metadata.json"

            environment_metadata = pipeline.collect_environment_metadata(
                run_id=run_id,
                run_start_time_utc=run_start_utc,
            )
            environment_metadata = pipeline.finalize_environment_metadata(
                metadata=environment_metadata,
                run_end_time_utc=run_end_utc,
                total_runtime_seconds=total_runtime_seconds,
            )
            pipeline.save_json(environment_metadata_path, environment_metadata)

            pipeline.write_run_summary(
                pipeline.ARTIFACT_ROOT,
                {
                    "run_id": run_id,
                    "status": run_status,
                    "selected_operator": selected_operator,
                    "artifact_root": str(pipeline.ARTIFACT_ROOT),
                    "resolved_config_path": str(resolved_config_path) if resolved_config_path else None,
                    "environment_metadata_path": str(environment_metadata_path) if environment_metadata_path else None,
                    "failure_traceback_path": str(failure_traceback_path),
                    "gcs_run_prefix": str(pipeline.CFG.get("gcs_run_prefix") or ""),
                    "run_start_time_utc": run_start_utc,
                    "run_end_time_utc": run_end_utc,
                    "total_runtime_seconds": total_runtime_seconds,
                },
            )

            if not args.skip_email:
                failure_html = (
                    f"<h1>Training failed</h1>"
                    f"<p><strong>run_id:</strong> {html.escape(run_id)}</p>"
                    f"<pre>{html.escape(traceback_text)}</pre>"
                )
                pipeline.send_email_report(
                    subject=f"[FAILURE] run_id={run_id} operator={selected_operator}",
                    html_body=failure_html,
                    to_email=str(pipeline.CFG.get("email_to") or ""),
                    attachments=[
                        p for p in [
                            resolved_config_path,
                            environment_metadata_path,
                            failure_traceback_path,
                        ]
                        if p is not None
                    ],
                    smtp_settings=pipeline.get_smtp_settings(pipeline.CFG),
                )
        except Exception:
            pipeline.LOGGER.exception("Failed while handling top-level exception.")

        raise


if __name__ == "__main__":
    main()

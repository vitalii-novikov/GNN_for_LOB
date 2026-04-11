# %% Imports

import argparse
import copy
import html
import json
import logging
import os
import platform
import random
import smtplib
import socket
import subprocess
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

import torch
import torch.nn as nn
import torch.nn.functional as F

from splits import build_or_load_split_bundle

warnings.filterwarnings("ignore", category=FutureWarning)

# %% Logging and utility helpers

LOGGER = logging.getLogger("train")

CFG: Dict[str, Any] = {}
ASSETS: List[str] = []
TARGET_ASSET: str = ""
ASSET2IDX: Dict[str, int] = {}
TARGET_NODE: int = 0
ARTIFACT_ROOT = Path(".")
ARTIFACT_BASE_ROOT = Path(".")
RUN_ID = ""
CONFIG_PATH = Path("train_config.yaml")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FREQ = ""
HORIZON_MINUTES = 0
HORIZON_BARS = 0
LOOKBACK_BARS = 0
RELATION_WINDOWS: List[int] = []
RELATION_LAGS: List[int] = []
PURGE_GAP_BARS = 0
EXPECTED_DELTA = pd.Timedelta(seconds=60)

RELATION_NAMES: List[str] = ["price_dep", "order_flow", "liquidity"]
EDGE_LIST: List[Tuple[str, str]] = []
EDGE_NAMES: List[str] = []
EDGE_INDEX = torch.empty((0, 2), dtype=torch.long)
EDGE_SRC_IDX = torch.empty((0,), dtype=torch.long)
EDGE_DST_IDX = torch.empty((0,), dtype=torch.long)

EPS = 1e-12
df = pd.DataFrame()
TIMESTAMPS = pd.Series(dtype="datetime64[ns, UTC]")
X_NODE_RAW = np.empty((0, 0, 0), dtype=np.float32)
NODE_FEATURE_NAMES: List[str] = []
RELATION_STATE_MAP: Dict[str, Dict[str, np.ndarray]] = {}
X_REL_EDGE_RAW = np.empty((0, 0, 0, 0), dtype=np.float32)
EDGE_FEATURE_NAMES: List[str] = []
TARGET_MID = np.empty((0,), dtype=np.float64)
TARGET_LR_1BAR = np.empty((0,), dtype=np.float64)
Y_RET = np.empty((0,), dtype=np.float32)
Y_DIR = np.empty((0,), dtype=np.float32)
Y_TRADE = np.empty((0,), dtype=np.float32)
Y_DIR_MASK = np.empty((0,), dtype=np.float32)
Y_EXIT_TYPE = np.empty((0,), dtype=np.int64)
Y_TTE = np.empty((0,), dtype=np.float32)
Y_TIMEOUT = np.empty((0,), dtype=np.float32)
TARGET_SUMMARY: Dict[str, Any] = {}
TRADE_LABEL_ABS_RETURN_THRESHOLD = 0.0
EXIT_TYPE_TO_IDX: Dict[str, int] = {"vertical": 0, "upper": 1, "lower": 2}
EXIT_TYPE_NAMES: List[str] = ["vertical", "upper", "lower"]

T = 0
FIRST_VALID_T = 0
LAST_VALID_T = 0
SAMPLE_T = np.empty((0,), dtype=np.int64)
N_SAMPLES = 0
IDX_PREHOLDOUT = np.empty((0,), dtype=np.int64)
IDX_HOLDOUT = np.empty((0,), dtype=np.int64)
WALK_FORWARD_SPLITS: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
IDX_TRAIN_FINAL = np.empty((0,), dtype=np.int64)
IDX_VAL_FINAL = np.empty((0,), dtype=np.int64)
IDX_TEST_FINAL = np.empty((0,), dtype=np.int64)


def configure_logging(log_dir: Path, run_id: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train.log"
    formatter = logging.Formatter(
        fmt=f"%(asctime)s | %(levelname)s | {run_id} | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def parse_bool_arg(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        f"Invalid boolean value: {value}. Use one of true/false/1/0/yes/no."
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_yaml(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def safe_relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except Exception:
        return str(path)


def get_git_commit_sha() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        sha = result.stdout.strip()
        return sha or None
    except Exception:
        return None


def generate_run_id() -> str:
    timestamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
    sha = get_git_commit_sha()
    return f"{timestamp}_{sha}" if sha else f"{timestamp}_manual"


def build_artifact_root_paths(artifact_root_value: str, freq_value: Any) -> Tuple[Path, Path]:
    artifact_root_value = str(artifact_root_value or "").strip() or "./artifacts"
    configured_root = Path(artifact_root_value).expanduser()
    artifact_root_base = configured_root.parent
    artifact_root_name = configured_root.name or "artifacts"
    freq_name = normalize_freq_name(str(freq_value or "1min"))
    artifact_suffix = utc_now().strftime("%m%d%H%M%S")
    artifact_root = artifact_root_base / f"{artifact_root_name}_{freq_name}_{artifact_suffix}"
    return artifact_root_base, artifact_root


def build_gcs_run_prefix(prefix: str, run_id: str) -> str:
    prefix = str(prefix or "").strip()
    if not prefix:
        return ""
    if "{run_id}" in prefix:
        return prefix.format(run_id=run_id)
    if prefix.rstrip("/").endswith(run_id):
        return prefix.rstrip("/")
    return f"{prefix.rstrip('/')}/{run_id}"


def parse_gs_uri(uri: str) -> Tuple[str, str]:
    if not str(uri).startswith("gs://"):
        raise ValueError(f"Expected gs:// URI, got: {uri}")
    remainder = str(uri)[5:]
    bucket, _, blob = remainder.partition("/")
    if not bucket:
        raise ValueError(f"Missing bucket in GCS URI: {uri}")
    return bucket, blob


def artifact_uri_join(prefix: str, relative_path: str) -> str:
    return f"{prefix.rstrip('/')}/{relative_path.lstrip('/')}"


def runtime_path_summary() -> Dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "config_path": str(CONFIG_PATH),
        "artifact_root": str(ARTIFACT_ROOT),
        "artifact_root_base": str(ARTIFACT_BASE_ROOT),
        "data_dir": str(Path(CFG["data_dir"])),
    }


# %% Config loading and runtime env resolution

def load_config(config_path: Path) -> Dict[str, Any]:
    config_path = config_path.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    suffix = config_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        loaded = load_yaml(config_path)
    elif suffix == ".json":
        loaded = read_json(config_path)
    else:
        raise ValueError(f"Unsupported config extension: {config_path.suffix}")

    if loaded is None:
        loaded = {}

    if not isinstance(loaded, dict):
        raise ValueError("Config root must be a mapping/object.")

    return loaded



def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None



def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge_dicts(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out



def get_nested_value(mapping: Dict[str, Any], path: str, default: Any = None) -> Any:
    cursor: Any = mapping
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


MEMORYGRAPH_OPERATORS = {"conv", "mpnn"}


def normalize_memorygraph_operator(value: Any, default: str = "conv") -> str:
    mode = str(first_not_none(value, default)).strip().lower().replace("-", "_")
    if mode not in MEMORYGRAPH_OPERATORS:
        raise ValueError(
            f"Unsupported memorygraph operator: {value}. "
            f"Expected one of: {sorted(MEMORYGRAPH_OPERATORS)}"
        )
    return mode


def maybe_memorygraph_operator(value: Any) -> Optional[str]:
    if value in {None, ""}:
        return None
    raw = str(value).strip().lower().replace("-", "_")
    if raw in MEMORYGRAPH_OPERATORS:
        return raw
    return None


def canonicalize_memorygraph_operator_name(
    operator_name: Any,
    default_operator: str = "conv",
) -> str:
    default_operator = normalize_memorygraph_operator(default_operator, default="conv")
    raw = maybe_memorygraph_operator(operator_name)

    if raw is None:
        if operator_name in {None, ""}:
            return default_operator
        raise ValueError(
            f"Unsupported memorygraph graph_operator: {operator_name}. "
            "Use one of: conv, mpnn."
        )

    if raw in MEMORYGRAPH_OPERATORS:
        return raw
    return default_operator


def normalize_memorygraph_operator_candidate_list(
    operator_candidates: Any,
    default_operator: str,
) -> List[str]:
    if operator_candidates is None:
        return []
    if isinstance(operator_candidates, str):
        raw_items = [x.strip() for x in operator_candidates.split(",") if x.strip()]
    else:
        raw_items = [str(x).strip() for x in operator_candidates if str(x).strip()]
    normalized: List[str] = []
    seen: set = set()
    for item in raw_items:
        canonical_name = canonicalize_memorygraph_operator_name(
            operator_name=item,
            default_operator=default_operator,
        )
        if canonical_name in seen:
            continue
        seen.add(canonical_name)
        normalized.append(canonical_name)
    return normalized



def resolve_runtime_overrides(cfg: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    resolved = copy.deepcopy(cfg)
    run_id = os.getenv("RUN_ID") or generate_run_id()

    data_dir = Path(os.getenv("DATA_DIR") or resolved.get("data_dir") or "./data").expanduser()
    artifact_root_base, artifact_root = build_artifact_root_paths(
        os.getenv("ARTIFACT_ROOT") or resolved.get("artifact_root") or "./artifacts",
        os.getenv("FREQ") or resolved.get("freq") or "1min",
    )

    resolved["data_dir"] = str(data_dir)
    resolved["artifact_root_base"] = str(artifact_root_base)
    resolved["artifact_root"] = str(artifact_root)
    resolved["run_id"] = run_id
    resolved["config_path"] = str(config_path)

    resolved["gcs_data_prefix"] = os.getenv("GCS_DATA_PREFIX", resolved.get("gcs_data_prefix") or "")
    raw_gcs_run_prefix = os.getenv("GCS_RUN_PREFIX", resolved.get("gcs_run_prefix") or "")
    resolved["gcs_run_prefix"] = build_gcs_run_prefix(raw_gcs_run_prefix, run_id) if raw_gcs_run_prefix else ""

    resolved["email_to"] = os.getenv("EMAIL_TO", resolved.get("email_to") or "")
    resolved["email_from"] = os.getenv("EMAIL_FROM", resolved.get("email_from") or "")
    resolved["smtp_host"] = os.getenv("SMTP_HOST", resolved.get("smtp_host") or "")
    resolved["smtp_port"] = int(os.getenv("SMTP_PORT", resolved.get("smtp_port") or 587))
    resolved["smtp_user"] = os.getenv("SMTP_USER", resolved.get("smtp_user") or "")
    resolved["smtp_password"] = os.getenv("SMTP_PASSWORD", resolved.get("smtp_password") or "")
    resolved["smtp_use_tls"] = parse_bool(
        os.getenv("SMTP_USE_TLS", resolved.get("smtp_use_tls")),
        default=parse_bool(resolved.get("smtp_use_tls"), True),
    )

    resolved["machine_type"] = os.getenv("MACHINE_TYPE", resolved.get("machine_type") or "")
    resolved["gcp_region"] = os.getenv("GCP_REGION", resolved.get("gcp_region") or "")
    resolved["container_image"] = os.getenv("CONTAINER_IMAGE", resolved.get("container_image") or "")
    resolved["split_file"] = os.getenv("SPLIT_FILE", resolved.get("split_file") or "")
    resolved["local_run"] = parse_bool(
        os.getenv("LOCAL_RUN", resolved.get("local_run")),
        default=parse_bool(resolved.get("local_run"), False),
    )

    env_overrides = {
        "label_mode": os.getenv("LABEL_MODE"),
        "objective_mode": os.getenv("OBJECTIVE_MODE"),
        "graph_readout_mode": os.getenv("GRAPH_READOUT_MODE"),
        "edge_feature_mode": os.getenv("EDGE_FEATURE_MODE"),
        "backtest_exit_mode": os.getenv("BACKTEST_EXIT_MODE"),
    }
    for key, value in env_overrides.items():
        if value is not None:
            resolved[key] = value
    return resolved



def add_cli_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph-operator", type=str, default=None)
    parser.add_argument("--run-full-operator-ablation", type=parse_bool_arg, default=None)
    parser.add_argument("--operator-candidates", nargs="+", default=None)
    parser.add_argument("--target-asset", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--horizon-minutes", type=int, default=None)
    parser.add_argument("--lookback-bars-override", type=int, default=None)
    parser.add_argument("--relation-windows-override", nargs="+", type=int, default=None)
    parser.add_argument("--relation-lags-bars", nargs="+", type=int, default=None)

    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--artifact-root", type=str, default=None)
    parser.add_argument("--split-file", type=str, default=None)
    parser.add_argument("--local-run", type=parse_bool_arg, default=None)
    parser.add_argument("--data-slice-start-frac", type=float, default=None)
    parser.add_argument("--data-slice-end-frac", type=float, default=None)
    parser.add_argument("--final-holdout-frac", type=float, default=None)

    parser.add_argument("--num-train-folds", type=int, default=None)
    parser.add_argument("--train-min-frac", type=float, default=None)
    parser.add_argument("--val-window-frac", type=float, default=None)
    parser.add_argument("--test-window-frac", type=float, default=None)
    parser.add_argument("--purge-gap-extra-bars", type=int, default=None)

    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None)

    parser.add_argument("--node-hidden-dim", type=int, default=None)
    parser.add_argument("--edge-hidden-dim", type=int, default=None)
    parser.add_argument("--target-hidden-dim", type=int, default=None)
    parser.add_argument("--graph-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--fusion-hidden-dim", type=int, default=None)
    parser.add_argument("--memorygraph-default-operator", type=str, default=None)
    parser.add_argument("--memorygraph-operator-candidates", nargs="+", default=None)
    parser.add_argument("--memorygraph-relation-fusion-mode", type=str, default=None)
    parser.add_argument("--memorygraph-adjacency-rank", type=int, default=None)
    parser.add_argument("--memorygraph-adjacency-scale", type=float, default=None)
    parser.add_argument("--memorygraph-node-memory-dim", type=int, default=None)
    parser.add_argument("--memorygraph-edge-memory-dim", type=int, default=None)
    parser.add_argument("--memorygraph-memory-dropout", type=float, default=None)
    parser.add_argument("--memorygraph-stateful-training", type=parse_bool_arg, default=None)
    parser.add_argument("--memorygraph-chunk-len", type=int, default=None)
    parser.add_argument("--memorygraph-detach-interval", type=int, default=None)
    parser.add_argument("--memorygraph-eval-chunk-len", type=int, default=None)
    parser.add_argument("--memorygraph-calibration-frac", type=float, default=None)
    parser.add_argument("--memorygraph-min-calibration-samples", type=int, default=None)
    parser.add_argument("--memorygraph-vertical-barrier-bars", type=int, default=None)

    parser.add_argument("--label-mode", type=str, default=None)
    parser.add_argument("--objective-mode", type=str, default=None)
    parser.add_argument("--backtest-exit-mode", type=str, default=None)
    parser.add_argument("--graph-readout-mode", type=str, default=None)
    parser.add_argument("--graph-global-pool", nargs="+", default=None)
    parser.add_argument("--use-target-global-attention", type=parse_bool_arg, default=None)
    parser.add_argument("--edge-feature-mode", type=str, default=None)
    parser.add_argument("--learned-pairwise-hidden-dim", type=int, default=None)
    parser.add_argument("--meta-labeling-enabled", type=parse_bool_arg, default=None)
    parser.add_argument("--trade-label-requires-first-touch", type=parse_bool_arg, default=None)
    parser.add_argument("--mask-timeout-for-direction", type=parse_bool_arg, default=None)

    parser.add_argument("--tb-pt-sl-mode", type=str, default=None)
    parser.add_argument("--tb-upper-barrier-bps", type=float, default=None)
    parser.add_argument("--tb-lower-barrier-bps", type=float, default=None)
    parser.add_argument("--tb-vol-lookback-bars", type=int, default=None)
    parser.add_argument("--tb-vol-mult-up", type=float, default=None)
    parser.add_argument("--tb-vol-mult-down", type=float, default=None)
    parser.add_argument("--tb-min-barrier-bps", type=float, default=None)
    parser.add_argument("--tb-max-barrier-bps", type=float, default=None)
    parser.add_argument("--tb-vertical-use-horizon", type=parse_bool_arg, default=None)

    parser.add_argument("--loss-w-trade", type=float, default=None)
    parser.add_argument("--loss-w-dir", type=float, default=None)
    parser.add_argument("--loss-w-ret", type=float, default=None)
    parser.add_argument("--loss-w-utility", type=float, default=None)
    parser.add_argument("--loss-w-exit-type", type=float, default=None)
    parser.add_argument("--loss-w-tte", type=float, default=None)
    parser.add_argument("--utility-tanh-k", type=float, default=None)
    parser.add_argument("--huber-beta", type=float, default=None)
    parser.add_argument("--adj-l1-lambda", type=float, default=None)
    parser.add_argument("--cost-bps-per-side", type=float, default=None)
    parser.add_argument("--trade-label-buffer-bps", type=float, default=None)
    parser.add_argument("--false-positive-penalty", type=float, default=None)
    parser.add_argument("--timeout-penalty", type=float, default=None)
    parser.add_argument("--execution-cost-multiplier", type=float, default=None)
    parser.add_argument("--use-cost-in-label", type=parse_bool_arg, default=None)

    parser.add_argument("--thr-trade-grid", nargs="+", type=float, default=None)
    parser.add_argument("--thr-dir-grid", nargs="+", type=float, default=None)
    parser.add_argument("--min-validation-trades", type=int, default=None)
    parser.add_argument("--min-validation-coverage", type=float, default=None)
    parser.add_argument("--threshold-search-metric", type=str, default=None)
    parser.add_argument("--allow-timeout-trades", type=parse_bool_arg, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-threads", type=int, default=None)

    parser.add_argument("--gcs-data-prefix", type=str, default=None)
    parser.add_argument("--gcs-run-prefix", type=str, default=None)
    parser.add_argument("--email-to", type=str, default=None)
    parser.add_argument("--email-from", type=str, default=None)
    parser.add_argument("--smtp-host", type=str, default=None)
    parser.add_argument("--smtp-port", type=int, default=None)
    parser.add_argument("--smtp-user", type=str, default=None)
    parser.add_argument("--smtp-password", type=str, default=None)
    parser.add_argument("--smtp-use-tls", type=parse_bool_arg, default=None)



def _set_if_not_none(cfg: Dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        cfg[key] = value



def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = copy.deepcopy(cfg)

    scalar_overrides = {
        "graph_operator": args.graph_operator,
        "run_full_operator_ablation": args.run_full_operator_ablation,
        "target_asset": args.target_asset,
        "freq": args.freq,
        "horizon_minutes": args.horizon_minutes,
        "data_dir": args.data_dir,
        "artifact_root": args.artifact_root,
        "split_file": args.split_file,
        "local_run": args.local_run,
        "data_slice_start_frac": args.data_slice_start_frac,
        "data_slice_end_frac": args.data_slice_end_frac,
        "final_holdout_frac": args.final_holdout_frac,
        "num_train_folds": args.num_train_folds,
        "train_min_frac": args.train_min_frac,
        "val_window_frac": args.val_window_frac,
        "test_window_frac": args.test_window_frac,
        "purge_gap_extra_bars": args.purge_gap_extra_bars,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
        "node_hidden_dim": args.node_hidden_dim,
        "edge_hidden_dim": args.edge_hidden_dim,
        "target_hidden_dim": args.target_hidden_dim,
        "graph_layers": args.graph_layers,
        "dropout": args.dropout,
        "fusion_hidden_dim": args.fusion_hidden_dim,
        "memorygraph_default_operator": args.memorygraph_default_operator,
        "memorygraph_relation_fusion_mode": args.memorygraph_relation_fusion_mode,
        "memorygraph_adjacency_rank": args.memorygraph_adjacency_rank,
        "memorygraph_adjacency_scale": args.memorygraph_adjacency_scale,
        "memorygraph_node_memory_dim": args.memorygraph_node_memory_dim,
        "memorygraph_edge_memory_dim": args.memorygraph_edge_memory_dim,
        "memorygraph_memory_dropout": args.memorygraph_memory_dropout,
        "memorygraph_stateful_training": args.memorygraph_stateful_training,
        "memorygraph_chunk_len": args.memorygraph_chunk_len,
        "memorygraph_detach_interval": args.memorygraph_detach_interval,
        "memorygraph_eval_chunk_len": args.memorygraph_eval_chunk_len,
        "memorygraph_calibration_frac": args.memorygraph_calibration_frac,
        "memorygraph_min_calibration_samples": args.memorygraph_min_calibration_samples,
        "memorygraph_vertical_barrier_bars": args.memorygraph_vertical_barrier_bars,
        "label_mode": args.label_mode,
        "objective_mode": args.objective_mode,
        "backtest_exit_mode": args.backtest_exit_mode,
        "graph_readout_mode": args.graph_readout_mode,
        "use_target_global_attention": args.use_target_global_attention,
        "edge_feature_mode": args.edge_feature_mode,
        "learned_pairwise_hidden_dim": args.learned_pairwise_hidden_dim,
        "meta_labeling_enabled": args.meta_labeling_enabled,
        "trade_label_requires_first_touch": args.trade_label_requires_first_touch,
        "mask_timeout_for_direction": args.mask_timeout_for_direction,
        "triple_barrier_pt_sl_mode": args.tb_pt_sl_mode,
        "triple_barrier_upper_barrier_bps": args.tb_upper_barrier_bps,
        "triple_barrier_lower_barrier_bps": args.tb_lower_barrier_bps,
        "triple_barrier_vol_lookback_bars": args.tb_vol_lookback_bars,
        "triple_barrier_vol_barrier_mult_up": args.tb_vol_mult_up,
        "triple_barrier_vol_barrier_mult_down": args.tb_vol_mult_down,
        "triple_barrier_min_barrier_bps": args.tb_min_barrier_bps,
        "triple_barrier_max_barrier_bps": args.tb_max_barrier_bps,
        "triple_barrier_vertical_barrier_use_horizon": args.tb_vertical_use_horizon,
        "loss_w_trade": args.loss_w_trade,
        "loss_w_dir": args.loss_w_dir,
        "loss_w_ret": args.loss_w_ret,
        "loss_w_utility": args.loss_w_utility,
        "loss_w_exit_type": args.loss_w_exit_type,
        "loss_w_tte": args.loss_w_tte,
        "utility_tanh_k": args.utility_tanh_k,
        "huber_beta": args.huber_beta,
        "adj_l1_lambda": args.adj_l1_lambda,
        "cost_bps_per_side": args.cost_bps_per_side,
        "trade_label_buffer_bps": args.trade_label_buffer_bps,
        "false_positive_penalty": args.false_positive_penalty,
        "timeout_penalty": args.timeout_penalty,
        "execution_cost_multiplier": args.execution_cost_multiplier,
        "use_cost_in_label": args.use_cost_in_label,
        "min_validation_trades": args.min_validation_trades,
        "min_validation_coverage": args.min_validation_coverage,
        "threshold_search_metric": args.threshold_search_metric,
        "allow_timeout_trades": args.allow_timeout_trades,
        "seed": args.seed,
        "num_threads": args.num_threads,
        "gcs_data_prefix": args.gcs_data_prefix,
        "gcs_run_prefix": args.gcs_run_prefix,
        "email_to": args.email_to,
        "email_from": args.email_from,
        "smtp_host": args.smtp_host,
        "smtp_port": args.smtp_port,
        "smtp_user": args.smtp_user,
        "smtp_password": args.smtp_password,
        "smtp_use_tls": args.smtp_use_tls,
    }
    for key, value in scalar_overrides.items():
        _set_if_not_none(merged, key, value)

    if args.operator_candidates is not None:
        merged["operator_candidates"] = [str(x) for x in args.operator_candidates]
    if args.memorygraph_operator_candidates is not None:
        merged["memorygraph_operator_candidates"] = [str(x) for x in args.memorygraph_operator_candidates]
    if args.relation_lags_bars is not None:
        merged["relation_lags_bars"] = [int(x) for x in args.relation_lags_bars]
    if args.thr_trade_grid is not None:
        merged["thr_trade_grid"] = [float(x) for x in args.thr_trade_grid]
    if args.thr_dir_grid is not None:
        merged["thr_dir_grid"] = [float(x) for x in args.thr_dir_grid]
    if args.graph_global_pool is not None:
        merged["graph_global_pool"] = [str(x) for x in args.graph_global_pool]

    freq = normalize_freq_name(merged.get("freq", cfg.get("freq", "1min")))

    if args.lookback_bars_override is not None:
        lookback_map = copy.deepcopy(merged.get("lookback_bars_by_freq") or {})
        lookback_map[freq] = int(args.lookback_bars_override)
        merged["lookback_bars_by_freq"] = lookback_map

    if args.relation_windows_override is not None:
        relation_map = copy.deepcopy(merged.get("relation_windows_bars_by_freq") or {})
        relation_map[freq] = [int(x) for x in args.relation_windows_override]
        merged["relation_windows_bars_by_freq"] = relation_map

    return merged



def resolve_extended_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(cfg)

    triple_barrier_defaults = {
        "pt_sl_mode": "fixed",
        "upper_barrier_bps": 8.0,
        "lower_barrier_bps": 8.0,
        "vol_lookback_bars": 30,
        "vol_barrier_mult_up": 1.8,
        "vol_barrier_mult_down": 1.8,
        "min_barrier_bps": 4.0,
        "max_barrier_bps": 30.0,
        "vertical_barrier_use_horizon": True,
    }
    meta_labeling_defaults = {
        "enabled": True,
        "trade_label_requires_first_touch": True,
        "mask_timeout_for_direction": True,
    }
    architecture_defaults = {
        "graph_readout_mode": "target_plus_attn_global",
        "graph_global_pool": ["mean", "max"],
        "use_target_global_attention": True,
        "edge_feature_mode": "hybrid",
        "learned_pairwise_hidden_dim": 48,
    }
    memorygraph_defaults = {
        "default_operator": "conv",
        "relation_fusion_mode": "attention",
        "adjacency_rank": 16,
        "adjacency_scale": 1.0,
        "node_memory_dim": None,
        "edge_memory_dim": None,
        "memory_dropout": None,
        "stateful_training": True,
        "chunk_len": 96,
        "detach_interval": 96,
        "eval_chunk_len": 192,
        "calibration_frac": 0.4,
        "min_calibration_samples": 64,
        "vertical_barrier_bars": None,
        "operator_candidates": [
            "conv",
            "mpnn",
        ],
    }
    loss_defaults = {
        "loss_w_trade": float(resolved.get("loss_w_trade", 0.35)),
        "loss_w_dir": float(resolved.get("loss_w_dir", 0.60)),
        "loss_w_ret": float(resolved.get("loss_w_ret", 0.15)),
        "loss_w_utility": float(resolved.get("loss_w_utility", 0.80)),
        "loss_w_exit_type": float(resolved.get("loss_w_exit_type", 0.0) or 0.0),
        "loss_w_tte": float(resolved.get("loss_w_tte", 0.0) or 0.0),
        "utility_tanh_k": float(resolved.get("utility_tanh_k", 1.8)),
        "false_positive_penalty": float(resolved.get("false_positive_penalty", 0.20) or 0.20),
        "timeout_penalty": float(resolved.get("timeout_penalty", 0.10) or 0.10),
        "execution_cost_multiplier": float(resolved.get("execution_cost_multiplier", 1.0) or 1.0),
        "use_cost_in_label": parse_bool(resolved.get("use_cost_in_label"), True),
    }
    backtest_defaults = {
        "exit_mode": "realized_event",
        "threshold_search_metric": "composite",
        "allow_timeout_trades": True,
    }

    resolved["label_mode"] = str(first_not_none(
        resolved.get("label_mode"),
        get_nested_value(resolved, "label_mode"),
        "fixed_horizon",
    ))
    resolved["objective_mode"] = str(first_not_none(
        resolved.get("objective_mode"),
        get_nested_value(resolved, "objective_mode"),
        "execution_aware",
    ))

    tb_cfg = deep_merge_dicts(triple_barrier_defaults, get_nested_value(resolved, "triple_barrier", {}))
    ml_cfg = deep_merge_dicts(meta_labeling_defaults, get_nested_value(resolved, "meta_labeling", {}))
    arch_cfg = deep_merge_dicts(architecture_defaults, get_nested_value(resolved, "architecture", {}))
    memorygraph_cfg = deep_merge_dicts(memorygraph_defaults, get_nested_value(resolved, "memorygraph", {}))
    loss_cfg = deep_merge_dicts(loss_defaults, get_nested_value(resolved, "loss", {}))
    backtest_cfg = deep_merge_dicts(backtest_defaults, get_nested_value(resolved, "backtest", {}))

    resolved["triple_barrier_pt_sl_mode"] = str(first_not_none(resolved.get("triple_barrier_pt_sl_mode"), tb_cfg.get("pt_sl_mode"), "fixed"))
    resolved["triple_barrier_upper_barrier_bps"] = float(first_not_none(resolved.get("triple_barrier_upper_barrier_bps"), tb_cfg.get("upper_barrier_bps"), 8.0))
    resolved["triple_barrier_lower_barrier_bps"] = float(first_not_none(resolved.get("triple_barrier_lower_barrier_bps"), tb_cfg.get("lower_barrier_bps"), 8.0))
    resolved["triple_barrier_vol_lookback_bars"] = int(first_not_none(resolved.get("triple_barrier_vol_lookback_bars"), tb_cfg.get("vol_lookback_bars"), 30))
    resolved["triple_barrier_vol_barrier_mult_up"] = float(first_not_none(resolved.get("triple_barrier_vol_barrier_mult_up"), tb_cfg.get("vol_barrier_mult_up"), 1.8))
    resolved["triple_barrier_vol_barrier_mult_down"] = float(first_not_none(resolved.get("triple_barrier_vol_barrier_mult_down"), tb_cfg.get("vol_barrier_mult_down"), 1.8))
    resolved["triple_barrier_min_barrier_bps"] = float(first_not_none(resolved.get("triple_barrier_min_barrier_bps"), tb_cfg.get("min_barrier_bps"), 4.0))
    resolved["triple_barrier_max_barrier_bps"] = float(first_not_none(resolved.get("triple_barrier_max_barrier_bps"), tb_cfg.get("max_barrier_bps"), 30.0))
    resolved["triple_barrier_vertical_barrier_use_horizon"] = parse_bool(
        first_not_none(resolved.get("triple_barrier_vertical_barrier_use_horizon"), tb_cfg.get("vertical_barrier_use_horizon"), True),
        True,
    )

    resolved["meta_labeling_enabled"] = parse_bool(first_not_none(resolved.get("meta_labeling_enabled"), ml_cfg.get("enabled"), True), True)
    resolved["trade_label_requires_first_touch"] = parse_bool(
        first_not_none(resolved.get("trade_label_requires_first_touch"), ml_cfg.get("trade_label_requires_first_touch"), True),
        True,
    )
    resolved["mask_timeout_for_direction"] = parse_bool(
        first_not_none(resolved.get("mask_timeout_for_direction"), ml_cfg.get("mask_timeout_for_direction"), True),
        True,
    )

    resolved["graph_readout_mode"] = str(first_not_none(resolved.get("graph_readout_mode"), arch_cfg.get("graph_readout_mode"), "target_plus_attn_global"))
    graph_global_pool = first_not_none(resolved.get("graph_global_pool"), arch_cfg.get("graph_global_pool"), ["mean", "max"])
    if isinstance(graph_global_pool, str):
        graph_global_pool = [x.strip() for x in graph_global_pool.split(",") if x.strip()]
    resolved["graph_global_pool"] = [str(x) for x in graph_global_pool]
    resolved["use_target_global_attention"] = parse_bool(
        first_not_none(resolved.get("use_target_global_attention"), arch_cfg.get("use_target_global_attention"), True),
        True,
    )
    resolved["edge_feature_mode"] = str(first_not_none(resolved.get("edge_feature_mode"), arch_cfg.get("edge_feature_mode"), "hybrid"))
    resolved["learned_pairwise_hidden_dim"] = int(first_not_none(resolved.get("learned_pairwise_hidden_dim"), arch_cfg.get("learned_pairwise_hidden_dim"), 48))
    resolved["memorygraph_default_operator"] = normalize_memorygraph_operator(
        first_not_none(resolved.get("memorygraph_default_operator"), memorygraph_cfg.get("default_operator"), "conv"),
        default="conv",
    )
    resolved["memorygraph_relation_fusion_mode"] = str(
        first_not_none(resolved.get("memorygraph_relation_fusion_mode"), memorygraph_cfg.get("relation_fusion_mode"), "attention")
    ).strip().lower().replace("-", "_")
    resolved["memorygraph_adjacency_rank"] = int(
        first_not_none(resolved.get("memorygraph_adjacency_rank"), memorygraph_cfg.get("adjacency_rank"), 16)
    )
    resolved["memorygraph_adjacency_scale"] = float(
        first_not_none(resolved.get("memorygraph_adjacency_scale"), memorygraph_cfg.get("adjacency_scale"), 1.0)
    )
    resolved["memorygraph_node_memory_dim"] = int(
        first_not_none(
            resolved.get("memorygraph_node_memory_dim"),
            memorygraph_cfg.get("node_memory_dim"),
            resolved.get("node_hidden_dim"),
            64,
        )
    )
    resolved["memorygraph_edge_memory_dim"] = int(
        first_not_none(
            resolved.get("memorygraph_edge_memory_dim"),
            memorygraph_cfg.get("edge_memory_dim"),
            resolved.get("edge_hidden_dim"),
            32,
        )
    )
    resolved["memorygraph_memory_dropout"] = float(
        first_not_none(
            resolved.get("memorygraph_memory_dropout"),
            memorygraph_cfg.get("memory_dropout"),
            resolved.get("dropout"),
            0.1,
        )
    )
    resolved["memorygraph_stateful_training"] = parse_bool(
        first_not_none(
            resolved.get("memorygraph_stateful_training"),
            memorygraph_cfg.get("stateful_training"),
            True,
        ),
        True,
    )
    resolved["memorygraph_chunk_len"] = int(
        first_not_none(
            resolved.get("memorygraph_chunk_len"),
            memorygraph_cfg.get("chunk_len"),
            96,
        )
    )
    resolved["memorygraph_detach_interval"] = int(
        first_not_none(
            resolved.get("memorygraph_detach_interval"),
            memorygraph_cfg.get("detach_interval"),
            resolved["memorygraph_chunk_len"],
        )
    )
    resolved["memorygraph_eval_chunk_len"] = int(
        first_not_none(
            resolved.get("memorygraph_eval_chunk_len"),
            memorygraph_cfg.get("eval_chunk_len"),
            max(resolved["memorygraph_chunk_len"], 1),
        )
    )
    resolved["memorygraph_calibration_frac"] = float(
        first_not_none(
            resolved.get("memorygraph_calibration_frac"),
            memorygraph_cfg.get("calibration_frac"),
            0.4,
        )
    )
    resolved["memorygraph_min_calibration_samples"] = int(
        first_not_none(
            resolved.get("memorygraph_min_calibration_samples"),
            memorygraph_cfg.get("min_calibration_samples"),
            64,
        )
    )
    vertical_barrier_bars_value = first_not_none(
        resolved.get("memorygraph_vertical_barrier_bars"),
        resolved.get("vertical_barrier_bars"),
        memorygraph_cfg.get("vertical_barrier_bars"),
    )
    resolved["memorygraph_vertical_barrier_bars"] = (
        None if vertical_barrier_bars_value in {None, ""} else int(vertical_barrier_bars_value)
    )

    if resolved["memorygraph_relation_fusion_mode"] not in {"attention", "mean"}:
        raise ValueError(
            "Unsupported memorygraph_relation_fusion_mode: "
            f"{resolved['memorygraph_relation_fusion_mode']}. Expected one of: attention, mean"
        )
    if resolved["memorygraph_adjacency_rank"] <= 0:
        raise ValueError("memorygraph_adjacency_rank must be > 0")
    if resolved["memorygraph_adjacency_scale"] <= 0.0:
        raise ValueError("memorygraph_adjacency_scale must be > 0")
    if resolved["memorygraph_node_memory_dim"] <= 0:
        raise ValueError("memorygraph_node_memory_dim must be > 0")
    if resolved["memorygraph_edge_memory_dim"] <= 0:
        raise ValueError("memorygraph_edge_memory_dim must be > 0")
    if resolved["memorygraph_memory_dropout"] < 0.0:
        raise ValueError("memorygraph_memory_dropout must be >= 0")
    if resolved["memorygraph_chunk_len"] <= 0:
        raise ValueError("memorygraph_chunk_len must be > 0")
    if resolved["memorygraph_detach_interval"] <= 0:
        raise ValueError("memorygraph_detach_interval must be > 0")
    if resolved["memorygraph_eval_chunk_len"] <= 0:
        raise ValueError("memorygraph_eval_chunk_len must be > 0")
    if not (0.05 <= resolved["memorygraph_calibration_frac"] < 0.95):
        raise ValueError("memorygraph_calibration_frac must be in [0.05, 0.95)")
    if resolved["memorygraph_min_calibration_samples"] <= 0:
        raise ValueError("memorygraph_min_calibration_samples must be > 0")
    if (
        resolved["memorygraph_vertical_barrier_bars"] is not None
        and resolved["memorygraph_vertical_barrier_bars"] <= 0
    ):
        raise ValueError("memorygraph_vertical_barrier_bars must be > 0 when provided")

    memorygraph_graph_operator = memorygraph_cfg.get("graph_operator")
    if memorygraph_graph_operator not in {None, ""}:
        resolved["graph_operator"] = canonicalize_memorygraph_operator_name(
            memorygraph_graph_operator,
            default_operator=resolved["memorygraph_default_operator"],
        )
    else:
        resolved["graph_operator"] = canonicalize_memorygraph_operator_name(
            maybe_memorygraph_operator(resolved.get("graph_operator")),
            default_operator=resolved["memorygraph_default_operator"],
        )

    operator_candidates_source = resolved.get("memorygraph_operator_candidates")
    if operator_candidates_source is None:
        operator_candidates_source = memorygraph_cfg.get("operator_candidates")
    if operator_candidates_source is None:
        operator_candidates_source = resolved.get("operator_candidates")
    if operator_candidates_source is None:
        operator_candidates_source = memorygraph_defaults["operator_candidates"]
    resolved["operator_candidates"] = normalize_memorygraph_operator_candidate_list(
        operator_candidates_source,
        default_operator=resolved["memorygraph_default_operator"],
    )
    if not resolved["operator_candidates"]:
        raise ValueError("operator_candidates must contain at least one memorygraph operator")

    for key, value in loss_cfg.items():
        flat_key = key
        if flat_key in resolved and resolved[flat_key] is not None:
            continue
        resolved[flat_key] = copy.deepcopy(value)

    resolved["threshold_search_metric"] = str(first_not_none(
        resolved.get("threshold_search_metric"),
        backtest_cfg.get("threshold_search_metric"),
        "composite",
    ))
    resolved["allow_timeout_trades"] = parse_bool(
        first_not_none(resolved.get("allow_timeout_trades"), backtest_cfg.get("allow_timeout_trades"), True),
        True,
    )
    default_exit_mode = "realized_event" if str(resolved["label_mode"]) == "triple_barrier" else "fixed_horizon"
    raw_exit_mode = str(first_not_none(
        resolved.get("backtest_exit_mode"),
        backtest_cfg.get("exit_mode"),
        default_exit_mode,
    )).strip().lower().replace("-", "_")
    resolved["memorygraph_requested_backtest_exit_mode"] = raw_exit_mode
    resolved["memorygraph_predicted_exit_diagnostics_enabled"] = True
    if raw_exit_mode == "fixed_horizon":
        resolved["backtest_exit_mode"] = "fixed_horizon"
    elif raw_exit_mode in {"realized_event", "oracle_realized_event"}:
        resolved["backtest_exit_mode"] = "realized_event"
    elif raw_exit_mode == "predicted_horizon":
        # Legacy configs may still request predicted_horizon. Keep the auxiliary heads
        # and diagnostics, but benchmark the model with the entry-only protocol.
        resolved["backtest_exit_mode"] = default_exit_mode
    else:
        resolved["backtest_exit_mode"] = raw_exit_mode

    if resolved["label_mode"] not in {"fixed_horizon", "triple_barrier"}:
        raise ValueError(f"Unsupported label_mode: {resolved['label_mode']}")
    if resolved["objective_mode"] not in {"standard", "execution_aware"}:
        raise ValueError(f"Unsupported objective_mode: {resolved['objective_mode']}")
    if resolved["graph_readout_mode"] not in {"target_only", "target_plus_global", "target_plus_attn_global"}:
        raise ValueError(f"Unsupported graph_readout_mode: {resolved['graph_readout_mode']}")
    if resolved["edge_feature_mode"] not in {"handcrafted_only", "hybrid"}:
        raise ValueError(f"Unsupported edge_feature_mode: {resolved['edge_feature_mode']}")
    if resolved["triple_barrier_pt_sl_mode"] not in {"fixed", "volatility_scaled"}:
        raise ValueError(f"Unsupported triple_barrier_pt_sl_mode: {resolved['triple_barrier_pt_sl_mode']}")
    if resolved["backtest_exit_mode"] not in {"fixed_horizon", "realized_event"}:
        raise ValueError(f"Unsupported backtest_exit_mode: {resolved['backtest_exit_mode']}")
    if resolved["threshold_search_metric"] not in {"pnl_sum", "pnl_per_trade", "sharpe_like", "composite"}:
        raise ValueError(f"Unsupported threshold_search_metric: {resolved['threshold_search_metric']}")
    if (
        str(resolved["label_mode"]) == "triple_barrier"
        and not bool(resolved["triple_barrier_vertical_barrier_use_horizon"])
        and resolved["memorygraph_vertical_barrier_bars"] is None
    ):
        raise ValueError(
            "triple_barrier.vertical_barrier_use_horizon=false requires memorygraph.vertical_barrier_bars "
            "(or top-level memorygraph_vertical_barrier_bars / vertical_barrier_bars)."
        )

    resolved["triple_barrier"] = {
        "pt_sl_mode": resolved["triple_barrier_pt_sl_mode"],
        "upper_barrier_bps": resolved["triple_barrier_upper_barrier_bps"],
        "lower_barrier_bps": resolved["triple_barrier_lower_barrier_bps"],
        "vol_lookback_bars": resolved["triple_barrier_vol_lookback_bars"],
        "vol_barrier_mult_up": resolved["triple_barrier_vol_barrier_mult_up"],
        "vol_barrier_mult_down": resolved["triple_barrier_vol_barrier_mult_down"],
        "min_barrier_bps": resolved["triple_barrier_min_barrier_bps"],
        "max_barrier_bps": resolved["triple_barrier_max_barrier_bps"],
        "vertical_barrier_use_horizon": resolved["triple_barrier_vertical_barrier_use_horizon"],
    }
    resolved["meta_labeling"] = {
        "enabled": resolved["meta_labeling_enabled"],
        "trade_label_requires_first_touch": resolved["trade_label_requires_first_touch"],
        "mask_timeout_for_direction": resolved["mask_timeout_for_direction"],
    }
    resolved["architecture"] = {
        "graph_readout_mode": resolved["graph_readout_mode"],
        "graph_global_pool": resolved["graph_global_pool"],
        "use_target_global_attention": resolved["use_target_global_attention"],
        "edge_feature_mode": resolved["edge_feature_mode"],
        "learned_pairwise_hidden_dim": resolved["learned_pairwise_hidden_dim"],
    }
    resolved["memorygraph"] = {
        "default_operator": resolved["memorygraph_default_operator"],
        "relation_fusion_mode": resolved["memorygraph_relation_fusion_mode"],
        "adjacency_rank": resolved["memorygraph_adjacency_rank"],
        "adjacency_scale": resolved["memorygraph_adjacency_scale"],
        "node_memory_dim": resolved["memorygraph_node_memory_dim"],
        "edge_memory_dim": resolved["memorygraph_edge_memory_dim"],
        "memory_dropout": resolved["memorygraph_memory_dropout"],
        "stateful_training": resolved["memorygraph_stateful_training"],
        "chunk_len": resolved["memorygraph_chunk_len"],
        "detach_interval": resolved["memorygraph_detach_interval"],
        "eval_chunk_len": resolved["memorygraph_eval_chunk_len"],
        "calibration_frac": resolved["memorygraph_calibration_frac"],
        "min_calibration_samples": resolved["memorygraph_min_calibration_samples"],
        "vertical_barrier_bars": resolved["memorygraph_vertical_barrier_bars"],
        "graph_operator": resolved["graph_operator"],
        "operator_candidates": list(resolved["operator_candidates"]),
        "primary_benchmark_role": "entry_only",
        "requested_backtest_exit_mode": resolved["memorygraph_requested_backtest_exit_mode"],
        "predicted_exit_diagnostics_enabled": resolved["memorygraph_predicted_exit_diagnostics_enabled"],
    }
    resolved["loss"] = {
        "loss_w_trade": resolved["loss_w_trade"],
        "loss_w_dir": resolved["loss_w_dir"],
        "loss_w_ret": resolved["loss_w_ret"],
        "loss_w_utility": resolved["loss_w_utility"],
        "loss_w_exit_type": resolved["loss_w_exit_type"],
        "loss_w_tte": resolved["loss_w_tte"],
        "utility_tanh_k": resolved["utility_tanh_k"],
        "false_positive_penalty": resolved["false_positive_penalty"],
        "timeout_penalty": resolved["timeout_penalty"],
        "execution_cost_multiplier": resolved["execution_cost_multiplier"],
        "use_cost_in_label": resolved["use_cost_in_label"],
    }
    resolved["backtest"] = {
        "exit_mode": resolved["backtest_exit_mode"],
        "requested_exit_mode": resolved["memorygraph_requested_backtest_exit_mode"],
        "threshold_search_metric": resolved["threshold_search_metric"],
        "allow_timeout_trades": resolved["allow_timeout_trades"],
    }

    return resolved


def resolve_vertical_barrier_bars(cfg: Dict[str, Any]) -> int:
    if parse_bool(cfg.get("triple_barrier_vertical_barrier_use_horizon"), True):
        return int(HORIZON_BARS)

    value = first_not_none(
        cfg.get("memorygraph_vertical_barrier_bars"),
        get_nested_value(cfg, "memorygraph", {}).get("vertical_barrier_bars"),
        cfg.get("vertical_barrier_bars"),
    )
    if value in {None, ""}:
        raise ValueError(
            "vertical barrier is configured separately, but no vertical_barrier_bars was provided."
        )
    return int(value)


def effective_tte_cap_bars(cfg: Dict[str, Any]) -> int:
    return int(max(1, HORIZON_BARS))


def memorygraph_tbptt_len(cfg: Dict[str, Any]) -> int:
    return int(max(1, min(int(cfg["memorygraph_chunk_len"]), int(cfg["memorygraph_detach_interval"]))))


def split_validation_indices_for_memorygraph(idx_val: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    idx_val = np.asarray(idx_val, dtype=np.int64)
    if idx_val.size < 4:
        raise ValueError("Validation split is too short to separate early stopping and threshold calibration.")

    sorted_idx = np.sort(idx_val)
    calibration_frac = float(cfg["memorygraph_calibration_frac"])
    requested_calibration = max(
        int(round(sorted_idx.size * calibration_frac)),
        int(cfg["memorygraph_min_calibration_samples"]),
    )
    calibration_count = min(max(1, requested_calibration), sorted_idx.size - 1)
    early_stop_count = sorted_idx.size - calibration_count
    if early_stop_count <= 0:
        raise ValueError("Calibration split consumed the entire validation window.")

    early_stop_idx = sorted_idx[:early_stop_count].astype(np.int64)
    calibration_idx = sorted_idx[early_stop_count:].astype(np.int64)
    if early_stop_idx.size == 0 or calibration_idx.size == 0:
        raise ValueError("Both early_stop_val and threshold_calibration_val must be non-empty.")
    return early_stop_idx, calibration_idx


def split_indices_into_contiguous_segments(indices: np.ndarray) -> List[np.ndarray]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return []

    sorted_idx = np.sort(indices)
    segments: List[List[int]] = [[int(sorted_idx[0])]]
    for current in sorted_idx[1:]:
        current = int(current)
        if current == segments[-1][-1] + 1:
            segments[-1].append(current)
        else:
            segments.append([current])
    return [np.asarray(segment, dtype=np.int64) for segment in segments if segment]


@dataclass
class StatefulChunk:
    sample_indices: np.ndarray
    is_segment_start: bool


def build_stateful_chunks(indices: np.ndarray, cfg: Dict[str, Any], for_eval: bool = False) -> List[StatefulChunk]:
    chunk_len = int(cfg["memorygraph_eval_chunk_len"] if for_eval else memorygraph_tbptt_len(cfg))
    chunks: List[StatefulChunk] = []
    for segment in split_indices_into_contiguous_segments(indices):
        start = 0
        while start < len(segment):
            stop = min(start + chunk_len, len(segment))
            chunks.append(
                StatefulChunk(
                    sample_indices=segment[start:stop].astype(np.int64),
                    is_segment_start=bool(start == 0),
                )
            )
            start = stop
    return chunks


def stateful_chunk_to_batch(
    chunk_sample_indices: np.ndarray,
    x_node_scaled: np.ndarray,
    x_rel_edge_scaled: np.ndarray,
) -> Dict[str, torch.Tensor]:
    chunk_sample_indices = np.asarray(chunk_sample_indices, dtype=np.int64)
    raw_t = SAMPLE_T[chunk_sample_indices].astype(np.int64)
    batch = {
        "x_node_seq": torch.from_numpy(x_node_scaled[raw_t]).unsqueeze(0),
        "x_edge_seq": torch.from_numpy(x_rel_edge_scaled[raw_t]).unsqueeze(0),
        "y_ret": torch.from_numpy(Y_RET[raw_t]).unsqueeze(0),
        "y_trade": torch.from_numpy(Y_TRADE[raw_t]).unsqueeze(0),
        "y_dir": torch.from_numpy(Y_DIR[raw_t]).unsqueeze(0),
        "y_dir_mask": torch.from_numpy(Y_DIR_MASK[raw_t]).unsqueeze(0),
        "y_exit_type": torch.from_numpy(Y_EXIT_TYPE[raw_t]).unsqueeze(0),
        "y_tte": torch.from_numpy(Y_TTE[raw_t]).unsqueeze(0),
        "sample_idx": torch.from_numpy(chunk_sample_indices).unsqueeze(0),
        "raw_t": torch.from_numpy(raw_t).unsqueeze(0),
    }
    return batch


def detach_memorygraph_state(state: Optional[Dict[str, torch.Tensor]]) -> Optional[Dict[str, torch.Tensor]]:
    if state is None:
        return None
    return {
        key: value.detach()
        for key, value in state.items()
    }

# %% Run metadata and environment metadata

def initialize_runtime_globals(cfg: Dict[str, Any]) -> None:
    global CFG, ASSETS, TARGET_ASSET, ASSET2IDX, TARGET_NODE
    global ARTIFACT_ROOT, ARTIFACT_BASE_ROOT, RUN_ID, CONFIG_PATH, DEVICE
    global FREQ, HORIZON_MINUTES, HORIZON_BARS, LOOKBACK_BARS, RELATION_WINDOWS, RELATION_LAGS, PURGE_GAP_BARS, EXPECTED_DELTA
    global EDGE_LIST, EDGE_NAMES, EDGE_INDEX, EDGE_SRC_IDX, EDGE_DST_IDX

    CFG = copy.deepcopy(cfg)
    RUN_ID = str(cfg["run_id"])
    CONFIG_PATH = Path(cfg["config_path"])
    ARTIFACT_BASE_ROOT = Path(cfg["artifact_root_base"])
    ARTIFACT_ROOT = ensure_dir(Path(cfg["artifact_root"]))
    ASSETS = list(CFG["assets"])
    TARGET_ASSET = str(CFG["target_asset"])
    if TARGET_ASSET not in ASSETS:
        raise AssertionError("Target asset must be one of the configured assets.")
    ASSET2IDX = {asset: i for i, asset in enumerate(ASSETS)}
    TARGET_NODE = ASSET2IDX[TARGET_ASSET]

    FREQ = normalize_freq_name(CFG["freq"])
    HORIZON_MINUTES = int(CFG["horizon_minutes"])
    HORIZON_BARS = horizon_bars_from_clock_minutes(FREQ, HORIZON_MINUTES)
    LOOKBACK_BARS = get_freq_specific_lookback(CFG)
    RELATION_WINDOWS = get_freq_specific_relation_windows(CFG)
    RELATION_LAGS = [int(x) for x in CFG["relation_lags_bars"]]
    PURGE_GAP_BARS = HORIZON_BARS + int(CFG["purge_gap_extra_bars"])
    EXPECTED_DELTA = expected_timedelta(FREQ)

    EDGE_LIST = build_edge_list(ASSETS, add_self_loops=True)
    EDGE_NAMES = [f"{src}->{dst}" for src, dst in EDGE_LIST]
    EDGE_INDEX = torch.tensor([[ASSET2IDX[src], ASSET2IDX[dst]] for src, dst in EDGE_LIST], dtype=torch.long)
    EDGE_SRC_IDX = EDGE_INDEX[:, 0]
    EDGE_DST_IDX = EDGE_INDEX[:, 1]

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_threads = int(CFG.get("num_threads") or max(1, (os.cpu_count() or 4) // 2))
    seed = int(CFG.get("seed") or 1001)
    torch.set_num_threads(max(1, num_threads))
    seed_everything(seed)

    LOGGER.info("DEVICE=%s", DEVICE)
    LOGGER.info("SEED=%s | NUM_THREADS=%s", seed, max(1, num_threads))
    LOGGER.info("ASSETS=%s | TARGET_ASSET=%s", ASSETS, TARGET_ASSET)
    LOGGER.info("ARTIFACT_ROOT=%s", ARTIFACT_ROOT.resolve())
    LOGGER.info(
        "FREQ=%s | HORIZON_BARS=%s | LOOKBACK_BARS=%s | RELATION_WINDOWS=%s | RELATION_LAGS=%s | PURGE_GAP_BARS=%s",
        FREQ,
        HORIZON_BARS,
        LOOKBACK_BARS,
        RELATION_WINDOWS,
        RELATION_LAGS,
        PURGE_GAP_BARS,
    )


def collect_environment_metadata(run_id: str, run_start_time_utc: str) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "machine_type": CFG.get("machine_type") or None,
        "region": CFG.get("gcp_region") or None,
        "container_image": CFG.get("container_image") or None,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "os_name": os.name,
        "cpu_count": os.cpu_count(),
        "cuda_available": bool(torch.cuda.is_available()),
        "current_device": str(DEVICE),
        "git_commit_sha": get_git_commit_sha(),
        "run_start_time_utc": run_start_time_utc,
        "run_end_time_utc": None,
        "total_runtime_seconds": None,
    }


def finalize_environment_metadata(
    metadata: Dict[str, Any],
    run_end_time_utc: str,
    total_runtime_seconds: float,
) -> Dict[str, Any]:
    out = copy.deepcopy(metadata)
    out["run_end_time_utc"] = run_end_time_utc
    out["total_runtime_seconds"] = float(total_runtime_seconds)
    return out

# %% GCS helpers

def get_smtp_settings(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "host": str(cfg.get("smtp_host") or "").strip(),
        "port": int(cfg.get("smtp_port") or 587),
        "user": str(cfg.get("smtp_user") or "").strip(),
        "password": str(cfg.get("smtp_password") or "").strip(),
        "use_tls": bool(cfg.get("smtp_use_tls")),
        "from_email": str(cfg.get("email_from") or cfg.get("smtp_user") or "").strip(),
    }


def send_email_report(
    subject: str,
    html_body: str,
    to_email: str,
    attachments: List[Path],
    smtp_settings: Dict[str, Any],
) -> None:
    """Send HTML email with optional attachments over SMTP.

    Args:
        subject: Email subject line.
        html_body: HTML body content.
        to_email: Recipient email or comma-separated list.
        attachments: Files to attach if they exist.
        smtp_settings: SMTP configuration with host, port, user, password, use_tls, from_email.
    """
    host = str(smtp_settings.get("host") or "").strip()
    port = int(smtp_settings.get("port") or 587)
    user = str(smtp_settings.get("user") or "").strip()
    password = str(smtp_settings.get("password") or "").strip()
    use_tls = bool(smtp_settings.get("use_tls"))
    from_email = str(smtp_settings.get("from_email") or user).strip()
    recipients = [x.strip() for x in str(to_email or "").split(",") if x.strip()]

    if not host or not recipients or not from_email:
        LOGGER.warning("Email sending skipped because SMTP/email settings are incomplete.")
        return

    message = MIMEMultipart()
    message["Subject"] = subject
    message["From"] = from_email
    message["To"] = ", ".join(recipients)
    message.attach(MIMEText(html_body, "html", "utf-8"))

    for attachment in attachments:
        if not attachment.exists():
            LOGGER.warning("Skipping missing attachment: %s", attachment)
            continue
        part = MIMEBase("application", "octet-stream")
        with open(attachment, "rb") as f:
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{attachment.name}"')
        message.attach(part)

    with smtplib.SMTP(host, port, timeout=30) as smtp:
        smtp.ehlo()
        if use_tls:
            smtp.starttls()
            smtp.ehlo()
        if user:
            smtp.login(user, password)
        smtp.sendmail(from_email, recipients, message.as_string())
    LOGGER.info("Email report sent to %s", recipients)

# %% Frequency and data settings helpers

def seed_everything(seed: int = 1001) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_freq_name(freq: str) -> str:
    f = str(freq).strip().lower()
    alias_map = {
        "1s": "1sec",
        "1sec": "1sec",
        "1second": "1sec",
        "1m": "1min",
        "1min": "1min",
        "1minute": "1min",
        "5m": "5min",
        "5min": "5min",
        "5minute": "5min",
    }
    if f not in alias_map:
        raise ValueError(f"Unsupported frequency: {freq}")
    return alias_map[f]


def freq_to_seconds(freq: str) -> int:
    f = normalize_freq_name(freq)
    return {"1sec": 1, "1min": 60, "5min": 300}[f]


def expected_timedelta(freq: str) -> pd.Timedelta:
    return pd.Timedelta(seconds=freq_to_seconds(freq))


def horizon_bars_from_clock_minutes(freq: str, horizon_minutes: int = 5) -> int:
    horizon_seconds = int(horizon_minutes) * 60
    bar_seconds = freq_to_seconds(freq)
    if horizon_seconds % bar_seconds != 0:
        raise ValueError(
            f"Horizon of {horizon_minutes} minutes is not an integer number of bars for freq={freq}"
        )
    return horizon_seconds // bar_seconds


def get_freq_specific_lookback(cfg: Dict[str, Any]) -> int:
    freq = normalize_freq_name(cfg["freq"])
    return int(cfg["lookback_bars_by_freq"][freq])


def get_freq_specific_relation_windows(cfg: Dict[str, Any]) -> List[int]:
    freq = normalize_freq_name(cfg["freq"])
    return [int(x) for x in cfg["relation_windows_bars_by_freq"][freq]]


RELATION_NAMES: List[str] = ["price_dep", "order_flow", "liquidity"]


def build_edge_list(assets: List[str], add_self_loops: bool = True) -> List[Tuple[str, str]]:
    edges = [(src, dst) for src in assets for dst in assets if src != dst]
    if add_self_loops:
        edges.extend([(a, a) for a in assets])
    return edges


def initialize_tensor_state(cfg: Dict[str, Any]) -> None:
    global df, TIMESTAMPS, X_NODE_RAW, NODE_FEATURE_NAMES, RELATION_STATE_MAP
    global X_REL_EDGE_RAW, EDGE_FEATURE_NAMES, TARGET_MID, TARGET_LR_1BAR
    global Y_RET, Y_DIR, Y_TRADE, Y_DIR_MASK, Y_EXIT_TYPE, Y_TTE, Y_TIMEOUT, TARGET_SUMMARY
    global TRADE_LABEL_ABS_RETURN_THRESHOLD
    global T, FIRST_VALID_T, LAST_VALID_T, SAMPLE_T, N_SAMPLES
    global IDX_PREHOLDOUT, IDX_HOLDOUT, WALK_FORWARD_SPLITS, IDX_TRAIN_FINAL, IDX_VAL_FINAL, IDX_TEST_FINAL

    df = load_and_align_assets(cfg)
    TIMESTAMPS = pd.to_datetime(df["timestamp"], utc=True)

    X_NODE_RAW, NODE_FEATURE_NAMES, RELATION_STATE_MAP = build_node_features_and_relation_states(df, cfg)
    X_REL_EDGE_RAW, EDGE_FEATURE_NAMES = build_multigraph_relation_tensor(
        relation_state_map=RELATION_STATE_MAP,
        edge_list=EDGE_LIST,
        windows=RELATION_WINDOWS,
        lags=RELATION_LAGS,
        use_fisher_z=bool(cfg["use_fisher_z_for_corr"]),
    )

    TARGET_MID = df[f"mid_{TARGET_ASSET}"].to_numpy(dtype=np.float64)
    TARGET_LR_1BAR = df[f"lr_{TARGET_ASSET}"].to_numpy(dtype=np.float64)

    target_pack = build_supervision_targets(TARGET_MID, cfg)
    Y_RET = target_pack["y_ret"].astype(np.float32)
    Y_DIR = target_pack["y_dir"].astype(np.float32)
    Y_TRADE = target_pack["y_trade"].astype(np.float32)
    Y_DIR_MASK = target_pack["y_dir_mask"].astype(np.float32)
    Y_EXIT_TYPE = target_pack["y_exit_type"].astype(np.int64)
    Y_TTE = target_pack["y_tte"].astype(np.float32)
    Y_TIMEOUT = target_pack["y_timeout"].astype(np.float32)
    TARGET_SUMMARY = copy.deepcopy(target_pack["summary"])
    TRADE_LABEL_ABS_RETURN_THRESHOLD = float(target_pack["trade_label_abs_return_threshold"])

    T = len(df)
    FIRST_VALID_T = LOOKBACK_BARS - 1
    LAST_VALID_T = T - HORIZON_BARS - 1
    if LAST_VALID_T < FIRST_VALID_T:
        raise RuntimeError(
            f"Not enough rows after slicing. Need lookback={LOOKBACK_BARS} and horizon={HORIZON_BARS}."
        )

    SAMPLE_T = np.arange(FIRST_VALID_T, LAST_VALID_T + 1, dtype=np.int64)
    N_SAMPLES = len(SAMPLE_T)

    split_bundle = build_or_load_split_bundle(
        cfg=cfg,
        n_samples=N_SAMPLES,
        gap_bars=PURGE_GAP_BARS,
        sample_t=SAMPLE_T,
        timestamps=TIMESTAMPS,
        y_trade=Y_TRADE,
        y_dir=Y_DIR,
        y_dir_mask=Y_DIR_MASK,
        y_exit_type=Y_EXIT_TYPE,
        y_tte=Y_TTE,
        target_summary=TARGET_SUMMARY,
        artifact_root=ARTIFACT_ROOT,
        logger=LOGGER,
    )
    IDX_PREHOLDOUT = split_bundle.idx_preholdout
    IDX_HOLDOUT = split_bundle.idx_holdout
    WALK_FORWARD_SPLITS = split_bundle.walk_forward_splits
    IDX_TRAIN_FINAL = split_bundle.idx_train_final
    IDX_VAL_FINAL = split_bundle.idx_val_final
    IDX_TEST_FINAL = split_bundle.idx_test_final

    LOGGER.info("Aligned dataframe shape=%s | time range=%s -> %s", df.shape, TIMESTAMPS.iloc[0], TIMESTAMPS.iloc[-1])
    LOGGER.info("X_NODE_RAW shape=%s | X_REL_EDGE_RAW shape=%s", X_NODE_RAW.shape, X_REL_EDGE_RAW.shape)
    LOGGER.info(
        "Label mode=%s | objective_mode=%s | valid_target_count=%s | trade threshold=%s | target_summary=%s",
        cfg["label_mode"],
        cfg["objective_mode"],
        int(np.isfinite(Y_RET).sum()),
        TRADE_LABEL_ABS_RETURN_THRESHOLD,
        TARGET_SUMMARY,
    )
    LOGGER.info(
        "Samples: T=%s | FIRST_VALID_T=%s | LAST_VALID_T=%s | N_SAMPLES=%s",
        T,
        FIRST_VALID_T,
        LAST_VALID_T,
        N_SAMPLES,
    )
    LOGGER.info(
        "Splits: preholdout=%s | holdout=%s | num_folds=%s | train_final=%s | val_final=%s | holdout_final=%s",
        len(IDX_PREHOLDOUT),
        len(IDX_HOLDOUT),
        len(WALK_FORWARD_SPLITS),
        len(IDX_TRAIN_FINAL),
        len(IDX_VAL_FINAL),
        len(IDX_TEST_FINAL),
    )

# %% Data loading and alignment

EPS = 1e-12


def choose_existing_column(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(f"Missing required column for {what}. Tried: {candidates}")


def infer_timestamp_column(df: pd.DataFrame) -> str:
    return choose_existing_column(
        df,
        ["system_time", "timestamp", "time", "datetime"],
        "timestamp",
    )


def infer_book_column(df: pd.DataFrame, side_prefix: str, level: int) -> str:
    candidates = [
        f"{side_prefix}_notional_{level}",
        f"{side_prefix}_vol_{level}",
        f"{side_prefix}_{level}",
    ]
    return choose_existing_column(df, candidates, f"{side_prefix} level {level}")


def validate_regular_time_index(
    frame: pd.DataFrame,
    expected_delta: pd.Timedelta,
    name: str,
    *,
    fill_missing: bool = True,
    log_limit: int = 10,
) -> pd.DataFrame:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{name}: expected DatetimeIndex, got {type(frame.index)}")
    if len(frame.index) < 2:
        raise ValueError(f"{name}: need at least 2 timestamps for regularity checks")

    frame = frame.sort_index()

    if frame.index.has_duplicates:
        dup_count = int(frame.index.duplicated().sum())
        raise ValueError(f"{name}: time index contains {dup_count} duplicate timestamps")

    diffs = frame.index.to_series().diff().dropna()
    bad_mask = diffs != expected_delta
    if bad_mask.any():
        bad_positions = np.where(bad_mask.to_numpy())[0][:5]
        irregular_examples = []
        for pos in bad_positions:
            irregular_examples.append(
                {
                    "prev": str(frame.index[pos]),
                    "curr": str(frame.index[pos + 1]),
                    "gap": str(diffs.iloc[pos]),
                }
            )

        if not fill_missing:
            raise ValueError(
                f"{name}: irregular time index for configured freq={FREQ}. "
                f"Expected every gap to be {expected_delta}, but found {int(bad_mask.sum())} irregular gaps. "
                f"Examples: {irregular_examples}"
            )

        full_index = pd.date_range(
            start=frame.index.min(),
            end=frame.index.max(),
            freq=expected_delta,
            tz=frame.index.tz,
        )
        missing_index = full_index.difference(frame.index)
        if len(missing_index) == 0:
            raise ValueError(
                f"{name}: irregular index detected but missing timestamps could not be inferred. "
                f"Examples: {irregular_examples}"
            )

        frame = frame.reindex(full_index).ffill()
        inserted_examples = [str(ts) for ts in missing_index[:log_limit]]
        print(
            f"[{name}] Forward-fill repair: inserted {len(missing_index)} missing rows "
            f"for freq={FREQ}. Example timestamps: {inserted_examples}"
        )

    repaired_diffs = frame.index.to_series().diff().dropna()
    if (repaired_diffs != expected_delta).any():
        raise RuntimeError(
            f"{name}: failed to regularize time index. Expected all deltas to be {expected_delta}."
        )

    inferred = repaired_diffs.mode().iloc[0]
    if inferred != expected_delta:
        raise ValueError(f"{name}: inferred step {inferred} does not match expected {expected_delta}")

    return frame


def load_one_asset_raw(asset: str, cfg: Dict[str, Any]) -> pd.DataFrame:
    freq = normalize_freq_name(cfg["freq"])
    path = Path(cfg["data_dir"]) / f"{asset}_{freq}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV for asset={asset}: {path}")

    raw = pd.read_csv(path)
    start = int(len(raw) * float(cfg["data_slice_start_frac"]))
    end = int(len(raw) * float(cfg["data_slice_end_frac"]))
    raw = raw.iloc[start:end].copy()

    ts_col = infer_timestamp_column(raw)
    if freq == "1sec":
        raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce").dt.floor("1s")
    elif freq == "1min":
        raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce").dt.floor("min")
    elif freq == "5min":
        raw["timestamp"] = pd.to_datetime(raw[ts_col], utc=True, errors="coerce").dt.floor("5min")

    raw = raw.dropna(subset=["timestamp"]).sort_values("timestamp")
    raw = raw[~raw["timestamp"].duplicated(keep="last")].copy()
    raw = raw.set_index("timestamp")
    raw = validate_regular_time_index(raw, EXPECTED_DELTA, name=f"{asset} raw", fill_missing=True)

    midpoint_col = choose_existing_column(raw, ["midpoint", "mid", "price"], "midpoint")
    spread_col = choose_existing_column(raw, ["spread"], "spread")
    buys_col = choose_existing_column(raw, ["buys"], "buys")
    sells_col = choose_existing_column(raw, ["sells"], "sells")

    out = pd.DataFrame(index=raw.index)
    out[f"mid_{asset}"] = pd.to_numeric(raw[midpoint_col], errors="coerce")
    out[f"spread_{asset}"] = pd.to_numeric(raw[spread_col], errors="coerce")
    out[f"buys_{asset}"] = pd.to_numeric(raw[buys_col], errors="coerce")
    out[f"sells_{asset}"] = pd.to_numeric(raw[sells_col], errors="coerce")

    book_levels = int(cfg["book_levels"])
    for level in range(book_levels):
        bid_col = infer_book_column(raw, "bids", level)
        ask_col = infer_book_column(raw, "asks", level)
        out[f"bids_notional_{asset}_{level}"] = pd.to_numeric(raw[bid_col], errors="coerce")
        out[f"asks_notional_{asset}_{level}"] = pd.to_numeric(raw[ask_col], errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan)
    missing_counts = out.isna().sum()
    if missing_counts.any():
        bad_cols = missing_counts[missing_counts > 0].to_dict()
        raise ValueError(
            f"{asset}: required columns contain NaNs after parsing. Columns with missing values: {bad_cols}"
        )

    out = validate_regular_time_index(out, EXPECTED_DELTA, name=f"{asset} standardized", fill_missing=True)
    return out


def load_and_align_assets(cfg: Dict[str, Any]) -> pd.DataFrame:
    aligned: Optional[pd.DataFrame] = None
    for asset in ASSETS:
        one = load_one_asset_raw(asset, cfg)
        aligned = one if aligned is None else aligned.join(one, how="inner")

    if aligned is None or len(aligned) == 0:
        raise RuntimeError("No data available after multi-asset loading and alignment")

    aligned = aligned.sort_index()
    aligned = aligned[~aligned.index.duplicated(keep="last")].copy()
    aligned = validate_regular_time_index(
        aligned,
        EXPECTED_DELTA,
        name="aligned multigraph index",
        fill_missing=True,
    )

    for asset in ASSETS:
        log_mid = np.log(aligned[f"mid_{asset}"].astype(float).to_numpy() + EPS)
        lr = np.zeros(len(aligned), dtype=np.float64)
        lr[1:] = np.diff(log_mid)
        aligned[f"lr_{asset}"] = lr.astype(np.float32)

    aligned = aligned.reset_index().rename(columns={"index": "timestamp"})
    return aligned



# %% Feature engineering

def safe_log1p_np(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(x, 0.0))


def bounded_log_ratio(num: np.ndarray, den: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return np.log((num + eps) / (den + eps))


def build_node_features_and_relation_states(
    df_: pd.DataFrame,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, List[str], Dict[str, Dict[str, np.ndarray]]]:
    book_levels = int(cfg["book_levels"])
    top_levels = int(cfg["top_levels"])
    near_levels = int(cfg["near_levels"])

    if top_levels > book_levels:
        raise ValueError("top_levels must be <= book_levels")
    if near_levels >= book_levels:
        raise ValueError("near_levels must be < book_levels")

    node_feature_names = [
        "lr_1bar",
        "rel_spread",
        "log_buys",
        "log_sells",
        "flow_imbalance",
        "depth_imbalance_total",
        "top_imbalance_0",
        "top_imbalance_1",
        "top_imbalance_2",
        "top_imbalance_3",
        "top_imbalance_4",
        "bid_near_far_ratio",
        "ask_near_far_ratio",
        "depth_imbalance_near",
        "depth_imbalance_far",
    ]

    per_asset_node_features: List[np.ndarray] = []
    relation_state_map: Dict[str, Dict[str, np.ndarray]] = {rel: {} for rel in RELATION_NAMES}

    for asset in ASSETS:
        lr = df_[f"lr_{asset}"].to_numpy(dtype=np.float32)
        mid = df_[f"mid_{asset}"].to_numpy(dtype=np.float32)
        spread = df_[f"spread_{asset}"].to_numpy(dtype=np.float32)
        buys = df_[f"buys_{asset}"].to_numpy(dtype=np.float32)
        sells = df_[f"sells_{asset}"].to_numpy(dtype=np.float32)

        rel_spread = spread / (mid + EPS)
        log_buys = safe_log1p_np(buys).astype(np.float32)
        log_sells = safe_log1p_np(sells).astype(np.float32)
        flow_imbalance = ((buys - sells) / (buys + sells + EPS)).astype(np.float32)

        bids = np.stack(
            [df_[f"bids_notional_{asset}_{i}"].to_numpy(dtype=np.float32) for i in range(book_levels)],
            axis=1,
        )
        asks = np.stack(
            [df_[f"asks_notional_{asset}_{i}"].to_numpy(dtype=np.float32) for i in range(book_levels)],
            axis=1,
        )

        bid_total = bids.sum(axis=1)
        ask_total = asks.sum(axis=1)
        depth_imbalance_total = ((bid_total - ask_total) / (bid_total + ask_total + EPS)).astype(np.float32)

        top_imbalances = []
        for i in range(top_levels):
            bi = bids[:, i]
            ai = asks[:, i]
            top_imbalances.append(((bi - ai) / (bi + ai + EPS)).astype(np.float32))
        top_imbalances = np.stack(top_imbalances, axis=1)

        bid_near = bids[:, :near_levels].sum(axis=1)
        ask_near = asks[:, :near_levels].sum(axis=1)
        bid_far = bids[:, near_levels:].sum(axis=1)
        ask_far = asks[:, near_levels:].sum(axis=1)

        bid_near_far_ratio = (bid_near / (bid_far + EPS)).astype(np.float32)
        ask_near_far_ratio = (ask_near / (ask_far + EPS)).astype(np.float32)
        depth_imbalance_near = ((bid_near - ask_near) / (bid_near + ask_near + EPS)).astype(np.float32)
        depth_imbalance_far = ((bid_far - ask_far) / (bid_far + ask_far + EPS)).astype(np.float32)

        asset_node = np.column_stack(
            [
                lr,
                rel_spread,
                log_buys,
                log_sells,
                flow_imbalance,
                depth_imbalance_total,
                top_imbalances[:, 0],
                top_imbalances[:, 1],
                top_imbalances[:, 2],
                top_imbalances[:, 3],
                top_imbalances[:, 4],
                bid_near_far_ratio,
                ask_near_far_ratio,
                depth_imbalance_near,
                depth_imbalance_far,
            ]
        ).astype(np.float32)

        per_asset_node_features.append(asset_node)

        price_state = lr.astype(np.float32)
        turnover_log = safe_log1p_np(buys + sells).astype(np.float32)
        flow_state = (flow_imbalance * turnover_log).astype(np.float32)

        depth_shape = np.tanh(
            bounded_log_ratio(bid_near_far_ratio + 1.0, ask_near_far_ratio + 1.0)
        ).astype(np.float32)
        liquidity_state = (
            -np.log1p(np.maximum(rel_spread, 0.0) * 1e4)
            + 0.50 * depth_imbalance_total
            + 0.25 * depth_imbalance_near
            + 0.25 * depth_shape
        ).astype(np.float32)

        relation_state_map["price_dep"][asset] = np.nan_to_num(price_state, nan=0.0, posinf=0.0, neginf=0.0)
        relation_state_map["order_flow"][asset] = np.nan_to_num(flow_state, nan=0.0, posinf=0.0, neginf=0.0)
        relation_state_map["liquidity"][asset] = np.nan_to_num(liquidity_state, nan=0.0, posinf=0.0, neginf=0.0)

    x_node = np.stack(per_asset_node_features, axis=1).astype(np.float32)
    x_node = np.nan_to_num(x_node, nan=0.0, posinf=0.0, neginf=0.0)
    return x_node, node_feature_names, relation_state_map



# %% Relation tensor construction

def fisher_z_transform(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -0.999999, 0.999999)
    return 0.5 * np.log((1.0 + x) / (1.0 - x))


def rolling_dependence_feature_matrix(
    src_series: pd.Series,
    dst_series: pd.Series,
    windows: List[int],
    lags: List[int],
    use_fisher_z: bool = True,
) -> np.ndarray:
    features: List[np.ndarray] = []

    src_series = src_series.astype(float)
    dst_series = dst_series.astype(float)

    for lag in lags:
        shifted_src = src_series.shift(int(lag)) if int(lag) > 0 else src_series
        for window in windows:
            window_int = int(window)
            min_periods = max(1, min(window_int, max(3, window_int // 2)))

            corr = shifted_src.rolling(window=window_int, min_periods=min_periods).corr(dst_series)
            cov = shifted_src.rolling(window=window_int, min_periods=min_periods).cov(dst_series)
            var = shifted_src.rolling(window=window_int, min_periods=min_periods).var()
            mean_prod = (shifted_src * dst_series).rolling(window=window_int, min_periods=min_periods).mean()

            corr_arr = corr.to_numpy(dtype=np.float64)
            beta_arr = (cov / (var + EPS)).to_numpy(dtype=np.float64)
            mean_prod_arr = mean_prod.to_numpy(dtype=np.float64)

            corr_arr = np.nan_to_num(corr_arr, nan=0.0, posinf=0.0, neginf=0.0)
            beta_arr = np.nan_to_num(beta_arr, nan=0.0, posinf=0.0, neginf=0.0)
            mean_prod_arr = np.nan_to_num(mean_prod_arr, nan=0.0, posinf=0.0, neginf=0.0)

            if use_fisher_z:
                corr_arr = fisher_z_transform(corr_arr)

            features.extend(
                [
                    corr_arr.astype(np.float32),
                    beta_arr.astype(np.float32),
                    mean_prod_arr.astype(np.float32),
                ]
            )

    return np.stack(features, axis=1).astype(np.float32)


def build_multigraph_relation_tensor(
    relation_state_map: Dict[str, Dict[str, np.ndarray]],
    edge_list: List[Tuple[str, str]],
    windows: List[int],
    lags: List[int],
    use_fisher_z: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    relation_tensors: List[np.ndarray] = []
    edge_feature_names: List[str] = []

    for lag in lags:
        for window in windows:
            edge_feature_names.extend(
                [
                    f"lag{lag}_win{window}_corr",
                    f"lag{lag}_win{window}_beta",
                    f"lag{lag}_win{window}_meanprod",
                ]
            )

    for rel in RELATION_NAMES:
        per_edge = []
        for src, dst in edge_list:
            src_series = pd.Series(relation_state_map[rel][src])
            dst_series = pd.Series(relation_state_map[rel][dst])
            edge_mat = rolling_dependence_feature_matrix(
                src_series=src_series,
                dst_series=dst_series,
                windows=windows,
                lags=lags,
                use_fisher_z=use_fisher_z,
            )
            per_edge.append(edge_mat)

        rel_tensor = np.stack(per_edge, axis=1).astype(np.float32)
        rel_tensor = np.nan_to_num(rel_tensor, nan=0.0, posinf=0.0, neginf=0.0)
        relation_tensors.append(rel_tensor)

    x_rel_edge = np.stack(relation_tensors, axis=1).astype(np.float32)
    return x_rel_edge, edge_feature_names




# %% Target construction

def forward_log_return_from_mid(mid: np.ndarray, horizon_bars: int) -> np.ndarray:
    if horizon_bars <= 0:
        raise ValueError("horizon_bars must be positive")

    mid = np.asarray(mid, dtype=np.float64)
    out = np.full(len(mid), np.nan, dtype=np.float32)
    log_mid = np.log(mid + EPS)

    if len(mid) <= horizon_bars:
        return out

    out[:-horizon_bars] = (log_mid[horizon_bars:] - log_mid[:-horizon_bars]).astype(np.float32)
    return out



def round_trip_cost_as_log_return(cost_bps_per_side: float) -> float:
    return 3.0 * float(cost_bps_per_side) * 1e-4



def compute_trade_edge_threshold_log_return(cfg: Dict[str, Any]) -> float:
    threshold = float(cfg.get("trade_label_buffer_bps", 0.0)) * 1e-4
    if parse_bool(cfg.get("use_cost_in_label"), True):
        threshold += float(cfg.get("execution_cost_multiplier", 1.0)) * round_trip_cost_as_log_return(
            float(cfg.get("cost_bps_per_side", 0.0))
        )
    return float(threshold)



def compute_triple_barrier_bps(mid: np.ndarray, cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    n = len(mid)
    default_up = float(cfg["triple_barrier_upper_barrier_bps"])
    default_down = float(cfg["triple_barrier_lower_barrier_bps"])
    pt_sl_mode = str(cfg["triple_barrier_pt_sl_mode"])

    if pt_sl_mode == "fixed":
        return (
            np.full(n, default_up, dtype=np.float64),
            np.full(n, default_down, dtype=np.float64),
        )

    log_mid = np.log(np.asarray(mid, dtype=np.float64) + EPS)
    lr_1bar = np.zeros(n, dtype=np.float64)
    if n > 1:
        lr_1bar[1:] = np.diff(log_mid)

    lookback = max(3, int(cfg["triple_barrier_vol_lookback_bars"]))
    vol = pd.Series(lr_1bar).rolling(
        window=lookback,
        min_periods=max(3, lookback // 3),
    ).std().to_numpy(dtype=np.float64)
    vol_bps = np.abs(vol) * 1e4

    up = vol_bps * float(cfg["triple_barrier_vol_barrier_mult_up"])
    down = vol_bps * float(cfg["triple_barrier_vol_barrier_mult_down"])
    up = np.clip(up, float(cfg["triple_barrier_min_barrier_bps"]), float(cfg["triple_barrier_max_barrier_bps"]))
    down = np.clip(down, float(cfg["triple_barrier_min_barrier_bps"]), float(cfg["triple_barrier_max_barrier_bps"]))

    up = np.where(np.isfinite(up), up, default_up)
    down = np.where(np.isfinite(down), down, default_down)
    return up.astype(np.float64), down.astype(np.float64)



def build_fixed_horizon_targets(mid: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    y_ret = forward_log_return_from_mid(mid, horizon_bars=HORIZON_BARS).astype(np.float32)
    valid_mask = np.isfinite(y_ret)
    meta_labeling_enabled = parse_bool(cfg.get("meta_labeling_enabled"), True)

    threshold = compute_trade_edge_threshold_log_return(cfg)
    y_trade = np.full(len(mid), np.nan, dtype=np.float32)
    if meta_labeling_enabled:
        y_trade[valid_mask] = (np.abs(y_ret[valid_mask]) > threshold).astype(np.float32)
    else:
        y_trade[valid_mask] = 1.0

    y_dir = np.full(len(mid), 0.5, dtype=np.float32)
    y_dir[valid_mask & (y_ret > 0.0)] = 1.0
    y_dir[valid_mask & (y_ret < 0.0)] = 0.0

    y_dir_mask = np.zeros(len(mid), dtype=np.float32)
    if meta_labeling_enabled:
        y_dir_mask[valid_mask & (y_trade > 0.5)] = 1.0
    else:
        y_dir_mask[valid_mask] = 1.0

    y_exit_type = np.full(len(mid), -1, dtype=np.int64)
    y_exit_type[valid_mask] = EXIT_TYPE_TO_IDX["vertical"]

    y_tte = np.full(len(mid), np.nan, dtype=np.float32)
    y_tte[valid_mask] = float(HORIZON_BARS)

    y_timeout = np.zeros(len(mid), dtype=np.float32)
    y_timeout[valid_mask] = 1.0

    summary = {
        "label_mode": "fixed_horizon",
        "n_valid": int(valid_mask.sum()),
        "n_trade": int(np.nansum(y_trade > 0.5)),
        "n_upper": 0,
        "n_lower": 0,
        "n_vertical": int(valid_mask.sum()),
        "trade_rate": float(np.nanmean(y_trade[valid_mask])) if valid_mask.any() else float("nan"),
        "avg_tte_bars": float(np.nanmean(y_tte[valid_mask])) if valid_mask.any() else float("nan"),
    }
    return {
        "y_ret": y_ret,
        "y_dir": y_dir,
        "y_trade": y_trade,
        "y_dir_mask": y_dir_mask,
        "y_exit_type": y_exit_type,
        "y_tte": y_tte,
        "y_timeout": y_timeout,
        "summary": summary,
        "trade_label_abs_return_threshold": threshold,
    }



def build_triple_barrier_targets(mid: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    mid = np.asarray(mid, dtype=np.float64)
    n = len(mid)
    y_ret = np.full(n, np.nan, dtype=np.float32)
    y_dir = np.full(n, 0.5, dtype=np.float32)
    y_trade = np.full(n, np.nan, dtype=np.float32)
    y_dir_mask = np.zeros(n, dtype=np.float32)
    y_exit_type = np.full(n, -1, dtype=np.int64)
    y_tte = np.full(n, np.nan, dtype=np.float32)
    y_timeout = np.zeros(n, dtype=np.float32)

    log_mid = np.log(mid + EPS)
    up_bps, down_bps = compute_triple_barrier_bps(mid, cfg)
    trade_edge_threshold = compute_trade_edge_threshold_log_return(cfg)
    vertical_barrier_bars = resolve_vertical_barrier_bars(cfg)
    meta_labeling_enabled = parse_bool(cfg.get("meta_labeling_enabled"), True)
    trade_label_requires_first_touch = parse_bool(cfg.get("trade_label_requires_first_touch"), True)
    mask_timeout_for_direction = parse_bool(cfg.get("mask_timeout_for_direction"), True)

    n_upper = 0
    n_lower = 0
    n_vertical = 0

    for t in range(0, max(0, n - vertical_barrier_bars)):
        if not np.isfinite(log_mid[t]):
            continue

        upper_lr = max(float(up_bps[t]) * 1e-4, 1e-8)
        lower_lr = max(float(down_bps[t]) * 1e-4, 1e-8)
        path = log_mid[t + 1: t + vertical_barrier_bars + 1] - log_mid[t]
        if len(path) == 0 or not np.isfinite(path).all():
            continue

        up_hits = np.where(path >= upper_lr - 1e-12)[0]
        down_hits = np.where(path <= -lower_lr + 1e-12)[0]
        first_up = int(up_hits[0]) + 1 if up_hits.size else None
        first_down = int(down_hits[0]) + 1 if down_hits.size else None

        if first_up is not None and (first_down is None or first_up <= first_down):
            exit_type = "upper"
            tte = first_up
            realized_return = float(upper_lr)
            side_label = 1.0
            n_upper += 1
        elif first_down is not None and (first_up is None or first_down < first_up):
            exit_type = "lower"
            tte = first_down
            realized_return = float(-lower_lr)
            side_label = 0.0
            n_lower += 1
        else:
            exit_type = "vertical"
            tte = vertical_barrier_bars
            realized_return = float(path[-1])
            side_label = 1.0 if realized_return > 0.0 else 0.0
            y_timeout[t] = 1.0
            n_vertical += 1

        if meta_labeling_enabled:
            if exit_type != "vertical":
                y_trade[t] = 1.0 if abs(realized_return) > trade_edge_threshold else 0.0
            elif trade_label_requires_first_touch:
                y_trade[t] = 0.0
            else:
                y_trade[t] = 1.0 if abs(realized_return) > trade_edge_threshold else 0.0
        else:
            y_trade[t] = 1.0

        if exit_type == "vertical" and mask_timeout_for_direction:
            y_dir_mask[t] = 0.0
        else:
            if meta_labeling_enabled:
                y_dir_mask[t] = 1.0 if y_trade[t] > 0.5 else 0.0
            else:
                y_dir_mask[t] = 1.0

        y_ret[t] = float(realized_return)
        y_dir[t] = float(side_label)
        y_exit_type[t] = EXIT_TYPE_TO_IDX[exit_type]
        y_tte[t] = float(tte)

    valid_mask = np.isfinite(y_ret)
    summary = {
        "label_mode": "triple_barrier",
        "n_valid": int(valid_mask.sum()),
        "n_trade": int(np.nansum(y_trade > 0.5)),
        "n_upper": int(n_upper),
        "n_lower": int(n_lower),
        "n_vertical": int(n_vertical),
        "trade_rate": float(np.nanmean(y_trade[valid_mask])) if valid_mask.any() else float("nan"),
        "avg_tte_bars": float(np.nanmean(y_tte[valid_mask])) if valid_mask.any() else float("nan"),
        "avg_abs_realized_return": float(np.nanmean(np.abs(y_ret[valid_mask]))) if valid_mask.any() else float("nan"),
        "vertical_barrier_bars": int(vertical_barrier_bars),
        "meta_labeling_enabled": bool(meta_labeling_enabled),
        "trade_label_requires_first_touch": bool(trade_label_requires_first_touch),
    }
    return {
        "y_ret": y_ret,
        "y_dir": y_dir,
        "y_trade": y_trade,
        "y_dir_mask": y_dir_mask,
        "y_exit_type": y_exit_type,
        "y_tte": y_tte,
        "y_timeout": y_timeout,
        "summary": summary,
        "trade_label_abs_return_threshold": trade_edge_threshold,
    }



def build_supervision_targets(mid: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if str(cfg.get("label_mode", "fixed_horizon")) == "triple_barrier":
        return build_triple_barrier_targets(mid, cfg)
    return build_fixed_horizon_targets(mid, cfg)

# %% Scaling helpers

def fit_robust_scaler_train_only_3d(
    raw_array: np.ndarray,
    sample_t: np.ndarray,
    train_sample_indices: np.ndarray,
    max_abs_value: float,
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if raw_array.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape={raw_array.shape}")

    last_train_t = int(sample_t[int(train_sample_indices[-1])])
    train_slice = raw_array[: last_train_t + 1]
    flat_train = train_slice.reshape(-1, raw_array.shape[-1])

    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(float(q_low), float(q_high)),
    )
    scaler.fit(flat_train)

    flat_all = raw_array.reshape(-1, raw_array.shape[-1])
    scaled_all = scaler.transform(flat_all).reshape(raw_array.shape).astype(np.float32)
    scaled_all = np.clip(scaled_all, -float(max_abs_value), float(max_abs_value))
    scaled_all = np.nan_to_num(scaled_all, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    params = {
        "center_": scaler.center_.astype(np.float32),
        "scale_": scaler.scale_.astype(np.float32),
        "max_abs_value": float(max_abs_value),
        "last_train_raw_t": last_train_t,
    }
    if last_train_t > int(sample_t[int(train_sample_indices[-1])]):
        raise AssertionError("Scaler fit window exceeded train boundary")
    return scaled_all, params


def apply_robust_scaler_params_3d(raw_array: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    center = np.asarray(params["center_"], dtype=np.float32)
    scale = np.asarray(params["scale_"], dtype=np.float32)
    max_abs_value = float(params["max_abs_value"])

    flat = raw_array.reshape(-1, raw_array.shape[-1]).astype(np.float32)
    flat = (flat - center) / (scale + 1e-12)
    flat = np.clip(flat, -max_abs_value, max_abs_value)
    flat = np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0)
    return flat.reshape(raw_array.shape).astype(np.float32)


def fit_relation_scalers_train_only(
    raw_rel_array: np.ndarray,
    relation_names: List[str],
    sample_t: np.ndarray,
    train_sample_indices: np.ndarray,
    max_abs_value: float,
    q_low: float,
    q_high: float,
) -> Tuple[np.ndarray, Dict[str, Dict[str, Any]]]:
    if raw_rel_array.ndim != 4:
        raise ValueError(f"Expected 4D relation array, got shape={raw_rel_array.shape}")

    scaled = np.zeros_like(raw_rel_array, dtype=np.float32)
    params: Dict[str, Dict[str, Any]] = {}

    for r, rel in enumerate(relation_names):
        rel_scaled, rel_params = fit_robust_scaler_train_only_3d(
            raw_array=raw_rel_array[:, r, :, :],
            sample_t=sample_t,
            train_sample_indices=train_sample_indices,
            max_abs_value=max_abs_value,
            q_low=q_low,
            q_high=q_high,
        )
        scaled[:, r, :, :] = rel_scaled
        params[rel] = rel_params

    return scaled, params


def apply_relation_scalers(
    raw_rel_array: np.ndarray,
    relation_names: List[str],
    relation_scaler_params: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    if raw_rel_array.ndim != 4:
        raise ValueError(f"Expected 4D relation array, got shape={raw_rel_array.shape}")

    scaled = np.zeros_like(raw_rel_array, dtype=np.float32)
    for r, rel in enumerate(relation_names):
        scaled[:, r, :, :] = apply_robust_scaler_params_3d(
            raw_rel_array[:, r, :, :],
            relation_scaler_params[rel],
        )
    return scaled.astype(np.float32)



# %% Model definition

class NodeStepProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_nodes: int, dropout: float):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.asset_emb = nn.Parameter(torch.randn(n_nodes, hidden_dim) * 0.02)
        self.post = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = h + self.asset_emb.view(1, 1, self.asset_emb.size(0), -1)
        return self.post(h)


class EdgeStepProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationAxisFusion(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        fusion_hidden_dim: int,
        mode: str,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_relations = int(num_relations)
        self.mode = str(mode)

        if self.mode == "attention":
            self.rel_emb = nn.Parameter(torch.randn(num_relations, hidden_dim) * 0.02)
            self.score_mlp = nn.Sequential(
                nn.Linear(hidden_dim, fusion_hidden_dim),
                nn.GELU(),
                nn.Linear(fusion_hidden_dim, 1),
            )
        elif self.mode == "mean":
            self.rel_emb = None
            self.score_mlp = None
        else:
            raise ValueError(f"Unsupported RelationAxisFusion mode: {self.mode}")

    def forward(self, relation_tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if relation_tensor.ndim < 4:
            raise ValueError(f"Expected relation tensor with shape [B, R, Items, D], got {relation_tensor.shape}")

        if self.mode == "mean":
            fused = relation_tensor.mean(dim=1)
            weights = torch.full(
                relation_tensor.shape[:-1],
                1.0 / max(float(self.num_relations), 1.0),
                dtype=relation_tensor.dtype,
                device=relation_tensor.device,
            )
            return fused, weights

        view_shape = [1] * relation_tensor.ndim
        view_shape[1] = self.num_relations
        view_shape[-1] = self.hidden_dim
        rel_bias = self.rel_emb.view(*view_shape)
        scores = self.score_mlp(relation_tensor + rel_bias).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        fused = (weights.unsqueeze(-1) * relation_tensor).sum(dim=1)
        return fused, weights


class HybridEdgeFeatureFusion(nn.Module):
    """Fuse handcrafted relation tensors with learnable pairwise node interactions.

    The learnable pairwise path is shared across relations but relation-conditioned through an
    embedding that is concatenated to the pairwise src/dst feature block. This keeps the original
    handcrafted relation semantics intact while adding adaptive pairwise context.
    """

    def __init__(
        self,
        node_dim: int,
        edge_hidden_dim: int,
        num_relations: int,
        pair_hidden_dim: int,
        dropout: float,
        edge_feature_mode: str,
    ):
        super().__init__()
        self.edge_feature_mode = str(edge_feature_mode)
        self.num_relations = int(num_relations)
        self.pair_hidden_dim = int(pair_hidden_dim)

        if self.edge_feature_mode == "hybrid":
            self.rel_emb = nn.Parameter(torch.randn(num_relations, node_dim) * 0.02)
            self.pair_mlp = nn.Sequential(
                nn.Linear(5 * node_dim, pair_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(pair_hidden_dim, edge_hidden_dim),
            )
            self.fuse_mlp = nn.Sequential(
                nn.Linear(2 * edge_hidden_dim, edge_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
            )
            self.gate_net = nn.Sequential(
                nn.Linear(2 * edge_hidden_dim, edge_hidden_dim),
                nn.GELU(),
                nn.Linear(edge_hidden_dim, edge_hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self.rel_emb = None
            self.pair_mlp = None
            self.fuse_mlp = None
            self.gate_net = None

    def forward(self, node_seq: torch.Tensor, edge_seq: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.edge_feature_mode != "hybrid":
            return edge_seq

        src_idx = edge_index[:, 0].to(node_seq.device)
        dst_idx = edge_index[:, 1].to(node_seq.device)
        h_src = node_seq[:, :, src_idx, :]
        h_dst = node_seq[:, :, dst_idx, :]
        pair_base = torch.cat([h_src, h_dst, h_src - h_dst, h_src * h_dst], dim=-1)

        fused_relations: List[torch.Tensor] = []
        for rel_idx in range(self.num_relations):
            rel_emb = self.rel_emb[rel_idx].view(1, 1, 1, -1).expand(
                pair_base.size(0),
                pair_base.size(1),
                pair_base.size(2),
                -1,
            )
            pair_feat = self.pair_mlp(torch.cat([pair_base, rel_emb], dim=-1))
            handcrafted_feat = edge_seq[:, :, rel_idx, :, :]
            gate_input = torch.cat([handcrafted_feat, pair_feat], dim=-1)
            gate = self.gate_net(gate_input)
            fused = gate * handcrafted_feat + (1.0 - gate) * pair_feat
            fused = fused + self.fuse_mlp(torch.cat([handcrafted_feat, pair_feat], dim=-1))
            fused_relations.append(fused)
        return torch.stack(fused_relations, dim=2)



def aggregate_messages_to_dst(msg: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    out = msg.new_zeros(msg.size(0), n_nodes, msg.size(-1))
    for e in range(msg.size(1)):
        out[:, int(dst_idx[e].item()), :] += msg[:, e, :]
    return out



def edge_softmax_by_dst(logits: torch.Tensor, dst_idx: torch.Tensor, n_nodes: int) -> torch.Tensor:
    out = torch.zeros_like(logits)
    for node in range(n_nodes):
        mask = dst_idx == node
        out[:, mask] = torch.softmax(logits[:, mask], dim=1)
    return out



class AdaptiveGraphConnectivity(nn.Module):
    def __init__(
        self,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        adjacency_rank: int,
        adjacency_scale: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.adjacency_scale = float(adjacency_scale)

        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        hidden_dim = max(8, int(edge_dim))
        self.edge_score_net = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        self.src_emb = nn.Parameter(torch.randn(n_nodes, adjacency_rank) * 0.02)
        self.dst_emb = nn.Parameter(torch.randn(n_nodes, adjacency_rank) * 0.02)
        self.edge_bias = nn.Parameter(torch.zeros(edge_index.size(0)))

    def forward(self, latest_edge_state: torch.Tensor) -> torch.Tensor:
        batch_size = int(latest_edge_state.size(0))
        device = latest_edge_state.device
        dtype = latest_edge_state.dtype

        edge_logits = self.edge_score_net(latest_edge_state).squeeze(-1)
        pair_logits = self.adjacency_scale * (
            (self.src_emb[self.src_idx] * self.dst_emb[self.dst_idx]).sum(dim=-1) + self.edge_bias
        )
        pair_logits = pair_logits.view(1, -1).expand(batch_size, -1).to(device=device, dtype=dtype)
        logits = edge_logits + pair_logits
        return edge_softmax_by_dst(logits, self.dst_idx, self.n_nodes)


class BaseTemporalConvLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        self.src_proj = nn.Linear(node_dim, node_dim)
        self.edge_shift = nn.Linear(edge_dim, node_dim)
        self.self_proj = nn.Linear(node_dim, node_dim)
        self.agg_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor, adjacency_weight: torch.Tensor) -> torch.Tensor:
        h_src = node_state[:, self.src_idx, :]
        msg = adjacency_weight.unsqueeze(-1) * (self.src_proj(h_src) + self.edge_shift(edge_state))
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)
        update = self.self_proj(node_state) + self.agg_proj(agg)
        return self.norm(node_state + self.dropout(F.gelu(update)))


class BaseTemporalMPNNLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        self.src_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.self_proj = nn.Linear(node_dim, node_dim)
        self.agg_proj = nn.Linear(node_dim, node_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(node_dim)

    def forward(self, node_state: torch.Tensor, edge_state: torch.Tensor, adjacency_weight: torch.Tensor) -> torch.Tensor:
        h_src = node_state[:, self.src_idx, :]
        h_dst = node_state[:, self.dst_idx, :]
        gate = torch.sigmoid(self.gate_net(torch.cat([h_src, h_dst, edge_state], dim=-1)))
        msg = adjacency_weight.unsqueeze(-1) * (gate * self.src_proj(h_src) + self.edge_proj(edge_state))
        agg = aggregate_messages_to_dst(msg, self.dst_idx, self.n_nodes)
        update = self.self_proj(node_state) + self.agg_proj(agg)
        return self.norm(node_state + self.dropout(F.gelu(update)))


class MemoryOperatorBlock(nn.Module):
    def __init__(
        self,
        operator_name: str,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        edge_index: torch.Tensor,
        num_layers: int,
        dropout: float,
        adjacency_rank: int,
        adjacency_scale: float,
        default_operator: str,
    ):
        super().__init__()
        canonical_name = canonicalize_memorygraph_operator_name(
            operator_name=operator_name,
            default_operator=default_operator,
        )
        self.operator_name = canonical_name
        self.connectivity = AdaptiveGraphConnectivity(
            edge_dim=edge_dim,
            n_nodes=n_nodes,
            edge_index=edge_index,
            adjacency_rank=adjacency_rank,
            adjacency_scale=adjacency_scale,
        )

        layer_cls = BaseTemporalConvLayer if self.operator_name == "conv" else BaseTemporalMPNNLayer
        self.layers = nn.ModuleList(
            [
                layer_cls(
                    node_dim=node_dim,
                    edge_dim=edge_dim,
                    n_nodes=n_nodes,
                    edge_index=edge_index,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        collect_regularization: bool = False,
        return_aux: bool = False,
    ):
        learned_adjacency = self.connectivity(edge_state)
        out = node_state

        for layer in self.layers:
            out = layer(out, edge_state, learned_adjacency)

        if not collect_regularization and not return_aux:
            return out

        reg = {
            "adj_l1": learned_adjacency.abs().mean(),
        }
        aux = {
            "learned_adjacency": learned_adjacency,
        }

        if collect_regularization and return_aux:
            return out, reg, aux
        if collect_regularization:
            return out, reg
        return out, aux


class EdgeMemoryUpdater(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_memory_dim: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.register_buffer("src_idx", edge_index[:, 0].clone())
        self.register_buffer("dst_idx", edge_index[:, 1].clone())
        self.edge_memory_dim = int(edge_memory_dim)
        self.memory_cell = nn.GRUCell(edge_dim + 4 * node_dim, edge_memory_dim)
        self.memory_to_edge = nn.Sequential(
            nn.LayerNorm(edge_memory_dim),
            nn.Linear(edge_memory_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * edge_dim, edge_dim),
            nn.GELU(),
            nn.Linear(edge_dim, edge_dim),
            nn.Sigmoid(),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * edge_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(edge_dim, edge_dim),
        )
        self.norm = nn.LayerNorm(edge_dim)

    def forward(
        self,
        node_state: torch.Tensor,
        edge_state: torch.Tensor,
        edge_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_src = node_state[:, self.src_idx, :]
        h_dst = node_state[:, self.dst_idx, :]
        pair_feat = torch.cat([h_src, h_dst, h_src - h_dst, h_src * h_dst], dim=-1)

        bsz, n_edges, _ = edge_state.shape
        flat_input = torch.cat([edge_state, pair_feat], dim=-1).reshape(bsz * n_edges, -1)
        flat_memory = edge_memory.reshape(bsz * n_edges, -1)
        new_edge_memory = self.memory_cell(flat_input, flat_memory).view(bsz, n_edges, self.edge_memory_dim)

        memory_edge = self.memory_to_edge(new_edge_memory)
        fuse_input = torch.cat([edge_state, memory_edge], dim=-1)
        gate = self.gate_net(fuse_input)
        enriched_edge = self.norm(edge_state + gate * memory_edge + self.fuse_mlp(fuse_input))
        return new_edge_memory, enriched_edge


class NodeMemoryUpdater(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_memory_dim: int,
        node_memory_dim: int,
        num_relations: int,
        fusion_hidden_dim: int,
        fusion_mode: str,
        n_nodes: int,
        edge_index: torch.Tensor,
        dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.node_memory_dim = int(node_memory_dim)
        self.num_relations = int(num_relations)
        self.register_buffer("dst_idx", edge_index[:, 1].clone())

        indeg = torch.zeros(n_nodes, dtype=torch.float32)
        for dst in edge_index[:, 1].tolist():
            indeg[int(dst)] += 1.0
        self.register_buffer("indeg", indeg.clamp_min(1.0))

        self.edge_to_node = nn.Sequential(
            nn.Linear(edge_memory_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.relation_node_fusion = RelationAxisFusion(
            hidden_dim=node_dim,
            num_relations=num_relations,
            fusion_hidden_dim=fusion_hidden_dim,
            mode=fusion_mode,
        )
        self.relation_edge_fusion = RelationAxisFusion(
            hidden_dim=node_dim,
            num_relations=num_relations,
            fusion_hidden_dim=fusion_hidden_dim,
            mode=fusion_mode,
        )
        self.memory_cell = nn.GRUCell(2 * node_dim, node_memory_dim)
        self.memory_to_node = nn.Sequential(
            nn.LayerNorm(node_memory_dim),
            nn.Linear(node_memory_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.gate_net = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.GELU(),
            nn.Linear(node_dim, node_dim),
            nn.Sigmoid(),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        relation_node_state: torch.Tensor,
        relation_edge_memory: torch.Tensor,
        node_memory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, n_rel, n_edges, _ = relation_edge_memory.shape
        flat_edge_memory = relation_edge_memory.reshape(bsz * n_rel, n_edges, -1)
        flat_edge_context = aggregate_messages_to_dst(
            self.edge_to_node(flat_edge_memory),
            self.dst_idx,
            self.n_nodes,
        )
        relation_edge_context = flat_edge_context.view(bsz, n_rel, self.n_nodes, -1)
        relation_edge_context = relation_edge_context / self.indeg.view(1, 1, -1, 1)

        fused_node_state, relation_node_weights = self.relation_node_fusion(relation_node_state)
        fused_edge_context, relation_edge_weights = self.relation_edge_fusion(relation_edge_context)

        _, n_nodes, _ = fused_node_state.shape
        flat_input = torch.cat([fused_node_state, fused_edge_context], dim=-1).reshape(bsz * n_nodes, -1)
        flat_memory = node_memory.reshape(bsz * n_nodes, -1)
        new_node_memory = self.memory_cell(flat_input, flat_memory).view(bsz, n_nodes, self.node_memory_dim)

        memory_node = self.memory_to_node(new_node_memory)
        fuse_input = torch.cat([fused_node_state, memory_node], dim=-1)
        gate = self.gate_net(fuse_input)
        enriched_node = self.norm(fused_node_state + gate * memory_node + self.fuse_mlp(fuse_input))
        aux = {
            "relation_node_weights": relation_node_weights,
            "relation_edge_weights": relation_edge_weights,
            "fused_edge_context": fused_edge_context,
        }
        return new_node_memory, enriched_node, aux


class MemoryAugmentedGraphBlock(nn.Module):
    def __init__(
        self,
        operator_name: str,
        node_dim: int,
        edge_dim: int,
        num_relations: int,
        fusion_hidden_dim: int,
        fusion_mode: str,
        n_nodes: int,
        edge_index: torch.Tensor,
        num_layers: int,
        dropout: float,
        adjacency_rank: int,
        adjacency_scale: float,
        default_operator: str,
        node_memory_dim: int,
        edge_memory_dim: int,
        memory_dropout: float,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_edges = int(edge_index.size(0))
        self.num_relations = int(num_relations)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.node_memory_dim = int(node_memory_dim)
        self.edge_memory_dim = int(edge_memory_dim)
        self.node_memory_seed = nn.Parameter(torch.zeros(1, n_nodes, node_memory_dim))
        self.edge_memory_seed = nn.Parameter(torch.zeros(1, num_relations, self.n_edges, edge_memory_dim))
        self.relation_node_bias = nn.Parameter(torch.randn(1, num_relations, 1, node_dim) * 0.02)
        self.node_memory_to_input = nn.Sequential(
            nn.LayerNorm(node_memory_dim),
            nn.Linear(node_memory_dim, node_dim),
            nn.GELU(),
            nn.Dropout(memory_dropout),
            nn.Linear(node_dim, node_dim),
        )
        self.edge_memory_to_input = nn.Sequential(
            nn.LayerNorm(edge_memory_dim),
            nn.Linear(edge_memory_dim, edge_dim),
            nn.GELU(),
            nn.Dropout(memory_dropout),
            nn.Linear(edge_dim, edge_dim),
        )
        self.input_node_norm = nn.LayerNorm(node_dim)
        self.input_edge_norm = nn.LayerNorm(edge_dim)
        self.edge_memory_updater = EdgeMemoryUpdater(
            node_dim=node_dim,
            edge_dim=edge_dim,
            edge_memory_dim=edge_memory_dim,
            edge_index=edge_index,
            dropout=memory_dropout,
        )
        self.graph_block = MemoryOperatorBlock(
            operator_name=operator_name,
            node_dim=node_dim,
            edge_dim=edge_dim,
            n_nodes=n_nodes,
            edge_index=edge_index,
            num_layers=num_layers,
            dropout=dropout,
            adjacency_rank=adjacency_rank,
            adjacency_scale=adjacency_scale,
            default_operator=default_operator,
        )
        self.node_memory_updater = NodeMemoryUpdater(
            node_dim=node_dim,
            edge_memory_dim=edge_memory_dim,
            node_memory_dim=node_memory_dim,
            num_relations=num_relations,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_mode=fusion_mode,
            n_nodes=n_nodes,
            edge_index=edge_index,
            dropout=memory_dropout,
        )

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        return {
            "node_memory": self.node_memory_seed.expand(batch_size, -1, -1).to(device=device, dtype=dtype),
            "edge_memory": self.edge_memory_seed.expand(batch_size, -1, -1, -1).to(device=device, dtype=dtype),
        }

    def _prepare_state(
        self,
        prev_state: Optional[Dict[str, torch.Tensor]],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        if prev_state is None:
            return self.init_state(batch_size=batch_size, device=device, dtype=dtype)

        node_memory = prev_state["node_memory"].to(device=device, dtype=dtype)
        edge_memory = prev_state["edge_memory"].to(device=device, dtype=dtype)
        if node_memory.size(0) != batch_size or edge_memory.size(0) != batch_size:
            raise ValueError("prev_state batch size does not match current batch size")
        return {
            "node_memory": node_memory,
            "edge_memory": edge_memory,
        }

    def _reshape_graph_aux(self, graph_aux: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
        reshaped: Dict[str, torch.Tensor] = {}
        for key, value in graph_aux.items():
            if isinstance(value, torch.Tensor) and value.ndim == 2 and value.size(0) == batch_size * self.num_relations:
                reshaped[key] = value.view(batch_size, self.num_relations, self.n_edges)
            else:
                reshaped[key] = value
        return reshaped

    def forward(
        self,
        node_seq: torch.Tensor,
        edge_seq: torch.Tensor,
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
        collect_regularization: bool = False,
        return_aux: bool = False,
        return_state: bool = False,
    ):
        bsz, seq_len, _, _ = node_seq.shape
        state = self._prepare_state(
            prev_state=prev_state,
            batch_size=bsz,
            device=node_seq.device,
            dtype=node_seq.dtype,
        )
        node_memory = state["node_memory"]
        edge_memory = state["edge_memory"]

        output_steps: List[torch.Tensor] = []
        adj_l1_terms: List[torch.Tensor] = []
        last_graph_aux: Dict[str, torch.Tensor] = {}
        last_node_aux: Dict[str, torch.Tensor] = {}
        last_relation_node_state: Optional[torch.Tensor] = None
        last_relation_edge_state: Optional[torch.Tensor] = None

        for t in range(seq_len):
            base_node_input = node_seq[:, t, :, :]
            relation_node_input = base_node_input.unsqueeze(1).expand(-1, self.num_relations, -1, -1)
            relation_node_input = relation_node_input + self.relation_node_bias + self.node_memory_to_input(node_memory).unsqueeze(1)
            relation_node_input = self.input_node_norm(relation_node_input)

            relation_edge_input = edge_seq[:, t, :, :, :] + self.edge_memory_to_input(edge_memory)
            relation_edge_input = self.input_edge_norm(relation_edge_input)

            flat_relation_node_input = relation_node_input.reshape(bsz * self.num_relations, self.n_nodes, self.node_dim)
            flat_relation_edge_input = relation_edge_input.reshape(bsz * self.num_relations, self.n_edges, self.edge_dim)
            flat_edge_memory = edge_memory.reshape(bsz * self.num_relations, self.n_edges, self.edge_memory_dim)
            flat_edge_memory, flat_relation_edge_state = self.edge_memory_updater(
                flat_relation_node_input,
                flat_relation_edge_input,
                flat_edge_memory,
            )
            edge_memory = flat_edge_memory.view(bsz, self.num_relations, self.n_edges, self.edge_memory_dim)
            relation_edge_state = flat_relation_edge_state.view(bsz, self.num_relations, self.n_edges, self.edge_dim)

            graph_out = self.graph_block(
                flat_relation_node_input,
                flat_relation_edge_state,
                collect_regularization=collect_regularization,
                return_aux=return_aux,
            )

            if collect_regularization and return_aux:
                flat_relation_node_state, graph_reg, graph_aux = graph_out
            elif collect_regularization:
                flat_relation_node_state, graph_reg = graph_out
                graph_aux = {}
            elif return_aux:
                flat_relation_node_state, graph_aux = graph_out
                graph_reg = {}
            else:
                flat_relation_node_state = graph_out
                graph_reg = {}
                graph_aux = {}

            relation_node_state = flat_relation_node_state.view(bsz, self.num_relations, self.n_nodes, self.node_dim)

            if collect_regularization:
                adj_l1_terms.append(graph_reg["adj_l1"])

            node_memory, fused_node_state, node_aux = self.node_memory_updater(
                relation_node_state,
                edge_memory,
                node_memory,
            )
            output_steps.append(fused_node_state)

            if return_aux:
                last_graph_aux = self._reshape_graph_aux(graph_aux, batch_size=bsz)
                last_node_aux = node_aux
                last_relation_node_state = relation_node_state
                last_relation_edge_state = relation_edge_state

        out = torch.stack(output_steps, dim=1)
        next_state = {
            "node_memory": node_memory,
            "edge_memory": edge_memory,
        }

        if not collect_regularization and not return_aux and not return_state:
            return out

        reg = {
            "adj_l1": torch.stack(adj_l1_terms).mean() if adj_l1_terms else out.new_zeros(()),
        }
        aux = {
            "graph_operator": self.graph_block.operator_name,
        }
        if return_aux:
            aux.update(last_graph_aux)
            aux["final_node_memory"] = node_memory
            aux["final_edge_memory"] = edge_memory
            aux.update(last_node_aux)
            if last_relation_node_state is not None:
                aux["final_relation_node_state"] = last_relation_node_state
            if last_relation_edge_state is not None:
                aux["final_relation_edge_state"] = last_relation_edge_state

        pieces: List[Any] = [out]
        if collect_regularization:
            pieces.append(reg)
        if return_aux:
            pieces.append(aux)
        if return_state:
            pieces.append(next_state)
        if len(pieces) == 1:
            return pieces[0]
        return tuple(pieces)


class GraphReadout(nn.Module):
    def __init__(
        self,
        node_dim: int,
        out_dim: int,
        mode: str,
        global_pool: List[str],
        use_target_global_attention: bool,
        dropout: float,
    ):
        super().__init__()
        self.mode = str(mode)
        self.global_pool = [str(x) for x in global_pool]
        self.use_target_global_attention = bool(use_target_global_attention)
        feature_multiplier = 1
        if self.mode in {"target_plus_global", "target_plus_attn_global"}:
            feature_multiplier += len(self.global_pool)
        if self.mode == "target_plus_attn_global" and self.use_target_global_attention:
            feature_multiplier += 1
        self.out_proj = nn.Sequential(
            nn.LayerNorm(node_dim * feature_multiplier),
            nn.Linear(node_dim * feature_multiplier, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _pool_global(self, fused_node_seq: torch.Tensor) -> List[torch.Tensor]:
        pooled: List[torch.Tensor] = []
        if "mean" in self.global_pool:
            pooled.append(fused_node_seq.mean(dim=2))
        if "max" in self.global_pool:
            pooled.append(fused_node_seq.max(dim=2).values)
        return pooled

    def _target_attention_context(self, fused_node_seq: torch.Tensor, target_node: int) -> torch.Tensor:
        target_seq = fused_node_seq[:, :, target_node, :]
        score = torch.einsum("blnd,bld->bln", fused_node_seq, target_seq) / np.sqrt(float(fused_node_seq.size(-1)))
        attn = torch.softmax(score, dim=-1)
        context = torch.einsum("bln,blnd->bld", attn, fused_node_seq)
        return context

    def forward(self, fused_node_seq: torch.Tensor, target_node: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        pieces = [fused_node_seq[:, :, target_node, :]]
        aux: Dict[str, torch.Tensor] = {}

        if self.mode in {"target_plus_global", "target_plus_attn_global"}:
            pooled = self._pool_global(fused_node_seq)
            for tensor in pooled:
                pieces.append(tensor)
            if pooled:
                aux["global_context"] = torch.cat(pooled, dim=-1)
        if self.mode == "target_plus_attn_global" and self.use_target_global_attention:
            attn_ctx = self._target_attention_context(fused_node_seq, target_node)
            pieces.append(attn_ctx)
            aux["target_attention_context"] = attn_ctx

        readout_seq = torch.cat(pieces, dim=-1)
        projected = self.out_proj(readout_seq)
        aux["readout_seq"] = projected
        return projected, aux


def make_prediction_head(hidden_dim: int, dropout: float, out_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class MemoryGraphTemporalFusionModel(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        n_nodes: int,
        target_node: int,
        relation_names: List[str],
        cfg: Dict[str, Any],
    ):
        super().__init__()

        self.n_nodes = int(n_nodes)
        self.target_node = int(target_node)
        self.relation_names = list(relation_names)
        self.tte_cap_bars = int(effective_tte_cap_bars(cfg))

        node_hidden_dim = int(cfg["node_hidden_dim"])
        edge_hidden_dim = int(cfg["edge_hidden_dim"])
        target_hidden_dim = int(cfg["target_hidden_dim"])
        graph_layers = int(cfg["graph_layers"])
        dropout = float(cfg["dropout"])
        default_operator = normalize_memorygraph_operator(
            cfg.get("memorygraph_default_operator"),
            default="conv",
        )
        self.operator_name = canonicalize_memorygraph_operator_name(
            operator_name=cfg.get("graph_operator"),
            default_operator=default_operator,
        )

        self.node_encoder = NodeStepProjector(
            input_dim=node_dim,
            hidden_dim=node_hidden_dim,
            dropout=dropout,
            n_nodes=n_nodes,
        )
        self.edge_encoder = EdgeStepProjector(
            input_dim=edge_dim,
            hidden_dim=edge_hidden_dim,
            dropout=dropout,
        )
        self.hybrid_edge_fusion = HybridEdgeFeatureFusion(
            node_dim=node_hidden_dim,
            edge_hidden_dim=edge_hidden_dim,
            num_relations=len(self.relation_names),
            pair_hidden_dim=int(cfg["learned_pairwise_hidden_dim"]),
            dropout=dropout,
            edge_feature_mode=str(cfg["edge_feature_mode"]),
        )
        self.graph_block = MemoryAugmentedGraphBlock(
            operator_name=self.operator_name,
            node_dim=node_hidden_dim,
            edge_dim=edge_hidden_dim,
            num_relations=len(self.relation_names),
            fusion_hidden_dim=int(cfg["fusion_hidden_dim"]),
            fusion_mode=str(first_not_none(cfg.get("memorygraph_relation_fusion_mode"), "attention")),
            n_nodes=n_nodes,
            edge_index=EDGE_INDEX,
            num_layers=graph_layers,
            dropout=dropout,
            adjacency_rank=int(first_not_none(cfg.get("memorygraph_adjacency_rank"), 16)),
            adjacency_scale=float(first_not_none(cfg.get("memorygraph_adjacency_scale"), 1.0)),
            default_operator=default_operator,
            node_memory_dim=int(first_not_none(cfg.get("memorygraph_node_memory_dim"), node_hidden_dim)),
            edge_memory_dim=int(first_not_none(cfg.get("memorygraph_edge_memory_dim"), edge_hidden_dim)),
            memory_dropout=float(first_not_none(cfg.get("memorygraph_memory_dropout"), dropout)),
        )
        self.readout = GraphReadout(
            node_dim=node_hidden_dim,
            out_dim=target_hidden_dim,
            mode=str(cfg["graph_readout_mode"]),
            global_pool=list(cfg["graph_global_pool"]),
            use_target_global_attention=bool(cfg["use_target_global_attention"]),
            dropout=dropout,
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(target_hidden_dim),
            nn.Linear(target_hidden_dim, target_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.trade_head = make_prediction_head(target_hidden_dim, dropout, out_dim=1)
        self.dir_head = make_prediction_head(target_hidden_dim, dropout, out_dim=1)
        self.return_head = make_prediction_head(target_hidden_dim, dropout, out_dim=1)
        self.exit_type_head = make_prediction_head(target_hidden_dim, dropout, out_dim=len(EXIT_TYPE_NAMES))
        self.tte_head = make_prediction_head(target_hidden_dim, dropout, out_dim=1)

    def init_state(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, torch.Tensor]:
        sample_param = next(self.parameters())
        return self.graph_block.init_state(
            batch_size=batch_size,
            device=device or sample_param.device,
            dtype=dtype or sample_param.dtype,
        )

    def forward(
        self,
        x_node_seq: torch.Tensor,
        x_edge_seq: torch.Tensor,
        prev_state: Optional[Dict[str, torch.Tensor]] = None,
        return_state: bool = False,
        return_aux: bool = False,
    ) -> Dict[str, torch.Tensor]:
        x_node_seq = torch.nan_to_num(x_node_seq, nan=0.0, posinf=0.0, neginf=0.0)
        x_edge_seq = torch.nan_to_num(x_edge_seq, nan=0.0, posinf=0.0, neginf=0.0)

        node_seq = self.node_encoder(x_node_seq)
        edge_seq = self.edge_encoder(x_edge_seq)
        edge_seq = self.hybrid_edge_fusion(node_seq, edge_seq, EDGE_INDEX.to(x_node_seq.device))
        graph_block_out = self.graph_block(
            node_seq,
            edge_seq,
            prev_state=prev_state,
            collect_regularization=True,
            return_aux=return_aux,
            return_state=return_state,
        )
        graph_aux: Dict[str, torch.Tensor] = {}
        next_state: Optional[Dict[str, torch.Tensor]] = None
        if return_aux and return_state:
            fused_node_seq, graph_reg, graph_aux, next_state = graph_block_out
        elif return_aux:
            fused_node_seq, graph_reg, graph_aux = graph_block_out
        elif return_state:
            fused_node_seq, graph_reg, next_state = graph_block_out
        else:
            fused_node_seq, graph_reg = graph_block_out
        readout_seq, readout_aux = self.readout(fused_node_seq, self.target_node)
        shared_state = self.output_proj(readout_seq)

        trade_logit = self.trade_head(shared_state).squeeze(-1)
        dir_logit = self.dir_head(shared_state).squeeze(-1)
        return_pred = self.return_head(shared_state).squeeze(-1)
        exit_type_logit = self.exit_type_head(shared_state)
        tte_pred = torch.clamp(
            F.softplus(self.tte_head(shared_state).squeeze(-1)) + 1.0,
            min=1.0,
            max=float(self.tte_cap_bars),
        )

        outputs = {
            "trade_logit": torch.nan_to_num(trade_logit, nan=0.0, posinf=0.0, neginf=0.0),
            "dir_logit": torch.nan_to_num(dir_logit, nan=0.0, posinf=0.0, neginf=0.0),
            "return_pred": torch.nan_to_num(return_pred, nan=0.0, posinf=0.0, neginf=0.0),
            "fixed_pred": torch.nan_to_num(return_pred, nan=0.0, posinf=0.0, neginf=0.0),
            "exit_type_logit": torch.nan_to_num(exit_type_logit, nan=0.0, posinf=0.0, neginf=0.0),
            "tte_pred": torch.nan_to_num(tte_pred, nan=float(HORIZON_BARS), posinf=float(HORIZON_BARS), neginf=1.0),
            "adj_l1": graph_reg["adj_l1"],
        }

        if return_aux:
            outputs["fused_node_seq"] = fused_node_seq
            outputs["relation_edge_seq"] = edge_seq
            outputs.update(graph_aux)
            outputs.update(readout_aux)
        if return_state and next_state is not None:
            outputs["next_state"] = next_state

        return outputs


# %% Losses and metrics

def rmse_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def mae_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))



def ic_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) == 0:
        return float("nan")
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])



def safe_roc_auc(y_true_binary: np.ndarray, score: np.ndarray) -> float:
    y_true_binary = np.asarray(y_true_binary)
    score = np.asarray(score, dtype=np.float64)
    mask = np.isfinite(y_true_binary) & np.isfinite(score)

    y = y_true_binary[mask]
    s = score[mask]

    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")



def finite_or_default(value: Any, default: float) -> float:
    try:
        v = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return v



def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))



def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)



def multiclass_accuracy_np(y_true: np.ndarray, logits: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.isfinite(y_true)
    if mask.sum() == 0:
        return float("nan")
    pred = np.argmax(logits[mask], axis=1)
    return float(np.mean(pred == y_true[mask]))



def trade_activation_torch(trade_logit: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    if parse_bool(cfg.get("meta_labeling_enabled"), True):
        return torch.sigmoid(trade_logit)
    return torch.ones_like(trade_logit)


def trade_activation_numpy(trade_logit: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    if parse_bool(cfg.get("meta_labeling_enabled"), True):
        return sigmoid_np(trade_logit)
    return np.ones_like(np.asarray(trade_logit, dtype=np.float64), dtype=np.float64)


def compute_soft_position_torch(
    trade_logit: torch.Tensor,
    dir_logit: torch.Tensor,
    k: float,
    cfg: Dict[str, Any],
) -> torch.Tensor:
    return trade_activation_torch(trade_logit, cfg) * torch.tanh(float(k) * dir_logit)



def compute_soft_utility_numpy(
    trade_logit: np.ndarray,
    dir_logit: np.ndarray,
    returns: np.ndarray,
    k: float,
    cfg: Dict[str, Any],
) -> np.ndarray:
    p_trade = trade_activation_numpy(trade_logit, cfg)
    soft_dir = np.tanh(float(k) * np.asarray(dir_logit, dtype=np.float64))
    ret = np.asarray(returns, dtype=np.float64)
    return p_trade * soft_dir * ret


@dataclass
class LossState:
    pos_weight_trade: float
    pos_weight_dir: float



def compute_positive_class_weight(labels: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64)
    labels = labels[np.isfinite(labels)]
    if len(labels) == 0:
        return 1.0
    pos = float((labels > 0.5).sum())
    neg = float(len(labels) - pos)
    if pos <= 0.0:
        return 1.0
    return float(min(10.0, max(1.0, neg / pos)))



def build_loss_state(idx_train: np.ndarray) -> LossState:
    raw_t_train = SAMPLE_T[idx_train]
    trade_train = Y_TRADE[raw_t_train].astype(np.float64) if parse_bool(CFG.get("meta_labeling_enabled"), True) else np.ones(len(raw_t_train), dtype=np.float64)
    dir_train = Y_DIR[raw_t_train][Y_DIR_MASK[raw_t_train] > 0.5].astype(np.float64)

    return LossState(
        pos_weight_trade=compute_positive_class_weight(trade_train),
        pos_weight_dir=compute_positive_class_weight(dir_train) if len(dir_train) > 0 else 1.0,
    )



def masked_binary_cross_entropy_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor,
) -> torch.Tensor:
    valid = mask > 0.5
    if not valid.any():
        return logits.new_zeros(())
    return F.binary_cross_entropy_with_logits(logits[valid], targets[valid], pos_weight=pos_weight)



def compute_standard_multitask_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_state: LossState,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    trade_logit = outputs["trade_logit"].view(-1)
    dir_logit = outputs["dir_logit"].view(-1)
    return_pred = outputs["return_pred"].view(-1)
    exit_type_logit = outputs["exit_type_logit"].reshape(-1, outputs["exit_type_logit"].shape[-1])
    tte_pred = outputs["tte_pred"].view(-1)

    y_ret = batch["y_ret"].view(-1).float()
    y_trade = batch["y_trade"].view(-1).float()
    y_dir = batch["y_dir"].view(-1).float()
    y_dir_mask = batch["y_dir_mask"].view(-1).float()
    y_exit_type = batch["y_exit_type"].view(-1).long()
    y_tte = batch["y_tte"].view(-1).float()

    trade_pos_weight = torch.tensor(loss_state.pos_weight_trade, dtype=torch.float32, device=trade_logit.device)
    dir_pos_weight = torch.tensor(loss_state.pos_weight_dir, dtype=torch.float32, device=trade_logit.device)

    if parse_bool(cfg.get("meta_labeling_enabled"), True):
        trade_loss = F.binary_cross_entropy_with_logits(trade_logit, y_trade, pos_weight=trade_pos_weight)
    else:
        trade_loss = trade_logit.new_zeros(())
    dir_loss = masked_binary_cross_entropy_with_logits(dir_logit, y_dir, y_dir_mask, dir_pos_weight)
    ret_loss = F.smooth_l1_loss(return_pred, y_ret, beta=float(cfg["huber_beta"]))
    exit_type_loss = F.cross_entropy(exit_type_logit, y_exit_type)
    tte_loss = F.smooth_l1_loss(tte_pred, y_tte, beta=1.0)

    zero = trade_loss.new_zeros(())
    adj_reg = float(cfg["adj_l1_lambda"]) * outputs["adj_l1"]
    total_loss = (
        float(cfg["loss_w_trade"]) * trade_loss
        + float(cfg["loss_w_dir"]) * dir_loss
        + float(cfg["loss_w_ret"]) * ret_loss
        + float(cfg.get("loss_w_exit_type", 0.0)) * exit_type_loss
        + float(cfg.get("loss_w_tte", 0.0)) * tte_loss
        + adj_reg
    )

    return {
        "total_loss": total_loss,
        "trade_loss": trade_loss.detach(),
        "dir_loss": dir_loss.detach(),
        "ret_loss": ret_loss.detach(),
        "exit_type_loss": exit_type_loss.detach(),
        "tte_loss": tte_loss.detach(),
        "utility_loss": zero.detach(),
        "false_positive_penalty_loss": zero.detach(),
        "timeout_penalty_loss": zero.detach(),
        "adj_reg": adj_reg.detach(),
        "soft_utility_mean": zero.detach(),
    }



def compute_execution_aware_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_state: LossState,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    base = compute_standard_multitask_loss(outputs, batch, loss_state, cfg)

    trade_logit = outputs["trade_logit"].view(-1)
    dir_logit = outputs["dir_logit"].view(-1)
    y_ret = batch["y_ret"].view(-1).float()
    y_trade = batch["y_trade"].view(-1).float()
    y_exit_type = batch["y_exit_type"].view(-1).long()

    soft_position = compute_soft_position_torch(
        trade_logit=trade_logit,
        dir_logit=dir_logit,
        k=float(cfg["utility_tanh_k"]),
        cfg=cfg,
    )
    exec_cost = float(cfg.get("execution_cost_multiplier", 1.0)) * round_trip_cost_as_log_return(
        float(cfg["cost_bps_per_side"])
    )
    soft_utility = soft_position * y_ret - torch.abs(soft_position) * exec_cost
    utility_loss = -soft_utility.mean()

    soft_trade = trade_activation_torch(trade_logit, cfg)
    if parse_bool(cfg.get("meta_labeling_enabled"), True):
        false_positive_mask = (y_trade < 0.5).float()
        false_positive_penalty_loss = (soft_trade * false_positive_mask).mean()
    else:
        false_positive_penalty_loss = soft_trade.new_zeros(())

    timeout_mask = (y_exit_type == EXIT_TYPE_TO_IDX["vertical"]).float()
    if str(cfg.get("label_mode", "fixed_horizon")) == "triple_barrier":
        timeout_penalty_loss = (soft_trade * timeout_mask).mean()
    else:
        timeout_penalty_loss = soft_trade.new_zeros(())

    total_loss = (
        base["total_loss"]
        + float(cfg["loss_w_utility"]) * utility_loss
        + float(cfg.get("false_positive_penalty", 0.0)) * false_positive_penalty_loss
        + float(cfg.get("timeout_penalty", 0.0)) * timeout_penalty_loss
    )
    base["total_loss"] = total_loss
    base["utility_loss"] = utility_loss.detach()
    base["false_positive_penalty_loss"] = false_positive_penalty_loss.detach()
    base["timeout_penalty_loss"] = timeout_penalty_loss.detach()
    base["soft_utility_mean"] = soft_utility.mean().detach()
    return base



def compute_total_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    loss_state: LossState,
    cfg: Dict[str, Any],
) -> Dict[str, torch.Tensor]:
    if str(cfg.get("objective_mode", "execution_aware")) == "standard":
        return compute_standard_multitask_loss(outputs, batch, loss_state, cfg)
    return compute_execution_aware_loss(outputs, batch, loss_state, cfg)


# %% Evaluation and backtesting

def apply_threshold_pair(
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    exit_type_prob: Optional[np.ndarray],
    thr_trade: float,
    thr_dir: float,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    del exit_type_prob, cfg
    trade_prob = np.asarray(trade_prob, dtype=np.float64)
    dir_prob = np.asarray(dir_prob, dtype=np.float64)

    trade_gate = trade_prob >= float(thr_trade)
    long_mask = trade_gate & (dir_prob > float(thr_dir))
    short_mask = trade_gate & (dir_prob < (1.0 - float(thr_dir)))
    return long_mask.astype(bool), short_mask.astype(bool)


def threshold_trade_grid_values(cfg: Dict[str, Any]) -> List[float]:
    return [float(x) for x in cfg["thr_trade_grid"]]


def predicted_exit_type_from_prob(exit_type_prob: np.ndarray) -> np.ndarray:
    exit_type_prob = np.asarray(exit_type_prob, dtype=np.float64)
    if exit_type_prob.ndim != 2 or exit_type_prob.shape[1] != len(EXIT_TYPE_NAMES):
        return np.full(exit_type_prob.shape[0], EXIT_TYPE_TO_IDX["vertical"], dtype=np.int64)
    return np.argmax(exit_type_prob, axis=1).astype(np.int64)


def predicted_exit_bars_from_tte(tte_pred: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    tte_pred = np.asarray(tte_pred, dtype=np.float64)
    exit_bars = np.asarray(np.round(tte_pred), dtype=np.int64)
    exit_bars[~np.isfinite(tte_pred)] = effective_tte_cap_bars(cfg)
    return np.clip(exit_bars, 1, effective_tte_cap_bars(cfg))


def oracle_exit_bars_from_labels(y_tte: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    oracle_cfg = copy.deepcopy(cfg)
    oracle_cfg["backtest_exit_mode"] = "realized_event"
    return get_exit_bars_for_backtest(y_tte, oracle_cfg)


def backtest_uses_label_derived_exit(cfg: Dict[str, Any]) -> bool:
    return (
        str(cfg.get("backtest_exit_mode", "fixed_horizon")) == "realized_event"
        and str(cfg.get("label_mode")) == "triple_barrier"
    )


def primary_backtest_mode_label(cfg: Dict[str, Any]) -> str:
    if backtest_uses_label_derived_exit(cfg):
        return "realized_event_entry_model"
    if str(cfg.get("backtest_exit_mode", "fixed_horizon")) == "fixed_horizon":
        return "fixed_horizon_entry_model"
    return f"{cfg.get('backtest_exit_mode', 'fixed_horizon')}_entry_model"


def get_exit_bars_for_backtest(y_tte: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    y_tte = np.asarray(y_tte, dtype=np.float64)
    if backtest_uses_label_derived_exit(cfg):
        exit_bars = np.asarray(np.round(y_tte), dtype=np.int64)
        exit_bars[~np.isfinite(y_tte)] = effective_tte_cap_bars(cfg)
        return np.clip(exit_bars, 1, effective_tte_cap_bars(cfg))
    return np.full(len(y_tte), effective_tte_cap_bars(cfg), dtype=np.int64)


def realized_return_from_raw_path(entry_raw_t: int, exit_bars: int) -> Tuple[int, float]:
    entry_raw_t = int(entry_raw_t)
    exit_raw_t = int(min(entry_raw_t + int(max(1, exit_bars)), len(TARGET_MID) - 1))
    entry_log_mid = float(np.log(float(TARGET_MID[entry_raw_t]) + EPS))
    exit_log_mid = float(np.log(float(TARGET_MID[exit_raw_t]) + EPS))
    return exit_raw_t, float(exit_log_mid - entry_log_mid)



def sequential_event_backtest_from_masks(
    y_true: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
    raw_t_indices: np.ndarray,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    cfg: Dict[str, Any],
    timestamps: Optional[pd.Series] = None,
    build_trades: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_exit_type = np.asarray(y_exit_type, dtype=np.int64)
    y_tte = np.asarray(y_tte, dtype=np.float64)
    raw_t_indices = np.asarray(raw_t_indices, dtype=np.int64)
    long_mask = np.asarray(long_mask, dtype=bool)
    short_mask = np.asarray(short_mask, dtype=bool)
    exit_bars_arr = get_exit_bars_for_backtest(y_tte, cfg)

    n = len(raw_t_indices)
    round_trip_cost = round_trip_cost_as_log_return(float(cfg["cost_bps_per_side"]))
    backtest_mode = primary_backtest_mode_label(cfg)
    backtest_uses_oracle_exit = backtest_uses_label_derived_exit(cfg)

    rows: List[Dict[str, Any]] = []
    pnl_list: List[float] = []
    gross_list: List[float] = []
    side_list: List[int] = []
    win_list: List[int] = []
    correct_list: List[int] = []
    timeout_trade_count = 0
    vertical_exit_count = 0
    upper_exit_count = 0
    lower_exit_count = 0

    i = 0
    while i < n:
        go_long = bool(long_mask[i])
        go_short = bool(short_mask[i])

        if go_long and go_short:
            go_long = False
            go_short = False

        if go_long:
            side = 1
        elif go_short:
            side = -1
        else:
            i += 1
            continue

        entry_raw_t = int(raw_t_indices[i])
        realized_return = float(y_true[i])
        exit_bars = int(max(1, exit_bars_arr[i]))
        gross_pnl = float(side * realized_return)
        net_pnl = float(gross_pnl - round_trip_cost)
        exit_type_idx = (
            int(y_exit_type[i])
            if 0 <= int(y_exit_type[i]) < len(EXIT_TYPE_NAMES)
            else EXIT_TYPE_TO_IDX["vertical"]
        )
        exit_type_name = EXIT_TYPE_NAMES[exit_type_idx]

        if exit_type_name == "vertical":
            vertical_exit_count += 1
            timeout_trade_count += 1
        elif exit_type_name == "upper":
            upper_exit_count += 1
        elif exit_type_name == "lower":
            lower_exit_count += 1

        pnl_list.append(net_pnl)
        gross_list.append(gross_pnl)
        side_list.append(side)
        win_list.append(int(net_pnl > 0.0))
        correct_list.append(int(side * realized_return > 0.0))

        if build_trades:
            exit_raw_t = int(min(entry_raw_t + exit_bars, len(TIMESTAMPS) - 1))
            entry_ts = pd.Timestamp(timestamps.iloc[entry_raw_t]) if timestamps is not None else pd.NaT
            exit_ts = pd.Timestamp(timestamps.iloc[exit_raw_t]) if (timestamps is not None and exit_raw_t < len(timestamps)) else pd.NaT
            rows.append(
                {
                    "entry_local_idx": i,
                    "entry_raw_t": entry_raw_t,
                    "exit_raw_t": exit_raw_t,
                    "entry_timestamp": entry_ts,
                    "exit_timestamp": exit_ts,
                    "side": side,
                    "exit_type": exit_type_name,
                    "time_to_exit_bars": exit_bars,
                    "realized_return": realized_return,
                    "gross_pnl": gross_pnl,
                    "net_pnl": net_pnl,
                    "backtest_mode": backtest_mode,
                }
            )

        i += exit_bars

    n_trades = len(pnl_list)
    pnl_sum = float(np.sum(pnl_list)) if n_trades else 0.0
    gross_pnl_sum = float(np.sum(gross_list)) if n_trades else 0.0
    pnl_per_trade = float(pnl_sum / n_trades) if n_trades else float("nan")
    sign_accuracy = float(np.mean(correct_list)) if n_trades else float("nan")
    win_rate = float(np.mean(win_list)) if n_trades else float("nan")
    long_trades = int(sum(1 for side in side_list if side == 1))
    short_trades = int(sum(1 for side in side_list if side == -1))
    long_pnl_sum = float(np.sum([p for p, side in zip(pnl_list, side_list) if side == 1])) if n_trades else 0.0
    short_pnl_sum = float(np.sum([p for p, side in zip(pnl_list, side_list) if side == -1])) if n_trades else 0.0
    trade_rate = float(n_trades / n) if n > 0 else float("nan")

    if n_trades >= 2 and np.std(pnl_list, ddof=1) > 1e-12:
        sharpe_like = float(np.mean(pnl_list) / np.std(pnl_list, ddof=1) * np.sqrt(n_trades))
    else:
        sharpe_like = float("nan")

    metrics = {
        "gross_pnl_sum": gross_pnl_sum,
        "pnl_sum": pnl_sum,
        "pnl_per_trade": pnl_per_trade,
        "n_trades": int(n_trades),
        "trade_rate": trade_rate,
        "sign_accuracy": sign_accuracy,
        "win_rate": win_rate,
        "long_trades": long_trades,
        "short_trades": short_trades,
        "long_pnl_sum": long_pnl_sum,
        "short_pnl_sum": short_pnl_sum,
        "sharpe_like": sharpe_like,
        "timeout_trade_count": int(timeout_trade_count),
        "upper_exit_trade_count": int(upper_exit_count),
        "lower_exit_trade_count": int(lower_exit_count),
        "vertical_exit_trade_count": int(vertical_exit_count),
        "backtest_mode": backtest_mode,
        "backtest_uses_oracle_exit": backtest_uses_oracle_exit,
    }

    trades_df = pd.DataFrame(rows)
    return metrics, trades_df


def sequential_event_backtest_from_masks_oracle(
    y_true: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
    raw_t_indices: np.ndarray,
    long_mask: np.ndarray,
    short_mask: np.ndarray,
    cfg: Dict[str, Any],
    timestamps: Optional[pd.Series] = None,
    build_trades: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    oracle_cfg = copy.deepcopy(cfg)
    oracle_cfg["backtest_exit_mode"] = "realized_event"
    return sequential_event_backtest_from_masks(
        y_true=y_true,
        y_exit_type=y_exit_type,
        y_tte=y_tte,
        raw_t_indices=raw_t_indices,
        long_mask=long_mask,
        short_mask=short_mask,
        cfg=oracle_cfg,
        timestamps=timestamps,
        build_trades=build_trades,
    )



def threshold_selection_key(bt_metrics: Dict[str, float], feasible: bool, coverage: float, cfg: Dict[str, Any]) -> Tuple[float, ...]:
    metric_name = str(cfg.get("threshold_search_metric", "composite"))
    if metric_name == "pnl_sum":
        primary = finite_or_default(bt_metrics.get("pnl_sum"), -1e9)
        secondary = finite_or_default(bt_metrics.get("pnl_per_trade"), -1e9)
        tertiary = finite_or_default(bt_metrics.get("sharpe_like"), -1e9)
    elif metric_name == "pnl_per_trade":
        primary = finite_or_default(bt_metrics.get("pnl_per_trade"), -1e9)
        secondary = finite_or_default(bt_metrics.get("pnl_sum"), -1e9)
        tertiary = finite_or_default(bt_metrics.get("sharpe_like"), -1e9)
    elif metric_name == "sharpe_like":
        primary = finite_or_default(bt_metrics.get("sharpe_like"), -1e9)
        secondary = finite_or_default(bt_metrics.get("pnl_sum"), -1e9)
        tertiary = finite_or_default(bt_metrics.get("pnl_per_trade"), -1e9)
    else:
        primary = finite_or_default(bt_metrics.get("pnl_sum"), -1e9)
        secondary = finite_or_default(bt_metrics.get("pnl_per_trade"), -1e9)
        tertiary = finite_or_default(bt_metrics.get("sign_accuracy"), -1e9)

    return (
        1.0 if feasible else 0.0,
        primary,
        secondary,
        tertiary,
        finite_or_default(coverage, -1e9),
        float(bt_metrics.get("n_trades", 0.0)),
    )



def search_best_threshold_pair(
    y_ret: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
    trade_prob: np.ndarray,
    dir_prob: np.ndarray,
    raw_t_indices: np.ndarray,
    cfg: Dict[str, Any],
    timestamps: pd.Series,
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, float], pd.DataFrame]:
    rows: List[Dict[str, Any]] = []
    best_pair: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, float]] = None
    best_key: Optional[Tuple[float, ...]] = None

    for thr_trade in threshold_trade_grid_values(cfg):
        for thr_dir in cfg["thr_dir_grid"]:
            long_mask, short_mask = apply_threshold_pair(
                trade_prob=trade_prob,
                dir_prob=dir_prob,
                exit_type_prob=None,
                thr_trade=float(thr_trade),
                thr_dir=float(thr_dir),
                cfg=cfg,
            )
            active_mask = long_mask | short_mask
            coverage = float(active_mask.mean()) if len(active_mask) else float("nan")

            bt_metrics, _ = sequential_event_backtest_from_masks(
                y_true=y_ret,
                y_exit_type=y_exit_type,
                y_tte=y_tte,
                raw_t_indices=raw_t_indices,
                long_mask=long_mask,
                short_mask=short_mask,
                cfg=cfg,
                timestamps=None,
                build_trades=False,
            )

            feasible = (
                bt_metrics["n_trades"] >= int(cfg["min_validation_trades"])
                and finite_or_default(coverage, 0.0) >= float(cfg["min_validation_coverage"])
            )

            row = {
                "thr_trade": float(thr_trade),
                "thr_dir": float(thr_dir),
                "coverage": coverage,
                "feasible": bool(feasible),
                **bt_metrics,
            }
            rows.append(row)

            key = threshold_selection_key(bt_metrics, feasible, coverage, cfg)
            if best_key is None or key > best_key:
                best_key = key
                best_pair = {
                    "thr_trade": float(thr_trade),
                    "thr_dir": float(thr_dir),
                    "coverage": coverage,
                    "feasible": bool(feasible),
                    "selection_stage": "threshold_calibration_val",
                }
                best_metrics = copy.deepcopy(bt_metrics)

    if best_pair is None or best_metrics is None:
        raise RuntimeError("Threshold-pair search failed to identify a valid validation pair.")

    best_long_mask, best_short_mask = apply_threshold_pair(
        trade_prob=trade_prob,
        dir_prob=dir_prob,
        exit_type_prob=None,
        thr_trade=float(best_pair["thr_trade"]),
        thr_dir=float(best_pair["thr_dir"]),
        cfg=cfg,
    )
    best_metrics, best_trades_df = sequential_event_backtest_from_masks(
        y_true=y_ret,
        y_exit_type=y_exit_type,
        y_tte=y_tte,
        raw_t_indices=raw_t_indices,
        long_mask=best_long_mask,
        short_mask=best_short_mask,
        cfg=cfg,
        timestamps=timestamps,
        build_trades=True,
    )
    best_metrics["coverage"] = float((best_long_mask | best_short_mask).mean()) if len(best_long_mask) else float("nan")

    grid_df = pd.DataFrame(rows).sort_values(
        by=["feasible", "pnl_sum", "pnl_per_trade", "sharpe_like", "n_trades"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)

    return best_pair, grid_df, best_metrics, best_trades_df



def evaluate_prediction_pack(
    pred_pack: Dict[str, Any],
    cfg: Dict[str, Any],
    selected_threshold_pair: Optional[Dict[str, Any]] = None,
    search_threshold_pair_on_pack: bool = False,
    perform_backtest: bool = True,
) -> Tuple[Dict[str, Any], Optional[pd.DataFrame], Dict[str, Any]]:
    y_ret = np.asarray(pred_pack["y_ret"], dtype=np.float64)
    y_trade = np.asarray(pred_pack["y_trade"], dtype=np.float64)
    y_dir = np.asarray(pred_pack["y_dir"], dtype=np.float64)
    y_dir_mask = np.asarray(pred_pack["y_dir_mask"], dtype=np.float64)
    y_exit_type = np.asarray(pred_pack["y_exit_type"], dtype=np.int64)
    y_tte = np.asarray(pred_pack["y_tte"], dtype=np.float64)
    return_pred = np.asarray(pred_pack["return_pred"], dtype=np.float64)
    fixed_pred = np.asarray(pred_pack["fixed_pred"], dtype=np.float64)
    trade_logit = np.asarray(pred_pack["trade_logit"], dtype=np.float64)
    dir_logit = np.asarray(pred_pack["dir_logit"], dtype=np.float64)
    trade_prob = np.asarray(pred_pack["trade_prob"], dtype=np.float64)
    dir_prob = np.asarray(pred_pack["dir_prob"], dtype=np.float64)
    exit_type_logit = np.asarray(pred_pack["exit_type_logit"], dtype=np.float64)
    raw_t = np.asarray(pred_pack["raw_t"], dtype=np.int64)
    tte_pred = np.asarray(pred_pack["tte_pred"], dtype=np.float64)
    exit_type_prob = np.asarray(pred_pack["exit_type_prob"], dtype=np.float64)
    predicted_exit_bars = predicted_exit_bars_from_tte(tte_pred, cfg)
    predicted_exit_type = predicted_exit_type_from_prob(exit_type_prob)

    trade_auc = safe_roc_auc(y_trade, trade_prob)
    dir_valid = y_dir_mask > 0.5
    dir_auc = safe_roc_auc(y_dir[dir_valid], dir_prob[dir_valid]) if dir_valid.any() else float("nan")
    exit_type_acc = multiclass_accuracy_np(y_exit_type, exit_type_logit)

    soft_utility_vec = compute_soft_utility_numpy(
        trade_logit=trade_logit,
        dir_logit=dir_logit,
        returns=y_ret,
        k=float(cfg["utility_tanh_k"]),
        cfg=cfg,
    )
    soft_utility_mean = float(np.mean(soft_utility_vec)) if len(soft_utility_vec) else float("nan")
    denom = float(np.mean(np.abs(y_ret)) + 1e-12) if len(y_ret) else 1.0
    scaled_soft_utility = float(soft_utility_mean / denom) if np.isfinite(soft_utility_mean) else float("nan")
    selection_score = float(
        finite_or_default(scaled_soft_utility, 0.0)
        + 0.45 * finite_or_default(dir_auc, 0.5)
        + 0.15 * finite_or_default(trade_auc, 0.5)
    )

    threshold_grid_df: Optional[pd.DataFrame] = None
    trades_df = pd.DataFrame()
    threshold_metrics: Dict[str, Any] = {
        "gross_pnl_sum": float("nan"),
        "pnl_sum": float("nan"),
        "pnl_per_trade": float("nan"),
        "n_trades": 0,
        "trade_rate": float("nan"),
        "sign_accuracy": float("nan"),
        "win_rate": float("nan"),
        "long_trades": 0,
        "short_trades": 0,
        "long_pnl_sum": float("nan"),
        "short_pnl_sum": float("nan"),
        "sharpe_like": float("nan"),
        "timeout_trade_count": 0,
        "upper_exit_trade_count": 0,
        "lower_exit_trade_count": 0,
        "vertical_exit_trade_count": 0,
        "coverage": float("nan"),
        "backtest_mode": "skipped_threshold_free_eval",
        "backtest_uses_oracle_exit": False,
    }

    if perform_backtest and (search_threshold_pair_on_pack or selected_threshold_pair is None):
        selected_threshold_pair, threshold_grid_df, threshold_metrics, trades_df = search_best_threshold_pair(
            y_ret=y_ret,
            y_exit_type=y_exit_type,
            y_tte=y_tte,
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            raw_t_indices=raw_t,
            cfg=cfg,
            timestamps=TIMESTAMPS,
        )
    elif perform_backtest:
        long_mask, short_mask = apply_threshold_pair(
            trade_prob=trade_prob,
            dir_prob=dir_prob,
            exit_type_prob=None,
            thr_trade=float(selected_threshold_pair["thr_trade"]),
            thr_dir=float(selected_threshold_pair["thr_dir"]),
            cfg=cfg,
        )
        threshold_metrics, trades_df = sequential_event_backtest_from_masks(
            y_true=y_ret,
            y_exit_type=y_exit_type,
            y_tte=y_tte,
            raw_t_indices=raw_t,
            long_mask=long_mask,
            short_mask=short_mask,
            cfg=cfg,
            timestamps=TIMESTAMPS,
            build_trades=True,
        )
        threshold_metrics["coverage"] = float((long_mask | short_mask).mean()) if len(long_mask) else float("nan")
    else:
        selected_threshold_pair = {
            "thr_trade": float(threshold_trade_grid_values(cfg)[0]),
            "thr_dir": float(cfg["thr_dir_grid"][0]),
            "coverage": float("nan"),
            "feasible": True,
            "selection_stage": "threshold_free_only",
        }

    metrics = {
        "rmse": rmse_np(y_ret, return_pred),
        "mae": mae_np(y_ret, return_pred),
        "ic": ic_np(y_ret, return_pred),
        "trade_auc": trade_auc,
        "dir_auc": dir_auc,
        "exit_type_accuracy": exit_type_acc,
        "avg_predicted_tte": float(np.mean(tte_pred)) if len(tte_pred) else float("nan"),
        "avg_true_tte": float(np.mean(y_tte)) if len(y_tte) else float("nan"),
        "soft_utility_mean": soft_utility_mean,
        "scaled_soft_utility": scaled_soft_utility,
        "selection_score": selection_score,
        "early_stop_score": selection_score,
        "predicted_exit_bar_mean": float(np.mean(predicted_exit_bars)) if len(predicted_exit_bars) else float("nan"),
        "predicted_timeout_rate": float(np.mean(predicted_exit_type == EXIT_TYPE_TO_IDX["vertical"])) if len(predicted_exit_type) else float("nan"),
        **threshold_metrics,
        "trades_df": trades_df,
        "selected_threshold_pair": copy.deepcopy(selected_threshold_pair),
    }
    return metrics, threshold_grid_df, selected_threshold_pair


def checkpoint_key_from_metrics(metrics):
    return (
        finite_or_default(metrics.get("selection_score"), -1e9),
        finite_or_default(metrics.get("scaled_soft_utility"), -1e9),
        finite_or_default(metrics.get("dir_auc"), -1e9),
        finite_or_default(metrics.get("trade_auc"), -1e9),
        finite_or_default(metrics.get("ic"), -1e9),
    )



def better_selection_key(candidate: Tuple[float, ...], incumbent: Optional[Tuple[float, ...]]) -> bool:
    if incumbent is None:
        return True
    return candidate > incumbent


# %% Prediction helpers

@torch.no_grad()
def predict_on_indices(
    model: nn.Module,
    x_node_scaled: np.ndarray,
    x_rel_edge_scaled: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> Dict[str, Any]:
    model.eval()
    trade_logits, dir_logits = [], []
    return_preds, exit_type_logits, tte_preds = [], [], []
    y_ret_all, y_trade_all, y_dir_all, y_dir_mask_all = [], [], [], []
    y_exit_type_all, y_tte_all = [], []
    sample_idx_all, raw_t_all = [], []
    state: Optional[Dict[str, torch.Tensor]] = None

    for chunk in build_stateful_chunks(indices=indices, cfg=CFG, for_eval=True):
        if chunk.is_segment_start:
            state = None

        batch = stateful_chunk_to_batch(
            chunk_sample_indices=chunk.sample_indices,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
        )
        x_node_seq = batch["x_node_seq"].to(DEVICE).float()
        x_edge_seq = batch["x_edge_seq"].to(DEVICE).float()

        outputs = model(
            x_node_seq,
            x_edge_seq,
            prev_state=state,
            return_state=True,
            return_aux=False,
        )
        state = detach_memorygraph_state(outputs.get("next_state"))

        trade_logits.append(outputs["trade_logit"].detach().cpu().numpy().reshape(-1))
        dir_logits.append(outputs["dir_logit"].detach().cpu().numpy().reshape(-1))
        return_preds.append(outputs["return_pred"].detach().cpu().numpy().reshape(-1))
        exit_type_logits.append(outputs["exit_type_logit"].detach().cpu().numpy().reshape(-1, len(EXIT_TYPE_NAMES)))
        tte_preds.append(outputs["tte_pred"].detach().cpu().numpy().reshape(-1))

        y_ret_all.append(batch["y_ret"].detach().cpu().numpy().reshape(-1))
        y_trade_all.append(batch["y_trade"].detach().cpu().numpy().reshape(-1))
        y_dir_all.append(batch["y_dir"].detach().cpu().numpy().reshape(-1))
        y_dir_mask_all.append(batch["y_dir_mask"].detach().cpu().numpy().reshape(-1))
        y_exit_type_all.append(batch["y_exit_type"].detach().cpu().numpy().reshape(-1))
        y_tte_all.append(batch["y_tte"].detach().cpu().numpy().reshape(-1))
        sample_idx_all.append(batch["sample_idx"].detach().cpu().numpy().reshape(-1))
        raw_t_all.append(batch["raw_t"].detach().cpu().numpy().reshape(-1))

    if not trade_logits:
        return {
            "trade_logit": np.empty((0,), dtype=np.float64),
            "dir_logit": np.empty((0,), dtype=np.float64),
            "return_pred": np.empty((0,), dtype=np.float64),
            "fixed_pred": np.empty((0,), dtype=np.float64),
            "exit_type_logit": np.empty((0, len(EXIT_TYPE_NAMES)), dtype=np.float64),
            "tte_pred": np.empty((0,), dtype=np.float64),
            "trade_prob": np.empty((0,), dtype=np.float64),
            "dir_prob": np.empty((0,), dtype=np.float64),
            "exit_type_prob": np.empty((0, len(EXIT_TYPE_NAMES)), dtype=np.float64),
            "y_ret": np.empty((0,), dtype=np.float64),
            "y_trade": np.empty((0,), dtype=np.float64),
            "y_dir": np.empty((0,), dtype=np.float64),
            "y_dir_mask": np.empty((0,), dtype=np.float64),
            "y_exit_type": np.empty((0,), dtype=np.int64),
            "y_tte": np.empty((0,), dtype=np.float64),
            "sample_idx": np.empty((0,), dtype=np.int64),
            "raw_t": np.empty((0,), dtype=np.int64),
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
        }

    trade_logit_arr = np.concatenate(trade_logits, axis=0).astype(np.float64)
    dir_logit_arr = np.concatenate(dir_logits, axis=0).astype(np.float64)
    return_pred_arr = np.concatenate(return_preds, axis=0).astype(np.float64)
    exit_type_logit_arr = np.concatenate(exit_type_logits, axis=0).astype(np.float64)
    tte_pred_arr = np.concatenate(tte_preds, axis=0).astype(np.float64)
    y_ret_arr = np.concatenate(y_ret_all, axis=0).astype(np.float64)
    y_trade_arr = np.concatenate(y_trade_all, axis=0).astype(np.float64)
    y_dir_arr = np.concatenate(y_dir_all, axis=0).astype(np.float64)
    y_dir_mask_arr = np.concatenate(y_dir_mask_all, axis=0).astype(np.float64)
    y_exit_type_arr = np.concatenate(y_exit_type_all, axis=0).astype(np.int64)
    y_tte_arr = np.concatenate(y_tte_all, axis=0).astype(np.float64)
    sample_idx_arr = np.concatenate(sample_idx_all, axis=0).astype(np.int64)
    raw_t_arr = np.concatenate(raw_t_all, axis=0).astype(np.int64)

    return {
        "trade_logit": trade_logit_arr,
        "dir_logit": dir_logit_arr,
        "return_pred": return_pred_arr,
        "fixed_pred": return_pred_arr,
        "exit_type_logit": exit_type_logit_arr,
        "tte_pred": tte_pred_arr,
        "trade_prob": trade_activation_numpy(trade_logit_arr, CFG),
        "dir_prob": sigmoid_np(dir_logit_arr),
        "exit_type_prob": softmax_np(exit_type_logit_arr, axis=1),
        "y_ret": y_ret_arr,
        "y_trade": y_trade_arr,
        "y_dir": y_dir_arr,
        "y_dir_mask": y_dir_mask_arr,
        "y_exit_type": y_exit_type_arr,
        "y_tte": y_tte_arr,
        "sample_idx": sample_idx_arr,
        "raw_t": raw_t_arr,
        "timestamp": TIMESTAMPS.iloc[raw_t_arr].reset_index(drop=True),
    }


# %% Training helpers

@dataclass
class SplitArtifacts:
    model_state: Dict[str, torch.Tensor]
    node_scaler_params: Dict[str, Any]
    relation_scaler_params: Dict[str, Dict[str, Any]]
    loss_state: Dict[str, float]
    best_epoch: int
    best_checkpoint_key: Tuple[float, ...]
    best_checkpoint_summary: Dict[str, Any]
    selected_threshold_pair: Dict[str, Any]
    validation_threshold_grid: pd.DataFrame
    early_stop_val_metrics: Dict[str, Any]
    calibration_val_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    test_metrics: Dict[str, Any]
    early_stop_predictions: Dict[str, Any]
    calibration_predictions: Dict[str, Any]
    val_predictions: Dict[str, Any]
    test_predictions: Dict[str, Any]
    timing: Dict[str, Any]
    epoch_durations_sec: List[float]



def build_run_cfg(base_cfg: Dict[str, Any], operator_name: str, is_ablation_context: bool) -> Dict[str, Any]:
    run_cfg = build_model_runtime_cfg(base_cfg=base_cfg, operator_name=operator_name, is_ablation_context=is_ablation_context)
    return run_cfg



def build_model_runtime_cfg(base_cfg: Dict[str, Any], operator_name: str, is_ablation_context: bool) -> Dict[str, Any]:
    run_cfg = copy.deepcopy(base_cfg)
    run_cfg["graph_operator"] = str(operator_name)
    if is_ablation_context and bool(base_cfg["ablation_fast_mode"]):
        run_cfg["epochs"] = int(base_cfg["ablation_epochs"])
        run_cfg["patience"] = int(base_cfg["ablation_patience"])
    return run_cfg



def build_model_for_cfg(cfg: Dict[str, Any]) -> MemoryGraphTemporalFusionModel:
    model = MemoryGraphTemporalFusionModel(
        node_dim=X_NODE_RAW.shape[-1],
        edge_dim=X_REL_EDGE_RAW.shape[-1],
        n_nodes=len(ASSETS),
        target_node=TARGET_NODE,
        relation_names=RELATION_NAMES,
        cfg=cfg,
    ).to(DEVICE)
    return model


def run_stateful_training_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_node_scaled: np.ndarray,
    x_rel_edge_scaled: np.ndarray,
    idx_train: np.ndarray,
    loss_state: LossState,
    cfg: Dict[str, Any],
) -> Dict[str, List[float]]:
    stats = {
        "train_total_loss": [],
        "train_trade_loss": [],
        "train_dir_loss": [],
        "train_ret_loss": [],
        "train_exit_type_loss": [],
        "train_tte_loss": [],
        "train_utility_loss": [],
        "train_false_positive_penalty": [],
        "train_timeout_penalty": [],
        "train_adj_reg": [],
        "train_soft_utility": [],
    }
    state: Optional[Dict[str, torch.Tensor]] = None

    for chunk in build_stateful_chunks(indices=idx_train, cfg=cfg, for_eval=False):
        if chunk.is_segment_start:
            state = None

        batch = stateful_chunk_to_batch(
            chunk_sample_indices=chunk.sample_indices,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
        )
        x_node_seq = batch["x_node_seq"].to(DEVICE).float()
        x_edge_seq = batch["x_edge_seq"].to(DEVICE).float()
        target_batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
            if key not in {"x_node_seq", "x_edge_seq", "sample_idx", "raw_t"}
        }

        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            x_node_seq,
            x_edge_seq,
            prev_state=state,
            return_state=True,
            return_aux=False,
        )
        next_state = outputs.pop("next_state", None)
        loss_pack = compute_total_loss(
            outputs=outputs,
            batch=target_batch,
            loss_state=loss_state,
            cfg=cfg,
        )
        loss_pack["total_loss"].backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(cfg["grad_clip"]))
        optimizer.step()
        state = detach_memorygraph_state(next_state)

        stats["train_total_loss"].append(float(loss_pack["total_loss"].detach().cpu().item()))
        stats["train_trade_loss"].append(float(loss_pack["trade_loss"].cpu().item()))
        stats["train_dir_loss"].append(float(loss_pack["dir_loss"].cpu().item()))
        stats["train_ret_loss"].append(float(loss_pack["ret_loss"].cpu().item()))
        stats["train_exit_type_loss"].append(float(loss_pack["exit_type_loss"].cpu().item()))
        stats["train_tte_loss"].append(float(loss_pack["tte_loss"].cpu().item()))
        stats["train_utility_loss"].append(float(loss_pack["utility_loss"].cpu().item()))
        stats["train_false_positive_penalty"].append(float(loss_pack["false_positive_penalty_loss"].cpu().item()))
        stats["train_timeout_penalty"].append(float(loss_pack["timeout_penalty_loss"].cpu().item()))
        stats["train_adj_reg"].append(float(loss_pack["adj_reg"].cpu().item()))
        stats["train_soft_utility"].append(float(loss_pack["soft_utility_mean"].cpu().item()))

    return stats



def train_one_split(
    split_name: str,
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    cfg: Dict[str, Any],
    evaluate_test_split: bool = True,
) -> SplitArtifacts:
    split_start_perf = time.perf_counter()
    split_start_utc = utc_now_iso()

    x_node_scaled, node_scaler_params = fit_robust_scaler_train_only_3d(
        raw_array=X_NODE_RAW,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_node_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )

    x_rel_edge_scaled, relation_scaler_params = fit_relation_scalers_train_only(
        raw_rel_array=X_REL_EDGE_RAW,
        relation_names=RELATION_NAMES,
        sample_t=SAMPLE_T,
        train_sample_indices=idx_train,
        max_abs_value=float(cfg["max_abs_edge_feature"]),
        q_low=float(cfg["scaler_quantile_low"]),
        q_high=float(cfg["scaler_quantile_high"]),
    )

    loss_state = build_loss_state(idx_train)
    early_stop_idx, calibration_idx = split_validation_indices_for_memorygraph(idx_val, cfg)

    model = build_model_for_cfg(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )

    best_state = None
    best_epoch = -1
    best_checkpoint_key: Optional[Tuple[float, ...]] = None
    best_checkpoint_summary: Dict[str, Any] = {}
    bad_epochs = 0
    epoch_durations_sec: List[float] = []
    best_epoch_cumulative_train_sec: Optional[float] = None

    LOGGER.info(
        "[%s][%s] validation policy: epoch val_* metrics are threshold-free early-stop metrics; "
        "threshold calibration on the tail validation slice runs once after checkpoint selection.",
        split_name,
        cfg["graph_operator"],
    )

    for epoch in range(1, int(cfg["epochs"]) + 1):
        epoch_start_perf = time.perf_counter()
        model.train()
        epoch_stats = run_stateful_training_epoch(
            model=model,
            optimizer=optimizer,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
            idx_train=idx_train,
            loss_state=loss_state,
            cfg=cfg,
        )

        epoch_duration_sec = float(time.perf_counter() - epoch_start_perf)
        epoch_durations_sec.append(epoch_duration_sec)

        early_stop_pred_pack = predict_on_indices(
            model=model,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
            indices=early_stop_idx,
            batch_size=int(cfg["batch_size"]),
        )
        early_stop_metrics, _unused_grid, default_threshold_pair = evaluate_prediction_pack(
            pred_pack=early_stop_pred_pack,
            cfg=cfg,
            selected_threshold_pair=None,
            search_threshold_pair_on_pack=False,
            perform_backtest=False,
        )

        checkpoint_key = checkpoint_key_from_metrics(early_stop_metrics)
        scheduler.step(float(early_stop_metrics["selection_score"]))

        LOGGER.info(
            "[%s][%s] epoch=%02d loss=%.6f trade_bce=%.6f dir_bce=%.6f ret_huber=%.6f exit_ce=%.6f tte_huber=%.6f utility_loss=%.6f fp_pen=%.6f timeout_pen=%.6f adj_reg=%.6f train_soft_util=%.6f val_selection=%.6f val_soft_util=%.6f val_dir_auc=%.4f val_trade_auc=%.4f val_ic=%.4f lr=%.2e epoch_sec=%.2f",
            split_name,
            cfg["graph_operator"],
            epoch,
            np.mean(epoch_stats["train_total_loss"]),
            np.mean(epoch_stats["train_trade_loss"]),
            np.mean(epoch_stats["train_dir_loss"]),
            np.mean(epoch_stats["train_ret_loss"]),
            np.mean(epoch_stats["train_exit_type_loss"]),
            np.mean(epoch_stats["train_tte_loss"]),
            np.mean(epoch_stats["train_utility_loss"]),
            np.mean(epoch_stats["train_false_positive_penalty"]),
            np.mean(epoch_stats["train_timeout_penalty"]),
            np.mean(epoch_stats["train_adj_reg"]),
            np.mean(epoch_stats["train_soft_utility"]),
            early_stop_metrics["selection_score"],
            early_stop_metrics["scaled_soft_utility"],
            finite_or_default(early_stop_metrics["dir_auc"], float("nan")),
            finite_or_default(early_stop_metrics["trade_auc"], float("nan")),
            finite_or_default(early_stop_metrics["ic"], float("nan")),
            optimizer.param_groups[0]["lr"],
            epoch_duration_sec,
        )

        if better_selection_key(checkpoint_key, best_checkpoint_key):
            best_checkpoint_key = checkpoint_key
            best_epoch = int(epoch)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch_cumulative_train_sec = float(sum(epoch_durations_sec))
            best_checkpoint_summary = {
                "epoch": int(epoch),
                "checkpoint_key": list(checkpoint_key),
                "selected_threshold_pair": None,
                "early_stop_val_metrics": {
                    k: v
                    for k, v in early_stop_metrics.items()
                    if k not in {"trades_df", "selected_threshold_pair"}
                },
                "early_stop_sample_count": int(len(early_stop_idx)),
                "threshold_calibration_sample_count": int(len(calibration_idx)),
                "threshold_free_selection": True,
                "default_threshold_pair": copy.deepcopy(default_threshold_pair),
                "epoch_duration_sec": epoch_duration_sec,
                "cumulative_train_duration_sec": best_epoch_cumulative_train_sec,
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= int(cfg["patience"]):
            LOGGER.info("[%s][%s] early stopping at epoch %s", split_name, cfg["graph_operator"], epoch)
            break

    if best_state is None or best_checkpoint_key is None:
        raise RuntimeError(f"[{split_name}] no best checkpoint was captured during training")

    model.load_state_dict(best_state)

    early_stop_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=early_stop_idx,
        batch_size=int(cfg["batch_size"]),
    )
    early_stop_val_metrics, _, _ = evaluate_prediction_pack(
        pred_pack=early_stop_pred_pack,
        cfg=cfg,
        selected_threshold_pair=None,
        search_threshold_pair_on_pack=False,
        perform_backtest=False,
    )

    calibration_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=calibration_idx,
        batch_size=int(cfg["batch_size"]),
    )
    calibration_val_metrics, validation_threshold_grid, selected_threshold_pair = evaluate_prediction_pack(
        pred_pack=calibration_pred_pack,
        cfg=cfg,
        selected_threshold_pair=None,
        search_threshold_pair_on_pack=True,
    )
    selected_threshold_pair = copy.deepcopy(selected_threshold_pair)
    selected_threshold_pair["selection_stage"] = "threshold_calibration_val"
    selected_threshold_pair["calibration_sample_count"] = int(len(calibration_idx))

    val_pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=idx_val,
        batch_size=int(cfg["batch_size"]),
    )
    val_metrics, _, _ = evaluate_prediction_pack(
        pred_pack=val_pred_pack,
        cfg=cfg,
        selected_threshold_pair=selected_threshold_pair,
        search_threshold_pair_on_pack=False,
        perform_backtest=True,
    )
    val_metrics["threshold_calibration_split"] = "tail_of_validation"
    val_metrics["early_stop_sample_count"] = int(len(early_stop_idx))
    val_metrics["threshold_calibration_sample_count"] = int(len(calibration_idx))

    LOGGER.info(
        "[%s][%s] VAL calibrated selection=%.6f soft_util=%.6f dir_auc=%.4f trade_auc=%.4f val_pnl_sum=%.6f val_ppt=%.6f val_trades=%s thr_trade=%.2f thr_dir=%.2f",
        split_name,
        cfg["graph_operator"],
        val_metrics["selection_score"],
        val_metrics["scaled_soft_utility"],
        finite_or_default(val_metrics["dir_auc"], float("nan")),
        finite_or_default(val_metrics["trade_auc"], float("nan")),
        finite_or_default(val_metrics["pnl_sum"], float("nan")),
        finite_or_default(val_metrics["pnl_per_trade"], float("nan")),
        int(val_metrics["n_trades"]),
        float(selected_threshold_pair["thr_trade"]),
        float(selected_threshold_pair["thr_dir"]),
    )

    if evaluate_test_split:
        test_pred_pack = predict_on_indices(
            model=model,
            x_node_scaled=x_node_scaled,
            x_rel_edge_scaled=x_rel_edge_scaled,
            indices=idx_test,
            batch_size=int(cfg["batch_size"]),
        )
        test_metrics, _, _ = evaluate_prediction_pack(
            pred_pack=test_pred_pack,
            cfg=cfg,
            selected_threshold_pair=selected_threshold_pair,
            search_threshold_pair_on_pack=False,
            perform_backtest=True,
        )
        test_metrics["threshold_calibration_sample_count"] = int(len(calibration_idx))
    else:
        test_pred_pack = {}
        test_metrics = {}

    split_end_utc = utc_now_iso()
    split_duration_sec = float(time.perf_counter() - split_start_perf)

    LOGGER.info(
        "[%s][%s] best_epoch=%s best_key=%s selected_threshold_pair=(thr_trade=%.2f, thr_dir=%.2f) early_stop_n=%s calibration_n=%s",
        split_name,
        cfg["graph_operator"],
        best_epoch,
        best_checkpoint_key,
        selected_threshold_pair["thr_trade"],
        selected_threshold_pair["thr_dir"],
        len(early_stop_idx),
        len(calibration_idx),
    )
    if evaluate_test_split:
        LOGGER.info(
            "[%s][%s] TEST selection=%.6f soft_util=%.6f dir_auc=%.4f trade_auc=%.4f rmse=%.6f ic=%.4f pnl_sum=%.6f pnl_per_trade=%.6f n_trades=%s split_duration_sec=%.2f",
            split_name,
            cfg["graph_operator"],
            test_metrics["selection_score"],
            test_metrics["scaled_soft_utility"],
            finite_or_default(test_metrics["dir_auc"], float("nan")),
            finite_or_default(test_metrics["trade_auc"], float("nan")),
            test_metrics["rmse"],
            finite_or_default(test_metrics["ic"], float("nan")),
            finite_or_default(test_metrics["pnl_sum"], float("nan")),
            finite_or_default(test_metrics["pnl_per_trade"], float("nan")),
            int(test_metrics["n_trades"]),
            split_duration_sec,
        )
    else:
        LOGGER.info(
            "[%s][%s] blind holdout evaluation deferred until the final comparison stage | split_duration_sec=%.2f",
            split_name,
            cfg["graph_operator"],
            split_duration_sec,
        )

    timing = {
        "split_name": split_name,
        "split_train_start_utc": split_start_utc,
        "split_train_end_utc": split_end_utc,
        "split_train_duration_sec": split_duration_sec,
        "best_epoch_cumulative_train_duration_sec": best_epoch_cumulative_train_sec,
    }

    return SplitArtifacts(
        model_state=copy.deepcopy(model.state_dict()),
        node_scaler_params=node_scaler_params,
        relation_scaler_params=relation_scaler_params,
        loss_state={
            "pos_weight_trade": float(loss_state.pos_weight_trade),
            "pos_weight_dir": float(loss_state.pos_weight_dir),
        },
        best_epoch=best_epoch,
        best_checkpoint_key=best_checkpoint_key,
        best_checkpoint_summary=best_checkpoint_summary,
        selected_threshold_pair=copy.deepcopy(selected_threshold_pair),
        validation_threshold_grid=validation_threshold_grid.copy(),
        early_stop_val_metrics=early_stop_val_metrics,
        calibration_val_metrics=calibration_val_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        early_stop_predictions=early_stop_pred_pack,
        calibration_predictions=calibration_pred_pack,
        val_predictions=val_pred_pack,
        test_predictions=test_pred_pack,
        timing=timing,
        epoch_durations_sec=epoch_durations_sec,
    )

# %% Bundle save/load helpers

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_dict()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, tuple):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonable(v) for v in obj]
    return obj


def save_bundle(
    bundle_dir: Path,
    bundle_name: str,
    model_state: Dict[str, torch.Tensor],
    node_scaler_params: Dict[str, Any],
    relation_scaler_params: Dict[str, Dict[str, Any]],
    cfg: Dict[str, Any],
    meta: Dict[str, Any],
) -> Dict[str, Path]:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = bundle_dir / f"{bundle_name}.pt"
    meta_path = bundle_dir / f"{bundle_name}_meta.json"

    payload = {
        "bundle_name": bundle_name,
        "cfg": copy.deepcopy(cfg),
        "model_state": model_state,
        "node_scaler_params": node_scaler_params,
        "relation_scaler_params": relation_scaler_params,
        "relation_names": RELATION_NAMES,
        "assets": ASSETS,
        "target_asset": TARGET_ASSET,
        "freq": FREQ,
        "expected_delta_seconds": freq_to_seconds(FREQ),
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_bars": HORIZON_BARS,
        "lookback_bars": LOOKBACK_BARS,
        "meta": meta,
    }

    torch.save(payload, str(bundle_path))

    meta_json = {
        "bundle_name": bundle_name,
        "bundle_file": bundle_path.name,
        "cfg": _jsonable(cfg),
        "relation_names": RELATION_NAMES,
        "assets": ASSETS,
        "target_asset": TARGET_ASSET,
        "freq": FREQ,
        "expected_delta_seconds": freq_to_seconds(FREQ),
        "horizon_minutes": HORIZON_MINUTES,
        "horizon_bars": HORIZON_BARS,
        "lookback_bars": LOOKBACK_BARS,
        **_jsonable(meta),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_json, f, indent=2)

    return {"bundle": bundle_path, "meta": meta_path}


def load_bundle(bundle_dir: Path, bundle_name: str) -> Dict[str, Any]:
    bundle_path = bundle_dir / f"{bundle_name}.pt"
    if not bundle_path.exists():
        raise FileNotFoundError(bundle_path)
    try:
        loaded = torch.load(str(bundle_path), map_location="cpu", weights_only=False)
    except TypeError:
        loaded = torch.load(str(bundle_path), map_location="cpu")
    return loaded


# %% CV runner and operator selection

def is_scalar_metric(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating, str, bool)) or value is None


def flatten_metrics_row(prefix: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in metrics.items():
        if k in {"trades_df", "selected_threshold_pair"}:
            continue
        if is_scalar_metric(v):
            row[f"{prefix}{k}" if prefix else k] = v
    return row


def flatten_threshold_pair(prefix: str, pair: Dict[str, Any]) -> Dict[str, Any]:
    row = {}
    for k, v in pair.items():
        if is_scalar_metric(v):
            row[f"{prefix}{k}" if prefix else k] = v
    return row


def run_cv_for_operator(
    operator_name: str,
    base_cfg: Dict[str, Any],
    is_ablation_context: bool = True,
) -> Dict[str, Any]:
    run_cfg = build_run_cfg(base_cfg=base_cfg, operator_name=operator_name, is_ablation_context=is_ablation_context)
    operator_dir = ARTIFACT_ROOT / operator_name
    operator_dir.mkdir(parents=True, exist_ok=True)

    cv_rows: List[Dict[str, Any]] = []
    best_cv_bundle_name: Optional[str] = None
    best_cv_key: Optional[Tuple[float, ...]] = None
    fold_bundle_names: List[str] = []
    fold_timings: List[Dict[str, Any]] = []

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(WALK_FORWARD_SPLITS, start=1):
        LOGGER.info(
            "OPERATOR=%s | FOLD %s/%s train=%s val=%s test=%s",
            operator_name,
            fold_idx,
            len(WALK_FORWARD_SPLITS),
            len(idx_train),
            len(idx_val),
            len(idx_test),
        )

        artifacts = train_one_split(
            split_name=f"{operator_name}_fold_{fold_idx:02d}",
            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,
            cfg=run_cfg,
        )

        bundle_name = f"{operator_name}_fold_{fold_idx:02d}_best"
        fold_bundle_names.append(bundle_name)

        save_bundle(
            bundle_dir=operator_dir,
            bundle_name=bundle_name,
            model_state=artifacts.model_state,
            node_scaler_params=artifacts.node_scaler_params,
            relation_scaler_params=artifacts.relation_scaler_params,
            cfg=run_cfg,
            meta={
                "kind": "cv_fold_best",
                "run_id": RUN_ID,
                "operator_name": operator_name,
                "fold_idx": fold_idx,
                "best_epoch": artifacts.best_epoch,
                "best_checkpoint_key": list(artifacts.best_checkpoint_key),
                "best_checkpoint_summary": artifacts.best_checkpoint_summary,
                "loss_state": artifacts.loss_state,
                "selected_threshold_pair": artifacts.selected_threshold_pair,
                "early_stop_val_metrics": {
                    k: v
                    for k, v in artifacts.early_stop_val_metrics.items()
                    if k not in {"trades_df", "selected_threshold_pair"}
                },
                "calibration_val_metrics": {
                    k: v
                    for k, v in artifacts.calibration_val_metrics.items()
                    if k not in {"trades_df", "selected_threshold_pair"}
                },
                "timing": artifacts.timing,
                "epoch_durations_sec": artifacts.epoch_durations_sec,
                "idx_train": idx_train.tolist(),
                "idx_val": idx_val.tolist(),
                "idx_test": idx_test.tolist(),
            },
        )

        artifacts.validation_threshold_grid.to_csv(
            operator_dir / f"{bundle_name}_validation_threshold_grid.csv",
            index=False,
        )
        artifacts.val_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_val_trades.csv",
            index=False,
        )
        artifacts.calibration_val_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_calibration_trades.csv",
            index=False,
        )
        artifacts.test_metrics["trades_df"].to_csv(
            operator_dir / f"{bundle_name}_test_trades.csv",
            index=False,
        )

        row = {
            "run_id": RUN_ID,
            "operator": operator_name,
            "fold": fold_idx,
            "best_epoch": artifacts.best_epoch,
            "best_checkpoint_key": json.dumps(list(artifacts.best_checkpoint_key)),
            "fold_train_start_utc": artifacts.timing["split_train_start_utc"],
            "fold_train_end_utc": artifacts.timing["split_train_end_utc"],
            "fold_train_duration_sec": artifacts.timing["split_train_duration_sec"],
            "best_epoch_cumulative_train_duration_sec": artifacts.timing["best_epoch_cumulative_train_duration_sec"],
            **flatten_threshold_pair("selected_", artifacts.selected_threshold_pair),
            **flatten_metrics_row("early_stop_", artifacts.early_stop_val_metrics),
            **flatten_metrics_row("calibration_", artifacts.calibration_val_metrics),
            **flatten_metrics_row("val_", artifacts.val_metrics),
            **flatten_metrics_row("test_", artifacts.test_metrics),
        }
        cv_rows.append(row)
        fold_timings.append(copy.deepcopy(artifacts.timing))

        fold_key = (
            finite_or_default(artifacts.test_metrics.get("pnl_sum"), -1e9),
            finite_or_default(artifacts.test_metrics.get("selection_score"), -1e9),
            finite_or_default(artifacts.test_metrics.get("dir_auc"), -1e9),
            finite_or_default(artifacts.test_metrics.get("pnl_per_trade"), -1e9),
            *artifacts.best_checkpoint_key,
        )
        if better_selection_key(fold_key, best_cv_key):
            best_cv_key = fold_key
            best_cv_bundle_name = bundle_name

    if best_cv_bundle_name is None or best_cv_key is None:
        raise RuntimeError(f"{operator_name}: no best CV bundle selected")

    cv_results_df = pd.DataFrame(cv_rows)
    cv_results_df.to_csv(operator_dir / f"{operator_name}_cv_results_summary.csv", index=False)

    cv_mean_numeric = cv_results_df.mean(numeric_only=True).to_dict()
    cv_mean_summary_row = {
        "run_id": RUN_ID,
        "operator": operator_name,
        "graph_operator": operator_name,
        "cv_mean_test_selection_score": float(cv_mean_numeric.get("test_selection_score", np.nan)),
        "cv_mean_test_scaled_soft_utility": float(cv_mean_numeric.get("test_scaled_soft_utility", np.nan)),
        "cv_mean_test_dir_auc": float(cv_mean_numeric.get("test_dir_auc", np.nan)),
        "cv_mean_test_trade_auc": float(cv_mean_numeric.get("test_trade_auc", np.nan)),
        "cv_mean_test_rmse": float(cv_mean_numeric.get("test_rmse", np.nan)),
        "cv_mean_test_ic": float(cv_mean_numeric.get("test_ic", np.nan)),
        "cv_mean_test_pnl_sum": float(cv_mean_numeric.get("test_pnl_sum", np.nan)),
        "cv_mean_test_pnl_per_trade": float(cv_mean_numeric.get("test_pnl_per_trade", np.nan)),
        "cv_mean_test_sharpe_like": float(cv_mean_numeric.get("test_sharpe_like", np.nan)),
        "cv_mean_fold_train_duration_sec": float(cv_mean_numeric.get("fold_train_duration_sec", np.nan)),
        "cv_total_fold_train_duration_sec": float(cv_results_df["fold_train_duration_sec"].sum()),
    }
    cv_mean_df = pd.DataFrame([cv_mean_summary_row])
    cv_mean_df.to_csv(operator_dir / f"{operator_name}_cv_mean_summary.csv", index=False)

    LOGGER.info("CV_RESULTS_DF [%s]\n%s", operator_name, cv_results_df)
    LOGGER.info("CV mean metrics [%s]\n%s", operator_name, cv_mean_df)

    return {
        "operator_name": operator_name,
        "cfg": run_cfg,
        "artifact_dir": operator_dir,
        "fold_bundle_names": fold_bundle_names,
        "best_cv_bundle_name": best_cv_bundle_name,
        "last_cv_bundle_name": fold_bundle_names[-1],
        "best_cv_key": best_cv_key,
        "cv_results_df": cv_results_df,
        "cv_mean_df": cv_mean_df,
        "fold_timings": fold_timings,
    }


def select_best_operator_from_cv_runs(operator_runs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    best_operator_name: Optional[str] = None
    best_key: Optional[Tuple[float, ...]] = None

    for operator_name, run_out in operator_runs.items():
        row = run_out["cv_mean_df"].iloc[0].to_dict()
        key = (
            finite_or_default(row["cv_mean_test_pnl_sum"], -1e9),
            finite_or_default(row["cv_mean_test_selection_score"], -1e9),
            finite_or_default(row["cv_mean_test_dir_auc"], -1e9),
            finite_or_default(row["cv_mean_test_pnl_per_trade"], -1e9),
        )
        row["cv_operator_selection_key"] = json.dumps(list(key))
        rows.append(row)

        if better_selection_key(key, best_key):
            best_key = key
            best_operator_name = operator_name

    if best_operator_name is None:
        raise RuntimeError("No operator selected from CV runs")

    operator_comparison_df = pd.DataFrame(rows).sort_values(
        by=[
            "cv_mean_test_pnl_sum",
            "cv_mean_test_selection_score",
            "cv_mean_test_dir_auc",
            "cv_mean_test_pnl_per_trade",
        ],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    operator_comparison_df.to_csv(ARTIFACT_ROOT / "operator_cv_comparison_summary.csv", index=False)

    selected_operator_run = operator_runs[best_operator_name]
    LOGGER.info("OPERATOR_CV_COMPARISON_DF\n%s", operator_comparison_df)
    LOGGER.info("SELECTED_OPERATOR_FROM_CV=%s | KEY=%s", best_operator_name, best_key)

    return {
        "selected_operator_name": best_operator_name,
        "selected_operator_run": selected_operator_run,
        "selected_operator_key": best_key,
        "operator_comparison_df": operator_comparison_df,
    }

# %% Post-CV evaluation and production fit

@torch.no_grad()
def evaluate_saved_bundle_on_indices(
    bundle_dir: Path,
    bundle_name: str,
    indices: np.ndarray,
    label: str,
) -> Dict[str, Any]:
    loaded = load_bundle(bundle_dir, bundle_name)
    cfg = loaded["cfg"]

    model = build_model_for_cfg(cfg)
    model.load_state_dict(loaded["model_state"])
    model.eval()

    x_node_scaled = apply_robust_scaler_params_3d(X_NODE_RAW, loaded["node_scaler_params"])
    x_rel_edge_scaled = apply_relation_scalers(
        raw_rel_array=X_REL_EDGE_RAW,
        relation_names=loaded["relation_names"],
        relation_scaler_params=loaded["relation_scaler_params"],
    )

    pred_pack = predict_on_indices(
        model=model,
        x_node_scaled=x_node_scaled,
        x_rel_edge_scaled=x_rel_edge_scaled,
        indices=indices.astype(np.int64),
        batch_size=int(cfg["batch_size"]),
    )

    selected_threshold_pair = copy.deepcopy(loaded["meta"]["selected_threshold_pair"])

    metrics, _, _ = evaluate_prediction_pack(
        pred_pack=pred_pack,
        cfg=cfg,
        selected_threshold_pair=selected_threshold_pair,
        search_threshold_pair_on_pack=False,
    )

    LOGGER.info("%s | bundle_name=%s", label, bundle_name)
    LOGGER.info(
        "selected_threshold_pair=(thr_trade=%.2f, thr_dir=%.2f)",
        selected_threshold_pair["thr_trade"],
        selected_threshold_pair["thr_dir"],
    )
    LOGGER.info(
        "selection=%.6f soft_util=%.6f dir_auc=%.4f trade_auc=%.4f rmse=%.6f mae=%.6f ic=%.4f pnl_sum=%.6f "
        "pnl_per_trade=%.6f n_trades=%s trade_rate=%.4f sign_accuracy=%.4f",
        metrics["selection_score"],
        metrics["scaled_soft_utility"],
        finite_or_default(metrics["dir_auc"], float("nan")),
        finite_or_default(metrics["trade_auc"], float("nan")),
        metrics["rmse"],
        metrics["mae"],
        finite_or_default(metrics["ic"], float("nan")),
        finite_or_default(metrics["pnl_sum"], float("nan")),
        finite_or_default(metrics["pnl_per_trade"], float("nan")),
        int(metrics["n_trades"]),
        finite_or_default(metrics["trade_rate"], float("nan")),
        finite_or_default(metrics["sign_accuracy"], float("nan")),
    )

    return {
        "pred_pack": pred_pack,
        "metrics": metrics,
        "selected_threshold_pair": selected_threshold_pair,
    }


def build_final_holdout_trade_log_df(
    pred_pack: Dict[str, Any],
    trades_df: pd.DataFrame,
    selected_threshold_pair: Dict[str, Any],
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    trade_log_columns = [
        "trade_no",
        "entry_id",
        "exit_id",
        "entry_sample_id",
        "exit_sample_id",
        "entry_raw_t_id",
        "exit_raw_t_id",
        "entry_timestamp",
        "exit_timestamp",
        "side",
        "side_name",
        "trade_prob_at_entry",
        "dir_prob_at_entry",
        "return_pred_at_entry",
        "trade_prob_minus_threshold",
        "dir_decision_margin",
        "selected_thr_trade",
        "selected_thr_dir",
        "time_to_exit_bars",
        "time_to_exit_minutes",
        "exit_type",
        "realized_return",
        "gross_pnl",
        "net_pnl",
        "is_win",
        "is_direction_correct",
    ]
    if trades_df.empty:
        return pd.DataFrame(columns=trade_log_columns)

    sample_idx_arr = np.asarray(pred_pack["sample_idx"], dtype=np.int64)
    trade_prob_arr = np.asarray(pred_pack["trade_prob"], dtype=np.float64)
    dir_prob_arr = np.asarray(pred_pack["dir_prob"], dtype=np.float64)
    return_pred_arr = np.asarray(pred_pack["return_pred"], dtype=np.float64)
    raw_t_to_sample_id = {int(raw_t): int(sample_id) for sample_id, raw_t in enumerate(np.asarray(SAMPLE_T, dtype=np.int64))}
    minutes_per_bar = float(freq_to_seconds(str(cfg["freq"]))) / 60.0
    thr_trade = float(selected_threshold_pair["thr_trade"])
    thr_dir = float(selected_threshold_pair["thr_dir"])

    rows: List[Dict[str, Any]] = []
    for trade_no, trade in enumerate(trades_df.to_dict(orient="records"), start=1):
        entry_local_idx = int(trade["entry_local_idx"])
        entry_raw_t = int(trade["entry_raw_t"])
        exit_raw_t = int(trade["exit_raw_t"])
        side = int(trade["side"])
        side_name = "long" if side > 0 else "short"
        entry_sample_id = (
            int(sample_idx_arr[entry_local_idx])
            if 0 <= entry_local_idx < len(sample_idx_arr)
            else raw_t_to_sample_id.get(entry_raw_t)
        )
        exit_sample_id = raw_t_to_sample_id.get(exit_raw_t)
        trade_prob = float(trade_prob_arr[entry_local_idx]) if 0 <= entry_local_idx < len(trade_prob_arr) else float("nan")
        dir_prob = float(dir_prob_arr[entry_local_idx]) if 0 <= entry_local_idx < len(dir_prob_arr) else float("nan")
        return_pred = float(return_pred_arr[entry_local_idx]) if 0 <= entry_local_idx < len(return_pred_arr) else float("nan")
        time_to_exit_bars = int(trade["time_to_exit_bars"])
        direction_margin = (
            float(dir_prob - thr_dir)
            if side > 0
            else float((1.0 - thr_dir) - dir_prob)
        )
        realized_return = float(trade["realized_return"])
        net_pnl = float(trade["net_pnl"])

        rows.append(
            {
                "trade_no": trade_no,
                "entry_id": entry_sample_id,
                "exit_id": exit_sample_id,
                "entry_sample_id": entry_sample_id,
                "exit_sample_id": exit_sample_id,
                "entry_raw_t_id": entry_raw_t,
                "exit_raw_t_id": exit_raw_t,
                "entry_timestamp": trade.get("entry_timestamp"),
                "exit_timestamp": trade.get("exit_timestamp"),
                "side": side,
                "side_name": side_name,
                "trade_prob_at_entry": trade_prob,
                "dir_prob_at_entry": dir_prob,
                "return_pred_at_entry": return_pred,
                "trade_prob_minus_threshold": float(trade_prob - thr_trade),
                "dir_decision_margin": direction_margin,
                "selected_thr_trade": thr_trade,
                "selected_thr_dir": thr_dir,
                "time_to_exit_bars": time_to_exit_bars,
                "time_to_exit_minutes": float(time_to_exit_bars * minutes_per_bar),
                "exit_type": str(trade["exit_type"]),
                "realized_return": realized_return,
                "gross_pnl": float(trade["gross_pnl"]),
                "net_pnl": net_pnl,
                "is_win": int(net_pnl > 0.0),
                "is_direction_correct": int(side * realized_return > 0.0),
            }
        )

    return pd.DataFrame(rows, columns=trade_log_columns)


def run_selected_operator_post_cv_and_production(
    operator_run: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    operator_name = str(operator_run["operator_name"])
    operator_dir = Path(operator_run["artifact_dir"])
    run_cfg = copy.deepcopy(operator_run["cfg"])
    best_cv_bundle_name = str(operator_run["best_cv_bundle_name"])
    last_cv_bundle_name = str(operator_run["last_cv_bundle_name"])

    production_artifacts = train_one_split(
        split_name=f"{operator_name}_production_refit",
        idx_train=IDX_TRAIN_FINAL,
        idx_val=IDX_VAL_FINAL,
        idx_test=IDX_TEST_FINAL,
        cfg=run_cfg,
        evaluate_test_split=False,
    )

    production_bundle_name = f"{operator_name}_production_best"
    save_bundle(
        bundle_dir=operator_dir,
        bundle_name=production_bundle_name,
        model_state=production_artifacts.model_state,
        node_scaler_params=production_artifacts.node_scaler_params,
        relation_scaler_params=production_artifacts.relation_scaler_params,
        cfg=run_cfg,
        meta={
            "kind": "production_best",
            "run_id": RUN_ID,
            "operator_name": operator_name,
            "best_epoch": production_artifacts.best_epoch,
            "best_checkpoint_key": list(production_artifacts.best_checkpoint_key),
            "best_checkpoint_summary": production_artifacts.best_checkpoint_summary,
            "loss_state": production_artifacts.loss_state,
            "selected_threshold_pair": production_artifacts.selected_threshold_pair,
            "early_stop_val_metrics": {
                k: v
                for k, v in production_artifacts.early_stop_val_metrics.items()
                if k not in {"trades_df", "selected_threshold_pair"}
            },
            "calibration_val_metrics": {
                k: v
                for k, v in production_artifacts.calibration_val_metrics.items()
                if k not in {"trades_df", "selected_threshold_pair"}
            },
            "timing": production_artifacts.timing,
            "epoch_durations_sec": production_artifacts.epoch_durations_sec,
            "idx_train": IDX_TRAIN_FINAL.tolist(),
            "idx_val": IDX_VAL_FINAL.tolist(),
            "idx_test": IDX_TEST_FINAL.tolist(),
        },
    )

    production_artifacts.validation_threshold_grid.to_csv(
        operator_dir / f"{production_bundle_name}_validation_threshold_grid.csv",
        index=False,
    )
    production_artifacts.val_metrics["trades_df"].to_csv(
        operator_dir / f"{production_bundle_name}_val_trades.csv",
        index=False,
    )
    production_artifacts.calibration_val_metrics["trades_df"].to_csv(
        operator_dir / f"{production_bundle_name}_calibration_trades.csv",
        index=False,
    )

    holdout_evaluations = [
        {
            "model_role": "best_cv_model",
            "bundle_name": best_cv_bundle_name,
            "label": f"FINAL BLIND HOLDOUT [{operator_name}] USING BEST CV MODEL",
        },
        {
            "model_role": "last_cv_fold_model",
            "bundle_name": last_cv_bundle_name,
            "label": f"FINAL BLIND HOLDOUT [{operator_name}] USING LAST CV FOLD MODEL",
        },
        {
            "model_role": "final_refit_model",
            "bundle_name": production_bundle_name,
            "label": f"FINAL BLIND HOLDOUT [{operator_name}] USING FINAL REFIT MODEL",
            "fit_start_utc": production_artifacts.timing["split_train_start_utc"],
            "fit_end_utc": production_artifacts.timing["split_train_end_utc"],
            "fit_duration_sec": production_artifacts.timing["split_train_duration_sec"],
            "best_epoch_cumulative_train_duration_sec": production_artifacts.timing[
                "best_epoch_cumulative_train_duration_sec"
            ],
        },
    ]

    holdout_rows: List[Dict[str, Any]] = []
    for eval_spec in holdout_evaluations:
        holdout_result = evaluate_saved_bundle_on_indices(
            bundle_dir=operator_dir,
            bundle_name=str(eval_spec["bundle_name"]),
            indices=IDX_HOLDOUT,
            label=str(eval_spec["label"]),
        )
        row = {
            "run_id": RUN_ID,
            "operator": operator_name,
            "holdout_stage": "final_blind_holdout_comparison",
            "model_role": str(eval_spec["model_role"]),
            "model_name": str(eval_spec["bundle_name"]),
            "selected_for_production": bool(str(eval_spec["model_role"]) == "final_refit_model"),
            "holdout_is_diagnostic_only": True,
            **flatten_metrics_row("", holdout_result["metrics"]),
            **flatten_threshold_pair("selected_", holdout_result["selected_threshold_pair"]),
        }
        for extra_key in [
            "fit_start_utc",
            "fit_end_utc",
            "fit_duration_sec",
            "best_epoch_cumulative_train_duration_sec",
        ]:
            if extra_key in eval_spec:
                row[extra_key] = eval_spec[extra_key]
        holdout_rows.append(row)
        holdout_result["metrics"]["trades_df"].to_csv(
            operator_dir / f"{operator_name}_final_holdout_{eval_spec['model_role']}_trades.csv",
            index=False,
        )
        if str(eval_spec["model_role"]) == "last_cv_fold_model":
            last_cv_trade_log_df = build_final_holdout_trade_log_df(
                pred_pack=holdout_result["pred_pack"],
                trades_df=holdout_result["metrics"]["trades_df"],
                selected_threshold_pair=holdout_result["selected_threshold_pair"],
                cfg=cfg,
            )
            last_cv_trade_log_df.to_csv(
                operator_dir / f"{operator_name}_final_holdout_last_cv_trade_log.csv",
                index=False,
            )

    final_holdout_comparison_df = pd.DataFrame(holdout_rows)
    final_holdout_comparison_df.to_csv(
        operator_dir / f"{operator_name}_final_holdout_model_comparison_summary.csv",
        index=False,
    )

    summary_row = {
        "run_id": RUN_ID,
        "operator": operator_name,
        "graph_operator": operator_name,
        "selected_cv_bundle_name": best_cv_bundle_name,
        "last_cv_bundle_name": last_cv_bundle_name,
        "production_bundle_name": production_bundle_name,
        "selected_for_production_model_name": production_bundle_name,
        "selected_for_production_model_role": "final_refit_model",
        "holdout_policy": "diagnostic_only_no_reselection",
        "cv_model_selection_policy": "best_validation_checkpoint_across_cv_folds",
        "cv_mean_test_selection_score": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_selection_score"]),
        "cv_mean_test_dir_auc": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_dir_auc"]),
        "cv_mean_test_pnl_sum": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_test_pnl_sum"]),
        "cv_mean_fold_train_duration_sec": float(operator_run["cv_mean_df"].iloc[0]["cv_mean_fold_train_duration_sec"]),
        "cv_total_fold_train_duration_sec": float(operator_run["cv_mean_df"].iloc[0]["cv_total_fold_train_duration_sec"]),
        "production_fit_start_utc": production_artifacts.timing["split_train_start_utc"],
        "production_fit_end_utc": production_artifacts.timing["split_train_end_utc"],
        "production_fit_duration_sec": production_artifacts.timing["split_train_duration_sec"],
    }

    for role in ["best_cv_model", "last_cv_fold_model", "final_refit_model"]:
        role_df = final_holdout_comparison_df.loc[final_holdout_comparison_df["model_role"] == role]
        if role_df.empty:
            continue
        role_row = role_df.iloc[0].to_dict()
        for key, value in role_row.items():
            if key in {"run_id", "operator", "holdout_stage", "model_role"}:
                continue
            summary_row[f"{role}_{key}"] = value

    selected_operator_summary_df = pd.DataFrame([summary_row])
    operator_summary_path = operator_dir / f"{operator_name}_final_summary.csv"
    selected_operator_summary_df.to_csv(operator_summary_path, index=False)

    return {
        "operator_name": operator_name,
        "artifact_dir": operator_dir,
        "final_holdout_comparison_df": final_holdout_comparison_df,
        "selected_operator_summary_df": selected_operator_summary_df,
        "operator_summary_path": operator_summary_path,
        "production_artifacts": production_artifacts,
    }

# %% Final report builder

def find_single_file(root: Path, pattern: str) -> Optional[Path]:
    matches = sorted(root.rglob(pattern))
    return matches[0] if matches else None


def build_artifact_manifest(root: Path) -> List[Dict[str, Any]]:
    manifest: List[Dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        manifest.append(
            {
                "relative_path": safe_relpath(path, root),
                "size_bytes": path.stat().st_size,
            }
        )
    return manifest


def read_optional_yaml_or_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() in {".yaml", ".yml"}:
        return load_yaml(path)
    if path.suffix.lower() == ".json":
        return read_json(path)
    return {}


def seconds_to_rounded_minutes(value: Any) -> Optional[int]:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(numeric):
        return None
    return int(round(numeric / 60.0))


def format_minutes_label(value: Any) -> str:
    minutes = seconds_to_rounded_minutes(value)
    return f"{minutes} min" if minutes is not None else ""


def normalize_holdout_model_state(model_role: Any) -> str:
    mapping = {
        "best_cv_model": "best_cv",
        "last_cv_fold_model": "last_cv",
        "final_refit_model": "final_refit",
    }
    return mapping.get(str(model_role), str(model_role))


def reorder_columns(df: pd.DataFrame, preferred_columns: List[str]) -> pd.DataFrame:
    preferred = [column for column in preferred_columns if column in df.columns]
    remainder = [column for column in df.columns if column not in preferred]
    return df.loc[:, preferred + remainder]


def render_key_value_table(pairs: Iterable[Tuple[str, Any]]) -> str:
    rows = "".join(
        f"<tr><th style='text-align:left;padding:6px 10px;border:1px solid #ddd;background:#f7f7f7'>{html.escape(str(key))}</th>"
        f"<td style='padding:6px 10px;border:1px solid #ddd'>{html.escape(str(value)) if value not in (None, '') else ''}</td></tr>"
        for key, value in pairs
    )
    return f"<table>{rows}</table>" if rows else "<p>No data found.</p>"


def render_dataframe_html(df: pd.DataFrame, empty_message: str) -> str:
    if df.empty:
        return f"<p>{html.escape(empty_message)}</p>"
    printable = df.copy().where(pd.notnull(df), "")
    return printable.to_html(
        index=False,
        border=0,
        escape=True,
        float_format=lambda value: f"{value:.6f}",
    )


def build_final_report(artifact_root: Path, gcs_run_prefix: Optional[str] = None) -> Dict[str, Path]:
    """Build final_report.csv and final_report.html from saved summary files on disk."""
    resolved_config_path = find_single_file(artifact_root, "resolved_config.yaml") or find_single_file(
        artifact_root,
        "resolved_config.json",
    )
    env_meta_path = find_single_file(artifact_root, "environment_metadata.json")
    run_summary_path = find_single_file(artifact_root, "run_summary.json")
    gcs_summary_path = find_single_file(artifact_root, "gcs_upload_summary.json")

    env_meta = read_optional_yaml_or_json(env_meta_path)
    run_summary = read_optional_yaml_or_json(run_summary_path)
    resolved_config = read_optional_yaml_or_json(resolved_config_path)
    gcs_summary = read_optional_yaml_or_json(gcs_summary_path)
    manifest = build_artifact_manifest(artifact_root)

    run_id = str(run_summary.get("run_id") or env_meta.get("run_id") or RUN_ID)
    selected_operator = str(run_summary.get("selected_operator") or "")
    model_pipeline = str(resolved_config.get("model_pipeline") or CFG.get("model_pipeline") or "")
    gcs_prefix = gcs_run_prefix or run_summary.get("gcs_run_prefix") or gcs_summary.get("gcs_run_prefix") or ""
    generated_at_utc = utc_now_iso()

    holdout_paths = sorted(artifact_root.rglob("*_final_holdout_model_comparison_summary.csv"))
    operator_sections: List[Tuple[str, pd.DataFrame]] = []
    operator_names: List[str] = []
    final_report_frames: List[pd.DataFrame] = []
    operator_timing_rows: List[Dict[str, Any]] = []

    preferred_holdout_columns = [
        "model_state",
        "gross_pnl_sum",
        "pnl_sum",
        "pnl_per_trade",
        "n_trades",
        "trade_rate",
        "sign_accuracy",
        "win_rate",
        "sharpe_like",
        "dir_auc",
        "trade_auc",
        "rmse",
        "mae",
        "ic",
        "exit_type_accuracy",
        "selection_score",
        "scaled_soft_utility",
        "long_trades",
        "short_trades",
        "long_pnl_sum",
        "short_pnl_sum",
        "timeout_trade_count",
        "upper_exit_trade_count",
        "lower_exit_trade_count",
        "vertical_exit_trade_count",
        "selected_thr_trade",
        "selected_thr_dir",
        "selected_coverage",
        "fit_duration_min",
        "best_epoch_cumulative_train_duration_min",
        "model_name",
    ]
    model_state_order = {"best_cv": 0, "last_cv": 1, "final_refit": 2}

    for holdout_path in holdout_paths:
        operator_name = holdout_path.name.replace("_final_holdout_model_comparison_summary.csv", "")
        holdout_df = pd.read_csv(holdout_path)
        if holdout_df.empty:
            continue

        operator_names.append(operator_name)
        display_df = holdout_df.copy()
        display_df["model_state"] = display_df["model_role"].map(normalize_holdout_model_state)
        display_df["model_state_order"] = display_df["model_state"].map(model_state_order).fillna(999)

        if "fit_duration_sec" in display_df.columns:
            display_df["fit_duration_min"] = display_df["fit_duration_sec"].apply(seconds_to_rounded_minutes)
        if "best_epoch_cumulative_train_duration_sec" in display_df.columns:
            display_df["best_epoch_cumulative_train_duration_min"] = display_df[
                "best_epoch_cumulative_train_duration_sec"
            ].apply(seconds_to_rounded_minutes)

        display_df = (
            display_df.sort_values(["model_state_order", "model_state"])
            .drop(columns=["model_state_order"], errors="ignore")
            .drop(
                columns=[
                    "run_id",
                    "operator",
                    "holdout_stage",
                    "model_role",
                    "selected_for_production",
                    "holdout_is_diagnostic_only",
                    "fit_start_utc",
                    "fit_end_utc",
                    "fit_duration_sec",
                    "best_epoch_cumulative_train_duration_sec",
                ],
                errors="ignore",
            )
            .reset_index(drop=True)
        )
        display_df = reorder_columns(display_df, preferred_holdout_columns)
        operator_sections.append((operator_name, display_df))

        csv_df = display_df.copy()
        csv_df.insert(0, "artifact_root_name", artifact_root.name)
        csv_df.insert(1, "run_id", run_id)
        csv_df.insert(2, "model_pipeline", model_pipeline)
        csv_df.insert(3, "selected_operator_by_cv", selected_operator)
        csv_df.insert(4, "operator", operator_name)
        csv_df.insert(5, "deployment_reference", csv_df["model_state"].astype(str) == "last_cv")
        final_report_frames.append(csv_df)

        cv_mean_path = holdout_path.parent / f"{operator_name}_cv_mean_summary.csv"
        cv_mean_df = pd.read_csv(cv_mean_path) if cv_mean_path.exists() else pd.DataFrame()
        cv_total_minutes = None
        cv_mean_fold_minutes = None
        if not cv_mean_df.empty:
            cv_row = cv_mean_df.iloc[0].to_dict()
            cv_total_minutes = seconds_to_rounded_minutes(cv_row.get("cv_total_fold_train_duration_sec"))
            cv_mean_fold_minutes = seconds_to_rounded_minutes(cv_row.get("cv_mean_fold_train_duration_sec"))

        final_refit_row = holdout_df.loc[holdout_df["model_role"].astype(str) == "final_refit_model"]
        final_refit_fit_minutes = None
        final_refit_best_epoch_minutes = None
        if not final_refit_row.empty:
            final_refit_fit_minutes = seconds_to_rounded_minutes(final_refit_row.iloc[0].get("fit_duration_sec"))
            final_refit_best_epoch_minutes = seconds_to_rounded_minutes(
                final_refit_row.iloc[0].get("best_epoch_cumulative_train_duration_sec")
            )

        operator_timing_rows.append(
            {
                "operator": operator_name,
                "cv_mean_fold_train_min": cv_mean_fold_minutes,
                "cv_total_train_min": cv_total_minutes,
                "final_refit_fit_min": final_refit_fit_minutes,
                "final_refit_best_epoch_train_min": final_refit_best_epoch_minutes,
            }
        )

    final_report_df = pd.concat(final_report_frames, axis=0, ignore_index=True) if final_report_frames else pd.DataFrame()

    final_report_csv = artifact_root / "final_report.csv"
    final_report_html = artifact_root / "final_report.html"
    final_report_df.to_csv(final_report_csv, index=False)

    operators_text = ", ".join(operator_names) if operator_names else None
    summary_pairs = [
        ("Run ID", run_id),
        ("Artifact folder", artifact_root.name),
        ("Artifact root", str(artifact_root)),
        ("Model pipeline", model_pipeline or None),
        ("Operators with final_holdout diagnostics", operators_text),
        ("Final holdout policy", "diagnostic_only_no_reselection"),
        ("Primary deployable reference", "last_cv"),
        ("Upper-bound reference", "final_refit"),
        ("Threshold calibration", "validation_only"),
        ("GCS run prefix", gcs_prefix or None),
        ("Resolved config", str(resolved_config_path) if resolved_config_path else None),
        ("Environment metadata", str(env_meta_path) if env_meta_path else None),
        ("Run summary", str(run_summary_path) if run_summary_path else None),
    ]
    config_summary_text = html.escape(yaml.safe_dump(resolved_config, sort_keys=False, allow_unicode=True)) if resolved_config else ""
    artifact_manifest_df = pd.DataFrame(manifest)
    operator_timing_df = pd.DataFrame(operator_timing_rows)

    run_timing_pairs = [
        ("Run start UTC", env_meta.get("run_start_time_utc") or run_summary.get("run_start_time_utc")),
        ("Run end UTC", env_meta.get("run_end_time_utc") or run_summary.get("run_end_time_utc")),
        (
            "Total runtime",
            format_minutes_label(env_meta.get("total_runtime_seconds") or run_summary.get("total_runtime_seconds")),
        ),
    ]

    operator_sections_html = "".join(
        f"<h2>Final holdout - {html.escape(operator_name)}</h2>"
        f"{render_dataframe_html(display_df, f'No final_holdout metrics found for {operator_name}.')}"
        for operator_name, display_df in operator_sections
    )

    environment_pairs = [
        (key, value)
        for key, value in env_meta.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ]

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>{html.escape(f"[SUCCESS] {artifact_root.name}")}</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1, h2 {{ margin-bottom: 8px; }}
table {{ border-collapse: collapse; margin: 12px 0 24px 0; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 6px 10px; vertical-align: top; }}
th {{ background: #f7f7f7; text-align: left; }}
pre {{ background: #f7f7f7; padding: 12px; border: 1px solid #ddd; overflow-x: auto; }}
.small {{ color: #666; font-size: 12px; }}
</style>
</head>
<body>
<h1>{html.escape(f"[SUCCESS] {artifact_root.name}")}</h1>
<p class="small">Generated at {html.escape(generated_at_utc)}</p>

<h2>Summary</h2>
<p>Each operator is reported independently on the same final_holdout for three model states: <strong>best_cv</strong>, <strong>last_cv</strong>, and <strong>final_refit</strong>. Thresholds remain fixed from validation only. <strong>last_cv</strong> is the primary deployable reference, while <strong>final_refit</strong> is the larger-sample upper-bound benchmark.</p>
{render_key_value_table(summary_pairs)}

<h2>Timing</h2>
{render_key_value_table(run_timing_pairs)}
{render_dataframe_html(operator_timing_df, 'No operator timing summary found.')}

{operator_sections_html if operator_sections_html else '<h2>Final holdout</h2><p>No final_holdout metrics found.</p>'}

<h2>Config summary</h2>
<pre>{config_summary_text}</pre>

<h2>Artifact manifest</h2>
{render_dataframe_html(artifact_manifest_df, 'No artifacts found.')}

<h2>Environment metadata</h2>
{render_key_value_table(environment_pairs)}
</body>
</html>"""

    final_report_html.write_text(html_doc, encoding="utf-8")
    LOGGER.info("Built final report files: %s | %s", final_report_csv, final_report_html)
    return {"csv": final_report_csv, "html": final_report_html}

# %% Main entrypoint

def expected_required_files(cfg: Dict[str, Any]) -> List[str]:
    freq = normalize_freq_name(cfg["freq"])
    return [f"{asset}_{freq}.csv" for asset in cfg["assets"]]


def write_resolved_config(cfg: Dict[str, Any], artifact_root: Path) -> Path:
    resolved_path = artifact_root / "resolved_config.yaml"
    save_yaml(resolved_path, cfg)
    return resolved_path


def write_run_summary(artifact_root: Path, summary: Dict[str, Any]) -> Path:
    path = artifact_root / "run_summary.json"
    save_json(path, summary)
    return path



def build_success_email_attachments(
    artifact_root: Path,
    report_paths: Dict[str, Path],
    resolved_config_path: Optional[Path],
) -> List[Path]:
    candidates = [
        report_paths.get("csv"),
        report_paths.get("html"),
        resolved_config_path,
        artifact_root / "run_summary.json",
        artifact_root / "environment_metadata.json",
    ]
    candidates.extend(sorted(artifact_root.rglob("*_final_holdout_model_comparison_summary.csv")))
    candidates.extend(sorted(artifact_root.rglob("*_final_holdout_last_cv_trade_log.csv")))
    attachments: List[Path] = []
    seen: set = set()
    for candidate in candidates:
        if candidate is None:
            continue
        candidate = Path(candidate)
        if not candidate.exists():
            continue
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        attachments.append(candidate)
    return attachments

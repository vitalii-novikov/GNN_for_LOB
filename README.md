# Real-Time Market Microstructure Modeling with Temporal Graph Neural Networks

This repository contains the experimental code and final artifacts for the master's thesis **“Real-Time Market Microstructure Modeling with Temporal Graph Neural Networks”** by **Vitalii Novikov** (University of Applied Sciences Technikum Wien, submitted on **30.05.2026**). The submitted thesis PDF is stored at `paper_artifacts/MastersThesis_Novikov_Vitalii_TGNN (signed).pdf`.

The project is a controlled benchmark for short-horizon cryptocurrency limit order book (LOB) prediction. It compares graph-based neural architectures on ADA, BTC, and ETH LOB snapshots at **5-minute**, **1-minute**, and **1-second** resolution. ETH is the target asset; ADA and BTC provide cross-asset relational context.

The benchmark compares three model families — **base GNN**, **multigraph**, and **memorygraph** — with convolution-style and message-passing operators under a shared deployment-oriented evaluation pipeline: leakage-controlled walk-forward validation, triple-barrier targets, final chronological holdout, threshold selection, transaction-cost-aware event backtesting, and consistent artifact reporting.

> **Scope note:** this is a master's-thesis benchmark and reproducibility artifact. It is **not** financial advice, not a production trading system, and not a claim that the models can be deployed profitably without further data, execution, risk, and infrastructure work.

## Thesis summary

The thesis studies whether temporal graph neural networks can extract short-horizon market microstructure signals from a small cryptocurrency asset graph. All model families use the same asset universe, feature construction, relation-state construction, target logic, validation protocol, final holdout interval, thresholding rules, and event-based backtest. This keeps the comparison focused on how the architectures encode graph, relation, and temporal information.

The main deployment-oriented reference in the thesis is the last chronological cross-validation model (`last_cv` in the code/artifacts), while the final refit model is treated as an informative robustness comparison. Selected verified examples from `final_runs/*/final_report.csv` show that `base_gnn` with the adaptive convolution-style operator had the strongest post-cost last-CV result at both 5min and 1min resolution. At 1sec resolution, several runs produced positive gross signal, but transaction costs and high turnover consumed the edge.

## What this repository is / is not

This repository is the code-and-artifact companion to the submitted thesis. It is intended to make the benchmark design, experiment workflow, and final run outputs inspectable. It is not a packaged library, hosted service, trading bot, or complete public data release.

## Repository map

| Path | Purpose |
| --- | --- |
| `train.py` | Main orchestration entry point. It loads config, dispatches to the selected model pipeline, downloads/validates data, runs cross-validation/final-holdout evaluation, builds reports, and optionally uploads artifacts. |
| `train_config.yaml` | Default experiment template: data frequency, asset universe, feature/target settings, split parameters, model hyperparameters, operator candidates, and GCS artifact prefixes. Treat owner-specific credentials or private infrastructure values as local-only secrets. |
| `models/base_gnn_pipeline.py` | Single-graph baseline family, including static/prior/adaptive graph variants and Conv/MPNN-style operators. |
| `models/multigraph_pipeline.py` | Relation-preserving multigraph family with price-dependence, order-flow, and liquidity relation states. |
| `models/memorygraph_pipeline.py` | Stateful recurrent graph family with node/edge memory updates and chunked stateful training/evaluation. |
| `splits.py` | Chronological pre-holdout/final-holdout construction, walk-forward folds, purge-gap validation, final production split, and split summaries. |
| `gcp_utils.py` | Google Cloud Storage helpers for downloading required data files and uploading/refreshing training artifacts. |
| `Dockerfile` | Container runtime used for experiments. It installs `requirements.txt`, copies the repository, and starts `python train.py`. |
| `requirements.txt` | Python dependencies used by the training pipelines. |
| `final_runs/` | Final exported experiment artifacts used for thesis reporting and alignment checks. |
| `paper_artifacts/` | Submitted thesis PDF and supporting paper/table/figure artifacts. |

## Model families

- **Base GNN (`base_gnn`)**: a single-graph baseline that compares static, prior-informed, and adaptive adjacency variants with convolution-style and MPNN operators.
- **Multigraph (`multigraph`)**: a relation-preserving temporal GNN that keeps separate relation channels for price dependence, order flow, and liquidity before fusing them for ETH prediction.
- **Memorygraph (`memorygraph`)**: a stateful temporal graph model that carries node and edge memory through chunks to test whether recurrent graph state improves deployment-oriented performance.

All families share the same high-level supervised task: predict ETH short-horizon event outcomes from recent LOB-derived features and cross-asset relational context.

## Experiment workflow

The experiments were run with a containerized workflow:

1. Configure an experiment through `train_config.yaml` and/or CLI overrides.
2. Build a Docker image from this repository.
3. Push the image to an Artifact Registry repository.
4. Launch Vertex AI Custom Jobs on Google Cloud Platform.
5. Let `train.py` write local artifacts inside the container and upload run outputs to GCS when configured.
6. Preserve the final thesis-aligned outputs in `final_runs/`.

Raw LOB data is not assumed to be committed to this repository. The pipelines expect data either in the configured local `data_dir` or through a configured GCS data prefix.

## GCP / Vertex AI execution pattern

The thesis experiments were run as containerized Vertex AI Custom Jobs. The snippet below is a **sanitized historical reproduction pattern**; replace every placeholder with your own project, image, bucket, and service-account values before running it.

```bash
export MACHINE_TYPE="e2-standard-4"
export IMAGE_URI="<REGION>-docker.pkg.dev/<PROJECT_ID>/<ARTIFACT_REGISTRY_REPO>/<IMAGE_NAME>:<TAG>"
export PROJECT_ID="<PROJECT_ID>"
export REGION="<REGION>"
export SA_EMAIL="<SERVICE_ACCOUNT_EMAIL>"
export JOB_SUFFIX="<OPTIONAL_JOB_SUFFIX>"

run_exp () {
  local EXP_NAME="$1"
  local EXTRA_ARGS="$2"

  gcloud ai custom-jobs create \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --display-name="${EXP_NAME}${JOB_SUFFIX}" \
    --service-account="$SA_EMAIL" \
    --worker-pool-spec="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}" \
    --args="^|^--artifact-root=./art_${EXP_NAME}${JOB_SUFFIX}|${EXTRA_ARGS}"
}
```

Example extra arguments can select the model family, frequency, operator, and artifact/data prefixes:

```bash
run_exp "5min_base_conv" \
  "--config=train_config.yaml|--model-pipeline=base_gnn|--freq=5min|--graph-operator=adaptive_conv|--gcs-data-prefix=<GCS_DATA_PREFIX>|--gcs-run-prefix=<GCS_RUN_PREFIX>|--skip-email"
```

For public or shared use, keep cloud credentials, service-account identifiers, SMTP/e-mail settings, and bucket names outside tracked files. Use environment variables, a secret manager, or a private deployment wrapper instead of committing them to the repository.

## Local and Docker usage

A local run requires the expected raw data files to exist under the configured `data_dir`, unless you configure GCS access. A minimal local smoke run pattern is:

```bash
python train.py \
  --config train_config.yaml \
  --model-pipeline multigraph \
  --freq 1min \
  --artifact-root ./art_local \
  --local-run true \
  --skip-gcs-upload \
  --skip-email
```

Docker build/run pattern:

```bash
docker build -t gnn-for-lob:local .

docker run --rm \
  -v "$PWD/final_runs:/app/final_runs" \
  -v "<LOCAL_DATA_DIR>:/data:ro" \
  gnn-for-lob:local \
  --config train_config.yaml \
  --model-pipeline multigraph \
  --freq 1min \
  --artifact-root ./art_docker \
  --local-run true \
  --skip-gcs-upload \
  --skip-email
```

Adjust `train_config.yaml` or pass CLI overrides for model family, graph operator, data path, GCS prefixes, and artifact root.

## Final artifacts

`final_runs/` contains the final exported runs used for thesis reporting. The top-level run directories are:

- `5min-base-gnn/`
- `5min-memory-gnn/`
- `5min-multi-gnn/`
- `1min-base-gnn-conv/`
- `1min-base-gnn-mpnn/`
- `1min-memory-gnn/`
- `1min-multi-gnn-conv/`
- `1min-multi-gnn-mpnn/`
- `1sec-base-gnn-conv/`
- `1sec-base-gnn-mpnn/`
- `1sec-memory-gnn-conv/`
- `1sec-memory-gnn-mpnn/`
- `1sec-multi-gnn-conv/`
- `1sec-multi-gnn-mpnn/`

Some directories contain multiple operator subruns when an ablation or operator comparison was executed together. Common files include:

| Artifact | Meaning |
| --- | --- |
| `final_report.csv` / `final_report.html` | Consolidated final benchmark report for the run directory. |
| `run_summary.json` | Run status, selected operator, artifact paths, GCS upload metadata, and runtime summary. |
| `resolved_config.yaml` | Fully resolved configuration after config/env/CLI overrides. |
| `environment_metadata.json` | Runtime environment metadata and timestamps. |
| `train.log` | Training and evaluation log. |
| `splits/split_summary.json` | Chronological split and holdout summary. |
| `*_fold_*_best.pt` / `*_production_best.pt` | Saved PyTorch checkpoints. |
| `*_meta.json` | Metadata for saved model states. |
| `*_validation_threshold_grid.csv` | Threshold-search results used for deployment-style decision rules. |
| `*_final_holdout_*_trades.csv` and `*_trade_log.csv` | Final-holdout trade-level outputs and event-backtest diagnostics. |
| `*_final_holdout_model_comparison_summary.csv` | Comparison of last-CV, best-CV, and final-refit model states. |

## Key findings (selected verified examples)

The following are selected examples from the final deployment-reference (`last_cv`) reports and thesis abstract. They should be read as benchmark findings under the thesis assumptions, not as general trading-system claims.

| Frequency | Run / operator | Deployment-reference result |
| --- | --- | --- |
| 5min | `final_runs/5min-base-gnn/`, `base_gnn` + `adaptive_conv` | `net_pnl = 0.020356` over 26 trades. |
| 1min | `final_runs/1min-base-gnn-conv/`, `base_gnn` + `adaptive_conv` | `net_pnl = 0.020094` over 132 trades. |
| 1sec | Multiple 1sec runs | Several configurations produced positive `gross_pnl`, but all selected 1sec deployment-reference results remained negative after transaction costs. |

A central conclusion is that additional graph complexity did not automatically translate into more robust post-cost performance. In this benchmark, simpler base-graph models were the most reliable at 5min and 1min, while relation-preserving and memory-based models were still useful diagnostics for gross signal extraction, ranking quality, and turnover/cost sensitivity.

## Thesis and paper artifacts

- Submitted thesis PDF: `paper_artifacts/MastersThesis_Novikov_Vitalii_TGNN (signed).pdf`
- Final holdout alignment table: `paper_artifacts/final_holdout_alignment_table.csv` and `paper_artifacts/final_holdout_alignment_table.md`
- Figure-generation notebook: `paper_artifacts/thesis_figures.ipynb`

## Limitations and safety notes

- The submitted thesis had not yet been assessed at the time this README was written.
- Raw exchange/LOB data is external to the repository unless provided separately.
- Reported PnL values are cumulative log-return-style benchmark metrics under a simplified transaction-cost proxy; they are not brokerage statements or live-trading results.
- The 1sec setting is especially sensitive to turnover, latency, fees, spread, and market-impact assumptions.
- Before publishing or sharing the repository, review configuration files for owner-specific infrastructure values and move any credentials/secrets to environment variables or a secret manager.

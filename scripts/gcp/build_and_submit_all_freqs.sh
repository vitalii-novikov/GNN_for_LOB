#!/usr/bin/env bash
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-directed-pier-491014-e4}"
REGION="${REGION:-europe-west4}"
REPO_NAME="${REPO_NAME:-universal-piplines}"
IMAGE_NAME="${IMAGE_NAME:-univ1}"
IMAGE_TAG="${IMAGE_TAG:-$(date -u +%Y%m%d-%H%M%S)}"
IMAGE_URI="${IMAGE_URI:-${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}}"
JOB_PREFIX="${JOB_PREFIX:-lob}"
SA_EMAIL="${SA_EMAIL:-vertex-train-sa@${PROJECT_ID}.iam.gserviceaccount.com}"
MACHINE_TYPE="${MACHINE_TYPE:-e2-standard-4}"
CONFIG_PATH="${CONFIG_PATH:-train_config.yaml}"
DATA_DIR="${DATA_DIR:-./data}"
GCS_RUN_PREFIX="${GCS_RUN_PREFIX:-gs://hft_lob/runs/{run_id}}"
FREQUENCIES="${FREQUENCIES:-1sec 1min 5min}"
EXTRA_TRAIN_ARGS_CSV="${EXTRA_TRAIN_ARGS_CSV:-}"

if ! command -v gcloud >/dev/null 2>&1; then
    echo "gcloud CLI is required but was not found in PATH." >&2
    exit 1
fi

echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Repository: ${REPO_NAME}"
echo "Image URI: ${IMAGE_URI}"
echo "Frequencies: ${FREQUENCIES}"

gcloud config set project "${PROJECT_ID}" >/dev/null

if ! gcloud artifacts repositories describe "${REPO_NAME}" \
    --project="${PROJECT_ID}" \
    --location="${REGION}" >/dev/null 2>&1; then
    echo "Creating Artifact Registry repository ${REPO_NAME}..."
    gcloud artifacts repositories create "${REPO_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --repository-format=docker \
        --description="Vertex training images for GNN_for_LOB"
else
    echo "Artifact Registry repository ${REPO_NAME} already exists."
fi

echo "Building image ${IMAGE_URI} with Cloud Build..."
gcloud builds submit \
    --project="${PROJECT_ID}" \
    --tag "${IMAGE_URI}" \
    .

IFS=' ' read -r -a freq_array <<< "${FREQUENCIES}"
for freq in "${freq_array[@]}"; do
    freq_slug="${freq//[^a-zA-Z0-9]/-}"
    job_name="${JOB_PREFIX}-${freq_slug}-${IMAGE_TAG}"
    args_csv="--config,${CONFIG_PATH},--freq,${freq},--data-dir,${DATA_DIR},--gcs-run-prefix,${GCS_RUN_PREFIX}"

    if [[ -n "${EXTRA_TRAIN_ARGS_CSV}" ]]; then
        args_csv="${args_csv},${EXTRA_TRAIN_ARGS_CSV}"
    fi

    echo "Submitting Vertex AI custom job ${job_name}..."
    gcloud ai custom-jobs create \
        --project="${PROJECT_ID}" \
        --region="${REGION}" \
        --display-name="${job_name}" \
        --service-account="${SA_EMAIL}" \
        --worker-pool-spec="machine-type=${MACHINE_TYPE},replica-count=1,container-image-uri=${IMAGE_URI}" \
        --args="${args_csv}"
done

echo "Done."
echo "Built image: ${IMAGE_URI}"

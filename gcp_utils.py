import logging
from pathlib import Path
from typing import Iterable, List, Tuple

LOGGER = logging.getLogger("train")


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


def _get_storage_client():
    try:
        from google.cloud import storage
    except ImportError as exc:
        raise RuntimeError(
            "google-cloud-storage is not installed. "
            "Use local mode or install the GCP dependencies before enabling GCP access."
        ) from exc
    return storage.Client()


def download_from_gcs(required_files: List[str], gcs_data_prefix: str, local_data_dir: Path) -> List[Path]:
    local_data_dir.mkdir(parents=True, exist_ok=True)
    local_paths = [local_data_dir / filename for filename in required_files]
    missing = [p for p in local_paths if not p.exists()]
    if not missing:
        LOGGER.info("All required data files already exist locally in %s", local_data_dir)
        return local_paths

    if not gcs_data_prefix:
        missing_names = ", ".join(p.name for p in missing)
        raise FileNotFoundError(
            f"Missing local data files and GCS_DATA_PREFIX is not configured. Missing: {missing_names}"
        )

    bucket_name, blob_prefix = parse_gs_uri(gcs_data_prefix)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    for local_path in missing:
        object_name = f"{blob_prefix.rstrip('/')}/{local_path.name}" if blob_prefix else local_path.name
        blob = bucket.blob(object_name)
        LOGGER.info("Downloading %s to %s", artifact_uri_join(f"gs://{bucket_name}", object_name), local_path)
        if not blob.exists(client):
            raise FileNotFoundError(f"GCS object not found: gs://{bucket_name}/{object_name}")
        blob.download_to_filename(str(local_path))
        if not local_path.exists():
            raise FileNotFoundError(f"Download completed without local file present: {local_path}")

    return local_paths


def upload_artifacts_to_gcs(local_artifact_root: Path, gcs_run_prefix: str) -> List[str]:
    if not gcs_run_prefix:
        raise ValueError("gcs_run_prefix is empty")

    bucket_name, blob_prefix = parse_gs_uri(gcs_run_prefix)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    uploaded: List[str] = []
    for path in sorted(local_artifact_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(local_artifact_root).as_posix()
        object_name = f"{blob_prefix.rstrip('/')}/{rel}" if blob_prefix else rel
        blob = bucket.blob(object_name)
        LOGGER.info("Uploading %s to gs://%s/%s", path, bucket_name, object_name)
        blob.upload_from_filename(str(path))
        uploaded.append(f"gs://{bucket_name}/{object_name}")

    return uploaded


def refresh_gcs_artifacts(
    local_artifact_root: Path,
    gcs_run_prefix: str,
    paths: Iterable[Path],
) -> List[str]:
    if not gcs_run_prefix:
        raise ValueError("gcs_run_prefix is empty")

    bucket_name, blob_prefix = parse_gs_uri(gcs_run_prefix)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)

    uploaded: List[str] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        rel = path.relative_to(local_artifact_root).as_posix()
        object_name = f"{blob_prefix.rstrip('/')}/{rel}" if blob_prefix else rel
        LOGGER.info("Refreshing %s on gs://%s/%s", path, bucket_name, object_name)
        bucket.blob(object_name).upload_from_filename(str(path))
        uploaded.append(f"gs://{bucket_name}/{object_name}")
    return uploaded

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("train")


@dataclass
class SplitBundle:
    idx_preholdout: np.ndarray
    idx_holdout: np.ndarray
    walk_forward_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
    idx_train_final: np.ndarray
    idx_val_final: np.ndarray
    idx_test_final: np.ndarray
    summary: Dict[str, Any]
    split_file: Optional[Path] = None
    summary_file: Optional[Path] = None


def make_preholdout_and_holdout_split(
    n_samples: int,
    holdout_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    holdout_n = max(1, int(round(n_samples * float(holdout_frac))))
    preholdout_n = n_samples - gap_bars - holdout_n
    if preholdout_n <= 0:
        raise RuntimeError("Not enough samples left after reserving purge gap and final holdout.")

    idx_preholdout = np.arange(0, preholdout_n, dtype=np.int64)
    idx_holdout = np.arange(preholdout_n + gap_bars, preholdout_n + gap_bars + holdout_n, dtype=np.int64)

    if len(idx_holdout) != holdout_n:
        raise RuntimeError("Holdout indices were not constructed correctly.")
    if idx_holdout[-1] >= n_samples:
        raise RuntimeError("Holdout indices exceed available sample count.")
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise RuntimeError("Pre-holdout and holdout indices overlap.")

    return idx_preholdout, idx_holdout


def assert_sorted_unique_indices(indices: np.ndarray, name: str) -> None:
    if len(indices) == 0:
        raise AssertionError(f"{name} is empty")
    if not np.all(indices[:-1] < indices[1:]):
        raise AssertionError(f"{name} must be strictly increasing and unique")


def assert_time_order_and_purge(
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,
    gap_bars: int,
    label: str,
) -> None:
    assert_sorted_unique_indices(idx_train, f"{label}.train")
    assert_sorted_unique_indices(idx_val, f"{label}.val")
    assert_sorted_unique_indices(idx_test, f"{label}.test")

    if len(np.intersect1d(idx_train, idx_val)) > 0:
        raise AssertionError(f"{label}: train and val overlap")
    if len(np.intersect1d(idx_train, idx_test)) > 0:
        raise AssertionError(f"{label}: train and test overlap")
    if len(np.intersect1d(idx_val, idx_test)) > 0:
        raise AssertionError(f"{label}: val and test overlap")

    train_last = int(idx_train[-1])
    val_first = int(idx_val[0])
    val_last = int(idx_val[-1])
    test_first = int(idx_test[0])

    if not train_last < val_first < test_first:
        raise AssertionError(f"{label}: split windows are not strictly time ordered")

    if (val_first - train_last) <= gap_bars:
        raise AssertionError(f"{label}: purge gap between train and val is not respected")
    if (test_first - val_last) <= gap_bars:
        raise AssertionError(f"{label}: purge gap between val and test is not respected")


def make_exact_walk_forward_splits(
    idx_preholdout: np.ndarray,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    gap_bars: int,
    num_folds: int,
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n_pre = len(idx_preholdout)
    num_folds = int(num_folds)
    train_min = max(1, int(round(n_pre * float(train_min_frac))))
    val_n = max(1, int(round(n_pre * float(val_window_frac))))
    test_n = max(1, int(round(n_pre * float(test_window_frac))))

    max_train_end = n_pre - (2 * gap_bars) - val_n - test_n
    if max_train_end <= train_min:
        raise RuntimeError(
            "Not enough pre-holdout data to create the requested number of exact walk-forward folds."
        )

    train_ends = np.linspace(train_min, max_train_end, num=num_folds)
    train_ends = np.round(train_ends).astype(np.int64)

    if len(np.unique(train_ends)) != num_folds:
        raise RuntimeError(
            "Exact fold construction produced duplicate train end points. "
            "Reduce num_train_folds or adjust split fractions."
        )

    splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for fold_idx, train_end in enumerate(train_ends, start=1):
        val_start = int(train_end) + gap_bars
        val_end = val_start + val_n
        test_start = val_end + gap_bars
        test_end = test_start + test_n

        if test_end > n_pre:
            raise RuntimeError(f"Fold {fold_idx} exceeds the pre-holdout boundary.")

        idx_train = idx_preholdout[: int(train_end)].copy()
        idx_val = idx_preholdout[val_start:val_end].copy()
        idx_test = idx_preholdout[test_start:test_end].copy()

        if len(idx_train) == 0 or len(idx_val) == 0 or len(idx_test) == 0:
            raise RuntimeError(f"Fold {fold_idx} produced an empty split.")

        splits.append((idx_train, idx_val, idx_test))

    if len(splits) != num_folds:
        raise RuntimeError(f"Expected exactly {num_folds} walk-forward folds, got {len(splits)}.")
    return splits


def make_final_production_split(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    val_window_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_pre = len(idx_preholdout)
    val_n = max(1, int(round(n_pre * float(val_window_frac))))
    train_end = n_pre - gap_bars - val_n

    if train_end <= 0:
        raise RuntimeError("Not enough pre-holdout samples for final production split.")

    idx_train_final = idx_preholdout[:train_end].copy()
    idx_val_final = idx_preholdout[train_end + gap_bars: train_end + gap_bars + val_n].copy()
    idx_test_final = idx_holdout.copy()

    if len(idx_val_final) != val_n:
        raise RuntimeError("Final validation window was not created correctly.")
    return idx_train_final, idx_val_final, idx_test_final


def validate_all_splits(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    idx_train_final: np.ndarray,
    idx_val_final: np.ndarray,
    idx_test_final: np.ndarray,
    gap_bars: int,
    num_expected_folds: int,
    n_samples: Optional[int] = None,
) -> None:
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise AssertionError("Pre-holdout and holdout overlap")

    if len(cv_splits) != int(num_expected_folds):
        raise AssertionError(
            f"Expected exactly {int(num_expected_folds)} CV folds, got {len(cv_splits)}"
        )

    if n_samples is not None:
        for name, arr in {
            "idx_preholdout": idx_preholdout,
            "idx_holdout": idx_holdout,
            "idx_train_final": idx_train_final,
            "idx_val_final": idx_val_final,
            "idx_test_final": idx_test_final,
        }.items():
            if len(arr) == 0:
                raise AssertionError(f"{name} is empty")
            if int(arr.min()) < 0 or int(arr.max()) >= int(n_samples):
                raise AssertionError(f"{name} contains indices outside [0, {n_samples})")
        for fold_idx, (idx_train, idx_val, idx_test) in enumerate(cv_splits, start=1):
            for split_name, arr in {
                "train": idx_train,
                "val": idx_val,
                "test": idx_test,
            }.items():
                if int(arr.min()) < 0 or int(arr.max()) >= int(n_samples):
                    raise AssertionError(f"cv_fold_{fold_idx}.{split_name} contains out-of-range indices")

    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(cv_splits, start=1):
        assert_time_order_and_purge(idx_train, idx_val, idx_test, gap_bars, label=f"cv_fold_{fold_idx}")

        if not np.all(np.isin(idx_train, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: train contains non-preholdout indices")
        if not np.all(np.isin(idx_val, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: val contains non-preholdout indices")
        if not np.all(np.isin(idx_test, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: test contains non-preholdout indices")
        if len(np.intersect1d(idx_test, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV test")
        if len(np.intersect1d(idx_train, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV train")
        if len(np.intersect1d(idx_val, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV val")

    assert_time_order_and_purge(
        idx_train_final,
        idx_val_final,
        idx_test_final,
        gap_bars,
        label="final_production_split",
    )
    if not np.all(np.isin(idx_train_final, idx_preholdout)):
        raise AssertionError("Final train contains non-preholdout indices")
    if not np.all(np.isin(idx_val_final, idx_preholdout)):
        raise AssertionError("Final val contains non-preholdout indices")
    if not np.array_equal(idx_test_final, idx_holdout):
        raise AssertionError("Final production test must equal the final holdout exactly")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, float):
        return obj if np.isfinite(obj) else None
    return obj


def _safe_rate(numerator: np.ndarray, denominator_mask: np.ndarray) -> Optional[float]:
    if not denominator_mask.any():
        return None
    return float(np.mean(numerator[denominator_mask]))


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if not mask.any():
        return None
    return float(np.mean(values[mask]))


def _split_time_bounds(
    indices: np.ndarray,
    sample_t: np.ndarray,
    timestamps: pd.Series,
) -> Dict[str, Optional[str]]:
    if len(indices) == 0:
        return {"start_timestamp_utc": None, "end_timestamp_utc": None}
    raw_t = sample_t[indices]
    start_ts = pd.Timestamp(timestamps.iloc[int(raw_t[0])]).isoformat()
    end_ts = pd.Timestamp(timestamps.iloc[int(raw_t[-1])]).isoformat()
    return {"start_timestamp_utc": start_ts, "end_timestamp_utc": end_ts}


def summarize_split(
    split_name: str,
    indices: np.ndarray,
    sample_t: np.ndarray,
    timestamps: pd.Series,
    y_trade: np.ndarray,
    y_dir: np.ndarray,
    y_dir_mask: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
) -> Dict[str, Any]:
    raw_t = sample_t[indices]
    valid_mask = np.isfinite(y_tte[raw_t])
    trade_mask = valid_mask & (y_trade[raw_t] > 0.5)
    dir_mask = valid_mask & (y_dir_mask[raw_t] > 0.5)
    upper_mask = valid_mask & (y_exit_type[raw_t] == 1)
    lower_mask = valid_mask & (y_exit_type[raw_t] == 2)
    vertical_mask = valid_mask & (y_exit_type[raw_t] == 0)

    summary = {
        "split_name": split_name,
        "n_samples": int(len(indices)),
        "sample_index_start": int(indices[0]) if len(indices) else None,
        "sample_index_end": int(indices[-1]) if len(indices) else None,
        "raw_t_start": int(raw_t[0]) if len(raw_t) else None,
        "raw_t_end": int(raw_t[-1]) if len(raw_t) else None,
        "n_valid_targets": int(valid_mask.sum()),
        "n_trade": int(trade_mask.sum()),
        "n_direction_labeled": int(dir_mask.sum()),
        "n_up": int(upper_mask.sum()),
        "n_flat": int(vertical_mask.sum()),
        "n_down": int(lower_mask.sum()),
        "n_upper_exit": int(upper_mask.sum()),
        "n_vertical_exit": int(vertical_mask.sum()),
        "n_lower_exit": int(lower_mask.sum()),
        "trade_rate": _safe_rate(y_trade[raw_t], valid_mask),
        "share_up": _safe_mean(upper_mask.astype(np.float32), valid_mask),
        "share_flat": _safe_mean(vertical_mask.astype(np.float32), valid_mask),
        "share_down": _safe_mean(lower_mask.astype(np.float32), valid_mask),
        "avg_tte_bars": _safe_mean(y_tte[raw_t], valid_mask),
        "time_bounds": _split_time_bounds(indices=indices, sample_t=sample_t, timestamps=timestamps),
    }
    return summary


def build_split_summary(
    cfg: Dict[str, Any],
    sample_t: np.ndarray,
    timestamps: pd.Series,
    y_trade: np.ndarray,
    y_dir: np.ndarray,
    y_dir_mask: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
    target_summary: Dict[str, Any],
    purge_gap_bars: int,
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    walk_forward_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    idx_train_final: np.ndarray,
    idx_val_final: np.ndarray,
    idx_test_final: np.ndarray,
    source_split_file: Optional[Path] = None,
) -> Dict[str, Any]:
    cv_summary: List[Dict[str, Any]] = []
    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(walk_forward_splits, start=1):
        cv_summary.append(
            {
                "fold_idx": fold_idx,
                "train": summarize_split(
                    split_name=f"cv_fold_{fold_idx:02d}.train",
                    indices=idx_train,
                    sample_t=sample_t,
                    timestamps=timestamps,
                    y_trade=y_trade,
                    y_dir=y_dir,
                    y_dir_mask=y_dir_mask,
                    y_exit_type=y_exit_type,
                    y_tte=y_tte,
                ),
                "val": summarize_split(
                    split_name=f"cv_fold_{fold_idx:02d}.val",
                    indices=idx_val,
                    sample_t=sample_t,
                    timestamps=timestamps,
                    y_trade=y_trade,
                    y_dir=y_dir,
                    y_dir_mask=y_dir_mask,
                    y_exit_type=y_exit_type,
                    y_tte=y_tte,
                ),
                "test": summarize_split(
                    split_name=f"cv_fold_{fold_idx:02d}.test",
                    indices=idx_test,
                    sample_t=sample_t,
                    timestamps=timestamps,
                    y_trade=y_trade,
                    y_dir=y_dir,
                    y_dir_mask=y_dir_mask,
                    y_exit_type=y_exit_type,
                    y_tte=y_tte,
                ),
            }
        )

    return {
        "source_split_file": str(source_split_file) if source_split_file else None,
        "label_mode": str(cfg.get("label_mode")),
        "freq": str(cfg.get("freq")),
        "target_asset": str(cfg.get("target_asset")),
        "n_samples": int(len(sample_t)),
        "purge_gap_bars": int(purge_gap_bars),
        "num_train_folds": int(cfg["num_train_folds"]),
        "target_summary": _jsonable(target_summary),
        "preholdout": summarize_split(
            split_name="preholdout",
            indices=idx_preholdout,
            sample_t=sample_t,
            timestamps=timestamps,
            y_trade=y_trade,
            y_dir=y_dir,
            y_dir_mask=y_dir_mask,
            y_exit_type=y_exit_type,
            y_tte=y_tte,
        ),
        "holdout": summarize_split(
            split_name="holdout",
            indices=idx_holdout,
            sample_t=sample_t,
            timestamps=timestamps,
            y_trade=y_trade,
            y_dir=y_dir,
            y_dir_mask=y_dir_mask,
            y_exit_type=y_exit_type,
            y_tte=y_tte,
        ),
        "cv_folds": cv_summary,
        "final_production_split": {
            "train": summarize_split(
                split_name="final_production.train",
                indices=idx_train_final,
                sample_t=sample_t,
                timestamps=timestamps,
                y_trade=y_trade,
                y_dir=y_dir,
                y_dir_mask=y_dir_mask,
                y_exit_type=y_exit_type,
                y_tte=y_tte,
            ),
            "val": summarize_split(
                split_name="final_production.val",
                indices=idx_val_final,
                sample_t=sample_t,
                timestamps=timestamps,
                y_trade=y_trade,
                y_dir=y_dir,
                y_dir_mask=y_dir_mask,
                y_exit_type=y_exit_type,
                y_tte=y_tte,
            ),
            "test": summarize_split(
                split_name="final_production.test",
                indices=idx_test_final,
                sample_t=sample_t,
                timestamps=timestamps,
                y_trade=y_trade,
                y_dir=y_dir,
                y_dir_mask=y_dir_mask,
                y_exit_type=y_exit_type,
                y_tte=y_tte,
            ),
        },
    }


def save_split_bundle(bundle: SplitBundle, output_dir: Path) -> SplitBundle:
    output_dir.mkdir(parents=True, exist_ok=True)
    split_file = output_dir / "split_indices.npz"
    summary_file = output_dir / "split_summary.json"

    arrays: Dict[str, np.ndarray] = {
        "idx_preholdout": bundle.idx_preholdout.astype(np.int64),
        "idx_holdout": bundle.idx_holdout.astype(np.int64),
        "idx_train_final": bundle.idx_train_final.astype(np.int64),
        "idx_val_final": bundle.idx_val_final.astype(np.int64),
        "idx_test_final": bundle.idx_test_final.astype(np.int64),
    }
    for fold_idx, (idx_train, idx_val, idx_test) in enumerate(bundle.walk_forward_splits, start=1):
        arrays[f"cv_fold_{fold_idx:02d}_train"] = idx_train.astype(np.int64)
        arrays[f"cv_fold_{fold_idx:02d}_val"] = idx_val.astype(np.int64)
        arrays[f"cv_fold_{fold_idx:02d}_test"] = idx_test.astype(np.int64)

    np.savez_compressed(split_file, **arrays)
    summary_file.write_text(
        json.dumps(_jsonable(bundle.summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    bundle.split_file = split_file
    bundle.summary_file = summary_file
    return bundle


def load_split_bundle(
    split_file: Path,
    summary_file: Optional[Path] = None,
) -> SplitBundle:
    split_file = Path(split_file).expanduser().resolve()
    if not split_file.exists():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    if summary_file is None:
        candidate = split_file.with_name("split_summary.json")
        summary_file = candidate if candidate.exists() else None

    with np.load(split_file, allow_pickle=False) as arrays:
        idx_preholdout = arrays["idx_preholdout"].astype(np.int64)
        idx_holdout = arrays["idx_holdout"].astype(np.int64)
        idx_train_final = arrays["idx_train_final"].astype(np.int64)
        idx_val_final = arrays["idx_val_final"].astype(np.int64)
        idx_test_final = arrays["idx_test_final"].astype(np.int64)

        walk_forward_splits: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        fold_idx = 1
        while True:
            prefix = f"cv_fold_{fold_idx:02d}"
            train_key = f"{prefix}_train"
            val_key = f"{prefix}_val"
            test_key = f"{prefix}_test"
            if train_key not in arrays.files:
                break
            walk_forward_splits.append(
                (
                    arrays[train_key].astype(np.int64),
                    arrays[val_key].astype(np.int64),
                    arrays[test_key].astype(np.int64),
                )
            )
            fold_idx += 1

    summary: Dict[str, Any] = {}
    if summary_file is not None and Path(summary_file).exists():
        summary = json.loads(Path(summary_file).read_text(encoding="utf-8"))

    return SplitBundle(
        idx_preholdout=idx_preholdout,
        idx_holdout=idx_holdout,
        walk_forward_splits=walk_forward_splits,
        idx_train_final=idx_train_final,
        idx_val_final=idx_val_final,
        idx_test_final=idx_test_final,
        summary=summary,
        split_file=split_file,
        summary_file=Path(summary_file) if summary_file is not None else None,
    )


def log_split_summary(bundle: SplitBundle, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or LOGGER

    logger.info(
        "Active split bundle: split_file=%s | summary_file=%s",
        bundle.split_file,
        bundle.summary_file,
    )
    logger.info(
        "Split overview: preholdout=%s | holdout=%s | cv_folds=%s | final_train=%s | final_val=%s | final_test=%s | purge_gap_bars=%s",
        len(bundle.idx_preholdout),
        len(bundle.idx_holdout),
        len(bundle.walk_forward_splits),
        len(bundle.idx_train_final),
        len(bundle.idx_val_final),
        len(bundle.idx_test_final),
        bundle.summary.get("purge_gap_bars"),
    )

    def _log_one(prefix: str, split_summary: Dict[str, Any]) -> None:
        logger.info(
            "%s | n=%s valid=%s trade_rate=%s up/flat/down=(%s/%s/%s) start=%s end=%s",
            prefix,
            split_summary.get("n_samples"),
            split_summary.get("n_valid_targets"),
            split_summary.get("trade_rate"),
            split_summary.get("share_up"),
            split_summary.get("share_flat"),
            split_summary.get("share_down"),
            split_summary.get("time_bounds", {}).get("start_timestamp_utc"),
            split_summary.get("time_bounds", {}).get("end_timestamp_utc"),
        )

    _log_one("preholdout", bundle.summary.get("preholdout", {}))
    _log_one("holdout", bundle.summary.get("holdout", {}))

    for fold_summary in bundle.summary.get("cv_folds", []):
        fold_idx = int(fold_summary.get("fold_idx", 0))
        _log_one(f"cv_fold_{fold_idx:02d}.train", fold_summary.get("train", {}))
        _log_one(f"cv_fold_{fold_idx:02d}.val", fold_summary.get("val", {}))
        _log_one(f"cv_fold_{fold_idx:02d}.test", fold_summary.get("test", {}))

    final_summary = bundle.summary.get("final_production_split", {})
    _log_one("final_production.train", final_summary.get("train", {}))
    _log_one("final_production.val", final_summary.get("val", {}))
    _log_one("final_production.test", final_summary.get("test", {}))


def build_or_load_split_bundle(
    cfg: Dict[str, Any],
    n_samples: int,
    gap_bars: int,
    sample_t: np.ndarray,
    timestamps: pd.Series,
    y_trade: np.ndarray,
    y_dir: np.ndarray,
    y_dir_mask: np.ndarray,
    y_exit_type: np.ndarray,
    y_tte: np.ndarray,
    target_summary: Dict[str, Any],
    artifact_root: Path,
    logger: Optional[logging.Logger] = None,
) -> SplitBundle:
    logger = logger or LOGGER
    split_file_cfg = str(cfg.get("split_file") or "").strip()
    source_split_file: Optional[Path] = None

    if split_file_cfg:
        source_split_file = Path(split_file_cfg).expanduser().resolve()
        bundle = load_split_bundle(source_split_file)
        logger.info("Loaded precomputed splits from %s", source_split_file)
    else:
        idx_preholdout, idx_holdout = make_preholdout_and_holdout_split(
            n_samples=n_samples,
            holdout_frac=float(cfg["final_holdout_frac"]),
            gap_bars=gap_bars,
        )
        walk_forward_splits = make_exact_walk_forward_splits(
            idx_preholdout=idx_preholdout,
            train_min_frac=float(cfg["train_min_frac"]),
            val_window_frac=float(cfg["val_window_frac"]),
            test_window_frac=float(cfg["test_window_frac"]),
            gap_bars=gap_bars,
            num_folds=int(cfg["num_train_folds"]),
        )
        idx_train_final, idx_val_final, idx_test_final = make_final_production_split(
            idx_preholdout=idx_preholdout,
            idx_holdout=idx_holdout,
            val_window_frac=float(cfg["val_window_frac"]),
            gap_bars=gap_bars,
        )
        bundle = SplitBundle(
            idx_preholdout=idx_preholdout,
            idx_holdout=idx_holdout,
            walk_forward_splits=walk_forward_splits,
            idx_train_final=idx_train_final,
            idx_val_final=idx_val_final,
            idx_test_final=idx_test_final,
            summary={},
        )

    validate_all_splits(
        idx_preholdout=bundle.idx_preholdout,
        idx_holdout=bundle.idx_holdout,
        cv_splits=bundle.walk_forward_splits,
        idx_train_final=bundle.idx_train_final,
        idx_val_final=bundle.idx_val_final,
        idx_test_final=bundle.idx_test_final,
        gap_bars=gap_bars,
        num_expected_folds=int(cfg["num_train_folds"]),
        n_samples=n_samples,
    )

    bundle.summary = build_split_summary(
        cfg=cfg,
        sample_t=sample_t,
        timestamps=timestamps,
        y_trade=y_trade,
        y_dir=y_dir,
        y_dir_mask=y_dir_mask,
        y_exit_type=y_exit_type,
        y_tte=y_tte,
        target_summary=target_summary,
        purge_gap_bars=gap_bars,
        idx_preholdout=bundle.idx_preholdout,
        idx_holdout=bundle.idx_holdout,
        walk_forward_splits=bundle.walk_forward_splits,
        idx_train_final=bundle.idx_train_final,
        idx_val_final=bundle.idx_val_final,
        idx_test_final=bundle.idx_test_final,
        source_split_file=source_split_file,
    )

    save_split_bundle(bundle, artifact_root / "splits")
    log_split_summary(bundle, logger=logger)
    return bundle

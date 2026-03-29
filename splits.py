import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SplitIndices:
    train: np.ndarray
    gap_before_val: np.ndarray
    val: np.ndarray
    gap_before_test: np.ndarray
    test: np.ndarray


@dataclass
class SplitPlan:
    freq: str
    lookback_bars: int
    horizon_minutes: int
    horizon_bars: int
    gap_bars: int
    n_samples: int
    first_valid_t: int
    last_valid_t: int
    sample_index_offset: int
    preholdout_indices: np.ndarray
    holdout_gap_indices: np.ndarray
    holdout_indices: np.ndarray
    cv_folds: List[SplitIndices]
    production_split: SplitIndices
    params: Dict[str, Any]
    metadata: Dict[str, Any]


def _to_numpy_int64(values: Sequence[int]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.int64)


def _make_block(start: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.empty((0,), dtype=np.int64)
    return np.arange(int(start), int(start) + int(length), dtype=np.int64)


def _require_non_empty(indices: np.ndarray, name: str) -> None:
    if len(indices) == 0:
        raise AssertionError(f"{name} is empty")


def _ensure_contiguous(indices: np.ndarray, name: str) -> None:
    if len(indices) <= 1:
        return
    if not np.all(np.diff(indices) == 1):
        raise AssertionError(f"{name} must be contiguous")


def _window_from_total_with_gap(window_name: str, total_n: int, gap_bars: int) -> int:
    usable_n = int(total_n) - int(gap_bars)
    if usable_n <= 0:
        raise RuntimeError(
            f"{window_name} window is too small for gap_bars={gap_bars}. "
            f"Increase the corresponding *_window_frac or reduce the gap."
        )
    return usable_n


def make_preholdout_and_holdout_split(
    n_samples: int,
    holdout_frac: float,
    gap_bars: int,
) -> Tuple[np.ndarray, np.ndarray]:
    holdout_n = max(1, int(round(n_samples * float(holdout_frac))))
    preholdout_n = n_samples - int(gap_bars) - holdout_n
    if preholdout_n <= 0:
        raise RuntimeError("Not enough samples left after reserving purge gap and final holdout.")

    idx_preholdout = _make_block(0, preholdout_n)
    idx_holdout = _make_block(preholdout_n + int(gap_bars), holdout_n)

    if len(idx_holdout) != holdout_n:
        raise RuntimeError("Holdout indices were not constructed correctly.")
    if idx_holdout[-1] >= n_samples:
        raise RuntimeError("Holdout indices exceed available sample count.")
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise RuntimeError("Pre-holdout and holdout indices overlap.")

    return idx_preholdout, idx_holdout


def assert_sorted_unique_indices(indices: np.ndarray, name: str) -> None:
    _require_non_empty(indices, name)
    if not np.all(indices[:-1] < indices[1:]):
        raise AssertionError(f"{name} must be strictly increasing and unique")


def assert_time_order_and_purge(
    split: SplitIndices,
    gap_bars: int,
    label: str,
) -> None:
    assert_sorted_unique_indices(split.train, f"{label}.train")
    assert_sorted_unique_indices(split.gap_before_val, f"{label}.gap_before_val")
    assert_sorted_unique_indices(split.val, f"{label}.val")
    assert_sorted_unique_indices(split.gap_before_test, f"{label}.gap_before_test")
    assert_sorted_unique_indices(split.test, f"{label}.test")

    if len(split.gap_before_val) != int(gap_bars):
        raise AssertionError(f"{label}: gap_before_val length must equal gap_bars={gap_bars}")
    if len(split.gap_before_test) != int(gap_bars):
        raise AssertionError(f"{label}: gap_before_test length must equal gap_bars={gap_bars}")

    for name, indices in [
        ("train", split.train),
        ("gap_before_val", split.gap_before_val),
        ("val", split.val),
        ("gap_before_test", split.gap_before_test),
        ("test", split.test),
    ]:
        _ensure_contiguous(indices, f"{label}.{name}")

    windows = [
        ("train", split.train),
        ("gap_before_val", split.gap_before_val),
        ("val", split.val),
        ("gap_before_test", split.gap_before_test),
        ("test", split.test),
    ]
    for left_pos, (left_name, left_idx) in enumerate(windows):
        for right_name, right_idx in windows[left_pos + 1:]:
            if len(np.intersect1d(left_idx, right_idx)) > 0:
                raise AssertionError(f"{label}: {left_name} and {right_name} overlap")

    if split.train[-1] + 1 != split.gap_before_val[0]:
        raise AssertionError(f"{label}: gap_before_val does not start immediately after train")
    if split.gap_before_val[-1] + 1 != split.val[0]:
        raise AssertionError(f"{label}: val does not start immediately after gap_before_val")
    if split.val[-1] + 1 != split.gap_before_test[0]:
        raise AssertionError(f"{label}: gap_before_test does not start immediately after val")
    if split.gap_before_test[-1] + 1 != split.test[0]:
        raise AssertionError(f"{label}: test does not start immediately after gap_before_test")


def make_exact_walk_forward_splits(
    idx_preholdout: np.ndarray,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    gap_bars: int,
    num_folds: int,
) -> List[SplitIndices]:
    n_pre = len(idx_preholdout)
    num_folds = int(num_folds)
    train_min = max(1, int(round(n_pre * float(train_min_frac))))
    val_total_n = max(int(gap_bars) + 1, int(round(n_pre * float(val_window_frac))))
    test_total_n = max(int(gap_bars) + 1, int(round(n_pre * float(test_window_frac))))
    val_n = _window_from_total_with_gap("val", val_total_n, gap_bars)
    test_n = _window_from_total_with_gap("test", test_total_n, gap_bars)

    max_train_end = n_pre - val_total_n - test_total_n
    if max_train_end <= train_min:
        raise RuntimeError(
            "Not enough pre-holdout data to create the requested number of walk-forward folds. "
            "Reduce num_train_folds or shrink val/test windows."
        )

    train_ends = np.linspace(train_min, max_train_end, num=num_folds)
    train_ends = np.round(train_ends).astype(np.int64)

    if len(np.unique(train_ends)) != num_folds:
        raise RuntimeError(
            "Walk-forward fold construction produced duplicate train end points. "
            "Reduce num_train_folds or adjust split fractions."
        )

    splits: List[SplitIndices] = []
    for fold_idx, train_end in enumerate(train_ends, start=1):
        train_end = int(train_end)
        gap_val_start = train_end
        val_start = gap_val_start + int(gap_bars)
        val_end = val_start + int(val_n)
        gap_test_start = val_end
        test_start = gap_test_start + int(gap_bars)
        test_end = test_start + int(test_n)

        if test_end > n_pre:
            raise RuntimeError(f"Fold {fold_idx} exceeds the pre-holdout boundary.")

        split = SplitIndices(
            train=idx_preholdout[:train_end].copy(),
            gap_before_val=idx_preholdout[gap_val_start:val_start].copy(),
            val=idx_preholdout[val_start:val_end].copy(),
            gap_before_test=idx_preholdout[gap_test_start:test_start].copy(),
            test=idx_preholdout[test_start:test_end].copy(),
        )
        if len(split.train) == 0 or len(split.val) == 0 or len(split.test) == 0:
            raise RuntimeError(f"Fold {fold_idx} produced an empty train/val/test split.")
        splits.append(split)

    if len(splits) != num_folds:
        raise RuntimeError(f"Expected exactly {num_folds} walk-forward folds, got {len(splits)}.")
    return splits


def make_final_production_split(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    val_window_frac: float,
    gap_bars: int,
) -> SplitIndices:
    n_pre = len(idx_preholdout)
    val_total_n = max(int(gap_bars) + 1, int(round(n_pre * float(val_window_frac))))
    val_n = _window_from_total_with_gap("final validation", val_total_n, gap_bars)
    train_end = n_pre - val_total_n

    if train_end <= 0:
        raise RuntimeError("Not enough pre-holdout samples for final production split.")

    gap_before_val = idx_preholdout[train_end: train_end + int(gap_bars)].copy()
    idx_val_final = idx_preholdout[train_end + int(gap_bars):].copy()
    expected_gap_before_holdout = _make_block(int(idx_preholdout[-1]) + 1, int(gap_bars))

    split = SplitIndices(
        train=idx_preholdout[:train_end].copy(),
        gap_before_val=gap_before_val,
        val=idx_val_final,
        gap_before_test=expected_gap_before_holdout,
        test=idx_holdout.copy(),
    )
    if len(split.val) != val_n:
        raise RuntimeError("Final validation window was not created correctly.")
    return split


def validate_all_splits(
    idx_preholdout: np.ndarray,
    idx_holdout: np.ndarray,
    cv_splits: Sequence[SplitIndices],
    production_split: SplitIndices,
    gap_bars: int,
    num_folds: int,
) -> None:
    if len(np.intersect1d(idx_preholdout, idx_holdout)) > 0:
        raise AssertionError("Pre-holdout and holdout overlap")

    holdout_gap = _make_block(int(idx_preholdout[-1]) + 1, int(gap_bars))
    if len(np.intersect1d(holdout_gap, idx_preholdout)) > 0 or len(np.intersect1d(holdout_gap, idx_holdout)) > 0:
        raise AssertionError("Holdout gap overlaps with pre-holdout or holdout")

    if len(cv_splits) != int(num_folds):
        raise AssertionError(f"Expected exactly {int(num_folds)} CV folds, got {len(cv_splits)}")

    for fold_idx, split in enumerate(cv_splits, start=1):
        assert_time_order_and_purge(split, gap_bars, label=f"cv_fold_{fold_idx}")

        if not np.all(np.isin(split.train, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: train contains non-preholdout indices")
        if not np.all(np.isin(split.gap_before_val, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: gap_before_val contains non-preholdout indices")
        if not np.all(np.isin(split.val, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: val contains non-preholdout indices")
        if not np.all(np.isin(split.gap_before_test, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: gap_before_test contains non-preholdout indices")
        if not np.all(np.isin(split.test, idx_preholdout)):
            raise AssertionError(f"cv_fold_{fold_idx}: test contains non-preholdout indices")

        if len(np.intersect1d(split.test, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV test")
        if len(np.intersect1d(split.train, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV train")
        if len(np.intersect1d(split.val, idx_holdout)) > 0:
            raise AssertionError(f"cv_fold_{fold_idx}: final holdout leaked into CV val")

    assert_time_order_and_purge(production_split, gap_bars, label="final_production_split")
    if not np.all(np.isin(production_split.train, idx_preholdout)):
        raise AssertionError("Final train contains non-preholdout indices")
    if not np.all(np.isin(production_split.gap_before_val, idx_preholdout)):
        raise AssertionError("Final gap_before_val contains non-preholdout indices")
    if not np.all(np.isin(production_split.val, idx_preholdout)):
        raise AssertionError("Final val contains non-preholdout indices")
    if not np.array_equal(production_split.gap_before_test, holdout_gap):
        raise AssertionError("Final gap_before_test must match the holdout purge gap exactly")
    if not np.array_equal(production_split.test, idx_holdout):
        raise AssertionError("Final production test must equal the final holdout exactly")


def build_split_plan(
    *,
    freq: str,
    lookback_bars: int,
    horizon_minutes: int,
    horizon_bars: int,
    gap_bars: int,
    n_samples: int,
    first_valid_t: int,
    last_valid_t: int,
    sample_index_offset: Optional[int] = None,
    final_holdout_frac: float,
    train_min_frac: float,
    val_window_frac: float,
    test_window_frac: float,
    num_folds: int,
    metadata: Optional[Dict[str, Any]] = None,
) -> SplitPlan:
    idx_preholdout, idx_holdout = make_preholdout_and_holdout_split(
        n_samples=n_samples,
        holdout_frac=float(final_holdout_frac),
        gap_bars=gap_bars,
    )
    cv_splits = make_exact_walk_forward_splits(
        idx_preholdout=idx_preholdout,
        train_min_frac=float(train_min_frac),
        val_window_frac=float(val_window_frac),
        test_window_frac=float(test_window_frac),
        gap_bars=int(gap_bars),
        num_folds=int(num_folds),
    )
    production_split = make_final_production_split(
        idx_preholdout=idx_preholdout,
        idx_holdout=idx_holdout,
        val_window_frac=float(val_window_frac),
        gap_bars=int(gap_bars),
    )
    validate_all_splits(
        idx_preholdout=idx_preholdout,
        idx_holdout=idx_holdout,
        cv_splits=cv_splits,
        production_split=production_split,
        gap_bars=int(gap_bars),
        num_folds=int(num_folds),
    )

    holdout_gap_indices = _make_block(int(idx_preholdout[-1]) + 1, int(gap_bars))
    return SplitPlan(
        freq=str(freq),
        lookback_bars=int(lookback_bars),
        horizon_minutes=int(horizon_minutes),
        horizon_bars=int(horizon_bars),
        gap_bars=int(gap_bars),
        n_samples=int(n_samples),
        first_valid_t=int(first_valid_t),
        last_valid_t=int(last_valid_t),
        sample_index_offset=int(first_valid_t if sample_index_offset is None else sample_index_offset),
        preholdout_indices=idx_preholdout,
        holdout_gap_indices=holdout_gap_indices,
        holdout_indices=idx_holdout,
        cv_folds=cv_splits,
        production_split=production_split,
        params={
            "final_holdout_frac": float(final_holdout_frac),
            "train_min_frac": float(train_min_frac),
            "val_window_frac": float(val_window_frac),
            "test_window_frac": float(test_window_frac),
            "num_folds": int(num_folds),
            "window_fraction_includes_preceding_gap": True,
        },
        metadata=dict(metadata or {}),
    )


def split_indices_to_dict(split: SplitIndices) -> Dict[str, Any]:
    return {
        "train": split.train.tolist(),
        "gap_before_val": split.gap_before_val.tolist(),
        "val": split.val.tolist(),
        "gap_before_test": split.gap_before_test.tolist(),
        "test": split.test.tolist(),
    }


def split_indices_from_dict(payload: Dict[str, Any]) -> SplitIndices:
    return SplitIndices(
        train=_to_numpy_int64(payload["train"]),
        gap_before_val=_to_numpy_int64(payload["gap_before_val"]),
        val=_to_numpy_int64(payload["val"]),
        gap_before_test=_to_numpy_int64(payload["gap_before_test"]),
        test=_to_numpy_int64(payload["test"]),
    )


def split_plan_to_dict(plan: SplitPlan) -> Dict[str, Any]:
    return {
        "plan_version": 1,
        "freq": plan.freq,
        "lookback_bars": int(plan.lookback_bars),
        "horizon_minutes": int(plan.horizon_minutes),
        "horizon_bars": int(plan.horizon_bars),
        "gap_bars": int(plan.gap_bars),
        "n_samples": int(plan.n_samples),
        "first_valid_t": int(plan.first_valid_t),
        "last_valid_t": int(plan.last_valid_t),
        "sample_index_offset": int(plan.sample_index_offset),
        "preholdout_indices": plan.preholdout_indices.tolist(),
        "holdout_gap_indices": plan.holdout_gap_indices.tolist(),
        "holdout_indices": plan.holdout_indices.tolist(),
        "cv_folds": [split_indices_to_dict(split) for split in plan.cv_folds],
        "production_split": split_indices_to_dict(plan.production_split),
        "params": dict(plan.params),
        "metadata": dict(plan.metadata),
    }


def split_plan_from_dict(payload: Dict[str, Any]) -> SplitPlan:
    return SplitPlan(
        freq=str(payload["freq"]),
        lookback_bars=int(payload["lookback_bars"]),
        horizon_minutes=int(payload["horizon_minutes"]),
        horizon_bars=int(payload["horizon_bars"]),
        gap_bars=int(payload["gap_bars"]),
        n_samples=int(payload["n_samples"]),
        first_valid_t=int(payload["first_valid_t"]),
        last_valid_t=int(payload["last_valid_t"]),
        sample_index_offset=int(payload.get("sample_index_offset", payload["first_valid_t"])),
        preholdout_indices=_to_numpy_int64(payload["preholdout_indices"]),
        holdout_gap_indices=_to_numpy_int64(payload["holdout_gap_indices"]),
        holdout_indices=_to_numpy_int64(payload["holdout_indices"]),
        cv_folds=[split_indices_from_dict(item) for item in payload["cv_folds"]],
        production_split=split_indices_from_dict(payload["production_split"]),
        params=dict(payload.get("params") or {}),
        metadata=dict(payload.get("metadata") or {}),
    )


def save_split_plan(path: Path, plan: SplitPlan) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split_plan_to_dict(plan), indent=2), encoding="utf-8")
    return path


def load_split_plan(path: Path) -> SplitPlan:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return split_plan_from_dict(payload)


def _describe_indices(indices: np.ndarray) -> str:
    if len(indices) == 0:
        return "-"
    if len(indices) == 1:
        return f"{int(indices[0])} (n=1)"
    return f"{int(indices[0])}-{int(indices[-1])} (n={len(indices)})"


def build_split_plan_log_table(plan: SplitPlan) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for fold_idx, split in enumerate(plan.cv_folds, start=1):
        rows.append(
            {
                "split": f"cv_fold_{fold_idx:02d}",
                "train": _describe_indices(split.train),
                "gap_1": _describe_indices(split.gap_before_val),
                "val": _describe_indices(split.val),
                "gap_2": _describe_indices(split.gap_before_test),
                "test": _describe_indices(split.test),
            }
        )

    rows.append(
        {
            "split": "final_refit",
            "train": _describe_indices(plan.production_split.train),
            "gap_1": _describe_indices(plan.production_split.gap_before_val),
            "val": _describe_indices(plan.production_split.val),
            "gap_2": _describe_indices(plan.production_split.gap_before_test),
            "test": _describe_indices(plan.production_split.test),
        }
    )
    return pd.DataFrame(rows)


def split_plan_summary(plan: SplitPlan) -> Dict[str, Any]:
    return {
        "freq": plan.freq,
        "lookback_bars": int(plan.lookback_bars),
        "horizon_minutes": int(plan.horizon_minutes),
        "horizon_bars": int(plan.horizon_bars),
        "gap_bars": int(plan.gap_bars),
        "n_samples": int(plan.n_samples),
        "preholdout_n": int(len(plan.preholdout_indices)),
        "holdout_gap_n": int(len(plan.holdout_gap_indices)),
        "holdout_n": int(len(plan.holdout_indices)),
        "num_cv_folds": int(len(plan.cv_folds)),
        "window_fraction_includes_preceding_gap": bool(
            plan.params.get("window_fraction_includes_preceding_gap", False)
        ),
    }

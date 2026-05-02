# Final Holdout Alignment Table

Source: `final_runs/*/splits/split_summary.json`, `split_indices.npz`, `resolved_config.yaml`.

Interpretation:
- `literal_match_within_freq = True` means the saved `idx_holdout` arrays are exactly identical within that frequency regime.
- `final_test_equals_holdout = True` means `final_production.test` is exactly the same saved holdout interval.
- `semantic_alignment_vs_1min` compares calendar-time alignment against the 1-minute reference window.

| run | freq | slice | holdout_frac | holdout_start_utc | holdout_end_utc | holdout_n | final_test_equals_holdout | literal_match_within_freq | effective_holdout_full_frac | delta_vs_1min | semantic_alignment_vs_1min |
| --- | --- | --- | ---: | --- | --- | ---: | --- | --- | --- | --- | --- |
| 1min-base-gnn-conv | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s | same_window |
| 1min-base-gnn-mpnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s | same_window |
| 1min-memory-gnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s | same_window |
| 1min-multi-gnn-conv | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s | same_window |
| 1min-multi-gnn-mpnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s | same_window |
| 1sec-base-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 1sec-base-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 1sec-memory-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 1sec-memory-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 1sec-multi-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 1sec-multi-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s | near_aligned_late_period |
| 5min-base-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:05:00+00:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end -120s | near_aligned_late_period |
| 5min-memory-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:05:00+00:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end -120s | near_aligned_late_period |
| 5min-multi-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:05:00+00:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end -120s | near_aligned_late_period |

## Frequency-level summary

| freq | runs | shared_holdout_start_utc | shared_holdout_end_utc | shared_holdout_n | note |
| --- | ---: | --- | --- | ---: | --- |
| 1min | 5 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:07:00+00:00 | 1543 | reference window |
| 1sec | 6 | 2021-04-17T03:25:34+00:00 | 2021-04-18T05:10:01+00:00 | 92668 | starts 34s later than 1min, ends 181s later |
| 5min | 3 | 2021-04-17T03:25:00+00:00 | 2021-04-18T05:05:00+00:00 | 309 | same start as 1min, ends 120s earlier |

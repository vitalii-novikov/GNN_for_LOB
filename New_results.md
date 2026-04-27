# 5\. New results

| Frequency | Model | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5min | base-gnn-conv | 0.028 | 0.020 | 26 | 0.62 | 0.70 |
| 5min | base-gnn-mpnn | 0.014 | 0.007 | 25 | 0.61 | 0.73 |
| 5min | multi-gnn-conv | 0.013 | 0.002 | 36 | 0.62 | 0.67 |
| 5min | multi-gnn-mpnn | 0.003 | \-0.009 | 41 | 0.63 | 0.71 |
| 5min | memory-gnn-conv | 0.009 | 0.004 | 17 | 0.61 | 0.73 |
| 5min | memory-gnn-mpnn | \-0.012 | \-0.037 | 83 | 0.54 | 0.73 |
| 1min | base-gnn-conv | 0.060 | 0.020 | 132 | 0.53 | 0.65 |
| 1min | base-gnn-mpnn | \-0.024 | \-0.079 | 181 | 0.50 | 0.64 |
| 1min | multi-gnn-conv | 0.026 | \-0.008 | 112 | 0.54 | 0.63 |
| 1min | multi-gnn-mpnn | \-0.003 | \-0.036 | 108 | 0.53 | 0.65 |
| 1min | memory-gnn-conv | \-0.031 | \-0.079 | 159 | 0.48 | 0.64 |
| 1min | memory-gnn-mpnn | 0.034 | 0.009 | 81 | 0.53 | 0.64 |
| 1sec | base-gnn-conv | 0.140 | \-0.094 | 779 | 0.60 | 0.84 |
| 1sec | base-gnn-mpnn | 0.053 | \-0.066 | 395 | 0.60 | 0.85 |
| 1sec | multi-gnn-conv | 0.082 | \-0.109 | 635 | 0.60 | 0.84 |
| 1sec | multi-gnn-mpnn | 0.080 | \-0.081 | 535 | 0.60 | 0.87 |
| 1sec | memory-gnn-conv | 0.412 | \-1.163 | 5251 | 0.59 | 0.49 |
| 1sec | memory-gnn-mpnn | 0.224 | \-0.281 | 1681 | 0.60 | 0.86 |

## 5.5. `last_CV` versus `final_refit` for selected models

### 5.5.1. Best 5-minute model

base-gnn-conv

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| `5min` | last\_cv | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | final\_refit | 0.01757 | 0.01127 | 21 | 0.630702 | 0.721795 |
|  |  | 160% | 181% |  | 98% | 97% |

### 5.5.2. Best 1-minute model

base-gnn-conv

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| `1min` | last\_cv | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | final\_refit | 0.007198 | \-0.012002 | 64 | 0.524286 | 0.635712 |
|  |  | 829% | **\+** |  | 100% | 102% |

### 5.5.3. Selected 1-second memorygraph case

memory-gnn-conv

Table 5.4. `last_CV` versus `final_refit` for the selected 1-second memorygraph case.

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| `1sec` | last\_cv | 0.412032 | \-1.163268 | 5251 | 0.588785 | 0.49005 |
| `1sec` | final\_refit | 0.443031 | \-0.954969 | 4660 | 0.592186 | 0.852874 |
|  |  | 93% | **\-** |  | 99% | 57% |

### 5.5.4. Interesting Case with 5 minute model (dir\_auc \> 70% at final\_refit)

`multi-gnn-conv`

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| 5min | last\_cv | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
|  | final\_refit | 0.061167 | 0.041367 | 66 | 0.703947 | 0.72092 |
|  |  | 21% | 5% |  | 88% | 93% |

### 
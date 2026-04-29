# 1. Introduction

## 1.1. Motivation of the topic

Financial markets generate large volumes of event-like information. At high frequency, prices, spreads, order-flow summaries, and visible depth change faster than a human analyst can inspect them manually. The limit order book is therefore a natural object for data science: it records the state of supply and demand near the current price and offers a dense, structured view of short-horizon market dynamics [2]. At the same time, it is one of the most difficult settings for predictive modelling. Useful signals are weak, non-stationary, highly regime-dependent, and quickly degraded by transaction costs, latency, threshold choice, and overfitting. These difficulties are consistent with well-known stylized facts of financial returns, including heavy tails, volatility clustering, and changing dependence structures [1].

This difficulty creates both a scientific and a practical motivation. Scientifically, limit order book prediction is a useful test case for machine learning on noisy sequential data. A model must process temporal dependence, cross-asset information, and changing liquidity conditions without relying on a stationary data-generating process. Practically, a forecast is not valuable only because it predicts the direction of a future move. It becomes valuable only if it can be translated into sufficiently selective trading decisions after costs. This thesis therefore treats predictive quality, gross signal quality, and net economic value as related but distinct layers of evidence.

Recent machine learning research has shown that deep architectures can extract useful representations from limit order book data. Convolutional and recurrent models have been used to learn local book structure and temporal dependencies, and large-scale studies have reported evidence that order-flow histories contain cross-instrument regularities [3-5]. These results motivate the use of representation learning, but they do not remove the need for careful evaluation. A model that produces a good ranking statistic can still fail as an entry model if it trades too frequently or if its positive gross edge is smaller than the cumulative cost burden.

Graph-based modelling provides an additional motivation. Financial assets do not evolve independently: their returns, order-flow pressure, and liquidity states can co-move, lead, lag, or diverge. A graph representation makes this relational structure explicit by representing assets as nodes and cross-asset dependencies as edges. Static graph neural networks are useful when the relation structure is fixed or slowly varying [6-9], but market microstructure is dynamic. For this reason, temporal graph neural networks, dynamic graph learning, and memory-augmented graph architectures are natural candidates for short-horizon market modelling [10-14].

The present thesis studies this idea in a deliberately controlled form. It does not attempt to build a full production trading system. Instead, it asks whether richer graph architectures improve a common entry-model benchmark when the data, targets, output heads, validation design, thresholding logic, and event-based trading evaluation are held as consistent as possible. The empirical question is therefore architectural: under a shared benchmark, does it help to preserve multiple relation channels, to add stateful memory, or to use a richer message-passing graph operator?

[image needed 1.1.1] (Conceptual flow from order book snapshots to graph-based entry decisions)

## 1.2. Research gap and thesis scope

The literature contains several relevant strands. Market microstructure research studies how order flow, liquidity, and book depth shape short-horizon price formation. Deep learning research on limit order books shows that neural architectures can extract features from high-dimensional book states. Graph neural network research provides methods for learning from relational data. Temporal graph learning extends this idea to systems whose node states, edge states, or interaction patterns change over time. Recent financial graph studies also show that multi-relational graph structures can be useful for stock prediction at lower frequencies.

The gap addressed here is narrower and more empirical. Many financial graph studies focus on daily or lower-frequency relations, while many limit order book studies model a single instrument without explicitly representing cross-asset graph structure. This thesis examines a small but controlled crypto limit order book setting in which ADA, BTC, and ETH form a three-node graph, ETH is the target asset, and cross-asset relation states are rebuilt at `5min`, `1min`, and `1sec` resolutions. The study focuses on whether graph family, graph operator, and temporal resolution change the usefulness of relational and memory-aware modelling under a common trading-oriented evaluation.

The scope is intentionally limited. The benchmark uses a fixed asset universe, a fixed target asset, a common triple-barrier target construction, and a shared non-overlapping event backtest. This design improves internal comparability, but it also means that the thesis evaluates entry models rather than complete trading systems with jointly optimized execution and exit policies. The resulting conclusions should therefore be interpreted as evidence about architecture under a controlled benchmark, not as evidence that any model is production-ready.

## 1.3. Research questions

The thesis is guided by four research questions.

**RQ1. Which graph family performs best under a controlled entry-model benchmark?**  
The first question asks whether the simpler single-graph baseline, the multi-relation graph family, or the stateful memory graph family produces the strongest final-holdout trading result when all families are evaluated under the same target construction, thresholding logic, and event-based backtest.

**RQ2. How important is the Conv-versus-MPNN operator choice inside each family?**  
Each family is evaluated with a Conv-style operator and an MPNN-style operator. This makes it possible to distinguish the effect of the broader family scaffold from the effect of the local graph interaction mechanism.

**RQ3. How does temporal resolution change the relative value of relational and memory mechanisms?**  
The `5min` and `1min` regimes solve the same 30-minute lookback and 5-minute horizon task, while the `1sec` regime uses a frequency-adapted two-minute lookback and two-minute horizon. This design allows the thesis to examine whether richer relation handling and recurrent memory become more useful as the observation frequency increases.

**RQ4. Are the conclusions stable under deployment-oriented model states?**  
The thesis distinguishes between `last_CV`, the final walk-forward fold model used as the primary deployment-oriented reference, and `final_refit`, a model refit on a larger pre-holdout sample. This question asks whether the same model remains attractive when viewed through both states, and what this implies for realistic deployment interpretation.

[image needed 1.3.1] (Research-question map linking graph family, operator, frequency, and deployment state)

## 1.4. Hypotheses

The empirical design tests five hypotheses.

**H1. The 1-minute regime should be the strongest shared-task benchmark.**  
Because the `1min` data preserve more intra-horizon dynamics than `5min` data while remaining less noisy than second-level data, the `1min` regime is expected to be the strongest of the two strict shared-task regimes.

**H2. Explicit multi-relation modelling should outperform the simpler baseline more clearly at finer resolutions.**  
The `multigraph` family is expected to benefit from preserving price-dependence, order-flow, and liquidity channels separately, especially when cross-asset dependencies evolve quickly.

**H3. Stateful memory should become more valuable as the market is observed more finely.**  
The `memorygraph` family is expected to be most useful at high frequency because recurrent state can, in principle, retain short-lived market information across contiguous observations.

**H4. Conv and MPNN operators should not be uniformly dominant across families.**  
The operator comparison is expected to be family- and frequency-dependent. A Conv-style operator may be more stable when edge structure is already regularized, while an MPNN-style operator may be more useful when messages need richer source-destination-edge conditioning.

**H5. `last_CV` and `final_refit` should tell a broadly consistent but not identical story.**  
The broad family-level conclusion is expected to remain similar across states, but individual model profitability may change when a larger pre-holdout sample is used for refitting.

## 1.5. Thesis contribution

The thesis contributes a controlled empirical comparison of graph-based market microstructure models across three temporal resolutions. Its contribution is not a new universal architecture, nor a claim of production deployment readiness. Instead, it provides evidence on three narrower issues:

1. whether a simpler single-graph baseline is sufficient under a common entry-model benchmark;
2. whether explicit multi-relation handling or stateful memory improves economic outcomes after costs;
3. why deployment-oriented model states, turnover, and cost drag must be included in the interpretation of high-frequency predictive models.

The main empirical finding is conservative. In the updated benchmark, `base_gnn + adaptive_conv` is the strongest shared-task model at both `5min` and `1min`. Richer graph mechanisms sometimes improve gross signal or ranking metrics, especially at `1sec`, but these gains do not reliably translate into positive net profitability after transaction costs.

# 2. Literature Background

## 2.1. Market microstructure and limit order book prediction

Market microstructure studies how trading rules, liquidity provision, order flow, and the organization of the limit order book affect price formation [2]. At short horizons, the visible book is informative because it contains the current distribution of buy and sell interest near the mid-price. However, short-horizon predictability is difficult to exploit. Return distributions are heavy-tailed, volatility clusters over time, and market regimes change [1]. These stylized facts make financial forecasting different from many stationary supervised-learning problems.

Limit order book modelling also creates an evaluation challenge. A direction classifier can appear useful under a conventional accuracy or AUC metric while still being economically weak if it triggers too many low-margin trades. For this reason, this thesis evaluates models through a friction-aware entry benchmark rather than through classification metrics alone. Directional AUC and trade AUC are retained as diagnostics, but `pnl_sum`, `gross_pnl_sum`, and `n_trades` are treated as central to the final interpretation.

## 2.2. Deep learning for limit order books

Deep learning research on limit order books has shown that neural networks can learn representations from high-dimensional book states [3]. DeepLOB is especially relevant because it combines convolutional components for local book structure with recurrent components for temporal dependence [5]. Large-scale studies of order-flow histories also suggest that neural models can identify cross-instrument regularities in price formation [4]. These studies motivate representation learning in market microstructure, but they also highlight the need for careful separation between predictive performance and economic performance.

The present thesis differs from single-instrument LOB prediction studies by making cross-asset relational structure explicit. Instead of treating ETH only as an isolated time series, ADA and BTC are included as context nodes. The resulting task is still modest in graph size, but it allows the thesis to test whether graph modelling adds value once all families share the same target and backtest.

## 2.3. Graph neural networks and message passing

Graph neural networks provide a general framework for learning from entities connected by relations. Graph convolutional networks and graph attention networks show how node representations can be updated using neighbourhood information, while message-passing neural networks provide a flexible formulation in which messages depend on source nodes, destination nodes, and edge attributes [6-9]. These ideas are directly relevant to financial data because assets can be represented as nodes and cross-asset dependence measures as edges.

In this thesis, the Conv-versus-MPNN distinction is used as a controlled operator comparison. Conv-style graph layers apply weighted source-node projections with edge-conditioned shifts. MPNN-style layers use richer gated messages that condition on source state, destination state, and edge state. The comparison therefore asks whether richer local message conditioning is economically useful under the same graph-family scaffold.

## 2.4. Temporal and dynamic graph learning

Many real systems are not static graphs. Node states, edge states, and interaction patterns can evolve over time. Temporal graph networks and dynamic graph representation learning address this problem by combining graph operators with temporal encoders, memory modules, or event-driven updates [10-12]. This literature is relevant to market microstructure because cross-asset relations are unlikely to remain fixed across regimes, liquidity states, and trading intensity.

The three families in this thesis instantiate this idea at different levels of complexity. `base_gnn` uses early relation fusion and a convolutional temporal backbone. `multigraph` preserves relation-specific graph pathways longer before fusing them. `memorygraph` uses recurrent node and edge memory inside a graph-processing loop. The empirical question is not whether these mechanisms are theoretically plausible; it is whether they improve a controlled friction-aware benchmark.

## 2.5. Financial graph learning

Financial applications of graph neural networks include stock relation modelling, portfolio prediction, risk propagation, and transaction-network analysis [13]. Recent multi-relational dynamic graph work is especially relevant because financial relations can be heterogeneous: firms, sectors, correlations, supply chains, and market co-movements can all define different relation channels [14]. This thesis adopts a microstructure version of the same general idea by constructing relation channels from price dependence, order-flow dependence, and liquidity dependence.

However, this thesis remains cautious about novelty. It does not claim to solve dynamic financial graph learning in general. Its contribution is an internally controlled comparison on a specific crypto LOB dataset and a specific entry-model benchmark.

# 3. Data and Methodology

## 3.1. Data source and study universe

The raw data source is the public Kaggle dataset *High-Frequency Crypto Limit Order Book Data* [15]. The local experiment pipelines use frequency-specific CSV files for ADA, BTC, and ETH at `1sec`, `1min`, and `5min` resolutions. ETH is the target asset, while ADA and BTC provide relational context. The source data are already organized as frequency-specific order book summaries rather than raw message streams, so the study does not reconstruct the book from individual order messages.

The local files contain midpoint price, spread, buy and sell flow summaries, and depth values for fifteen bid-side and fifteen ask-side levels. The preprocessing task is therefore to standardize timestamps, align the three assets on a common clock, derive node and edge features, and construct labels for the target asset.

[image needed 3.1.1] (Dataset structure: assets, frequencies, order book fields, and target asset)

## 3.2. Graph representation

Each timestamp is represented as a directed graph over three assets: ADA, BTC, and ETH. The graph is complete and includes self-loops. Nodes correspond to assets. Edges correspond to ordered source-destination asset pairs. Node states and edge states vary over time.

The graph is small, but the design is useful for a controlled thesis benchmark. It allows the model to use cross-asset information without introducing a large universe in which many additional modelling choices would become difficult to isolate. The small universe also makes the results easier to interpret: the benchmark tests whether relational structure is useful under controlled conditions, not whether scale alone improves performance.

## 3.3. Node features

For each asset and timestamp, the implemented node feature block contains:

1. one-bar log return;
2. relative spread;
3. log-transformed buys;
4. log-transformed sells;
5. flow imbalance;
6. total depth imbalance;
7. top-level depth imbalances for the first five book levels;
8. bid near/far depth ratio;
9. ask near/far depth ratio;
10. near-depth imbalance;
11. far-depth imbalance.

These features summarize local price change, order-flow pressure, and the shape of visible bid-ask depth. They are not intended to exhaust all possible microstructure features. Their purpose is to define a common input representation for the architecture comparison.

## 3.4. Relation states and edge features

The pipelines construct three relation-state series per asset:

1. `price_dep`, represented by the asset log return;
2. `order_flow`, represented by flow imbalance scaled by log turnover;
3. `liquidity`, represented by a spread-and-depth composite based on relative spread, total depth imbalance, near-depth imbalance, and a bounded near/far depth shape ratio.

For every ordered asset pair and relation channel, rolling dependence features are computed over lag-window combinations. The edge tensor contains rolling correlation, rolling beta, and rolling mean product. Correlations are Fisher-z transformed when configured. This design gives all model families access to the same relation-aware edge representation.

[image needed 3.4.1] (Node features, relation states, and edge-feature construction)

## 3.5. Scaling and leakage control

Node and edge tensors are scaled with robust statistics fitted only on training data for the relevant fold or refit split. The scaled values are clipped to bounded ranges. This fold-specific scaling is important because global scaling over the full sample would leak information from later periods into earlier model fitting. The thesis therefore treats scaling as part of the validation design, not as a separate preprocessing step applied once to the full dataset.

## 3.6. Frequency-specific experimental regimes

The experimental design evaluates six model variants in each of three frequency regimes:

1. `base_gnn + adaptive_conv`;
2. `base_gnn + adaptive_mpnn`;
3. `multigraph + dynamic_rel_conv`;
4. `multigraph + dynamic_edge_mpnn`;
5. `memorygraph + conv`;
6. `memorygraph + mpnn`.

The `5min` and `1min` regimes solve the same clock-time task: a 30-minute lookback and a 5-minute forecast horizon. This corresponds to 6 lookback bars and 1 horizon bar at `5min`, and 30 lookback bars and 5 horizon bars at `1min`. For both regimes, the working sample uses the first 90% of the local series, and the final holdout is the final 10% of that working sample.

The `1sec` regime uses a frequency-adapted task: a 2-minute lookback and a 2-minute horizon, corresponding to 120 bars each. The working sample is restricted to the interval from 50% to 90% of the full second-level series. The final holdout fraction is increased to 0.225 so that the blind evaluation period is better aligned with the late-period market segment used in the slower-frequency experiments.

| Frequency | Working data slice | Final holdout fraction | Lookback | Horizon | Interpretation |
| :--- | :--- | ---: | :--- | :--- | :--- |
| `5min` | `0.0-0.9` | 0.10 | 30 min = 6 bars | 5 min = 1 bar | Strict shared-task benchmark |
| `1min` | `0.0-0.9` | 0.10 | 30 min = 30 bars | 5 min = 5 bars | Strict shared-task benchmark |
| `1sec` | `0.5-0.9` | 0.225 | 2 min = 120 bars | 2 min = 120 bars | Frequency-adapted high-frequency stress test |

The consequence is important for interpretation. `5min` and `1min` are directly comparable as shared-task regimes. `1sec` remains comparable within its own frequency but should not be treated as a perfectly symmetric continuation of the slower regimes.

[image needed 3.6.1] (Frequency regimes and clock-time horizon conversion)

## 3.7. Target construction and learning objective

All models are trained under a common multi-task triple-barrier framework. For each valid timestamp, the future ETH midpoint path is followed until one of three event types occurs: the upper barrier is touched, the lower barrier is touched, or the vertical barrier is reached. The upper and lower barriers are volatility-scaled from an 8 basis point base level, use a 30-bar volatility lookback, are multiplied by 1.8, and are clipped between 4 and 30 basis points. The vertical barrier is set equal to the prediction horizon.

The pipelines construct five target outputs:

1. realized return;
2. trade relevance label;
3. direction label;
4. exit-type label;
5. time-to-exit label.

The trade label is meta-labeled and depends on whether the future move remains economically meaningful after a friction-aware threshold. Direction labels are masked when timeout outcomes are configured as uninformative for directional supervision. The shared multi-task objective combines trade classification, direction classification, return regression, utility-based supervision, exit-type classification, and time-to-exit regression. This common target interface is central to the benchmark because it prevents different families from being evaluated under different prediction tasks.

## 3.8. Entry-model benchmark and backtesting logic

The thesis evaluates models as entry models. The trade head determines whether a trade candidate is active. The direction head determines whether the position is long or short. The exit is generated by the same realized event rule for all families. Exit-type and time-to-exit heads are retained as auxiliary learning targets and diagnostics, but they do not define family-specific exit policies in the main benchmark.

The backtest is sequential and non-overlapping. Once a position is opened, no new position can be opened until the current position is closed. For trade `i`, gross PnL is computed as:

\[
\text{gross\_pnl}_i = s_i \cdot r_i,
\]

where `s_i` is the trade side and `r_i` is the realized log return up to the event exit. Net PnL subtracts a round-trip cost proxy:

\[
\text{net\_pnl}_i = \text{gross\_pnl}_i - c_{rt}.
\]

With `cost_bps_per_side = 1.0`, the implemented cost proxy is:

\[
c_{rt} = 3 \times 1.0 \times 10^{-4} = 0.0003.
\]

The reported `pnl_sum` values are therefore interpreted directly as exported post-cost benchmark outputs, not as manually reconstructed table values.

[image needed 3.8.1] (Entry-model benchmark: score thresholds, trade side, realized event exit, and net PnL)

## 3.9. Validation design and deployment-oriented model states

The experiments use purged walk-forward validation. The working sample is split into a pre-holdout development region and a final blind holdout region. Within the pre-holdout region, folds follow a chronological train-gap-validation-gap-test structure. The purge gaps are necessary because triple-barrier labels depend on future price paths; adjacent observations can otherwise share overlapping future information.

The thesis distinguishes three model states:

1. `best_CV`, the strongest selected cross-validation checkpoint;
2. `last_CV`, the final walk-forward fold model and the primary deployment-oriented reference;
3. `final_refit`, a model refit on the largest available pre-holdout sample and used as a larger-sample robustness check.

The full benchmark in the Results chapter uses `last_CV`. This is appropriate because a live deployment would most closely resemble taking the latest trained fold model forward into an unseen period. `final_refit` is informative because it shows whether using more pre-holdout data changes the final-holdout interpretation, but it cannot replace `last_CV` as the deployment-oriented reference. A refit can improve some ranking metrics while worsening realized trading outcomes.

[image needed 3.9.1] (Chronological split: pre-holdout, final holdout, walk-forward folds, and purge gaps)

# 4. Model Families

## 4.1. Shared architectural conventions

All three model families operate on the same two input tensors: a node sequence and a relation-aware edge sequence. The node sequence has dimensions corresponding to batch, lookback length, assets, and node features. The edge sequence has dimensions corresponding to batch, lookback length, relation channels, directed edges, and edge features.

All families also share the same output heads:

1. `trade_logit`;
2. `dir_logit`;
3. `return_pred`;
4. `exit_type_logit`;
5. `tte_pred`.

The key differences are therefore not the target variables or output interface. They are the temporal backbone, the treatment of relation channels, and the local graph operator.

## 4.2. The `base_gnn` family

The `base_gnn` family is the single-graph baseline. It first encodes node and edge histories with dilated causal convolutional blocks. It then fuses handcrafted relation-edge features with a learnable pairwise node-interaction path. The relation axis is collapsed before graph processing, producing one fused edge representation per directed pair.

The graph component is implemented through a single graph operator block. In the thesis benchmark, the two relevant variants are:

1. `adaptive_conv`, a Conv-style graph operator with adaptive adjacency;
2. `adaptive_mpnn`, an MPNN-style graph operator with adaptive adjacency.

After graph processing, `base_gnn` uses a target-centered readout that combines the ETH target-node representation with global graph context. A final causal temporal trunk processes the target-centered sequence before the shared prediction heads are evaluated.

## 4.3. The `multigraph` family

The `multigraph` family extends the baseline by preserving relation-specific graph pathways. It uses the same convolutional node and edge temporal encoders as `base_gnn`, but it does not collapse the relation axis before graph processing. Instead, it applies a dedicated relation graph block to each relation channel: price dependence, order flow, and liquidity.

The two tested operators are:

1. `dynamic_rel_conv`, the Conv-style relation-specific variant;
2. `dynamic_edge_mpnn`, the MPNN-style relation-specific variant.

After relation-specific graph processing, the model applies attention-based relation fusion and then uses the same target-centered readout and target temporal trunk as the baseline. The family therefore tests whether late relation fusion is more useful than early relation fusion under the same entry-model task.

## 4.4. The `memorygraph` family

The `memorygraph` family is the most distinct architecture. It replaces the deep convolutional temporal backbone with recurrent node and edge memory. Raw node inputs are projected per step, raw edge inputs are projected per step, and the resulting states are processed through a memory-augmented graph block.

The memory block maintains node memory and relation-specific edge memory. Edge memory is updated with a recurrent cell conditioned on edge state and source-destination node interactions. Node memory is then updated after graph message passing by aggregating relation-specific edge-memory context and relation-specific node states. The two tested graph operators are:

1. `conv`, a Conv-style graph operator inside the recurrent memory loop;
2. `mpnn`, an MPNN-style graph operator inside the recurrent memory loop.

The final fused node sequence is passed to the shared target-centered readout and prediction heads. There is no separate target temporal trunk because temporal accumulation is already performed by recurrent memory.

[image needed 4.4.1] (Architectural comparison: early relation fusion, late relation fusion, and recurrent graph memory)

# 5. Results

## 5.1. Reading the result tables

The Results chapter reports the deployment-oriented `last_CV` benchmark first. This is the primary empirical comparison because `last_CV` most closely represents a model that has been trained chronologically and then carried forward into a blind final-holdout interval. The main metrics are:

1. `gross_pnl_sum`, the sum of pre-cost directional trade returns;
2. `pnl_sum`, the post-cost net trading result;
3. `n_trades`, the number of executed non-overlapping trades;
4. `dir_auc`, a directional ranking diagnostic;
5. `trade_auc`, a trade-relevance ranking diagnostic.

The interpretation gives priority to `pnl_sum`, but not in isolation. A positive `pnl_sum` based on very few trades is weaker evidence than a positive result supported by a larger and more stable trade set. Likewise, a strong AUC is not sufficient evidence of deployment value if it does not translate into selective post-cost trades.

## 5.2. Overview of the eighteen `last_CV` benchmark models

Table 5.1 reports the updated `last_CV` benchmark. The table uses the short labels from the run artifacts: `base-gnn-conv` corresponds to `base_gnn + adaptive_conv`, `base-gnn-mpnn` corresponds to `base_gnn + adaptive_mpnn`, `multi-gnn-conv` corresponds to `multigraph + dynamic_rel_conv`, `multi-gnn-mpnn` corresponds to `multigraph + dynamic_edge_mpnn`, and the memory labels correspond to `memorygraph + conv/mpnn`.

| Frequency | Model | Gross PnL sum | PnL sum | N trades | Dir AUC | Trade AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `base-gnn-conv` | 0.028 | 0.020 | 26 | 0.62 | 0.70 |
| `5min` | `base-gnn-mpnn` | 0.014 | 0.007 | 25 | 0.61 | 0.73 |
| `5min` | `multi-gnn-conv` | 0.013 | 0.002 | 36 | 0.62 | 0.67 |
| `5min` | `multi-gnn-mpnn` | 0.003 | -0.009 | 41 | 0.63 | 0.71 |
| `5min` | `memory-gnn-conv` | 0.009 | 0.004 | 17 | 0.61 | 0.73 |
| `5min` | `memory-gnn-mpnn` | -0.012 | -0.037 | 83 | 0.54 | 0.73 |
| `1min` | `base-gnn-conv` | 0.060 | 0.020 | 132 | 0.53 | 0.65 |
| `1min` | `base-gnn-mpnn` | -0.024 | -0.079 | 181 | 0.50 | 0.64 |
| `1min` | `multi-gnn-conv` | 0.026 | -0.008 | 112 | 0.54 | 0.63 |
| `1min` | `multi-gnn-mpnn` | -0.003 | -0.036 | 108 | 0.53 | 0.65 |
| `1min` | `memory-gnn-conv` | -0.031 | -0.079 | 159 | 0.48 | 0.64 |
| `1min` | `memory-gnn-mpnn` | 0.034 | 0.009 | 81 | 0.53 | 0.64 |
| `1sec` | `base-gnn-conv` | 0.140 | -0.094 | 779 | 0.60 | 0.84 |
| `1sec` | `base-gnn-mpnn` | 0.053 | -0.066 | 395 | 0.60 | 0.85 |
| `1sec` | `multi-gnn-conv` | 0.082 | -0.109 | 635 | 0.60 | 0.84 |
| `1sec` | `multi-gnn-mpnn` | 0.080 | -0.081 | 535 | 0.60 | 0.87 |
| `1sec` | `memory-gnn-conv` | 0.412 | -1.163 | 5251 | 0.59 | 0.49 |
| `1sec` | `memory-gnn-mpnn` | 0.224 | -0.281 | 1681 | 0.60 | 0.86 |

Three patterns are central.

First, the updated shared-task winner is `base-gnn-conv` at both `5min` and `1min`. This is the most important correction relative to the earlier draft narrative. The strongest shared-task evidence no longer supports the claim that the MPNN variant of the baseline is the best model. The strongest evidence supports the simpler baseline family and its Conv-style adaptive operator.

Second, richer graph structures do not produce superior post-cost shared-task performance. At `5min`, `multi-gnn-conv` and `memory-gnn-conv` remain positive but below the baseline Conv winner. At `1min`, the only richer-family positive result is `memory-gnn-mpnn`, but its net result is less than half of the `base-gnn-conv` result.

Third, the `1sec` regime shows strong gross signal but poor net realization. Every listed `1sec` model is net negative after costs. The memory-based models are especially informative: `memory-gnn-conv` produces the largest gross result in the entire benchmark, but it also executes 5251 trades and loses heavily after costs. The issue is therefore not simply absence of signal. It is the failure to convert high-frequency signal into sufficiently selective post-cost trades.

[image needed 5.2.1] (Benchmark heatmap: PnL sum by frequency, family, and operator)

## 5.3. Frequency-specific results

### 5.3.1. The `5min` regime

The `5min` regime produces the clearest positive shared-task result. `base_gnn + adaptive_conv` achieves the highest net outcome, with `gross_pnl_sum = 0.028` and `pnl_sum = 0.020` over 26 trades. The second-best baseline variant, `base_gnn + adaptive_mpnn`, remains positive but materially lower, with `pnl_sum = 0.007` over 25 trades.

The richer graph families show isolated strengths but not superior deployment outcomes. `multigraph + dynamic_rel_conv` remains slightly positive with `pnl_sum = 0.002`, while `memorygraph + conv` reaches `pnl_sum = 0.004`. Both results indicate that richer relation or memory mechanisms can produce usable trades in this slower regime, but neither challenges the baseline Conv winner. The weakest result is `memorygraph + mpnn`, which trades much more often than the positive memory variant and ends at `pnl_sum = -0.037`.

The main `5min` interpretation is therefore conservative. At the coarsest shared-task frequency, early relation fusion with a stable adaptive Conv-style operator is sufficient to produce the strongest post-cost result. Additional architectural complexity is not rewarded by the primary economic metric.

### 5.3.2. The `1min` regime

The `1min` regime is a richer stress test because it contains more observations and produces more trades. It also confirms the updated shared-task conclusion. `base_gnn + adaptive_conv` is again the strongest model, with `gross_pnl_sum = 0.060`, `pnl_sum = 0.020`, and 132 trades. This result is economically similar to the best `5min` result in net terms, but it is supported by a larger number of trades and a larger gross edge.

The larger gross edge does not fully translate into net improvement because turnover is higher. This is the first sign of a pattern that becomes dominant at `1sec`: increasing temporal resolution can reveal more trading opportunities, but it also increases the importance of selectivity and cost control.

The `memorygraph + mpnn` variant is the strongest richer-family model at `1min`, with `pnl_sum = 0.009` over 81 trades. This is meaningful evidence that recurrent graph memory can be economically useful at minute frequency. However, it remains below the baseline Conv winner. The `multigraph` variants both finish net negative, despite non-trivial ranking diagnostics. This weakens the hypothesis that explicit relation-channel preservation is automatically valuable at finer shared-task resolution.

### 5.3.3. The `1sec` regime

The `1sec` regime creates the strongest separation between gross signal and net profitability. All six listed models are net negative. This does not mean that the models detect no structure. On the contrary, `memory-gnn-conv` reaches `gross_pnl_sum = 0.412`, and `memory-gnn-mpnn` reaches `gross_pnl_sum = 0.224`. The problem is that these gross edges are accompanied by very high trade counts. With a round-trip cost proxy of 0.0003, 5251 trades imply an approximate cumulative cost burden of 1.5753, far larger than the gross edge of `memory-gnn-conv`.

The baseline and multigraph families also fail to remain net positive at `1sec`. Their losses are smaller than the memorygraph Conv loss because they trade less aggressively, but they still do not clear the cost hurdle. The best `1sec` net result in the updated table is `base-gnn-mpnn` at `pnl_sum = -0.066`, which is still not deployment-grade.

The high-frequency conclusion is therefore not that memory is useless. The evidence is more precise: memory can surface short-lived gross signal, but under the current thresholding and entry-only benchmark it does not impose enough selectivity. The model extracts opportunities faster than the trading rule can filter them economically.

[image needed 5.3.3] (Gross versus net PnL at `1sec`, highlighting turnover cost drag)

## 5.4. Answer to RQ1: which graph family performs best?

The answer to RQ1 is that the `base_gnn` family performs best overall under the controlled entry-model benchmark. This conclusion is strongest in the two strict shared-task regimes. At both `5min` and `1min`, the top model is `base_gnn + adaptive_conv`. The family-level result is therefore robust across the two comparable clock-time tasks.

The conclusion should be stated carefully. The result does not prove that richer relation modelling or recurrent memory is generally inferior. It shows that, under this dataset, feature construction, thresholding logic, and event-based entry benchmark, the simpler baseline family produces the strongest post-cost shared-task evidence. The richer families sometimes improve gross signal or ranking diagnostics, but they do not convert those advantages into superior net profitability.

The updated single-model conclusion is also clear: the strongest shared-task specification is `base_gnn + adaptive_conv`, not `base_gnn + adaptive_mpnn`.

## 5.5. Answer to RQ2: how important is the Conv-versus-MPNN choice?

The Conv-versus-MPNN choice is important, but the updated evidence changes its interpretation. It is no longer defensible to claim that MPNN dominates the shared-task benchmark. At `5min`, the Conv variant is better for all three families on `pnl_sum`: `base-gnn-conv` beats `base-gnn-mpnn`, `multi-gnn-conv` beats `multi-gnn-mpnn`, and `memory-gnn-conv` beats `memory-gnn-mpnn`. At `1min`, the pattern is mixed but still strongly favors Conv for the baseline: `base-gnn-conv` is the regime winner, while `base-gnn-mpnn` is strongly negative. In contrast, the memory family favors MPNN at `1min`.

At `1sec`, MPNN reduces net losses for `base_gnn`, `multigraph`, and `memorygraph`, but none of the variants becomes profitable. This means the operator effect is real but conditional. A richer message-passing operator can improve some high-frequency outcomes, yet it does not solve the cost-drag problem by itself.

The most defensible answer to RQ2 is therefore that operator choice must be evaluated jointly with family and frequency. Conv is the stronger shared-task operator in the primary benchmark, especially for the winning baseline family. MPNN remains useful as a diagnostic and in some frequency-family combinations, but it should not be treated as uniformly superior.

## 5.6. Answer to RQ3: how does temporal resolution affect richer relation and memory mechanisms?

The results do not support the simple hypothesis that finer temporal resolution automatically increases the economic value of richer graph mechanisms. Moving from `5min` to `1min` increases the gross edge and number of trades for the winning baseline Conv model, but it does not materially improve net profitability. Moving to `1sec` increases the evidence of gross signal in memory-based models, but net profitability deteriorates sharply because trade counts and transaction costs dominate.

For `multigraph`, the evidence is weak on the primary economic metric. It does not beat `base_gnn` in any frequency regime. This does not mean that relation-channel preservation is conceptually unhelpful. It means that, in this benchmark, preserving relation channels deeper into the network does not produce better post-cost entries.

For `memorygraph`, the evidence is more nuanced. The family becomes most distinctive at `1sec`, where memory-based models produce the largest gross signals. Yet the same regime exposes their main weakness: recurrent memory increases responsiveness, but the current entry rule does not filter enough trades. The value of memory therefore appears more clearly before costs than after costs.

## 5.7. Answer to RQ4: are conclusions stable between `last_CV` and `final_refit`?

The `last_CV` versus `final_refit` comparison shows that the two states are related but not interchangeable. `last_CV` remains the primary deployment-oriented state because it resembles carrying the most recent fold model into an unseen period. `final_refit` is useful as a robustness check because it trains on a larger pre-holdout sample, but a larger training sample does not guarantee a better realized trading outcome.

### 5.7.1. Best 5-minute selected model: `base-gnn-conv`

| Frequency | Training cycle | Gross PnL sum | PnL sum | N trades | Dir AUC | Trade AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `last_CV` | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | `final_refit` | 0.017570 | 0.011270 | 21 | 0.630702 | 0.721795 |

The `5min` baseline Conv model remains positive after refitting. Its net result declines, but its ranking metrics improve. This is a relatively stable deployment interpretation: the model remains economically positive in both states, although `final_refit` is not superior on the primary economic metric.

### 5.7.2. Best 1-minute selected model: `base-gnn-conv`

| Frequency | Training cycle | Gross PnL sum | PnL sum | N trades | Dir AUC | Trade AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `1min` | `last_CV` | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | `final_refit` | 0.007198 | -0.012002 | 64 | 0.524286 | 0.635712 |

The `1min` case is less stable. The `last_CV` model is profitable, but the `final_refit` model turns net negative. The AUC measures change only modestly, which means the economic deterioration is not fully explained by a collapse in ranking quality. Instead, the translation from scores to realized trades changes enough to alter post-cost profitability. This case is a strong reason to keep `last_CV` central in deployment-oriented evaluation.

### 5.7.3. Selected 1-second memorygraph case: `memory-gnn-conv`

| Frequency | Training cycle | Gross PnL sum | PnL sum | N trades | Dir AUC | Trade AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `1sec` | `last_CV` | 0.412032 | -1.163268 | 5251 | 0.588785 | 0.490050 |
| `1sec` | `final_refit` | 0.443031 | -0.954969 | 4660 | 0.592186 | 0.852874 |

This comparison is analytically important because it shows that refitting can improve signal diagnostics without producing a deployable net result. `final_refit` improves gross PnL, reduces the trade count, improves `dir_auc`, and sharply improves `trade_auc`, yet the model remains strongly negative after costs. The central high-frequency problem remains excessive turnover relative to the available gross edge.

### 5.7.4. Informative 5-minute refit case: `multi-gnn-conv`

| Frequency | Training cycle | Gross PnL sum | PnL sum | N trades | Dir AUC | Trade AUC |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `last_CV` | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| `5min` | `final_refit` | 0.061167 | 0.041367 | 66 | 0.703947 | 0.720920 |

The `multi-gnn-conv` refit case is notable because the final-refit directional AUC exceeds 70% and the net result improves substantially. This does not overturn the primary benchmark, because the thesis prioritizes `last_CV` for deployment-oriented comparison. It does, however, show that relation-aware modelling can become attractive under a larger-sample refit state. The correct interpretation is therefore not that `multigraph` is irrelevant. It is that its strongest evidence appears in a refit robustness case rather than in the primary deployment-oriented benchmark.

[image needed 5.7.1] (Deployment-state comparison: `last_CV` versus `final_refit` for selected models)

## 5.8. Hypothesis assessment

**H1. The 1-minute regime should be the strongest shared-task benchmark.**  
This hypothesis is not supported on the primary economic metric. The best `5min` and `1min` models have nearly equal net PnL, but `5min` is slightly higher (`0.020356` versus `0.020094`). The `1min` regime remains important because it produces more trades and a larger gross edge, but it is not stronger in strict net-profit terms.

**H2. Explicit multi-relation modelling should outperform the simpler baseline more clearly at finer resolutions.**  
This hypothesis is not supported by the `last_CV` economic benchmark. `multigraph` does not beat `base_gnn` at `5min`, `1min`, or `1sec`. The informative `5min` `multi-gnn-conv` final-refit case gives partial evidence that multi-relation modelling can improve under a larger-sample state, but this is not enough to overturn the primary deployment-oriented conclusion.

**H3. Stateful memory should become more valuable as the market is observed more finely.**  
This hypothesis is partially supported before costs but not supported after costs. At `1sec`, `memorygraph` produces the strongest gross signals, which is consistent with the idea that recurrent state can capture short-lived high-frequency structure. However, those signals are expressed through too many trades, and both memory variants are strongly negative after costs.

**H4. Conv and MPNN operators should not be uniformly dominant across families.**  
This hypothesis is supported, but the updated evidence favors Conv more strongly in the shared-task benchmark. Conv is the winning operator for the baseline at both `5min` and `1min`, and it also beats MPNN across all families at `5min`. MPNN reduces losses at `1sec` and helps memorygraph at `1min`, but it is not uniformly superior.

**H5. `last_CV` and `final_refit` should tell a broadly consistent but not identical story.**  
This hypothesis is partially supported. Both states show that no model is deployment-ready at `1sec`, and both support caution about treating predictive metrics as sufficient. However, model-level conclusions can change materially. The `1min` baseline Conv winner turns negative after refitting, while the `5min` multigraph Conv model becomes much stronger under final refit. The states must therefore be reported separately.

# 6. Discussion

## 6.1. Main findings by research question

The main finding is that architectural complexity did not guarantee better trading outcomes. Under the strict shared-task benchmark, the simpler `base_gnn` family is strongest, and the winning specification is `base_gnn + adaptive_conv`. This result is methodologically important because it was obtained under a common target construction, common output interface, common thresholding framework, and common event-based backtest.

For RQ1, the strongest family is `base_gnn`. For RQ2, the operator effect is substantial and conditional, but Conv is the stronger shared-task choice. For RQ3, finer temporal resolution increases the visibility of gross high-frequency signal, especially for memorygraph, but it also increases cost drag and turnover risk. For RQ4, `last_CV` and `final_refit` provide complementary but non-equivalent evidence.

A central theme is that predictive quality, gross signal extraction, and net profitability must be separated. The `1sec` memorygraph results make this most visible. A model can generate a large positive gross PnL and still fail economically because the number of trades is too high. Conversely, a model with less dramatic ranking statistics can be more useful if it is more selective.

## 6.2. Comparison with previous work

The results are consistent with the broader limit order book literature in one respect: short-horizon market data contain learnable structure [3-5]. The presence of positive gross PnL in several models, especially at `1sec`, supports the idea that order book and cross-asset microstructure features contain information about subsequent movement.

However, the results also qualify optimistic interpretations of deep learning for market prediction. Prior work such as DeepLOB emphasizes the ability of neural models to learn from high-dimensional order book states [5]. This thesis shows that, in a graph-based crypto setting, representation learning alone is not sufficient. The economic translation layer matters. Thresholds, trade frequency, costs, and deployment state can change the conclusion even when ranking metrics appear reasonable.

The graph-learning literature motivates the use of relation-aware models, and recent financial graph studies motivate multi-relational dynamic modelling [6-14]. The present findings are more cautious. Multi-relation modelling is plausible and sometimes useful, but in the primary `last_CV` benchmark it does not beat the simpler baseline. This does not contradict the graph literature; rather, it shows that graph complexity must be evaluated against the specific economic objective and data regime.

The temporal graph and memory literature also provides a useful comparison [11-12]. Temporal memory is designed to preserve information across changing graph states. The memorygraph results support this idea before costs: the largest `1sec` gross signals come from memory-based models. Yet they also show that memory without sufficiently selective trading control can amplify turnover. In market microstructure, the ability to detect many short-lived opportunities is not enough if the opportunities are too small relative to costs.

## 6.3. Scientific implications

The scientific implication is that graph-based market microstructure research should evaluate architecture under deployment-aware metrics, not only under predictive metrics. AUC, accuracy, and regression error remain useful, but they do not settle whether a model is economically meaningful. The benchmark must include trade count, gross PnL, net PnL, and cost sensitivity.

A second implication is that relation modelling should be treated as an empirical design choice rather than an assumed improvement. Preserving multiple relation channels is theoretically attractive, especially in finance, but the updated benchmark shows that early fusion in a simpler baseline can outperform late relation-specific processing under some conditions.

A third implication concerns temporal resolution. Higher frequency can increase the quantity of detectable signal, but it also raises the cost of acting on that signal. Research that evaluates high-frequency models without explicit turnover and cost analysis risks overstating practical value.

## 6.4. Practical and deployment implications

The strongest deployment-oriented evidence favors `base_gnn + adaptive_conv` in the shared-task regimes. Even this conclusion should be interpreted cautiously. The model is not a production trading system; it is the best entry model in a controlled final-holdout benchmark. A real deployment would require latency modelling, exchange-specific fee and slippage assumptions, monitoring for regime drift, capital constraints, live order placement logic, and risk controls.

`last_CV` is deployment-relevant because it resembles the chronological situation of using the latest available model on an unseen future segment. `final_refit` adds information about whether a larger training sample changes the holdout result, but it cannot replace `last_CV`. The `1min` example demonstrates why: the refit model preserves similar ranking diagnostics but turns net negative. The `5min` multigraph Conv example demonstrates the opposite possibility: a model that is weak in `last_CV` can become strong under final refit. Both cases show that deployment interpretation depends on model state.

The updated results imply that realistic deployment should prioritize selectivity and cost robustness. The high-frequency memory models are not deployable in their current form because their gross edge is overwhelmed by turnover. Future deployment-oriented work should therefore evaluate trade-rate controls, threshold stability, cost sensitivity, and no-trade calibration before treating high-frequency graph signals as practically useful.

[image needed 6.4.1] (Deployment interpretation: predictive ranking, gross signal, net profitability, and model state)

## 6.5. Limitations, weaknesses, and sources of bias

Several limitations affect the interpretation of the thesis.

First, the asset universe is small. The graph contains only ADA, BTC, and ETH, with ETH as the target. This improves interpretability but creates asset-selection bias. A larger universe could change the value of multi-relation modelling because more relation pathways would be available.

Second, the data source is limited to a specific crypto LOB dataset. Crypto markets have distinctive features, including continuous trading, high volatility, exchange-specific liquidity, and changing market regimes. Results from this dataset should not be generalized automatically to equities, futures, or other exchanges.

Third, the benchmark is vulnerable to market-regime bias. A chronological final holdout is more realistic than random splitting, but it still represents a particular late-period segment. A different holdout period could produce different rankings.

Fourth, the label construction introduces label-construction bias. Triple-barrier labels depend on volatility estimates, barrier multipliers, timeout handling, and the treatment of economically meaningful moves. These choices are defensible but not neutral.

Fifth, threshold-selection bias remains possible. Thresholds are selected on validation data and then applied to the final holdout. This is better than selecting thresholds on the holdout itself, but the threshold grid and feasibility constraints still shape the trading outcomes.

Sixth, the cost model is simplified. A constant round-trip proxy is useful for controlled comparison, but real trading costs include exchange fees, bid-ask spread, queue position, slippage, market impact, latency, and failed execution. The high-frequency conclusions are especially sensitive to this issue.

Seventh, the `1sec` regime is frequency-adapted rather than directly equivalent to the slower regimes. It uses a shorter clock-time task and a restricted working sample for feasibility. The `1sec` results should therefore be interpreted as a high-frequency stress test rather than a direct extension of the shared `5min`/`1min` task.

Finally, the architecture comparison is controlled but not perfect. The main benchmark fixes the key task and evaluation components, but practical runs can include frequency-specific and family-specific configuration choices required for computational feasibility. This should be treated as a threat to internal validity when making strong claims about architecture alone.

# 7. Conclusions and Future Research

## 7.1. Overall conclusion

This thesis asked whether graph-based and memory-aware architectures improve a controlled market microstructure entry benchmark. The updated evidence supports a conservative answer. The `base_gnn` family performs best overall in the strict shared-task regimes, and the strongest shared-task specification is `base_gnn + adaptive_conv`. Richer relation handling and recurrent memory do not produce superior `last_CV` net profitability under the current benchmark.

The results do not imply that graph complexity is unhelpful in general. They show that complexity must be justified by post-cost economic evidence. `multigraph` produces an informative `5min` final-refit case with strong directional AUC and positive net PnL, and `memorygraph` produces the strongest `1sec` gross signal. These findings indicate that richer architectures can capture useful structure. The problem is that the structure is not consistently converted into deployment-grade net profitability.

The most important conceptual conclusion is that high-frequency model evaluation must separate three layers: predictive ranking, gross signal extraction, and net trading performance. A model can be good at the first two and still fail at the third. This distinction is not a minor implementation detail; it is central to any serious data science evaluation of trading models.

## 7.2. Future research

The first direction is turnover-aware model design. The `1sec` experiments show that memory-based graph models can identify many short-lived opportunities, but they lack sufficient selectivity. Future work should investigate cost-aware objectives, stricter no-trade calibration, sparse event-driven state updates, and threshold policies that explicitly penalize excessive trading.

The second direction is execution-aware evaluation. The present benchmark fixes a common realized-event exit rule in order to preserve fairness. Future work could keep the common entry benchmark for comparability and then test the strongest entry models under adaptive exits, richer slippage assumptions, exchange-specific fees, latency constraints, and queue-position modelling.

The third direction is larger-universe graph modelling. A three-asset graph is useful for a controlled thesis benchmark, but it may understate the value of relation-specific architectures. A larger crypto or cross-asset universe would provide a stronger test of whether multigraph methods become more valuable when relation diversity increases.

The fourth direction is stability analysis. The current thesis reports final-holdout outcomes and selected `last_CV` versus `final_refit` comparisons. Future research should add fold-level dispersion, bootstrap confidence intervals for economic metrics, pairwise model comparison tests, and regime-specific performance summaries.

The fifth direction is richer bias and robustness evaluation. Future work should explicitly test sensitivity to the holdout period, asset universe, barrier parameters, cost assumptions, threshold grids, and data source. This would make the conclusions more robust and help distinguish genuine architecture effects from benchmark-specific outcomes.

[image needed 7.2.1] (Future research roadmap: selectivity, execution realism, larger graphs, and robustness)

# References

[1] Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance, 1(2), 223-236. PDF: https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf

[2] Cont, R., Stoikov, S., & Talreja, R. (2010). *A stochastic model for order book dynamics*. Operations Research, 58(3), 549-563. PDF: https://rama.cont.perso.math.cnrs.fr/pdf/CST2010.pdf

[3] Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting, 37(8), 852-866. arXiv/PDF: https://arxiv.org/abs/1705.03233

[4] Sirignano, J., & Cont, R. (2019). *Universal features of price formation in financial markets: perspectives from deep learning*. Quantitative Finance, 19(9), 1449-1459. arXiv/PDF: https://arxiv.org/abs/1803.06917

[5] Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep convolutional neural networks for limit order books*. IEEE Transactions on Signal Processing, 67(11), 3001-3012. PDF: https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf

[6] Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. International Conference on Learning Representations. arXiv/PDF: https://arxiv.org/abs/1609.02907

[7] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). *Neural message passing for quantum chemistry*. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1263-1272. PDF: https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

[8] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph attention networks*. International Conference on Learning Representations. arXiv/PDF: https://arxiv.org/abs/1710.10903

[9] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2021). *A comprehensive survey on graph neural networks*. IEEE Transactions on Neural Networks and Learning Systems, 32(1), 4-24. arXiv/PDF: https://arxiv.org/abs/1901.00596

[10] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). *Connecting the dots: Multivariate time series forecasting with graph neural networks*. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 753-763. arXiv/PDF: https://arxiv.org/abs/2005.11650

[11] Kazemi, S. M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., & Poupart, P. (2020). *Representation learning for dynamic graphs: A survey*. Journal of Machine Learning Research, 21(70), 1-73. PDF: https://jmlr.csail.mit.edu/papers/volume21/19-447/19-447.pdf

[12] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). *Temporal graph networks for deep learning on dynamic graphs*. arXiv preprint arXiv:2006.10637. arXiv/PDF: https://arxiv.org/abs/2006.10637

[13] Wang, J., Zhang, S., Xiao, Y., & Song, R. (2022). *A review on graph neural network methods in financial applications*. Journal of Data Science, 20(2), 111-134. PDF: https://jds-online.org/journal/JDS/article/1279/file/pdf

[14] Qian, H., Zhou, H., Zhao, Q., Chen, H., Yao, H., Wang, J., Liu, Z., Yu, F., Zhang, Z., & Zhou, J. (2024). *MDGNN: Multi-relational dynamic graph neural network for comprehensive and dynamic stock investment prediction*. Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14642-14650. arXiv/PDF: https://arxiv.org/abs/2402.06633

[15] Martinsn. *High-Frequency Crypto Limit Order Book Data*. Kaggle dataset. Dataset page: https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data

# Graph Neural Networks for Limit Order Book Prediction under Deployment-Oriented Evaluation

## Abstract

This thesis studies whether graph-based neural architectures improve short-horizon limit order book prediction when they are evaluated under a common, deployment-oriented protocol. The empirical setting is a cryptocurrency limit order book dataset containing ADA, BTC, and ETH snapshots at 5-minute, 1-minute, and 1-second resolution. ETH is treated as the target asset, while ADA and BTC provide relational market context. The study compares three model families: a single-graph baseline (`base_gnn`), a multi-relation graph architecture (`multigraph`), and a stateful recurrent graph architecture (`memorygraph`). Each family is evaluated with a convolution-style graph operator and a message-passing neural network operator, producing six model variants per frequency regime and eighteen primary benchmark configurations overall.

The methodological contribution of the thesis is a controlled benchmark rather than a claim of production trading readiness. All model families share the same asset universe, feature construction, relation-state construction, triple-barrier target logic, multi-task learning objective, purged walk-forward validation design, final holdout interval, thresholding logic, and event-based backtest. This design isolates architectural differences while keeping the trading evaluation comparable. The central deployment distinction is between `last_CV`, the model state closest to a realistic deployment reference, and `final_refit`, a larger-sample refit state that is informative but not a substitute for chronological deployment evidence.

The updated empirical results show that the strongest `last_CV` model at both 5-minute and 1-minute frequency is `base-gnn-conv`, not the previously assumed MPNN baseline. At 5-minute frequency, `base-gnn-conv` achieves `pnl_sum = 0.020356` over 26 trades. At 1-minute frequency, the same family and operator achieves `pnl_sum = 0.020094` over 132 trades. At 1-second frequency, all benchmark models remain net negative after transaction costs. The most important high-frequency finding is therefore not that no signal exists, but that gross signal and deployable net profitability diverge sharply once turnover and costs are included. The `memory-gnn-conv` model, for example, produces the largest 1-second gross signal, but its very high trade count makes the net result strongly negative.

The thesis concludes that, under the controlled entry-model benchmark used here, the simpler single-graph baseline is the most reliable architecture. More complex relation-preserving and memory-based models produce useful diagnostic evidence, especially in ranking quality and gross signal extraction, but they do not establish a robust post-cost advantage. The main unresolved challenge is therefore not merely extracting short-horizon microstructure signal. It is converting that signal into sufficiently selective, stable, and cost-aware trading decisions.

# 1. Introduction

## 1.1. Motivation of the topic

Limit order books are among the most information-rich data structures in electronic financial markets. At each moment, they record the visible supply and demand available at different price levels. For an algorithmic trading system, the limit order book is therefore not only a price record but a dynamic description of market pressure, liquidity, and short-term imbalance. Predicting future price movement from such data is scientifically difficult because the relevant signals are weak, noisy, nonstationary, and strongly affected by market frictions.

The difficulty is especially clear at high frequency. At longer horizons, price dynamics may be partly explained by macroeconomic information, firm-specific news, or broad risk factors. At the level of minutes or seconds, however, the prediction problem is dominated by microstructure effects: queue pressure, bid-ask spread variation, short-lived liquidity shocks, cross-asset spillovers, and order-flow imbalance. These effects are not independent across assets. A movement in BTC may change the informational environment for ETH and ADA; a liquidity shock in one asset may alter relative risk appetite across the crypto market. A model that treats each asset as an isolated time series may therefore miss a relevant part of the market state.

Graph neural networks offer a natural representation for this problem because they model entities as nodes and relations as edges. In the present thesis, the nodes are assets and the edges summarize rolling dependence channels between assets. This does not mean that the true market is literally a three-node graph. Rather, the graph formalism provides a disciplined way to ask whether relational information improves short-horizon prediction when all models are evaluated on the same task.

[image needed 1.1.1] (Crypto limit order book prediction as a dynamic cross-asset graph)

The motivation for temporal and memory-aware graph models follows from the same logic. Market relations are not stable constants. They vary across regimes, volatility states, and liquidity conditions. Foundational graph convolution and message-passing methods show how node information can be propagated over a graph [4], [5]. Multivariate time-series graph models extend this idea by learning relations among variables over time [6]. Temporal graph networks and dynamic graph models then add mechanisms for evolving interactions and memory [7], [8]. In finance, recent work has explored graph methods for asset relations and stock prediction [9], [10]. However, much of that literature works at daily frequency, uses static or slowly changing graphs, or focuses primarily on predictive accuracy rather than friction-aware trading evaluation.

Limit order book research has also shown that deep learning can extract useful high-frequency representations. DeepLOB combines convolutional filters and recurrent components to predict price movements from order book states [2], while Sirignano and Cont provide evidence that deep learning can identify common price-formation structure across equities [3]. These works motivate deep representation learning for market microstructure. They do not, however, eliminate the need for deployment-aware evaluation. A model can rank future directions well and still fail as a trading system if it trades too often, ignores costs, or is evaluated on a temporally contaminated split.

This thesis is positioned at that intersection. It does not attempt to build a complete production trading system. Instead, it asks a narrower and more scientifically controllable question: under a common target, common features, common validation procedure, and common event-based backtest, which graph architecture is most reliable for limit order book prediction? The answer is important because architectural complexity is often intuitively attractive in financial machine learning. Multi-relation graph processing and recurrent memory seem well matched to high-frequency markets. Yet such mechanisms are only useful for trading if they improve deployable, post-cost performance rather than only intermediate predictive metrics.

## 1.2. Research gap and contribution

The research gap addressed by this thesis is not the absence of deep learning for order books, nor the absence of graph neural networks in finance. Both areas already contain substantial work. The gap is the lack of a controlled, deployment-oriented comparison of graph-family design choices for high-frequency limit order book prediction under a common trading benchmark.

Several points make this gap practically relevant. First, many financial forecasting papers report classification accuracy, AUC, or return-prediction metrics without fully separating gross signal from post-cost profitability. Second, graph-based financial models are often evaluated on daily stock data, where relation dynamics and transaction-cost pressure differ substantially from minute-level or second-level limit order book data. Third, richer architectures may be compared under different targets or execution assumptions, making it unclear whether gains are due to the architecture or to the surrounding evaluation procedure. Fourth, final refitting on larger samples can look attractive academically but may not represent how a model would actually be selected and deployed after chronological validation.

The contribution of this thesis is therefore empirical and methodological:

1. It constructs a common graph-based limit order book benchmark across 5-minute, 1-minute, and 1-second regimes.
2. It compares three model families under matched inputs and matched targets: `base_gnn`, `multigraph`, and `memorygraph`.
3. It compares Conv-style and MPNN-style graph operators inside each family.
4. It evaluates results using both predictive diagnostics and event-based trading metrics.
5. It separates `last_CV` from `final_refit` to distinguish deployable chronological evidence from larger-sample refit evidence.

[image needed 1.2.1] (Thesis contribution: architecture comparison plus deployment-aware evaluation)

The thesis avoids exaggerated novelty claims. The individual building blocks, including graph convolution, message passing, temporal convolutions, recurrent memory, and triple-barrier labeling, are not new in isolation. The novelty lies in combining them into a controlled empirical comparison for crypto limit order book prediction and interpreting the results through a deployment-aware lens.

## 1.3. Research aim

The aim of this thesis is to determine whether richer graph-based architectures improve short-horizon limit order book prediction and trading performance when evaluated under an apples-to-apples, friction-aware benchmark.

The core object of interest is the model family. The study asks whether a simple single-graph representation is sufficient, whether preserving multiple relation channels improves the result, and whether stateful memory becomes valuable at higher temporal resolution. A second object of interest is the graph operator: a Conv-style operator versus a message-passing neural network operator. A third object of interest is deployment stability: whether the same conclusions hold when moving from the last chronological cross-validation state to a final refit state.

# 2. Research Questions and Hypotheses

This thesis studies graph-based market microstructure models under a deliberately controlled protocol. The objective is not to compare loosely related trading systems that differ in targets, exits, or validation logic. The objective is to compare graph architectures under a common supervised learning problem, common input construction, common multi-task objective, and common event-based backtest.

The comparison is organized along two axes. The first axis is the model family:

1. `base_gnn`, a single fused graph baseline.
2. `multigraph`, a relation-preserving graph architecture.
3. `memorygraph`, a stateful recurrent graph architecture.

The second axis is the graph interaction mechanism:

1. a Conv-style graph operator.
2. an MPNN-style graph operator.

Consequently, each family is evaluated in two matched variants, producing six primary models per frequency regime:

1. `base_gnn + adaptive_conv` (`base-gnn-conv`)
2. `base_gnn + adaptive_mpnn` (`base-gnn-mpnn`)
3. `multigraph + dynamic_rel_conv` (`multi-gnn-conv`)
4. `multigraph + dynamic_edge_mpnn` (`multi-gnn-mpnn`)
5. `memorygraph + conv` (`memory-gnn-conv`)
6. `memorygraph + mpnn` (`memory-gnn-mpnn`)

These six models are evaluated at three temporal resolutions:

1. `5min`, with a 30-minute lookback and a 5-minute horizon.
2. `1min`, with a 30-minute lookback and a 5-minute horizon.
3. `1sec`, with a 2-minute lookback and a 2-minute horizon.

The study therefore contains eighteen primary experimental configurations. Within each frequency regime, the comparison is apples-to-apples because the input representation, target construction, thresholding logic, and final trading benchmark are fixed while only the family scaffold and graph operator vary.

## 2.1. Research questions

### RQ1. Which graph family performs best under a controlled entry-model benchmark?

The first research question asks which of the three graph families performs best when evaluated under the same entry-model task. This is the central question of the thesis because it isolates the value of the architectural scaffold from the value of family-specific trading rules.

### RQ2. How important is the Conv-versus-MPNN operator choice inside each family?

Each family is instantiated with two graph interaction mechanisms. This makes it possible to ask whether performance differences are explained mainly by the broader family scaffold or by the local graph operator. The answer is important because the same operator comparison is repeated inside all three families.

### RQ3. How does temporal resolution change the value of relation-preserving and memory-aware mechanisms?

The 5-minute and 1-minute experiments solve the same clock-time task, whereas the 1-second experiments use a shorter task that remains realistic at ultra-high frequency. This design makes it possible to examine whether richer relational modeling and persistent memory become more useful as temporal resolution increases.

### RQ4. Are the conclusions stable under deployment-oriented model states?

All models are interpreted through deployment-relevant model states. The `last_CV` model is the primary deployable reference because it comes from the most recent chronological walk-forward fold before the final holdout. The `final_refit` model is a larger-sample refit state. This leads to a practical question: do the same model families remain preferable when evaluated in a deployment-oriented setting rather than under a more optimistic refit setting?

[image needed 2.1.1] (Research-question map linking family, operator, frequency, and deployment state)

## 2.2. Hypotheses

### H1. The 1-minute regime should be the strongest shared-task benchmark.

Under the common 30-minute lookback and 5-minute horizon task, the 1-minute regime is expected to outperform the 5-minute regime on the main comparison metrics. The reason is that 1-minute data preserve substantially more intra-horizon information than 5-minute aggregation while remaining less noisy and less costly to trade than second-level data.

### H2. Explicit multi-relation modeling should outperform the simpler baseline more clearly at finer resolutions.

The `multigraph` family is expected to deliver larger gains over `base_gnn` at 1-minute than at 5-minute frequency, and potentially stronger gains at 1-second frequency, because relation-specific cross-asset dependencies may evolve rapidly at short horizons.

### H3. Stateful memory should become more valuable as the market is observed more finely.

The `memorygraph` family is expected to be most competitive at 1-second frequency, moderately competitive at 1-minute frequency, and least differentiated at 5-minute frequency. This hypothesis follows from the architecture: `memorygraph` replaces the fully convolutional temporal encoding of the other families with recurrent memory, which should be useful when signals are transient and recent state matters.

### H4. Conv and MPNN operators should not be uniformly dominant across families.

The Conv-versus-MPNN comparison is expected to be family-dependent rather than universal. An MPNN operator may be useful when interactions require richer conditioning on source, destination, and edge states, whereas a Conv-style operator may be more stable when the edge structure is already strongly regularized.

### H5. `last_CV` and `final_refit` should tell a broadly consistent family-level story.

Although `final_refit` may improve some metrics by using a larger pre-holdout sample, the broad family ranking is expected to remain similar across the two states. A family that only looks attractive after refitting, but not under the realistic `last_CV` benchmark, would be less convincing from a deployment perspective.

# 3. Methodology

## 3.1. Data source and study universe

The raw data source is the public Kaggle dataset *High-Frequency Crypto Limit Order Book Data* by Martinsn, which provides frequency-specific cryptocurrency limit order book snapshots for multiple assets, including ADA, BTC, and ETH, at `1sec`, `1min`, and `5min` resolutions [1]. The data are distributed as order book snapshots organized by price level rather than as raw exchange message streams.

The present study uses a fixed three-node asset universe:

1. ADA
2. BTC
3. ETH

ETH is the target asset. ADA and BTC provide relational market context. Because the source data are already distributed in frequency-specific tables, no bespoke reconstruction of the limit order book from raw order messages is required. The preprocessing task is instead to standardize timestamps, align assets on a common clock, and derive node and edge features from the available order book summaries.

The local data files used in the pipelines contain midpoint price, spread, buy and sell flow summaries, and 15 bid-side and 15 ask-side depth values. These fields are the foundation for all node features and relation features used in the benchmark.

## 3.2. Graph input representation

All models use the same graph input representation within a frequency regime. The graph is a directed complete graph over the three assets with self-loops. The nodes are fixed, but node states and edge states vary over time.

Formally, each model receives:

1. a node sequence \(X^{(n)} \in R^{B \times L \times N \times F_n}\)
2. a relation-aware edge sequence \(X^{(e)} \in R^{B \times L \times R \times E \times F_e}\)

where \(B\) is batch size, \(L\) is the lookback length, \(N = 3\) is the number of assets, \(R = 3\) is the number of relation channels, and \(E\) is the number of directed edges including self-loops.

The three relation channels are:

1. `price_dep`, based on asset log returns.
2. `order_flow`, based on flow imbalance scaled by log turnover.
3. `liquidity`, based on spread, depth imbalance, near-depth imbalance, and near/far depth shape.

[image needed 3.2.1] (Three-asset directed graph with price, order-flow, and liquidity relation channels)

## 3.3. Node features

For each asset and each time step, the node feature block summarizes local price behavior, order-flow pressure, and depth structure. The implemented node features are:

1. one-bar log return.
2. relative spread.
3. log-transformed buys.
4. log-transformed sells.
5. flow imbalance.
6. total depth imbalance.
7. top-level depth imbalances for the first five book levels.
8. bid near/far depth ratio.
9. ask near/far depth ratio.
10. near-depth imbalance.
11. far-depth imbalance.

This feature set is deliberately microstructure-oriented. It does not use external news, social media, or macroeconomic variables. That choice keeps the thesis focused on the information available inside the aligned cross-asset order book state.

## 3.4. Relation states and edge features

Edge features are constructed from rolling cross-asset dependence measures. For every ordered asset pair and every relation channel, the pipeline computes lagged rolling features over frequency-specific windows:

1. rolling correlation.
2. rolling beta.
3. rolling mean product.

When configured, rolling correlations are Fisher-\(z\) transformed before scaling. The edge tensor therefore represents relation-specific dependence among assets rather than only a fixed adjacency prior.

This design is important for fair comparison. All three model families operate on the same handcrafted relation states and the same learnable pairwise edge-fusion path. The architectures differ in how they process and fuse this information, not in whether they receive richer or poorer input data.

## 3.5. Scaling and leakage control

Node and edge tensors are robustly scaled on training data only, using fold-specific quantile statistics. The transformed features are then clipped to bounded ranges before model fitting. This prevents train-test leakage through scaling and reduces the influence of extreme observations. Because the same scaling approach is used for all families, feature preprocessing does not favor any architecture.

## 3.6. Frequency-specific experimental regimes

The experimental design contains eighteen primary runs: six model variants for each of the three frequency regimes.

The `5min` and `1min` regimes solve the same clock-time task:

1. lookback window = 30 minutes.
2. forecast horizon = 5 minutes.

This corresponds to 6 lookback bars and 1 horizon bar at `5min`, and 30 lookback bars and 5 horizon bars at `1min`.

The `1sec` regime uses a frequency-adapted task:

1. lookback window = 2 minutes = 120 bars.
2. forecast horizon = 2 minutes = 120 bars.

The 1-second working sample is restricted to the interval from 50% to 90% of the full second-level series. This keeps training computationally feasible while preserving a late-period high-frequency comparison. The final holdout fraction is increased to align the 1-second blind evaluation interval as closely as possible with the final holdout interval used in the slower-frequency experiments.

| Frequency | Working data slice | Final holdout fraction | Lookback | Horizon | CV folds |
| :--- | :--- | ---: | :--- | :--- | ---: |
| `5min` | `0.0-0.9` of the full series | `0.10` | 30 min = 6 bars | 5 min = 1 bar | 4 |
| `1min` | `0.0-0.9` of the full series | `0.10` | 30 min = 30 bars | 5 min = 5 bars | 4 |
| `1sec` | `0.5-0.9` of the full series | `0.225` | 2 min = 120 bars | 2 min = 120 bars | 2 |

[image needed 3.6.1] (Full sample, working sample, pre-holdout region, and final holdout by frequency)

The `5min` and `1min` regimes are therefore directly comparable as a strict shared-task benchmark. The `1sec` regime is apples-to-apples within its own frequency, but it should be interpreted as a frequency-adapted high-frequency stress test rather than as a perfectly symmetric continuation of the 30-minute/5-minute task.

## 3.7. Target construction and shared learning objective

All model families are trained under the same multi-task triple-barrier framework. For each valid timestamp \(t\), the future path of the ETH midpoint is followed until one of three mutually exclusive events occurs:

1. the upper barrier is touched.
2. the lower barrier is touched.
3. the vertical barrier is reached.

The barrier system is volatility-scaled. In the default benchmark configuration, the upper and lower barriers start from 8 basis points, are rescaled using rolling volatility estimated over a 30-bar lookback, are multiplied by 1.8, and are clipped to the interval from 4 to 30 basis points. The vertical barrier is set equal to the prediction horizon.

From this future path, the pipeline constructs a common target set:

1. realized return.
2. trade relevance label.
3. direction label.
4. exit-type label.
5. time-to-exit label.

The trade label is meta-labeled and depends on whether the future move remains economically meaningful after a friction-aware threshold is applied. Direction labels are masked when timeout outcomes are configured as uninformative for directional supervision.

All families share the same output interface:

1. `trade_logit`
2. `dir_logit`
3. `return_pred`
4. `exit_type_logit`
5. `tte_pred`

The multi-task objective combines trade classification, direction classification, return regression, utility-based supervision, exit-type classification, and time-to-exit regression. In the benchmark configuration, the loss weights are:

1. `loss_w_trade = 0.35`
2. `loss_w_dir = 0.65`
3. `loss_w_ret = 0.15`
4. `loss_w_utility = 0.85`
5. `loss_w_exit_type = 0.05`
6. `loss_w_tte = 0.03`

This shared target design preserves comparability. The models differ in how they encode temporal and graph structure, not in what they are asked to predict.

## 3.8. Common entry-model backtest

The trading evaluation is formulated as a common entry-model benchmark. In the primary backtest:

1. the trade head determines whether a trade candidate is active.
2. the direction head determines whether the candidate becomes a long or short position.
3. the exit is generated by the same realized event rule for all families.

Exit-type and time-to-exit heads are retained as auxiliary learning targets and diagnostics, but they do not define a family-specific trade-closing policy in the main benchmark. This choice is especially important for `memorygraph`, because a stateful architecture could otherwise be evaluated under a different execution policy from the other families. The common entry-model benchmark improves internal validity by holding execution logic fixed.

The trading evaluation uses a sequential non-overlapping event-based backtest. Once a position is opened, no new position can be opened until the current one is closed. This makes turnover interpretable and avoids overlapping position exposure.

For trade \(i\), gross PnL is computed as:

\[
\text{gross\_pnl}_i = s_i \cdot r_i,
\]

where \(s_i \in \{-1, +1\}\) is the trade side and \(r_i\) is the realized log return up to the realized event exit. Net PnL is:

\[
\text{net\_pnl}_i = \text{gross\_pnl}_i - c_{\text{rt}},
\]

where the round-trip transaction-cost proxy is:

\[
c_{\text{rt}} = 3 \times \text{cost\_bps\_per\_side} \times 10^{-4}.
\]

With `cost_bps_per_side = 1.0`, this gives:

\[
c_{\text{rt}} = 0.0003.
\]

The cost model is deliberately simple. It is sufficient for a controlled friction-aware benchmark, but it should not be interpreted as a complete execution simulator.

[image needed 3.8.1] (Entry-model backtest: trade score, direction score, realized event exit, gross and net PnL)

## 3.9. Validation design and deployment-oriented model states

The experiments use purged walk-forward validation. Each working sample is divided into:

1. a pre-holdout region used for model development.
2. a final holdout region used only for blind final evaluation.

Within the pre-holdout region, each walk-forward fold follows a chronological train-gap-validation-gap-test structure. The purge gaps are necessary because triple-barrier labels depend on future price evolution; adjacent observations can have overlapping future windows and would otherwise leak information across split boundaries.

[image needed 3.9.1] (Purged walk-forward validation with train, purge gap, validation, purge gap, test, and final holdout)

The study distinguishes three model states:

1. `best_CV`, the strongest selected cross-validation checkpoint.
2. `last_CV`, the model from the last chronological walk-forward fold.
3. `final_refit`, the model refit on the largest possible pre-holdout sample.

The main thesis benchmark uses `last_CV`. This state is the most deployment-relevant reference because it approximates a model selected from the most recent chronological validation cycle before the final holdout. The `final_refit` state adds a useful larger-sample comparison, but it cannot replace `last_CV`: refitting changes the training sample, may change the score-to-trade conversion, and is not itself evidence that the same model would have been selected in a live walk-forward process.

## 3.10. Metrics

The main empirical metrics are:

1. `gross_pnl_sum`, the sum of pre-cost directional trade returns.
2. `pnl_sum`, the sum of post-cost trade returns.
3. `n_trades`, the number of executed trades.
4. `dir_auc`, the AUC of the direction head.
5. `trade_auc`, the AUC of the trade head.

The primary economic metric is `pnl_sum`. The `gross_pnl_sum` metric separates raw signal extraction from the effect of transaction costs. The `n_trades` metric shows whether the result is supported by meaningful trading activity or by a small number of positions. The AUC metrics are valuable diagnostics for ranking quality, but they are not sufficient evidence of deployable profitability.

[image needed 3.10.1] (Metric hierarchy: ranking quality, gross signal, turnover, and net profitability)

## 3.11. Fair-comparison principle

Within each frequency regime, only two aspects are allowed to vary:

1. the family scaffold (`base_gnn`, `multigraph`, `memorygraph`).
2. the local graph operator (Conv or MPNN).

The following elements are held fixed within a regime:

1. asset universe and target asset.
2. node-feature construction.
3. relation-state construction.
4. edge-feature construction.
5. label construction.
6. multi-task output interface.
7. thresholding logic.
8. event-based backtest.
9. split protocol.
10. final holdout interval.

This is the methodological basis for treating the benchmark as an architecture comparison rather than as a comparison of unrelated trading systems.

# 4. Detailed Description of the Tested Models

## 4.1. Shared architectural conventions

All three model families operate on the same node and edge tensors and produce the same multi-task outputs. They also share a hybrid edge-fusion mechanism that augments handcrafted relation features with learnable pairwise node interactions. The main architectural differences concern:

1. when relation channels are fused.
2. whether the temporal backbone is convolutional or recurrent.
3. whether the local graph operator is Conv-style or MPNN-style.

[image needed 4.1.1] (Architectural comparison of `base_gnn`, `multigraph`, and `memorygraph`)

The two local graph operator types can be summarized as follows. The Conv-style operator applies a weighted source-node projection plus an edge-conditioned shift term. The MPNN-style operator computes gated messages conditioned on source node state, destination node state, and edge state. This makes the MPNN operator more expressive, but not automatically more profitable.

## 4.2. The `base_gnn` family

The `base_gnn` family is the single-graph baseline. It is evaluated through two adaptive operators:

1. `adaptive_conv`
2. `adaptive_mpnn`

The temporal component is fully convolutional. Node inputs are projected into hidden space, augmented with learned asset embeddings, and processed by dilated causal residual convolution blocks. Edge inputs are processed by a separate temporal edge encoder. After graph processing and readout, the target-centered sequence is passed through a second causal temporal trunk.

The graph component first fuses relation-aware edge features, then collapses the relation axis into a single edge representation. A single graph operator block is then applied using adaptive adjacency. This means that `base_gnn` tests whether an early-fused relation representation is sufficient for the benchmark.

The readout concatenates the target-node representation with global graph context, including mean and max pooling and optional target-to-global attention. The resulting target-centered representation is mapped to the shared multi-task prediction heads.

## 4.3. The `multigraph` family

The `multigraph` family extends the baseline by preserving relation channels deeper into the graph-processing stage. It is evaluated in two matched variants:

1. `dynamic_rel_conv`
2. `dynamic_edge_mpnn`

The temporal component is structurally similar to `base_gnn`: node and edge histories are encoded with dilated causal convolution blocks, and the target readout is processed by a causal temporal trunk. The difference is in graph processing. Instead of collapsing the relation axis before message passing, the model constructs a separate relation graph block for each relation channel.

For each relation, the Conv variant computes dynamic edge scores and applies normalized source-node projections and edge-conditioned shifts. The MPNN variant uses gated messages conditioned jointly on source state, destination state, and edge state. After relation-specific processing, the model applies learned relation attention fusion.

The central design question for `multigraph` is whether preserving price-dependence, order-flow, and liquidity relations as separate graph pathways improves the final trading benchmark relative to early relation fusion.

## 4.4. The `memorygraph` family

The `memorygraph` family is the most distinct architecture in the study. It is evaluated with:

1. `conv`
2. `mpnn`

Unlike `base_gnn` and `multigraph`, it does not rely on a deep causal-convolutional temporal encoder. Instead, it uses stateful recurrent memory. Raw node and edge inputs are first projected at each time step. A `MemoryAugmentedGraphBlock` then maintains node memory and relation-specific edge memory across contiguous chunks.

The edge memory update uses recurrent cells conditioned on current edge state, source-node state, destination-node state, and pairwise node interactions. The node memory update aggregates relation-specific edge-memory context to nodes, fuses relation-specific node and edge contexts, and updates node memory with another recurrent cell. Training uses contiguous stateful chunks with truncated backpropagation through time.

Inside each recurrent step, the graph operator is either Conv-style or MPNN-style. The key difference from the other families is that graph interaction occurs inside a recurrent memory loop. The operator acts on state-enriched representations rather than on a fully pre-encoded temporal sequence.

This gives `memorygraph` a qualitatively different inductive bias:

1. `base_gnn` uses early relation fusion and convolutional temporal modeling.
2. `multigraph` uses late relation fusion and convolutional temporal modeling.
3. `memorygraph` uses relation-aware recurrent state and stateful graph updates.

[image needed 4.4.1] (Memorygraph recurrent node and edge memory update)

# 5. Results

This chapter reports the empirical benchmark. The main evidence is the deployment-oriented `last_CV` comparison across all eighteen primary model-frequency configurations. The chapter then discusses frequency-specific outcomes, answers the research questions, compares selected `last_CV` and `final_refit` cases, and evaluates the hypotheses.

The main interpretive rule is that `pnl_sum` is the primary economic outcome, `gross_pnl_sum` indicates pre-cost signal extraction, and `n_trades` is necessary for understanding whether the economic result is operationally meaningful. AUC values are interpreted as ranking diagnostics, not as sufficient evidence of tradability.

## 5.1. Benchmark overview

Table 5.1 reports the updated `last_CV` benchmark. Within each frequency, the six models are directly comparable because they use the same input representation, target construction, validation logic, and event-based backtest. The `5min` and `1min` regimes are also directly comparable to each other because they solve the same 30-minute lookback / 5-minute horizon task. The `1sec` regime should be interpreted as a high-frequency stress test with its own adapted task.

[image needed 5.1.1] (Eighteen-model benchmark grid by frequency, family, and operator)

| Frequency | Model | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `base-gnn-conv` | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | `base-gnn-mpnn` | 0.014415 | 0.006915 | 25 | 0.614912 | 0.727631 |
| `5min` | `multi-gnn-conv` | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| `5min` | `multi-gnn-mpnn` | 0.002941 | -0.009359 | 41 | 0.625439 | 0.707304 |
| `5min` | `memory-gnn-conv` | 0.009459 | 0.004359 | 17 | 0.611842 | 0.734196 |
| `5min` | `memory-gnn-mpnn` | -0.012463 | -0.037363 | 83 | 0.537719 | 0.726026 |
| `1min` | `base-gnn-conv` | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | `base-gnn-mpnn` | -0.024239 | -0.078539 | 181 | 0.504512 | 0.642097 |
| `1min` | `multi-gnn-conv` | 0.025767 | -0.007833 | 112 | 0.541996 | 0.634787 |
| `1min` | `multi-gnn-mpnn` | -0.003371 | -0.035771 | 108 | 0.528405 | 0.645004 |
| `1min` | `memory-gnn-conv` | -0.031247 | -0.078947 | 159 | 0.479399 | 0.638928 |
| `1min` | `memory-gnn-mpnn` | 0.033605 | 0.009305 | 81 | 0.529844 | 0.635665 |
| `1sec` | `base-gnn-conv` | 0.139515 | -0.094185 | 779 | 0.599753 | 0.841804 |
| `1sec` | `base-gnn-mpnn` | 0.052679 | -0.065821 | 395 | 0.599093 | 0.848558 |
| `1sec` | `multi-gnn-conv` | 0.081710 | -0.108790 | 635 | 0.597838 | 0.839544 |
| `1sec` | `multi-gnn-mpnn` | 0.079777 | -0.080723 | 535 | 0.600538 | 0.868370 |
| `1sec` | `memory-gnn-conv` | 0.412032 | -1.163268 | 5251 | 0.588785 | 0.490050 |
| `1sec` | `memory-gnn-mpnn` | 0.223788 | -0.280512 | 1681 | 0.596713 | 0.863699 |

Three patterns define the updated benchmark.

First, `base-gnn-conv` is the strongest model at both `5min` and `1min`. This is the most important correction relative to the old interpretation of the results. The benchmark no longer supports the claim that `base_gnn + adaptive_mpnn` is the strongest model at the shared-task frequencies. The updated evidence favors the Conv variant of the baseline family.

Second, all 1-second models remain net negative after transaction costs. Several of them produce positive gross signal, and some produce large gross signal, but none converts that signal into positive `pnl_sum`. The 1-second result is therefore not a search for a small positive winner. It is a demonstration that gross signal and net deployability separate sharply under high turnover.

Third, richer architectures do not dominate the primary economic benchmark. `multigraph` sometimes improves ranking diagnostics or gross PnL, and `memorygraph` produces the largest gross signal at 1-second frequency. Nevertheless, neither family establishes a robust post-cost advantage over the simpler `base_gnn` scaffold under the common entry-model evaluation.

## 5.2. Frequency-specific results

### 5.2.1. Five-minute regime

The `5min` regime produces the clearest economically positive block of results. The best model is `base-gnn-conv`, with `gross_pnl_sum = 0.028156`, `pnl_sum = 0.020356`, and 26 trades. The second-best model is `base-gnn-mpnn`, with `pnl_sum = 0.006915` over 25 trades. The two baseline variants therefore occupy the top two economic positions.

The more complex families remain informative but not dominant. `multi-gnn-conv` is mildly positive, with `pnl_sum = 0.001958`, while `multi-gnn-mpnn` is negative. `memory-gnn-conv` is also mildly positive, with `pnl_sum = 0.004359`, while `memory-gnn-mpnn` is the weakest 5-minute model with `pnl_sum = -0.037363` and 83 trades.

The ranking metrics show why economic interpretation cannot rely on AUC alone. `multi-gnn-mpnn` has the highest `dir_auc` in the 5-minute block (`0.625439`), and `memory-gnn-conv` has the highest `trade_auc` (`0.734196`). Neither is the best economic model. The best deployable outcome comes from the baseline Conv model, which combines a positive gross signal with a modest number of trades and limited cost drag.

### 5.2.2. One-minute regime

The `1min` regime is the richer shared-task stress test because it uses the same 30-minute lookback and 5-minute horizon as the 5-minute regime, but with more granular input information. The winner remains `base-gnn-conv`, with `gross_pnl_sum = 0.059694`, `pnl_sum = 0.020094`, and 132 trades.

This result is important because the gross signal is much larger than at 5-minute frequency, but the net result is almost identical. The reason is turnover. The 1-minute model extracts more pre-cost signal, but the larger number of trades absorbs most of the incremental edge through transaction costs. The 1-minute benchmark is therefore not stronger in strict net-profit terms, but it is stronger as a stress test of whether signal survives more active trading.

The second-best 1-minute model is `memory-gnn-mpnn`, with `pnl_sum = 0.009305` over 81 trades. This is the strongest shared-task result for `memorygraph` and suggests that recurrent memory can be useful at minute-level resolution. However, the result remains below the baseline Conv winner.

The `multigraph` family does not produce positive net PnL at 1-minute frequency. `multi-gnn-conv` has positive gross PnL (`0.025767`) but ends at `pnl_sum = -0.007833`; `multi-gnn-mpnn` is also negative. This does not show that relation-specific modeling contains no information. It shows that, under the present thresholding and cost assumptions, relation-specific information does not translate into superior net profitability.

### 5.2.3. One-second regime

The `1sec` regime creates the sharpest separation between gross signal and net deployability. All models finish negative on `pnl_sum`. The least negative model is `base-gnn-mpnn` at `pnl_sum = -0.065821`, followed by `multi-gnn-mpnn` at `-0.080723`, `base-gnn-conv` at `-0.094185`, and `multi-gnn-conv` at `-0.108790`. The memory models are substantially more negative.

The gross results tell a different story. `memory-gnn-conv` produces the largest gross signal in the entire benchmark (`gross_pnl_sum = 0.412032`), and `memory-gnn-mpnn` produces the second-largest 1-second gross signal (`0.223788`). These numbers indicate that the memory architecture is extracting high-frequency structure. However, `memory-gnn-conv` executes 5251 trades and ends at `pnl_sum = -1.163268`; `memory-gnn-mpnn` executes 1681 trades and ends at `pnl_sum = -0.280512`.

At the benchmark cost of `0.0003` per trade, the cumulative cost burden for `memory-gnn-conv` is approximately `1.5753`. This is much larger than its gross signal of `0.412032`. The model is therefore not failing because it lacks raw signal. It is failing because the signal is expressed through too many trades.

[image needed 5.2.3] (Gross versus net PnL at 1-second frequency: signal extraction overwhelmed by turnover costs)

The 1-second evidence is central to the thesis. It shows why deployment-oriented evaluation must separate ranking quality, gross signal, and net tradability. A model can be directionally informative and still economically unsuitable under realistic friction assumptions.

## 5.3. Answer to RQ1: which graph family performs best?

The answer to RQ1 is that `base_gnn` performs best overall under the controlled entry-model benchmark.

The strongest evidence comes from the two shared-task regimes. At `5min`, the best model is `base-gnn-conv` with `pnl_sum = 0.020356`. At `1min`, the best model is again `base-gnn-conv`, with `pnl_sum = 0.020094`. Both results are positive, and both are obtained by the same family and operator.

At `1sec`, no family produces positive net PnL. This means the high-frequency regime cannot be used to identify a robust deployment winner. Instead, it shows that all families face a cost and turnover barrier under the current entry policy.

The family-level conclusion is therefore conservative but clear. Under fixed targets, fixed features, fixed validation, fixed thresholds, and fixed event-based exits, the simpler single-graph baseline is the most reliable architecture. The richer families may contain useful signal, but they do not produce a stronger post-cost benchmark result.

## 5.4. Answer to RQ2: how important is the Conv-versus-MPNN choice?

The operator choice is important, but its effect is not universal.

At `5min`, Conv outperforms MPNN on `pnl_sum` in all three families:

1. `base_gnn`: `0.020356` versus `0.006915`.
2. `multigraph`: `0.001958` versus `-0.009359`.
3. `memorygraph`: `0.004359` versus `-0.037363`.

At `1min`, Conv remains better for `base_gnn` and `multigraph`, but `memorygraph` reverses in favor of MPNN:

1. `base_gnn`: `0.020094` versus `-0.078539`.
2. `multigraph`: `-0.007833` versus `-0.035771`.
3. `memorygraph`: `-0.078947` versus `0.009305`.

At `1sec`, all models are net negative, but MPNN is less negative than Conv in each family:

1. `base_gnn`: `-0.065821` versus `-0.094185`.
2. `multigraph`: `-0.080723` versus `-0.108790`.
3. `memorygraph`: `-0.280512` versus `-1.163268`.

The operator therefore changes economic outcomes materially. The strongest shared-task model is Conv-based, but the least negative 1-second models are MPNN-based. The correct conclusion is not that Conv is always better or that MPNN is always better. The operator must be selected jointly with the family scaffold, frequency regime, and cost-sensitive trading policy.

## 5.5. Answer to RQ3: how does temporal resolution affect richer relation and memory mechanisms?

The results do not support the hypothesis that finer temporal resolution automatically increases the net economic value of richer graph mechanisms.

For `multigraph`, relation-specific processing does not beat the baseline on `pnl_sum` at any frequency. At 5-minute frequency, it is mildly positive in the Conv variant but below the baseline. At 1-minute frequency, both variants are net negative. At 1-second frequency, both variants have positive gross signal but negative net PnL.

For `memorygraph`, the answer is more nuanced. The family becomes most distinctive at 1-second frequency, exactly where stateful memory was expected to matter most. Its gross results are the largest in the benchmark. This provides partial evidence that recurrent memory can surface high-frequency opportunities. However, the same results show that memory also produces excessive trading activity under the current policy. The net effect is strongly negative.

The best interpretation is therefore two-layered. Finer temporal resolution appears to increase the amount of extractable short-horizon signal, especially for memory-based models. At the same time, it increases the penalty for insufficient trade selectivity. In the current benchmark, the cost and turnover effect dominates the signal-extraction effect.

## 5.6. Answer to RQ4: are conclusions stable between `last_CV` and `final_refit`?

The deployment-state comparison shows that `last_CV` and `final_refit` are related but not interchangeable. The `last_CV` state remains the main deployment reference because it is produced by the final chronological cross-validation fold. The `final_refit` state is useful because it tests what happens when a model is refit on a larger pre-holdout sample, but it does not replace the chronological evidence.

[image needed 5.6.1] (`last_CV` versus `final_refit`: deployable chronological state versus larger-sample refit)

### 5.6.1. Best 5-minute model: `base-gnn-conv`

| Frequency | Training cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `last_CV` | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | `final_refit` | 0.017570 | 0.011270 | 21 | 0.630702 | 0.721795 |

The 5-minute winner remains positive after refitting. Its net PnL declines, but its ranking metrics improve. This is the cleanest deployment-stability case in the selected comparisons. It also shows why final refitting should not be assumed to improve the main economic metric: more training data improve AUC here, but not `pnl_sum`.

### 5.6.2. Best 1-minute model: `base-gnn-conv`

| Frequency | Training cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `1min` | `last_CV` | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | `final_refit` | 0.007198 | -0.012002 | 64 | 0.524286 | 0.635712 |

The 1-minute winner is less stable. The `last_CV` model is clearly positive, while the `final_refit` version turns negative. The AUC values change only modestly, which suggests that the underlying ranking quality remains similar while the score-to-trade conversion becomes less economically favorable. This case reinforces the deployment argument: a model can look similar in predictive diagnostics but materially different in realized trading performance.

### 5.6.3. Selected 1-second memorygraph case: `memory-gnn-conv`

| Frequency | Training cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `1sec` | `last_CV` | 0.412032 | -1.163268 | 5251 | 0.588785 | 0.490050 |
| `1sec` | `final_refit` | 0.443031 | -0.954969 | 4660 | 0.592186 | 0.852874 |

The selected 1-second case is the most informative high-frequency stress example. Refitting increases gross PnL, reduces the trade count, improves `trade_auc`, and makes the net result less negative. Nevertheless, the model remains strongly unprofitable after costs. The central 1-second conclusion therefore survives refitting: memory-based high-frequency signal is present, but it is not sufficiently selective under the current benchmark.

### 5.6.4. Interesting 5-minute refit case: `multi-gnn-conv`

| Frequency | Training cycle | Gross pnl sum | pnl sum | N trades | dir auc | trade auc |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| `5min` | `last_CV` | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| `5min` | `final_refit` | 0.061167 | 0.041367 | 66 | 0.703947 | 0.720920 |

This case is analytically important because the `final_refit` version of `multi-gnn-conv` reaches `dir_auc > 70%` and a strongly positive `pnl_sum`. It shows that relation-preserving graph processing can become highly effective under a larger-sample refit. However, it does not overturn the primary `last_CV` conclusion. In the deployment-oriented state, `multi-gnn-conv` is only weakly positive and remains below `base-gnn-conv`. The case should therefore be interpreted as evidence of refit sensitivity and potential future value, not as proof that `multigraph` is already the strongest deployable family.

## 5.7. Hypothesis assessment

### H1. The 1-minute regime should be the strongest shared-task benchmark.

H1 is not supported on the primary economic metric. The strongest 1-minute model reaches `pnl_sum = 0.020094`, while the strongest 5-minute model reaches `pnl_sum = 0.020356`. The difference is small, but the hypothesis predicts 1-minute superiority, which is not observed. The 1-minute regime remains important because it produces more trades and much larger gross signal, but it is not the strongest shared-task regime in strict net terms.

### H2. Explicit multi-relation modeling should outperform the simpler baseline more clearly at finer resolutions.

H2 is not supported on the primary economic benchmark. `multigraph` does not beat `base_gnn` on `pnl_sum` at 5-minute, 1-minute, or 1-second frequency. The interesting `multi-gnn-conv` final-refit case provides evidence that relation-specific modeling can become strong under some training states, but the deployment-oriented `last_CV` benchmark does not support a general multigraph advantage.

### H3. Stateful memory should become more valuable as the market is observed more finely.

H3 is partially supported for gross signal extraction but not supported for net deployable performance. At 1-second frequency, `memorygraph` produces the strongest gross PnL values in the study. However, both memory variants remain net negative, and `memory-gnn-conv` is especially negative because of excessive turnover. Memory helps reveal short-lived signal, but the current benchmark does not show that it improves post-cost profitability.

### H4. Conv and MPNN operators should not be uniformly dominant across families.

H4 is supported. Conv dominates the 5-minute net results and produces the strongest shared-task model overall. At 1-minute frequency, Conv remains better for `base_gnn` and `multigraph`, while MPNN is better for `memorygraph`. At 1-second frequency, MPNN is less negative in all families on net PnL. The operator effect is therefore economically meaningful and frequency-dependent.

### H5. `last_CV` and `final_refit` should tell a broadly consistent family-level story.

H5 is partially supported. The selected comparisons do not overturn the broad conclusion that `base_gnn` is the most reliable deployment-oriented family and that 1-second models remain net negative. However, the model-level story can change materially. The 1-minute `base-gnn-conv` turns negative after refitting, while the 5-minute `multi-gnn-conv` becomes very strong after refitting. The evidence therefore supports reporting both states transparently and prioritizing `last_CV` for deployment interpretation.

# 6. Discussion

## 6.1. Main findings

The main answer to RQ1 is that `base_gnn` is the strongest family under the controlled entry-model benchmark. The updated results show that `base-gnn-conv` is the best model at both shared-task frequencies. This finding is important because it contradicts a simple complexity-based expectation. More complex graph processing does not automatically improve deployable trading performance.

The answer to RQ2 is that the Conv-versus-MPNN choice matters substantially. However, the preferred operator depends on frequency and family. Conv is strongest in the shared-task winners, while MPNN is less negative at 1-second frequency. The operator should therefore be treated as a substantive modeling decision rather than a minor implementation detail.

The answer to RQ3 is that finer temporal resolution increases both opportunity and risk. The 1-second memory models reveal strong gross signal, but they also generate excessive turnover. The thesis therefore separates signal extraction from deployable profitability: richer temporal mechanisms can reveal more opportunities without producing a better trading system.

The answer to RQ4 is that deployment state matters. `last_CV` is the main practical reference because it respects chronological model selection. `final_refit` is useful, but it can improve, weaken, or qualitatively alter economic outcomes. A thesis that reported only refit performance would give an incomplete and potentially misleading deployment interpretation.

## 6.2. Comparison with previous work

The findings are broadly consistent with earlier limit order book research in one respect: deep models can extract meaningful structure from order book data. DeepLOB demonstrates that convolutional and recurrent neural networks can forecast short-term price movements from LOB states [2]. Sirignano and Cont show that deep learning can identify stable price-formation patterns across large equity universes [3]. The present thesis supports the premise that microstructure data contain learnable signal. The positive gross PnL values, especially at 1-minute and 1-second frequency, would be difficult to explain if no signal were present.

At the same time, the thesis is more cautious about economic interpretation. Much of the predictive modeling literature emphasizes accuracy, AUC, or return prediction. The updated results show why such metrics are insufficient for deployment. The 1-second memory models extract the largest gross signal but fail after costs. This finding aligns with the broader financial machine learning principle that backtests must account for chronological validation, transaction costs, and selection effects [15].

The graph-learning literature motivates the use of relational representations. Kipf and Welling provide a scalable graph convolution framework [4], while Gilmer et al. formalize message passing as a general neural architecture [5]. Wu et al. show that graph learning can improve multivariate time-series forecasting by capturing dependencies among variables [6]. The present thesis applies the same general idea to cross-asset market microstructure, but its results are more conservative than a graph-optimistic interpretation would suggest. Relation-aware modeling improves some diagnostics and produces an interesting 5-minute final-refit case, but it does not dominate the deployment benchmark.

The temporal graph literature also helps interpret the `memorygraph` results. Temporal Graph Networks use memory modules and graph operators for dynamic graphs [7], while EvolveGCN adapts graph convolution parameters through time [8]. These works suggest that dynamic graph mechanisms should be useful when relations evolve. The thesis finds partial support for this idea at the gross-signal level: memory is most distinctive at 1-second frequency. However, the trading benchmark reveals an additional constraint that is less prominent in generic dynamic-graph tasks: frequent state updates can amplify turnover, and turnover can destroy net profitability.

Recent financial graph work such as MDGNN studies multi-relational dynamic graph neural networks for stock investment prediction [10]. The present thesis is related but not equivalent. MDGNN works in a broader stock-investment setting, while this thesis studies short-horizon crypto limit order book prediction with event-based trading evaluation. The difference matters. A graph architecture that is useful for daily or lower-frequency investment prediction may not automatically remain superior at minute or second frequency, where transaction costs and threshold calibration dominate the realized outcome.

Overall, the thesis contributes a cautionary empirical result to the literature. It does not reject graph neural networks for finance. Instead, it shows that, in high-frequency microstructure prediction, architectural sophistication must be evaluated through the full chain from ranking quality to gross signal to turnover and net PnL.

## 6.3. Implications

The scientific implication is that controlled comparisons are essential. Without common targets, common features, common validation, and common exits, it is difficult to know whether a graph model is genuinely better or merely evaluated under more favorable conditions. The thesis shows that the simplest family can win when confounding procedural differences are removed.

The practical implication is that deployment-oriented model selection should prioritize post-cost performance and stability. A model with attractive AUC may still be unsuitable if it triggers too many trades. Conversely, a model with modest AUC can be economically useful if it is selective and cost-aware. For trading applications, ranking quality, gross PnL, trade count, and net PnL must be interpreted together.

The organizational implication is that financial machine learning teams should avoid treating final refit performance as the sole decision criterion. A final refit model may be useful before deployment, but its behavior should be compared against walk-forward evidence. The `last_CV` state is closer to how a model would be selected under a live chronological process.

The broader positive consequence of this research is methodological discipline. A controlled benchmark can prevent overclaiming and can identify where future work should focus. The broader negative consequence is that even academically promising models can encourage excessive trading if they are optimized for signal extraction without sufficient attention to costs, thresholds, and execution constraints.

[image needed 6.3.1] (From predictive model to deployable trading system: where signal can fail)

## 6.4. Limitations, weaknesses, and bias

The study has several limitations.

First, the sample is limited. The benchmark uses a fixed three-asset crypto universe: ADA, BTC, and ETH. This keeps the graph interpretable but restricts generalization. The results may not transfer to larger crypto universes, equities, futures, foreign exchange, or less liquid assets.

Second, the data source may introduce exchange and collection bias. The dataset is a public Kaggle dataset rather than a proprietary exchange feed with full message-level detail, queue position, order cancellations, latency, and execution reports. The order book snapshots are suitable for the thesis benchmark, but they are not equivalent to the information available to a colocated production trading system.

Third, the temporal sample may contain market-regime bias. The final holdout is a specific late segment of the available period. If this interval has unusual volatility, liquidity, or directional behavior, the measured ranking of architectures may reflect that regime. A broader study would repeat the benchmark across multiple calendar periods and market states.

Fourth, the label construction may introduce bias. Triple-barrier labels depend on barrier widths, volatility scaling, timeout treatment, and the decision to mask some direction labels. These choices are defensible, but they shape what the model learns. A different barrier system could change the apparent value of memory or relation-specific processing.

Fifth, threshold-selection bias is possible. Trade and direction thresholds are selected from finite grids on validation data. This is more disciplined than choosing thresholds on the final holdout, but it still means that realized trading behavior depends on a particular calibration procedure. The strong divergence between gross and net results at 1-second frequency suggests that threshold design is one of the main unresolved issues.

Sixth, the transaction-cost model is simplified. The benchmark uses a fixed cost proxy. It does not model exchange fees by venue and account tier, bid-ask queue priority, partial fills, market impact, latency, adverse selection, or slippage during volatile intervals. This limitation matters most at 1-second frequency, where costs dominate the results.

Seventh, the model family set is restricted. The thesis compares three graph families and two graph-operator styles. It does not test transformer-based LOB models, hybrid attention-GNN architectures, reinforcement-learning exits, probabilistic calibration layers, or explicitly turnover-regularized objectives. The conclusion is therefore conditional on the tested family set.

Eighth, hyperparameter search is limited. The benchmark is designed for fair comparison, not exhaustive optimization. Some architectures, especially `multigraph` and `memorygraph`, might improve under broader tuning. The 5-minute `multi-gnn-conv` final-refit case suggests that richer models may be sensitive to training state and calibration.

Ninth, deployment remains conceptual rather than production-complete. The backtest is sequential and event-based, but it is not a live execution system. It does not include live data ingestion, order placement, monitoring, risk limits, capital allocation, latency measurement, or operational failure handling. The thesis can support deployment-oriented conclusions, but not production-profit claims.

Finally, several forms of bias are relevant even though demographic bias is not central to this application. Market-regime bias can occur if the sample overrepresents a particular volatility state. Asset-selection bias can occur because ETH is the target and only ADA and BTC provide context. Exchange/data-source bias can occur if the order book snapshots do not represent broader market liquidity. Label-construction bias can occur through barrier and timeout rules. Cost-model bias can occur through simplified transaction costs. Temporal-slice bias can occur through reliance on a single final holdout period. Model-selection bias can occur if repeated experimentation implicitly adapts to the final benchmark. These biases do not invalidate the thesis, but they define the boundaries of its claims.

# 7. Conclusions and Future Research

## 7.1. Conclusion

This thesis asked whether graph-based architectures improve limit order book prediction under a controlled, deployment-oriented benchmark. The updated empirical answer is clear: the strongest evidence favors the simpler `base_gnn` family, specifically `base-gnn-conv`, at both shared-task frequencies.

At 5-minute frequency, `base-gnn-conv` achieves the best `last_CV` net result with `pnl_sum = 0.020356`. At 1-minute frequency, `base-gnn-conv` again achieves the best result with `pnl_sum = 0.020094`. At 1-second frequency, all models are net negative after transaction costs. The core high-frequency finding is therefore the divergence between gross signal and net deployability. Memory-based models identify substantial gross signal, but the trading policy converts that signal into too many trades.

The thesis does not show that graph neural networks are ineffective for market microstructure. It shows something more precise: under a fair entry-model benchmark, additional relation-specific processing and recurrent memory do not automatically produce better post-cost trading performance. The strongest architecture is the one that best balances signal extraction, selectivity, and turnover.

The deployment-state analysis reinforces this conclusion. `last_CV` should remain the primary deployment reference because it respects chronological model selection. `final_refit` is useful but can alter economic outcomes in both directions. The 5-minute baseline refit remains positive, the 1-minute baseline refit turns negative, and the 5-minute `multi-gnn-conv` refit becomes very strong. These cases show why both states should be reported transparently.

The final thesis conclusion is therefore disciplined rather than promotional: the best current evidence supports `base-gnn-conv` as the most reliable model in the controlled benchmark, while the main research challenge is to make richer short-horizon signals selective enough to survive transaction costs.

## 7.2. Future research

The first direction for future research is turnover-aware modeling. The 1-second experiments show that signal extraction is not enough. Future models should include stricter trade gating, explicit no-trade calibration, cost-aware threshold optimization, and objectives that penalize excessive turnover directly.

The second direction is execution-aware evaluation. The present benchmark fixes the realized-event exit rule for fairness. A natural extension is to evaluate the strongest entry models under adaptive exit policies, latency-aware execution, slippage models, partial-fill assumptions, and venue-specific fee schedules.

The third direction is broader robustness analysis. The benchmark should be repeated across multiple market regimes, longer calendar samples, and additional final holdout periods. This would help distinguish architecture effects from temporal-slice effects.

The fourth direction is a larger relational universe. A three-node graph is interpretable but small. Adding more crypto assets, stablecoins, sector proxies, derivatives, or cross-venue liquidity measures would create a stronger test of whether `multigraph` becomes more valuable when the relation space is richer.

The fifth direction is selective memory. The current memory model appears capable of finding high-frequency gross signal but not of controlling trade frequency. Future work should examine sparse memory updates, event-triggered memory writes, confidence-aware state resets, and memory mechanisms coupled to explicit trade-rate constraints.

The sixth direction is uncertainty quantification. Future versions of the benchmark should report fold-level dispersion, bootstrap confidence intervals, pairwise model-comparison tests, drawdown statistics, and cost-sensitivity curves. These additions would strengthen the statistical interpretation of the architecture ranking.

[image needed 7.2.1] (Future research roadmap: turnover-aware learning, execution-aware testing, larger graphs, selective memory, robustness)

# References and Working Source List

[1] Martinsn. *High-Frequency Crypto Limit Order Book Data*. Kaggle dataset. Link: https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data/data

[2] Zhang, Z., Zohren, S., and Roberts, S. "DeepLOB: Deep Convolutional Neural Networks for Limit Order Books." *IEEE Transactions on Signal Processing*, 67(11), 3001-3012, 2019. PDF: https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID3519855_code3934917.pdf?abstractid=3519855

[3] Sirignano, J., and Cont, R. "Universal Features of Price Formation in Financial Markets: Perspectives from Deep Learning." *Quantitative Finance*, 19(9), 1449-1459, 2019. Preprint/PDF: https://arxiv.org/pdf/1803.06917

[4] Kipf, T. N., and Welling, M. "Semi-Supervised Classification with Graph Convolutional Networks." *International Conference on Learning Representations (ICLR)*, 2017. PDF: https://arxiv.org/pdf/1609.02907

[5] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., and Dahl, G. E. "Neural Message Passing for Quantum Chemistry." *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1263-1272, 2017. PDF: https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf

[6] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., and Zhang, C. "Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks." *Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD)*, 753-763, 2020. PDF: https://arxiv.org/pdf/2005.11650

[7] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., and Bronstein, M. "Temporal Graph Networks for Deep Learning on Dynamic Graphs." arXiv preprint, 2020. PDF: https://arxiv.org/pdf/2006.10637

[8] Pareja, A., Domeniconi, G., Chen, J., Ma, T., Suzumura, T., Kanezashi, H., Kaler, T., Schardl, T. B., and Leiserson, C. E. "EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs." *Proceedings of the AAAI Conference on Artificial Intelligence*, 34(04), 5363-5370, 2020. PDF: https://arxiv.org/pdf/1902.10191

[9] Wang, J., Zhang, S., Xiao, Y., and Song, R. "A Review on Graph Neural Network Methods in Financial Applications." arXiv preprint, 2021. PDF: https://arxiv.org/pdf/2111.15367

[10] Qian, H., Zhou, H., Zhao, Q., Chen, H., Yao, H., Wang, J., Liu, Z., Yu, F., Zhang, Z., and Zhou, J. "MDGNN: Multi-Relational Dynamic Graph Neural Network for Comprehensive and Dynamic Stock Investment Prediction." *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(13), 14642-14650, 2024. Link: https://ojs.aaai.org/index.php/AAAI/article/view/29381

[11] Khemani, B., Patil, S., Kotecha, K., and Tanwar, S. "A Review of Graph Neural Networks: Concepts, Architectures, Techniques, Challenges, Datasets, Applications, and Future Directions." *Journal of Big Data*, 11, Article 18, 2024. PDF: https://link.springer.com/content/pdf/10.1186/s40537-023-00876-4.pdf

[12] Liu, Y., Liu, Q., Zhang, J. W., Feng, H., Wang, Z., Zhou, Z., and Chen, W. "Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 19414-19426, 2022. PDF: https://proceedings.neurips.cc/paper_files/paper/2022/file/7b102c908e9404dd040599c65db4ce3e-Paper-Conference.pdf

[13] Jin, M., Zheng, Y., Li, Y.-F., Chen, S., Yang, B., and Pan, S. "Multivariate Time Series Forecasting with Dynamic Graph Neural ODEs." *IEEE Transactions on Knowledge and Data Engineering*, 35(9), 9168-9180, 2023. Link: https://ieeexplore.ieee.org/document/9950330

[14] Ortu, M., Uras, N., Conversano, C., Bartolucci, S., and Destefanis, G. "On Technical Trading and Social Media Indicators for Cryptocurrency Price Classification through Deep Learning." *Expert Systems with Applications*, 198, 116804, 2022. Link: https://doi.org/10.1016/j.eswa.2022.116804

[15] Lopez de Prado, M. *Advances in Financial Machine Learning*. Wiley, 2018. Publisher link: https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

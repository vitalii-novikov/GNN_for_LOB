# 2. Research Questions and Hypotheses

This thesis studies graph-based market microstructure models under a deliberately controlled apples-to-apples protocol. The objective is not to compare loosely related trading systems that differ in targets, exits, or validation logic, but to compare several graph families under a common supervised task, a common feature space, a common multi-task objective, and a common event-based evaluation protocol. The empirical focus is therefore architectural rather than procedural.

The comparison is organized along two orthogonal axes. The first axis is the model family: `base_gnn`, `multigraph`, and `memorygraph`. The second axis is the graph interaction mechanism: a convolution-style operator (Conv) and a message-passing neural network operator (MPNN). Consequently, each family is evaluated in two matched variants, producing six models per frequency regime:

1. `base_gnn + adaptive_conv`
2. `base_gnn + adaptive_mpnn`
3. `multigraph + dynamic_rel_conv`
4. `multigraph + dynamic_edge_mpnn`
5. `memorygraph + conv`
6. `memorygraph + mpnn`

These six models are evaluated at three temporal resolutions:

1. `5min`, with a 30-minute lookback and a 5-minute horizon
2. `1min`, with a 30-minute lookback and a 5-minute horizon
3. `1sec`, with a 2-minute lookback and a 2-minute horizon

The study therefore contains eighteen experimental configurations. Within each frequency regime, the comparison is apples-to-apples because the input representation, target construction, thresholding logic, and final trading benchmark are fixed, while only the family scaffold and the Conv-versus-MPNN operator vary.

## 2.1. Research Questions

### RQ1. Which graph family performs best under a controlled entry-model benchmark?

The first question asks which of the three graph families performs best when they are evaluated under the same entry-model task. This is the core question of the thesis, because it isolates the value of the architectural scaffold itself rather than the value of family-specific execution rules.

### RQ2. How important is the Conv-versus-MPNN operator choice inside each family?

Each family is instantiated with two graph interaction mechanisms. This makes it possible to ask whether performance differences are primarily explained by the broader family scaffold or by the local graph operator. The answer is important because the same comparison is repeated inside all three families, which creates a structurally consistent operator study rather than a one-off ablation.

### RQ3. How does temporal resolution change the relative value of relational and memory mechanisms?

The 5-minute and 1-minute experiments solve the same clock-time prediction task, whereas the 1-second experiments use a shorter task that remains realistic at ultra-high frequency. This design makes it possible to study whether richer relational modeling and persistent memory become more useful as temporal resolution increases.

### RQ4. Are the conclusions stable under deployment-oriented model states?

All models are evaluated not only in terms of the statistically best cross-validation checkpoint, but also in terms of the `last_CV` model, which serves as the primary deployable reference, and the `final_refit` model, which serves as a larger-sample upper bound. This leads to a practical question: do the same model families remain preferable when evaluated in a deployment-oriented setting rather than under idealized refitting?

## 2.2. Hypotheses

### H1. The 1-minute regime should be the strongest shared-task benchmark.

Under the common 30-minute lookback and 5-minute horizon task, the 1-minute regime is expected to outperform the 5-minute regime on the main comparison metrics. The reason is that 1-minute data preserve substantially more intra-horizon dynamics than 5-minute aggregation while remaining less noisy than second-level data.

### H2. Explicit multi-relation modeling should outperform the simpler baseline more clearly at finer resolutions.

The `multigraph` family is expected to deliver larger gains over `base_gnn` at 1-minute than at 5-minute frequency, and potentially stronger gains still at 1-second frequency, because richer relation-specific modeling should matter more when short-horizon cross-asset dependencies evolve rapidly.

### H3. Stateful memory should become more valuable as the market is observed more finely.

The `memorygraph` family is expected to be most competitive at 1-second frequency, moderately competitive at 1-minute frequency, and least differentiated at 5-minute frequency. This hypothesis follows directly from the architecture: `memorygraph` replaces the purely convolutional temporal encoding of the other families with a stateful recurrent memory mechanism, which should be most useful when signals are transient and regime adaptation must be fast.

### H4. Conv and MPNN operators should not be uniformly dominant across families.

The Conv-versus-MPNN comparison is expected to be family-dependent rather than universal. In particular, an MPNN operator may be more beneficial when node-to-node interactions require richer conditioning on source, destination, and edge states, whereas a Conv-style operator may be more stable when the edge structure is already strongly regularized by the input representation.

### H5. `last_CV` and `final_refit` should tell a consistent family-level story.

Although `final_refit` is expected to provide a somewhat more optimistic estimate than `last_CV`, the broad ranking of the model families should remain similar across the two states. A family that only looks attractive under full refitting, but not under the realistic `last_CV` benchmark, would be much less convincing from a deployment perspective.


# 3. Methodology

## 3.1. Data source and study universe

The raw data source is the public Kaggle dataset *High-Frequency Crypto Limit Order Book Data* by Martinsn, which provides frequency-specific crypto limit order book snapshots for multiple assets, including ADA, BTC, and ETH, at `1sec`, `1min`, and `5min` resolutions, with order book information organized by depth level rather than as raw message streams ([dataset page](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data/data)).

The present study uses a fixed three-node asset universe:

1. ADA
2. BTC
3. ETH

ETH is treated as the target asset, while ADA and BTC provide relational context. Because the source data were already distributed in frequency-specific, level-organized tables, no bespoke order-book reconstruction, message-level aggregation, or manual frequency conversion was required. The preprocessing burden was therefore intentionally limited to timestamp standardization, multi-asset alignment on a common clock, and the derivation of secondary microstructure and cross-asset dependence features from the already available order book summaries.

In the locally stored CSV files used by the pipelines, each asset-frequency table contains midpoint price, spread, buy and sell flow summaries, and 15 bid-side and 15 ask-side depth values. These fields form the basis of the node features and relation features used by all three model families.

## 3.2. Input representation and derived features

All experiments use the same graph input representation within a given frequency regime. The graph is a directed complete graph over the three assets, augmented with self-loops. As a result, the set of nodes is fixed, but node states and edge states vary over time.

### 3.2.1. Node features

For each asset and each time step, the node feature block is constructed from the aligned order book table. The implemented node features are:

1. one-bar log return
2. relative spread
3. log-transformed buys
4. log-transformed sells
5. flow imbalance
6. total depth imbalance
7. top-level depth imbalances for the first five book levels
8. bid near/far depth ratio
9. ask near/far depth ratio
10. near-depth imbalance
11. far-depth imbalance

These features summarize three central aspects of market microstructure: local price change, order-flow pressure, and the shape of the bid-ask depth profile.

### 3.2.2. Relation states and edge features

To construct graph edges, the pipelines first build three relation-state series per asset:

1. `price_dep`, represented by the asset log return
2. `order_flow`, represented by flow imbalance scaled by log turnover
3. `liquidity`, represented by a spread-and-depth composite based on relative spread, total depth imbalance, near-depth imbalance, and a bounded near/far depth shape ratio

For every ordered asset pair and every relation channel, rolling dependence features are then computed over several lagged windows. The edge tensor contains, for each lag-window combination:

1. rolling correlation
2. rolling beta
3. rolling mean product

When configured, correlations are Fisher-\(z\) transformed before scaling. This means that all model families operate on the same relation-aware edge representation rather than on family-specific graph features.

### 3.2.3. Scaling and leakage control

Node and edge tensors are robustly scaled on training data only, using fold-specific quantile statistics, and then clipped to bounded ranges before model fitting. This procedure is shared across the graph families and prevents the architectural comparison from being distorted by inconsistent feature scaling or train-test leakage.

## 3.3. Frequency-specific experimental regimes

The experimental design consists of eighteen runs: six model variants for each of the three frequency regimes. For every frequency, the same six graph-model instances are evaluated:

1. `base_gnn + adaptive_conv`
2. `base_gnn + adaptive_mpnn`
3. `multigraph + dynamic_rel_conv`
4. `multigraph + dynamic_edge_mpnn`
5. `memorygraph + conv`
6. `memorygraph + mpnn`

This means that every graph family is explicitly tested with both a Conv-style operator and an MPNN-style operator.

### 3.3.1. Shared-task benchmark at 5-minute and 1-minute frequency

The `5min` and `1min` regimes solve the same clock-time task:

1. lookback window = 30 minutes
2. forecast horizon = 5 minutes

This corresponds to:

1. `5min`: 6 lookback bars and 1 horizon bar
2. `1min`: 30 lookback bars and 5 horizon bars

For both regimes, the working sample is taken from the first 90% of the full series (`data_slice_start_frac = 0.0`, `data_slice_end_frac = 0.9`), and the final 10% of that working sample is reserved as the blind final holdout (`final_holdout_frac = 0.1`).

### 3.3.2. Adapted high-frequency regime at 1-second frequency

The `1sec` regime uses a shorter task:

1. lookback window = 2 minutes = 120 bars
2. forecast horizon = 2 minutes = 120 bars

The working sample is intentionally restricted to the interval `0.5-0.9` of the full second-level series, which corresponds to 40% of all available observations. This reduction is deliberate. It keeps training computationally feasible while preserving a meaningful ultra-high-frequency comparison.

The final holdout fraction is increased to `0.225` so that the final holdout interval in calendar time aligns as closely as possible with the final holdout interval used in the `1min` and `5min` experiments. In other words, the 1-second study uses less data for development, but its blind evaluation interval is intentionally matched to the same late-period market segment as the slower-frequency experiments.

Table 3.1 summarizes the resulting task design.

| Frequency | Working data slice | Final holdout fraction | Lookback | Horizon |
|---|---|---:|---|---|
| `5min` | `0.0-0.9` of the full series | `0.10` | 30 min = 6 bars | 5 min = 1 bar |
| `1min` | `0.0-0.9` of the full series | `0.10` | 30 min = 30 bars | 5 min = 5 bars |
| `1sec` | `0.5-0.9` of the full series | `0.225` | 2 min = 120 bars | 2 min = 120 bars |

The consequence is methodologically important. The `5min` and `1min` regimes form a strict shared-task benchmark. The `1sec` regime remains apples-to-apples within its own frequency, but it should be interpreted as a frequency-adapted extension rather than as a perfectly symmetric continuation of the 30-minute/5-minute benchmark.

## 3.4. Target construction and shared learning objective

All three families are trained under the same multi-task triple-barrier framework. For each valid timestamp \(t\), the future path of the ETH midpoint is followed until one of three mutually exclusive event types occurs:

1. the upper barrier is touched
2. the lower barrier is touched
3. the vertical barrier is reached

The barrier system is volatility-scaled. In the default benchmark configuration, the upper and lower barriers start from 8 basis points, are rescaled using rolling volatility estimated over a 30-bar lookback, are multiplied by 1.8, and are clipped to the interval from 4 to 30 basis points. The vertical barrier is set equal to the prediction horizon.

From this future path, the pipelines construct a common target set:

1. realized return
2. trade relevance label
3. direction label
4. exit-type label
5. time-to-exit label

The trade label is meta-labeled and depends on whether the future move remains economically meaningful after a friction-aware threshold is applied. Direction labels are masked when the configuration declares timeout outcomes to be uninformative for directional supervision.

The crucial methodological point is that all model families share the same target set and the same output interface. The multi-task objective combines trade classification, direction classification, return regression, utility-based supervision, exit-type classification, and time-to-exit regression. In the default benchmark configuration, the loss weights are:

1. `loss_w_trade = 0.35`
2. `loss_w_dir = 0.65`
3. `loss_w_ret = 0.15`
4. `loss_w_utility = 0.85`
5. `loss_w_exit_type = 0.05`
6. `loss_w_tte = 0.03`

Additional penalties for false positives, timeout-sensitive behavior, and execution cost are shared at the configuration level. This design preserves comparability: the models differ in how they encode temporal and graph structure, not in what they are asked to predict.

## 3.5. Common entry-model benchmark

To ensure apples-to-apples trading evaluation, the benchmark is formulated as an entry-model comparison for all graph families. In the primary backtest:

1. the trade head determines whether a trade candidate is active
2. the direction head determines whether the candidate becomes a long or short position
3. the exit is generated by the same realized event rule for all families

Exit-type and time-to-exit heads are therefore retained as auxiliary learning targets and diagnostic outputs, but they do not define a family-specific trade-closing policy in the main benchmark. This choice is especially important for `memorygraph`, whose architecture is capable of representing persistent state and could otherwise be evaluated under a different execution policy than the other families.

## 3.6. Validation design and deployment-oriented model states

The experiments use purged walk-forward validation. Each working sample is divided into:

1. a pre-holdout region used for model development
2. a final holdout region used only for the final blind evaluation

Within the pre-holdout region, the walk-forward folds follow a chronological train-gap-validation-gap-test structure. The purge gaps are necessary because triple-barrier labels depend on future price evolution; adjacent observations would otherwise have overlapping future windows and could leak information across split boundaries.

For every operator and every family, the study distinguishes three model states:

1. `best_CV`: the strongest selected cross-validation model
2. `last_CV`: the model from the last walk-forward fold, treated as the primary deployable reference
3. `final_refit`: the model refit on the largest possible pre-holdout sample, treated as an optimistic upper bound

All three states are evaluated on the same final holdout interval. This distinction is one of the strong methodological features of the study because it prevents the final conclusions from depending exclusively on the most optimistic training scenario.

## 3.7. Backtesting logic and performance metrics

The trading evaluation uses a sequential non-overlapping event-based backtest. Once a position is opened, no new position can be opened until the current one is closed. This makes the benchmark conservative and keeps turnover interpretable.

For each executed trade \(i\), the implementation computes

\[
\text{gross\_pnl}_i = s_i \cdot r_i,
\]

where \(s_i \in \{-1, +1\}\) is the trade side and \(r_i\) is the realized log return up to the realized event exit. Net PnL is then computed as

\[
\text{net\_pnl}_i = \text{gross\_pnl}_i - c_{\text{rt}},
\]

where the round-trip transaction-cost proxy is

\[
c_{\text{rt}} = 3 \times \text{cost\_bps\_per\_side} \times 10^{-4}.
\]

With the default configuration `cost_bps_per_side = 1.0`, this yields a per-trade round-trip cost of

\[
c_{\text{rt}} = 0.0003
\]

in log-return units.

Accordingly:

1. `gross_pnl_sum` is the sum of pre-cost directional trade returns
2. `pnl_sum` is the sum of post-cost trade returns

This distinction is central to the interpretation of the results because several models can generate attractive gross trading signals while still failing after transaction costs are deducted.

The metrics emphasized most strongly in the thesis are the `last_CV` values of:

1. `gross_pnl_sum`
2. `pnl_sum`
3. `sign_accuracy`
4. `sharpe_like`
5. `dir_auc`
6. `trade_auc`

These metrics jointly capture economic usefulness, directional quality, trade-selection quality, and the turnover-adjusted stability of the realized trade sequence. In the interpretation of final results, `pnl_sum` is treated as the primary economic criterion, while `gross_pnl_sum` is used to separate raw signal quality from the effect of trading frictions.

## 3.8. Fair-comparison principle

The governing principle of the experimental design is that, within each frequency regime, only two aspects are allowed to vary:

1. the family scaffold (`base_gnn`, `multigraph`, `memorygraph`)
2. the local graph operator (Conv or MPNN)

The following components are held fixed within a regime:

1. asset universe and target asset
2. node-feature construction
3. relation-state construction
4. edge-feature construction
5. label construction
6. multi-task output interface
7. thresholding logic
8. event-based backtest
9. split protocol
10. final holdout interval

This is what makes the comparison apples-to-apples. The only principled asymmetry is the deliberate task adaptation at `1sec`, which changes the horizon, lookback, and working sample in order to create a realistic ultra-high-frequency benchmark rather than an impractically long second-level task.


# 4. Detailed Description of the Tested Models

## 4.1. Shared architectural conventions

All three graph families operate on the same pair of tensors:

1. a node sequence \(X^{(n)} \in \mathbb{R}^{B \times L \times N \times F_n}\)
2. a relation-aware edge sequence \(X^{(e)} \in \mathbb{R}^{B \times L \times R \times E \times F_e}\)

where \(B\) is batch size, \(L\) is sequence length, \(N=3\) is the number of assets, \(R=3\) is the number of relation channels (`price_dep`, `order_flow`, `liquidity`), and \(E\) is the number of directed edges including self-loops.

All three families also share the same output heads:

1. `trade_logit`
2. `dir_logit`
3. `return_pred`
4. `exit_type_logit`
5. `tte_pred`

In addition, all families use a hybrid edge-fusion block that augments handcrafted relation features with a learnable pairwise node-interaction path. The crucial architectural differences therefore lie in the temporal backbone and in how the graph operator is applied.

The temporal component is not identical across the three families:

1. `base_gnn` and `multigraph` use a causal convolutional temporal stack
2. `memorygraph` uses a stateful recurrent memory mechanism with truncated backpropagation through time

This distinction is central to the thesis and must be kept explicit.

## 4.2. The `base_gnn` family

The `base_gnn` family serves as the single-graph baseline. In the benchmark it is evaluated through two adaptive operators: `adaptive_conv` and `adaptive_mpnn`.

### 4.2.1. Temporal component

The temporal backbone of `base_gnn` is fully convolutional.

At the node level, the model applies:

1. a linear projection from raw node features to `node_hidden_dim`
2. a learned asset embedding added to every node
3. a stack of dilated causal residual convolution blocks

Each `CausalConv1dBlock` contains:

1. a causal dilated `Conv1d`
2. a `GELU` activation
3. dropout
4. a `1x1` projection
5. a residual connection
6. layer normalization

With the default configuration, the node encoder uses three such blocks with exponentially increasing dilation factors, which enlarges the temporal receptive field without using recurrent state.

The edge pathway is analogous. A dedicated `EdgeTemporalEncoder` first linearly projects raw edge features and then processes them with its own stack of dilated causal convolution blocks. This means that both node histories and relation-edge histories are temporally encoded before graph message passing begins.

After graph processing and readout, the model applies a second temporal block, the `TargetTemporalTrunk`, to the target-centered readout sequence. This trunk is again a stack of causal convolution blocks. The last temporal state is then passed through a post-projection MLP before the prediction heads are evaluated.

In summary, the temporal component of `base_gnn` is a three-stage causal-convolutional system:

1. node temporal encoder
2. edge temporal encoder
3. target temporal trunk

### 4.2.2. Graph component

The graph component is implemented through the `SingleGraphOperatorBlock`. Before message passing, the model fuses handcrafted relation-edge features with a learnable pairwise node-interaction path using `HybridEdgeFeatureFusion`. It then collapses the relation axis through `EdgeRelationFusion`, which can use either attention-based or mean fusion.

The resulting single edge representation is passed to a graph block that contains two sub-components:

1. an adjacency factory
2. a stack of graph operator layers

The adjacency factory can produce static, prior-based, or adaptive adjacency matrices. In the experiments used in this thesis, `adaptive_conv` and `adaptive_mpnn` both use the adaptive adjacency mode, which combines edge-conditioned scores with low-rank learnable source and destination embeddings.

The local graph operator is then chosen as either:

1. `BaseTemporalConvLayer`, which applies weighted source-node projection plus an edge-conditioned shift term
2. `BaseTemporalMPNNLayer`, which computes edge-conditioned gated messages using source state, destination state, and edge state

Although the internal class names contain the word "temporal," these layers operate across graph structure at each time step of the temporally encoded sequence; they are graph operators, not the main temporal backbone.

### 4.2.3. Readout and prediction heads

After graph processing, `base_gnn` uses a target-centered readout. The `GraphReadout` module concatenates:

1. the target-node representation
2. global mean pooling and global max pooling
3. an optional target-to-global attention context

The resulting sequence is projected into `target_hidden_dim`, processed by the target temporal trunk, and finally mapped to the shared multi-task heads. This architecture makes `base_gnn` the simplest family in the study: it uses one fused graph, one adaptive adjacency mechanism, and one choice of local operator.

## 4.3. The `multigraph` family

The `multigraph` family extends the baseline by retaining relation-specific processing deeper into the network. It is evaluated in two matched forms: `dynamic_rel_conv` and `dynamic_edge_mpnn`.

### 4.3.1. Temporal component

The temporal component of `multigraph` is structurally the same as in `base_gnn`:

1. a `NodeTemporalEncoder`
2. an `EdgeTemporalEncoder`
3. a `TargetTemporalTrunk`

All three modules are again based on stacks of dilated causal residual convolution blocks. Consequently, the major difference between `multigraph` and `base_gnn` is not the temporal backbone, but the graph-processing logic applied after temporal encoding.

### 4.3.2. Graph component

The defining feature of `multigraph` is that it preserves the three relation channels longer. After the same hybrid edge-fusion stage used in `base_gnn`, the model does not immediately collapse the relation axis. Instead, it constructs one dedicated `RelationGraphBlock` per relation.

Each relation block contains a stack of graph layers using one of two operators:

1. `DynamicRelConvLayer`
2. `DynamicEdgeMPNNLayer`

`DynamicRelConvLayer` is the Conv-style variant. It computes an edge score from the relation-specific edge state, normalizes scores with a destination-wise edge softmax, and applies the resulting attention weights to source-node projections and edge-conditioned shifts.

`DynamicEdgeMPNNLayer` is the MPNN-style variant. It uses a gating network conditioned jointly on source-node state, destination-node state, and edge state. Messages are then aggregated to destination nodes and normalized by in-degree.

Because this process is repeated independently for each relation channel, `multigraph` can represent price dependence, order-flow dependence, and liquidity dependence as distinct graph pathways rather than as a single fused edge signal.

After the per-relation graph blocks, the model applies `RelationAttentionFusion`, which learns a weighted combination of the relation-specific node sequences. This late fusion is the key design decision that differentiates `multigraph` from `base_gnn`.

### 4.3.3. Readout and prediction heads

Once the relation-specific node sequences are fused, `multigraph` uses the same target-centered `GraphReadout` and the same `TargetTemporalTrunk` as the baseline. Therefore, the family comparison between `base_gnn` and `multigraph` isolates a clear architectural question: is it better to collapse relation structure early into a single edge representation, or to preserve relation-specific graph processing until late fusion?

## 4.4. The `memorygraph` family

The `memorygraph` family is the most distinct architecture in the study. It is also evaluated in two matched operator variants, `conv` and `mpnn`, but unlike the previous families it does not use a deep causal-convolutional temporal encoder. Instead, it relies on persistent recurrent state.

### 4.4.1. Temporal component

The temporal backbone of `memorygraph` is stateful and recurrent.

Raw node inputs are first processed by `NodeStepProjector`, which consists of:

1. a linear projection
2. a learned asset embedding
3. layer normalization
4. a feed-forward projection with `GELU` and dropout

Raw edge inputs are processed by `EdgeStepProjector`, a small MLP with layer normalization, `GELU`, and dropout. Unlike `base_gnn` and `multigraph`, these modules do not encode long temporal histories by themselves. They only create per-step hidden representations.

The actual temporal mechanism appears inside `MemoryAugmentedGraphBlock`, which carries state across time and across contiguous training chunks. This block maintains:

1. node memory
2. relation-specific edge memory

At each time step, the current projected node input is combined with a learned relation bias and a projection of the current node memory. The current projected edge input is combined with a projection of the current edge memory. The model then updates its recurrent state in two stages.

First, `EdgeMemoryUpdater` applies a `GRUCell` to the concatenation of:

1. the current edge representation
2. source-node state
3. destination-node state
4. pairwise differences and products of source and destination states

This produces updated edge memory, which is then transformed back into an enriched edge state through a gated fusion layer.

Second, after graph message passing, `NodeMemoryUpdater` aggregates relation-specific edge-memory context to nodes, fuses relation-specific node states and relation-specific edge contexts, and updates node memory with another `GRUCell`. The updated node memory is then transformed back into the node state through a gated fusion mechanism.

Training is performed on contiguous stateful chunks with truncated backpropagation through time. This makes `memorygraph` fundamentally different from the convolutional families: its temporal modeling capacity comes primarily from recurrent state propagation, not from a stack of dilated temporal convolutions.

### 4.4.2. Graph component

Within each recurrent step, graph interaction is handled by `MemoryOperatorBlock`. This block first learns a data-dependent adjacency with `AdaptiveGraphConnectivity`, which combines an edge-conditioned score with low-rank source and destination embeddings.

The graph operator is then chosen as either:

1. `conv`, implemented through `BaseTemporalConvLayer`
2. `mpnn`, implemented through `BaseTemporalMPNNLayer`

As in the other families, the Conv-style variant uses weighted source-node projection plus an edge-conditioned shift, whereas the MPNN-style variant uses gated messages conditioned on source nodes, destination nodes, and edge states.

The crucial difference is that in `memorygraph` these operators are embedded inside a recurrent memory loop. They therefore operate not on a fully pre-encoded temporal sequence, but on state-enriched per-step representations that already include persistent node and edge memory.

### 4.4.3. Relation handling, readout, and heads

`memorygraph` also preserves relation structure deeper into the architecture. Relation-aware node states and relation-aware edge memories are fused with `RelationAxisFusion`, which can use either mean fusion or learned attention. This happens after recurrent updates rather than before graph processing, which allows relation information to influence both memory formation and message passing.

The readout stage is simpler than in the convolutional families. After the recurrent graph block produces the fused node sequence, the model applies the same `GraphReadout` module used by the other families and then a lightweight `output_proj` MLP. There is no separate target temporal trunk, because temporal aggregation has already been performed by the recurrent memory process itself.

This design gives `memorygraph` a qualitatively different inductive bias:

1. `base_gnn` uses early relation fusion and a convolutional temporal stack
2. `multigraph` uses late relation fusion but still relies on a convolutional temporal stack
3. `memorygraph` uses relation-aware recurrent state and stateful graph updates

For this reason, `memorygraph` is the most informative test of whether persistent memory is genuinely beneficial for high-frequency market microstructure prediction once the evaluation is reduced to the same entry-model benchmark used for the other families.

# Results

## All models Final-CV results. 

| Frequency | Model | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----: | ----- | ----- | ----- | ----- |
| `5min` | `base-gnn-mpnn` | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | `base-gnn-conv` | 0.014415 | 0.006915 | 25 | 0.614912 | 0.727631 |
| `5min` | `multi-gnn-mpnn` | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| `5min` | `multi-gnn-conv` | 0.002941 | \-0.009359 | 41 | 0.625439 | 0.707304 |
| `5min` | `memory-gnn-mpnn` | 0.009459 | 0.004359 | 17 | 0.611842 | 0.734196 |
| `5min` | `memory-gnn-conv` | \-0.012463 | \-0.037363 | 83 | 0.537719 | 0.726026 |
| `1min` | `base-gnn-mpnn` | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | `base-gnn-conv` | \-0.024239 | \-0.078539 | 181 | 0.504512 | 0.642097 |
| `1min` | `multi-gnn-mpnn` | \-0.000594 | \-0.005694 | 17 | 0.573684 | 0.693639 |
| `1min` | `multi-gnn-conv` | 0.005622 | 0.003522 | 7 | 0.569737 | 0.670006 |
| `1min` | `memory-gnn-mpnn` | \-0.031247 | \-0.078947 | 159 | 0.479399 | 0.638928 |
| `1min` | `memory-gnn-conv` | 0.033605 | 0.009305 | 81 | 0.529844 | 0.635665 |
| `1sec` | `base-gnn-mpnn` | 0.000599 | 0.000149 | 1 | 0.594069 | 0.997705 |
| `1sec` | `base-gnn-conv` | 0.004917 | \-0.008883 | 46 | 0.601218 | 0.875124 |
| `1sec` | `multi-gnn-mpnn` | 0.00013 | \-0.00347 | 6 | 0.60167 | 0.998611 |
| `1sec` | `multi-gnn-conv` | 0.141935 | \-0.082465 | 748 | 0.595509 | 0.870613 |
| `1sec` | `memory-gnn-mpnn` | 0.412032 | \-1.163268 | 5251 | 0.588785 | 0.49005 |
| `1sec` | `memory-gnn-conv` | 0.223788 | \-0.280512 | 1681 | 0.596713 | 0.863699 |

## Best models Final-CV vs. Final-refit results.

### 5min \- base-gnn (adaptive\_conv)

run\_exp "304\_5min\_bg\_guard\_lr020" "--model-pipeline=base\_gnn|--freq=5min|--backtest-exit-mode=realized\_event|--lr=0.0002"

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `5min` | last\_cv | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | final\_refit | 0.01757 | 0.01127 | 21 | 0.630702 | 0.721795 |
|  |  | 160% | 181% |  | 98% | 97% |

### 1min \- base-gnn (adaptive\_conv)

run\_exp "311\_1min\_bg\_adconv\_lr020\_ep30" "--model-pipeline=base\_gnn|--freq=1min|--graph-operator=adaptive\_conv|--run-full-operator-ablation=false|--backtest-exit-mode=realized\_event|--lr=0.0002|--epochs=30|--patience=5|--batch-size=64"

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `1min` | last\_cv | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | final\_refit | 0.007198 | \-0.012002 | 64 | 0.524286 | 0.635712 |
|  |  | 829% | **\+** |  | 100% | 102% |

### 1sec \- memory-gnn (conv)

run\_exp "1sec02\_emg\_conv" "--config=train\_config.yaml|--model-pipeline=memorygraph|--freq=1sec|--horizon-minutes=2|--backtest-exit-mode=realized\_event|--graph-operator=conv|--run-full-operator-ablation=false|--lookback-bars-override=120|--relation-windows-override|30|60|120|--num-threads=8|--train-min-frac=0.35|--val-window-frac=0.08|--test-window-frac=0.08|--data-slice-start-frac=0.50|--data-slice-end-frac=0.90|--final-holdout-frac=0.225|--num-train-folds=2|--batch-size=64|--dropout=0.15|--memorygraph-memory-dropout=0.15|--lr=0.00025|--epochs=12|--patience=3|--node-hidden-dim=24|--edge-hidden-dim=12|--target-hidden-dim=32|--fusion-hidden-dim=12|--graph-layers=1|--memorygraph-node-memory-dim=24|--memorygraph-edge-memory-dim=12|--memorygraph-adjacency-rank=12|--loss-w-tte=0.05|--loss-w-exit-type=0.01|--memorygraph-chunk-len=512|--memorygraph-detach-interval=512|--memorygraph-eval-chunk-len=4096"

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `1sec` | last\_cv | 0.412032 | \-1.163268 | 5251 | 0.588785 | 0.49005 |
| `1sec` | final\_refit | 0.443031 | \-0.954969 | 4660 | 0.592186 | 0.852874 |
|  |  | 93% | **\-** |  | 99% | 57% |


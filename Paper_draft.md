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
3. `n_trades`
4. `dir_auc`
5. `trade_auc`

These metrics jointly capture raw signal quality, friction-adjusted economic usefulness, turnover intensity, directional discrimination, and trade-selection quality. In the interpretation of final results, `pnl_sum` is treated as the primary economic criterion, `gross_pnl_sum` is used to separate signal quality from transaction-cost effects, and `n_trades` is used to diagnose whether high gross performance is being achieved through economically unsustainable turnover. Supplementary metrics such as `sign_accuracy` and `sharpe_like` remain useful diagnostics, but they are not the primary reporting metrics in the final benchmark tables.

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

# 5. Results and Discussion

## 5.1. Cross-model comparison on `last_CV`

Table 5.1 reports the primary benchmark comparison for all eighteen models using the `last_CV` state. This table is the main empirical reference in the thesis because `last_CV` is the closest proxy to a realistically deployable model. The results immediately reveal three recurring patterns.

First, the economically best model is not always the model with the best ranking metrics. Across several frequencies, the strongest `dir_auc` or `trade_auc` value is achieved by a more complex family, while the strongest `pnl_sum` is achieved by a simpler model with a more favorable turnover profile.

Second, the relationship between `gross_pnl_sum` and `pnl_sum` becomes progressively more fragile as frequency increases. At `5min`, positive gross performance often remains positive after costs. At `1min`, the gap between gross and net performance becomes materially larger. At `1sec`, several models generate large positive gross PnL but still end with strongly negative net PnL, indicating that turnover and cost drag dominate the raw signal.

Third, the Conv-versus-MPNN comparison is clearly family-dependent. The operator that maximizes PnL in one family and frequency does not remain optimal elsewhere. This directly supports the view that local message-passing choice must be interpreted jointly with the broader architectural scaffold.

Table 5.1. `last_CV` benchmark comparison across all model-frequency combinations.

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

## 5.2. Results at 5-minute frequency

The 5-minute regime produces the clearest economic winner in the entire benchmark. `base_gnn + adaptive_mpnn` achieves both the highest `gross_pnl_sum` (0.028156) and the highest `pnl_sum` (0.020356) with only 26 trades. The second-best net result is the Conv variant of the same family, `base_gnn + adaptive_conv`, with `pnl_sum = 0.006915` and 25 trades. This means that, under the coarsest shared-task benchmark, the simplest family is also the most economically robust.

The more complex families do show isolated strengths. `multigraph + dynamic_rel_conv` delivers the highest `dir_auc` at 0.625439, while `memorygraph + mpnn` achieves the highest `trade_auc` at 0.734196. However, neither of these advantages translates into the best net trading outcome. The `multigraph` Conv variant ends with `pnl_sum = -0.009359`, and `memorygraph + mpnn` remains positive but weaker than the base model at `pnl_sum = 0.004359`.

The weakest 5-minute result is `memorygraph + conv`, which produces 83 trades and ends at `pnl_sum = -0.037363`. This is an early sign of the turnover problem that becomes much more pronounced at higher frequencies: once the model begins to trade too often, the friction-adjusted benchmark can deteriorate rapidly even if `trade_auc` remains acceptable.

Taken together, the 5-minute results suggest that under a short-horizon but relatively coarse observation regime, additional architectural complexity is not required to obtain the strongest economic result. What matters more is a favorable balance between signal quality and trade frequency.

## 5.3. Results at 1-minute frequency

The 1-minute regime is more nuanced. `base_gnn + adaptive_mpnn` again delivers the highest net performance, with `gross_pnl_sum = 0.059694` and `pnl_sum = 0.020094` over 132 trades. This is a substantially larger gross edge than in the 5-minute regime, but the final net gain is almost the same as at 5 minutes because the extra turnover absorbs most of the incremental signal.

The second-best net result comes from `memorygraph + conv`, which reaches `gross_pnl_sum = 0.033605` and `pnl_sum = 0.009305` over 81 trades. This is the strongest showing of the `memorygraph` family in the shared-task benchmark and indicates that recurrent state can be economically useful at minute-level resolution, even though it does not surpass the base MPNN baseline.

The most interesting discrepancy appears in the `multigraph` family. `multigraph + dynamic_edge_mpnn` achieves the highest `dir_auc` (0.573684) and the highest `trade_auc` (0.693639) among all six 1-minute models, yet its `pnl_sum` is negative at `-0.005694`, and it executes only 17 trades. `multigraph + dynamic_rel_conv` is economically positive (`pnl_sum = 0.003522`) but still trails both `base_gnn + adaptive_mpnn` and `memorygraph + conv`.

This means that, at 1 minute, richer relation-aware modeling clearly improves ranking-style metrics, but those improvements do not automatically translate into superior net profitability under the standardized entry benchmark. The most successful model remains the economically more robust base family with the MPNN operator.

## 5.4. Results at 1-second frequency

The 1-second regime produces the sharpest divergence between gross and net performance. On a gross basis, the strongest models are the memory-based variants. `memorygraph + mpnn` reaches `gross_pnl_sum = 0.412032`, and `memorygraph + conv` reaches `gross_pnl_sum = 0.223788`. `multigraph + dynamic_rel_conv` is also strongly positive in gross terms at 0.141935. However, all three models are decisively negative after costs:

1. `memorygraph + mpnn`: `pnl_sum = -1.163268`, `n_trades = 5251`
2. `memorygraph + conv`: `pnl_sum = -0.280512`, `n_trades = 1681`
3. `multigraph + dynamic_rel_conv`: `pnl_sum = -0.082465`, `n_trades = 748`

Under the benchmark commission assumption, the cost burden on these high-turnover models is extremely large. For example, 5,251 trades imply an approximate round-trip cost load of 1.5753 log-return units, which is far above the gross edge of 0.412032. The 1-second results therefore show that a model can be directionally informative and still fail economically once turnover is priced realistically.

The only positive `pnl_sum` at 1 second is produced by `base_gnn + adaptive_mpnn`, with `pnl_sum = 0.000149`. However, this result is based on a single trade, which makes it too sparse to treat as robust economic evidence. Similarly, `multigraph + dynamic_edge_mpnn` posts exceptionally strong classification metrics (`dir_auc = 0.601670`, `trade_auc = 0.998611`) but executes only 6 trades and still remains slightly negative in net terms.

The correct interpretation is therefore not that the 1-second regime fails to produce predictive structure. On the contrary, the gross PnL and AUC values show that meaningful signal exists. The problem is that, under the present cost model and thresholding regime, that signal does not survive economically once turnover becomes large.

## 5.5. `last_CV` versus `final_refit` for selected models

To understand whether the main conclusions are stable under additional retraining, the thesis compares `last_CV` and `final_refit` for selected representative models.

The comparison tables in this subsection are kept in the same exported form as the experiment artifacts. For traceability, this includes the final summary row generated by the reporting pipeline.

### 5.5.1. Best 5-minute model

The 5-minute comparison shows that the selected best model remains economically positive after refitting, but the optimistic `final_refit` state is not superior on the main economic metrics. `last_CV` records `gross_pnl_sum = 0.028156` and `pnl_sum = 0.020356`, whereas `final_refit` falls to `gross_pnl_sum = 0.017570` and `pnl_sum = 0.011270`. At the same time, `dir_auc` rises from 0.617105 to 0.630702 and `trade_auc` rises from 0.700447 to 0.721795. Thus, the refit model looks better as a ranker but worse as a trading system.

Table 5.2. `last_CV` versus `final_refit` for the selected 5-minute winner.

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `5min` | last\_cv | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| `5min` | final\_refit | 0.01757 | 0.01127 | 21 | 0.630702 | 0.721795 |
|  |  | 160% | 181% |  | 98% | 97% |

### 5.5.2. Best 1-minute model

The 1-minute comparison produces an even stronger version of the same pattern. `last_CV` yields `gross_pnl_sum = 0.059694` and `pnl_sum = 0.020094`, while `final_refit` drops to `gross_pnl_sum = 0.007198` and becomes net negative at `pnl_sum = -0.012002`. The AUC values remain close (`dir_auc` from 0.525927 to 0.524286, `trade_auc` from 0.648157 to 0.635712), but the economic deterioration is substantial. This suggests that the additional fit available to `final_refit` does not automatically improve deployable performance and may in some cases amplify calibration instability or overfit to the expanded development sample.

Table 5.3. `last_CV` versus `final_refit` for the selected 1-minute winner.

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `1min` | last\_cv | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| `1min` | final\_refit | 0.007198 | \-0.012002 | 64 | 0.524286 | 0.635712 |
|  |  | 829% | **\+** |  | 100% | 102% |

### 5.5.3. Selected 1-second memorygraph case

For the 1-second regime, the most informative comparison is the strongest gross-edge memorygraph case. Here `final_refit` does improve the result, but only partially. `gross_pnl_sum` rises from 0.412032 to 0.443031, `pnl_sum` improves from `-1.163268` to `-0.954969`, `dir_auc` rises from 0.588785 to 0.592186, and `trade_auc` rises sharply from 0.490050 to 0.852874, while `n_trades` falls from 5,251 to 4,660. Nevertheless, the model remains strongly net negative. Therefore, refitting can reduce damage and improve ranking quality, but it does not solve the basic turnover-cost problem at 1 second.

Table 5.4. `last_CV` versus `final_refit` for the selected 1-second memorygraph case.

| Frequency | Training Cycle | Gross pnl sum | pnl sum | N trades | dir auc  | trade auc |
| :---- | :---- | ----- | ----- | ----- | ----- | ----- |
| `1sec` | last\_cv | 0.412032 | \-1.163268 | 5251 | 0.588785 | 0.49005 |
| `1sec` | final\_refit | 0.443031 | \-0.954969 | 4660 | 0.592186 | 0.852874 |
|  |  | 93% | **\-** |  | 99% | 57% |

## 5.6. Hypothesis evaluation

### H1. The 1-minute regime should be the strongest shared-task benchmark.

This hypothesis is not supported by the reported results. The best 1-minute model, `base_gnn + adaptive_mpnn`, reaches `pnl_sum = 0.020094`, which is slightly below the best 5-minute result of `0.020356`. The highest 5-minute `dir_auc` (0.625439) and highest 5-minute `trade_auc` (0.734196) are also above the corresponding 1-minute maxima of 0.573684 and 0.693639. The 1-minute regime does deliver a larger gross edge, but that advantage is largely offset by higher turnover.

### H2. Explicit multi-relation modeling should outperform the simpler baseline more clearly at finer resolutions.

This hypothesis is only partially supported. On predictive metrics, the evidence is favorable to `multigraph`: at 1 minute, `multigraph + dynamic_edge_mpnn` achieves the best `dir_auc` and `trade_auc`, and at 1 second, `multigraph + dynamic_edge_mpnn` again produces the best AUC values. However, these predictive advantages do not translate into superior net PnL. In the economically central comparison, `base_gnn + adaptive_mpnn` remains the strongest model at both 5 minutes and 1 minute.

### H3. Stateful memory should become more valuable as temporal resolution increases.

This hypothesis is not supported in the primary economic benchmark, although it receives partial support in gross-signal terms. At 1 second, `memorygraph + mpnn` produces the largest `gross_pnl_sum` of the entire study (0.412032), and `memorygraph + conv` is also strongly positive in gross terms. However, both models are decisively negative after transaction costs. At 1 minute, `memorygraph + conv` is economically competitive but still not the benchmark winner, and at 5 minutes the best memory model remains below the best base model. Thus, persistent memory appears valuable for raw signal extraction, but not for cost-adjusted profitability under the current benchmark.

### H4. Conv and MPNN operators should not be uniformly dominant across families.

This hypothesis is supported. Inside `base_gnn`, the MPNN operator is consistently stronger on net PnL across all three frequencies. Inside `multigraph`, the ranking is mixed: MPNN is stronger at 5 minutes and less negative at 1 second, whereas Conv is economically better at 1 minute. Inside `memorygraph`, the best operator changes by frequency: MPNN is better at 5 minutes, while Conv is economically better at 1 minute and materially less harmful than MPNN at 1 second. The operator effect is therefore clearly family- and frequency-specific.

### H5. `last_CV` and `final_refit` should tell a consistent family-level story.

This hypothesis is only partially supported. The broad substantive conclusion remains stable: increased architectural complexity does not overturn the core finding that friction-adjusted robustness is difficult to preserve as frequency increases. However, `final_refit` is not a uniformly optimistic upper bound. At 5 minutes and 1 minute, `final_refit` is economically worse than `last_CV` for the selected winning models, while at 1 second it improves the selected memorygraph case without restoring profitability. The practical implication is that `last_CV` should remain the primary deployment reference, not merely a conservative secondary benchmark.

## 5.7. Additional result fields that should be included in the final thesis

The current tables already capture the main economic and predictive comparison well. However, several additional fields would materially improve the final interpretation of the hypotheses.

### 5.7.1. Strongly recommended additions

1. `pnl_per_trade`
   This would make it much easier to separate genuine edge from turnover-driven gross PnL, especially at `1sec`.

2. `trade_rate` or `coverage`
   Raw `n_trades` is useful, but it is not normalized by sample length. A normalized trade frequency is essential for comparing `1sec`, `1min`, and `5min` results more fairly.

3. `gross_pnl_sum - pnl_sum`
   This directly measures the cost burden. It would make the 1-second failure mode immediately visible without requiring the reader to reconstruct it manually.

4. Fold dispersion for `pnl_sum`, `dir_auc`, and `trade_auc`
   A `cv_mean +/- std` or at least a min-max range would be highly valuable for judging stability and for supporting or rejecting the deployment-oriented hypothesis more rigorously.

5. `selected_thr_trade`, `selected_thr_dir`, and selected coverage
   These values would show whether economic underperformance is driven by weak ranking quality or by an unfavorable operating point chosen during calibration.

### 5.7.2. Useful supplementary additions

1. `long_trades`, `short_trades`, `long_pnl_sum`, `short_pnl_sum`
   These would reveal whether a model's performance is structurally one-sided.

2. `win_rate` and `sign_accuracy`
   These metrics are less important than net PnL, but they would help explain whether a model is losing because it is often wrong or because it is correct on moves that are too small after costs.

3. `best_CV`, `last_CV`, and `final_refit` for all 18 models
   This would enable a much stronger and more general evaluation of Hypothesis H5 than the selected-model comparisons alone.

4. Exit-type composition or average realized holding time
   These fields would help explain whether specific models lose money because they overproduce short-lived or timeout-heavy trades.

## 5.8. Main answer to the research question

The main research question asked which graph family performs best under a controlled apples-to-apples entry-model benchmark, and how that answer changes with temporal resolution.

The empirical answer is clear. Under the present benchmark, the most economically robust family is `base_gnn`, especially in its MPNN variant. It produces the best net result at both 5-minute and 1-minute frequency and remains the only family with a non-negative 1-second net result, although the 1-second case is too sparse to be considered robust.

`multigraph` and `memorygraph` do add value, but not in the way initially hypothesized. Their strongest contribution is to predictive discrimination and raw signal extraction rather than to final net profitability. `multigraph` often achieves the best `dir_auc` and `trade_auc`, while `memorygraph` can generate the largest gross edge at the finest frequency. However, these advantages are not sufficient to overcome transaction costs and turnover under the current evaluation protocol.

Therefore, the main answer of the thesis is that richer graph structure and persistent memory do not automatically outperform a simpler adaptive graph baseline once the comparison is reduced to the same friction-aware entry benchmark. Under realistic costs, economic robustness favors the simpler model family unless the more complex models are calibrated more aggressively against turnover.

## 5.9. Overall thesis conclusion

The main contribution of this thesis is to show that controlled architectural comparison matters. When graph families are evaluated under the same targets, the same entry logic, the same cost model, and the same validation protocol, the ranking of models is more conservative than a purely predictive comparison would suggest.

The shared-task benchmark at 5 minutes and 1 minute shows that the adaptive base graph model is the most reliable economic performer. The `multigraph` family demonstrates that explicit relation-aware processing can improve classification quality, but these gains are not sufficient on their own to dominate in net PnL. The `memorygraph` family demonstrates that stateful memory can generate substantial gross edge, especially at 1 second, but that such edge is highly vulnerable to transaction-cost drag when turnover is not tightly controlled.

The thesis therefore delivers a precise empirical message: in high-frequency crypto limit order book prediction, architectural richness first appears in ranking metrics and gross signal quality, but deployable profitability depends at least as much on turnover control and calibration as on representation power.

## 5.10. Future research

Several follow-up directions arise naturally from these findings.

First, future work should explicitly optimize for turnover-aware deployment rather than treating calibration as a secondary stage. The present results strongly suggest that the 1-second failures are not failures of raw signal generation, but failures of friction-aware operating-point selection.

Second, future work should report and compare `best_CV`, `last_CV`, and `final_refit` for all eighteen model configurations. This would allow a more complete study of stability and would strengthen the deployment-oriented claims of the thesis.

Third, the comparison should be extended to normalized turnover metrics such as trade rate, average holding time, and PnL per trade. These metrics would make it easier to explain why certain families look attractive in gross terms but weak in net terms.

Fourth, the current three-asset graph could be expanded to a larger crypto universe. This is especially relevant for `multigraph`, whose relation-specific design may become more valuable when the graph contains a richer set of cross-asset interactions.

Fifth, `memorygraph` deserves a dedicated second-stage study as an execution-aware model rather than only as an entry model. In the present thesis it was intentionally constrained by the apples-to-apples benchmark. A natural continuation would be to test whether its auxiliary exit heads can recover value in a fair but explicitly two-stage entry-plus-exit setting.

Finally, future research should explore more realistic cost models, alternative threshold-calibration objectives, and explicit target-turnover constraints. These are the most direct ways to test whether the gross edge observed in the more complex models can be converted into robust net profitability.

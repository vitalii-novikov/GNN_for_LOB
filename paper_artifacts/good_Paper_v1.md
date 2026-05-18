Author: **Vitalii Novikov**

Degree program: **Master Artificial Intelligence & Data Science SS2026**

1st reviewer master’s thesis: **Dr. Rosana De Oliveira Gomes**

github: [https://github.com/vitalii-novikov/GNN\_for\_LOB](https://github.com/vitalii-novikov/GNN_for_LOB)

Supplementary benchmark evidence is reported in Appendix B.

# **Real-Time Market Microstructure Modeling with Temporal Graph Neural Networks**

## **Abstract**

This thesis compares graph-based neural architectures for short-horizon limit order book prediction under a common, deployment-oriented benchmark. The empirical setting is a cryptocurrency limit order book dataset containing ADA, BTC, and ETH snapshots at five-minute, one-minute, and one-second resolution. ETH is the target asset, while ADA and BTC provide relational market context. The study compares three model families: a single-graph baseline (base\_gnn), a multi-relation graph architecture (multigraph), and a stateful recurrent graph architecture (memorygraph). Each family is tested with a convolution-style graph operator and a message-passing neural network operator, producing eighteen primary benchmark configurations (3 frequencies, 3 architecture families, 2 operators).

The thesis is a controlled benchmark, not a production trading-system claim. All models share the same asset universe, feature construction, relation-state construction, triple-barrier target logic, multi-task output heads, purged walk-forward validation, final holdout interval, thresholding logic, and event-based backtest. This design makes architectural comparison more reliable because the models differ mainly in how they encode graph, relation, and temporal information. The central evaluation distinction is between ‘last\_CV’, the final chronological cross-validation model used as the main deployment-oriented reference, and ‘final\_refit’, a larger-sample refit state used as an informative robustness comparison.

The strongest last\_CV result at both five-minute and one-minute resolution is obtained by base-gnn-conv. At 5min, it achieves net\_pnl \= 0.020356 over 26 trades. At 1min, it achieves net\_pnl \= 0.020094 over 132 trades. At 1sec, all models remain negative after transaction costs. The one-second experiments are still informative because several models produce positive gross\_pnl; the problem is that the gross edge is consumed by excessive turnover and the round-trip cost proxy. The clearest example is memory-gnn-conv, which produces the largest one-second gross signal but ends with strongly negative net PnL because it executes 5251 trades.

The thesis concludes that, under this controlled entry-model benchmark, the simpler base\_gnn architecture is the most reliable family. More complex relation-preserving and memory-based models provide useful diagnostics, especially for gross signal extraction and ranking quality, but they do not establish a robust post-cost advantage. Overall, the findings indicate that the central challenge in this setting is not only identifying short-horizon microstructure signals, but translating them into sufficiently selective, stable, and cost-aware trading decisions.

## **Keywords**

limit order book; market microstructure; graph neural networks; temporal graph learning; walk-forward validation; triple-barrier labels; transaction-cost-aware backtesting

## **1\. Introduction**

### **1.1. Motivation of the Topic**

Financial markets generate high-volume streams of event-like information. At high frequency, prices, spreads, order-flow summaries, and visible depth change faster than a human analyst can inspect them manually. The limit order book (LOB) is therefore a natural object for data science: it records the visible supply and demand around the current price and provides a structured view of short-horizon market dynamics \[2\], \[16\]. It is also a difficult prediction environment. Useful signals are weak, non-stationary, regime-dependent, and highly sensitive to latency, costs, threshold choice, and model-selection bias. These difficulties are consistent with well-known stylized facts of financial returns, including heavy tails, volatility clustering, and changing dependence structures \[1\]. They also appear in cryptocurrency LOBs, where empirical studies report similarities to mature electronic markets but also shallower books and relatively high liquidity costs \[17\].

This difficulty creates both a scientific and a practical motivation. Scientifically, LOB prediction is a demanding test case for machine learning on noisy sequential data. A model must process temporal dependence, cross-asset information, changing liquidity, and shifting dependence patterns without relying on a stationary data-generating process. Practically, a forecast is not valuable only because it predicts a future direction. It becomes valuable only if it can be translated into selective decisions after transaction costs and turnover are considered. This thesis therefore treats predictive quality, gross signal quality, and net economic value as related but distinct layers of evidence.

Recent machine learning research has shown that deep architectures can learn useful representations from LOB data. Convolutional and recurrent models learn local book structure and temporal dependence; large-scale order-flow studies report cross-instrument regularities; and recent LOB work has moved toward attention, transformer, survival-analysis, and topology-aware representations \[3\]-\[4\]-\[5\], \[18\]-\[19\]-\[20\], \[25\]. These results motivate representation learning, but they do not remove the need for chronological validation. In financial prediction, random splits can leak information across time, and labels based on future price paths can create overlap between neighbouring observations. For this reason, the present benchmark uses a purged walk-forward design and a final chronological holdout rather than random cross-validation \[23\].

Graph-based modelling provides an additional motivation. Financial assets do not evolve independently: returns, order-flow pressure, spreads, and liquidity states can co-move, lead, lag, or diverge. A graph representation makes this relational structure explicit by representing assets as nodes and cross-asset dependence measures as edges. Static graph neural networks are suitable when relations are fixed or slowly varying \[6\]-\[8\]-\[9\], while temporal and dynamic graph learning address settings in which node states, edge states, or interaction patterns evolve over time \[10\]-\[11\]-\[12\], \[21\], \[22\]. This makes graph-based modelling plausible for cross-asset market microstructure, even though the empirical value of this additional structure must be tested rather than assumed. Figure 1.1 summarizes the conceptual pipeline studied in the thesis.

![][image1]

| *Figure 1.1 \- Conceptual pipeline*  |
| :---: |

The present thesis studies this idea in a deliberately controlled form. It does not attempt to build a full production trading system. Instead, it asks whether richer graph architectures improve a common entry-model benchmark when the data, targets, output heads, validation design, thresholding logic, and event-based trading evaluation are held as consistent as possible. The empirical question is therefore architectural: under a shared benchmark, does it help to preserve multiple relation channels, to add stateful memory, or to use a richer message-passing graph operator?

### **1.2. Research Gap and Thesis Scope**

The literature contains several relevant strands. Market microstructure explains why order flow, liquidity, and the organization of the book matter for short-horizon price formation \[2\], \[16\]. Deep learning for LOB prediction shows that neural networks can learn from high-dimensional book states \[3\], \[4\], \[5\], \[18\], \[19\], \[20\], \[25\]. Graph neural networks provide tools for learning from relational data \[6\], \[7\], \[9\], while dynamic graph methods extend this idea to systems whose states and relations evolve over time \[11\], \[12\], \[21\], \[22\]. Financial graph learning also shows how relation structures can be useful in stock prediction and other financial tasks \[13\], \[14\].

The gap addressed here is narrower and more empirical. Many financial graph studies focus on daily or lower-frequency relations, while many LOB studies model a single instrument without explicitly representing cross-asset graph structure. This thesis examines a small but controlled crypto limit order book setting in which ADA, BTC, and ETH form a three-node graph, ETH is the target asset, and cross-asset relation states are rebuilt at 5min, 1min, and 1sec resolutions. The study focuses on whether graph family, graph operator, and temporal resolution change the usefulness of relational and memory-aware modelling under a common trading-oriented evaluation.

The scope is intentionally limited. The benchmark uses a fixed asset universe, a fixed target asset, a common triple-barrier target construction, and a shared non-overlapping event backtest. This design improves internal comparability, but it also means that the thesis evaluates entry models rather than complete trading systems with jointly optimized execution and exit policies. The resulting conclusions should therefore be interpreted as evidence about architecture under a controlled benchmark, not as evidence that any model is ready for production deployment.

### **1.3. Research Aim**

The aim of this thesis is to determine whether richer graph-based architectures improve short-horizon limit order book prediction and trading performance when evaluated under an apples-to-apples, friction-aware benchmark.

The core object of interest is the model family. The study asks whether a simple single-graph representation is sufficient, whether preserving multiple relation channels improves the result, and whether stateful memory becomes valuable at higher temporal resolution. A second object of interest is the graph operator: a Conv-style operator versus a message-passing neural network operator. A third object of interest is deployment stability: whether the same conclusions hold when moving from the last chronological cross-validation state to a final refit state.

### **1.4. Research Questions**

The thesis is guided by four research questions. Figure 1.2 shows how the questions connect model family, graph operator, temporal resolution, and deployment-oriented model state. Table 1.1 then maps each question to the research method.

**RQ1. Which graph family performs best under a controlled entry-model benchmark?**  
The first question asks whether the simpler single-graph baseline, the multi-relation graph family, or the stateful memory graph family produces the strongest final-holdout trading result when all families are evaluated under the same target construction, thresholding logic, and event-based backtest.

**RQ2. How important is the Conv-versus-MPNN operator choice inside each family?**  
Each family is evaluated with a Conv-style operator and an MPNN-style operator. This makes it possible to distinguish the effect of the broader family scaffold from the effect of the local graph interaction mechanism.

**RQ3. How does temporal resolution change the relative value of relational and memory mechanisms?**  
The 5min and 1min regimes solve the same 30-minute lookback and five-minute horizon task, while the 1sec regime uses a frequency-adapted two-minute lookback and two-minute horizon. This design allows the thesis to examine whether richer relation handling and recurrent memory become more useful as the observation frequency increases.

**RQ4. Are the conclusions stable under deployment-oriented model states?**  
The thesis distinguishes between last\_CV, the final walk-forward fold model used as the primary deployment-oriented reference, and final\_refit, a model refit on a larger pre-holdout sample. This question asks whether the same model remains attractive when viewed through both states, and what this implies for realistic deployment interpretation.

![Figure 1.2 - Research question map][image2]

*Figure 1.2 \- Research question map*

**Table 1.1. Research questions, methods, and expected result types.**

| Research question | Applied research method | Expected result type |
| :---: | :---: | :---: |
| RQ1. Which graph family performs best? | Controlled benchmarking across base\_gnn, multigraph, and memorygraph under the same data, labels, splits, thresholds, and backtest. | Ranked model-family comparison on final-holdout economic and diagnostic metrics. |
| RQ2. How important is the Conv-versus-MPNN operator choice? | Within-family ablation-style comparison of Conv and MPNN operators across three frequencies. | Operator-level evidence showing whether performance changes are family- and frequency-dependent. |
| RQ3. How does temporal resolution affect relation and memory mechanisms? | Frequency-regime comparison between 5min, 1min, and frequency-adapted 1sec experiments. | Interpretation of how gross signal, turnover, and post-cost outcomes change with temporal resolution. |
| RQ4. Are conclusions stable between last\_CV and final\_refit? | Deployment-state comparison using selected final-holdout benchmark results. | Cautious stability assessment that separates chronological deployment evidence from larger-sample refit evidence. |

### **1.5. Hypotheses**

The empirical design tests five hypotheses.

**H1. The one-minute regime should be the strongest shared-task benchmark.**  
Because the 1min data preserve more intra-horizon dynamics than 5min data while remaining less noisy than second-level data, the 1min regime is expected to be the strongest of the two strict shared-task regimes.

**H2. Explicit multi-relation modelling should outperform the simpler baseline more clearly at finer resolutions.**  
The multigraph family is expected to benefit from preserving price-dependence, order-flow, and liquidity channels separately, especially when cross-asset dependencies evolve quickly.

**H3. Stateful memory should become more valuable as the market is observed more finely.**  
The memorygraph family is expected to be most useful at high frequency because recurrent state can, in principle, retain short-lived market information across contiguous observations.

**H4. Conv and MPNN operators should not be uniformly dominant across families.**  
The operator comparison is expected to be family- and frequency-dependent. A Conv-style operator may be more stable when edge structure is already regularized, while an MPNN-style operator may be more useful when messages need richer source-destination-edge conditioning.

**H5. last\_CV and final\_refit should provide broadly consistent but not necessarily equivalent evidence.**
The broad family-level conclusion is expected to remain similar across states, while individual model profitability and diagnostic metrics may change when the larger pre-holdout sample is used for refitting. The hypothesis does not assume that refitting uniformly improves `net_pnl`; instead, it treats the comparison as evidence about deployment sensitivity.

### **1.6. Thesis Contribution**

The thesis contributes a controlled empirical comparison of graph-based market microstructure models across three temporal resolutions. Its contribution is not a new universal architecture, nor a claim of production deployment readiness. Instead, it provides evidence on three narrower issues:

1. whether a simpler single-graph baseline is sufficient under a common entry-model benchmark;

2. whether explicit multi-relation handling or stateful memory improves economic outcomes after costs;

3. why deployment-oriented model states, turnover, and cost drag must be included in the interpretation of high-frequency predictive models.

From a data science perspective, the thesis contributes not only a model comparison but also a controlled evaluation workflow for noisy, non-stationary sequential data. The workflow emphasizes graph-based representation learning, leakage-aware preprocessing, chronological validation, multi-task supervised learning, cost-sensitive post-processing, and metric selection. These elements are central to the thesis because the empirical conclusion depends not only on model architecture, but also on how data are split, scaled, labelled, calibrated, and evaluated.

The main empirical finding is conservative. In the main benchmark, base-gnn-conv is the strongest last\_CV model at both 5min and 1min. Richer graph mechanisms sometimes improve gross signal or ranking metrics, especially at 1sec, but these gains do not reliably translate into positive net profitability after transaction costs.

## **2\. Literature Background**

### **2.1. Market Microstructure and Limit Order Book Prediction**

Market microstructure studies how trading rules, liquidity provision, order flow, and the organization of the limit order book affect price formation \[2\], \[16\]. At short horizons, the visible book is informative because it contains the current distribution of buy and sell interest around the mid-price. However, short-horizon predictability is difficult to exploit. Return distributions are heavy-tailed, volatility clusters over time, and dependence structures change \[1\]. These features make financial forecasting different from many supervised-learning problems with stable sampling assumptions.

LOB modelling also creates an evaluation challenge. A direction classifier can appear useful under accuracy or AUC while remaining economically weak if it triggers many low-margin trades. This gap is especially important in cryptocurrency markets. Empirical work on Bitcoin LOBs reports that crypto order books share some stylized facts with mature markets but can also be relatively shallow, with higher liquidity costs \[17\]. These conditions make transaction costs and turnover central to interpretation.

For this reason, this thesis evaluates models through a friction-aware entry benchmark rather than through classification metrics alone. Directional AUC and trade AUC are retained as diagnostics, but net\_pnl, gross\_pnl, pnl\_per\_trade, n\_trades, and trade\_rate are required to interpret whether a predictive signal survives as an economic signal.

### **2.2. Deep Learning for Limit Order Books**

Deep learning research on LOBs has shown that neural networks can learn representations from high-dimensional book states \[3\], \[25\]. DeepLOB is especially relevant because it combines convolutional components for local book structure with recurrent components for temporal dependence \[5\]. Large-scale order-flow studies also suggest that neural models can identify cross-instrument regularities in price formation \[4\]. These studies motivate representation learning in market microstructure, but they also highlight the need to separate predictive performance from economic performance.

Recent LOB research has broadened the model space. Arroyo et al. use convolutional-transformer survival models to estimate fill probabilities, connecting LOB representation learning to execution-aware decisions \[18\]. Jung and Lee study attention-based sequence-to-sequence forecasting of multi-level LOB states, emphasizing high dimensionality, irregular timing, and spatiotemporal dependencies \[19\]. Briola, Bartolucci, and Aste propose HLOB, which uses information-filtering graph structure to study information persistence across LOB levels \[20\]. These studies are not direct baselines for the present thesis, because the current benchmark is a cross-asset graph entry-model comparison rather than a full LOB reconstruction or execution-probability model. They are nevertheless relevant because they show that recent LOB research increasingly treats market microstructure as structured, temporal, and evaluation-sensitive data.

The present thesis differs from single-instrument LOB prediction studies by making cross-asset relational structure explicit. Instead of treating ETH only as an isolated time series, ADA and BTC are included as context nodes. The resulting graph is small, but it allows the thesis to test whether graph modelling adds value once all families share the same target construction, validation protocol, and trading evaluation.

### **2.3. Graph Neural Networks and Message Passing**

Graph neural networks provide a general framework for learning from entities connected by relations. Graph convolutional networks and graph attention networks show how node representations can be updated using neighbourhood information, while message-passing neural networks provide a flexible formulation in which messages depend on source nodes, destination nodes, and edge attributes \[6\]-\[8\]-\[9\]. These ideas are directly relevant to financial data because assets can be represented as nodes and cross-asset dependence measures as edges.

In this thesis, the Conv-versus-MPNN distinction is used as a controlled operator comparison. Conv-style graph layers apply weighted source-node projections with edge-conditioned shifts. MPNN-style layers use richer gated messages that condition on source state, destination state, and edge state. The comparison therefore asks whether richer local message conditioning is economically useful under the same graph-family scaffold.

### **2.4. Temporal and Dynamic Graph Learning**

Many real systems are not static graphs. Node states, edge states, and interaction patterns can evolve over time. Temporal graph networks and dynamic graph representation learning address this problem by combining graph operators with temporal encoders, memory modules, or event-driven updates \[10\]-\[11\]-\[12\]. Recent surveys reinforce that dynamic GNNs are designed for settings where topology or attributes change over time, and that open challenges include scalability, heterogeneous information, and memory-enhanced modelling \[21\], \[22\]. This literature is relevant to market microstructure because cross-asset relations are unlikely to remain fixed across regimes, liquidity states, and trading intensity.

The three families in this thesis instantiate this idea at different levels of complexity. base\_gnn uses early relation fusion and a convolutional temporal backbone. multigraph preserves relation-specific graph pathways longer before fusing them. memorygraph uses recurrent node and edge memory inside a graph-processing loop. The empirical question is not whether these mechanisms are theoretically plausible; it is whether they improve a controlled friction-aware benchmark after costs.

### **2.5. Financial Graph Learning**

Financial applications of graph neural networks include stock relation modelling, portfolio prediction, risk propagation, fraud detection, and transaction-network analysis \[13\]. Financial graphs are often heterogeneous or time-varying because relations can be induced by sectors, ownership, supply chains, correlations, news, or market co-movements \[13\]. Recent multi-relational dynamic graph work is especially relevant because it treats financial relations as heterogeneous and temporally evolving rather than as a single fixed adjacency matrix \[14\].

This thesis adopts a microstructure version of the same general idea by constructing relation channels from price dependence, order-flow dependence, and liquidity dependence. The design is deliberately modest: it does not claim to solve dynamic financial graph learning in general. Its contribution is an internally controlled comparison on a specific crypto LOB dataset and a specific entry-model benchmark.

### **2.6. Evaluation, Backtesting, and Leakage Control**

Financial machine learning requires evaluation methods that differ from many standard predictive-modelling workflows. Labels can depend on future price paths, adjacent samples can share overlapping horizons, and repeated experimentation can create backtest overfitting. The triple-barrier method and purged cross-validation framework are therefore central references for the target and validation design used in this thesis \[23\]. 

Transaction-cost-aware evaluation is also required. A high-frequency model may produce useful ranking statistics or positive gross PnL while failing after fees, spread, slippage, and turnover are considered. Execution-aware LOB research emphasizes that order placement and fill probabilities are themselves prediction problems, not details that can be ignored \[18\]. Algorithmic and high-frequency trading texts similarly treat costs, adverse selection, inventory risk, and execution design as central constraints on trading value \[24\]. The present thesis therefore keeps the backtest simple and common across models, but interprets net\_pnl, gross\_pnl, pnl\_per\_trade, n\_trades, and trade\_rate together.

The literature therefore motivates the empirical design that follows. Market microstructure explains why the prediction problem is noisy and friction-sensitive; deep LOB models motivate neural representation learning; GNN and temporal graph methods motivate cross-asset and memory-aware architectures; and financial machine learning evaluation literature motivates triple-barrier labels, purged walk-forward validation, and post-cost interpretation. The next chapter translates these ideas into the controlled data representation and evaluation protocol used in the thesis.

## **3\. Data and Methodology**

### **3.1. Data Source and Study Universe**

The raw data source is the public Kaggle dataset *High-Frequency Crypto Limit Order Book Data* by Martinsn, which provides frequency-specific cryptocurrency limit order book snapshots for multiple assets, including ADA, BTC, and ETH, at 1sec, 1min, and 5min resolutions \[15\]. The data are distributed as order book snapshots organized by price level rather than as raw exchange message streams.

The present study uses a fixed three-node asset universe:

1. ADA

2. BTC

3. ETH

ETH is the target asset. ADA and BTC provide relational market context. Because the source data are already distributed in frequency-specific tables, no bespoke reconstruction of the limit order book from raw order messages is required. The preprocessing task is instead to standardize timestamps, align assets on a common clock, and derive node and edge features from the available order book summaries.

The local data files used in the pipelines contain midpoint price, spread, buy and sell flow summaries, and 15 bid-side and 15 ask-side depth values. These fields are the foundation for all node features and relation features used in the benchmark.

### **3.2. Graph Input Representation**

All models use the same graph input representation within a frequency regime. The graph is a directed complete graph over the three assets with self-loops. The nodes are fixed, but node states and edge states vary over time.

Formally, each model receives:

1. a node sequence XnRBLNFn

2. a relation-aware edge sequence XeRBLREFe

where B is batch size, L is the lookback length, N=3 is the number of assets, R=3 is the number of relation channels, and E is the number of directed edges including self-loops.

The three relation channels are:

1. price\_dep, based on asset log returns.

2. order\_flow, based on flow imbalance scaled by log turnover.

3. liquidity, based on spread, depth imbalance, near-depth imbalance, and near/far depth shape.

The overall graph representation is summarized in Figure 3.1.

![][image3]*Figure 3.1 \- Graph input representation*

All families receive the same graph-structured input: a three-node directed complete graph with self-loops, ETH as target, dynamic node states, and three relation-aware edge channels.

### **3.3. Node Features**

For each asset and each time step, the node feature block summarizes local price behavior, order-flow pressure, and depth structure. The implemented node features are:

1. one-bar log return.

2. relative spread.

3. log-transformed buys.

4. log-transformed sales.

5. flow imbalance.

6. total depth imbalance.

7. top-level depth imbalances for the first five book levels.

8. bid near/far depth ratio.

9. ask near/far depth ratio.

10. near-depth imbalance.

11. far-depth imbalance.

This feature set is deliberately microstructure-oriented. It does not use external news, social media, or macroeconomic variables. That choice keeps the thesis focused on the information available inside the aligned cross-asset order book state.

### **3.4. Relation States and Edge Features**

Edge features are constructed from rolling cross-asset dependence measures. For every ordered asset pair and every relation channel, the pipeline computes lagged rolling features over frequency-specific windows:

1. rolling correlation.

2. rolling beta.

3. rolling mean product.

When configured, rolling correlations are Fisher-z transformed before scaling. The edge tensor therefore represents relation-specific dependence among assets rather than only a fixed adjacency prior.

This design is important for fair comparison. All three model families operate on the same handcrafted relation states and the same learnable pairwise edge-fusion path. The architectures differ in how they process and fuse this information, not in whether they receive richer or poorer input data.

### **3.5. Scaling and Leakage Control**

Node and edge tensors are robustly scaled on training data only, using fold-specific quantile statistics. The transformed features are then clipped to bounded ranges before model fitting. This prevents train-test leakage through scaling and reduces the influence of extreme observations. Because the same scaling approach is used for all families, feature preprocessing does not favor any architecture.

Leakage control is especially important in this thesis because the target labels are constructed from future ETH midpoint paths. The validation design therefore avoids random splits and treats chronological separation, fold-specific preprocessing, and purge gaps as part of the experimental method rather than as implementation details. This follows the financial machine learning argument that conventional cross-validation can be misleading when labels overlap in time or when repeated strategy selection creates overfitting risk \[23\].

### **3.6. Frequency-Specific Experimental Regimes**

The experimental design contains eighteen primary runs: six model variants for each of the three frequency regimes.

The 5min and 1min regimes solve the same clock-time task:

1. lookback window \= 30 minutes.

2. forecast horizon \= 5 minutes.

This corresponds to six lookback bars and one horizon bar at 5min, and 30 lookback bars and five horizon bars at 1min.

The 1sec regime uses a frequency-adapted task:

1. lookback window \= 2 minutes \= 120 bars.

2. forecast horizon \= 2 minutes \= 120 bars.

The one-second working sample is restricted to the interval from 50% to 90% of the full second-level series. This keeps training computationally feasible while preserving a late-period high-frequency comparison. The final holdout fraction is increased to align the one-second blind evaluation interval as closely as possible with the final holdout interval used in the slower-frequency experiments. Table 3.1 summarizes these frequency-specific settings.

**Table 3.1. Frequency-specific experimental regimes and validation settings.**

| Frequency | Working data slice | Final holdout fraction | Lookback | Horizon | CV folds |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 5min | 0.0-0.9 of the full series | 0.10 | 30 min \= 6 bars | 5 min \= 1 bar | 4 |
| 1min | 0.0-0.9 of the full series | 0.10 | 30 min \= 30 bars | 5 min \= 5 bars | 4 |
| 1sec | 0.5-0.9 of the full series | 0.225 | 2 min \= 120 bars | 2 min \= 120 bars | 2 |

The frequency-specific sample design is shown schematically in Figure 3.2.

![Figure 3.2 - Split design][image4]

*Figure 3.2 \- Split design*

Figure 3.2 makes the comparability boundary explicit. The 5min and 1min regimes share the same broad working-sample logic and late final holdout, so their results form the strict shared-task benchmark. The 1sec regime is internally consistent, but its restricted working slice and enlarged holdout show why it should be interpreted as a frequency-adapted high-frequency stress test rather than as a perfectly symmetric continuation of the 30-minute lookback and five-minute horizon task.

### **3.7. Target Construction and Shared Learning Objective**

All model families are trained under the same multi-task triple-barrier framework. The triple-barrier approach labels an observation by following the future price path until an upper barrier, lower barrier, or vertical time barrier is reached \[23\]. In this thesis, for each valid timestamp t, the future path of the ETH midpoint is followed until one of three mutually exclusive events occurs:

1. the upper barrier is touched.

2. the lower barrier is touched.

3. the vertical barrier is reached.

The barrier system is volatility-scaled. In the default benchmark configuration, the upper and lower barriers start from 8 basis points, are rescaled using rolling volatility estimated over a 30-bar lookback, are multiplied by 1.8, and are clipped to the interval from 4 to 30 basis points. The vertical barrier is set equal to the prediction horizon.

Figure 3.3 summarizes the triple-barrier target construction used before the common target set is derived.

![Figure 3.3 - Triple barrier][image5]

*Figure 3.3 \- Triple barrier*

The figure highlights that the label is path-dependent: the realized outcome is determined by the first touched barrier or by the vertical timeout, not only by the endpoint return at a fixed horizon. This is why the same construction can provide both an economic trade label and auxiliary information about direction, exit type, and time to exit.

From this future path, the pipeline constructs a common target set:

1. realized return.

2. trade relevance label.

3. direction label.

4. exit-type label.

5. time-to-exit label.

The trade label is meta-labeled and depends on whether the future move remains economically meaningful after a friction-aware threshold is applied. Direction labels are masked when timeout outcomes are configured as uninformative for directional supervision. This design uses the triple-barrier framework as a labelling device, not as evidence that a trading strategy is economically complete.

All families share the same output interface:

1. trade\_logit

2. dir\_logit

3. return\_pred

4. exit\_type\_logit

5. tte\_pred

The multi-task objective combines trade classification, direction classification, return regression, utility-based supervision, exit-type classification, and time-to-exit regression. In the benchmark configuration, the loss weights are:

1. loss\_w\_trade \= 0.35

2. loss\_w\_dir \= 0.65

3. loss\_w\_ret \= 0.15

4. loss\_w\_utility \= 0.85

5. loss\_w\_exit\_type \= 0.05

6. loss\_w\_tte \= 0.03

This shared target design preserves comparability. The models differ in how they encode temporal and graph structure, not in what they are asked to predict.

### **3.8. Common Entry-Model Backtest**

The trading evaluation is formulated as a common entry-model benchmark. In the primary backtest:

1. the trade head determines whether a trade candidate is active.

2. the direction head determines whether the candidate becomes a long or short position.

3. the exit is generated by the same realized event rule for all families.

Exit-type and time-to-exit heads are retained as auxiliary learning targets and diagnostics, but they do not define a family-specific trade-closing policy in the main benchmark. This choice is especially important for memorygraph, because a stateful architecture could otherwise be evaluated under a different execution policy from the other families. The common entry-model benchmark improves internal validity by holding execution logic fixed.

The trading evaluation uses a sequential non-overlapping event-based backtest. Once a position is opened, no new position can be opened until the current one is closed. This makes turnover interpretable and avoids overlapping position exposure. The design is intentionally simpler than an execution simulator. Execution-aware LOB research shows that fill probabilities, passive-versus-aggressive order placement, and queue dynamics can materially affect realized trading value \[18\]. Algorithmic trading literature similarly treats transaction costs, market impact, and adverse selection as core constraints \[24\]. The present benchmark therefore uses a transparent cost proxy and interprets the result as an entry-model comparison, not as a live execution claim.

For trade i, gross PnL is computed as:

gross\_pnli=siri,

where si{−1,+1} is the trade side and ri is the realized log return up to the realized event exit. Net PnL is:

net\_pnli=gross\_pnli−crt,

where the round-trip transaction-cost proxy is:

crt=3cost\_bps\_per\_side10−4.

With cost\_bps\_per\_side \= 1.0, this gives:

crt=0.0003.

The cost model is deliberately simple. It is sufficient for a controlled friction-aware benchmark, but it should not be interpreted as a complete execution simulator.

The common entry-model evaluation logic is illustrated in Figure 3.4.

![][image6]

*Figure 3.4 \- Entry backtest and PnL*

All model families are evaluated under the same entry-model backtest: trade activation, direction selection, realized-event exit, and a shared transaction-cost proxy that converts gross PnL to net PnL.

### **3.9. Validation Design and Deployment-Oriented Model States**

The experiments use purged walk-forward validation. Each working sample is divided into:

1. a pre-holdout region used for model development.

2. a final holdout region used only for blind final evaluation.

Within the pre-holdout region, each walk-forward fold follows a chronological train-gap-validation-gap-test structure. The purge gaps are necessary because triple-barrier labels depend on future price evolution; adjacent observations can have overlapping future windows and would otherwise leak information across split boundaries. This design follows the financial machine learning recommendation that path-dependent labels require time-aware splitting and purging rather than ordinary random cross-validation \[23\]. It also reduces, but does not eliminate, model-selection and backtest-overfitting risk.

The purged walk-forward design is shown schematically in Figure 3.5.

![Figure 3.5 - Walk-forward validation][image7]

*Figure 3.5 \- Walk-forward validation and final-holdout model states*

Figure 3.5 shows the main leakage-control logic of the experiments. Training, validation, and test periods move forward chronologically, while purge gaps separate adjacent blocks whose triple-barrier label windows could otherwise overlap. The final holdout remains outside this cycle, so last\_CV and final\_refit are compared on a genuinely later blind segment rather than on a reused validation period.

The study distinguishes two model states:

1. last\_CV, the model from the last chronological walk-forward fold and the primary deployment-oriented reference.

2. final\_refit, the model refit on the largest possible pre-holdout sample and used as a larger-sample comparison rather than a replacement deployment state.

The main thesis benchmark uses last\_CV. This state is the most deployment-relevant reference because it approximates a model selected from the most recent chronological validation cycle before the final holdout. The final\_refit state adds a useful larger-sample comparison, but it cannot replace last\_CV: refitting changes the training sample, may change the score-to-trade conversion, and is not itself evidence that the same model would have been selected in a live walk-forward process.

### **3.10. Metrics**

The main empirical metrics are:

1. gross\_pnl, the sum of pre-cost directional trade returns.

2. net\_pnl, the sum of post-cost trade returns.

3. pnl\_per\_trade, the average post-cost trade return.

4. n\_trades, the number of executed trades.

5. trade\_rate, the fraction of eligible events that become executed trades.

6. sign\_accuracy, the fraction of trades for which the predicted side matches the realized event direction.

7. win\_rate, the fraction of executed trades with positive gross or net outcome as implemented in the evaluation table.

8. sharpe\_like, a scale-free diagnostic of mean trade return relative to trade-return dispersion.

9. dir\_auc, the AUC of the direction head.

10. trade\_auc, the AUC of the trade head.

11. rmse, the return-regression error diagnostic.

The primary economic metric is net\_pnl. The gross\_pnl metric separates raw signal extraction from the effect of transaction costs. The pnl\_per\_trade, n\_trades, and trade\_rate metrics show whether the result is supported by selective trading or by high turnover. The AUC metrics are valuable diagnostics for ranking quality, but they are not sufficient evidence of deployable profitability. The main text emphasizes the most interpretable subset of the eleven metrics available for both last\_CV and final\_refit, while Appendix B reports the wider diagnostic tables.

### **3.11. Fair-Comparison Principle**

Within each frequency regime, only two aspects are allowed to vary:

1. the family scaffold (base\_gnn, multigraph, memorygraph).

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

## **4\. Model Families**

### **4.1. Shared Architectural Conventions**

All three model families operate on the same node and edge tensors and produce the same multi-task outputs. They also share a hybrid edge-fusion mechanism that augments handcrafted relation features with learnable pairwise node interactions. The main architectural differences concern:

1. when relation channels are fused.  
2. whether the temporal backbone is convolutional or recurrent.  
3. whether the local graph operator is Conv-style or MPNN-style.

**Table 4.1. Controlled architectural comparison across the three model families.**

| Design axis | base\_gnn | multigraph | memorygraph |
| :---: | :---: | :---: | :---: |
| Relation handling | Early fusion of relation channels | Separate relation pathways before fusion | Relation-aware recurrent state |
| Graph pathway | Single graph operator block | One graph block per relation | Graph operator inside a memory loop |
| Temporal mechanism | Causal convolutional encoders | Causal convolutional encoders | Recurrent node and edge memory across contiguous chunks |
| Statefulness | Window-based, no persistent memory | Window-based, no persistent memory | Persistent node and edge memory carried across chunks |
| Main architectural hypothesis | A fused relation representation is sufficient | Preserving relation channels improves graph updates | Recurrent state captures short-lived high-frequency structure |

The three families therefore differ not by their input data, target construction, output heads, or evaluation protocol, but by the point at which relation information is fused and by whether temporal information is represented through convolutional windows or recurrent state. Chapter 4 should therefore be read as a controlled architectural ablation rather than as a comparison between models trained under different information sets.

The architectural comparison is summarized in Figure 4.1.

![][image8]  
*Figure 4.1 — Comparison of the Architectures* 

The family comparison is controlled because the models share the same data, target construction, and output heads; only the architecture of relation handling and temporal processing changes.

The two local graph operator types can be summarized as follows. The Conv-style operator applies a weighted source-node projection plus an edge-conditioned shift term. The MPNN-style operator computes gated messages conditioned on source node state, destination node state, and edge state. This makes the MPNN operator more expressive, but not automatically more profitable.

### **4.2. The base\_gnn Family**

The base\_gnn family is the single-graph baseline. It is evaluated through two adaptive operators, corresponding to the benchmark variants base-gnn-conv and base-gnn-mpnn:

1. adaptive\_conv  
2. adaptive\_mpnn

Architecturally, base\_gnn tests whether the three relation channels can be compressed into a single edge representation before graph message passing without losing economically useful information.

The temporal component is fully convolutional. Node inputs are projected into hidden space, augmented with learned asset embeddings, and processed by dilated causal residual convolution blocks. Edge inputs are processed by a separate temporal edge encoder. After graph processing and readout, the target-centered sequence is passed through a second causal temporal trunk.

base\_gnn forward path:

| X\_node \-\> NodeTemporalEncoder X\_edge \-\> EdgeTemporalEncoder (node\_seq, edge\_seq) \-\> HybridEdgeFeatureFusion relation edge states \-\> EdgeRelationFusion fused edges \+ node states \-\> SingleGraphOperatorBlock node states \-\> target/global GraphReadout readout sequence \-\> TargetTemporalTrunk shared state \-\> trade, direction, return, exit-type, time-to-event heads |
| :---- |

The graph component first fuses relation-aware edge features, then collapses the relation axis into a single edge representation. A single graph operator block is then applied using adaptive adjacency. Early relation fusion therefore happens before graph message passing, and after that fusion the model has only one graph pathway. This makes base\_gnn the clean single-graph baseline of the benchmark rather than merely a weaker version of the richer families.

The readout concatenates the target-node representation with global graph context, including mean and max pooling and optional target-to-global attention. The resulting target-centered representation is mapped to the shared multi-task prediction heads.

![][image9]  
*Figure 4.2. Detailed architecture of the base\_gnn family.*

Therefore, base\_gnn compresses relation channels **before** graph message passing, then runs one graph pathway over the fused edge representation.

### **4.3. The multigraph Family**

The multigraph family extends the baseline by preserving relation channels deeper into the graph-processing stage. It is evaluated in two matched variants, corresponding to multi-gnn-conv and multi-gnn-mpnn:

1. dynamic\_rel\_conv  
2. dynamic\_edge\_mpnn

The temporal component is structurally similar to base\_gnn: node and edge histories are encoded with dilated causal convolution blocks, and the target readout is processed by a causal temporal trunk. The difference is in graph processing. Instead of collapsing the relation axis before message passing, the model constructs a separate relation graph block for each relation channel.

multigraph forward path:

| X\_node \-\> NodeTemporalEncoder X\_edge \-\> EdgeTemporalEncoder (node\_seq, edge\_seq) \-\> HybridEdgeFeatureFusion price\_dep edges \-\> RelationGraphBlock(price\_dep) order\_flow edges \-\> RelationGraphBlock(order\_flow) liquidity edges \-\> RelationGraphBlock(liquidity) relation node states \-\> RelationAttentionFusion fused node states \-\> target/global GraphReadout readout sequence \-\> TargetTemporalTrunk shared state \-\> trade, direction, return, exit-type, time-to-event heads |
| :---- |

For each relation, the Conv variant computes dynamic edge scores and applies normalized source-node projections and edge-conditioned shifts. The MPNN variant uses gated messages conditioned jointly on source state, destination state, and edge state. After relation-specific processing, the model applies learned relation attention fusion. The key architectural contrast with base\_gnn is therefore that multigraph delays relation fusion until after message passing. Price-dependence, order-flow, and liquidity induce separate node updates before learned relation attention combines them.

The central design question for multigraph is not simply whether a more complex model helps. It is whether relation semantics should remain separated during message passing, so that price-dependence, order-flow, and liquidity can shape node updates differently before being merged into a shared representation.

*![][image10]*  
*Figure 4.3. Detailed architecture of the multigraph family.*

Therefore, multigraph preserves relation-specific semantics through message passing and only fuses them **after** relation-specific graph updates.

### **4.4. The memorygraph Family**

The memorygraph family is the most distinct architecture in the study. It is evaluated with two variants, memory-gnn-conv and memory-gnn-mpnn:

1. conv  
2. mpnn

Unlike base\_gnn and multigraph, it does not rely on a deep causal-convolutional temporal encoder. Instead, it uses stateful recurrent memory. Raw node and edge inputs are first projected at each time step. A MemoryAugmentedGraphBlock then maintains node memory and relation-specific edge memory across contiguous chunks.

memorygraph recurrent path:

| X\_node\_t \-\> NodeStepProjector X\_edge\_t \-\> EdgeStepProjector (node\_t, edge\_t) \-\> HybridEdgeFeatureFusion edge\_t \+ source/destination node context \+ previous edge\_memory \-\> EdgeMemoryUpdater edge-enriched state \-\> AdaptiveGraphConnectivity \+ graph operator relation node states \+ relation edge-memory context \+ previous node\_memory \-\> NodeMemoryUpdater fused node sequence \-\> target/global GraphReadout readout state \-\> output projection shared state \-\> trade, direction, return, exit-type, time-to-event heads |
| :---- |

The edge memory update uses recurrent cells conditioned on current edge state, source-node state, destination-node state, and pairwise node interactions. The node memory update aggregates relation-specific edge-memory context to nodes, fuses relation-specific node and edge contexts, and updates node memory with another recurrent cell. Training uses contiguous stateful chunks with truncated backpropagation through time. In memorygraph, temporal modelling is therefore not primarily a convolution over a fixed lookback window. Temporal information is stored in recurrent node and relation-specific edge memories that are updated step by step across contiguous chunks.

Inside each recurrent step, the graph operator is either Conv-style or MPNN-style. The key difference from the other families is that graph interaction occurs inside a recurrent memory loop. The operator acts on state-enriched representations rather than on a fully pre-encoded temporal sequence. This makes memorygraph qualitatively different from the convolutional window-based logic of base\_gnn and multigraph, even when the same local Conv-versus-MPNN comparison is preserved.

This gives memorygraph a qualitatively different inductive bias:

1. base\_gnn uses early relation fusion and convolutional temporal modelling.  
2. multigraph uses late relation fusion and convolutional temporal modelling.  
3. memorygraph uses relation-aware recurrent state and stateful graph updates.

Figure 4.4 highlights the recurrent memory mechanism that differentiates memorygraph from the convolutional temporal families.   
![][image11]  
Therefore, temporal modelling in this case is represented through **recurrent node and edge memory**, and graph interaction occurs **inside** the recurrent update loop. 

Taken together, the three model families define the architectural axes used in the empirical comparison: early relation fusion, late relation fusion, and recurrent relation-aware memory. This provides the structure for the Results chapter, where family design is evaluated jointly with operator choice, temporal frequency, and deployment-oriented model state.

## **5\. Results**

This chapter reports the empirical benchmark. The main evidence is the deployment-oriented last\_CV comparison across all eighteen primary model-frequency configurations. The chapter then discusses frequency-specific outcomes, answers the research questions, compares selected last\_CV and final\_refit cases, and evaluates the hypotheses.

The main interpretive rule is that net\_pnl is the primary economic outcome, gross\_pnl indicates pre-cost signal extraction, and n\_trades, trade\_rate, and pnl\_per\_trade are necessary for understanding whether the economic result is operationally meaningful. AUC values, sign\_accuracy, win\_rate, sharpe\_like, and rmse are interpreted as diagnostics, not as sufficient evidence of tradability. Appendix B reports additional diagnostic metrics and explains the formula-helper rows used for last\_CV versus final\_refit comparisons.

### **5.1. Benchmark Overview**

Table 5.1 reports the main last\_CV benchmark. Within each frequency, the six models are directly comparable because they use the same input representation, target construction, validation logic, and event-based backtest. The 5min and 1min regimes are also directly comparable to each other because they solve the same 30-minute lookback / five-minute horizon task. The 1sec regime should be interpreted as a high-frequency stress test with its own adapted task.

Figure 5.1 provides a visual summary of the benchmark grid before the exact numerical values are reported in Table 5.1.

![][image12]

*Figure 5.1. Benchmark overview by frequency, graph family, and operator.*

Figure 5.1 visualizes `last_CV` net PnL (`net_pnl`) across all eighteen model-frequency configurations, grouped by temporal resolution, graph family, and graph operator. The main visual conclusion is that positive post-cost results are concentrated in the slower shared-task regimes, especially the base-gnn Conv specification, while every one-second configuration remains negative after costs.

**Table 5.1. last\_CV benchmark overview across all model-frequency configurations.**

| Frequency | Model | gross\_pnl | net\_pnl | N trades | dir\_auc | trade\_auc |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 5min | base-gnn-conv | 0.028156 | **0.020356** | 26 | 0.617105 | 0.700447 |
| 5min | base-gnn-mpnn | 0.014415 | 0.006915 | 25 | 0.614912 | 0.727631 |
| 5min | multi-gnn-conv | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| 5min | multi-gnn-mpnn | 0.002941 | \-0.009359 | 41 | **0.625439** | 0.707304 |
| 5min | memory-gnn-conv | 0.009459 | 0.004359 | 17 | 0.611842 | 0.734196 |
| 5min | memory-gnn-mpnn | \-0.012463 | \-0.037363 | 83 | 0.537719 | 0.726026 |
| 1min | base-gnn-conv | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| 1min | base-gnn-mpnn | \-0.024239 | \-0.078539 | 181 | 0.504512 | 0.642097 |
| 1min | multi-gnn-conv | 0.025767 | \-0.007833 | 112 | 0.541996 | 0.634787 |
| 1min | multi-gnn-mpnn | \-0.003371 | \-0.035771 | 108 | 0.528405 | 0.645004 |
| 1min | memory-gnn-conv | \-0.031247 | \-0.078947 | 159 | 0.479399 | 0.638928 |
| 1min | memory-gnn-mpnn | 0.033605 | 0.009305 | 81 | 0.529844 | 0.635665 |
| 1sec | base-gnn-conv | 0.139515 | \-0.094185 | 779 | 0.599753 | 0.841804 |
| 1sec | base-gnn-mpnn | 0.052679 | \-0.065821 | 395 | 0.599093 | 0.848558 |
| 1sec | multi-gnn-conv | 0.081710 | \-0.108790 | 635 | 0.597838 | 0.839544 |
| 1sec | multi-gnn-mpnn | 0.079777 | \-0.080723 | 535 | 0.600538 | **0.868370** |
| 1sec | memory-gnn-conv | **0.412032** | \-1.163268 | **5251** | 0.588785 | 0.490050 |
| 1sec | memory-gnn-mpnn | 0.223788 | \-0.280512 | 1681 | 0.596713 | 0.863699 |

Figure 5.2 moves the benchmark interpretation from endpoint totals to path-level behaviour. It shows the cumulative gross PnL paths for three representative final-holdout models: the best 5min model, the best 1min model, and the least-negative 1sec model after costs. The ETH midpoint index is included only as a market-context reference. The paths should therefore not be read as a price-following strategy chart, but as an event-based visualization of when realized trade exits add or subtract PnL.

![][image13]  
*Figure 5.2. Cumulative gross PnL paths for representative final-holdout models.*

The figure is useful for two reasons. First, all three representative models finish positive before transaction costs even though the ETH midpoint index declines over the same displayed window. This suggests that the gross PnL is not simply a passive exposure to a rising ETH market. Second, the paths show different ways of generating signal. The 5min base-gnn-conv path is sparse and stair-stepped because it executes only 26 trades in the full final holdout. The 1min base-gnn-conv path is more active and reaches the largest gross total among the representative shared-task models. The 1sec base-gnn-mpnn path also remains positive before costs, but it does so with substantially higher turnover than the 5min model.

Figure 5.3 repeats the same comparison after transaction costs. This is the deployment-relevant version of the path comparison because the thesis treats `net_pnl` as the primary economic metric. The visual comparison clarifies why the gross result alone is insufficient. The 1min base-gnn-conv model extracts more pre-cost signal than the 5min base-gnn-conv model, but it also executes many more trades. As a result, the two models finish with almost identical net PnL despite very different gross PnL and turnover profiles.

![][image14]  
*Figure 5.3. Cumulative net PnL paths for representative final-holdout models.*

The contrast between Figure 5.2 and Figure 5.3 is central to the main interpretation. The 5min base-gnn-conv model ends at `net_pnl = 0.020356` with 26 trades, while the 1min base-gnn-conv model ends at `net_pnl = 0.020094` with 132 trades. The 1min model therefore finds more gross opportunity, but a larger part of that opportunity is consumed by the fixed round-trip cost proxy. The 5min model is less active but more efficient per trade. The 1sec base-gnn-mpnn model illustrates the same issue more strongly: its gross PnL is positive, but its net PnL ends at `-0.065821` after costs.

Three patterns define the main benchmark.

First, base-gnn-conv is the strongest deployment-oriented model at both shared-task frequencies. At 5min it achieves `net_pnl = 0.020356`, and at 1min it achieves `net_pnl = 0.020094`. Figure 5.3 strengthens this conclusion by showing that both positive results are path-level outcomes, not only endpoint table values.

Second, the one-minute regime should not be described as clearly superior in net economic terms. It is superior in gross signal extraction relative to the 5min representative model, but not in post-cost profitability. The correct interpretation is that the 1min regime reveals more trading signal, while the 5min regime converts a smaller amount of signal into a similarly strong net result through lower turnover.

Third, the one-second regime separates signal extraction from deployability. Figure 5.2 shows that a positive gross signal exists at 1sec, but Figure 5.3 shows that this signal does not survive the transaction-cost proxy. The one-second result is therefore not simply a failure of prediction. It is a failure of cost-aware selectivity under the current entry-policy benchmark.

### **5.2. Frequency-Specific Results**

#### *5.2.1. Five-Minute Regime*

The 5min regime produces the clearest economically positive block of results. The best model is base-gnn-conv, with gross\_pnl \= 0.028156, net\_pnl \= 0.020356, and 26 trades. The second-best model is base-gnn-mpnn, with net\_pnl \= 0.006915 over 25 trades. The two baseline variants therefore occupy the top two economic positions.

The more complex families remain informative but not dominant. multi-gnn-conv is mildly positive, with net\_pnl \= 0.001958, while multi-gnn-mpnn is negative. memory-gnn-conv is also mildly positive, with net\_pnl \= 0.004359, while memory-gnn-mpnn is the weakest five-minute model with net\_pnl \= \-0.037363 and 83 trades.

The ranking metrics show why economic interpretation cannot rely on AUC alone. multi-gnn-mpnn has the highest dir\_auc in the five-minute block (0.625439), and memory-gnn-conv has the highest trade\_auc (0.734196). Neither is the best economic model. The best deployable outcome comes from the baseline Conv model, which combines a positive gross signal with a modest number of trades and limited cost drag.

#### *5.2.2. One-Minute Regime*

The 1min regime is the richer shared-task stress test because it uses the same 30-minute lookback and five-minute horizon as the five-minute regime, but with more granular input information. The winner remains base-gnn-conv, with gross\_pnl \= 0.059694, net\_pnl \= 0.020094, and 132 trades.

This result is important because the gross signal is much larger than at five-minute frequency, but the net result is almost identical. The reason is turnover. The one-minute model extracts more pre-cost signal, but the larger number of trades absorbs most of the incremental edge through transaction costs. The one-minute benchmark is therefore not stronger in strict net-profit terms, but it is stronger as a stress test of whether signal survives more active trading.

The second-best one-minute model is memory-gnn-mpnn, with net\_pnl \= 0.009305 over 81 trades. This is the strongest shared-task result for memorygraph and suggests that recurrent memory can be useful at minute-level resolution. However, the result remains below the baseline Conv winner.

The multigraph family does not produce positive net PnL at one-minute frequency. multi-gnn-conv has positive gross PnL (0.025767) but ends at net\_pnl \= \-0.007833; multi-gnn-mpnn is also negative. This does not show that relation-specific modelling contains no information. It shows that, under the present thresholding and cost assumptions, relation-specific information does not translate into superior net profitability.

#### *5.2.3. One-Second Regime*

The 1sec regime creates the sharpest separation between gross signal and net deployability. All one-second models finish negative on `net_pnl`. The least-negative model is base-gnn-mpnn with `net_pnl = -0.065821`, followed by multi-gnn-mpnn with `net_pnl = -0.080723`, base-gnn-conv with `net_pnl = -0.094185`, and multi-gnn-conv with `net_pnl = -0.108790`. The memorygraph variants are substantially more negative after costs.  
The gross results tell a different story. Figure 5.4 highlights this divergence between gross and net PnL in the one-second regime.

![Figure 5.2 - Gross vs net PnL][image15]

*Figure 5.4 \- Gross vs net PnL for 1sec models*

Figure 5.4 shows that all one-second models contain some pre-cost trading signal, but none of them preserve it after the transaction-cost proxy. The gap between the gross and net bars is therefore the key result of the figure: one-second performance is dominated by cost drag rather than by the absence of directional information. memory-gnn-conv produces the largest gross signal in the entire benchmark, with `gross_pnl = 0.412032`, but it is also the clearest example of this cost problem. memory-gnn-mpnn produces the second-largest one-second gross signal, with `gross_pnl = 0.223788`. These values show that the memory architecture is not simply failing to detect high-frequency structure. The problem is that its signal is too trade-intensive under the current cost model.

This point is visible in Figure 5.5. The solid orange line shows that memory-gnn-conv accumulates positive gross PnL over the final holdout. However, the dashed orange line declines strongly because the model executes 5251 trades. With the benchmark round-trip cost proxy of `0.0003`, the implied cumulative cost drag is approximately `1.5753`, which is far larger than the model's gross PnL of `0.412032`. The model therefore loses money after costs even though its pre-cost directional signal is positive.

**![][image16]**  
*Figure 5.5. One-second cumulative gross-versus-net PnL paths for memory-gnn-conv and base-gnn-mpnn.*

The base-gnn-mpnn lines in Figure 5.5 provide a lower-turnover reference. This model also has positive gross PnL and negative net PnL, but its cost drag is much smaller because it executes 395 trades rather than 5251\. Its implied cost drag is approximately `0.1185`, compared with `1.5753` for memory-gnn-conv. The visual scale is dominated by the memory model, so the baseline appears compressed, but the comparison is still informative: both models face the same cost rule, while the high-turnover memory model is punished much more severely.

The Appendix B diagnostics support the same interpretation. The one-second memory-gnn-conv result has `trade_rate = 0.056665`, which is far above the `trade_rate = 0.004263` of the one-second base-gnn-mpnn result. Its `sign_accuracy = 0.584841` is not poor in isolation, but its `pnl_per_trade = -0.000222` and `sharpe_like = -28.865252` show that small post-cost losses accumulate rapidly under high turnover. These diagnostics are reported in Appendix B because they support interpretation without changing the primary ranking by `net_pnl`.

The one-second evidence is central to the thesis. It supports a two-layer conclusion: memory-based graph models can extract high-frequency gross signal, but this signal is not sufficiently selective under the current benchmark. The main unresolved problem is therefore not only representation learning. It is the conversion of high-frequency signal into sparse, stable, and cost-aware trading decisions.

Across the three frequency regimes, the five-minute setting is the most efficient post-cost shared-task benchmark, while the one-minute setting reveals richer gross signal without a superior net outcome. The one-second setting operates as a turnover and cost-drag stress test, which motivates the research-question answers that follow, beginning with the family-level comparison in RQ1.

### **5.3. Answer to RQ1: Which Graph Family Performs Best?**

The answer to RQ1 is that base\_gnn performs best overall under the controlled entry-model benchmark.

The strongest evidence comes from the two shared-task regimes. At 5min, the best model is base-gnn-conv with net\_pnl \= 0.020356. At 1min, the best model is again base-gnn-conv, with net\_pnl \= 0.020094. Both results are positive, and both are obtained by the same family and operator.

At 1sec, no family produces positive net PnL. This means the high-frequency regime cannot be used to identify a robust deployment winner. Instead, it shows that all families face a cost and turnover barrier under the current entry policy.

The family-level conclusion is therefore conservative but clear. Under fixed targets, fixed features, fixed validation, fixed thresholds, and fixed event-based exits, the simpler single-graph baseline is the most reliable architecture. The richer families may contain useful signal, but they do not produce a stronger post-cost benchmark result.

### **5.4. Answer to RQ2: How Important Is the Conv-versus-MPNN Operator Choice?**

The operator choice is important, but its effect is not universal.

At 5min, Conv outperforms MPNN on net\_pnl in all three families:

1. base\_gnn: 0.020356 versus 0.006915.

2. multigraph: 0.001958 versus \-0.009359.

3. memorygraph: 0.004359 versus \-0.037363.

At 1min, Conv remains better for base\_gnn and multigraph, but memorygraph reverses in favor of MPNN:

1. base\_gnn: 0.020094 versus \-0.078539.

2. multigraph: \-0.007833 versus \-0.035771.

3. memorygraph: \-0.078947 versus 0.009305.

At 1sec, all models are net negative, but MPNN is less negative than Conv in each family:

1. base\_gnn: \-0.065821 versus \-0.094185.

2. multigraph: \-0.080723 versus \-0.108790.

3. memorygraph: \-0.280512 versus \-1.163268.

The operator therefore changes economic outcomes materially. The strongest shared-task model is Conv-based, but the least negative one-second models are MPNN-based. The correct conclusion is not that Conv is always better or that MPNN is always better. The operator must be selected jointly with the family scaffold, frequency regime, and cost-sensitive trading policy.

### **5.5. Answer to RQ3: How Does Temporal Resolution Affect Relation and Memory Mechanisms?**

The results do not support the hypothesis that finer temporal resolution automatically increases the net economic value of richer graph mechanisms.

For multigraph, relation-specific processing does not beat the baseline on net\_pnl at any frequency. At five-minute frequency, it is mildly positive in the Conv variant but below the baseline. At one-minute frequency, both variants are net negative. At one-second frequency, both variants have positive gross signal but negative net PnL.

For memorygraph, the answer is more nuanced. The family becomes most distinctive at one-second frequency, exactly where stateful memory was expected to matter most. Its gross results are the largest in the benchmark. This provides partial evidence that recurrent memory can surface high-frequency opportunities. However, the same results show that memory also produces excessive trading activity under the current policy. The net effect is strongly negative.

The best interpretation is therefore two-layered. Finer temporal resolution appears to increase the amount of extractable short-horizon signal, especially for memory-based models. At the same time, it increases the penalty for insufficient trade selectivity. In the current benchmark, the cost and turnover effect dominates the signal-extraction effect.

### **5.6. Answer to RQ4: Are Conclusions Stable Between last\_CV and final\_refit?**

The deployment-state comparison shows that last\_CV and final\_refit are related but not interchangeable. The last\_CV state remains the main deployment reference because it is produced by the final chronological cross-validation fold. The final\_refit state is useful because it tests what happens when a model is refit on a larger pre-holdout sample, but it does not replace the chronological evidence.

The conceptual distinction between these states is summarized in Figure 5.6 before the selected numerical comparisons are reported.

*![][image17]*

*Figure 5.6. `last_CV` versus `final_refit` as deployment-oriented model states*

Figure 5.6 shows that refitting is informative but not mechanically beneficial. Several final_refit points improve ranking diagnostics, yet the economic arrows do not move uniformly upward: the five-minute multigraph case improves strongly, the one-minute baseline worsens, and the selected one-second memory case remains negative after costs. The selected comparisons in Sections 5.6.1-5.6.4 therefore treat final_refit as a larger-sample robustness comparison, while last_CV remains the primary deployment-oriented state defined in Section 3.9.

#### *5.6.1. Best Five-Minute Model: base-gnn-conv*

Table 5.2 reports the last\_CV and final\_refit comparison for the best five-minute model.

**Table 5.2. last\_CV versus final\_refit for the best five-minute model.**

| Frequency | Training cycle | gross\_pnl | net\_pnl | n\_trades | dir\_auc | trade\_auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | last\_CV | 0.028156 | 0.020356 | 26 | 0.617105 | 0.700447 |
| 5min | final\_refit | 0.017570 | 0.011270 | 21 | 0.630702 | 0.721795 |

The five-minute winner remains positive after refitting. Its net PnL declines, but its ranking metrics improve. This is the cleanest deployment-stability case in the selected comparisons. It also shows why final refitting should not be assumed to improve the main economic metric: more training data improve AUC here, but not net\_pnl.

#### *5.6.2. Best One-Minute Model: base-gnn-conv*

Table 5.3 reports the same deployment-state comparison for the best one-minute model.

**Table 5.3. last\_CV versus final\_refit for the best one-minute model.**

| Frequency | Training cycle | gross\_pnl | net\_pnl | n\_trades | dir\_auc | trade\_auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1min | last\_CV | 0.059694 | 0.020094 | 132 | 0.525927 | 0.648157 |
| 1min | final\_refit | 0.007198 | \-0.012002 | 64 | 0.524286 | 0.635712 |

The one-minute winner is less stable. The last\_CV model is clearly positive, while the final\_refit version turns negative. The AUC values change only modestly, which suggests that the underlying ranking quality remains similar while the score-to-trade conversion becomes less economically favorable. This case reinforces the deployment argument: a model can look similar in predictive diagnostics but materially different in realized trading performance.

#### *5.6.3. Selected One-Second Memorygraph Case: memory-gnn-conv*

Table 5.4 reports the selected one-second memorygraph comparison because this case is the clearest high-frequency turnover example.

**Table 5.4. last\_CV versus final\_refit for the selected one-second memorygraph case.**

| Frequency | Training cycle | gross\_pnl | net\_pnl | n\_trades | dir\_auc | trade\_auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1sec | last\_CV | 0.412032 | \-1.163268 | 5251 | 0.588785 | 0.490050 |
| 1sec | final\_refit | 0.443031 | \-0.954969 | 4660 | 0.592186 | 0.852874 |

The selected one-second case is the most informative high-frequency stress example. Refitting increases gross PnL, reduces the trade count, improves trade\_auc, and makes the net result less negative. Nevertheless, the model remains strongly unprofitable after costs. The central one-second conclusion therefore survives refitting: memory-based high-frequency signal is present, but it is not sufficiently selective under the current benchmark.

#### *5.6.4. Informative Five-Minute Refit Case: multi-gnn-conv*

Table 5.5 reports the five-minute multigraph refit case because it shows that final\_refit can materially change model-level interpretation.

**Table 5.5. last\_CV versus final\_refit for the informative five-minute multigraph case.**

| Frequency | Training cycle | gross\_pnl | net\_pnl | n\_trades | dir\_auc | trade\_auc |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | last\_CV | 0.012758 | 0.001958 | 36 | 0.616667 | 0.672097 |
| 5min | final\_refit | 0.061167 | 0.041367 | 66 | 0.703947 | 0.720920 |

This case is analytically important because the final\_refit version of multi-gnn-conv reaches dir\_auc \> 70% and a strongly positive net\_pnl. It shows that relation-preserving graph processing can become highly effective under a larger-sample refit. However, it does not overturn the primary last\_CV conclusion. In the deployment-oriented state, multi-gnn-conv is only weakly positive and remains below base-gnn-conv. The case should therefore be interpreted as evidence of refit sensitivity and potential future value, not as proof that multigraph is already the strongest deployable family.

Overall, the selected state comparisons show that refitting can improve diagnostics and can sometimes improve `net_pnl`, but it does not uniformly improve deployability. The mixed economic response means that last\_CV remains the deployment-oriented benchmark reference. The final\_refit state instead provides complementary evidence about larger-sample sensitivity before the hypothesis assessment summarizes this distinction.

### **5.7. Hypothesis Assessment**

#### *H1. The one-minute regime should be the strongest shared-task benchmark.*

H1 is not supported on the primary economic metric. The strongest one-minute model reaches net\_pnl \= 0.020094, while the strongest five-minute model reaches net\_pnl \= 0.020356. The difference is small, but the hypothesis predicts one-minute superiority, which is not observed. The one-minute regime remains important because it produces more trades and much larger gross signal, but it is not the strongest shared-task regime in strict net terms.

#### *H2. Explicit multi-relation modelling should outperform the simpler baseline more clearly at finer resolutions.*

H2 is not supported on the primary economic benchmark. multigraph does not beat base\_gnn on net\_pnl at five-minute, one-minute, or one-second frequency. The interesting multi-gnn-conv final-refit case provides evidence that relation-specific modelling can become strong under some training states, but the deployment-oriented last\_CV benchmark does not support a general multigraph advantage.

#### *H3. Stateful memory should become more valuable as the market is observed more finely.*

H3 is partially supported for gross signal extraction but not supported for net deployable performance. At one-second frequency, memorygraph produces the strongest gross PnL values in the study. However, both memory variants remain net negative, and memory-gnn-conv is especially negative because of excessive turnover. Memory helps reveal short-lived signal, but the current benchmark does not show that it improves post-cost profitability.

#### *H4. Conv and MPNN operators should not be uniformly dominant across families.*

H4 is supported. Conv dominates the five-minute net results and produces the strongest shared-task model overall. At one-minute frequency, Conv remains better for base\_gnn and multigraph, while MPNN is better for memorygraph. At one-second frequency, MPNN is less negative in all families on net PnL. The operator effect is therefore economically meaningful and frequency-dependent.

#### *H5. last\_CV and final\_refit should provide broadly consistent but not necessarily equivalent evidence.*

H5 is partially supported. The selected comparisons do not overturn the broad conclusion that base\_gnn is the most reliable deployment-oriented family and that one-second models remain net negative. However, the model-level story can change materially, and the economic response is mixed rather than mechanically positive. The one-minute base-gnn-conv turns negative after refitting, while the five-minute multi-gnn-conv becomes very strong after refitting and several diagnostics improve. The evidence therefore supports reporting both states transparently and prioritizing last\_CV for deployment interpretation.

### **5.8. Summary of Research Question Answers**

Table 5.6 consolidates the answers to the four research questions. The table is included to make the empirical storyline explicit before the thesis moves from results to discussion.

**Table 5.6. Summary of answers to the research questions.**

| Research question | Main empirical answer | Key evidence | Interpretation |
| :---- | :---- | :---- | :---- |
| RQ1. Which graph family performs best under a controlled entry-model benchmark? | base\_gnn performs best overall. | base-gnn-conv is the strongest last\_CV model at both 5min and 1min, with net\_pnl \= 0.020356 and net\_pnl \= 0.020094, respectively. | The simpler single-graph baseline is the most reliable architecture under the tested benchmark. |
| RQ2. How important is the Conv-versus-MPNN operator choice? | Operator choice materially changes outcomes, but no operator is universally dominant. | Conv is strongest in the shared-task winners, while MPNN is less negative for all families at 1sec. | Operator choice should be evaluated jointly with family scaffold, frequency, and cost-sensitive thresholding. |
| RQ3. How does temporal resolution affect relation and memory mechanisms? | Higher temporal resolution increases gross signal visibility but also increases turnover and cost pressure. | memory-gnn-conv has the largest 1sec gross signal, but it is strongly net negative because it executes 5251 trades. | Finer data can reveal short-lived signal, but selectivity becomes the main bottleneck. |
| RQ4. Are conclusions stable between last\_CV and final\_refit? | The broad story is stable, but model-level economics can change substantially. | 5min base-gnn-conv remains positive after refit, 1min base-gnn-conv turns negative, and 5min multi-gnn-conv becomes very strong after refit. | final\_refit is informative as a larger-sample comparison, but last\_CV remains the primary deployment-oriented reference. |

### **5.9. Summary of Hypothesis Assessment**

Table 5.7 summarizes the hypothesis assessment. This summary separates unsupported, partially supported, and supported claims and clarifies which conclusions are based on net economic outcomes rather than only on predictive diagnostics.

**Table 5.7. Summary of hypothesis assessment.**

| Hypothesis | Status | Main reason |
| :---- | :---- | :---- |
| H1. The one-minute regime should be the strongest shared-task benchmark. | Not supported | The best one-minute net result (net\_pnl \= 0.020094) is slightly below the best five-minute net result (net\_pnl \= 0.020356). |
| H2. Explicit multi-relation modelling should outperform the simpler baseline more clearly at finer resolutions. | Not supported | multigraph does not beat base\_gnn on last\_CV net\_pnl at 5min, 1min, or 1sec. |
| H3. Stateful memory should become more valuable as the market is observed more finely. | Partially supported | memorygraph produces the strongest one-second gross signal, but it remains strongly negative after costs. |
| H4. Conv and MPNN operators should not be uniformly dominant across families. | Supported | Conv dominates the shared-task winners, while MPNN is less negative across all families at 1sec. |
| H5. last\_CV and final\_refit should provide broadly consistent but non-equivalent evidence. | Partially supported | The broad deployment conclusion remains cautious, while diagnostics and individual model outcomes can change materially after refitting without implying uniform `net_pnl` improvement. |

## **6\. Discussion**

### **6.1. Main Findings by Research Question**

The main finding is that architectural complexity did not guarantee better trading outcomes. Under the strict shared-task benchmark, the simpler base\_gnn family is strongest, and the winning specification is base-gnn-conv. This result is methodologically important because it was obtained under a common target construction, common output interface, common thresholding framework, and common event-based backtest.

For RQ1, the strongest family is base\_gnn. For RQ2, the operator effect is substantial and conditional, but Conv is the stronger shared-task choice. For RQ3, finer temporal resolution increases the visibility of gross high-frequency signal, especially for memorygraph, but it also increases cost drag and turnover risk. For RQ4, last\_CV and final\_refit provide complementary but non-equivalent evidence: final\_refit can improve diagnostics, but it does not uniformly improve `net_pnl` or replace the deployment-oriented last\_CV reference.

A central theme is that predictive quality, gross signal extraction, and net profitability must be separated. The 1sec memorygraph results make this most visible. A model can generate a large positive gross PnL and still fail economically because the number of trades is too high. Conversely, a model with less dramatic ranking statistics can be more useful if it is more selective.

### **6.2. Comparison with Previous Work**

The results are consistent with the broader LOB literature in one respect: short-horizon market data contain learnable structure \[3\]-\[4\]-\[5\], \[25\]. The presence of positive gross PnL in several models, especially at 1sec, supports the idea that order book and cross-asset microstructure features contain information about subsequent movement.

However, the results also qualify optimistic interpretations of deep learning for market prediction. DeepLOB and related neural LOB studies emphasize the ability of convolutional, recurrent, and attention-based models to learn from high-dimensional order book states \[5\], \[19\], \[25\]. Recent work such as HLOB further suggests that structured representations can capture information persistence inside the book \[20\]. This thesis shows that, in a graph-based crypto setting, representation learning alone is not sufficient. The economic translation layer matters. Thresholds, trade frequency, costs, and deployment state can change the conclusion even when ranking metrics appear reasonable.

The graph-learning literature motivates the use of relation-aware models, and recent financial graph studies motivate multi-relational dynamic modelling \[6\]-\[8\]-\[9\]-\[10\]-\[11\]-\[12\]-\[13\]-\[14\], \[21\], \[22\]. The present findings are more cautious. Multi-relation modelling is plausible and sometimes useful, but in the primary last\_CV benchmark it does not beat the simpler baseline. This does not contradict the graph literature; rather, it shows that graph complexity must be evaluated against the specific economic objective, market regime, asset universe, and cost model.

The temporal graph and memory literature also provides a useful comparison \[11\], \[12\], \[21\]. Temporal memory is designed to preserve information across changing graph states. The memorygraph results support this idea before costs: the largest 1sec gross signals come from memory-based models. Yet they also show that memory without sufficiently selective trading control can amplify turnover. In market microstructure, the ability to detect many short-lived opportunities is not enough if the opportunities are too small relative to costs.

### **6.3. Scientific Implications**

The scientific implication is that graph-based market microstructure research should evaluate architecture under deployment-aware metrics, not only under predictive metrics. AUC, accuracy, and regression error remain useful, but they do not settle whether a model is economically meaningful. The benchmark must include trade count, gross PnL, net PnL, and cost sensitivity.

A second implication is that relation modelling should be treated as an empirical design choice rather than an assumed improvement. Preserving multiple relation channels is theoretically attractive, especially in finance, but the main benchmark shows that early fusion in a simpler baseline can outperform late relation-specific processing under some conditions.

A third implication concerns temporal resolution. Higher frequency can increase the quantity of detectable signal, but it also raises the cost of acting on that signal. Research that evaluates high-frequency models without explicit turnover and cost analysis risks overstating practical value.

### **6.4. Practical and Deployment Implications**

The strongest deployment-oriented evidence favors base-gnn-conv in the shared-task regimes. Even this conclusion should be interpreted cautiously. The model is not a production trading system; it is the best entry model in a controlled final-holdout benchmark. A real deployment would require latency modelling, exchange-specific fee and slippage assumptions, monitoring for regime drift, capital constraints, live order placement logic, risk controls, and operational monitoring.

last\_CV is deployment-relevant because it resembles the chronological situation of using the latest available model on an unseen future segment. final\_refit adds information about whether a larger training sample changes the holdout result, but it cannot replace last\_CV. The 1min example demonstrates why: the refit model preserves similar ranking diagnostics but turns net negative. The 5min multigraph Conv example demonstrates the opposite possibility: a model that is weak in last\_CV can become strong under final refit. Both cases show that deployment interpretation depends on model state.

The main results imply that realistic deployment should prioritize selectivity and cost robustness. The high-frequency memory models are not suitable for deployment in their current form because their gross edge is overwhelmed by turnover. Future deployment-oriented work should therefore evaluate trade-rate controls, threshold stability, cost sensitivity, and no-trade calibration before treating high-frequency graph signals as practically useful.

From an impact perspective, the immediate value of the thesis is methodological rather than operational. It provides a reproducible benchmark for comparing graph architectures under chronological validation and transaction costs. Potential positive impacts include better model-selection discipline, clearer reporting of high-frequency ML limitations, and reduced risk of over-interpreting ranking metrics. Potential negative impacts include encouraging automated trading experiments without adequate risk controls if the results are misread as live-profitability evidence. Regulatory and ethical considerations are therefore linked to transparency, overfitting control, risk disclosure, and the avoidance of claims that exceed the evidence.

Figure 6.1 summarizes the practical interpretation chain that connects predictions to deployment-oriented evidence.

**Figure 6.1. Deployment interpretation from prediction to post-cost evidence.**  

 prediction quality  
 (ranking diagnostics)  
         │  
         ▼  
 gross signal extraction  
 (\`gross\_pnl\` before cost)  
         │  
         ▼  
 trade selectivity / turnover  
 (\`pnl\_per\_trade\`, \`n\_trades\`, \`trade\_rate\`)  
         │  
         ▼  
 transaction-cost adjustment  
 (gross edge must survive frictions)  
         │  
         ▼  
 model-state stability  
 (\`last\_CV\` vs \`final\_refit\` interpretation)  
         │  
         ▼  
 deployment-informative evidence  
 (`net_pnl` after cost, with stable interpretation)

The figure should be read as a filtering chain rather than as a simple performance ladder. A model can pass the earlier diagnostic steps and still fail later if turnover, transaction costs, or model-state instability remove the economic value. This is exactly the pattern observed in the one-second memory models: predictive and gross-PnL evidence exists, but it does not survive the deployment-oriented filters.

### **6.5. Limitations, Weaknesses, and Sources of Bias**

Several limitations affect the interpretation of the thesis. They do not invalidate the benchmark, but they define the scope within which the conclusions are valid.

First, the asset universe is small. The graph contains only ADA, BTC, and ETH, with ETH as the sole target asset. This improves interpretability and keeps the architecture comparison controlled, but it creates asset-selection bias. The results may not transfer to larger crypto universes, equities, futures, foreign exchange, or less liquid instruments. A larger universe could also change the value of multigraph, because more assets would create more relation pathways.

Second, the data source is a public Kaggle dataset rather than a proprietary exchange feed. The order book snapshots are suitable for a thesis benchmark, but they are not equivalent to full message-level exchange data. They do not contain all order submissions, cancellations, queue-position information, partial fills, latency measurements, or execution reports. This snapshot-based design limits what can be concluded about live execution.

Third, the benchmark may contain market-regime and temporal-slice bias. A chronological final holdout is more realistic than random splitting, but it still represents a particular late segment of the available sample. If this interval has unusual volatility, liquidity, or directional behavior, the measured ranking of architectures may partly reflect that regime. A broader study would repeat the benchmark across multiple calendar periods, volatility regimes, and market states.

Fourth, the label construction introduces label-construction bias. Triple-barrier labels depend on barrier widths, rolling volatility estimates, timeout treatment, and the decision to mask some direction labels. These choices are defensible and shared across models, but they shape what the models learn. A different barrier system could change the apparent value of relation-preserving or memory-based architectures.

Fifth, threshold-selection bias remains possible. Trade and direction thresholds are selected from finite validation grids and then applied to the final holdout. This is more disciplined than tuning on the holdout itself, but realized trading performance still depends on the threshold grid, feasibility constraints, and no-trade calibration. The one-second results indicate that threshold design is one of the main unresolved weaknesses of the benchmark.

Sixth, the transaction-cost model is simplified. A constant round-trip proxy is useful for controlled comparison, but real trading costs include exchange fees, bid-ask spread, queue priority, slippage, market impact, latency, adverse selection, and failed execution. The high-frequency conclusions are especially sensitive to this limitation because costs dominate the one-second outcomes.

Seventh, the model-family set is limited. The thesis compares three graph families and two graph-operator styles. It does not test transformer-based LOB models, hybrid attention-GNN architectures, reinforcement-learning exits, probabilistic calibration layers, or explicitly turnover-regularized objectives. The conclusion is therefore conditional on the tested model families.

Eighth, hyperparameter search is limited. The benchmark is designed for fair comparison, not exhaustive optimization. Some architectures, especially multigraph and memorygraph, might improve under broader tuning. The informative five-minute multi-gnn-conv final\_refit case suggests that richer models may be sensitive to training state and calibration.

Ninth, the 1sec regime is frequency-adapted rather than directly equivalent to the slower regimes. It uses a shorter clock-time task, a restricted working sample, fewer cross-validation folds, and an enlarged holdout fraction for feasibility and alignment. The 1sec results should therefore be interpreted as a high-frequency stress test rather than as a perfectly symmetric extension of the 5min and 1min shared task.

Tenth, deployment remains conceptual rather than operationally complete. The backtest is sequential and event-based, but it is not a live trading system. It does not include live data ingestion, order placement, monitoring, risk limits, capital allocation, latency measurement, operational failure handling, or compliance controls. The thesis can support deployment-oriented conclusions, but not live-profitability claims.

Finally, model-selection bias is a general risk in empirical machine learning. Even when the final holdout is protected, repeated experimentation can indirectly shape modelling choices. Reporting both last\_CV and final\_refit, preserving the full benchmark table, and emphasizing limitations reduces this risk, but cannot eliminate it completely.

The most relevant bias categories in this thesis are temporal, asset-selection, liquidity-regime, and model-selection bias. Demographic fairness bias is not central because the dataset contains order book states rather than personal data. However, this does not remove the need for bias discussion: a model trained on a short crypto period, a small asset universe, and a single target asset may encode market-specific conditions that do not generalize.

## **7\. Conclusions and Future Research**

### **7.1. Overall Conclusion**

This thesis asked whether graph-based and memory-aware architectures improve short-horizon limit order book prediction under a controlled, deployment-oriented benchmark. The main empirical answer is conservative. The strongest last\_CV evidence favors the simpler base\_gnn family, specifically base-gnn-conv, at both shared-task frequencies.

At five-minute frequency, base-gnn-conv achieves the best last\_CV net result with net\_pnl \= 0.020356 over 26 trades. At one-minute frequency, base-gnn-conv again achieves the best result with net\_pnl \= 0.020094 over 132 trades. At one-second frequency, all models are net negative after transaction costs. The core high-frequency finding is therefore the divergence between gross signal and net deployability. memory-gnn-conv produces the largest one-second gross signal, but it fails after costs because the trading policy expresses that signal through excessive turnover.

The thesis does not show that graph neural networks are ineffective for market microstructure. It shows something more precise: under a fair entry-model benchmark, additional relation-specific processing and recurrent memory do not automatically produce better post-cost trading performance. The strongest architecture is the one that best balances signal extraction, selectivity, and turnover.

The deployment-state analysis reinforces this conclusion. last\_CV should remain the primary deployment reference because it respects chronological model selection. final\_refit is useful but can alter economic outcomes in both directions. The five-minute baseline refit remains positive, the one-minute baseline refit turns negative, and the five-minute multi-gnn-conv refit becomes very strong. These cases show why both states should be reported transparently and why final\_refit should not replace chronological deployment evidence.

The final thesis conclusion is therefore disciplined rather than promotional: richer graph architectures extract useful signals in some cases, but they do not establish a robust post-cost advantage over the simpler base\_gnn benchmark under the tested dataset, model families, and evaluation protocol. The main unresolved challenge is not merely extracting short-horizon microstructure signal. It is converting that signal into sufficiently selective, stable, and cost-aware trading decisions.

### **7.2. Future Research**

Future research should focus first on turnover-aware modelling. The 1sec experiments show that memory-based graph models can identify many short-lived opportunities, but they lack sufficient selectivity. Future work should investigate cost-aware objectives, stricter no-trade calibration, sparse event-driven state updates, confidence-aware memory resets, and threshold policies that explicitly penalize excessive trading.

A second direction is execution-aware evaluation. The present benchmark fixes a common realized-event exit rule in order to preserve fairness. Future work could keep the common entry benchmark for comparability and then test the strongest entry models under adaptive exits, richer slippage assumptions, exchange-specific fees, latency constraints, queue-position modelling, partial-fill assumptions, and adverse-selection scenarios.

A third direction is larger-universe graph modelling. A three-asset graph is useful for a controlled thesis benchmark, but it may understate the value of relation-specific architectures. Adding more crypto assets, stablecoins, sector proxies, derivatives, or cross-venue liquidity measures would create a stronger test of whether multigraph becomes more valuable when the relation space is richer.

A fourth direction is selective memory. The current memorygraph models appear capable of finding high-frequency gross signal but not of controlling trade frequency. Future models should examine sparse memory updates, event-triggered memory writes, confidence-aware state resets, and memory mechanisms coupled to explicit trade-rate constraints.

A fifth direction is robustness and uncertainty analysis. Future versions of the benchmark should report fold-level dispersion, bootstrap confidence intervals for economic metrics, pairwise model-comparison tests, regime-specific performance summaries, drawdown statistics, and cost-sensitivity curves. These additions would make it easier to distinguish genuine architecture effects from temporal-slice effects or threshold-selection effects.

A final direction is systematic cost sensitivity. Because the main high-frequency weakness is the gap between gross and net performance, future research should report cost-sensitivity curves across fee, spread, slippage, and latency assumptions. This would clarify whether a model is close to viability under realistic execution improvements or whether its signal is too small relative to unavoidable trading frictions.

## **Appendix A. Model Configuration Summary**

This appendix summarizes the main benchmark configuration in a compact form. It is not intended to replace the Methodology chapter; instead, it provides a quick reference for the model-family, operator, frequency, and target-design choices used throughout the thesis. Table A.1 maps model labels to families and operators, Table A.2 lists the frequency-specific task design, and Table A.3 summarizes the shared target and loss configuration.

**Table A.1. Model-family and operator mapping.**

| Benchmark label | Family | Graph operator | Main architectural distinction |
| :---- | :---- | :---- | :---- |
| base-gnn-conv | base\_gnn | adaptive\_conv | Early relation fusion with Conv-style graph processing. |
| base-gnn-mpnn | base\_gnn | adaptive\_mpnn | Early relation fusion with MPNN-style message passing. |
| multi-gnn-conv | multigraph | dynamic\_rel\_conv | Relation-specific graph pathways with Conv-style processing. |
| multi-gnn-mpnn | multigraph | dynamic\_edge\_mpnn | Relation-specific graph pathways with MPNN-style processing. |
| memory-gnn-conv | memorygraph | conv | Stateful recurrent node-edge memory with Conv-style graph updates. |
| memory-gnn-mpnn | memorygraph | mpnn | Stateful recurrent node-edge memory with MPNN-style graph updates. |

**Table A.2. Frequency and task configuration.**

| Frequency | Lookback | Forecast horizon | Working data slice | Final holdout fraction | CV folds | Interpretation |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | 30 minutes \= 6 bars | 5 minutes \= 1 bar | 0.0-0.9 of the full series | 0.10 | 4 | Strict shared-task benchmark. |
| 1min | 30 minutes \= 30 bars | 5 minutes \= 5 bars | 0.0-0.9 of the full series | 0.10 | 4 | Strict shared-task benchmark. |
| 1sec | 2 minutes \= 120 bars | 2 minutes \= 120 bars | 0.5-0.9 of the full series | 0.225 | 2 | Frequency-adapted high-frequency stress test. |

**Table A.3. Shared target and loss configuration.**

| Component | Configuration used in the benchmark |
| :---- | :---- |
| Target asset | ETH |
| Context assets | ADA and BTC |
| Target construction | Volatility-scaled triple-barrier framework |
| Output heads | trade\_logit, dir\_logit, return\_pred, exit\_type\_logit, tte\_pred |
| Loss weights | loss\_w\_trade \= 0.35, loss\_w\_dir \= 0.65, loss\_w\_ret \= 0.15, loss\_w\_utility \= 0.85, loss\_w\_exit\_type \= 0.05, loss\_w\_tte \= 0.03 |
| Main economic metric | net\_pnl |
| Cost proxy | c\_rt \= 0.0003 with cost\_bps\_per\_side \= 1.0 |

## **Appendix B. Additional Benchmark Tables**

This appendix records supplementary diagnostic summaries that support the benchmark interpretation. The main text uses the most interpretable subset of the metrics. The appendix keeps the wider diagnostic evidence available without overloading the Results chapter.

The source benchmark tables contain semi-empty formula-helper rows between each last\_CV and final\_refit pair. These rows are comparison helpers, not additional experiments. A typical formula has the form \=IF(F8\>0,IF(F9\>0,F8/F9,"+"),"-"). A numeric value therefore means that both the last\_CV and final\_refit entries are positive and the cell reports their ratio. A \+ marker means that the last\_CV value is positive while the final\_refit value is not positive. A \- marker means that the last\_CV value is not positive. These markers are useful for internal consistency checks, but they are excluded from the main thesis tables because they mix ratios and symbols. Table B.1 summarizes the best or least-negative last\_CV model by frequency, Table B.2 isolates the high-frequency turnover example, Table B.3 reports the full last\_CV diagnostic metrics, and Table B.4 reports the corresponding final\_refit diagnostics.

**Table B.1. Best or least-negative last\_CV model by frequency.**

| Frequency | Best or least-negative model | gross\_pnl | net\_pnl | n\_trades | Interpretation |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | base-gnn-conv | 0.028156 | 0.020356 | 26 | Best positive shared-task result. |
| 1min | base-gnn-conv | 0.059694 | 0.020094 | 132 | Best one-minute result, with larger gross signal but higher turnover. |
| 1sec | base-gnn-mpnn | 0.052679 | \-0.065821 | 395 | Least negative one-second model; still not deployment-grade after costs. |

**Table B.2. Main high-frequency turnover example.**

| Model | Frequency | gross\_pnl | net\_pnl | n\_trades | Approximate cumulative cost | Interpretation |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| memory-gnn-conv | 1sec | 0.412032 | \-1.163268 | 5251 | 1.5753 | Strong gross signal is overwhelmed by excessive turnover and transaction costs. |

**Table B.3. Full last\_CV diagnostic table.**

| Frequency | Model | gross\_pnl | net\_pnl | pnl\_per\_trade | n\_trades | trade\_rate | sign\_accuracy | win\_rate | sharpe\_like | dir\_auc | trade\_auc | rmse |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | base-gnn-conv | 0.028156 | 0.020356 | 0.000783 | 26 | 0.084142 | 0.692308 | 0.692308 | 1.914418 | 0.617105 | 0.700447 | 0.028769 |
| 5min | base-gnn-mpnn | 0.014415 | 0.006915 | 0.000277 | 25 | 0.080906 | 0.560000 | 0.560000 | 0.508262 | 0.614912 | 0.727631 | 0.038402 |
| 5min | multi-gnn-conv | 0.012758 | 0.001958 | 0.000054 | 36 | 0.116505 | 0.555556 | 0.555556 | 0.127740 | 0.616667 | 0.672097 | 0.041531 |
| 5min | multi-gnn-mpnn | 0.002941 | \-0.009359 | \-0.000228 | 41 | 0.132686 | 0.487805 | 0.487805 | \-0.594956 | 0.625439 | 0.707304 | 0.029870 |
| 5min | memory-gnn-conv | 0.009459 | 0.004359 | 0.000256 | 17 | 0.055016 | 0.588235 | 0.588235 | 0.366933 | 0.611842 | 0.734196 | 0.044066 |
| 5min | memory-gnn-mpnn | \-0.012463 | \-0.037363 | \-0.000450 | 83 | 0.268608 | 0.445783 | 0.445783 | \-1.607095 | 0.537719 | 0.726026 | 0.024487 |
| 1min | base-gnn-conv | 0.059694 | 0.020094 | 0.000152 | 132 | 0.085548 | 0.583333 | 0.568182 | 0.644455 | 0.525927 | 0.648157 | 0.002269 |
| 1min | base-gnn-mpnn | \-0.024239 | \-0.078539 | \-0.000434 | 181 | 0.117304 | 0.469613 | 0.436464 | \-2.489121 | 0.504512 | 0.642097 | 0.003080 |
| 1min | multi-gnn-conv | 0.025767 | \-0.007833 | \-0.000070 | 112 | 0.072586 | 0.544643 | 0.535714 | \-0.288013 | 0.541996 | 0.634787 | 0.005728 |
| 1min | multi-gnn-mpnn | \-0.003371 | \-0.035771 | \-0.000331 | 108 | 0.069994 | 0.490741 | 0.481481 | \-1.294441 | 0.528405 | 0.645004 | 0.005223 |
| 1min | memory-gnn-conv | \-0.031247 | \-0.078947 | \-0.000497 | 159 | 0.103046 | 0.459119 | 0.452830 | \-2.410248 | 0.479399 | 0.638928 | 0.002455 |
| 1min | memory-gnn-mpnn | 0.033605 | 0.009305 | 0.000115 | 81 | 0.052495 | 0.567901 | 0.555556 | 0.380381 | 0.529844 | 0.635665 | 0.010463 |
| 1sec | base-gnn-conv | 0.139515 | \-0.094185 | \-0.000121 | 779 | 0.008406 | 0.684211 | 0.680359 | \-6.506930 | 0.599753 | 0.841804 | 0.000592 |
| 1sec | base-gnn-mpnn | 0.052679 | \-0.065821 | \-0.000167 | 395 | 0.004263 | 0.663291 | 0.658228 | \-8.430534 | 0.599093 | 0.848558 | 0.000569 |
| 1sec | multi-gnn-conv | 0.081710 | \-0.108790 | \-0.000171 | 635 | 0.006852 | 0.656693 | 0.650394 | \-10.771267 | 0.597838 | 0.839544 | 0.000539 |
| 1sec | multi-gnn-mpnn | 0.079777 | \-0.080723 | \-0.000151 | 535 | 0.005773 | 0.680374 | 0.676636 | \-7.960091 | 0.600538 | 0.868370 | 0.000504 |
| 1sec | memory-gnn-conv | 0.412032 | \-1.163268 | \-0.000222 | 5251 | 0.056665 | 0.584841 | 0.584079 | \-28.865252 | 0.588785 | 0.490050 | 0.001247 |
| 1sec | memory-gnn-mpnn | 0.223788 | \-0.280512 | \-0.000167 | 1681 | 0.018140 | 0.662106 | 0.660916 | \-17.034172 | 0.596713 | 0.863699 | 0.000578 |

**Table B.4. Full final\_refit diagnostic table.**

| Frequency | Model | gross\_pnl | net\_pnl | pnl\_per\_trade | n\_trades | trade\_rate | sign\_accuracy | win\_rate | sharpe\_like | dir\_auc | trade\_auc | rmse |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 5min | base-gnn-conv | 0.017570 | 0.011270 | 0.000537 | 21 | 0.067961 | 0.619048 | 0.619048 | 0.988114 | 0.630702 | 0.721795 | 0.021324 |
| 5min | base-gnn-mpnn | 0.020722 | 0.008722 | 0.000218 | 40 | 0.129450 | 0.575000 | 0.575000 | 0.545576 | 0.636842 | 0.715911 | 0.016092 |
| 5min | multi-gnn-conv | 0.061167 | 0.041367 | 0.000627 | 66 | 0.213592 | 0.681818 | 0.681818 | 2.182646 | 0.703947 | 0.720920 | 0.017109 |
| 5min | multi-gnn-mpnn | 0.008092 | 0.002992 | 0.000176 | 17 | 0.055016 | 0.588235 | 0.588235 | 0.309342 | 0.634211 | 0.707158 | 0.021148 |
| 5min | memory-gnn-conv | 0.029075 | \-0.000325 | \-0.000003 | 98 | 0.317152 | 0.540816 | 0.530612 | \-0.013141 | 0.631140 | 0.722914 | 0.016015 |
| 5min | memory-gnn-mpnn | 0.017905 | \-0.014495 | \-0.000134 | 108 | 0.349515 | 0.527778 | 0.518519 | \-0.548738 | 0.544298 | 0.737259 | 0.014369 |
| 1min | base-gnn-conv | 0.007198 | \-0.012002 | \-0.000188 | 64 | 0.041478 | 0.484375 | 0.484375 | \-0.555220 | 0.524286 | 0.635712 | 0.003175 |
| 1min | base-gnn-mpnn | 0.034198 | 0.006298 | 0.000068 | 93 | 0.060272 | 0.548387 | 0.548387 | 0.234208 | 0.513907 | 0.655608 | 0.003194 |
| 1min | multi-gnn-conv | 0.001944 | \-0.037056 | \-0.000285 | 130 | 0.084251 | 0.492308 | 0.492308 | \-1.153049 | 0.543084 | 0.641036 | 0.004606 |
| 1min | multi-gnn-mpnn | 0.021735 | \-0.011565 | \-0.000104 | 111 | 0.071938 | 0.540541 | 0.540541 | \-0.386487 | 0.522804 | 0.627002 | 0.003473 |
| 1min | memory-gnn-conv | \-0.011482 | \-0.049882 | \-0.000390 | 128 | 0.082955 | 0.484375 | 0.476562 | \-1.816363 | 0.525231 | 0.619879 | 0.002149 |
| 1min | memory-gnn-mpnn | 0.049531 | \-0.055769 | \-0.000159 | 351 | 0.227479 | 0.524217 | 0.504274 | \-1.323040 | 0.516055 | 0.562599 | 0.002287 |
| 1sec | base-gnn-conv | 0.262794 | \-0.264606 | \-0.000151 | 1758 | 0.018971 | 0.668942 | 0.665529 | \-11.677533 | 0.590194 | 0.828703 | 0.000511 |
| 1sec | base-gnn-mpnn | 0.108520 | \-0.089180 | \-0.000135 | 659 | 0.007111 | 0.688923 | 0.682853 | \-7.092366 | 0.598878 | 0.842822 | 0.000499 |
| 1sec | multi-gnn-conv | 0.091421 | \-0.073579 | \-0.000134 | 550 | 0.005935 | 0.703636 | 0.694545 | \-8.415791 | 0.597449 | 0.843536 | 0.000494 |
| 1sec | multi-gnn-mpnn | 0.057127 | \-0.078773 | \-0.000174 | 453 | 0.004888 | 0.657837 | 0.651214 | \-9.356826 | 0.600348 | 0.865638 | 0.000519 |
| 1sec | memory-gnn-conv | 0.443031 | \-0.954969 | \-0.000205 | 4660 | 0.050287 | 0.604506 | 0.603219 | \-25.126196 | 0.592186 | 0.852874 | 0.000529 |
| 1sec | memory-gnn-mpnn | 0.198876 | \-0.202524 | \-0.000151 | 1338 | 0.014439 | 0.671898 | 0.668909 | \-13.431875 | 0.593766 | 0.865245 | 0.000590 |

## **Appendix C. Final Holdout Alignment Table**

Source: final\_runs/\*/splits/split\_summary.json, split\_indices.npz, resolved\_config.yaml.

Interpretation: \- literal\_match\_within\_freq \= True means the saved idx\_holdout arrays are exactly identical within that frequency regime. \- final\_test\_equals\_holdout \= True means final\_production.test is exactly the same saved holdout interval. \- semantic\_alignment\_vs\_1min compares calendar-time alignment against the 1-minute reference window. Table C.1 reports the run-level alignment records.

**Table C.1. Final-holdout alignment records by run.**

| run | freq | slice | holdout\_frac | holdout\_start\_utc | holdout\_end\_utc | holdout\_n | final\_test\_equals\_holdout | literal\_match\_within\_freq | effective\_holdout\_full\_frac | delta\_vs\_1min |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| 1min-base-gnn-conv | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s |
| 1min-base-gnn-mpnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s |
| 1min-memory-gnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s |
| 1min-multi-gnn-conv | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s |
| 1min-multi-gnn-mpnn | 1min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | True | True | 0.809977-0.899942 | start 0s / end 0s |
| 1sec-base-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 1sec-base-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 1sec-memory-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 1sec-memory-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 1sec-multi-gnn-conv | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 1sec-multi-gnn-mpnn | 1sec | 0.5-0.9 | 0.225 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | True | True | 0.810000-0.899999 | start 34s / end 181s |
| 5min-base-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:05:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end \-120s |
| 5min-memory-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:05:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end \-120s |
| 5min-multi-gnn | 5min | 0.0-0.9 | 0.100 | 2021-04-17T03:25:00 | 2021-04-18T05:05:00 | 309 | True | True | 0.809883-0.899708 | start 0s / end \-120s |

## **Frequency-level summary**

Table C.2 summarizes the same alignment at frequency level.

**Table C.2. Frequency-level final-holdout summary.**

| freq | runs | shared\_holdout\_start\_utc | shared\_holdout\_end\_utc | shared\_holdout\_n | note |
| :---- | :---- | :---- | :---- | :---- | :---- |
| 1min | 5 | 2021-04-17T03:25:00 | 2021-04-18T05:07:00 | 1543 | reference window |
| 1sec | 6 | 2021-04-17T03:25:34 | 2021-04-18T05:10:01 | 92668 | starts 34s later than 1min, ends 181s later |
| 5min | 3 | 2021-04-17T03:25:00 | 2021-04-18T05:05:00 | 309 | same start as 1min, ends 120s earlier |

## **References**

\[1\] Cont, R. (2001). *Empirical properties of asset returns: stylized facts and statistical issues*. Quantitative Finance, 1(2), 223-236. PDF: [https://www.stat.rice.edu/\~dobelman/courses/texts/stylized.cont.2001.pdf](https://www.stat.rice.edu/~dobelman/courses/texts/stylized.cont.2001.pdf)

\[2\] Cont, R., Stoikov, S., & Talreja, R. (2010). *A stochastic model for order book dynamics*. Operations Research, 58(3), 549-563. PDF: [https://rama.cont.perso.math.cnrs.fr/pdf/CST2010.pdf](https://rama.cont.perso.math.cnrs.fr/pdf/CST2010.pdf)

\[3\] Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018). *Benchmark dataset for mid-price forecasting of limit order book data with machine learning methods*. Journal of Forecasting, 37(8), 852-866. arXiv/PDF: [https://arxiv.org/abs/1705.03233](https://arxiv.org/abs/1705.03233)

\[4\] Sirignano, J., & Cont, R. (2019). *Universal features of price formation in financial markets: perspectives from deep learning*. Quantitative Finance, 19(9), 1449-1459. arXiv/PDF: [https://arxiv.org/abs/1803.06917](https://arxiv.org/abs/1803.06917)

\[5\] Zhang, Z., Zohren, S., & Roberts, S. (2019). *DeepLOB: Deep convolutional neural networks for limit order books*. IEEE Transactions on Signal Processing, 67(11), 3001-3012. PDF: [https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf](https://www.oxford-man.ox.ac.uk/wp-content/uploads/2020/03/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books.pdf)

\[6\] Kipf, T. N., & Welling, M. (2017). *Semi-supervised classification with graph convolutional networks*. International Conference on Learning Representations. arXiv/PDF: [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)

\[7\] Gilmer, J., Schoenholz, S. S., Riley, P. F., Vinyals, O., & Dahl, G. E. (2017). *Neural message passing for quantum chemistry*. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1263-1272. PDF: [https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf](https://proceedings.mlr.press/v70/gilmer17a/gilmer17a.pdf)

\[8\] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph attention networks*. International Conference on Learning Representations. arXiv/PDF: [https://arxiv.org/abs/1710.10903](https://arxiv.org/abs/1710.10903)

\[9\] Wu, Z., Pan, S., Chen, F., Long, G., Zhang, C., & Yu, P. S. (2020). *A comprehensive survey on graph neural networks*. IEEE Transactions on Neural Networks and Learning Systems, 32(1), 4-24. arXiv/PDF: [https://arxiv.org/abs/1901.00596](https://arxiv.org/abs/1901.00596)

\[10\] Wu, Z., Pan, S., Long, G., Jiang, J., Chang, X., & Zhang, C. (2020). *Connecting the dots: Multivariate time series forecasting with graph neural networks*. Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, 753-763. arXiv/PDF: [https://arxiv.org/abs/2005.11650](https://arxiv.org/abs/2005.11650)

\[11\] Kazemi, S. M., Goel, R., Jain, K., Kobyzev, I., Sethi, A., Forsyth, P., & Poupart, P. (2020). *Representation learning for dynamic graphs: A survey*. Journal of Machine Learning Research, 21(70), 1-73. PDF: [https://jmlr.csail.mit.edu/papers/volume21/19-447/19-447.pdf](https://jmlr.csail.mit.edu/papers/volume21/19-447/19-447.pdf)

\[12\] Rossi, E., Chamberlain, B., Frasca, F., Eynard, D., Monti, F., & Bronstein, M. (2020). *Temporal graph networks for deep learning on dynamic graphs*. arXiv preprint arXiv:2006.10637. arXiv/PDF: [https://arxiv.org/abs/2006.10637](https://arxiv.org/abs/2006.10637)

\[13\] Wang, J., Zhang, S., Xiao, Y., & Song, R. (2022). *A review on graph neural network methods in financial applications*. Journal of Data Science, 20(2), 111-134. PDF: [https://jds-online.org/journal/JDS/article/1279/file/pdf](https://jds-online.org/journal/JDS/article/1279/file/pdf)

\[14\] Qian, H., Zhou, H., Zhao, Q., Chen, H., Yao, H., Wang, J., Liu, Z., Yu, F., Zhang, Z., & Zhou, J. (2024). *MDGNN: Multi-relational dynamic graph neural network for comprehensive and dynamic stock investment prediction*. Proceedings of the AAAI Conference on Artificial Intelligence, 38(13), 14642-14650. arXiv/PDF: [https://arxiv.org/abs/2402.06633](https://arxiv.org/abs/2402.06633)

\[15\] Martinsn. *High-Frequency Crypto Limit Order Book Data*. Kaggle dataset. Dataset page: [https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data)

\[--\] Biais, B., Hillion, P., & Spatt, C. (1995). *An empirical analysis of the limit order book and the order flow in the Paris Bourse*. The Journal of Finance, 50(5), 1655-1689. DOI: [https://doi.org/10.1111/j.1540-6261.1995.tb05192.x](https://doi.org/10.1111/j.1540-6261.1995.tb05192.x)

\[16\] Gould, M. D., Porter, M. A., Williams, S., McDonald, M., Fenn, D. J., & Howison, S. D. (2013). *Limit order books*. Quantitative Finance, 13(11), 1709-1742. DOI: [https://doi.org/10.1080/14697688.2013.803148](https://doi.org/10.1080/14697688.2013.803148)   \- access via WU account

\[17\] Schnaubelt, M., Rende, J., & Krauss, C. (2019). *Testing stylized facts of Bitcoin limit order books*. Journal of Risk and Financial Management, 12(1), 25\. DOI: [https://doi.org/10.3390/jrfm12010025](https://doi.org/10.3390/jrfm12010025)

\[18\] Arroyo, Á., Cartea, Á., Moreno-Pino, F., & Zohren, S. (2024). *Deep attentive survival analysis in limit order books: Estimating fill probabilities with convolutional-transformers*. Quantitative Finance, 24(1), 35-57. DOI: [https://doi.org/10.1080/14697688.2023.2286351](https://doi.org/10.1080/14697688.2023.2286351)

\[19\] Jung, J., & Lee, K. (2025). *Attention-based reading, highlighting, and forecasting of the limit order book*. Quantitative Finance, 25(7), 1015-1027. DOI: [https://doi.org/10.1080/14697688.2025.2522914](https://doi.org/10.1080/14697688.2025.2522914)

\[20\] Briola, A., Bartolucci, S., & Aste, T. (2025). *HLOB: Information persistence and structure in limit order books*. Expert Systems with Applications, 266, 126078\. DOI: [https://doi.org/10.1016/j.eswa.2024.126078](https://doi.org/10.1016/j.eswa.2024.126078)

\[21\] Zheng, Y., Yi, L., & Wei, Z. (2025). *A survey of dynamic graph neural networks*. Frontiers of Computer Science, 19, Article 196323\. DOI: [https://doi.org/10.1007/s11704-024-3853-2](https://doi.org/10.1007/s11704-024-3853-2)

\[22\] Corradini, F., Gerosa, F., Gori, M., Lucheroni, C., Piangerelli, M., & Zannotti, M. (2026). *A systematic literature review of spatio-temporal graph neural network models for time series forecasting and classification*. Neural Networks, 195, 108269\. DOI: [https://doi.org/10.1016/j.neunet.2025.108269](https://doi.org/10.1016/j.neunet.2025.108269)

\[23\] López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. ISBN: 978-1-119-48208-6. [https://books.google.com/books?hl=en\&lr=\&id=oU9KDwAAQBAJ\&oi=fnd\&pg=PR21\&dq=Advances+in+Financial+Machine+Learning\&ots=7VKLT0rD7v\&sig=uiAs8FQTgJFpkWWmbmsbEcQV9qo](https://books.google.com/books?hl=en&lr=&id=oU9KDwAAQBAJ&oi=fnd&pg=PR21&dq=Advances+in+Financial+Machine+Learning&ots=7VKLT0rD7v&sig=uiAs8FQTgJFpkWWmbmsbEcQV9qo)

\[\--\] Bailey, D. H., Borwein, J. M., López de Prado, M., & Zhu, Q. J. (2017). *The probability of backtest overfitting*. Journal of Computational Finance, 20(4), 39-69. DOI: [https://doi.org/10.21314/JCF.2016.322](https://doi.org/10.21314/JCF.2016.322)

\[24\] Cartea, Á., Jaimungal, S., & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*. Cambridge University Press. ISBN: 978-1-107-09114-6. [https://books.google.com/books?hl=en\&lr=\&id=5dMmCgAAQBAJ\&oi=fnd\&pg=PR13\&dq=Algorithmic+and+High-Frequency+Trading.+\&ots=4cFqMNHOdV\&sig=iB7S5Rkxv5-Qax8LpCXWC5VJciM](https://books.google.com/books?hl=en&lr=&id=5dMmCgAAQBAJ&oi=fnd&pg=PR13&dq=Algorithmic+and+High-Frequency+Trading.+&ots=4cFqMNHOdV&sig=iB7S5Rkxv5-Qax8LpCXWC5VJciM)

\[25\] Sirignano, J. (2019). *Deep learning for limit order books*. Quantitative Finance, 19(4), 549-570. DOI: [https://doi.org/10.1080/14697688.2018.1546053](https://doi.org/10.1080/14697688.2018.1546053)   \- access via WU account

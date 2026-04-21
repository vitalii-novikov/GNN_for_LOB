# **2. Research Questions and Hypotheses**

This thesis studies graph-based market microstructure models under an explicitly **apples-to-apples comparison protocol**. The central goal is not to compare arbitrary model implementations under partially different objectives, but to compare several graph families under the **same decision problem, the same target construction, the same loss structure, the same thresholding logic, and the same validation and holdout protocol** within each frequency regime.

The empirical design evaluates **three model families**:

* `base_gnn`,
* `multigraph`,
* `memorygraph`,

with **two operator variants per family**, yielding **six models per frequency regime**:

* `base_gnn + adaptive_conv`,
* `base_gnn + adaptive_mpnn`,
* `multigraph + dynamic_rel_conv`,
* `multigraph + dynamic_edge_mpnn`,
* `memorygraph + conv`,
* `memorygraph + mpnn`.

These six models are trained and evaluated at three temporal regimes:

* **5-minute data**, using a **30-minute lookback** and **5-minute horizon**,
* **1-minute data**, using a **30-minute lookback** and **5-minute horizon**,
* **1-second data**, using a **2-minute lookback** and **2-minute horizon**.

This produces **18 total experimental configurations**. The comparison is apples-to-apples **within each frequency regime**, because only the architecture family and graph operator vary. Cross-frequency interpretation is more nuanced: the **1-minute** and **5-minute** regimes form a strict shared-task benchmark, whereas the **1-second** regime is a frequency-adapted extension designed to remain computationally feasible and behaviorally meaningful at ultra-high frequency.

## **2.1. Main Research Question**

**When graph-based microstructure models are evaluated under the same entry-model benchmark protocol, how do `base_gnn`, `multigraph`, and `memorygraph` compare across 5-minute, 1-minute, and 1-second data, and how does the value of richer relational and memory mechanisms depend on temporal resolution?**

This main question emphasizes that the thesis is about **controlled architectural comparison**, not about loosely comparing models that solve partially different trading problems.

## **2.2. Specific Research Questions**

### **RQ1. Within-frequency apples-to-apples model comparison**

*Within a fixed frequency regime, which of the six candidate models performs best when all models share the same lookback window, forecast horizon, target construction, loss weights, thresholding logic, and event-based backtesting rule?*

This is the most direct question in the thesis. It asks whether performance differences can be attributed to architectural design rather than to differences in preprocessing, supervision, or evaluation.

### **RQ2. Family-level value of relational and memory mechanisms**

*Relative to the `base_gnn` baseline, do the `multigraph` and `memorygraph` families deliver consistent gains in predictive and trading-oriented performance under the same benchmark protocol?*

This question isolates the incremental value of two richer modeling ideas:

* explicit multi-relation graph structure in `multigraph`,
* persistent state and temporal memory in `memorygraph`.

### **RQ3. Operator sensitivity within each family**

*How important is the message-passing operator choice within each family? In particular, do `adaptive_conv` and `adaptive_mpnn`, `dynamic_rel_conv` and `dynamic_edge_mpnn`, and `memorygraph conv` and `memorygraph mpnn` exhibit materially different behavior under the same task definition?*

This question focuses on whether architectural family or operator variant explains more of the observed performance variation.

### **RQ4. Frequency dependence of architectural advantages**

*Do the relative gains from multigraph and memory-based architectures change across 5-minute, 1-minute, and 1-second regimes?*

This question addresses whether richer graph mechanisms become more valuable as the temporal regime becomes finer and more non-stationary.

### **RQ5. Benchmark comparability versus frequency adaptation**

*How should the results from the 1-second regime be interpreted relative to the 1-minute and 5-minute benchmarks, given that the 1-second task uses a shorter horizon, a shorter lookback, and a reduced working sample?*

This question is methodological as much as empirical. It clarifies that the thesis distinguishes between:

* a **strict shared-task benchmark** at 1-minute and 5-minute frequency,
* a **frequency-adapted ultra-high-frequency experiment** at 1-second frequency.

### **RQ6. Stability of conclusions under deployment-oriented evaluation**

*Do the principal conclusions remain stable when comparing the statistical best-CV model, the realistic `last_CV` deployment proxy, and the larger-sample `final_refit` model on the same final holdout interval?*

This question connects model development to realistic deployment logic. It asks whether the same model family remains preferable once operational constraints are taken seriously.

## **2.3. Research Hypotheses**

### **H1. 1-minute superiority over 5-minute in the shared-task benchmark**

*Under the common 30-minute lookback and 5-minute horizon benchmark, the 1-minute regime will outperform the 5-minute regime for the corresponding model families on the primary comparison metrics, especially `pnl_sum`, `dir_auc`, and `trade_auc`.*

The reasoning is that 1-minute data preserve substantially more within-horizon temporal structure than 5-minute aggregation while remaining much less noisy than raw second-level data. As a result, 1-minute inputs should provide a richer representation of short-horizon cross-asset microstructure dynamics.

### **H2. Architectural gains will be clearer at 1-minute than at 5-minute**

*The incremental value of `multigraph` and `memorygraph` relative to `base_gnn` will be larger at 1-minute frequency than at 5-minute frequency.*

This hypothesis reflects the idea that richer architectures need enough temporal and relational resolution to matter. At 5-minute frequency, the benchmark horizon corresponds to only one forward bar, so the room for exploiting more sophisticated temporal graph structure is more limited.

### **H3. Operator choice matters, but its effect is frequency-dependent**

*Within each family, operator differences will be small at 5-minute frequency and more visible at 1-minute and 1-second frequency, where finer temporal structure creates more room for operator-specific gains.*

This hypothesis does not assume that one operator will dominate universally. Rather, it proposes that operator sensitivity itself should increase as the data become more temporally detailed.

### **H4. The 1-second regime can become competitive after task adaptation**

*When the task is adapted to a 2-minute lookback and 2-minute horizon, 1-second models will become meaningfully competitive, particularly on direction-sensitive and trade-selection metrics, even if they do not universally dominate the 1-minute benchmark.*

This is a deliberately moderate hypothesis. It does not claim that ultra-high-frequency data are automatically superior, but that they become informative once the prediction task is aligned with the shorter lifetime of second-level signals.

### **H5. Memory should become more useful as temporal resolution increases**

*The relative advantage of `memorygraph` over the simpler graph baselines will increase as temporal resolution becomes finer, with the strongest potential gains expected in the 1-second regime, moderate gains at 1-minute, and the weakest gains at 5-minute frequency.*

This follows from the idea that persistent state and fast adaptation should matter most when the market evolves rapidly and local dependencies decay quickly.

### **H6. Deployment-oriented rankings will be broadly stable**

*Although `final_refit` should act as an optimistic upper bound, the main family-level conclusions obtained from final holdout evaluation will remain broadly consistent between `best_CV`, `last_CV`, and `final_refit` model states.*

This hypothesis is important for practical relevance. A result that only appears in the idealized refit setting but disappears in the realistic `last_CV` setting would be less convincing from a deployment perspective.

## **2.4. Research Contribution Logic**

The contribution logic of the thesis is built around **controlled architectural comparison under frequency-aware task design**.

First, the thesis provides a **strict apples-to-apples benchmark** at **1-minute** and **5-minute** frequency, where all six models are evaluated under the same clock-time lookback, the same horizon, the same label construction, the same thresholding rule, and the same event-based backtest.

Second, the thesis extends the comparison to **1-second data** under a deliberately adapted task definition. This extension is still apples-to-apples **within the 1-second regime**, because the same six models are compared under the same 1-second-specific setup. However, cross-frequency claims involving 1-second results are interpreted as claims about **frequency-specific suitability**, not as a simple universal ranking against the slower regimes.

This design allows the thesis to make a more precise contribution than a flat leaderboard would allow. It identifies:

* which model family performs best under a tightly controlled shared-task benchmark,
* whether graph operator choice matters materially within each family,
* whether richer relational and memory mechanisms become more valuable at finer temporal resolutions,
* and whether these conclusions remain stable under deployment-oriented evaluation logic.

# **3. Methodology**

## **3.1 Overall empirical design**

The empirical design consists of **18 experiments**, organized as:

* **3 frequency regimes**,
* **3 model families**,
* **2 operator variants per family**.

For each frequency regime, exactly the following six models are trained:

* `base_gnn + adaptive_conv`,
* `base_gnn + adaptive_mpnn`,
* `multigraph + dynamic_rel_conv`,
* `multigraph + dynamic_edge_mpnn`,
* `memorygraph + conv`,
* `memorygraph + mpnn`.

The frequency-specific task definitions are:

1. **5-minute regime**: `freq=5min`, `lookback=30min`, `horizon=5min`.
2. **1-minute regime**: `freq=1min`, `lookback=30min`, `horizon=5min`.
3. **1-second regime**: `freq=1sec`, `lookback=2min`, `horizon=2min`, with `data_slice_start_frac=0.5`, `data_slice_end_frac=0.9`, and `final_holdout_frac=0.225`.

This design serves two purposes simultaneously.

First, it creates a **direct cross-frequency benchmark** between **1-minute** and **5-minute** data. Both regimes solve the same clock-time forecasting problem: a 30-minute historical context and a 5-minute forward event horizon. The number of bars differs by frequency, but the economic task is identical.

Second, it creates a **frequency-adapted high-frequency extension** at **1-second** resolution. Here the task is intentionally shortened to 2 minutes because second-level signals are more local, more transient, and computationally more expensive to model over long horizons. The 1-second regime is therefore not framed as a perfectly symmetric benchmark against the slower frequencies, but as a controlled ultra-high-frequency sub-study.

The thesis therefore makes comparisons at two levels:

* **within-frequency apples-to-apples comparisons** across the six model configurations,
* **cross-frequency comparisons** with direct interpretation for 1-minute versus 5-minute and more cautious, regime-aware interpretation for 1-second versus slower data.

## **3.2 Data sampling and frequency-specific working samples**

The asset graph contains three cryptocurrency nodes:

* **ADA**,
* **BTC**,
* **ETH**,

with **ETH** treated as the target asset.

For the **1-minute** and **5-minute** regimes, the working sample is taken from the **0.0-0.9 segment** of the available time series, with the last part of that working sample reserved as the blind holdout region.

For the **1-second** regime, the working sample is deliberately restricted to the **0.5-0.9 segment** of the available series. This restriction is intentional and serves two goals.

First, it reduces computational cost. Second-level order-book data are much larger, and full-history training would disproportionately increase training time without necessarily improving the quality of the architectural comparison.

Second, the **final holdout fraction is increased to 0.225** in the 1-second regime so that the **calendar-time holdout interval aligns as closely as possible with the terminal holdout interval used in the 1-minute and 5-minute experiments**. This is an important design choice. It means that although the 1-second regime uses a smaller working sample, the final blind evaluation is still anchored to the same late-period market segment as the slower-frequency benchmarks.

This makes the 1-second design more informative than a purely arbitrary subsample reduction: it preserves a meaningful final comparison window while keeping experimentation tractable.

## **3.3 Graph representation and feature construction**

Across all experiments, the market is represented as a **directed asset graph** with a fixed asset universe. The topology is constant because the node set is fixed, but the graph remains dynamic because node states and edge attributes vary over time.

Each node is described by microstructure features summarizing:

* recent price dynamics,
* order-flow pressure,
* liquidity and order-book state.

Cross-asset edge features are constructed from rolling dependence statistics over recent histories of price, flow, and liquidity variables. These relation features are built consistently across experiments so that architectural comparisons are driven by **how models process the same graph information**, not by different feature definitions.

This graph construction is held constant across all 18 experiments. As a result, the thesis isolates the effect of architecture family and message-passing operator rather than conflating architecture with data representation.

## **3.4 Frequency-specific task design**

### **3.4.1 Shared-task benchmark: 5-minute and 1-minute**

The 5-minute and 1-minute regimes share the same clock-time task:

* **lookback window = 30 minutes**,
* **forecast horizon = 5 minutes**.

The bar-count implications are frequency-specific:

* at **5-minute frequency**, the lookback corresponds to **6 bars** and the horizon to **1 bar**;
* at **1-minute frequency**, the lookback corresponds to **30 bars** and the horizon to **5 bars**.

This design is deliberate. It ensures that both regimes solve the same economic forecasting problem even though the representation granularity differs. The comparison therefore answers the question: **which sampling frequency preserves more useful structure for the same short-horizon decision problem?**

### **3.4.2 Adapted high-frequency regime: 1-second**

The 1-second regime uses:

* **lookback window = 2 minutes**,
* **forecast horizon = 2 minutes**.

In bar counts, this corresponds to:

* **120 input bars**,
* **120 forward bars**.

This shorter setup is intentional. A 5-minute horizon at 1-second resolution would combine much greater computational cost with a target horizon that may exceed the persistence of many second-level signals. The shortened 1-second task therefore reflects **task adaptation**, not reduced rigor.

Within the 1-second regime, the comparison remains fully apples-to-apples because all six candidate models are trained and evaluated under exactly the same second-level configuration.

## **3.5 Target construction and shared learning task**

All model families are trained under a common **multi-task triple-barrier framework**. For each prediction timestamp, the future path of ETH is observed until one of the following event outcomes occurs first:

* the upper barrier is hit,
* the lower barrier is hit,
* the vertical barrier is reached.

This produces a shared target space consisting of:

* realized return,
* trade relevance,
* movement direction,
* exit type,
* time to exit.

The trade label functions as a meta-labeling component, identifying whether the future move is economically meaningful after accounting for frictions.

Crucially, the **same target set and the same loss weighting structure are preserved across all model families within each regime**. This means that `base_gnn`, `multigraph`, and `memorygraph` are not allowed to specialize through different supervision definitions. They all learn the same multi-task objective and differ only in how they encode temporal and relational information.

At the same time, the benchmark is formulated as an **entry-model comparison**. In the final trading evaluation, position entry is determined by the model's trade and direction outputs, while exit-type and time-to-exit predictions are treated as **auxiliary learning targets and diagnostic outputs**, not as architecture-specific privileges in the primary trading benchmark. This choice is essential for apples-to-apples comparison, especially when including a memory-based model that might otherwise benefit from a bespoke predicted-exit trading rule.

## **3.6 Model families and shared output interface**

The thesis compares three graph-model families.

The first family, **`base_gnn`**, acts as the baseline graph architecture and is evaluated with two operator variants:

* `adaptive_conv`,
* `adaptive_mpnn`.

The second family, **`multigraph`**, explicitly models separate relation channels and is evaluated with:

* `dynamic_rel_conv`,
* `dynamic_edge_mpnn`.

The third family, **`memorygraph`**, introduces persistent state and relation-aware temporal memory and is evaluated with:

* `conv`,
* `mpnn`.

Despite these architectural differences, all models expose the same output interface:

* trade probability,
* directional probability,
* return prediction,
* exit-type prediction,
* time-to-exit prediction.

This common interface serves two methodological purposes.

First, it enables a common multi-task learning objective across architectures. Second, it allows all architectures to be evaluated under the same **entry-decision benchmark protocol**, where trading decisions are based on the same thresholded trade and direction scores. This prevents the empirical comparison from being distorted by giving one family a fundamentally different execution policy.

## **3.7 Validation protocol**

Model development follows a **purged walk-forward protocol**.

Within each working sample, the data are split into:

* a **pre-holdout region** used for model development,
* a **final holdout region** reserved for final blind evaluation.

Within the pre-holdout region, the folds follow the structure:

**train → gap → validation → gap → test**

The purge gaps are necessary because triple-barrier labels depend on future price evolution, so adjacent observations may otherwise share overlapping future windows. Gap segments therefore reduce temporal leakage and produce a more conservative out-of-sample estimate.

This same validation principle is applied to all six models within each frequency regime. The absolute number of bars differs by frequency, but the clock-time and leakage logic remain aligned.

## **3.8 last_CV and final_refit logic**

A central distinction is made between three model states:

* **best_CV**,
* **last_CV**,
* **final_refit**.

The **best_CV model** is the strongest model selected inside cross-validation according to the validation protocol. It is a useful statistical reference because it reflects the best development-stage checkpoint under the pre-holdout folds.

The **last_CV model** is the most realistic deployment proxy. It represents the latest model that would be available after the walk-forward development process without assuming zero-friction retraining immediately before the unseen future period.

The **final_refit model** is trained on the largest possible pre-holdout sample and therefore serves as an idealized upper bound. It estimates what performance might look like if full retraining and redeployment were effectively frictionless.

All three model states are evaluated on the same final holdout interval. This preserves the strong logic already present in the study:

* `best_CV` measures the strongest selected development model,
* `last_CV` measures the most realistic deployable model,
* `final_refit` measures the optimistic upper bound.

This distinction is important because a model family that only looks attractive in the refit setting but loses its advantage in the `last_CV` setting would be less compelling from a practical perspective.

## **3.9 Thresholding and backtesting**

All models output continuous scores rather than direct trade actions. Therefore, threshold calibration is performed on validation data only. The calibrated decision rule maps model outputs into:

* long,
* short,
* no trade.

For apples-to-apples comparison, the primary trading rule is deliberately standardized across all model families:

* the **trade head** determines whether a trade is considered,
* the **direction head** determines whether that trade is long or short,
* the **same thresholding logic** is used for every model family within a regime.

After a position is opened, the benchmark backtest applies the **same external event-based exit rule** to every model. In other words, the primary comparison treats all six models as **entry models** under a common exit protocol. This is methodologically important because it ensures that architectural differences are not conflated with differences in trade-closing logic.

Any predictions related to exit type or time-to-exit remain valuable as auxiliary diagnostics, but they do not define the primary benchmark trading rule.

## **3.10 Evaluation logic**

The evaluation combines predictive and trading-oriented evidence, but the final comparison logic places particular emphasis on a small set of primary metrics.

The **primary comparison metrics** are:

* **`pnl_sum`**, as the main economic usefulness measure,
* **`dir_auc`**, as the main direction-quality measure,
* **`trade_auc`**, as the main trade-selection quality measure.

These three metrics are prioritized because together they capture:

* whether the model identifies economically useful trades,
* whether it gets the directional sign right,
* whether those signals translate into cumulative value under the standardized benchmark.

However, primary metrics are not sufficient on their own. Therefore, the thesis also reports secondary metrics that help interpret disagreements between models.

## **3.11 Sequential backtesting procedure**

Economic usefulness is assessed with a **sequential non-overlapping event-based backtest**.

Once a signal is generated at time \(t\), the simulated position remains open until the corresponding benchmark event exit is reached. No new trade may be opened before the current trade is closed. This non-overlapping rule prevents artificially inflated turnover and keeps the evaluation aligned with a single-position short-horizon trading interpretation.

The backtest is used as a comparative benchmark rather than as a full execution simulator. Its role is to determine whether the models produce economically meaningful entry signals under a shared exit rule and common friction assumptions.

## **3.12 Evaluation metrics**

All architectures are reported under the same set of metrics.

The **primary comparison metrics** are:

* cumulative profit and loss (**`pnl_sum`**),
* directional AUC (**`dir_auc`**),
* trade AUC (**`trade_auc`**).

The **secondary trading metrics** include:

* profit per trade,
* trade rate and signal coverage,
* sign accuracy,
* win rate,
* long and short trade counts,
* long-side and short-side PnL,
* Sharpe-like normalized performance.

The **secondary predictive metrics** include:

* RMSE and MAE for return prediction,
* information coefficient,
* exit-type accuracy,
* average predicted time-to-exit,
* average true time-to-exit.

These secondary metrics are important even when they are not used as the primary ranking criteria. For example:

* a model with strong `pnl_sum` but weak `trade_auc` may be economically useful yet unstable,
* a model with strong `dir_auc` but weak `pnl_sum` may be directionally accurate but poorly calibrated for trade selection,
* a model with strong `trade_auc` but weak `pnl_per_trade` may find many trades of low economic quality.

Accordingly, the final interpretation of results is based on the primary metrics first, with the secondary metrics used to explain *why* one model family outperforms another.

## **3.13 Fair-comparison principle**

The core methodological principle of the thesis is that **within each frequency regime, only the model family and operator are allowed to vary**.

Thus, for the six models compared at a given frequency, the following are held fixed:

* asset universe and target asset,
* graph input construction,
* lookback window and forecast horizon,
* data slice and holdout design,
* triple-barrier labeling framework,
* multi-task target set,
* loss weights,
* thresholding logic,
* backtesting rule,
* walk-forward validation protocol,
* final holdout evaluation logic.

This is what makes the comparison apples-to-apples.

The only important qualification concerns the **1-second regime**. There, the task is intentionally adapted to a shorter lookback and shorter horizon, and the working sample is reduced for computational reasons. However, even in that regime, the six candidate models are still compared under exactly the same second-level design. Therefore:

* **within 1-second**, the comparison remains apples-to-apples,
* **between 1-minute and 5-minute**, the comparison remains a strict shared-task benchmark,
* **between 1-second and the slower regimes**, the comparison should be interpreted as a comparison of frequency-specific suitability rather than a naïve universal ranking.

This distinction allows the thesis to remain both fair and realistic: fair in its architectural comparisons, and realistic in recognizing that ultra-high-frequency data often require a different task specification than minute-level benchmarks.

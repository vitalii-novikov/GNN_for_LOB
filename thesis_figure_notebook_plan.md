# Thesis Figure Notebook Plan and Skeleton

## Purpose

Этот файл задает скелет для будущего ноутбука `thesis_figures.ipynb`, в котором каждая figure из `co_om_thesis_enhanced.md` получает свой отдельный раздел.

Цель текущего артефакта:
- определить, какие figures лучше строить кодом в Python;
- какие лучше оформлять как ASCII / prompt / hybrid;
- зафиксировать краткую мотивацию выбора;
- для безкодовых частей сразу дать финальные заготовки: ASCII-схемы и промпты.

## Global notebook contract

- **Planned notebook name:** `thesis_figures.ipynb`
- **Suggested export directory:** `figures_generated/`
- **Figure naming convention:** `fig_1_1`, `fig_3_2`, `fig_5_3`, ...
- **Primary thesis source:** `co_om_thesis_enhanced.md`
- **Rule:** для эмпирических figures сначала использовать repo artifacts (`final_runs/**`, `splits/**`, `*_final_summary.csv`), а не ручной перенос чисел.
- **Rule:** для conceptual/hybrid figures обязательно хранить семантический чек-лист, чтобы финальная картинка не «уплыла» от смысла thesis.
- **Rule:** Figures 3.2 и 3.5 не дублируют друг друга:
  - **3.2** = frequency regimes + aligned holdout design;
  - **3.5** = purged walk-forward chronology + model states (`best_CV`, `last_CV`, `final_refit`).

## Notebook section template

Ниже шаблон, который должен повторяться в ноутбуке для каждой figure:

```markdown
## Figure X.Y — Title

**Why this format**
- 1 very short sentence.

**Metadata**
- Figure type: executable / conceptual / hybrid
- Evidence source: thesis-only / code-grounded / artifact-grounded / mixed
- Rendering method: Python / ASCII / Prompt / Hybrid / Manual-vector-first
- Primary inputs: exact file paths
- Acceptance check: what must be visible in the final figure

**Generation body**
- Python cell plan OR final ASCII OR final image prompt
```

---

## Figure 1.1 — Conceptual pipeline from limit order book snapshots to graph-based entry decisions

**Why this format**
- Hybrid is best because the figure is conceptual, but its objects are tightly tied to repo terminology and thesis pipeline stages.

**Metadata**
- Figure type: `hybrid`
- Evidence source: `mixed`
- Rendering method: `Prompt-first hybrid`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
  - `train_config.yaml`
  - `models/base_gnn_pipeline.py`
  - `models/multigraph_pipeline.py`
  - `models/memorygraph_pipeline.py`
- Acceptance check:
  - ADA, BTC, ETH shown as separate order-book inputs
  - node features and relation-aware edges shown explicitly
  - graph model prediction stage visible
  - entry decision and cost-aware final-holdout backtest visible

**Generation body — final image prompt**

```text
Create a clean academic systems diagram for a master's thesis in quantitative finance.

Title: "Conceptual pipeline from limit order book snapshots to graph-based entry decisions"

Style requirements:
- white background
- publication-quality vector-like look
- minimal color palette: dark blue, muted teal, gray, subtle orange accents
- professional academic layout, no marketing style
- landscape orientation
- sharp readable labels
- consistent arrow styles
- elegant, uncluttered composition

Diagram structure from left to right:
1. Three separate crypto limit order book snapshot panels labeled ADA, BTC, and ETH.
   - each panel should show bid/ask ladders with several levels
   - visually imply that they come from different sampling frequencies
2. Feature extraction block:
   - node features: returns, spread, order flow, depth imbalance, near/far depth shape
   - edge / relation features: price dependence, order flow relation, liquidity relation
3. Graph construction block:
   - three asset nodes: ADA, BTC, ETH
   - directed complete graph with self-loops
   - relation-aware edges across assets
4. Graph model block:
   - one unified graph neural network prediction stage
   - visually imply comparison of graph families, but do not overcrowd the figure
5. Output head block:
   - trade relevance
   - direction
   - return / exit related outputs
6. Decision block:
   - candidate trade activation
   - long / short decision
7. Evaluation block:
   - realized event exit
   - gross PnL
   - transaction-cost adjustment
   - final-holdout net PnL

Important semantic constraint:
This is not a live trading dashboard. It is a controlled research benchmark pipeline from LOB snapshots to graph-based entry decisions and cost-aware backtest evaluation.

Output should look like a thesis figure, not a product infographic.
```

---

## Figure 1.2 — Research-question map for the controlled graph benchmark

**Why this format**
- Python is enough because the figure is essentially a structured mapping from RQ1–RQ4 to benchmark dimensions.

**Metadata**
- Figure type: `conceptual-executable`
- Evidence source: `thesis-only`
- Rendering method: `Python`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - RQ1 mapped to model family
  - RQ2 mapped to Conv vs MPNN
  - RQ3 mapped to temporal resolution
  - RQ4 mapped to `last_CV` vs `final_refit`

**Generation body**
- Build a 4-row mapping diagram or matrix.
- Left column: `RQ1`–`RQ4`.
- Right side: benchmark dimensions with arrows / assignment cells.
- Keep the layout compact and thesis-like, not decorative.

---

## Figure 3.1 — Graph input representation for the three-asset limit order book benchmark

**Why this format**
- Python is appropriate because the graph topology is explicit and can be rendered reproducibly, but annotations must emphasize relation semantics.

**Metadata**
- Figure type: `executable`
- Evidence source: `code-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `train_config.yaml`
  - `models/base_gnn_pipeline.py`
  - `models/multigraph_pipeline.py`
  - `models/memorygraph_pipeline.py`
- Acceptance check:
  - ADA, BTC, ETH shown as nodes
  - directed complete graph with self-loops
  - relation channels `price_dep`, `order_flow`, `liquidity`
  - node features vs edge features visibly separated

**Generation body**
- Use a fixed node layout triangle.
- Show self-loops explicitly.
- Label edge semantics by relation channel.
- Add a small side annotation explaining node tensor and relation-aware edge tensor.

---

## Figure 3.2 — Frequency regimes and final-holdout split design

**Why this format**
- Python is the right choice because the split structure exists in `split_summary.json` and should be reproduced from artifacts, not hand-drawn.

**Metadata**
- Figure type: `executable`
- Evidence source: `artifact-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `final_runs/5min-base-gnn/splits/split_summary.json`
  - `final_runs/1min-base-gnn-conv/splits/split_summary.json`
  - `final_runs/1sec-base-gnn-conv/splits/split_summary.json`
- Acceptance check:
  - `5min`, `1min`, `1sec` shown side by side
  - pre-holdout and final holdout clearly distinguished
  - 1sec regime visibly marked as adapted high-frequency regime
  - alignment of final holdout intervals conveyed

**Generation body**
- One horizontal timeline per frequency.
- Show working slice, pre-holdout, holdout.
- Annotate lookback / horizon / folds briefly.
- Add a small note that `1sec` is a frequency-adapted stress test.

---

## Figure 3.3 — Triple-barrier target construction for the ETH midpoint

**Why this format**
- Hybrid is best because the core path/barrier visualization can be built in Python, but the explanatory annotations need tight manual control.

**Metadata**
- Figure type: `hybrid`
- Evidence source: `mixed`
- Rendering method: `Python-primary`
- Primary inputs:
  - `../Graph_Neural_Network_for_Market_Microstructure/dataset/ETH_1min.csv`
  - `../Graph_Neural_Network_for_Market_Microstructure/dataset/ETH_5min.csv`
  - `train_config.yaml`
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - target timestamp visible
  - upper, lower, vertical barriers visible
  - realized exit marked
  - realized return, trade relevance, direction label explained
  - representative-example selection rule documented

**Generation body**
- Select one reproducible example timestamp with a clear barrier hit.
- Plot midpoint path after target time.
- Draw upper/lower/vertical barriers.
- Add labels for realized exit, realized return, trade label, direction label.
- Record timestamp selection rule in notebook text.

---

## Figure 3.4 — Common entry-model backtest and post-cost PnL calculation

**Why this format**
- Hybrid fits best because the figure is a conceptual evaluation pipeline with formulas, not a raw empirical plot.

**Metadata**
- Figure type: `hybrid`
- Evidence source: `mixed`
- Rendering method: `Prompt-first hybrid`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
  - `train_config.yaml`
- Acceptance check:
  - trade head activation shown
  - direction choice shown
  - realized event exit shown
  - gross PnL to net PnL via cost proxy shown
  - cost formula conceptually visible

**Generation body — final image prompt**

```text
Create a clean academic flowchart for a quantitative finance thesis.

Title: "Common entry-model backtest and post-cost PnL calculation"

Visual style:
- white background
- minimal academic design
- dark navy text and lines, muted teal blocks, subtle orange for cost adjustment
- landscape layout
- crisp vector-like diagram
- no 3D effects, no glossy elements

Flow structure from left to right:
1. Model outputs block with two emphasized heads:
   - trade head
   - direction head
2. Decision block:
   - if trade head activates, open candidate position
   - direction head chooses long or short
3. Event-based holding block:
   - position remains open until realized event exit
   - upper barrier / lower barrier / vertical barrier concept may be shown compactly
4. Gross PnL block:
   - gross PnL computed from position side times realized return
5. Cost adjustment block:
   - subtract round-trip transaction cost proxy
6. Final output block:
   - net PnL on final holdout

Include compact mathematical flavor without overcrowding:
- gross_pnl = side × realized_return
- net_pnl = gross_pnl − cost_proxy

Important semantic constraint:
This is a controlled benchmark backtest illustration, not a full execution simulator and not a production trading system.
```

---

## Figure 3.5 — Purged walk-forward validation and deployment-oriented model states

**Why this format**
- Python is the strongest choice because both chronology and model-state comparison can be anchored to split artifacts and reported states.

**Metadata**
- Figure type: `executable`
- Evidence source: `artifact-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `final_runs/5min-base-gnn/splits/split_summary.json`
  - `final_runs/1min-base-gnn-conv/splits/split_summary.json`
  - `final_runs/1sec-base-gnn-conv/splits/split_summary.json`
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - train / purge / validation / purge / test visible
  - chronological order obvious
  - final holdout separate from CV process
  - `best_CV`, `last_CV`, `final_refit` produced from the diagram

**Generation body**
- Draw fold chronology explicitly.
- Use a second annotation layer to show how each model state is obtained.
- Keep it distinct from Figure 3.2 by focusing on fold mechanics rather than regime comparison.

---

## Figure 3.6 — Metric hierarchy for deployment-oriented interpretation

**Why this format**
- ASCII is enough because this is a hierarchy of interpretation, not a measured plot.

**Metadata**
- Figure type: `conceptual`
- Evidence source: `thesis-only`
- Rendering method: `ASCII`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
  - metric names from `final_runs/*/final_report.csv`
- Acceptance check:
  - AUC metrics shown as diagnostics
  - `gross_pnl_sum` shown as signal extraction
  - `n_trades` shown as turnover evidence
  - `pnl_sum` shown as primary economic outcome

**Generation body — final ASCII**

```text
                     DEPLOYMENT-ORIENTED METRIC HIERARCHY

                 ┌──────────────────────────────────────┐
                 │ Ranking diagnostics                  │
                 │  - dir_auc                           │
                 │  - trade_auc                         │
                 └──────────────────────────────────────┘
                                   │
                                   ▼
                 ┌──────────────────────────────────────┐
                 │ Pre-cost signal extraction           │
                 │  - gross_pnl_sum                     │
                 └──────────────────────────────────────┘
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
     ┌──────────────────────────────┐   ┌──────────────────────────────┐
     │ Turnover evidence            │   │ Selectivity / activity check │
     │  - n_trades                  │   │  - is the result supported   │
     │  - trade rate (optional)     │   │    by meaningful trading?    │
     └──────────────────────────────┘   └──────────────────────────────┘
                    │                             │
                    └──────────────┬──────────────┘
                                   ▼
                 ┌──────────────────────────────────────┐
                 │ Primary deployment outcome           │
                 │  - pnl_sum (post-cost net PnL)       │
                 └──────────────────────────────────────┘
```

---

## Figure 4.1 — Architecture comparison of `base_gnn`, `multigraph`, and `memorygraph`

**Why this format**
- Hybrid is best because the figure is conceptual but must stay faithful to actual architectural differences visible in the code.

**Metadata**
- Figure type: `hybrid`
- Evidence source: `code-grounded`
- Rendering method: `Prompt-first hybrid`
- Primary inputs:
  - `models/base_gnn_pipeline.py`
  - `models/multigraph_pipeline.py`
  - `models/memorygraph_pipeline.py`
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - `base_gnn` = early relation fusion
  - `multigraph` = relation-specific pathways before fusion
  - `memorygraph` = recurrent node-edge memory updates
  - shared output heads visible across all families

**Generation body — final image prompt**

```text
Create a publication-quality academic architecture comparison diagram for a master's thesis.

Title: "Architecture comparison of base_gnn, multigraph, and memorygraph"

Style:
- white background
- minimal, elegant, vector-like
- 3 aligned columns for the 3 model families
- restrained palette: navy, teal, gray, muted orange accents
- readable labels, no decorative clutter

Column 1: base_gnn
- show relation channels being fused early
- then one shared temporal / graph processing stream
- then common output heads

Column 2: multigraph
- show separate relation-specific graph pathways
- later learned fusion / aggregation
- then common output heads

Column 3: memorygraph
- show recurrent node and edge memory states
- graph interaction inside the recurrent loop
- then common output heads

Shared bottom section:
- common heads for trade relevance, direction, return, exit-related outputs

Important semantic constraint:
This is a controlled benchmark comparison of architectural families, not a generic neural-network infographic. The main contrast is when and how relation information is fused, and whether memory is stateful.
```

---

## Figure 4.2 — Recurrent node and edge memory update in `memorygraph`

**Why this format**
- Hybrid is necessary because the underlying mechanism is code-grounded but visually too complex for a purely auto-laid-out plot.

**Metadata**
- Figure type: `hybrid`
- Evidence source: `code-grounded`
- Rendering method: `Prompt-first hybrid`
- Primary inputs:
  - `models/memorygraph_pipeline.py`
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - edge memory depends on current edge state and node states
  - node memory update follows relation-specific edge context
  - recurrent loop visible
  - not confused with static temporal convolution

**Generation body — final image prompt**

```text
Create a clean academic recurrent-mechanism diagram for a graph neural network thesis.

Title: "Recurrent node and edge memory update in memorygraph"

Style:
- white background
- publication figure style
- vector-like clean blocks and arrows
- minimal palette: navy, teal, gray, subtle orange highlights
- emphasize information flow, not decoration

Diagram content:
1. Inputs to edge memory update:
   - current edge state
   - source node state
   - destination node state
   - pairwise node interaction
2. Edge memory update block
3. Relation-specific edge context aggregation
4. Node memory update block
5. Updated node state and updated edge state
6. Explicit recurrent loop across time steps

Semantic constraint:
This is a recurrent graph-memory mechanism. The figure must clearly show that graph interaction happens inside a time-evolving memory update loop, rather than in a purely feed-forward architecture.
```

---

## Figure 5.1 — Benchmark overview by frequency, graph family, and operator

**Why this format**
- Python is mandatory because this figure is a direct visual summary of benchmark results and should be generated from the result tables.

**Metadata**
- Figure type: `executable`
- Evidence source: `artifact-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `final_runs/*/final_report.csv`
  - optionally `*_final_summary.csv` for a stricter unified source
- Acceptance check:
  - all 18 primary `last_CV` model-frequency configurations included
  - grouping by frequency clear
  - grouping by family/operator clear
  - metric shown is `pnl_sum`
  - caution note that this is not a significance-test figure

**Generation body**
- Extract `last_cv` rows only.
- Normalize labels into 18 comparable benchmark entries.
- Use grouped bar chart or heatmap.
- Add visual separator between `5min`, `1min`, `1sec`.
- Add a small note: no uncertainty intervals / no formal dominance testing.

---

## Figure 5.2 — Gross versus net PnL at `1sec`

**Why this format**
- Python is best because the figure is a direct comparison of reported metrics and trade counts.

**Metadata**
- Figure type: `executable`
- Evidence source: `artifact-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `final_runs/1sec-*/final_report.csv`
- Acceptance check:
  - six 1sec models included
  - both `gross_pnl_sum` and `pnl_sum` visible
  - `n_trades` visible as annotation or secondary encoding
  - `memory-gnn-conv` cost-drag stands out clearly

**Generation body**
- Build paired bars for gross vs net PnL.
- Annotate or color-encode `n_trades`.
- Add a note explaining that cost burden dominates at extreme turnover.

---

## Figure 5.3 — `last_CV` versus `final_refit` as deployment-oriented model states

**Why this format**
- Python is appropriate because the comparison already exists in artifact tables and should be shown as a paired state comparison, not just as a conceptual diagram.

**Metadata**
- Figure type: `executable`
- Evidence source: `artifact-grounded`
- Rendering method: `Python`
- Primary inputs:
  - `final_runs/*/final_report.csv`
  - `final_runs/**/*final_holdout_model_comparison_summary.csv`
- Acceptance check:
  - `last_CV` vs `final_refit` explicitly contrasted
  - deployment-primary status of `last_CV` visible
  - `final_refit` shown as informative but non-primary
  - at least selected representative model cases included

**Generation body**
- Prefer slope chart or paired bars.
- Use a small subset of representative models if full matrix is too dense.
- Preserve semantic emphasis: deployment reference vs larger-sample diagnostic comparison.

---

## Figure 6.1 — Deployment interpretation from prediction to post-cost evidence

**Why this format**
- ASCII works well because this is a reasoning chain rather than a data graphic.

**Metadata**
- Figure type: `conceptual`
- Evidence source: `thesis-only`
- Rendering method: `ASCII`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - predictive ranking shown as first gate
  - gross signal extraction shown
  - trade selectivity shown
  - transaction-cost adjustment shown
  - model-state stability shown before deployment-informative conclusion

**Generation body — final ASCII**

```text
 prediction quality
 (ranking diagnostics)
         │
         ▼
 gross signal extraction
 (`gross_pnl_sum` before cost)
         │
         ▼
 trade selectivity / turnover
 (`n_trades`, activity discipline)
         │
         ▼
 transaction-cost adjustment
 (gross edge must survive frictions)
         │
         ▼
 model-state stability
 (`last_CV` vs `final_refit` interpretation)
         │
         ▼
 deployment-informative evidence
 (`pnl_sum` after cost, with stable interpretation)
```

---

## Figure 7.1 — Future research roadmap for deployment-oriented graph LOB prediction

**Why this format**
- Manual-vector-first is the safest option because the roadmap is dense, conceptual, and taxonomy-sensitive; prompt can help as a draft, but should not be the authoritative final form.

**Metadata**
- Figure type: `conceptual`
- Evidence source: `thesis-only`
- Rendering method: `Manual-vector-first with prompt fallback`
- Primary inputs:
  - `co_om_thesis_enhanced.md`
- Acceptance check:
  - all seven roadmap directions present
  - categories remain thesis-faithful
  - visual grouping is clear
  - no generic AI/fintech visual clichés

**Generation body — final image prompt fallback**

```text
Create an academic roadmap infographic for a master's thesis in graph-based market microstructure modelling.

Title: "Future research roadmap for deployment-oriented graph LOB prediction"

Style:
- white background
- elegant publication-quality layout
- vector-like academic figure
- restrained palette: navy, teal, gray, muted orange accents
- clean typography
- avoid futuristic or commercial fintech aesthetics

Central concept:
A roadmap centered on deployment-oriented graph LOB prediction.

Surrounding thematic branches:
1. turnover-aware learning
2. execution-aware evaluation
3. larger graph universes
4. selective memory mechanisms
5. regime robustness
6. uncertainty quantification
7. cost-sensitivity analysis

Composition guidance:
- use a hub-and-spoke or clustered roadmap layout
- each branch should look like a research direction, not a product feature
- emphasize structure, hierarchy, and academic clarity

Important semantic constraint:
This figure is a future research agenda for a controlled benchmark thesis. It should communicate research extensions, not implementation backlog items.
```

---

## Summary table

| Figure | Recommended rendering | Evidence source | Confidence |
|---|---|---|---|
| 1.1 | Hybrid (prompt-first) | Mixed | Medium |
| 1.2 | Python | Thesis-only | High |
| 3.1 | Python | Code-grounded | Medium |
| 3.2 | Python | Artifact-grounded | High |
| 3.3 | Hybrid (Python-primary) | Mixed | Medium |
| 3.4 | Hybrid (prompt-first) | Mixed | Medium |
| 3.5 | Python | Artifact-grounded | High |
| 3.6 | ASCII | Thesis-only | High |
| 4.1 | Hybrid (prompt-first) | Code-grounded | Medium |
| 4.2 | Hybrid (prompt-first) | Code-grounded | Medium |
| 5.1 | Python | Artifact-grounded | High |
| 5.2 | Python | Artifact-grounded | High |
| 5.3 | Python | Artifact-grounded | High |
| 6.1 | ASCII | Thesis-only | High |
| 7.1 | Manual-vector-first, prompt fallback | Thesis-only | Medium |

## Recommended implementation order for the future notebook

1. Start with highest-confidence executable figures:
   - 3.2
   - 3.5
   - 5.1
   - 5.2
   - 5.3
2. Then implement simple conceptual/executable figures:
   - 1.2
   - 3.1
3. Then fill hybrid figures:
   - 1.1
   - 3.3
   - 3.4
   - 4.1
   - 4.2
4. Finally finalize thesis-only conceptual visuals:
   - 3.6
   - 6.1
   - 7.1

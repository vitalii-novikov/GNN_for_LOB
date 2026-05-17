# Foundational Figure Prompts and Production Specs

Date: 2026-05-13  
Source text: `paper_artifacts/good_Paper_draft.md`  
Supporting artifact references:

- `paper_artifacts/figures_generated/figure_manifest.md`
- `paper_artifacts/figures_generated/figure_manifest.csv`
- `models/base_gnn_pipeline.py`
- `models/multigraph_pipeline.py`
- `train.py`
- `final_runs/*/final_report.csv`

## Scope note

These specs translate the current manuscript mentions of the following figures into production-ready prompts in the same style as `paper_artifacts/chapter4_figure_specs.md`:

- Figure 1.1 — Conceptual pipeline
- Figure 1.2 — Research-question map
- Figure 3.1 — Graph input representation
- Figure 3.4 — Entry backtest and PnL

## Evidence summary from the current manuscript

### Direct evidence

- `paper_artifacts/good_Paper_draft.md:37-44` states that Figure 1.1 summarizes the controlled pipeline from frequency-specific ADA, BTC, and ETH LOB snapshots to graph-based entry decisions and cost-aware final-holdout evaluation.
- `paper_artifacts/good_Paper_draft.md:62-86` states that Figure 1.2 connects the four research questions to model family, graph operator, temporal resolution, and deployment-oriented model state.
- `paper_artifacts/good_Paper_draft.md:185-208` states that Figure 3.1 shows a directed complete graph over ADA, BTC, and ETH with ETH as target and relation-aware edge channels for `price_dep`, `order_flow`, and `liquidity`.
- `paper_artifacts/good_Paper_draft.md:358-393` states that Figure 3.4 shows the common entry-model benchmark: trade-head activation, direction selection, realized event exit, and conversion from gross PnL to net PnL using a shared transaction-cost proxy.
- `paper_artifacts/figures_generated/figure_manifest.csv:2-6` confirms existing manifest conventions and gives explicit source-grounding for Figures 1.2, 3.1, and 3.4.

### Inference

- Figure 1.1 should be visually broader and more explanatory than Figures 3.1 and 3.4, because it synthesizes the whole benchmark logic rather than one isolated subsystem.
- Figure 1.2 should be schematic and relational, not process-oriented, because the manuscript frames it as a map between research questions and experimental dimensions.
- Figure 3.1 should be more formal and data-structural than Figure 1.1, because it sits in the methodology chapter and is described with node/edge tensors and relation channels.
- Figure 3.4 should emphasize fairness and shared evaluation logic across model families, because the text uses it to justify the common benchmark rather than to depict a production trading engine.

### Unknown / limit

- The repository does not contain a manifest entry for Figure 1.1, so its final source-artifact field is less constrained than the others.
- The exact rendering code for the existing figures is not inspected here; these specs are grounded primarily in manuscript language and the figure manifest.

---

## Shared design system for these four figures

### Visual goal

These figures should establish the thesis logic from introduction through methodology:

- what the benchmark studies
- how the research questions are organized
- how the graph input is represented
- how the shared trading evaluation works

### Style requirements

- academic, vector-first, publication-ready
- white or very light neutral background
- consistent typography and arrow style across figures
- simple color semantics with minimal decorative elements
- no screenshot-like layouts
- no raw code blocks or implementation-noise labels

### Shared semantic color mapping

- **blue**: data inputs and observed market information
- **orange**: relation / graph structure
- **green**: model or decision-processing stages
- **purple**: evaluation, readout, or result integration
- **gray**: contextual notes, shared constraints, or benchmark framing
- **red accent**: costs, friction, or cautionary benchmark limitations when needed

### What to avoid

- production-trading visual language that implies live deployment
- exchange-specific order-matching detail not discussed in the thesis
- excessive tensor algebra in introductory figures
- redundant labels repeated in full sentences inside the diagram

---

## Figure 1.1 — Conceptual pipeline from limit order book snapshots to graph-based entry decisions

### Narrative purpose

This figure should introduce the thesis as a **controlled end-to-end benchmark pipeline** from raw market observations to friction-aware evaluation.

### Main claim to communicate

The thesis does not study isolated predictions in the abstract. It studies a controlled pipeline in which frequency-specific multi-asset LOB inputs are transformed into graph representations, model outputs, entry decisions, and final post-cost evaluation.

### Evidence from the manuscript

- `paper_artifacts/good_Paper_draft.md:37-44`
- Figure caption at `paper_artifacts/good_Paper_draft.md:39-42`

### Recommended layout

Use a left-to-right staged pipeline with 6 major blocks:

1. **frequency-specific LOB snapshots**
   - ADA
   - BTC
   - ETH
   - 5min / 1min / 1sec
2. **feature construction**
   - node features
   - relation states
3. **three-asset graph representation**
   - ADA, BTC, ETH nodes
   - relation-aware edges
4. **graph model benchmark**
   - `base_gnn`
   - `multigraph`
   - `memorygraph`
   - Conv / MPNN operator choice
5. **entry-model outputs**
   - trade relevance
   - direction
   - return / auxiliary outputs
6. **cost-aware final-holdout evaluation**
   - event-based backtest
   - gross PnL
   - net PnL
   - deployment-oriented comparison

### Required visual emphasis

- show ADA, BTC, and ETH as a shared multi-asset input rather than isolated pipelines
- show that graph representation is the bridge between raw LOB data and model-family comparison
- show that the benchmark ends in **cost-aware** evaluation, not just prediction
- visually signal that the study is **controlled** and **not a full production trading system**

### Recommended side note

Add a small gray annotation near the final block:

`controlled entry-model benchmark, not a full execution simulator`

### Caption-ready wording

**Figure 1.1. Conceptual pipeline from limit order book snapshots to graph-based entry decisions.**  
The figure summarizes the controlled research pipeline studied in the thesis: frequency-specific ADA, BTC, and ETH limit order book snapshots are transformed into node and relation features, represented as a three-asset graph, processed by graph-model families under a shared benchmark, converted into entry-model decisions, and evaluated on a cost-aware final holdout.

### Production prompt

Create a clean academic vector diagram showing the conceptual thesis pipeline from limit order book data to final benchmark evaluation. Use a left-to-right flow. Start with a block for frequency-specific cryptocurrency LOB snapshots containing ADA, BTC, and ETH and labels 5min, 1min, and 1sec. Then show feature construction split into node features and relation states. Next show a three-asset graph representation with ADA, BTC, and ETH as nodes and relation-aware edges between them. Then show a graph benchmark block containing base_gnn, multigraph, and memorygraph, with a small note that each family is evaluated with Conv or MPNN operators. After that show entry-model outputs such as trade, direction, return, and auxiliary targets. End with a cost-aware final-holdout evaluation block showing event-based backtest, gross PnL, net PnL, and deployment-oriented comparison. Use blue for input data, orange for graph structure, green for model/decision stages, purple for evaluation/result integration, and gray for benchmark caveats. Add a small note that this is a controlled entry-model benchmark rather than a full production trading system. Keep the figure publication-ready and conceptually clear.

---

## Figure 1.2 — Research-question map for the controlled graph benchmark

### Narrative purpose

This figure should organize the thesis research questions into a compact conceptual map before the reader reaches the detailed methods table.

### Main claim to communicate

The four research questions are not independent; they jointly span four benchmark dimensions:

- model family
- graph operator
- temporal resolution
- deployment-oriented model state

### Evidence from the manuscript

- `paper_artifacts/good_Paper_draft.md:62-86`
- manifest entry `paper_artifacts/figures_generated/figure_manifest.csv:2`

### Recommended layout

Use a hub-and-spoke or matrix-style concept map.

Recommended center node:

`Controlled graph benchmark for ETH entry-model evaluation`

From the center, branch to four research-question nodes:

- **RQ1** — best graph family
- **RQ2** — Conv vs MPNN operator choice
- **RQ3** — temporal resolution and value of relation/memory mechanisms
- **RQ4** — stability between `last_CV` and `final_refit`

Each RQ node should visually connect to its key experimental dimension:

- RQ1 -> `base_gnn / multigraph / memorygraph`
- RQ2 -> `Conv / MPNN`
- RQ3 -> `5min / 1min / 1sec`
- RQ4 -> `last_CV / final_refit`

### Required visual emphasis

- make the four research questions visually parallel
- show that all four belong to one shared benchmark rather than four separate studies
- preserve the language of “controlled benchmark” and “deployment-oriented model state”

### Optional supporting row

At the bottom, add a light summary strip:

`shared data, shared targets, shared validation, shared backtest`

This reinforces the controlled-comparison logic from the manuscript.

### Caption-ready wording

**Figure 1.2. Research-question map for the controlled graph benchmark.**  
The figure maps the four research questions onto the core dimensions of the benchmark: model family, graph operator, temporal resolution, and deployment-oriented model state. It emphasizes that the thesis evaluates these questions within one shared controlled benchmark rather than through disconnected experiments.

### Production prompt

Create an academic concept-map figure for a thesis research design. Place a central node labeled Controlled graph benchmark for ETH entry-model evaluation. Around it, place four clearly balanced research-question nodes: RQ1 Which graph family performs best, RQ2 How important is Conv versus MPNN operator choice, RQ3 How does temporal resolution change the value of relation and memory mechanisms, and RQ4 Are conclusions stable between last_CV and final_refit. Connect each node to a compact dimension label: base_gnn / multigraph / memorygraph for RQ1, Conv / MPNN for RQ2, 5min / 1min / 1sec for RQ3, and last_CV / final_refit for RQ4. Add a subtle bottom strip saying shared data, shared targets, shared validation, shared backtest. Keep the style balanced, clean, vector-based, and suitable for an academic thesis.

---

## Figure 3.1 — Graph input representation for the three-asset limit order book benchmark

### Narrative purpose

This figure should formalize the common graph input used by all model families.

### Main claim to communicate

All families receive the same graph-structured input: a three-node directed complete graph with self-loops, ETH as target, dynamic node states, and three relation-aware edge channels.

### Evidence from the manuscript

- `paper_artifacts/good_Paper_draft.md:185-208`
- `paper_artifacts/good_Paper_draft.md:212-250`
- manifest entry `paper_artifacts/figures_generated/figure_manifest.csv:3`

### Recommended layout

Use a central graph diagram plus side annotations.

Core visual:

- three nodes:
  - ADA
  - BTC
  - ETH
- ETH visually marked as `target asset`
- directed edges between every pair
- self-loops shown subtly

Relation-layer annotation:

- `price_dep`
- `order_flow`
- `liquidity`

Side panels:

- left panel: node feature concept
  - local price behavior
  - order-flow pressure
  - depth structure
- right panel: edge feature concept
  - rolling correlation
  - rolling beta
  - rolling mean product

Optional tensor annotation band:

- `B` batch
- `L` lookback
- `N = 3` assets
- `R = 3` relation channels

### Required visual emphasis

- make it explicit that the graph is **directed complete** with self-loops
- make ETH visually distinct as the supervised target asset
- make relation channels appear as edge semantics, not as three separate disconnected graphs
- preserve the sense that node and edge states vary over time even though node identities are fixed

### Recommended caption note

If one extra note is needed:

`all model families share this same graph input representation`

### Caption-ready wording

**Figure 3.1. Graph input representation for the three-asset limit order book benchmark.**  
The figure shows the common graph input used throughout the benchmark: ADA, BTC, and ETH form a directed complete graph with self-loops, ETH is the target asset, node states summarize local microstructure information, and edges carry three relation-aware channels—price dependence, order flow, and liquidity.

### Production prompt

Create a formal academic vector diagram for the graph input representation used in a three-asset market-microstructure benchmark. Place a central directed complete graph with three nodes labeled ADA, BTC, and ETH, and show subtle self-loops on each node. Mark ETH clearly as the target asset. Represent the edges as relation-aware and annotate them with three semantic channels: price_dep, order_flow, and liquidity. Add a compact left panel summarizing node features such as local price behavior, order-flow pressure, and depth structure. Add a compact right panel summarizing edge features such as rolling correlation, rolling beta, and rolling mean product. Include a minimal tensor-notation band if space allows: B batch, L lookback, N=3 assets, R=3 relation channels. Use blue for node-side information, orange for relation/edge information, and gray or purple for structural annotations. Keep the layout precise, methodology-oriented, and publication-ready.

---

## Figure 3.4 — Common entry-model backtest and post-cost PnL calculation

### Narrative purpose

This figure should make the common trading evaluation logic transparent and show why the benchmark is fair across model families.

### Main claim to communicate

All model families are evaluated under the same entry-model backtest: trade activation, direction selection, realized-event exit, and a shared transaction-cost proxy that converts gross PnL to net PnL.

### Evidence from the manuscript

- `paper_artifacts/good_Paper_draft.md:358-393`
- manifest entry `paper_artifacts/figures_generated/figure_manifest.csv:6`

### Recommended layout

Use a left-to-right evaluation flow with one decision branch and one formula panel.

Main pipeline:

1. model output heads
   - trade head
   - direction head
   - auxiliary outputs
2. trade candidate activation
3. long / short decision
4. sequential non-overlapping event-based position
5. realized event exit
6. realized return
7. gross PnL
8. transaction-cost proxy
9. net PnL

### Required visual emphasis

- show that the trade head controls whether a candidate is active
- show that direction only matters after activation
- show that no new position opens until the active one closes
- show the common cost proxy as a shared benchmark rule, not a family-specific component

### Formula box

Include a compact side box or footer:

- `gross PnL_i = s_i r_i`
- `net PnL_i = gross PnL_i - c_rt`
- `c_rt = 3 * cost_bps_per_side * 10^-4`
- example: `cost_bps_per_side = 1.0 -> c_rt = 0.0003`

### Cautionary annotation

Add a gray or red-accent note:

`transparent cost proxy for controlled comparison, not a full execution simulator`

### Caption-ready wording

**Figure 3.4. Common entry-model backtest and post-cost PnL calculation.**  
The figure summarizes the shared trading evaluation used throughout the benchmark. The trade head determines whether a candidate is active, the direction head selects long or short exposure, positions are closed by the same realized event rule for all families, and a common transaction-cost proxy converts gross PnL into net PnL for friction-aware comparison.

### Production prompt

Create a publication-ready academic flow diagram for the common entry-model backtest used in a graph-based trading benchmark. Show a left-to-right sequence starting from model output heads with trade head, direction head, and auxiliary outputs. Then show trade candidate activation, then a long/short decision, then a sequential non-overlapping position block, then realized event exit, then realized return, then gross PnL, then subtraction of a shared transaction-cost proxy, and finally net PnL. Make it visually clear that a new position cannot open until the current one closes. Include a compact formula panel with gross PnL_i = s_i r_i, net PnL_i = gross PnL_i minus c_rt, c_rt = 3 times cost_bps_per_side times 10^-4, and the example cost_bps_per_side = 1.0 gives c_rt = 0.0003. Add a note that this is a transparent cost proxy for controlled comparison rather than a full execution simulator. Use green for decision stages, purple for evaluation and PnL stages, gray for benchmark constraints, and a red accent for transaction costs.

---

## Implementation checklist for the figure author

Before finalizing these figures, verify:

1. Figure 1.1 ends in cost-aware final-holdout evaluation rather than stopping at prediction.
2. Figure 1.1 does not imply a production execution engine.
3. Figure 1.2 maps every research question to the correct benchmark dimension.
4. Figure 1.2 visually reinforces the shared-benchmark logic.
5. Figure 3.1 clearly marks ETH as the target asset.
6. Figure 3.1 shows relation-aware edges for `price_dep`, `order_flow`, and `liquidity`.
7. Figure 3.4 shows the shared trade-activation -> direction -> realized-event-exit flow.
8. Figure 3.4 includes the common cost proxy and the caution that the benchmark is not a full execution simulator.
9. All four figures remain visually consistent with one another and with the Chapter 4 figure style.

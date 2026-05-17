# Chapter 4 Figure Prompts and Production Specs

Date: 2026-05-13  
Source text: `paper_artifacts/good_Paper_draft.md`  
Code-grounding sources:

- `models/base_gnn_pipeline.py`
- `models/multigraph_pipeline.py`
- `models/memorygraph_pipeline.py`

## Scope note

These specs translate the revised Chapter 4 text into figure-ready prompts for **Figures 4.1–4.4**.

Important numbering note:

- the current manuscript already contains **Figure 4.1** as the high-level family comparison;
- the current manuscript also uses **Figure 4.2** for the memorygraph recurrent-memory diagram;
- if the full Chapter 4 visual plan is adopted, the recommended final numbering becomes:
  - **Figure 4.1** — high-level family comparison
  - **Figure 4.2** — detailed `base_gnn` architecture
  - **Figure 4.3** — detailed `multigraph` architecture
  - **Figure 4.4** — detailed `memorygraph` architecture

So the current memorygraph figure should be renumbered or rebuilt as Figure 4.4 in the final manuscript layout.

---

## Shared design system for all Chapter 4 figures

### Visual goal

The Chapter 4 figures should read as a **controlled architectural experiment**, not as software-engineering diagrams.

### Style requirements

- academic, clean, vector-first layout
- white background
- limited color palette with stable semantic meaning across all four figures
- avoid screenshots, code snippets, tensor-shape clutter, or implementation-specific function arguments
- use concise labels that match Chapter 4 prose and pseudo-code

### Shared semantic color mapping

- **blue**: node-side temporal/state processing
- **orange**: edge / relation processing
- **green**: graph operator or graph message-passing stage
- **purple**: fusion / readout / output integration
- **gray**: shared outputs and common benchmark components
- **accent red or dark magenta**: recurrent memory/statefulness, only where needed

### Shared labels to keep consistent

- `X_node`
- `X_edge`
- `HybridEdgeFeatureFusion`
- `GraphReadout`
- `TargetTemporalTrunk` or `Output Projection` where applicable
- shared heads:
  - `trade`
  - `direction`
  - `return`
  - `exit type`
  - `time-to-event`

### Shared output-head block

All four figures should end in the same compact gray output block:

`shared multi-task heads -> trade / direction / return / exit type / time-to-event`

### What to avoid

- raw class definitions
- PyTorch helper names such as `torch.nan_to_num`, `reshape`, `squeeze`
- low-level training utilities
- regularization bookkeeping
- optimizer or loss details
- tensor dimension formulas unless absolutely necessary for a single annotation

---

## Figure 4.1 — Architecture comparison of `base_gnn`, `multigraph`, and `memorygraph`

### Narrative purpose

This is the **high-level comparison panel** for the whole chapter. It should show that the families share inputs and outputs, but differ in:

1. relation-fusion timing
2. graph-pathway structure
3. temporal mechanism
4. statefulness

### Main claim to communicate

The family comparison is controlled because the models share the same data, target construction, and output heads; only the architecture of relation handling and temporal processing changes.

### Recommended layout

Three vertical columns, one per family:

- left: `base_gnn`
- center: `multigraph`
- right: `memorygraph`

Each column should follow the same top-to-bottom structure:

`shared inputs -> family-specific processing -> shared readout/heads`

Add a small comparison strip or callout row underneath the three columns with four aligned labels:

- relation handling
- graph pathway
- temporal mechanism
- statefulness

### Family-specific content

#### `base_gnn`

- shared `X_node` and `X_edge`
- temporal encoders
- `HybridEdgeFeatureFusion`
- **early relation fusion**
- **single graph operator block**
- readout and shared heads

Short callout:

`early fusion before message passing`

#### `multigraph`

- shared `X_node` and `X_edge`
- temporal encoders
- `HybridEdgeFeatureFusion`
- **three relation-specific graph pathways**
  - `price_dep`
  - `order_flow`
  - `liquidity`
- **late relation attention fusion**
- readout and shared heads

Short callout:

`separate relation pathways before fusion`

#### `memorygraph`

- shared `X_node_t` and `X_edge_t`
- step projectors
- `HybridEdgeFeatureFusion`
- recurrent edge-memory update
- graph operator inside loop
- recurrent node-memory update
- readout and shared heads

Short callout:

`recurrent node-edge state across chunks`

### Required annotations

- place a bracket or marker over `base_gnn` showing **early relation fusion**
- place a bracket or marker over `multigraph` showing **late relation fusion**
- place a loop arrow around the core memorygraph block showing **recurrent state**
- add a small side note: `Conv or MPNN operator inside each family`

### Caption-ready wording

**Figure 4.1. Architecture comparison of base_gnn, multigraph, and memorygraph.**  
The figure compares the three model families under a shared benchmark. All families use the same node and edge inputs and the same multi-task output heads, but they differ in when relation information is fused, whether graph processing occurs through a single or relation-specific pathway, and whether temporal information is represented through convolutional windows or recurrent memory.

### Production prompt

Create an academic vector diagram comparing three graph-model families for a thesis chapter on market microstructure. Use three aligned columns labeled base_gnn, multigraph, and memorygraph. All columns must start from shared inputs X_node and X_edge and end in the same gray multi-task output heads block: trade, direction, return, exit type, time-to-event. In the base_gnn column, show node and edge temporal encoders, HybridEdgeFeatureFusion, early relation fusion, one single graph operator block, GraphReadout, and TargetTemporalTrunk. In the multigraph column, show the same shared encoders and HybridEdgeFeatureFusion, then three separate relation-specific graph pathways labeled price_dep, order_flow, and liquidity, then late relation attention fusion, GraphReadout, and TargetTemporalTrunk. In the memorygraph column, show step-wise node and edge projectors, HybridEdgeFeatureFusion, recurrent edge-memory update, graph operator inside a loop, recurrent node-memory update, GraphReadout, and output projection. Use consistent semantic colors: blue for node processing, orange for edge/relation processing, green for graph operators, purple for fusion/readout, gray for shared heads, and dark magenta for recurrent memory. Add concise callouts for early fusion, late fusion, and recurrent state. Keep the layout clean, white-background, publication-ready, and non-code-like.

---

## Figure 4.2 — Detailed `base_gnn` architecture

### Narrative purpose

This figure should explain why `base_gnn` is the **clean single-graph baseline**, not just a simpler model.

### Main claim to communicate

`base_gnn` compresses relation channels **before** graph message passing, then runs one graph pathway over the fused edge representation.

### Recommended layout

Single left-to-right pipeline with 7 blocks:

1. `X_node`
2. `NodeTemporalEncoder`
3. `X_edge`
4. `EdgeTemporalEncoder`
5. `HybridEdgeFeatureFusion`
6. `EdgeRelationFusion`
7. `SingleGraphOperatorBlock`
8. `GraphReadout`
9. `TargetTemporalTrunk`
10. shared heads

Because node and edge encoders run in parallel, the first part can be a two-lane structure that merges at `HybridEdgeFeatureFusion`.

### Required visual emphasis

- highlight `EdgeRelationFusion` as the decisive architectural bottleneck
- show the three relation channels entering fusion and leaving as one fused edge stream
- show only **one** graph pathway after fusion

### Suggested relation visual

Before fusion:

- `price_dep`
- `order_flow`
- `liquidity`

After fusion:

- one block labeled `fused edge representation`

### Minimal explanatory callouts

- `early relation fusion`
- `single graph pathway`
- `convolutional temporal backbone`

### Caption-ready wording

**Figure 4.2. Detailed architecture of the base_gnn family.**  
The figure shows the forward path of the single-graph baseline. Node and edge histories are encoded separately, combined through hybrid edge-feature fusion, and collapsed into a single fused edge representation before graph message passing. The model therefore tests whether early relation fusion is sufficient under the shared benchmark.

### Production prompt

Create a clean academic architecture diagram for the base_gnn family. Use a left-to-right pipeline. Show two parallel input lanes: X_node going into NodeTemporalEncoder and X_edge going into EdgeTemporalEncoder. Merge them in HybridEdgeFeatureFusion. Before fusion, visually indicate three relation channels: price_dep, order_flow, and liquidity. Then show EdgeRelationFusion collapsing those channels into one fused edge representation. After that, show exactly one SingleGraphOperatorBlock, then GraphReadout, then TargetTemporalTrunk, then the shared multi-task heads block with trade, direction, return, exit type, and time-to-event. Add concise callouts that say early relation fusion, single graph pathway, and convolutional temporal backbone. Make the diagram publication-ready, vector-style, white background, with no code or tensor-shape clutter.

---

## Figure 4.3 — Detailed `multigraph` architecture

### Narrative purpose

This figure should make the contrast with `base_gnn` visually obvious.

### Main claim to communicate

`multigraph` preserves relation-specific semantics through message passing and only fuses them **after** relation-specific graph updates.

### Recommended layout

Single left-to-right flow with a split middle section:

1. parallel node and edge temporal encoders
2. `HybridEdgeFeatureFusion`
3. branch into three relation-specific graph lanes
   - `RelationGraphBlock(price_dep)`
   - `RelationGraphBlock(order_flow)`
   - `RelationGraphBlock(liquidity)`
4. merge through `RelationAttentionFusion`
5. `GraphReadout`
6. `TargetTemporalTrunk`
7. shared heads

### Required visual emphasis

- the three relation lanes must be equally prominent
- fusion must happen **after** the three graph blocks
- the relation-attention block should look like a learned weighted merge, not a simple concatenation

### Minimal explanatory callouts

- `late relation fusion`
- `relation-specific message passing`
- `same temporal backbone as base_gnn`

### Recommended comparison device

Optionally add a small side annotation:

`contrast to base_gnn: fusion occurs after graph processing, not before`

### Caption-ready wording

**Figure 4.3. Detailed architecture of the multigraph family.**  
The figure shows the multi-relation architecture in which price-dependence, order-flow, and liquidity channels are processed by separate graph pathways before learned relation attention combines them. The design tests whether preserving relation semantics during message passing improves the benchmark relative to early fusion.

### Production prompt

Create a publication-ready vector architecture diagram for the multigraph family. Use a left-to-right pipeline with a central three-branch section. Show X_node into NodeTemporalEncoder and X_edge into EdgeTemporalEncoder, then merge them in HybridEdgeFeatureFusion. From there, split into three equally visible relation-specific graph lanes labeled price_dep, order_flow, and liquidity. Each lane should contain its own RelationGraphBlock. After the three lanes, merge them in a block labeled RelationAttentionFusion that visibly suggests learned weighted fusion. Then continue to GraphReadout, TargetTemporalTrunk, and the same shared multi-task heads block: trade, direction, return, exit type, time-to-event. Add short callouts for late relation fusion, relation-specific message passing, and same temporal backbone as base_gnn. Keep the figure academic, minimal, and non-code-like.

---

## Figure 4.4 — Detailed `memorygraph` architecture

### Narrative purpose

This figure should explain why `memorygraph` is qualitatively different from the other two families.

### Main claim to communicate

Temporal modelling is represented through **recurrent node and edge memory**, and graph interaction occurs **inside** the recurrent update loop.

### Recommended layout

Use a loop-based diagram rather than a simple straight pipeline.

Outer structure:

1. `X_node_t` and `X_edge_t`
2. step projectors
3. `HybridEdgeFeatureFusion`
4. recurrent core loop
5. `GraphReadout`
6. `Output Projection`
7. shared heads

Core loop structure:

- previous `edge_memory`
- `EdgeMemoryUpdater`
- `AdaptiveGraphConnectivity`
- graph operator (`Conv or MPNN`)
- relation-aware node state
- previous `node_memory`
- `NodeMemoryUpdater`
- updated `node_memory` and `edge_memory`
- loop arrow to next time step / next chunk step

### Required visual emphasis

- edge-memory update must happen before node-memory update
- the graph operator must appear **inside** the recurrent loop
- clearly distinguish:
  - current input state
  - previous memory state
  - updated memory state

### Minimal explanatory callouts

- `recurrent node-edge memory`
- `graph interaction inside memory loop`
- `state carried across contiguous chunks`

### Optional micro-annotation

Near the loop arrow:

`truncated BPTT during training`

Only include this if it does not clutter the figure.

### Caption-ready wording

**Figure 4.4. Detailed architecture of the memorygraph family.**  
The figure shows the recurrent node-edge memory mechanism used by memorygraph. At each step, projected node and edge inputs are combined with previous memory states, edge memory is updated first, graph interaction is applied to state-enriched representations, and node memory is then updated from relation-aware node and edge context. The design therefore represents temporal information through recurrent state rather than only through convolutional windows.

### Production prompt

Create a high-quality academic vector diagram for the memorygraph family using a loop-based architecture rather than a simple straight pipeline. Start with X_node_t and X_edge_t entering NodeStepProjector and EdgeStepProjector, then HybridEdgeFeatureFusion. Then show a recurrent core with previous edge_memory feeding into EdgeMemoryUpdater, followed by AdaptiveGraphConnectivity and a graph operator labeled Conv or MPNN, followed by NodeMemoryUpdater that also receives previous node_memory. The diagram must make clear that edge memory is updated before node memory and that graph interaction happens inside the recurrent loop. After the loop, show fused node sequence going into GraphReadout, then Output Projection, then the shared multi-task heads block with trade, direction, return, exit type, and time-to-event. Use dark magenta to emphasize memory state, and add concise callouts: recurrent node-edge memory, graph interaction inside memory loop, and state carried across contiguous chunks. Keep it publication-ready, clean, and free of code-level detail.

---

## Implementation checklist for the figure author

Before finalizing the figures, verify:

1. Figure 4.1 visibly communicates shared inputs and shared heads across all families.
2. Figure 4.2 clearly shows fusion **before** graph message passing.
3. Figure 4.3 clearly shows fusion **after** relation-specific graph message passing.
4. Figure 4.4 clearly shows the recurrent loop and the ordering:
   - edge memory update
   - graph interaction
   - node memory update
5. All figures use the same color semantics.
6. All labels are prose-friendly and match the Chapter 4 terminology exactly.
7. No figure includes implementation noise that the chapter deliberately avoids.
8. Figure numbering in the manuscript is updated consistently if Figures 4.2–4.4 are all introduced.

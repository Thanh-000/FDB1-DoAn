# Next Architecture Proposal

This note proposes the next full-system research direction after the project was rolled back to the compact `baseline_tree` production path.

The proposal does not add more weak model branches.
Instead, it keeps the accepted tree backbone and changes the layers around it that are most likely limiting final performance.

## Why a new direction is needed

The ablation results established three things:

- the compact tree ensemble is still the strongest accepted production path
- graph-derived production logic did not beat the baseline
- recent-window and time-decay production logic did not beat the baseline

This means the next improvement should not start from architectural novelty alone.
It should start from:

- model selection under temporal drift
- feature stability under time shift
- safer decision making under uncertainty

## Proposed system

The next architecture should be:

- stable fold-local preprocessing
- stable tabular feature backbone
- compact boosted-tree ensemble
- temporal model-selection layer
- selective-decision layer
- existing XAI and HITL layer

In short:

`Feature Factory -> Tree Ensemble -> Temporal Selector -> Selective Decision Layer -> XAI/HITL`

## Layer 1: stable feature factory

Keep the accepted tabular pipeline, but tighten feature quality:

- remove unstable or low-value fold-local features
- rank features by average contribution across folds
- penalize features with high fold-to-fold volatility
- favor compact, leakage-safe, temporally stable features

This is the lowest-risk source of gain because it improves the strongest existing system directly.

## Layer 2: compact tree ensemble

Keep the current core:

- `XGBoost`
- `LightGBM`
- `CatBoost`
- `XGB` meta-learner

Do not add new model families by default.
The project already showed that more branches usually increased complexity faster than they increased system-level value.

## Layer 3: temporal model selection

This is the main research change.

Instead of permanently changing the architecture with graph or recent branches, compare candidate configurations using a temporal assessment rule that favors what is best for the current time regime.

The core idea is:

- keep a small set of strong candidate baselines
- assess them on rolling or recent temporal windows
- choose the active production configuration using time-aware validation

Candidate configurations can include:

- `baseline_tree`
- `baseline_tree` with a tighter feature subset
- `baseline_tree` with a different threshold policy
- `baseline_tree` with a lighter or stronger meta learner

This keeps the system compact while still adapting to drift.

Reference:

- Elise Han, Chengpiao Huang, Kaizheng Wang. `Model Assessment and Selection under Temporal Distribution Shift`. ICML 2024.
  https://proceedings.mlr.press/v235/han24b.html

## Layer 4: selective decision layer

The project already has HITL outputs.
The next step is to turn uncertainty into a first-class decision mechanism.

The system should not only predict fraud score.
It should also decide whether the score is reliable enough for automatic action.

That leads to a three-way output:

- `ALLOW`
- `BLOCK`
- `REVIEW`

The logic should be built from:

- calibrated probability
- confidence or uncertainty proxy
- review-budget policy

This is more aligned with real fraud operations than maximizing a single thresholded metric alone.

Reference:

- Adam Fisch, Tommi Jaakkola, Regina Barzilay. `Calibrated Selective Classification`.
  https://arxiv.org/abs/2208.12084

## Layer 5: knowledge-enriched feature semantics

If a new modeling idea is introduced, it should enter through the feature layer first, not as a new heavyweight branch.

The recommended direction is to encode domain knowledge such as:

- merchant or account behavior concepts
- deterministic risk rules
- feature groups with known operational meaning
- column descriptions and concept groupings

This can strengthen the tabular backbone without forcing a new end-to-end architecture.

Reference:

- Juyong Kim, Chandler Squires, Pradeep Ravikumar. `Knowledge-Enriched Machine Learning for Tabular Data`.
  https://proceedings.mlr.press/v288/kim25a.html

## Research priorities

### Priority 1

Feature-stability pruning on the accepted baseline.

### Priority 2

Temporal model selection among a small number of strong baseline variants.

### Priority 3

Selective classification and review gating on top of the accepted baseline.

### Priority 4

Knowledge-enriched feature construction and grouping.

### Priority 5

Only after the above, revisit any new model family.

## What not to do next

Do not immediately return to:

- graph branches
- deep fusion branches
- larger ensembles
- architecture-heavy additions without a controlled baseline win

The project already spent enough evidence budget to know those are not the highest-ROI next step.

## Practical implementation path

The next implementation cycle should be:

1. export fold-wise feature importances from the accepted baseline
2. build a feature-stability score and prune weak unstable columns
3. benchmark a small set of baseline variants under rolling temporal selection
4. add a selective-decision layer with `ALLOW / BLOCK / REVIEW`
5. compare final operating metrics, not only AUPRC

## Expected benefit

If this direction works, the gain should come from:

- better robustness to temporal shift
- less noisy feature space
- fewer unsafe automatic decisions
- higher effective operational value, even when raw model family remains unchanged

That is the most defensible route to a better fraud system given the evidence collected in this repository.

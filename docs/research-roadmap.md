# Research Roadmap

This document defines the next research phase after rolling the IEEE-CIS notebook back to the last stable tabular backbone.

## Stable baseline

The current baseline to preserve is:

- fold-local preprocessing
- fold-local aggregate and graph-derived features
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID smoothing
- temporal evaluation on Colab

This baseline is the control system for all future ablations.

## Research goal

The next phase is not incremental notebook tuning. The goal is to identify one architecture direction with a realistic chance of improving temporal fraud detection under leakage-safe evaluation.

The project should answer:

1. Which additional view gives the strongest gain beyond the stable tabular backbone?
2. Does pre-training help when labels are sparse and temporal drift is strong?
3. Which direction is strong enough to justify integration into the main project?

## Priority order

### 1. TCN branch

Why first:

- lowest implementation risk
- faster than LSTM
- compatible with Colab
- easiest sequential baseline to benchmark cleanly

Research question:

- Does a short-window temporal convolution branch add useful sequential signal beyond the current tabular backbone?

Success criteria:

- improves `AUPRC` or `F1` under temporal split
- does not increase runtime excessively
- produces stable gains across folds

### 2. SCARF pre-training

Why second:

- strongest research value among label-efficient methods
- can be framed as semi-supervised fraud detection
- can be applied before a downstream tabular or sequential branch

Research question:

- Does contrastive pre-training on unlabeled transactions improve downstream fraud detection under temporal drift?

Success criteria:

- clear improvement over the same downstream model without pre-training
- gains remain after leakage-safe evaluation

### 3. SAINT

Why third:

- potentially strong for tabular fraud patterns
- sample-level attention may capture fraud cluster similarity

Main risk:

- memory cost
- harder Colab scaling

Research question:

- Can inter-sample attention capture fraud similarity patterns missed by tree ensembles?

Success criteria:

- competitive with the stable tabular baseline on temporal folds
- memory/runtime still manageable on Colab

### 4. GTAN

Why last:

- highest novelty
- closest to a real graph-temporal architecture for the problem

Main risk:

- highest implementation complexity
- scalability uncertainty on large data
- easiest direction to lose time on

Research question:

- Does a unified graph-temporal attention model outperform graph-derived features plus tabular boosting under realistic fraud evaluation?

Success criteria:

- standalone graph-temporal branch beats weak graph baselines
- later fusion improves the main system

## Recommended phase plan

### Phase 1. Re-establish clean control baseline

- keep the current rolled-back IEEE notebook unchanged
- rerun short screening to confirm the stable baseline still behaves as expected
- document final baseline metrics for future comparisons

Deliverable:

- one locked baseline configuration
- one baseline result table

### Phase 2. Low-risk extension

- build a `TCN` sequential branch first
- evaluate standalone
- then test late fusion against the baseline

Deliverable:

- ablation: baseline vs baseline + TCN

### Phase 3. Research contribution branch

- test `SCARF` pre-training on the downstream branch selected in Phase 2
- if resources remain, benchmark `SAINT`
- only after those two, consider a `GTAN` branch

Deliverable:

- ablation: no pre-training vs SCARF pre-training
- decision memo on whether `SAINT` or `GTAN` is worth full implementation

## Experimental rules

- run all major experiments on Google Colab
- keep temporal split and fold-local preprocessing unchanged
- compare every new branch against the same stable baseline
- do not integrate a branch into the main notebook unless it wins as a standalone or fusion experiment
- stop weak directions early

## Stop conditions

A direction should be dropped if:

- it fails to beat the stable baseline in screening
- it only improves train metrics but not temporal validation
- it adds too much runtime for negligible gain
- it requires large architectural complexity without measurable benefit

## Immediate next step

The next implementation target should be:

- `TCN` branch on Colab

This is the safest next research step and the most likely to produce a usable sequential improvement without destabilizing the project.

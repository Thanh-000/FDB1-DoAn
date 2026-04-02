# SCARF Fusion Design

This document defines how `SCARF` should be tested for controlled integration into the current stable IEEE-CIS backbone.

## Goal

Use `SCARF` as a representation-learning branch without disturbing the current stable tabular architecture.

The current stable backbone remains:

- fold-local preprocessing
- temporal and aggregate tabular features
- graph-derived motif features
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID smoothing

`SCARF` should be treated as an additional branch, not a replacement.

## What the current evidence says

From the standalone benchmarks:

- `SCARF + fine-tune` beats `no-pretrain`
- `SCARF + fine-tune` beats `SCARF + freeze`
- `SCARF` is stronger than the previously tested `TCN` and `GNN` branches

But:

- standalone `SCARF` still does not beat the main tabular backbone

That means the next test should be controlled late fusion rather than full architectural replacement.

## Fusion options

There are two candidate fusion paths.

### Option A: SCARF score fusion

Use the final fraud probability from `SCARF` as a fourth learner score.

Current meta input:

- `XGB`
- `LGB`
- `CatBoost`

New meta input:

- `XGB`
- `LGB`
- `CatBoost`
- `SCARF`

Advantages:

- simplest integration
- easiest to ablate
- lowest risk to the current notebook

This should be the first integration path.

### Option B: SCARF embedding fusion

Use the final hidden representation from the `SCARF` encoder and append it to the tabular feature matrix before training one or more downstream models.

Advantages:

- richer signal than a single score

Risks:

- much easier to overfit
- harder to keep the ablation readable
- more complex fold-local pipeline

This should only be attempted if score fusion shows positive signal.

## Recommended integration order

### Stage 1: score-only fusion

For each outer fold:

1. train the stable tabular backbone exactly as before
2. train `SCARF` on the same train fold
3. generate `SCARF` probabilities for:
   - inner validation slices for OOF training
   - outer validation fold
4. add `SCARF` score as one extra column in the meta matrix
5. compare:
   - baseline `Nx3`
   - fused `Nx4`

This is the first real decision point.

### Stage 2: score fusion with ablation

If Stage 1 helps, compare:

1. baseline backbone
2. baseline + `SCARF no-pretrain score`
3. baseline + `SCARF pretrain score`

This isolates whether the pretraining itself adds value after fusion.

### Stage 3: embedding fusion

Only if score fusion is positive and stable.

## Evaluation protocol

The fusion test must follow the same rules as the main notebook:

- temporal split only
- fold-local preprocessing only
- no leakage from validation into any preprocessing or pretraining fit
- compare on the same fold configuration

Primary metric:

- `AUPRC`

Secondary metrics:

- `ROC-AUC`
- `F1`
- `Precision`
- `Recall`

`SCARF` should only stay in the architecture if it improves `AUPRC` under temporal validation.

## Integration decision rule

Integrate `SCARF` into the main architecture only if:

- fused `AUPRC` is higher than baseline
- the gain is repeatable across folds
- the additional runtime is acceptable

Do not integrate if:

- only one fold improves
- `ROC-AUC` rises but `AUPRC` does not
- the gain is too small relative to added complexity

## Minimal implementation plan

1. extend the `SCARF` runner so it can return fold probabilities rather than just standalone metrics
2. create a small fusion harness notebook or script
3. inject `SCARF` score into the meta matrix as learner four
4. benchmark on short screening setup first
5. only then try full temporal confirmation

## Recommended next coding step

Implement `SCARF score fusion` only.

Do not start with embedding fusion.

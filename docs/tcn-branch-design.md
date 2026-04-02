# TCN Branch Design

This document defines the first sequential extension to the current stable IEEE-CIS baseline.

## Purpose

The `TCN` branch is intended to add a true sequential view on top of the current tabular backbone without reopening the complexity of `LSTM` or `GNN` research branches.

The branch must satisfy three constraints:

- leakage-safe under temporal validation
- cheap enough to run on Google Colab
- easy to fuse with the existing `XGB + LGB + CatBoost + XGB meta` pipeline

## Baseline context

The current baseline remains unchanged:

- fold-local preprocessing
- tabular and graph-derived fold-local features
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID smoothing

The `TCN` branch is an extension, not a replacement.

## Research question

Can a short-window temporal convolution branch capture transaction-order patterns that are not expressed by the current tabular backbone under strict temporal evaluation?

## Sequence definition

### Sequence entity

The default sequence key should be:

- `AccountID`

Fallback sequence keys for ablation:

- `card1`
- `device_fp_raw`

The first implementation should use only `AccountID`.

### Sequence target

For each transaction at time `t`, the branch predicts fraud for that transaction using only its own past sequence context.

### Window length

Default:

- `T = 10`

Reason:

- already aligned with prior project direction
- short enough for Colab
- long enough for recent fraud burst patterns

Possible ablations:

- `T = 5`
- `T = 20`

## Sequence contents

Each sequence element should contain only features that are safe to compute from the transaction row itself or from past-only transforms.

### Default per-step features

Start with a compact numeric feature set:

- `TransactionAmt`
- `LogAmt`
- `Hour`
- `DayOfWeek`
- `Amt_cents`
- `C1_div_C14`
- `C1`
- `C2`
- `C3`
- `C4`
- `C5`
- `C11`
- `C14`
- `D1`
- `D2`
- `D6`
- `D7`
- `D8`
- `D13`
- `D14`

Optional later extension:

- selected fold-local target-encoded or PCA features

The first implementation should avoid large feature tensors.

## Sequence construction rules

For each target transaction:

1. sort all transactions within the same `AccountID` by `TransactionDT`
2. collect up to `T-1` previous transactions
3. append the current transaction as the last step
4. left-pad with zeros if history is shorter than `T`
5. generate a binary mask where padded rows are `0`

This means the model always sees a fixed tensor:

- shape: `[batch, T, F]`

where:

- `T = 10`
- `F = number of per-step features`

## Model design

### Core block

Use a lightweight `Temporal Convolutional Network` with residual dilated 1D convolutions.

Recommended first version:

- input projection: `Linear(F -> 64)`
- TCN blocks: `3`
- hidden channels: `64`
- kernel size: `3`
- dilations: `1, 2, 4`
- dropout: `0.10`

### Output head

Use the final time step representation for classification:

- sequence encoder output -> last valid step
- `Linear(64 -> 32)`
- `ReLU`
- `Dropout(0.10)`
- `Linear(32 -> 1)`

Loss:

- `BCEWithLogitsLoss(pos_weight=...)`

Optional later ablation:

- focal loss

## Masking and padding

Because the sequence is left-padded, the branch must not treat padded timesteps as real events.

Implementation rule:

- either explicitly mask padded positions after each block
- or gather the final valid step representation using sequence lengths

The first implementation should gather the last valid step using lengths. This is simpler and less error-prone.

## Training protocol

### Environment

- Google Colab only
- GPU required

### Evaluation

Use the same temporal split policy as the main IEEE notebook:

- screening: short temporal split on Colab
- confirmation: strict walk-forward

### Data leakage rule

The sequence branch must obey the same causal rule as the tabular baseline:

- only past transactions may appear before the current transaction in the sequence
- no future transactions may leak into the sequence window
- any normalization object used for sequence features must be fit inside the fold

## Integration strategy

### Phase A: standalone branch

Train the `TCN` branch by itself and report:

- `AUPRC`
- `ROC-AUC`
- `F1`
- `Precision`
- `Recall`

Goal:

- determine whether the sequential branch has signal at all

### Phase B: late fusion

If standalone results are acceptable, integrate the branch as a new base learner score.

Current meta input:

- `XGB`
- `LGB`
- `CatBoost`

Extended meta input:

- `XGB`
- `LGB`
- `CatBoost`
- `TCN`

This keeps the rest of the notebook architecture stable.

### Phase C: embedding fusion

Only if score-level fusion works:

- expose a small `TCN` embedding
- concatenate that embedding with OOF score features for a stronger meta model

This is not part of the first implementation.

## Ablation plan

Minimum ablations:

1. baseline tabular only
2. standalone `TCN`
3. baseline + `TCN score`

Recommended ablations:

4. `T = 5` vs `T = 10`
5. `AccountID` sequence vs `card1` sequence
6. BCE vs focal loss

## Success criteria

The branch is worth keeping only if at least one of the following happens under temporal validation:

- `AUPRC` improves over baseline fusion
- `F1` improves without unacceptable precision collapse
- `Recall` improves materially at a usable precision level

The branch should be dropped if:

- standalone performance is weak
- fusion adds runtime but no measurable gain
- gains appear only on screening and disappear on strict temporal confirmation

## Implementation recommendation

The branch should be implemented outside the main notebook first, in a separate Colab-friendly script or notebook.

Suggested future files:

- `sequence_tcn.py`
- `sequence_tcn_runner.py`
- optional Colab notebook for sequence benchmarking

Only after the branch shows signal should it be merged into the main IEEE notebook.

## Immediate next implementation target

Build a standalone `TCN` benchmark on `AccountID` sequences with:

- `T = 10`
- compact 20-feature per-step input
- weighted BCE loss
- score-only output for later fusion

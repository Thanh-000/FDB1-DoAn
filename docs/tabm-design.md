# TabM Design Notes

This document defines the next deep-tabular direction to test after `SCARF` and `SAINT`.

## Why TabM

`TabM` is attractive because it targets the exact setting where this project has struggled:

- large tabular datasets
- strong tree baselines
- the need for a practical deep model rather than a fragile graph or sequence branch

Compared with earlier neural branches:

- it is more appropriate for tabular data than `TCN`
- it is less memory-sensitive than row-attention-heavy transformers
- it is easier to benchmark cleanly than retrieval- or graph-based alternatives

## Project goal

Test whether a `TabM`-style deep tabular model can become the strongest neural standalone branch under the same temporal split protocol used throughout the project.

## First benchmark scope

The first implementation should stay narrow and readable:

- standalone branch only
- same IEEE-CIS train split protocol as `SCARF` and `SAINT`
- compact feature set first
- no pretraining in the first pass

This benchmark is only meant to answer:

`Can a TabM-style model beat SAINT and get close to or beyond SCARF standalone performance?`

## Candidate feature set

Use the same compact numerical set already used for `SCARF` and `SAINT`:

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

The first pass should not use the full wide notebook feature table.

## Model shape

The benchmark should use a `TabM`-style design with:

- per-feature numerical projection
- shared MLP backbone
- lightweight ensemble heads
- averaged logits at inference time

This keeps the implementation faithful to the practical spirit of TabM:

- deep tabular
- ensemble-like robustness
- lower complexity than a full transformer

## Training protocol

- temporal split only
- one fold first for screening
- BCE with positive class weighting
- GPU on Colab

Primary metric:

- `AUPRC`

Secondary metrics:

- `ROC-AUC`
- `F1`
- `Precision`
- `Recall`

## Success criteria

The `TabM` branch is worth continuing only if:

- it clearly beats `TCN`
- it clearly beats or matches `SAINT`
- it gets close to or exceeds `SCARF` standalone

If it remains weak under the same temporal split, it should not be integrated further.

## Next ablations if the first pass is positive

1. wider feature set
2. deeper backbone
3. more ensemble heads
4. `SCARF` pretraining before TabM

## Integration rule

Even if standalone `TabM` is strong, do not inject it into the main IEEE notebook immediately.

It should first pass:

1. standalone screening
2. standalone repeatability on another fold
3. controlled fusion study

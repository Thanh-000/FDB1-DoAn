# MLP-PLR Fusion Design Notes

This document defines the controlled `MLP-PLR -> score fusion` experiment.

## Why fusion is now justified

`MLP-PLR` is the strongest standalone deep tabular branch tested so far under the temporal benchmark.

The best observed configuration is:

- `USE_TIME_FEATURE = False`
- `N_FREQUENCIES = 8`

Because of that, `MLP-PLR` is the first deep branch that is worth testing in controlled score fusion
against the stable production backbone.

## Fusion rule

The production backbone remains:

- `XGBoost`
- `LightGBM`
- `CatBoost`
- `XGB` meta-learner

The fusion experiment adds exactly one new score:

- `MLP-PLR probability`

This changes the meta input from:

- baseline `Nx3`

to:

- fused `Nx4`

## Evaluation rule

The experiment is valid only if:

1. temporal split is preserved
2. the tabular baseline uses the same fold data
3. only one variable changes: adding the `MLP-PLR` score

The primary metric remains `AUPRC`.

## Decision rule

Keep `MLP-PLR` out of the main IEEE notebook unless fusion improves:

- `AUPRC`

under the same temporal fold.

If `ROC-AUC` improves but `AUPRC` does not, integration is rejected.

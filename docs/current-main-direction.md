# Current Main Direction

This note records the current accepted direction of the project after the recent deep-branch and temporal-drift experiments.

## Current accepted main path

The project now treats the IEEE-CIS notebook as the primary production-path pipeline.

The current active system is:

- fold-local leakage-safe preprocessing
- temporal and aggregate tabular features
- graph-derived fold-local motif features
- long-history branch:
  - `XGBoost`
  - `LightGBM`
  - `CatBoost`
- recent-window branch:
  - `XGB_recent`
  - `LGB_recent`
  - `CatBoost_recent`
- time-decay sample weighting
- `XGB` meta-learner
- UID smoothing
- threshold selection via policy (`best_f1`, `best_f2`, or `recall_at_precision`)

## Why this became the main direction

Recent experiments showed a consistent pattern:

- several deep branches had real standalone signal
- but their integration into the main system did not improve the final system metric
- the more defensible improvement path was not "add a new architecture", but "improve temporal adaptation"

This was tested by adding:

- a recent-window branch trained only on the newest part of the training fold
- time-decay sample weights so newer examples matter more than older ones

Unlike the earlier deep-branch integrations, the recent branch was not ignored by the meta learner.

## Screening evidence

Under the `3x2` screening setup with:

- `ENABLE_TIME_DECAY = True`
- `TIME_DECAY_ALPHA = 2.0`
- `ENABLE_RECENT_BRANCH = True`
- `RECENT_TRAIN_FRAC = 0.35`

the following outer-fold results were obtained when each fold was run separately on Colab:

- Fold 1:
  - `AUPRC = 0.4995`
  - `ROC-AUC = 0.8493`
  - `F1 = 0.5123`
  - `Precision = 0.6312`
  - `Recall = 0.4311`
- Fold 2:
  - `AUPRC = 0.5263`
  - `ROC-AUC = 0.8870`
  - `F1 = 0.5186`
  - `Precision = 0.6487`
  - `Recall = 0.4320`
- Fold 3:
  - `AUPRC = 0.5342`
  - `ROC-AUC = 0.9082`
  - `F1 = 0.5293`
  - `Precision = 0.6703`
  - `Recall = 0.4373`

Approximate screening mean:

- `Mean AUPRC = 0.5200`
- `Mean ROC-AUC = 0.8815`
- `Mean F1 = 0.5201`
- `Mean Precision = 0.6501`
- `Mean Recall = 0.4334`

These gains are not dramatic, but they are the strongest system-level evidence seen so far on the main path.

## What was not accepted into the main path

The following were tested but not accepted into the active backbone:

- `GNN`
- `TCN`
- `SAINT`
- `SCARF score fusion`
- `TabM` integration
- `RealMLP`
- `MLP-PLR score fusion`

Some of these remain useful research baselines, but not accepted production-path components.

## Colab execution rule

For Colab free users, the recommended notebook workflow is:

- run exactly one outer fold per session
- use `RUN_OUTER_FOLD_ONLY = 0`, `1`, or `2`
- restart the runtime between folds

This does not change the fold result.
It only prevents RAM accumulation across multiple outer folds in the same Colab session.

## Practical conclusion

The project should continue from this point by:

1. keeping `recent-window + time-decay` as the active main-direction experiment
2. using per-fold Colab runs for stable execution
3. treating deep standalone branches as secondary research tracks unless they improve the full system metric after integration

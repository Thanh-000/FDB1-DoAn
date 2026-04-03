# Current Main Direction

This note records the current accepted direction of the project after the ablation study on the IEEE-CIS production notebook.

## Accepted production-path default

The accepted default system is now the `baseline_tree` preset in the IEEE-CIS notebook.

The default architecture is:

- fold-local leakage-safe preprocessing
- stable tabular feature backbone
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `XGB` meta-learner
- UID smoothing
- threshold selection by policy

## What was tested and rejected as default

The following production-path additions were tested against the same fold protocol and did not beat `baseline_tree`:

- `graph_only`
- `decay_only`
- `recent_only`
- `full_recent_decay`

In other words:

- graph-derived fold-local features did not improve the default system enough
- time-decay weighting did not beat the simpler baseline
- recent-window branch did not beat the simpler baseline
- combining graph + time-decay + recent did not beat the simpler baseline

## Why the default was rolled back

The project first explored a more complex temporal-adaptation path because several standalone research branches showed signal.

However, the controlled notebook ablations showed a simpler conclusion:

- the strongest production-path result still comes from the compact tree-based backbone
- additional complexity increased runtime and memory cost
- the extra components did not improve the main metric consistently

Therefore the default system was rolled back to the simpler and stronger baseline.

## Colab execution rule

For Colab free, the main notebook should still be run one outer fold at a time:

- `RUN_OUTER_FOLD_ONLY = 0`
- `RUN_OUTER_FOLD_ONLY = 1`
- `RUN_OUTER_FOLD_ONLY = 2`

This is an execution strategy only.
It does not change the fold result itself.

## Research tracks that remain open

Rolling back the default does not mean research is over.
It means the accepted production-path baseline is now clear, and further work must beat it explicitly.

The most promising remaining directions are:

- stronger feature-stability control
- better temporal validation and threshold policy study
- stronger tabular feature engineering with controlled ablation
- deep branches only when they improve the final ensemble, not just standalone benchmarks

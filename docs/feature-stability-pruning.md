# Feature Stability Pruning

This note defines the first concrete experiment in the next research phase.

The goal is to improve the accepted `baseline_tree` system by removing unstable features rather than adding new model branches.

## Why this experiment comes first

The current evidence suggests:

- the compact tree baseline is the strongest accepted system
- graph, recent, and decay additions did not beat it
- the most likely remaining source of avoidable error is unstable feature contribution under temporal shift

That makes feature stability the highest-ROI next target.

## Data source

The main notebook now supports exporting fold-wise base-model feature importances.

Relevant config:

- `EXPORT_FOLD_IMPORTANCES = True`
- `IMPORTANCE_EXPORT_DIR = './artifacts/feature_importance'`

Each outer fold writes:

- `fold_<k>_feature_importances.csv`

with columns:

- `fold`
- `model`
- `feature`
- `importance`

## Aggregation script

Use:

```powershell
python scripts/feature_stability_pruning.py "artifacts/feature_importance/*.csv"
```

Outputs:

- `artifacts/feature_stability/feature_stability_by_model.csv`
- `artifacts/feature_stability/feature_stability_overall.csv`

## Core metrics

The script computes:

- `mean_importance`
- `std_importance`
- `fold_coverage`
- `cv_importance`
- `stability_score`

Interpretation:

- high `mean_importance` is good
- high `fold_coverage` is good
- high `std_importance` is bad
- high `cv_importance` is bad
- high `stability_score` is good

## Pruning rule

The recommended first pruning pass is conservative:

- keep features with strong `stability_score`
- remove features with:
  - low mean importance
  - low fold coverage
  - high volatility across folds

Do not prune aggressively in the first pass.
The purpose is to remove obviously unstable columns, not to redesign the whole feature space at once.

## Experimental protocol

1. run the accepted baseline on outer folds with `EXPORT_FOLD_IMPORTANCES = True`
2. aggregate the exported CSV files with the script
3. inspect the top stable and unstable features
4. create a pruned feature list
5. rerun the baseline with only the retained feature set
6. compare against the current accepted baseline under the same temporal fold protocol

## Success condition

The pruning pass is worth keeping only if it improves at least one of:

- `AUPRC`
- `F1`
- operational precision/recall tradeoff

without causing a harmful drop in the others.

## Expected benefit

If this works, the gain should come from:

- less noisy temporal generalization
- lower runtime
- lower Colab memory pressure
- stronger trust in the accepted feature space

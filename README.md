# Digital Financial Transaction Fraud Detection

This repository now keeps only the accepted production-path code and the supporting research notes needed to justify it.

## Main contents

- [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb): main IEEE-CIS production notebook
- [notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb](./notebooks/MVS_XAI_Colab_DataPrep_Phase1.ipynb): preprocessing and preparation notebook
- [scripts/MVS_XAI_Dashboard.py](./scripts/MVS_XAI_Dashboard.py): dashboard prototype
- [docs/current-main-direction.md](./docs/current-main-direction.md): accepted system direction after ablation
- [docs/highest-result-research.md](./docs/highest-result-research.md): shortlist of next high-ROI improvements
- [docs/next-architecture-proposal.md](./docs/next-architecture-proposal.md): next full-system architecture proposal
- [docs/negative-results.md](./docs/negative-results.md): excluded branches and rationale

## Accepted architecture

The current main system is a compact tabular ensemble:

- fold-local leakage-safe preprocessing
- stable tabular feature backbone
- `XGBoost`
- `LightGBM`
- `CatBoost`
- `XGB` meta-learner
- UID smoothing
- threshold policy selection

Graph-derived production logic, temporal recent/decay branches, and deep-model fusion paths were removed from the codebase after controlled ablations failed to beat the baseline tree ensemble.

## Run on Colab

1. Open [notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb](./notebooks/MVS_XAI_Colab_IEEE_CIS.ipynb) in Google Colab.
2. Set `Runtime -> Change runtime type -> GPU`.
3. Mount or copy access to the IEEE-CIS dataset files:
   - `ieee-fraud-detection/train_transaction.csv`
   - `ieee-fraud-detection/train_identity.csv`
   - `ieee-fraud-detection/test_transaction.csv`
   - `ieee-fraud-detection/test_identity.csv`
4. Run cells from top to bottom.
5. On Colab free, prefer one outer fold per session:
   - `RUN_OUTER_FOLD_ONLY = 0`
   - `RUN_OUTER_FOLD_ONLY = 1`
   - `RUN_OUTER_FOLD_ONLY = 2`
   - and in full confirmation, continue with `3` and `4`

## Notes

- Raw datasets are excluded from Git because of size.
- Colab remains the preferred execution environment.
- Further improvement work should beat the accepted baseline explicitly before it is added back into the main notebook.

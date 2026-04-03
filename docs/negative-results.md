# Negative Results and Exclusion Rationale

This document records experimental branches that were tested and explicitly excluded from the current main architecture.

The purpose is to preserve technical reasoning and prevent weak branches from being reintroduced without new evidence.

## Current main architecture

The active system remains:

- fold-local preprocessing
- stable tabular backbone
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID smoothing
- temporal evaluation on Colab

This architecture remains the project baseline because it outperformed the experimental branches below under controlled ablation.

## 1. Heterogeneous GNN branch

### What was tested

A standalone graph branch was prototyped using a heterogeneous transaction-entity graph with node types such as:

- `txn`
- `acct`
- `card1`
- `addr1`
- `device`
- `pemail`
- `product`

The branch used a graph neural model and produced a fraud score for transaction nodes under temporal fold evaluation.

### Observed result

The branch produced extremely weak validation performance:

- very low `val_auprc`
- `val_roc_auc` close to random

The measured result was not competitive with the tabular baseline.

### Why it was excluded

The branch was excluded for the following reasons:

- standalone signal was too weak to justify integration
- graph branch complexity was high relative to measurable gain
- the implementation was not strong enough to outperform the simpler tabular backbone under leakage-safe temporal validation

### Interpretation

The result does not prove graph learning is useless for fraud detection.
It only shows that the tested graph design was not mature enough for this project stage.

Graph learning may be revisited later only if:

- a stronger architecture such as `GTAN` is implemented
- temporal graph design is much more explicit
- the branch wins standalone before fusion

## 2. TCN branch

### What was tested

A standalone `TCN` sequence branch was implemented with:

- `AccountID` as the default sequence key
- `T = 10`
- `20` per-step features
- temporal validation on Colab

### Observed result

The branch executed correctly and used GPU, but validation quality was weak:

- low `val_auprc`
- `val_roc_auc` only slightly above random
- poor `F1` at the default threshold

The branch was therefore not competitive with the tabular baseline.

### Why it was excluded

The branch was excluded from the main architecture because:

- standalone validation signal was insufficient
- it did not justify late fusion
- adding it to the main notebook would increase complexity without evidence of gain

### Interpretation

This negative result is still valuable:

- it shows that a simple sequential deep branch is not enough by itself
- it supports the view that stronger representation learning or pre-training is needed before adding new deep views

## 3. SAINT branch

### What was tested

A standalone `SAINT`-style tabular transformer branch was benchmarked with:

- a compact `20`-feature numerical subset
- temporal fold evaluation on Colab
- supervised-only training

### Observed result

The branch showed real signal and was stronger than the earlier `TCN` and `GNN` branches.

However:

- it did not clearly exceed the best standalone `SCARF` result
- it did not justify integration into the main tabular backbone

### Why it was not integrated

It was not integrated into the main architecture because:

- the gain over other neural standalone branches was not decisive
- it still remained below the stable tabular backbone in practical value
- the project needed a stronger reason than "non-trivial standalone signal" before increasing main-pipeline complexity

### Interpretation

`SAINT` remains a valid research baseline, but not an accepted production-path branch for this project state.

## 4. SCARF branch

### What was tested

A standalone `SCARF` branch was evaluated under:

- compact numerical feature subset
- contrastive pre-training
- downstream fine-tuning
- no-pretrain ablation
- frozen-encoder ablation

### Observed result

This branch produced the first clearly positive neural result:

- `SCARF pretrain + fine-tune` beat `no-pretrain`
- `SCARF pretrain + fine-tune` beat `SCARF + freeze`

So the representation-learning idea itself was validated.

### Why it was not integrated

The branch was still excluded from the main architecture because:

- standalone performance remained below the main tabular backbone
- `SCARF score fusion` into the main meta-learner did not improve the main metric `AUPRC`
- the tested fusion only increased complexity without measurable system gain

### Interpretation

`SCARF` is not a negative result in the same sense as `TCN` or `GNN`.
It is a partial positive result:

- useful as a research finding
- not yet useful as an integrated production branch

## 5. TabM branch

### What was tested

A standalone `TabM`-style deep tabular branch was benchmarked, then integrated as an optional fourth learner into the IEEE-CIS stacking notebook.

### Observed result

The standalone benchmark was the strongest deep-tabular result among the tested neural branches.

But when integrated into the main backbone:

- the meta-learner assigned `TabM` very low importance
- the integrated system did not show meaningful improvement over the baseline

### Why it was not integrated

`TabM` was removed from the main architecture because:

- integration into the stack did not materially improve the main system
- the added learner mostly duplicated existing tabular signal instead of contributing a distinct gain
- the stable tree-based backbone remained stronger in practice

### Interpretation

`TabM` is a valuable standalone benchmark and the strongest deep-tabular candidate tested so far.
However, for this project stage it is still a research branch, not part of the accepted main architecture.

## 6. MLP-PLR branch

### What was tested

A standalone `MLP-PLR` branch was benchmarked with:

- periodic-linear numerical embeddings
- a compact `20`-feature numerical subset
- temporal fold evaluation on Colab
- ablations on `TimeDays` and embedding frequency count

The strongest standalone configuration was:

- `USE_TIME_FEATURE = False`
- `N_FREQUENCIES = 8`

### Observed result

`MLP-PLR` became the strongest standalone deep branch tested in the project.

However, controlled score fusion into the main stacking system was negative:

- baseline meta (`Nx3`) remained stronger
- adding `MLP-PLR` score (`Nx4`) reduced `AUPRC`
- it also reduced `ROC-AUC` in the tested fusion benchmark

### Why it was not integrated

`MLP-PLR` was not integrated into the main architecture because:

- standalone strength did not translate into system-level gain
- score fusion duplicated or distorted the signal already captured by the tree-based ensemble
- the integration failed the same rule used for earlier branches: the final ensemble metric must improve

### Interpretation

`MLP-PLR` is a strong research benchmark and the best deep standalone branch tested so far.
But under the tested fusion path, it is not an accepted component of the production-path architecture.

## What these negative results imply

The project learned several important lessons:

1. strong leakage-safe tabular baselines are hard to beat
2. adding a new deep branch is not automatically useful
3. temporal robustness matters more than architectural novelty alone
4. positive standalone signal is not enough; integration must improve the system-level `AUPRC`
5. representation learning may still matter, but only if it changes the final ensemble in a measurable way

## Exclusion rule

An experimental branch is not integrated into the main architecture unless:

- it wins as a standalone model
- or it materially improves the main system after fusion

The tested `GNN`, `TCN`, `SAINT`, `SCARF fusion`, `TabM integration`, `MLP-PLR fusion`, `graph_only`, `decay_only`, `recent_only`, and `full_recent_decay` branches did not satisfy that rule.

## Consequence for the next phase

The next research phase should focus on:

- stronger feature-stability control on the accepted baseline
- threshold and operating-policy optimization
- tabular feature refinement with controlled ablation
- model changes only when they beat the compact baseline ensemble

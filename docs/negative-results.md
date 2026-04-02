# Negative Results and Exclusion Rationale

This document records experimental branches that were tested and explicitly excluded from the current main architecture.

The purpose is to preserve technical reasoning and prevent weak branches from being reintroduced without new evidence.

## Current main architecture

The active system remains:

- fold-local preprocessing
- tabular backbone with graph-derived fold-local features
- `XGBoost + LightGBM + CatBoost`
- `XGB` meta-learner
- UID smoothing
- temporal evaluation on Colab

This architecture remains the project baseline because it consistently outperformed the experimental branches below under stricter evaluation.

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

## What these negative results imply

The project learned three important lessons:

1. strong leakage-safe tabular baselines are hard to beat
2. adding a new deep branch is not automatically useful
3. temporal robustness matters more than architectural novelty alone

## Exclusion rule

An experimental branch is not integrated into the main architecture unless:

- it wins as a standalone model
- or it materially improves the main system after fusion

Neither the tested `GNN` branch nor the tested `TCN` branch satisfied that rule.

## Consequence for the next phase

The next research phase should focus on more principled directions:

- `SCARF` for representation pre-training
- `SAINT` for stronger tabular deep modeling
- later `GTAN` only if a graph-temporal branch is revisited at a much higher implementation level

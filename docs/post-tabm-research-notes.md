# Post-TabM Research Notes

This note records the most plausible research directions after the project tested:

- `GNN`
- `TCN`
- `SAINT`
- `SCARF`
- `TabM`

and found that none of the tested neural branches materially improved the final stacked tabular system after integration.

## What the recent experiments imply

The current evidence suggests:

1. the main bottleneck is not simply "missing a neural branch"
2. strong leakage-safe tree ensembles remain the project's most reliable backbone
3. standalone neural signal does not automatically survive system-level fusion

This means the next phase should focus on methods that change the *quality of representation or decision policy*, not just add one more branch.

## Most plausible next directions

### 1. Better deep-tabular backbones with stronger defaults

The strongest research question is no longer whether a neural model can produce signal at all.
It is whether a neural model can become strong enough to justify replacing or restructuring the current backbone.

The best candidates now are:

- `RealMLP`-style strong MLP baselines
- improved `TabM` variants with stronger defaults and larger feature sets
- carefully tuned `FT-Transformer` with better numerical embeddings

These directions are more promising than revisiting `TCN` or `GNN`.

### 2. Representation-learning that changes downstream features, not just scores

`SCARF` showed that pre-training helps a standalone branch, but score fusion did not help the full system.

That suggests a better question:

- can representation learning improve the *features* fed into the backbone instead of adding one more meta score?

Promising variants:

- encoder embeddings as compact learned features
- learned feature augmentation before GBDT training
- hard-negative mining or curriculum signals for downstream tabular learners

### 3. Temporal adaptation and drift-aware operation

The project repeatedly observed a gap between faster screening results and strict temporal confirmation.

This implies that:

- concept drift
- distribution shift
- threshold instability

are likely more important than raw model novelty.

This direction includes:

- rolling retraining
- temporal calibration
- drift-aware thresholding
- block-wise operating-policy tuning

### 4. Retrieval-style tabular learning

If the project wants a more novel neural direction later, the most defensible alternative to graph and sequence branches is:

- retrieval-augmented tabular learning

This is attractive in fraud because:

- fraud often appears as similarity to known anomalous neighborhoods
- retrieval may express that more directly than a generic neural branch

But it should only be attempted after the project stabilizes the backbone decision on deep-tabular modeling.

## Recommended priority order

1. restore the stable main backbone and keep it fixed
2. study stronger deep-tabular baselines rather than more side branches
3. explore representation-learning as feature augmentation, not immediate score fusion
4. investigate drift-aware evaluation and operation
5. only then revisit more novel retrieval- or graph-style directions

## Practical conclusion

At this stage, the strongest research move is:

- keep the stable tree ensemble as the accepted architecture
- treat deep models as standalone research baselines unless they improve the final system-level `AUPRC`

The project should not expand the main notebook again until a candidate wins under that rule.

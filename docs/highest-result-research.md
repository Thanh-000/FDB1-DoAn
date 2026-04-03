# Highest-Result Research Shortlist

This note ranks the most promising next improvements after the production notebook was rolled back to the `baseline_tree` architecture.

The key lesson so far is simple:

- many complex branches showed standalone signal
- but the default IEEE-CIS production path still performed best as a compact tree-based ensemble

So the next research phase should optimize the baseline more aggressively rather than adding more model families.

## Priority 1: feature stability pruning

### Why

The current system still carries a broad fold-local feature factory.
The most likely remaining source of lost performance is unstable feature contribution across temporal folds.

### Goal

Keep only the features that are:

- consistently useful across folds
- temporally stable
- not overly dependent on a narrow time segment

### Method

For each outer fold:

- collect feature importance from `XGB`, `LGB`, and `CatBoost`
- rank features by:
  - mean importance
  - fold-to-fold variance
  - sign/ordering stability where applicable

Then:

- remove features with high volatility and low average contribution
- especially inspect:
  - graph-derived columns
  - rare aggregate features
  - target-encoded columns with unstable lift

### Why this is high ROI

It directly targets the strongest existing system instead of introducing a new architecture risk.

## Priority 2: threshold and operating-policy optimization

### Why

The current notebook already shows that threshold choice moves:

- precision
- recall
- F1

quite significantly.

The system may still be underperforming because it is optimized for a generic threshold policy instead of the actual fraud-operation objective.

### Goal

Choose the final decision rule using business or research objectives such as:

- maximize recall subject to minimum precision
- maximize expected utility under asymmetric fraud cost
- optimize review queue size under HITL constraints

### Method

Systematically compare:

- `best_f1`
- `best_f2`
- `recall_at_precision`
- cost-based thresholding

This should be done on the accepted `baseline_tree` architecture first.

## Priority 3: feature-space refinement, not feature expansion

### Why

The project already learned that adding more views or more branches can hurt.
That suggests the next improvement should come from refining the current tabular feature space.

### Goal

Improve the quality of the current feature set, not its size.

### Method

Focus on:

- removing redundant fold-local aggregates
- revisiting temporal count / recency signals in lightweight tabular form
- keeping only low-leakage and stable features
- measuring every change via controlled ablation

### Expected value

This can improve both:

- generalization
- runtime and Colab stability

## Priority 4: stronger tabular optimization, not new branch fusion

### Why

The strongest neural branches did not improve the final system after integration.

So if model-side work continues, it should stay very close to the accepted tree baseline.

### Practical options

- tune `XGB` and `LGB` more carefully under the accepted baseline
- test whether one learner can be removed without hurting performance
- test whether a shallower meta learner is enough

The direction here is simplification and targeted tuning, not more branch diversity.

## Priority 5: deep models only as research baselines

### Why

Deep branches such as `SCARF`, `TabM`, and `MLP-PLR` are still useful research tools.
But they should not drive the production path unless they improve the final ensemble metric.

### Recommended role

Use them to answer research questions such as:

- what representations deep tabular models learn
- whether numerical embeddings improve standalone robustness
- whether pretraining helps under temporal split

But do not integrate them by default without a winning fusion result.

## Literature directions worth following

These papers are the most relevant next references for improvement work:

- Better default tabular baselines and strong MLP comparisons:
  - `Better by Default: Strong Pre-Tuned MLPs and Boosted Trees on Tabular Data`
  - https://arxiv.org/abs/2407.04491
- Temporal shift as a first-class challenge in tabular learning:
  - `Understanding the Limits of Deep Tabular Methods with Temporal Shift`
  - https://arxiv.org/abs/2502.20260
- Numerical embedding quality for tabular deep learning:
  - `On Embeddings for Numerical Features in Tabular Deep Learning`
  - https://arxiv.org/abs/2203.05556
- Concept-drift aware fraud pipelines:
  - `ROSFD: Robust Online Streaming Fraud Detection Against Concept Drift`
  - https://arxiv.org/abs/2504.10229

## Recommended order of work

1. feature stability pruning on the accepted baseline
2. threshold/decision-policy study on the accepted baseline
3. refine the tabular feature set with controlled ablation
4. only after that, revisit model-level tuning or simplified retraining strategies

## Practical conclusion

If the goal is the highest final result rather than the most novel architecture, the project should now optimize:

- the accepted tree baseline
- the stability of its feature space
- the final operating policy

That is a more defensible path than adding new standalone branches that do not survive system-level fusion.

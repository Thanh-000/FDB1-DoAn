# SCARF and SAINT Design Notes

This document defines the next two architecture directions to study before any new implementation is started.

The project should move to these directions only after preserving the current stable tabular baseline.

## Context

The following branches have already been tested and excluded from the main architecture:

- standalone heterogeneous `GNN`
- standalone `TCN`

That means the next research step should not be another lightweight add-on branch.

The next candidates should instead target representation quality directly.

## Why these two directions

### SCARF

`SCARF` is attractive because:

- it supports self-supervised or semi-supervised learning
- it can use the full transaction table without relying on labels for pre-training
- it is a good fit for fraud settings where labels are sparse and temporal drift is strong

### SAINT

`SAINT` is attractive because:

- it is designed for tabular learning rather than sequence or graph proxy tasks
- row attention may capture similarity between fraud clusters
- it provides a stronger neural tabular baseline than a simple TCN branch

## 1. SCARF design

### Goal

Learn a better transaction representation before downstream fraud classification.

### Core idea

Use contrastive pre-training on unlabeled transactions by creating two corrupted views of the same sample and training the encoder to match them while separating other samples in the batch.

### Candidate input features

Use a compact but information-rich subset from the stable baseline:

- `TransactionAmt`
- `LogAmt`
- `Hour`
- `DayOfWeek`
- `Amt_cents`
- `C1_div_C14`
- selected `C/D/M` numeric columns
- selected fold-local aggregate columns that are safe and stable
- optional `V_pca_*`

The first implementation should avoid the full wide feature table.

### Pre-training stage

Input:

- transactions from the IEEE train set
- unlabeled objective during pre-training

Process:

1. sample mini-batches
2. create two corrupted views per transaction
3. encode both views
4. optimize contrastive objective

### Downstream stage

After pre-training:

- freeze or partially fine-tune the encoder
- attach a small classification head
- train under the same temporal split protocol as the baseline

### First downstream target

The first downstream configuration should be:

- standalone neural classifier on top of the pre-trained encoder

Only if that branch shows signal should the embedding be considered for fusion with the main backbone.

### Key ablations

1. no pre-training vs SCARF pre-training
2. frozen encoder vs fine-tuned encoder
3. compact feature set vs expanded feature set

### Success criteria

SCARF is worth continuing only if:

- the downstream branch clearly beats the same architecture without pre-training
- gains remain under strict temporal validation

## 2. SAINT design

### Goal

Evaluate a stronger deep tabular model that can represent inter-sample similarity more explicitly than tree ensembles.

### Core idea

Treat the fraud dataset as a tabular learning problem with:

- column-wise attention
- row-wise or sample-wise attention

This gives the model a way to express similarity between transactions beyond hand-crafted groups.

### Input design

Use a controlled feature subset first:

- stable numeric features from the baseline
- selected categorical features or encoded versions
- no wide graph experiment in the first pass

The first SAINT benchmark should be narrow enough to fit comfortably on Colab.

### Training design

First implementation should use:

- supervised learning only
- short screening setup first
- temporal evaluation identical to the baseline

Pre-training should not be combined with SAINT in the first pass.

### Why not combine with SCARF immediately

Combining both at once would make the ablation unreadable.

The project needs clean evidence on:

- what SAINT does by itself
- what SCARF does by itself

Only later should the two be considered together.

### Key ablations

1. baseline tabular backbone vs standalone SAINT
2. compact feature set vs wider feature set
3. sample-attention enabled vs reduced variant if needed for memory

### Success criteria

SAINT is worth continuing only if:

- it reaches competitive `AUPRC` under temporal split
- or it provides a representation that later improves fusion

## Recommended execution order

1. keep current tabular baseline fixed
2. implement `SCARF` first as a representation-learning study
3. implement `SAINT` second as a stronger tabular deep model
4. compare both against the same stable baseline

## Integration rule

Neither `SCARF` nor `SAINT` should be integrated into the main notebook unless one of them:

- wins as a standalone direction
- or produces embeddings or scores that clearly improve the main system after controlled fusion

## Immediate next implementation target

If the project wants the strongest research value first:

- implement `SCARF`

If the project wants the safer engineering path first:

- implement `SAINT`

Given the recent negative results for `TCN` and `GNN`, `SCARF` is the stronger next research choice on paper because it addresses representation quality directly.

# MLP-PLR Design Notes

This document defines the `MLP-PLR` benchmark direction for the project.

## Why MLP-PLR

The recent tabular literature suggests that deep tabular models often live or die by how numerical
features are represented, not only by the backbone choice.

For this project, that matters because:

- `RealMLP` had real signal but did not beat `TabM`
- `SCARF` had positive standalone results but did not help after fusion
- the main deployment bottleneck still appears under temporal shift, not under i.i.d. conditions

`MLP-PLR` is worth testing because it directly targets the weak point of the current deep baselines:
numerical feature representation.

## Benchmark goal

Answer the question:

`Does a stronger numerical embedding scheme beat RealMLP/SAINT/SCARF standalone under the same temporal split protocol?`

## Scope

The first pass should be:

- standalone only
- temporal fold evaluation only
- compact feature set
- explicit time signal included
- no fusion into the main notebook

## Feature set

Use the same compact feature set as the previous neural benchmarks, with one additional temporal
feature:

- `TransactionAmt`
- `LogAmt`
- `Hour`
- `DayOfWeek`
- `Amt_cents`
- `C1_div_C14`
- `C1`
- `C2`
- `C3`
- `C4`
- `C5`
- `C11`
- `C14`
- `D1`
- `D2`
- `D6`
- `D7`
- `D8`
- `D13`
- `D14`
- `TimeDays`

`TimeDays` is added to let the model express temporal shift more directly than the previous neural
benchmarks.

## Model design

The benchmark should use:

- standardized numerical inputs
- periodic-linear numerical embeddings per feature
- residual MLP backbone
- dropout and normalization

This is not meant to be a full paper reproduction.
It is a controlled benchmark to test whether better numerical embeddings matter more than switching
to another high-level architecture.

## Success criteria

The branch is worth continuing only if:

- it beats `RealMLP`
- it is competitive with or better than `TabM`
- it clearly improves `AUPRC` over `SAINT`

## Next ablations if positive

1. more frequencies per feature
2. wider hidden layers
3. temporal feature variants beyond `TimeDays`
4. pretraining or representation transfer from `SCARF`

## Integration rule

Even if standalone `MLP-PLR` is strong, it should not be integrated into the main IEEE notebook
until:

1. it wins standalone
2. it is repeated on another temporal fold
3. a controlled integration study improves final system-level `AUPRC`

# RealMLP Design Notes

This document defines the `RealMLP` benchmark direction for the project.

## Why RealMLP

Recent tabular literature suggests that a strong MLP baseline with well-chosen defaults can be more competitive than many heavier deep-tabular alternatives.

For this project, that is attractive because:

- the current stable backbone is still tree-based
- `TCN`, `GNN`, `SAINT`, `SCARF fusion`, and `TabM integration` did not improve the final system
- a simpler but stronger neural tabular baseline is more defensible than adding another complex branch

## Benchmark goal

Answer the question:

`Can a strong MLP-style tabular model beat the currently tested neural baselines under the same temporal split protocol?`

## Scope

The first pass should be:

- standalone only
- temporal fold evaluation only
- compact numerical feature set
- no fusion into the main notebook

## Why not integrate immediately

The project now uses a strict rule:

- a candidate must win standalone first
- only then can it be considered for controlled fusion

This is especially important after the negative `TabM` integration result.

## First feature set

Use the same compact `20`-feature set already used for `SCARF`, `SAINT`, and `TabM`:

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

This keeps the benchmark directly comparable to the prior neural runs.

## Model design

The benchmark should use a `RealMLP`-style architecture with:

- scaled numerical inputs
- residual MLP blocks
- normalization
- dropout
- strong but simple defaults

The goal is not exact paper reproduction in the first pass.
The goal is a practical benchmark aligned with the paper's spirit:

- strong MLP baseline
- good tabular defaults
- reasonable time and memory on Colab

## Success criteria

The branch is worth continuing only if:

- it beats `SAINT`
- it beats or matches `TabM`
- it gets close to or exceeds the best `SCARF` standalone result

## Next ablations if positive

1. wider feature set
2. deeper residual stack
3. stronger normalization or activation variants
4. learned numerical embeddings before the backbone

## Integration rule

Even if standalone `RealMLP` is strong, it should not be integrated into the main IEEE notebook until:

1. it wins standalone
2. it is repeated on another temporal fold
3. a controlled fusion study improves the final system-level `AUPRC`

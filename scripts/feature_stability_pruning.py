import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"fold", "model", "feature", "importance"}


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for raw_path in paths:
        matched = sorted(Path().glob(raw_path)) if any(ch in raw_path for ch in "*?[]") else [Path(raw_path)]
        for path in matched:
            df = pd.read_csv(path)
            missing = REQUIRED_COLUMNS - set(df.columns)
            if missing:
                raise ValueError(f"{path} missing required columns: {sorted(missing)}")
            frames.append(df.loc[:, ["fold", "model", "feature", "importance"]].copy())
    if not frames:
        raise ValueError("No importance CSV files were found.")
    out = pd.concat(frames, ignore_index=True)
    out["fold"] = out["fold"].astype(int)
    out["model"] = out["model"].astype(str)
    out["feature"] = out["feature"].astype(str)
    out["importance"] = out["importance"].astype(float)
    return out


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_folds = max(int(df["fold"].nunique()), 1)

    per_model = (
        df.groupby(["model", "feature"], as_index=False)
        .agg(
            mean_importance=("importance", "mean"),
            std_importance=("importance", "std"),
            max_importance=("importance", "max"),
            fold_count=("fold", "nunique"),
        )
        .fillna({"std_importance": 0.0})
    )
    per_model["fold_coverage"] = per_model["fold_count"] / float(total_folds)
    per_model["cv_importance"] = np.where(
        per_model["mean_importance"] > 0,
        per_model["std_importance"] / per_model["mean_importance"],
        np.inf,
    )
    per_model["stability_score"] = (
        per_model["mean_importance"] * per_model["fold_coverage"] / (1.0 + per_model["std_importance"])
    )
    per_model = per_model.sort_values(
        ["model", "stability_score", "mean_importance"],
        ascending=[True, False, False],
    )

    overall = (
        per_model.groupby("feature", as_index=False)
        .agg(
            mean_importance=("mean_importance", "mean"),
            std_importance=("mean_importance", "std"),
            max_importance=("max_importance", "max"),
            mean_fold_coverage=("fold_coverage", "mean"),
            model_count=("model", "nunique"),
        )
        .fillna({"std_importance": 0.0})
    )
    overall["cv_importance"] = np.where(
        overall["mean_importance"] > 0,
        overall["std_importance"] / overall["mean_importance"],
        np.inf,
    )
    overall["stability_score"] = (
        overall["mean_importance"] * overall["mean_fold_coverage"] / (1.0 + overall["std_importance"])
    )
    overall = overall.sort_values(["stability_score", "mean_importance"], ascending=[False, False])
    return per_model, overall


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate fold-wise feature importances into stability scores.")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="CSV paths or globs. Each file must contain columns: fold, model, feature, importance.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/feature_stability",
        help="Directory for aggregated CSV outputs.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many top features to print in the console summary.",
    )
    args = parser.parse_args()

    df = load_inputs(args.inputs)
    per_model, overall = summarize(df)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_model_path = output_dir / "feature_stability_by_model.csv"
    overall_path = output_dir / "feature_stability_overall.csv"
    per_model.to_csv(per_model_path, index=False)
    overall.to_csv(overall_path, index=False)

    print(f"Saved: {per_model_path}")
    print(f"Saved: {overall_path}")
    print("")
    print("Top overall features:")
    print(overall.head(args.top_k).to_string(index=False))


if __name__ == "__main__":
    main()

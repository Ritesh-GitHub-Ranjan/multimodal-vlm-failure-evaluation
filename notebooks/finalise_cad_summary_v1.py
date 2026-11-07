from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # public repository root
RESULTS = ROOT / "results"
SOURCE = RESULTS / "rapid_analysis_v1"
OUT = RESULTS / "cad_finalisation_v1"

FILES = {
    "BLIP-2": RESULTS / "stage7_blip2_full_2401_results.csv",
    "LLaVA-1.5": RESULTS / "full2401_llava15_vast.csv",
    "InstructBLIP": RESULTS / "full2401_qa_prompt_instructblip_vast.csv",
}

CANDIDATE_PATH = SOURCE / "three_model_CAD_summary_10000_candidate.csv"
BOOTSTRAP_PATH = SOURCE / "paired_bootstrap_10000.csv"

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(condition, message):
    if not condition:
        raise ValueError(message)

def load_raw(model, path):
    df = pd.read_csv(path, keep_default_na=False)

    required = {
        "question_id", "corruption", "severity",
        "vqa_soft_accuracy", "confidence"
    }
    require(required.issubset(df.columns),
            f"{model}: missing required columns")

    df["question_id"] = pd.to_numeric(
        df["question_id"], errors="raise"
    ).astype(int)
    df["severity"] = pd.to_numeric(
        df["severity"], errors="raise"
    ).astype(int)

    for col in ["vqa_soft_accuracy", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    keys = ["question_id", "corruption", "severity"]
    require(not df.duplicated(keys).any(),
            f"{model}: duplicate question-condition records")

    return df

def main():
    for path in list(FILES.values()) + [CANDIDATE_PATH, BOOTSTRAP_PATH]:
        require(path.is_file(), f"Missing input: {path}")

    candidate = pd.read_csv(CANDIDATE_PATH)
    bootstrap = pd.read_csv(BOOTSTRAP_PATH)

    required_boot = {
        "model", "condition", "metric", "n_paired",
        "n_boot", "seed", "estimate", "ci_lower", "ci_upper"
    }
    require(required_boot.issubset(bootstrap.columns),
            "Bootstrap CSV schema is incomplete")

    cad_boot = bootstrap[bootstrap["metric"] == "CAD"].copy()
    require(len(cad_boot) == 60,
            "Expected exactly 60 CAD bootstrap records")

    reconstructed = []

    for model, path in FILES.items():
        df = load_raw(model, path)

        clean = df[
            (df["corruption"].str.lower() == "clean")
            & (df["severity"] == 0)
        ][["question_id", "vqa_soft_accuracy", "confidence"]]

        require(not clean["question_id"].duplicated().any(),
                f"{model}: duplicate clean question IDs")

        clean = clean.set_index("question_id")

        model_candidate = candidate[candidate["model"] == model]
        require(len(model_candidate) == 20,
                f"{model}: expected 20 candidate conditions")

        for _, row in model_candidate.iterrows():
            condition = df[
                (df["corruption"] == row["corruption"])
                & (df["severity"] == int(row["severity"]))
            ][["question_id", "vqa_soft_accuracy", "confidence"]]

            require(not condition["question_id"].duplicated().any(),
                    f"{model}: duplicate condition question IDs")

            condition = condition.set_index("question_id")

            paired = clean.join(
                condition,
                how="inner",
                lsuffix="_clean",
                rsuffix="_condition"
            )

            paired = paired.dropna(subset=[
                "vqa_soft_accuracy_clean",
                "confidence_clean",
                "vqa_soft_accuracy_condition",
                "confidence_condition"
            ])

            require(len(paired) >= 2,
                    f"{model}: insufficient valid pairs")

            accuracy_drop = (
                paired["vqa_soft_accuracy_clean"]
                - paired["vqa_soft_accuracy_condition"]
            )
            confidence_drop = (
                paired["confidence_clean"]
                - paired["confidence_condition"]
            )
            cad = accuracy_drop - confidence_drop

            condition_label = (
                f"{row['corruption']}_S{int(row['severity'])}"
            )

            match = cad_boot[
                (cad_boot["model"] == model)
                & (cad_boot["condition"] == condition_label)
            ]

            require(len(match) == 1,
                    f"{model}/{condition_label}: bootstrap match not unique")

            b = match.iloc[0]

            require(len(paired) == int(b["n_paired"]),
                    f"{model}/{condition_label}: paired n mismatch")

            reconstructed.append({
                "model": model,
                "corruption": row["corruption"],
                "severity": int(row["severity"]),
                "n_paired": len(paired),
                "mean_accuracy_clean":
                    paired["vqa_soft_accuracy_clean"].mean(),
                "mean_accuracy_condition":
                    paired["vqa_soft_accuracy_condition"].mean(),
                "mean_confidence_clean":
                    paired["confidence_clean"].mean(),
                "mean_confidence_condition":
                    paired["confidence_condition"].mean(),
                "mean_accuracy_drop": accuracy_drop.mean(),
                "mean_confidence_drop": confidence_drop.mean(),
                "CAD": cad.mean(),
                "CAD_bootstrap_95CI_lower": float(b["ci_lower"]),
                "CAD_bootstrap_95CI_upper": float(b["ci_upper"]),
                "model_key": row["model_key"],
                "condition_key": row["condition_key"],
                "bootstrap_resamples": int(b["n_boot"]),
                "bootstrap_seed": int(b["seed"]),
                "bootstrap_method": "paired percentile bootstrap"
            })

    summary = pd.DataFrame(reconstructed)

    require(len(summary) == 60, "Expected 60 reconstructed summary rows")
    require(not summary.duplicated(
        ["model", "corruption", "severity"]
    ).any(), "Duplicate reconstructed conditions")

    candidate = candidate.sort_values(
        ["model", "corruption", "severity"]
    ).reset_index(drop=True)
    summary = summary.sort_values(
        ["model", "corruption", "severity"]
    ).reset_index(drop=True)

    numeric_columns = [
        "n_paired", "mean_accuracy_clean",
        "mean_accuracy_condition", "mean_confidence_clean",
        "mean_confidence_condition", "mean_accuracy_drop",
        "mean_confidence_drop", "CAD",
        "CAD_bootstrap_95CI_lower", "CAD_bootstrap_95CI_upper"
    ]

    checks = []
    for col in numeric_columns:
        if col == "n_paired":
            passed = summary[col].equals(candidate[col])
        else:
            passed = np.allclose(
                summary[col], candidate[col],
                atol=1e-10, rtol=0
            )
        checks.append({
            "check": f"Reconstructed versus candidate: {col}",
            "passed": bool(passed)
        })

    require(all(x["passed"] for x in checks),
            "Reconstruction does not match the existing candidate")

    require(
        (summary["CAD"] - (
            summary["mean_accuracy_drop"]
            - summary["mean_confidence_drop"]
        )).abs().max() < 1e-10,
        "CAD identity check failed"
    )

    OUT.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        OUT / "three_model_CAD_summary_final.csv", index=False
    )
    pd.DataFrame(checks).to_csv(
        OUT / "reconstruction_validation.csv", index=False
    )

    manifest = []
    for model, path in FILES.items():
        manifest.append({
            "model": model,
            "source_file": str(path),
            "sha256": sha256(path)
        })

    for label, path in [
        ("candidate_summary", CANDIDATE_PATH),
        ("bootstrap_output", BOOTSTRAP_PATH)
    ]:
        manifest.append({
            "model": label,
            "source_file": str(path),
            "sha256": sha256(path)
        })

    pd.DataFrame(manifest).to_csv(
        OUT / "input_manifest.csv", index=False
    )

    print("CAD RECONSTRUCTION VALIDATION")
    print("=" * 55)
    for check in checks:
        print(("PASS" if check["passed"] else "FAIL"), check["check"])
    print("Rows reconstructed:", len(summary))
    print("Output directory:", OUT)
    print("Original files were not overwritten.")

if __name__ == "__main__":
    main()

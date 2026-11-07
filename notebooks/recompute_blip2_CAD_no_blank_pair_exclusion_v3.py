from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # public repository root
RESULTS = ROOT / "results"
OUT = RESULTS / "CAD_blip2_no_blank_pair_exclusion_v3"

INPUT = RESULTS / "stage7_blip2_full_2401_results.csv"
EXISTING = RESULTS / "three_model_CAD_summary_10000_validated.csv"

N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 20261029
EXPECTED_QUESTIONS = 2401

OUT.mkdir(parents=True, exist_ok=True)

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

df = pd.read_csv(
    INPUT,
    dtype={"question_id": str},
    keep_default_na=False
)

required = {
    "question_id", "corruption", "severity",
    "vqa_soft_accuracy", "confidence", "prediction", "n_tokens"
}
missing_columns = required - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

df["question_id"] = df["question_id"].astype(str)
df["severity"] = pd.to_numeric(df["severity"], errors="raise")
df["vqa_soft_accuracy"] = pd.to_numeric(
    df["vqa_soft_accuracy"], errors="raise"
)
df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
df["n_tokens"] = pd.to_numeric(df["n_tokens"], errors="raise")

keys = ["question_id", "corruption", "severity"]
if df.duplicated(keys).any():
    raise ValueError("Duplicate question-condition records detected.")

if df["question_id"].nunique() != EXPECTED_QUESTIONS:
    raise ValueError("Unexpected number of unique questions.")

clean_mask = df["corruption"].eq("clean") & df["severity"].eq(0)
clean = df.loc[
    clean_mask,
    ["question_id", "vqa_soft_accuracy", "confidence"]
].rename(columns={
    "vqa_soft_accuracy": "accuracy_clean",
    "confidence": "confidence_clean"
})

if len(clean) != EXPECTED_QUESTIONS:
    raise ValueError("Unexpected clean baseline row count.")
if clean["question_id"].nunique() != EXPECTED_QUESTIONS:
    raise ValueError("Clean baseline does not contain 2,401 unique questions.")
if clean["confidence_clean"].isna().any():
    raise ValueError("Missing clean baseline confidence.")

corrupted = df.loc[~df["corruption"].eq("clean")].copy()

paired_all = corrupted.merge(
    clean,
    on="question_id",
    how="left",
    validate="many_to_one"
)

if paired_all["accuracy_clean"].isna().any():
    raise ValueError("Missing clean accuracy after pairing.")

if paired_all["confidence_clean"].isna().any():
    raise ValueError("Missing clean confidence after pairing.")

paired_all["delta_accuracy"] = (
    paired_all["accuracy_clean"] - paired_all["vqa_soft_accuracy"]
)

paired_all["delta_confidence"] = (
    paired_all["confidence_clean"] - paired_all["confidence"]
)

# Retain all accuracy records. A CAD value requires valid confidence
# at both clean and corrupted conditions.
paired_all["CAD"] = (
    paired_all["delta_accuracy"] - paired_all["delta_confidence"]
)

paired_all["cad_valid"] = (
    paired_all["confidence_clean"].notna()
    & paired_all["confidence"].notna()
)

# Save the full paired inventory, including rows without calculable CAD.
paired_all.to_csv(OUT / "paired_inventory_all_2401.csv", index=False)

# Record the rows that cannot contribute to CAD.
missing_confidence = paired_all.loc[
    ~paired_all["cad_valid"]
].copy()
missing_confidence.to_csv(
    OUT / "missing_confidence_pairs.csv", index=False
)

rng = np.random.default_rng(BOOTSTRAP_SEED)
summary_rows = []
condition_audit = []

for (corruption, severity), group_all in paired_all.groupby(
    ["corruption", "severity"], sort=True
):
    group = group_all.loc[group_all["cad_valid"]].copy()
    group["_question_id_numeric"] = pd.to_numeric(group["question_id"], errors="raise")
    group = group.sort_values("_question_id_numeric", kind="mergesort").drop(
        columns="_question_id_numeric"
    ).reset_index(drop=True)

    n = len(group)
    if n == 0:
        raise ValueError(f"No valid CAD pairs for {corruption}, severity {severity}")

    if group["question_id"].nunique() != n:
        raise ValueError("Duplicate question IDs within a condition.")

    cad = group["CAD"].to_numpy(dtype=float)
    boot_means = np.empty(N_BOOTSTRAP, dtype=float)

    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = cad[idx].mean()

    lo, hi = np.quantile(boot_means, [0.025, 0.975])

    summary_rows.append({
        "model": "BLIP2",
        "corruption": corruption,
        "severity": int(severity),
        "n_paired": n,
        "n_questions_in_condition": len(group_all),
        "n_missing_confidence_pairs": len(group_all) - n,
        "mean_accuracy_clean": group["accuracy_clean"].mean(),
        "mean_accuracy_condition": group["vqa_soft_accuracy"].mean(),
        "mean_confidence_clean": group["confidence_clean"].mean(),
        "mean_confidence_condition": group["confidence"].mean(),
        "mean_accuracy_drop": group["delta_accuracy"].mean(),
        "mean_confidence_drop": group["delta_confidence"].mean(),
        "CAD": cad.mean(),
        "CAD_bootstrap_95CI_lower": lo,
        "CAD_bootstrap_95CI_upper": hi,
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_method": "paired percentile bootstrap; sequential condition RNG"
    })

    condition_audit.append({
        "corruption": corruption,
        "severity": int(severity),
        "rows_before_confidence_filter": len(group_all),
        "valid_CAD_pairs": n,
        "missing_confidence_pairs": len(group_all) - n,
        "unique_valid_question_ids": group["question_id"].nunique(),
        "empty_output_rows": group_all["prediction"].astype(str)
            .str.strip().eq("<EMPTY_OUTPUT>").sum()
    })

summary = pd.DataFrame(summary_rows)
audit = pd.DataFrame(condition_audit)

if len(summary) != 20:
    raise ValueError("Expected exactly 20 corruption-severity conditions.")

if sorted(summary["n_paired"].unique()) != [2398, 2400, 2401]:
    raise ValueError("Unexpected condition-specific sample sizes.")

summary.to_csv(OUT / "blip2_CAD_summary_no_blank_pair_exclusion.csv", index=False)
audit.to_csv(OUT / "condition_sample_audit.csv", index=False)

# Compare corrected point estimates with the existing BLIP-2 summary.
existing = pd.read_csv(EXISTING)
old = existing[
    existing["model"].astype(str).str.contains("BLIP2", case=False)
].copy()

comparison = summary.merge(
    old,
    on=["corruption", "severity"],
    how="left",
    suffixes=("_corrected", "_old"),
    validate="one_to_one"
)

if len(comparison) != 20 or comparison["CAD_old"].isna().any():
    raise ValueError("Could not match all 20 existing BLIP-2 conditions.")

comparison["CAD_point_change"] = (
    comparison["CAD_corrected"] - comparison["CAD_old"]
)
comparison["old_n_paired"] = comparison["n_paired_old"]
comparison["old_CI_lower"] = comparison["CAD_bootstrap_95CI_lower_old"]
comparison["old_CI_upper"] = comparison["CAD_bootstrap_95CI_upper_old"]

comparison.to_csv(
    OUT / "comparison_with_existing_blip2_CAD.csv",
    index=False
)

metadata = {
    "analysis": "BLIP-2 CAD without blanket question exclusions; deterministic question ordering",
    "input_file": str(INPUT),
    "input_sha256": sha256(INPUT),
    "existing_summary_file": str(EXISTING),
    "existing_summary_sha256": sha256(EXISTING),
    "starting_question_population": EXPECTED_QUESTIONS,
    "clean_baseline_questions": len(clean),
    "missing_confidence_pairs": len(missing_confidence),
    "condition_count": len(summary),
    "bootstrap_resamples": N_BOOTSTRAP,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "bootstrap_method": "paired percentile bootstrap; sequential RNG across sorted conditions; question IDs sorted numerically within condition",
    "cad_definition": "(clean accuracy - corrupted accuracy) - (clean confidence proxy - corrupted confidence proxy)",
    "missingness_policy": "Retain accuracy records; omit only pairs with missing confidence from CAD calculation.",
    "blank_output_policy": "Do not replace missing confidence with zero. Retain observed VQA soft accuracy.",
    "output_directory": str(OUT),
    "question_ordering": "Numeric ascending question_id within each condition before resampling, using an explicit numeric sort key",
    "files": [
        "paired_inventory_all_2401.csv",
        "missing_confidence_pairs.csv",
        "blip2_CAD_summary_no_blank_pair_exclusion.csv",
        "condition_sample_audit.csv",
        "comparison_with_existing_blip2_CAD.csv"
    ]
}

(OUT / "analysis_metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("=" * 85)
print("CORRECTED BLIP-2 CAD ANALYSIS COMPLETE")
print("=" * 85)
print("Output directory:", OUT)
print("Starting population:", EXPECTED_QUESTIONS)
print("Missing confidence pairs:", len(missing_confidence))
print("Conditions:", len(summary))
print("\nCondition sample sizes:")
print(audit.to_string(index=False))

print("\nCAD comparison with existing summary:")
print(comparison[
    ["corruption", "severity", "n_paired_corrected",
     "CAD_corrected", "CAD_old", "CAD_point_change",
     "CAD_bootstrap_95CI_lower_corrected",
     "CAD_bootstrap_95CI_upper_corrected",
     "old_CI_lower", "old_CI_upper"]
].to_string(index=False, float_format=lambda x: f"{x:.6f}"))

print("\nPASS: corrected BLIP-2 analysis saved separately.")
print("No existing files were overwritten.")

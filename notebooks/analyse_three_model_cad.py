from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # public repository root
RESULTS = ROOT / "results"

MODELS = {
    "BLIP2": RESULTS / "stage7_blip2_full_2401_results.csv",
    "LLaVA": RESULTS / "full2401_llava15_vast.csv",
    "InstructBLIP": RESULTS / "full2401_qa_prompt_instructblip_vast.csv",
}

N_QUESTIONS = 2401
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 42

# Exclude the same three BLIP-2 question IDs from every condition.
BLIP2_EXCLUDE_IDS = {"127971013", "278006002", "578131001"}

all_summaries = []
failure_records = []

for model_name, input_path in MODELS.items():
    print(f"\n{'=' * 65}\nMODEL: {model_name}")

    df = pd.read_csv(input_path, keep_default_na=False)
    required = {
        "question_id", "corruption", "severity",
        "vqa_soft_accuracy", "confidence", "prediction", "n_tokens"
    }
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"{model_name}: missing columns {sorted(missing_cols)}")

    df["question_id"] = df["question_id"].astype(str)
    df["severity"] = pd.to_numeric(df["severity"], errors="raise")
    df["vqa_soft_accuracy"] = pd.to_numeric(
        df["vqa_soft_accuracy"], errors="raise"
    )
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")
    df["n_tokens"] = pd.to_numeric(df["n_tokens"], errors="raise")

    keys = ["question_id", "corruption", "severity"]
    if df.duplicated(keys).any():
        raise ValueError(f"{model_name}: duplicate question-condition rows")

    # Preserve empty-output cases as a separate descriptive audit file.
    empty = df[
        df["prediction"].astype(str).str.strip().eq("<EMPTY_OUTPUT>")
    ].copy()
    if not empty.empty:
        empty.insert(0, "model", model_name)
        failure_records.append(empty)

    excluded_ids = BLIP2_EXCLUDE_IDS if model_name == "BLIP2" else set()
    if excluded_ids:
        observed = set(df["question_id"])
        if not excluded_ids.issubset(observed):
            raise ValueError("One or more pre-specified BLIP-2 IDs are absent.")
        df = df[~df["question_id"].isin(excluded_ids)].copy()

    expected_n = N_QUESTIONS - len(excluded_ids)
    clean_mask = df["corruption"].eq("clean") & df["severity"].eq(0)
    clean = df.loc[
        clean_mask,
        ["question_id", "vqa_soft_accuracy", "confidence"]
    ].rename(columns={
        "vqa_soft_accuracy": "accuracy_clean",
        "confidence": "confidence_clean",
    })

    if len(clean) != expected_n or clean["question_id"].nunique() != expected_n:
        raise ValueError(f"{model_name}: unexpected clean baseline size")

    corrupted = df.loc[~df["corruption"].eq("clean")].copy()
    paired = corrupted.merge(
        clean, on="question_id", how="left", validate="many_to_one"
    )

    if paired["accuracy_clean"].isna().any():
        raise ValueError(f"{model_name}: missing clean accuracy")
    if paired["confidence"].isna().any():
        raise ValueError(f"{model_name}: corrupted confidence still missing")
    if paired["confidence_clean"].isna().any():
        raise ValueError(f"{model_name}: clean confidence still missing")

    counts = paired.groupby(["corruption", "severity"])["question_id"].nunique()
    if len(counts) != 20 or not (counts == expected_n).all():
        raise ValueError(
            f"{model_name}: expected 20 conditions with {expected_n} paired questions"
        )

    paired["delta_accuracy"] = (
        paired["accuracy_clean"] - paired["vqa_soft_accuracy"]
    )
    paired["delta_confidence"] = (
        paired["confidence_clean"] - paired["confidence"]
    )
    paired["CAD"] = paired["delta_accuracy"] - paired["delta_confidence"]

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    summary_rows = []

    for (corruption, severity), group in paired.groupby(
        ["corruption", "severity"], sort=True
    ):
        cad = group["CAD"].to_numpy(dtype=float)
        n = len(cad)
        boot_means = np.empty(N_BOOTSTRAP)

        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, size=n)
            boot_means[b] = cad[idx].mean()

        lo, hi = np.quantile(boot_means, [0.025, 0.975])

        summary_rows.append({
            "model": model_name,
            "corruption": corruption,
            "severity": int(severity),
            "n_paired": n,
            "mean_accuracy_clean": group["accuracy_clean"].mean(),
            "mean_accuracy_condition": group["vqa_soft_accuracy"].mean(),
            "mean_confidence_clean": group["confidence_clean"].mean(),
            "mean_confidence_condition": group["confidence"].mean(),
            "mean_accuracy_drop": group["delta_accuracy"].mean(),
            "mean_confidence_drop": group["delta_confidence"].mean(),
            "CAD": cad.mean(),
            "CAD_bootstrap_95CI_lower": lo,
            "CAD_bootstrap_95CI_upper": hi,
        })

    paired_out = RESULTS / f"{model_name.lower()}_CAD_paired_validated.csv"
    summary_out = RESULTS / f"{model_name.lower()}_CAD_summary_validated.csv"
    paired.to_csv(paired_out, index=False)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_out, index=False)
    all_summaries.append(summary)

    print("Questions per condition:", expected_n)
    print("Corrupted paired rows:", len(paired))
    print("Conditions analysed:", len(summary))
    print("Saved:", paired_out)
    print("Saved:", summary_out)

combined = pd.concat(all_summaries, ignore_index=True)
combined_out = RESULTS / "three_model_CAD_summary_validated.csv"
combined.to_csv(combined_out, index=False)

if failure_records:
    failures = pd.concat(failure_records, ignore_index=True)
    failure_out = RESULTS / "three_model_empty_output_audit.csv"
    failures.to_csv(failure_out, index=False)
    print("\nEmpty-output audit:", failure_out)
    print(failures.groupby("model").size().to_string())

exclusion_out = RESULTS / "blip2_CAD_excluded_question_ids.txt"
exclusion_out.write_text(
    "\n".join(sorted(BLIP2_EXCLUDE_IDS)) + "\n",
    encoding="utf-8",
)

print("\nExcluded BLIP-2 question IDs saved:", exclusion_out)
print("Combined summary:", combined_out)
print(combined[
    ["model", "corruption", "severity", "n_paired", "CAD",
     "CAD_bootstrap_95CI_lower", "CAD_bootstrap_95CI_upper"]
].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
print("\nPASS: three-model paired CAD analysis completed.")

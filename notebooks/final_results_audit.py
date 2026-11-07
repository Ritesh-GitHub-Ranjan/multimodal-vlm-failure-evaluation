import pandas as pd
from pathlib import Path
import numpy as np

R = Path("results")
models = {
    "BLIP2": ("blip2", 2398),
    "LLaVA": ("llava", 2401),
    "InstructBLIP": ("instructblip", 2401),
}

failed = False

def check(label, condition):
    global failed
    print(f"{'PASS' if condition else 'FAIL'}  {label}")
    if not condition:
        failed = True

for model, (prefix, n) in models.items():
    print(f"\n{'='*15} {model} {'='*15}")
    p = pd.read_csv(R / f"{prefix}_CAD_paired_validated.csv")
    s = pd.read_csv(R / f"{prefix}_CAD_summary_validated.csv")

    check("Expected paired row count", len(p) == n * 20)
    check("Expected unique questions", p.question_id.nunique() == n)
    check("20 corrupted conditions", len(s) == 20)
    check("No duplicate question-condition rows",
          not p.duplicated(["question_id", "corruption", "severity"]).any())
    check("Expected paired sample sizes", (s.n_paired == n).all())
    check("CAD matches delta_accuracy - delta_confidence",
          np.allclose(p.CAD, p.delta_accuracy - p.delta_confidence,
                      atol=1e-9, rtol=0))
    check("Accuracy and confidence are within [0,1]",
          p.vqa_soft_accuracy.between(0, 1).all()
          and p.confidence.between(0, 1).all())
    check("Confidence baseline is constant within model",
          p.groupby("question_id").confidence_clean.nunique().max() == 1)
    check("All confidence intervals are ordered",
          (s.CAD_bootstrap_95CI_lower <=
           s.CAD_bootstrap_95CI_upper).all())

    agg = p.groupby(["corruption", "severity"]).agg(
        mean_accuracy_condition=("vqa_soft_accuracy", "mean"),
        mean_confidence_condition=("confidence", "mean"),
        mean_accuracy_clean=("accuracy_clean", "mean"),
        mean_confidence_clean=("confidence_clean", "mean"),
        CAD=("CAD", "mean"),
    ).reset_index()

    m = s.merge(agg, on=["corruption", "severity"],
                suffixes=("_summary", "_paired"),
                validate="one_to_one")

    for col in [
        "mean_accuracy_condition", "mean_confidence_condition",
        "mean_accuracy_clean", "mean_confidence_clean", "CAD"
    ]:
        err = (m[f"{col}_summary"] - m[f"{col}_paired"]).abs().max()
        check(f"{col}: summary matches paired data (max diff {err:.2g})",
              err < 1e-8)

three = pd.read_csv(R / "three_model_CAD_summary_validated.csv")
print("\n========== Combined summary ==========")
check("Combined summary has 60 rows", len(three) == 60)
check("Combined summary has 3 models", three.model.nunique() == 3)

for model, (prefix, _) in models.items():
    short = pd.read_csv(R / f"{prefix}_CAD_summary_validated.csv")
    label = short.model.iloc[0]
    combined = three[three.model == label].sort_values(
        ["corruption", "severity"]
    ).reset_index(drop=True)
    short = short.sort_values(
        ["corruption", "severity"]
    ).reset_index(drop=True)

    check(f"{model}: combined summary matches individual summary",
          len(combined) == 20 and
          np.allclose(combined.CAD, short.CAD, atol=1e-10, rtol=0))

print("\nFINAL:", "FAILURES FOUND — investigate before freezing"
      if failed else "ALL CHECKS PASSED")

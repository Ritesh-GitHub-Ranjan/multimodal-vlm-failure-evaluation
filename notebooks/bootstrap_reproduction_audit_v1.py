from pathlib import Path
import csv
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]  # public repository root
RESULTS = ROOT / "results"
OUT = RESULTS / "bootstrap_reproduction_audit_v1"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "BLIP-2": RESULTS / "stage7_blip2_full_2401_results.csv",
    "LLaVA-1.5": RESULTS / "full2401_llava15_vast.csv",
    "InstructBLIP": RESULTS / "full2401_qa_prompt_instructblip_vast.csv",
}

SAVED_BOOT = RESULTS / "rapid_analysis_v1/paired_bootstrap_10000.csv"

N_BOOT = 10_000
BOOT_SEED = 20261029
TOL = 1e-12

required = {
    "question_id", "corruption", "severity", "prediction",
    "confidence", "vqa_soft_accuracy", "n_tokens",
}

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def load_model(model, path):
    df = pd.read_csv(path, keep_default_na=False)

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{model}: missing columns {sorted(missing)}")

    for col in ["question_id", "severity", "n_tokens"]:
        df[col] = pd.to_numeric(df[col], errors="raise").astype(int)

    for col in ["confidence", "vqa_soft_accuracy"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["corruption"] = df["corruption"].astype(str)
    df["prediction"] = df["prediction"].astype(str)

    if df.duplicated(["question_id", "corruption", "severity"]).any():
        raise ValueError(f"{model}: duplicate question-condition keys")

    if not df["vqa_soft_accuracy"].between(0, 1).all():
        raise ValueError(f"{model}: accuracy outside [0,1]")

    bad_conf = (
        df["confidence"].notna()
        & ~df["confidence"].between(0, 1)
    )
    if bad_conf.any():
        raise ValueError(f"{model}: confidence outside [0,1]")

    if df.groupby("question_id").size().nunique() != 1:
        raise ValueError(f"{model}: inconsistent conditions per question")

    return df

print("=== BOOTSTRAP REPRODUCTION AUDIT ===")
print("Python:", sys.version)
print("Platform:", platform.platform())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("SciPy:", scipy.__version__)
print("Resamples:", N_BOOT)
print("Base seed:", BOOT_SEED)
print("Tolerance:", TOL)

# Load and verify the three primary input files.
data = {}
input_manifest = []

for model, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    print(f"\nLoading {model}: {path}")
    df = load_model(model, path)
    data[model] = df

    input_manifest.append({
        "model": model,
        "path": str(path),
        "sha256": sha256(path),
        "rows": len(df),
        "unique_questions": int(df.question_id.nunique()),
        "conditions": int(
            df[["corruption", "severity"]].drop_duplicates().shape[0]
        ),
        "missing_confidence": int(df.confidence.isna().sum()),
    })

    print(
        "Rows:", len(df),
        "| Questions:", df.question_id.nunique(),
        "| Conditions:", df[["corruption", "severity"]].drop_duplicates().shape[0],
    )

# Validate common question-condition grid.
reference = data["BLIP-2"]
ref_keys = set(zip(
    reference.question_id,
    reference.corruption,
    reference.severity
))

for model, df in data.items():
    keys = set(zip(df.question_id, df.corruption, df.severity))
    if keys != ref_keys:
        raise ValueError(f"{model}: question-condition grid differs")

# Reconstruct the exact condition ordering from the original script.
condition_rows = (
    reference[["corruption", "severity"]]
    .drop_duplicates()
    .assign(_clean=lambda x: x.corruption.str.lower().eq("clean"))
    .sort_values(
        ["_clean", "corruption", "severity"],
        ascending=[False, True, True]
    )
)

conditions = list(
    condition_rows[["corruption", "severity"]]
    .itertuples(index=False, name=None)
)

clean_conditions = [
    c for c in conditions if c[0].lower() == "clean"
]

if len(clean_conditions) != 1:
    raise ValueError(f"Expected one clean condition: {clean_conditions}")

CLEAN = clean_conditions[0]
CORRUPTED = [c for c in conditions if c != CLEAN]

if len(CORRUPTED) != 20:
    raise ValueError(f"Expected 20 corrupted conditions, got {len(CORRUPTED)}")

def label(c):
    return "clean" if c == CLEAN else f"{c[0]}_S{c[1]}"

def pivot_metric(df, metric):
    return df.pivot(
        index="question_id",
        columns=["corruption", "severity"],
        values=metric
    )

def condition_series(wide, cond):
    matches = [
        i for i, col in enumerate(wide.columns)
        if tuple(col) == tuple(cond)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one column for {cond}; found {len(matches)}"
        )
    return wide.iloc[:, matches[0]]

# Reproduce original generator creation and model-specific seeds.
rng_by_model = {
    model: np.random.default_rng(BOOT_SEED + i)
    for i, model in enumerate(data)
}

audit_rows = []
start_time = time.time()

for model, df in data.items():
    print(f"\nReproducing bootstrap: {model}")

    acc = pivot_metric(df, "vqa_soft_accuracy")
    conf = pivot_metric(df, "confidence")

    ids = acc.index.intersection(conf.index)
    acc, conf = acc.loc[ids], conf.loc[ids]

    clean_acc = condition_series(acc, CLEAN)
    clean_conf = condition_series(conf, CLEAN)

    rng = rng_by_model[model]
    actual_seed = BOOT_SEED + list(data).index(model)

    for condition_index, cond in enumerate(CORRUPTED, start=1):
        valid = (
            clean_acc.notna()
            & acc[cond].notna()
            & clean_conf.notna()
            & conf[cond].notna()
        )

        # Original script has CAD_EXCLUSIONS = set(), so no exclusions apply.
        pair_ids = valid.index[valid]

        a = (
            clean_acc.loc[pair_ids]
            - condition_series(acc, cond).loc[pair_ids]
        ).to_numpy(float)

        c = (
            clean_conf.loc[pair_ids]
            - condition_series(conf, cond).loc[pair_ids]
        ).to_numpy(float)

        cad = a - c
        n = len(pair_ids)

        if n < 2:
            raise ValueError(f"Too few pairs: {model} {label(cond)}")

        # Exact original resampling shape and shared indices for all metrics.
        indices = rng.integers(0, n, size=(N_BOOT, n))

        boot_a = a[indices].mean(axis=1)
        boot_c = c[indices].mean(axis=1)
        boot_cad = cad[indices].mean(axis=1)

        for metric, point, values in [
            ("accuracy_drop", a.mean(), boot_a),
            ("confidence_proxy_drop", c.mean(), boot_c),
            ("CAD", cad.mean(), boot_cad),
        ]:
            lo, hi = np.quantile(values, [0.025, 0.975])

            audit_rows.append({
                "model": model,
                "condition": label(cond),
                "metric": metric,
                "n_paired": n,
                "n_boot": N_BOOT,
                "seed": BOOT_SEED,
                "actual_rng_seed": actual_seed,
                "estimate": float(point),
                "ci_lower": float(lo),
                "ci_upper": float(hi),
                "estimate_pp": float(point * 100),
                "ci_lower_pp": float(lo * 100),
                "ci_upper_pp": float(hi * 100),
            })

        del indices, boot_a, boot_c, boot_cad

        if condition_index % 5 == 0:
            print(f"  Completed {condition_index}/20 conditions")

# Save independent results separately.
audit_df = pd.DataFrame(audit_rows)
audit_path = OUT / "paired_bootstrap_reproduced.csv"
audit_df.to_csv(audit_path, index=False)

# Compare against the original saved bootstrap.
if not SAVED_BOOT.is_file():
    raise FileNotFoundError(SAVED_BOOT)

saved_df = pd.read_csv(SAVED_BOOT, keep_default_na=False)

key_fields = ["model", "condition", "metric"]
numeric_fields = [
    "n_paired", "n_boot", "seed",
    "estimate", "ci_lower", "ci_upper",
    "estimate_pp", "ci_lower_pp", "ci_upper_pp",
]

if saved_df.duplicated(key_fields).any():
    raise ValueError("Duplicate keys in saved bootstrap file")

if audit_df.duplicated(key_fields).any():
    raise ValueError("Duplicate keys in reproduced bootstrap file")

saved_map = {
    tuple(str(r[k]) for k in key_fields): r
    for r in saved_df.to_dict("records")
}
audit_map = {
    tuple(str(r[k]) for k in key_fields): r
    for r in audit_df.to_dict("records")
}

comparison_rows = []

for key in sorted(set(saved_map) | set(audit_map)):
    old = saved_map.get(key)
    new = audit_map.get(key)

    if old is None or new is None:
        comparison_rows.append({
            "model": key[0],
            "condition": key[1],
            "metric": key[2],
            "status": "MISSING_ROW",
            "fields_mismatched": "row",
        })
        continue

    mismatches = []

    for field in numeric_fields:
        old_value = float(old[field])
        new_value = float(new[field])

        if abs(old_value - new_value) > TOL:
            mismatches.append(field)

    comparison_rows.append({
        "model": key[0],
        "condition": key[1],
        "metric": key[2],
        "status": "MATCH" if not mismatches else "MISMATCH",
        "fields_mismatched": ";".join(mismatches),
    })

comparison_df = pd.DataFrame(comparison_rows)
comparison_path = OUT / "bootstrap_comparison.csv"
comparison_df.to_csv(comparison_path, index=False)

matches = int(comparison_df.status.eq("MATCH").sum())
mismatches = int(comparison_df.status.eq("MISMATCH").sum())
missing = int(comparison_df.status.eq("MISSING_ROW").sum())

report = {
    "audit": "Independent paired bootstrap reproduction",
    "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "python": sys.version,
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scipy": scipy.__version__,
    "n_boot": N_BOOT,
    "base_seed_recorded_in_original": BOOT_SEED,
    "model_order": list(data),
    "actual_rng_seeds": {
        model: BOOT_SEED + i for i, model in enumerate(data)
    },
    "condition_order": [label(c) for c in CORRUPTED],
    "tolerance": TOL,
    "saved_rows": len(saved_df),
    "reproduced_rows": len(audit_df),
    "matching_rows": matches,
    "mismatching_rows": mismatches,
    "missing_rows": missing,
    "input_manifest": input_manifest,
    "saved_bootstrap_sha256": sha256(SAVED_BOOT),
    "reproduced_output_sha256": sha256(audit_path),
    "comparison_output_sha256": sha256(comparison_path),
    "elapsed_seconds": round(time.time() - start_time, 2),
}

report_path = OUT / "reproduction_report.json"
report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

print("\n" + "=" * 75)
print("BOOTSTRAP REPRODUCTION RESULTS")
print("=" * 75)
print("Saved rows:", len(saved_df))
print("Reproduced rows:", len(audit_df))
print("Matching rows:", matches)
print("Mismatching rows:", mismatches)
print("Missing rows:", missing)
print("Tolerance:", TOL)
print("Elapsed seconds:", report["elapsed_seconds"])

print("\nActual RNG seeds:")
for model, seed in report["actual_rng_seeds"].items():
    print(f"  {model}: {seed}")

print("\nAudit outputs:")
print(audit_path)
print(comparison_path)
print(report_path)

if matches == 180 and mismatches == 0 and missing == 0:
    print("\nRESULT: ALL 180 BOOTSTRAP ROWS REPRODUCED")
else:
    print("\nRESULT: REPRODUCTION DIFFERENCES REQUIRE INVESTIGATION")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]  # public repository root
RESULTS = ROOT / "results"
OUT = RESULTS / "thesis_figures"
OUT.mkdir(parents=True, exist_ok=True)

source = RESULTS / "three_model_CAD_summary_validated.csv"
df = pd.read_csv(source)

required = {
    "model", "corruption", "severity", "n_paired",
    "mean_accuracy_drop", "mean_confidence_drop", "CAD",
    "CAD_bootstrap_95CI_lower", "CAD_bootstrap_95CI_upper"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

models = ["BLIP2", "LLaVA", "InstructBLIP"]
corruptions = [
    "gaussian_noise", "motion_blur", "fog", "jpeg_compression"
]
labels = {
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
    "fog": "Fog",
    "jpeg_compression": "JPEG compression",
}
metrics = [
    ("mean_accuracy_drop", "Accuracy drop", "Accuracy drop (percentage points)"),
    ("mean_confidence_drop", "Confidence-proxy drop",
     "Confidence-proxy drop (percentage points)"),
    ("CAD", "Confidence–accuracy divergence (CAD)",
     "CAD (percentage points)"),
]

df = df[df["severity"].between(1, 5)].copy()

if len(df) != 60:
    raise ValueError(f"Expected 60 corrupted-condition rows, found {len(df)}")
if df.duplicated(["model", "corruption", "severity"]).any():
    raise ValueError("Duplicate model-condition rows detected")

for model in models:
    subset = df[df["model"] == model]
    if len(subset) != 20:
        raise ValueError(f"{model}: expected 20 conditions, found {len(subset)}")
    if set(subset["corruption"]) != set(corruptions):
        raise ValueError(f"{model}: unexpected corruption types")

# All drops and CAD are shown in percentage points for easier interpretation.
# CAD intervals are the saved, unadjusted percentile bootstrap intervals.
for column in [
    "mean_accuracy_drop", "mean_confidence_drop", "CAD",
    "CAD_bootstrap_95CI_lower", "CAD_bootstrap_95CI_upper"
]:
    df[column] = pd.to_numeric(df[column], errors="raise") * 100

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "figure.titlesize": 14,
})

for metric, title, ylabel in metrics:
    fig, axes = plt.subplots(
        nrows=1, ncols=3, figsize=(16, 4.8), sharex=True
    )

    for ax, model in zip(axes, models):
        subset = df[df["model"] == model]

        for corruption in corruptions:
            g = subset[subset["corruption"] == corruption].sort_values(
                "severity"
            )
            ax.plot(
                g["severity"], g[metric],
                marker="o", linewidth=1.8, markersize=4,
                label=labels[corruption]
            )

            if metric == "CAD":
                lower = g["CAD_bootstrap_95CI_lower"].to_numpy()
                upper = g["CAD_bootstrap_95CI_upper"].to_numpy()
                values = g["CAD"].to_numpy()
                ax.fill_between(
                    g["severity"].to_numpy(), lower, upper,
                    alpha=0.12
                )

        ax.axhline(0, linewidth=0.9, linestyle="--")
        ax.set_title(model)
        ax.set_xlabel("Corruption severity")
        ax.set_xticks([1, 2, 3, 4, 5])
        ax.grid(True, alpha=0.25)

    axes[0].set_ylabel(ylabel)
    handles, legend_labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc="lower center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.01)
    )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.12, 1, 0.92])

    filename = {
        "mean_accuracy_drop": "figure_accuracy_drop",
        "mean_confidence_drop": "figure_confidence_proxy_drop",
        "CAD": "figure_confidence_accuracy_divergence",
    }[metric]

    fig.savefig(OUT / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("Created:", filename + ".png and .pdf")

# Export the exact numeric data used by the figures.
export_cols = [
    "model", "corruption", "severity", "n_paired",
    "mean_accuracy_drop", "mean_confidence_drop", "CAD",
    "CAD_bootstrap_95CI_lower", "CAD_bootstrap_95CI_upper"
]
df[export_cols].sort_values(
    ["model", "corruption", "severity"]
).to_csv(OUT / "figure_source_data_percentage_points.csv", index=False)

print("\nOutput directory:", OUT)
print("Files:")
for path in sorted(OUT.iterdir()):
    print(f"  {path.name} ({path.stat().st_size / 1024:.1f} KB)")

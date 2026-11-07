#!/usr/bin/env python3
"""
Create thesis figure revision v3 from the already generated v2 SVGs.

This is a presentation-only revision:
- does not read or alter the numerical source data;
- does not recalculate statistics or confidence intervals;
- increases typography for thesis-page readability;
- reduces CAD confidence-band opacity.

Run from the thesis project root:
    python notebooks/refine_thesis_figures_v3.py
"""

from pathlib import Path
import hashlib
import json
import re
import sys

ROOT = Path.cwd()
INPUT_DIR = ROOT / "results" / "thesis_final_figures_v2"
OUTPUT_DIR = ROOT / "results" / "thesis_final_figures_v3"

FIGURES = (
    "figure_accuracy_drop.svg",
    "figure_confidence_proxy_drop.svg",
    "figure_confidence_accuracy_divergence.svg",
)

# Explicit mapping preserves the visual hierarchy while making labels
# more legible after the figures are reduced to thesis-page width.
FONT_SIZE_MAP = {
    "10": "13",  # tick labels
    "11": "14",  # secondary labels
    "12": "15",  # legend labels
    "14": "18",  # axis labels
    "16": "19",  # panel headings
    "21": "24",  # figure title
}

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def main() -> int:
    if not INPUT_DIR.is_dir():
        print(f"ERROR: v2 figure directory not found: {INPUT_DIR}")
        print("Run the v2 generator first, or correct INPUT_DIR in this script.")
        return 2

    missing = [name for name in FIGURES if not (INPUT_DIR / name).is_file()]
    if missing:
        print("ERROR: required v2 SVG file(s) missing:")
        for name in missing:
            print(f"  - {name}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    for name in FIGURES:
        source = INPUT_DIR / name
        target = OUTPUT_DIR / name
        svg = source.read_text(encoding="utf-8")

        if not svg.lstrip().startswith("<svg"):
            print(f"ERROR: {name} does not appear to be an SVG document.")
            return 2

        original_hash = sha256(source)

        # Replace only explicit font-size attributes. Geometry, plotted
        # coordinates, labels, data points, and axes remain unchanged.
        for old, new in FONT_SIZE_MAP.items():
            svg = re.sub(
                rf'font-size="{re.escape(old)}"',
                f'font-size="{new}"',
                svg,
            )

        # The v2 CAD generator uses opacity="0.10" for CI polygons.
        # Reduce only those polygon fills to avoid obscuring the series.
        if name == "figure_confidence_accuracy_divergence.svg":
            svg = svg.replace('opacity="0.10"', 'opacity="0.055"')

        target.write_text(svg, encoding="utf-8")

        # Basic structural checks after writing.
        written = target.read_text(encoding="utf-8")
        if written.count("<svg") != 1 or "</svg>" not in written:
            print(f"ERROR: SVG structure check failed for {name}")
            return 2

        records.append({
            "file": name,
            "input_sha256": original_hash,
            "output_sha256": sha256(target),
            "font_size_mapping": FONT_SIZE_MAP,
            "cad_confidence_band_opacity": (
                "0.055" if name == "figure_confidence_accuracy_divergence.svg"
                else None
            ),
            "data_or_geometry_recalculated": False,
        })
        print(f"Created: {target.relative_to(ROOT)}")

    manifest = {
        "revision": "thesis_final_figures_v3",
        "purpose": "presentation-only refinement of v2 SVG figures",
        "input_directory": str(INPUT_DIR.relative_to(ROOT)),
        "output_directory": str(OUTPUT_DIR.relative_to(ROOT)),
        "numerical_data_changed": False,
        "statistics_recalculated": False,
        "confidence_intervals_recalculated": False,
        "changes": [
            "increased SVG font sizes for thesis-page readability",
            "reduced CAD confidence-interval polygon opacity from 0.10 to 0.055",
        ],
        "figures": records,
    }

    manifest_path = OUTPUT_DIR / "figure_refinement_manifest_v3.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFIGURE REFINEMENT COMPLETE")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Manifest: {manifest_path}")
    print("No numerical values, plotted coordinates, or statistical results were changed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

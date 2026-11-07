# Multimodal Vision–Language Model Failure Evaluation

This repository contains the public research materials for an MSc study evaluating confidence–accuracy divergence in pretrained vision–language models under visual corruption.

Healthcare VQA is used as a high-risk evaluation domain. This is a study of model behaviour and evaluation limitations, not a healthcare product or clinical decision-support system.

## Research scope

The study evaluates three pretrained models:

- BLIP-2
- LLaVA-1.5
- InstructBLIP

The evaluation uses 2,401 VQA v2 validation image–question pairs, a clean baseline, and four visual corruption types at five severity levels:

- Gaussian noise
- Motion blur
- Fog
- JPEG compression

The central measure is Confidence–Accuracy Divergence (CAD), defined as the difference between the deterioration in answer accuracy and the deterioration in a token-confidence proxy, using paired clean and corrupted observations.

A positive CAD indicates that accuracy deteriorated more than the confidence proxy. A negative CAD indicates that the confidence proxy deteriorated more than accuracy.

## Main results

See:

- `results/cad/three_model_CAD_summary_final.csv`
- `results/cad/main_results_60_conditions.csv`
- `results/cad/CAD_FINALISATION_METHOD_NOTE_v1.md`
- `results/figures/`

The CAD measure is an aggregate relative-deterioration measure. The token-confidence proxy is not an answer-level calibrated probability. It should not be interpreted as one.

## Supporting analyses

- `results/proxy_ece/` contains aggregate proxy-ECE summaries.
- `results/spearman/` contains pair-matched condition-level association results.
- `results/bootstrap/` contains the independent bootstrap reproduction comparison and report.

These analyses support interpretation of the central CAD evaluation; they do not establish generalisation beyond the evaluated dataset, models, and corruption conditions.

## Code

Selected analysis and figure-generation scripts are provided in `notebooks/`.

The scripts document the analysis procedures. Full end-to-end reproduction requires obtaining the underlying dataset and model resources independently, configuring the required software environment, and following their respective terms. Raw question-level predictions, image files, model weights, and caches are not distributed in this repository.

## Data and licensing

The VQA v2 dataset and COCO image resources are not included. Users must obtain them from their respective official sources and review the applicable terms before use or redistribution.

This repository does not grant rights to third-party datasets, model weights, or pretrained model resources. Model names and associated marks belong to their respective owners.

No blanket licence is applied to third-party materials. A repository-level licence should only be added after the ownership and redistribution status of each included item has been reviewed.

## Methodological qualifications

- Confidence is measured using a token-confidence proxy, not a calibrated answer-level probability.
- CAD is an aggregate comparison of relative deterioration, not an individual answer calibration score.
- Confidence-dependent analyses use available paired confidence observations; missing confidence values are not replaced with zero.
- Bootstrap intervals are reported with their documented reproduction audit.
- Qualitative failure review was purposive and must not be interpreted as a prevalence estimate.
- Results do not establish performance on other datasets, domains, or model families.

## Evidence integrity

The published aggregate outputs are selected from a versioned research record. Historical files, raw predictions, question-level inventories, and visual-review materials are intentionally retained outside this public release.

For the research boundary and evidence mapping, see:

- `docs/RESEARCH_SCOPE.md`
- `docs/RESEARCH_EVIDENCE_REGISTER.csv`

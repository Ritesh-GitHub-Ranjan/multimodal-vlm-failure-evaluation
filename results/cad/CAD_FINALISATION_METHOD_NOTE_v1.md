# CAD Finalisation Method Note — Version 1

## Status

Numerical reconstruction passed. This record documents the preferred
no-manual-exclusion CAD analysis. Formal thesis-wide adoption remains
subject to statistical reconciliation and figure/table updates.

## Research scope

The analysis evaluates confidence–accuracy divergence under image
corruption in BLIP-2, LLaVA-1.5 and InstructBLIP, using the fixed
2,401-question VQA evaluation subset.

## Analysis population

No manual question-ID exclusions are applied.

For each model and corrupted condition, observations are paired with
the clean baseline using question ID. CAD uses complete pairs for which
clean and corrupted accuracy and confidence values are available.

The paired sample size is therefore condition-specific and may be
smaller than 2,401. Missing confidence is excluded only from analyses
that require confidence. Accuracy records are not removed from the
separate accuracy analysis, and missing confidence is never replaced
with zero.

## Measures

Accuracy is the saved VQA soft-accuracy score.

The confidence proxy is the exponentiated mean selected generated-token
log probability. It is a token-level confidence proxy, not a calibrated
probability that the complete answer is correct.

Accuracy drop = clean accuracy - corrupted-condition accuracy.

Confidence-proxy drop = clean confidence proxy -
corrupted-condition confidence proxy.

CAD = accuracy drop - confidence-proxy drop.

Positive CAD means that the mean accuracy drop exceeds the mean
confidence-proxy drop. Negative CAD means that the mean
confidence-proxy drop exceeds the mean accuracy drop.

## Bootstrap procedure

The existing bootstrap output uses 10,000 paired question-level
resamples and seed 20261029. Clean and corrupted observations remain
paired during resampling. The same sampled indices are used to calculate
accuracy drop, confidence-proxy drop and CAD.

The 95% confidence intervals use the 2.5th and 97.5th percentiles of
the bootstrap distributions.

The rapid-analysis implementation uses a separate random-number
generator for each model, initialised using the base seed plus the
model's iteration index.

## Reconstruction procedure

The finalisation script independently reconstructs the paired
condition-level means, drops and CAD from the three authoritative raw
CSV files.

It takes the 60 model-condition combinations from the existing
candidate summary and attaches the corresponding CAD confidence
intervals from the existing bootstrap output.

It does not rerun the bootstrap. It does not independently regenerate
the bootstrap confidence intervals.

The reconstruction is checked against the candidate summary for:
- paired sample size;
- clean and corrupted accuracy means;
- clean and corrupted confidence means;
- accuracy and confidence-proxy drops;
- CAD point estimates;
- lower and upper bootstrap confidence limits.

All 10 numerical checks passed across the 60 model-condition rows.

The reconstruction therefore establishes consistency between the raw
observations, reconstructed point estimates, existing bootstrap output
and candidate summary. It is not an independent replication of the
bootstrap procedure.

## Limitations and interpretation

The confidence proxy must not be described as whole-answer correctness
probability or conventional calibrated confidence.

CAD is an aggregate difference between two mean changes. It is not an
individual answer-level calibration score.

Results are limited to the three evaluated models, the selected VQA
subset and the four evaluated corruption types. They do not establish
a universal property of vision-language models.

The no-manual-exclusion policy is based on the proposal and examiner
records reviewed during this audit. If another approved protocol
document is found that explicitly prespecifies the three historical
BLIP-2 exclusions, this decision must be reconsidered and documented.

## Authoritative reconstructed output

results/cad_finalisation_v1/three_model_CAD_summary_final.csv

SHA-256:
e9cf60f6499c6ff6f98144bb33383a7c3d1b234a62a1f24f112d866055249793

## Reconstruction script

notebooks/finalise_cad_summary_v1.py

SHA-256:
85039e68dc3fedf0c02a2c095af73fb3318a3390561c0b7d435619064ae57247

## Validation output

results/cad_finalisation_v1/reconstruction_validation.csv

SHA-256:
3f16939b5233ce343da4aad956e11a26d435359dd66dec00b834b08bece11550

## Input manifest

results/cad_finalisation_v1/input_manifest.csv

SHA-256:
01be81798f13392f724305ff55cb214f9aa5e4aa3d9cc560a74d0cc4f3f9f66d

## Source data and bootstrap provenance

BLIP-2 raw CSV:
1befc275ff8354dee0e56b679d344895ea4537f3b71082aac007d1dbe4c703b2

LLaVA-1.5 raw CSV:
4df867281a4ceb59b746b5ed1f5b64f73946f7904865472a675de34f4cec6590

InstructBLIP raw CSV:
51e55c26b7a9d2c4393775c18b9779a58a34cc2a35432bd4261c1363fcff0166

Existing 10,000-resample bootstrap output:
54eac07f1cecd906fc3814105d08da4922a00b8475063a0135b18be76a696019

Original rapid-analysis script:
4be0e7ad637dafff6358ab036efd6f32f17dc97a3c9864efad0e572853ab576

## Historical outputs

Earlier exclusion-based CAD summaries and analysis outputs are retained
as historical records. They must not be silently mixed with this
no-manual-exclusion analysis.

## Finalisation status

The reconstructed 60-row summary passed all numerical validation
checks. Formal adoption in the thesis remains subject to the subsequent
statistical reconciliation and dependent figure/table audit.

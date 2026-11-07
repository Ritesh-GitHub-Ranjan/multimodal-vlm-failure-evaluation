# Research Scope and Boundaries

## Research objective

The study evaluates confidence–accuracy divergence in pretrained vision–language models under visual corruption. Healthcare visual question answering is the evaluation domain, not the intended product or application.

## Core problem

The single central problem is the relationship between model confidence proxies and answer accuracy when visual inputs are degraded.

## Models

- BLIP-2
- LLaVA-1.5
- InstructBLIP

## Dataset and evaluation conditions

The main evaluation uses 2,401 image–question pairs from the VQA v2 validation subset. Each model is evaluated under a clean baseline and four corruption types at five severity levels.

## Corruption conditions

- Gaussian noise
- Motion blur
- Fog
- JPEG compression

## Scope exclusions

The project does not:

- Train a large model from scratch.
- Develop a clinical decision-support system.
- Claim that the evaluated models are suitable for clinical use.
- Claim empirical generalisation beyond the evaluated models, dataset subset and corruption conditions.
- Treat token-derived confidence as calibrated answer-level probability.
- Treat purposive qualitative review as a prevalence estimate.

## Research boundary

The contribution is an evaluation and failure-analysis study. Any proposed extension beyond the frozen experimental design must be documented separately and must not silently replace the approved scope.

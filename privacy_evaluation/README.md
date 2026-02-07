# Privacy evaluation (reference implementation)

This directory contains supporting material for the privacy risk assessment performed in the study  
**_Safe data for safe medication_**.

It is intended to document the evaluation logic and outputs and to provide small helper scripts where useful.  
References and tool links are listed in the main repository README.

---

## Scope

In the study, privacy risk was assessed for both anonymized and synthetic datasets, including:

- singling-out risk
- linkability risk
- attribute inference risk
- membership inference risk

Anonymeter-based evaluations were used for both anonymized and synthetic data.  
Membership inference evaluation differs by data type (synthetic vs anonymized). 

---

## What is provided here

- A reference example showing how we invoked Anonymeter and how we structured outputs  
  (configured for dummy data and reduced runtime settings). The provided Anonymeter script is a simplified, shareable reference implementation.
- Lightweight helper scripts to aggregate results produced by the two membership inference evaluation tools.

This directory does not attempt to bundle or fully configure the external privacy evaluation frameworks; users should follow the upstream documentation and the methodological description in the paper.

---

## Intended use

These materials are intended to be:
- read and adapted,
- used as a guide for reimplementation,
- used to structure and compare privacy evaluation outputs across datasets.

They are not intended as a turnkey privacy attack pipeline.

---

## Membership inference

We configured both membership inference frameworks as described in the manuscript:

- Attacker model: Random Forest  
- Features: Histogram
- Targets: 10 "average" + 10 "outlier" records per dataset
  - Targets selected following Halilovic, M., Meurers, T., Otte, K. & Prasser, F. Are You the Outlier? Identifying Targets for Privacy Attacks on Health Datasets. Stud Health Technol Inform 316, 1224–1225 (2024).
- Sampling / repeats: sample size 500, 10 training + 10 test replicates per run, 30 runs total (≥95% coverage rationale)

Config notes:
- Shadow-model framework: `nIter=30`, `nShadows=10`, `sizeRawT=sizeSynT=500`, `nSynT=10` (targets passed as a list of target IDs).
- Phantom: `runCount=30`, `runTrainingCount=10`, `runTestCount=10`, `sizeSampleTraining=sizeSampleTest=500`, `overlap=0.8`.
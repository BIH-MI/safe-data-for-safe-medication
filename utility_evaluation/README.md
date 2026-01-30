# Utility evaluation (reference implementation)

This directory contains reference implementations to illustrate how utility was assessed in the study  
**_Safe data for safe medication_**.

The scripts provided here are not the original analysis code used in the study.  
They are intended to make the analysis logic and structure explicit and to support reimplementation on independent datasets.

---

## Scope

In our study, fidelity and utility assessment comprised multiple components, including cohort descriptives, covariate prevalence, correlation analyses, and outcome modeling.

The reference implementation in this directory focuses on the core analytical components most relevant for downstream decision-making:
- cohort descriptives by treatment group (DOAC vs VKA),
- prevalence of baseline binary covariates,
- incidence rates for selected outcomes, and
- adjusted and IPTW-weighted hazard ratios for major bleeding and all-cause mortality.

Additional analyses reported in the paper are not reproduced in full here.

---

## About the provided scripts

- The scripts are written to run on dummy data and prioritize portability and clarity.
- Model specifications may be simplified compared to the original study (e.g. linear age adjustment instead of spline-based terms) to ensure stability on the dummy datasets.
- Comments in the code indicate where the original analysis used more complex specifications.

For orientation, `run_utility.py` serves as a minimal entry point and `utility_metrics.py` contains the main computations.

---

## Intended use

These scripts are intended to be:
- read and understood,
- adapted to local data structures,
- reimplemented with dataset-specific preprocessing and validation.

They are not intended as a drop-in reproduction of the published analysis.

For full methodological details, please refer to the paper.

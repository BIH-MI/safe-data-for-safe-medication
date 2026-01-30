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

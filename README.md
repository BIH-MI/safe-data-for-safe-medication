# Safe Data for Safe Medication

This repository accompanies **_Safe data for safe medication_** and provides the configuration and tooling used to recreate our workflow.

It is meant to help others recreate the same setup on their own data by documenting key assumptions, parameter choices, and analysis steps.

---

## Scope and intent

This repository provides:
- configuration documentation for key study components,
- pointers to the external tools used, and
- small, shareable select reference scripts that illustrate how parts of the privacy and utility evaluation were executed.

It is not a full end-to-end pipeline; reimplementation requires mapping variables, cohort definitions, and I/O to local data structures.

---

## 1. Data availability

The original study dataset is not included due to privacy and data use restrictions. Please refer to the paper for details on the source data and cohort construction.

For convenience, the repository includes a small script in `data/` that generates dummy data matching the expected schema. This is intended only to demonstrate how the reference evaluation scripts run end-to-end (it is not clinically meaningful and should not be used for validation).

---

## 2. Anonymization with ARX

We used the ARX de-identification tool:

- Prasser, F., Eicher, J., Spengler, H., Bild, R. & Kuhn, K. A. Flexible data anonymization using ARX—Current status and challenges ahead. Softw: Pract Exper 50, 1277–1304 (2020), https://doi.org/10.1002/spe.2812
- Project: https://github.com/arx-deidentifier/arx

### 2.1 Configuration
In `anonymization/`, we provide YAML files that document our anonymization setup (attribute roles, protected variables, privacy model parameters).  
These files are intended for transparency and reimplementation, but they are not directly consumed by ARX.

Provided files:

- `data_config_ci.yml`  
  Documentation of our context-independent anonymization setup

- `data_config_cd.yml`  
  Documentation of our context-dependent anonymization setup

- `anonymization_config_K2.yml`  
  Documentation of the ARX privacy model and key parameters used.

These YAMLs specify:

- Attribute roles in each scenario (identifier, indirect identifier, sensitive, insensitive)
- The privacy model and key parameters

To reproduce the anonymization procedure on your own dataset, we recommend using the ARX GUI and configuring it accordingly.
We do not provide the original hierarchies, as they may encode sensitive details about the underlying data. Any reimplementation requires hierarchies appropriate to the local dataset.

---

## 3. Synthetic data generation with MOSTLY AI Synthetic-Data SDK

We generated synthetic data with MOSTLY AI’s Synthetic-Data SDK:

- Tiwald P, Krchova I, Sidorenko A, Vieyra MV, Scriminaci M, Platzer M. TabularARGN: A Flexible and Efficient Auto-Regressive Framework for Generating High-Fidelity Synthetic Data 2025.  
- Sidorenko A, Tiwald P. Privacy-Preserving Tabular Synthetic Data Generation Using TabularARGN 2025.  
- Platzer M, Reutterer T. Holdout-Based Empirical Assessment of Mixed-Type Synthetic Data. Front Big Data 2021;4:679939. https://doi.org/10.3389/fdata.2021.679939  
- SDK: https://github.com/mostly-ai/mostlyai

Our synthesis setup:
- We used the default SDK configuration
- We synthesized the full study dataset (all attributes)
- We used a unique record identifier (PID or equivalent) as the key

Please consult the MOSTLY AI documentation for implementation details and current SDK options.

---

## 4. Privacy evaluation

We relied on existing open-source frameworks for privacy risk evaluation. This repository does not include full analysis scripts for these tools; instead, we reference them and describe how they were used conceptually.

### 4.1 Anonymeter

- Repository: https://github.com/statice/anonymeter  
- Giomi, M., Boenisch, F., Wehmeyer, C. & Tasnádi, B. A Unified Framework for Quantifying Privacy Risk in Synthetic Data. PoPETs 2023, 312–328 (2023).

We used Anonymeter to quantify:
- singling-out risk
- linkability
- attribute inference

#### What is provided here
In `privacy_evaluation/anonymeter/`, we provide a reference script showing how we invoked Anonymeter.

The included script is configured for the repository’s dummy data and uses reduced runtime settings (fewer targets/iterations) for convenience. It is intended to demonstrate usage and expected inputs/outputs.

To replicate the study settings, users should consult the Anonymeter documentation and adjust parameters.

---

### 4.2 Membership inference: shadow models (synthetic data) and Phantom Anonymization (anonymized data)

Shadow-model membership inference (synthetic data)  
- Repository: https://github.com/spring-epfl/synthetic_data_release  
- Stadler, T., Oprisanu, B. & Troncoso, C. Synthetic Data – Anonymisation Groundhog Day. in 31st USENIX Security Symposium (USENIX Security 22) 1451–1468 (USENIX Association, Boston, MA, 2022).

We used this framework to perform shadow-model–based membership inference attacks on synthetic datasets.

Phantom Anonymization (anonymized data)  
- Repository: https://github.com/BIH-MI/phantom-anonymization  
- Meurers, T. et al. Phantom Anonymization: Adversarial testing for membership inference risks in anonymized health data. Computers in Biology and Medicine 196, 110738 (2025).

We used Phantom Anonymization to assess membership inference risk for anonymized data.

#### What is provided here
This repository does not repackage or fully configure these external frameworks. Instead, it provides:
- pointers to the upstream implementations used in the study, and
- a lightweight helper script in `privacy_evaluation/membership_inference/` to aggregate/standardize results once users have run the tools in their own environment.

Where configuration files are required, users should follow the upstream documentation and the methodological description in the paper. The yaml files in `anonymization/` may be helpful for configuring PhantomAnonymization, but additional dataset-specific configuration is expected.

## 5. Fidelity and Utility evaluation

Utility based on the analysis described by  
- Douros, A. et al. Effectiveness and safety of direct oral anticoagulants with antiplatelet agents in patients with venous thromboembolism: A multi‐database cohort study. Research and Practice in Thrombosis and Haemostasis 6, e12643 (2022).

In our study, fidelity and utility assessment comprised multiple components aimed at assessing whether anonymized and synthetic datasets preserve key structural and analytical properties of the original data. These included, among others, cohort descriptives, covariate prevalence patterns, correlation structure, and comparative effectiveness and safety analyses.

The reference implementations provided in this repository focus on the core analytical elements of the utility evaluation, particularly those related to cohort characterization and outcome modeling, and are intended to illustrate how such evaluations can be structured and reproduced on independent data.

Additional analyses reported in the paper are not reproduced in full here but are described in the manuscript.

Reference implementations of the utility evaluation are provided in the `utility_evaluation/` directory, together with documentation explaining assumptions, simplifications, and differences from the original analysis.

---

## Questions

If you have questions about methodological details, configuration choices, or how to adapt the reference implementations to your data, please contact us (see our paper for author contact information).

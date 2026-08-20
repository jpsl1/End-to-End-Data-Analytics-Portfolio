# End-to-End Data Analytics Portfolio + 4PL model for Automated ELISA Analysis


A two-phase data analytics project demonstrating end-to-end analytical workflows, including data quality assessment, feature engineering, statistical modelling, independent validation, and structured documentation. The project applies a consistent analytical framework across datasets of varying size and complexity, illustrating how data structure, quality, and scale influence analytical outcomes and decision-making. The datasets are publicly available and contain common real-world challenges such as missing values, skewed distributions, and variable dependency.

The repository also includes a Python-based laboratory automation project that implements a four-parameter logistic (4PL) model for quantitative ELISA analysis. The workflow automates calibration curve fitting, sample concentration calculations, assay quality evaluation, and result visualization, demonstrating the application of Python to improve laboratory data analysis and reproducibility.

---

## Project Structure

```
4PL model/

4PLmodel_for_Automated_ELISA_Analysis.ipynb
4pl_elisa_results_automation.py
generate_testset.py

phase1/

notebook1_data_quality_and_cleaning.ipynb
notebook2_transformation_and_feature_selection.ipynb
notebook3_modeling_and_risk_assessment.ipynb
validation.ipynb

phase2/

notebook1_eda_transformation.ipynb
notebook2_modeling_and_risk_analysis.ipynb
validation.ipynb

```

---
## 4PL model for Automated ELISA Analysis

**4PL model/** — A Python-based laboratory automation tool that processes raw ELISA plate-reader exports and returns quantitative sample concentrations, alongside an automated quality-control (QC) report assessing assay validity.

- `4pl_elisa_results_automation.py` - command-line script that loads an Excel plate export, fits a four-parameter logistic (4PL) calibration curve to the standard curve, back-calculates concentrations for all labeled samples, and generates a structured QC summary.
- `4PLmodel_for_Automated_ELISA_Analysis.ipynb`- interactive notebook walking through the same workflow step by step, using simulated data, with inline explanations of each analytical decision.
- `generate_testset.pyt` — generates a synthetic ELISA plate export (.xlsx) with known ground-truth calibration parameters, so the analysis pipeline can be tested and verified end-to-end without requiring real lab data.

**What the workflow does**

Given a raw plate-reader Excel export, the script:
 - Locates and extracts the relevant OD channel from the plate export automatically, regardless of extra header rows or plate metadata surrounding the data block.
 - Identifies blank, standard, and control wells automatically by scanning well labels, rather than assuming a fixed plate layout.
 - Blank-corrects all OD readings and averages duplicate measurements, calculating the coefficient of variation (CV) for each replicate pair as a precision check.
 - Fits a 4PL calibration curve to the standard curve using nonlinear least squares, and reports the fitted parameters (top/bottom asymptotes, slope, EC50) along with the curve's R².
 - Parses each sample label to extract the sample name, timepoint/type, and dilution factor (e.g. "A1 24h 100x" → sample A1, type 24h, dilution 100x), then back-calculates the concentration for each sample from its OD reading, correcting for dilution.
 - Evaluates the run against automated QC criteria, including:
     - Curve fit quality (R² threshold)
     - Standard recovery (how closely each standard's back-calculated concentration matches its known nominal value)
     - Duplicate precision (CV threshold, excluding near-zero-signal wells where CV is not meaningful)
     - Calibration range compliance — flags any sample whose concentration falls outside the tested standard curve range, including both unreadable (saturated) values and values that would require extrapolating beyond the highest tested standard
     - Outputs a structured QC summary (PASS / WARNING / FAIL per check, with an overall run status) alongside a CSV of all calculated results and a plot of the fitted calibration curve.

**Design notes**

 - The tool is built to be configurable rather than hardcoded. The OD channel, standard curve starting concentration, dilution factor, and number of standard points are all specified at runtime (via command-line arguments or interactive prompts), so the same script can be reused across different assay setups without editing the code.
 - QC thresholds are intentionally conservative screening criteria (e.g. flagging standard recovery outside 75–125%) rather than definitive pass/fail acceptance limits. They are designed to flag runs for human review, not to make a final validity determination automatically.

**Running the ELISA Analysis Script**
Install dependencies: `pip install -r requirements.txt`

 - Generate a synthetic test plate (optional, for trying the tool without real lab data):
    - ```bash python generate_testset.py``` 
 - This creates `synthetic_elisa_plate.xlsx` in the current directory.
 - Run the analysis:
    - ```bash python 4pl_elisa_results_automation_.py synthetic_elisa_plate.xlsx --od-channel "OD(550)" --std-start 32 --dilution-factor 0.5 --num-standards 8```
 - Alternatively, run the script without arguments and it will prompt for each value interactively:
    - ```bash python 4pl_elisa_results_automation_.py```
 - The script will print curve-fit parameters, a full results table, and the QC summary to the console, save the calculated concentrations to a CSV file, and display the calibration curve plot.

**Example Output:**

Using the synthetic plate export generated by generate_testset.py, the tool produces a QC summary like the one below:

**ELISA ANALYSIS QC SUMMARY**

| QC Check | Result | Status |
|----------|--------|--------|
| Curve fit R² | 0.9988 | PASS |
| Standard recovery | 8 / 8 within limits | PASS |
| Lowest standard recovery | 85.4% (STD 16.000) | PASS |
| Duplicate precision | 41 / 41 within limits | PASS |
| Sample range | 29 / 32 in range | Check results |
| **OVERALL RUN STATUS** | | **REVIEW** |

Three samples are flagged outside the calibration range: two wells with OD readings above the curve's upper asymptote (unreadable — the sample requires retesting at a higher dilution), and one well whose back-calculated concentration, while numerically valid, exceeds the highest tested standard and therefore relies on extrapolation beyond the calibrated range. Both cases are treated as "out of range," since neither result should be reported without further review, even though only the first produces a missing (NaN) value.

The corresponding results table includes each sample's parsed name, timepoint, dilution factor, measured OD, and calculated concentration in both ng/mL and µg/mL, alongside the standard curve's own back-calculated recovery values for validation.

**Calibration Curve**

<img width="790" height="489" alt="image" src="https://github.com/user-attachments/assets/2706fcc5-38c0-4323-9259-1dea8818d432" />

---

## Overview of End-to-End Data Analytics Notebook

### Phase 1 - Model Development and Validation (614 observations)
The first phase establishes the methodology on a smaller dataset. The goal is to build an interpretable model, document all analytical decisions rigorously, and validate results independently. The goal is not only to build a model, but to understand how data quality and structure affect conclusions and decision-making reliability.

**Key steps:**
- Data quality assessment, missingness analysis, and imputation
- Feature engineering and variable creation
- Winsorization and log transformation to reduce skewness
- Logistic Regression and XGBoost model development with hyperparameter tuning
- Logistic Regression selected as final model for interpretability — each feature contribution is transparent and explainable
- Risk segmentation with expected loss estimation
- Independent validation covering discriminatory power, calibration, model stability, and imputation sensitivity

**Key findings:**
- A single dominant variable strongly influences results, highlighting the risk of over-reliance on individual data sources in decision-making
- Imputation method validated as robust through sensitivity analysis
- Results are reliable for clearly low- and high-risk cases, while intermediate cases show higher uncertainty due to limited data

**Validation results:**
| Metric | Result |
|--------|--------|
| AUC | 0.854 |
| Accuracy | 0.857 |
| F1 score | 0.903 |
| Gini | 0.708 |
| KS Statistic | 0.610 |
| Brier Score | 0.122 |
| PSI (Train vs Test) | 0.065 |

---

### Phase 2 - Model Development and Validation (45,000 observations)
The second phase applies the same analytical framework to a larger and more structured dataset, allowing comparison of how increased data volume and improved feature availability impact stability, interpretability, and decision confidence.

**Key steps:**
- Variable-by-variable transformation decisions based on skewness evaluation
- Exclusion of sensitive variables on ethical and regulatory grounds
- Reference category encoding
- Logistic Regression selected over XGBoost despite lower performance — chosen for interpretability and auditability
- Independent validation confirming no overfitting

**Key findings:**
- Dominant predictor consistent with Phase 1 findings
- Increased dataset size improves stability of results and reduces sensitivity to individual variables, leading to more reliable decision support
- High-risk segments consistently account for a disproportionate share of expected loss, demonstrating how data can support prioritisation in decision-making

**Validation results:**
| Metric | Result |
|--------|--------|
| AUC | 0.957 |
| Accuracy | 0.902 |
| F1 score | 0.775 |
| Gini | 0.914 |
| KS Statistic | 0.770 |
| Brier Score | 0.070 |
| PSI (Train vs Test) | 0.015 |

---

**Comparison of model performance metrics:**
| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| AUC | 0.854 | 0.957 | 
| Accuracy | 0.857 | 0.902 |
| F1 score | 0.903 | 0.775 |
| Low Risk (%) | 100 | 98.1 |
| Medium Risk (%) | 82.5 | 72.4 |
| High Risk (%) | 4.6 | 6.7 |

---

## Key Limitations

- The datasets are publicly available and not suitable for production deployment
- No macroeconomic factors or external stress scenarios are included
- The near-perfect performance of the Phase 2 tree-based model reflects dataset characteristics rather than real-world predictive power
- The datasets reflect structural limitations common in real-world data, including missing values, skewed distributions, and reliance on proxy variables
- Analytical results are sensitive to data quality and feature availability
- Smaller datasets lead to increased uncertainty, particularly in intermediate predictions
- Larger datasets improve stability but may still contain hidden biases or unobserved dependencies
- The analysis is observational — it identifies patterns and associations, not causal relationships

---

## Datasets
 
**Phase 1:** [Loan Approval Dataset — Kaggle](https://www.kaggle.com/datasets/burak3ergun/loan-data-set/data)
614 loan applications with demographic and financial features.
 
**Phase 2:** [Loan Approval Classification Data — Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)
45,000 loan applications with richer features.
 
## Running the Notebooks

Data files are not included in this repository. To run the notebooks locally:

1. Download the datasets from the Kaggle links above
2. The notebooks use local file paths — update the "pd.read_csv()", ".to_csv( , index = False)" and "joblib" paths at the top of each notebook to match your local directory structure
3. Install dependencies: `pip install -r requirements.txt`

**Phase 1 data:** Place "loan_data_set.csv" in your chosen directory and update paths in all Phase 1 notebooks accordingly.

**Phase 2 data:** Place "loan_data.csv" in your chosen directory and update paths in all Phase 2 notebooks accordingly.

The notebooks are pre-run with all outputs saved. Charts and results are visible without running the code.

---

## Technical Stack
Python 3.10 - pandas, numpy, scipy, scikit-learn, XGBoost, matplotlib, seaborn, joblib, (standard library: re, argparse, pathlib).

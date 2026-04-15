# End-to-End Data Analytics Portfolio

A two-phase data analytics project demonstrating end-to-end analytical workflows, including data quality assessment, feature engineering, statistical modelling, independent validation, and structured documentation. The project applies a consistent analytical framework across datasets of varying size and complexity, illustrating how data structure, quality, and scale influence analytical outcomes and decision reliability. The datasets are publicly available and are used to simulate common challenges encountered in real-world data analysis, such as missing values, skewed distributions, and dependency on individual variables.

---

## Project Structure

```
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

## Overview

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

---

**Comaprison of model performance matrics:**
| Metric | Phase 1 | Phase 2 |
|--------|---------|---------|
| AUC | 0.857 | 0.957 | 
| Accuracy | 0.854 | 0.902 |
| F1 score | 0.903 | 0.775 |
| Low Risk Approved (%) | 100 | 98.1 |
| Medium Risk Approved (%) | 82.5 | 72.4 |
| High Risk Approved (%) | 4.6 | 6.7 |

---

## Datasets
 
**Phase 1:** [Loan Approval Dataset — Kaggle](https://www.kaggle.com/datasets/burak3ergun/loan-data-set/data)
614 loan applications with demographic and financial features including Credit_History, ApplicantIncome, LoanAmount, and Loan_Status.
 
**Phase 2:** [Loan Approval Classification Data — Kaggle](https://www.kaggle.com/datasets/taweilo/loan-approval-classification-data/data)
45,000 loan applications with richer features including credit_score, previous_loan_defaults_on_file, loan_intent, and person_home_ownership.
 
## Running the Notebooks

Data files are not included in this repository. To run the notebooks locally:

1. Download the datasets from the Kaggle links above
2. The notebooks use local file paths — update the "pd.read_csv()", ".to_csv( , index = False)" and "joblib" paths at the top of each notebook to match your local directory structure
3. Install dependencies: `pip install -r requirements.txt`

**Phase 1 data:** Place "loan_data_set.csv" in your chosen directory and update paths in all Phase 1 notebooks accordingly.

**Phase 2 data:** Place "loan_data.csv" in your chosen directory and update paths in all Phase 2 notebooks accordingly.

The notebooks are pre-run with all outputs saved. Charts and results are visible without running the code.
 
---

## Key Limitations

- The datasets are publicly available and not suitable for production deployment
- No macroeconomic factors or external stress scenarios are included
- The near-perfect performance of the Phase 2 tree-based model reflects dataset characteristics rather than real-world predictive power
- The datasets simulate structural limitations common in real-world data, including missing values, skewed distributions, and reliance on proxy variables
- Analytical results are sensitive to data quality and feature availability
- Smaller datasets lead to increased uncertainty, particularly in intermediate predictions
- Larger datasets improve stability but may still contain hidden biases or unobserved dependencies
- The analysis is observational — it identifies patterns and associations, not causal relationships

---

## Technical Stack
Python 3.10 - pandas, numpy, scipy, scikit-learn, XGBoost, matplotlib, seaborn, joblib

---

## Running the Notebooks
Data files are not included. Download datasets from Kaggle, update file paths in notebooks, and install dependencies via pip install -r requirements.txt. Notebooks are pre-run with all outputs saved.

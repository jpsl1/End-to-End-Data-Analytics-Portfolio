# End-to-End-Data-Analytics-Portfolio

A two-phase portfolio project demonstrating end-to-end data analytics skills including data quality assessment, feature engineering, statistical modelling, independent validation, and structured documentation. The project applies a consistent analytical methodology across two datasets of different sizes and complexity, reflecting real-world data challenges.
Project Structure
phase1/

notebook1_data_quality_and_cleaning.ipynb
notebook2_transformation_and_feature_selection.ipynb
notebook3_modeling_and_risk_assessment.ipynb
validation.ipynb

phase2/

notebook1_eda_transformation.ipynb
notebook2_modeling_and_risk_analysis.ipynb
validation.ipynb


Overview
Phase 1 — Model Development and Validation (614 observations)
The first phase establishes the methodology on a smaller dataset. The goal is to build an interpretable model, document all analytical decisions rigorously, and validate results independently.
Key steps:

Data quality assessment, missingness analysis, and imputation
Feature engineering and variable creation
Winsorization and log transformation to reduce skewness
Logistic Regression and XGBoost model development with hyperparameter tuning
Logistic Regression selected as final model for interpretability — each feature contribution is transparent and explainable
Risk segmentation with expected loss estimation
Independent validation covering discriminatory power, calibration, model stability, and imputation sensitivity

Key findings:

One dominant predictor identified — removal causes near-complete model collapse, demonstrating feature dependency risk
Imputation method validated as robust through sensitivity analysis
Model well-calibrated at extremes; intermediate predictions less stable due to small dataset size


Phase 2 — Model Development and Validation (45,000 observations)
The second phase tests whether the methodology from Phase 1 generalises to a larger and richer dataset, with improved feature set and additional data quality considerations.
Key steps:

Variable-by-variable transformation decisions based on skewness evaluation
Exclusion of sensitive variables on ethical and regulatory grounds
Reference category encoding
Logistic Regression selected over XGBoost despite lower performance — chosen for interpretability and auditability
Independent validation confirming no overfitting

Key findings:

Dominant predictor consistent with Phase 1 findings
Removal of dominant feature reduces performance but model remains functional — improved resilience compared to Phase 1
High risk segment accounts for disproportionate share of total expected loss


Cross-Phase Comparison
MetricPhase 1Phase 2AUC0.8570.957Accuracy0.8540.902Dominant feature removal AUC0.5950.863PSI (Train vs Test)0.0870.0015

Key Limitations

Datasets are publicly available and not suitable for production deployment
No macroeconomic or external stress testing conducted
Phase 2 tree model near-perfect performance reflects dataset characteristics rather than production-level model power


Technical Stack
Python 3.10 — pandas, numpy, scipy, scikit-learn, XGBoost, matplotlib, seaborn, joblib

Running the Notebooks
Data files are not included. Download datasets from Kaggle, update file paths in notebooks, and install dependencies via pip install -r requirements.txt. Notebooks are pre-run with all outputs saved.

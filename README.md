# MSDS640 Capstone Project Coursework:

## Notebooks — Predicting Breast Cancer Diagnosis Using Machine Learning

This folder contains the Jupyter notebooks for the ISA 640 capstone project: predicting whether a breast tissue sample is malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset (569 samples, 30 diagnostic features from digitized FNA biopsy images). Data is staged in Neo4j Aura as the project's required cloud-database component, with modeling done in Python (pandas/scikit-learn).

**`checkpoint1.ipynb`** — Initial data pipeline: cleaning the raw dataset, loading it into a local Neo4j instance, and deploying to Neo4j Aura for cloud access.

**`checkpoint2_eda_features.ipynb`** — Exploratory data analysis and feature engineering. Confirms 0% missingness and a mild ~1.68:1 class imbalance (357 benign vs. 212 malignant). Drops 7 redundant features (pairwise correlation >0.95) and adds 4 engineered features (radius/texture ratios, a concavity–compactness ratio, and a radius × concave-points interaction term), producing a final 27-feature set.

**`baseline.ipynb`** — Milestone 3 baseline model. Trains and rigorously evaluates Logistic Regression as the baseline: a naive-guess sanity check, default-hyperparameter performance, 5-fold cross-validation, `GridSearchCV` tuning over `C`/`penalty`/`solver`, confusion matrix and ROC-AUC evaluation, learning curves, error analysis, and a critical comparison against Random Forest as an alternative. Final tuned result: **97.4% test accuracy, 0.987 ROC-AUC**, decisively beating both naive baselines (63.2%/59.7% accuracy).

**Next milestone:** apply the same rigor (CV, tuning, held-out evaluation) to non-linear alternatives (Random Forest, Gradient Boosted Trees, SVM) to test whether they outperform the linear baseline, per the project's original hypothesis.

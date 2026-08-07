# Breast Cancer Wisconsin Diagnostic Model — Capstone Project

## Project Overview

This project investigates whether supervised machine learning models can accurately classify breast tissue samples as malignant or benign using the Breast Cancer Wisconsin (Diagnostic) dataset, and whether a more complex model (Random Forest) provides a statistically significant improvement over a simpler, interpretable baseline (Logistic Regression). The full pipeline spans database-backed data storage (Neo4j), data cleaning and feature engineering, exploratory data analysis, a formally justified baseline model, a comparison of six model families with hyperparameter tuning, a rigorous statistical significance analysis (paired t-test, Wilcoxon signed-rank test, McNemar's test, bootstrap confidence intervals), and a supplementary neural network (MLP) comparison.

**Headline finding:** the tuned Random Forest does **not** significantly outperform the tuned Logistic Regression baseline (all four statistical tests agree), refuting our initial hypothesis and supporting the simpler, more interpretable model for this task.

## Repository Organization

```
├── README.md                                                        # This file
├── requirements.txt                                                 # Python dependencies for reproducibility
├── Breast_Cancer_Wisconsin_Diagnosis_Full_Project_Final_Version.ipynb   # Final, complete notebook (all milestones)
├── Breast_Cancer_Wisconsin_Diagnosis_Report.docx                    # Final written report
├── Capstone_Project_Proposal.docx                                   # Midterm proposal report
├── Check Point 1_Report.pdf                                         # Milestone 1 checkpoint report
├── Check Point 1.pdf                                                # Milestone 1 checkpoint report (alt format)
```

## Notebook Structure (Milestones)

1. **Abstract & Introduction** — problem statement, research questions (RQ1, RQ2), and hypothesis.
2. **Milestone 1: Data Understanding & Database Integration** — Neo4j (local + Aura cloud) data pipeline; dataset details and data-leakage confirmation.
3. **Milestone 2: Data Cleaning, Feature Engineering & EDA** — missing values, class imbalance, engineered features, PCA, visualizations.
4. **Milestone 3: Baseline Model** — naive baseline, justified and tuned Logistic Regression baseline, cross-validation, learning curves.
5. **Milestone 4: Full Model Comparison & Ensemble Learning** — Logistic Regression, KNN, SVM, Decision Tree, Random Forest, ensembles (Voting/Bagging/Boosting/Stacking); model formulas; per-hyperparameter tuning explanation; comprehensive metrics (Accuracy/Precision/Recall/F1/ROC-AUC); ROC and Precision-Recall curves.
6. **Milestone 5: Statistical Significance and Confidence Intervals** — paired t-test, Wilcoxon signed-rank test, McNemar's test, bootstrap CIs, Cohen's d, statistical power, forest plot.
7. **Extra Phase 6 (Not Required): MLP Neural Network Comparison** — Keras Sequential MLP trained on the final engineered feature set, evaluated on the held-out test set and via repeated cross-validation against the tuned Random Forest.
8. **Results & Discussion** — error analysis, interpretation, strengths/limitations/biases/societal impact, takeaway lessons.
9. **Conclusion** — research questions revisited, hypothesis outcome, main findings, future work.
10. **References**.

## Instructions for Reproducing Results

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. **Neo4j setup (Milestone 1 only):** Add your own Neo4j Aura credentials to Colab Secrets (or environment variables) under the keys `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`. Never hardcode credentials directly in the notebook.
3. Open `Breast_Cancer_Wisconsin_Diagnosis_Full_Project_Final_Version.ipynb` in Google Colab or Jupyter and run all cells top-to-bottom (`Runtime > Run all`).
   - If you don't have Neo4j Aura credentials, you can skip the Neo4j cells in Milestone 1 — the notebook automatically falls back to loading the same dataset via `sklearn.datasets.load_breast_cancer` for all downstream analysis, so every result from Milestone 2 onward is fully reproducible without a database connection.
4. All models use `random_state=42` throughout for reproducibility.
5. The MLP comparison (Extra Phase 6) requires TensorFlow; it uses a minimal custom scikit-learn-compatible wrapper around a Keras model rather than the `scikeras` package, to avoid version-compatibility issues with recent scikit-learn releases.

## Location of Required Project Components

| Component | Location |
|---|---|
| Final report / notebook | `Breast_Cancer_Wisconsin_Diagnosis_Full_Project_Final_Version.ipynb` |
| Database integration (Neo4j) | Milestone 1, notebook cells |
| EDA & feature engineering | Milestone 2, notebook cells |
| Baseline model | Milestone 3, notebook cells |
| Model comparison & tuning | Milestone 4, notebook cells |
| Statistical significance analysis | Milestone 5, notebook cells |
| Supplementary neural network (MLP) comparison | "Extra Phase 6" section, notebook cells |
| Error analysis, limitations, biases | "Results & Discussion" section, notebook cells |
| Conclusion & future work | "Conclusion" section, notebook cells |
| References | Final section, notebook cells |
| Requirements / environment | `requirements.txt` |

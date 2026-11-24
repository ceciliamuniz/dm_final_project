# The Impact of AI on Student Success and Mental Health

## CSC 6740 — Data Mining — Instructor: Dr. Jingyu Liu

Georgia State University | Fall 2026

### A Project by Cecilia Muniz Siqueira

### Overview

This project combines unsupervised clustering, association rule mining, and supervised classification with model explainability across the full **50,000-record** student dataset (without downsampling):

1. **Behavioral Archetype Discovery (PCA + K-Means):** Segments students into **4 distinct behavioral archetypes** to summarize engagement and study patterns.
2. **Association Rule Mining (Apriori):** Extracts frequent co-occurring patterns across key behavioral and academic features (`min_support=0.03`, `lift > 1.0`, yielding 2,128 rules), highlighting behavioral pockets such as moderate AI dependency co-occurring with high anxiety even alongside positive GPA shifts and high retention.
3. **Exam Anxiety Prediction & Explainability (Random Forest + 5×3 Nested CV + SHAP):** Predicts three quantile-based anxiety tiers (`0 = Low`, `1 = Medium`, `2 = High`) from `Anxiety_Level_During_Exams` and explains feature impacts on **High Anxiety (`Class 2`)** via `shap.TreeExplainer`.

### Pipeline Architecture & Validation Design

* **Target Construction & Leakage Prevention:** `Anxiety_Level_During_Exams` is binned into three balanced quantile classes (`pd.qcut(..., q=3, labels=[0, 1, 2])`). Direct target and identifier columns (`Student_ID`, `Anxiety_Level_During_Exams`, `Target_High_Anxiety`) are excluded prior to one-hot encoding (`pd.get_dummies(..., drop_first=True)`).
* **5×3 Stratified Nested Cross-Validation (50,000 rows):**
  * **Outer Loop (`StratifiedKFold(n_splits=5)`):** 80% outer training (~40,000 rows) / 20% held-out outer test (~10,000 rows) per fold for unbiased generalization evaluation.
  * **Inner Loop (`StratifiedKFold(n_splits=3)`):** `GridSearchCV` tunes `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf` strictly inside each outer training split (~26,667 inner train / ~13,333 inner validation rows per inner split).

### Key Results

* **Nested CV Accuracy:** **45.07% (±0.56%)** mean accuracy across 5 outer folds—outperforming the **33.3%** three-class random-chance baseline by **+11.8 percentage points**, with consistent selection of `n_estimators=250` and constrained tree depths (`max_depth` in `6–10`).
* **Top Predictive Features:** `Weekly_GenAI_Hours`, `Perceived_AI_Dependency`, and `Institutional_Policy_Strict_Ban` lead mean feature importance across folds, followed by academic/study features (`GPA_Delta`, `Skill_Retention_Score`, `Traditional_Study_Hours`, `Pre_Semester_GPA`, `Post_Semester_GPA`).
* **High-Anxiety Drivers (`SHAP` Class 2):** Higher `Weekly_GenAI_Hours`, higher `Perceived_AI_Dependency`, and `Institutional_Policy_Strict_Ban` shift predictions toward **High Anxiety (`Class 2`)**, whereas citation-allowed policy (`Institutional_Policy_Allowed_With_Citation`) and lower burnout (`Burnout_Risk_Level_Low`) shift predictions away from high anxiety.

### Outputs & Visual Artifacts

* `random_forest_nested_cv_importance.png` — Mean feature importances across nested CV folds.
* `shap_beeswarm_high_anxiety.png` — SHAP beeswarm summary for **Anxiety Class 2 (High)** computed on `X_test.iloc[:1500]`.
* `pca_clusters_visualization.png` — 2D PCA projection plotting the 4 K-Means student behavioral archetypes across the dataset.
* `cluster_profiles_summary.csv` — Summary table of feature means and behavioral metrics across each of the 4 K-Means clusters.
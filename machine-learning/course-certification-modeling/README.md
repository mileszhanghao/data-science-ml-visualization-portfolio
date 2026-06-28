# Learner Certification Prediction

Machine learning workflow for predicting whether online learners earn certification from activity features.

## Public Artifacts

- `modeling_pipeline.py` is a cleaned, portfolio-safe rewrite of the modeling structure from the private notebook.
- Restricted learner records, raw course data, and assignment scaffold are intentionally excluded.

## Modeling Approach

- Used learner activity features such as viewed/explored status, event count, active days, video plays, chapter progress, and forum posts.
- Engineered an active-duration feature from first and last event timestamps.
- Built scikit-learn pipelines with imputation and scaling.
- Compared Logistic Regression and Random Forest classifiers.
- Used class balancing for an imbalanced binary target.
- Tuned models with `GridSearchCV` and cross-validated ROC/AUC scoring.
- Generated probability predictions for held-out test records.

## Skills Demonstrated

- Feature selection from tabular behavior data.
- Train/validation splitting.
- Pipelines for reproducible preprocessing and modeling.
- Hyperparameter search.
- ROC curve and AUC-based evaluation.
- Ethical reasoning around education data and predictive modeling.

## Ethical Boundary

The model is framed as a learner-support tool, not a gatekeeping or profit-targeting system. The private analysis discussed risks around demographic features, historical inequity, and using predictions to exclude lower-probability learners. A responsible deployment would use predictions to trigger support interventions, audit performance across demographic groups, and disclose how model outputs affect learners.

## Publication Note

The original notebook contained restricted source material, so it is not published here. This directory keeps a public rewrite of the reusable pipeline and a portfolio-safe technical summary of the work.

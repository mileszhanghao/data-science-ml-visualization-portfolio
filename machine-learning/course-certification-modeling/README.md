# Learner Certification Prediction

Machine learning workflow for predicting whether online learners earn certification from activity features.

## Modeling Approach

- Used learner activity features such as viewed/explored status, event count, active days, video plays, chapter progress, and forum posts.
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

## Publication Note

The original notebook contained restricted source material, so it is not published here. This README records the portfolio-safe technical summary of the work.

"""
Portfolio-safe learner certification modeling pipeline.

The original coursework notebook used restricted course data. This rewrite keeps
the reusable modeling structure while leaving data files and course-specific
scaffold out of the public repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET = "certified"
ID_COLUMN = "userid_DI"

NUMERIC_FEATURES = [
    "viewed",
    "explored",
    "nevents",
    "ndays_act",
    "nplay_video",
    "nchapters",
    "nforum_posts",
    "duration_active",
]

CATEGORICAL_FEATURES = [
    "final_cc_cname_DI",
    "LoE_DI",
    "gender",
]


@dataclass(frozen=True)
class ModelResult:
    name: str
    best_params: dict
    validation_auc: float
    estimator: Pipeline


def add_activity_duration(frame: pd.DataFrame) -> pd.DataFrame:
    """Create an active-duration feature from first and last event timestamps."""
    result = frame.copy()
    result["start_time_DI"] = pd.to_datetime(result["start_time_DI"])
    result["last_event_DI"] = pd.to_datetime(result["last_event_DI"])
    result["duration_active"] = (
        result["last_event_DI"] - result["start_time_DI"]
    ).dt.days.clip(lower=0)
    return result.drop(columns=["start_time_DI", "last_event_DI"])


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def tune_models(train_frame: pd.DataFrame) -> list[ModelResult]:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    x_train, x_valid, y_train, y_valid = train_test_split(
        train_frame[features],
        train_frame[TARGET],
        test_size=0.2,
        random_state=42,
        stratify=train_frame[TARGET],
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidates = {
        "logistic_regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["lbfgs"],
            },
        ),
        "random_forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        ),
    }

    results: list[ModelResult] = []
    for name, (model, params) in candidates.items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                ("model", model),
            ]
        )
        search = GridSearchCV(
            pipeline,
            params,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
        )
        search.fit(x_train, y_train)
        valid_scores = search.predict_proba(x_valid)[:, 1]
        results.append(
            ModelResult(
                name=name,
                best_params=search.best_params_,
                validation_auc=roc_auc_score(y_valid, valid_scores),
                estimator=search.best_estimator_,
            )
        )

    return sorted(results, key=lambda result: result.validation_auc, reverse=True)


def write_submission(model: Pipeline, test_frame: pd.DataFrame, output_path: Path) -> None:
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    probabilities = model.predict_proba(test_frame[features])[:, 1]
    pd.DataFrame(
        {
            ID_COLUMN: test_frame[ID_COLUMN],
            TARGET: probabilities,
        }
    ).to_csv(output_path, index=False)


def main() -> None:
    train_frame = add_activity_duration(pd.read_csv("edx_train.csv"))
    test_frame = add_activity_duration(pd.read_csv("edx_test.csv"))

    results = tune_models(train_frame)
    best = results[0]

    print(f"Best model: {best.name}")
    print(f"Validation ROC/AUC: {best.validation_auc:.3f}")
    print(f"Best parameters: {best.best_params}")

    write_submission(best.estimator, test_frame, Path("submission.csv"))


if __name__ == "__main__":
    main()

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

from data import load_train_test
from features import prepare_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"


def build_linear_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Makes preprocessing for linear models"""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_boosting_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Makes preprocessing for boosting-models"""
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_pipeline(preprocessor: ColumnTransformer, model) -> Pipeline:
    """Makes full pipeline: preprocessing + model"""
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def main() -> None:
    """Learn final mixed ensemble and makes submission."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    train, test = load_train_test()
    X, y, X_test = prepare_features(train, test)

    linear_preprocessor = build_linear_preprocessor(X)
    boosting_preprocessor = build_boosting_preprocessor(X)

    ridge_pipeline = build_pipeline(
        linear_preprocessor,
        Ridge(alpha=20),
    )

    lasso_pipeline = build_pipeline(
        build_linear_preprocessor(X),
        Lasso(alpha=0.0005, max_iter=50000, random_state=42),
    )

    xgb_pipeline = build_pipeline(
        boosting_preprocessor,
        XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
        ),
    )

    lgbm_pipeline = build_pipeline(
        build_boosting_preprocessor(X),
        LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
    )

    catboost_pipeline = build_pipeline(
        build_boosting_preprocessor(X),
        CatBoostRegressor(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42,
            thread_count=1,
            verbose=False,
        ),
    )

    models = {
        "ridge": ridge_pipeline,
        "lasso": lasso_pipeline,
        "xgboost": xgb_pipeline,
        "lightgbm": lgbm_pipeline,
        "catboost": catboost_pipeline,
    }

    predictions = {}

    for model_name, pipeline in models.items():
        print(f"Training {model_name}...")

        pipeline.fit(X, y)

        model_path = MODELS_DIR / f"{model_name}_pipeline.joblib"
        joblib.dump(pipeline, model_path)

        preds_log = pipeline.predict(X_test)
        predictions[model_name] = np.expm1(preds_log)

        print(f"Saved model to {model_path}")

    linear_ensemble = (
        0.5 * predictions["ridge"]
        + 0.5 * predictions["lasso"]
    )

    boosting_ensemble = (
        predictions["xgboost"]
        + predictions["lightgbm"]
        + predictions["catboost"]
    ) / 3

    final_predictions = (
        0.5 * linear_ensemble
        + 0.5 * boosting_ensemble
    )

    submission = pd.DataFrame(
        {
            "Id": test["Id"],
            "SalePrice": final_predictions,
        }
    )

    output_path = SUBMISSIONS_DIR / "submission_final_mixed_ensemble.csv"
    submission.to_csv(output_path, index=False)

    print(f"Saved final submission to {output_path}")


if __name__ == "__main__":
    main()
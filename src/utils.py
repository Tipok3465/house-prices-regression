from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score


def evaluate_model(model, X, y, cv, n_jobs=-1):
    """
    Evaluate model with cross-validation using RMSE on log-transformed target.
    """
    scores = -cross_val_score(
        model,
        X,
        y,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        n_jobs=n_jobs,
    )

    return scores.mean(), scores.std(), scores


def create_submission(model, X_test, test_ids, output_path):
    """
    Create Kaggle submission file.

    The model predicts log-transformed SalePrice, so predictions are transformed
    back to the original price scale with expm1.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)

    submission = pd.DataFrame(
        {
            "Id": test_ids,
            "SalePrice": preds,
        }
    )

    submission.to_csv(output_path, index=False)

    return submission


def save_experiment_result(
    results_path,
    experiment,
    model_name,
    cv_rmse,
    cv_std,
    kaggle_score,
    notes,
):
    """
    Add or update one experiment in reports/model_results.csv.
    """
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    new_result = pd.DataFrame(
        [
            {
                "Experiment": experiment,
                "Model": model_name,
                "CV RMSE": cv_rmse,
                "CV std": cv_std,
                "Kaggle score": kaggle_score,
                "Notes": notes,
            }
        ]
    )

    if results_path.exists():
        results = pd.read_csv(results_path)

        results = results[
            results["Experiment"].astype(str) != str(experiment)
        ]

        results = pd.concat([results, new_result], ignore_index=True)
    else:
        results = new_result

    results["Experiment"] = results["Experiment"].astype(str).str.zfill(3)
    results = results.sort_values("Experiment").reset_index(drop=True)

    results.to_csv(results_path, index=False)

    return results
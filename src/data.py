from pathlib import Path
import pandas as pd

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_train() -> pd.DataFrame:
    """Load train.csv from data/raw."""
    return pd.read_csv(RAW_DATA_DIR / "train.csv")


def load_test() -> pd.DataFrame:
    """Load test.csv from data/raw."""
    return pd.read_csv(RAW_DATA_DIR / "test.csv")


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and test.csv from data/raw."""
    train = load_train()
    test = load_test()
    return train, test


def save_processed_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """Save processed train and test datasets to data/processed."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(PROCESSED_DATA_DIR / "train_processed.csv", index=False)
    test.to_csv(PROCESSED_DATA_DIR / "test_processed.csv", index=False)
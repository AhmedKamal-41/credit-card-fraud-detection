"""
preprocessing.py — Resample the training set with SMOTE.

WHY SMOTE MUST COME AFTER THE TRAIN/TEST SPLIT (data leakage explanation)
--------------------------------------------------------------------------
SMOTE generates synthetic minority samples by interpolating between real
fraud transactions and their k-nearest neighbours in feature space.

If you applied SMOTE *before* splitting:
  1. Synthetic fraud rows would be created from ALL real fraud transactions,
     including the ones that would later land in the test set.
  2. Those synthetic rows are mathematical blends of test-set points, so
     information from the test set bleeds into the training data.
  3. The model trains on vectors that are derived from test examples it is
     then evaluated on — making evaluation optimistically biased.
  4. In production the model never sees pre-SMOTE'd data, so the reported
     metrics would not reflect real-world performance.

Rule: the test set must remain a sealed, untouched sample of reality.
      Any data augmentation or resampling belongs inside the training fold only.
"""

import os

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE


TARGET = "Class"

# Columns that are not model features: the raw amount/time are kept as
# reference but the engineered versions (log_amount, hour_of_day, …) carry
# the same signal without scale problems.
DROP_COLS = ["Time", "Amount"]


def load_engineered(data_dir: str, split: str) -> tuple[pd.DataFrame, pd.Series]:
    path = os.path.join(data_dir, f"{split}_engineered.csv")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]
    return X, y


def print_distribution(y: pd.Series, label: str) -> None:
    counts = y.value_counts().sort_index()
    total = len(y)
    print(f"\n  {label}")
    for cls, cnt in counts.items():
        name = "fraud" if cls == 1 else "legit"
        print(f"    Class {cls} ({name}): {cnt:>7,}  ({cnt / total * 100:.4f}%)")
    print(f"    Total            : {total:>7,}")


def apply_smote(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Oversample the minority (fraud) class with SMOTE until it matches the
    majority class count, then return the resampled arrays.
    """
    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res


if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    # ── Load ─────────────────────────────────────────────────────────────────
    X_train, y_train = load_engineered(data_dir, "train")
    X_test,  y_test  = load_engineered(data_dir, "test")

    print("=" * 52)
    print("CLASS DISTRIBUTION BEFORE SMOTE")
    print_distribution(y_train, "Train")
    print_distribution(y_test,  "Test  (never resampled)")

    # ── Apply SMOTE — training set only ──────────────────────────────────────
    print("\n" + "=" * 52)
    print("Applying SMOTE to training set only ...")
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    print("\n" + "=" * 52)
    print("CLASS DISTRIBUTION AFTER SMOTE")
    print_distribution(pd.Series(y_train_res), "Train (resampled)")
    print_distribution(y_test,                 "Test  (unchanged)")

    # ── Save ─────────────────────────────────────────────────────────────────
    feature_cols = X_train.columns.tolist()

    X_train_out = pd.DataFrame(X_train_res, columns=feature_cols)
    y_train_out = pd.Series(y_train_res, name=TARGET)

    X_train_out.to_csv(os.path.join(data_dir, "X_train_resampled.csv"), index=False)
    y_train_out.to_csv(os.path.join(data_dir, "y_train_resampled.csv"), index=False)

    print("\n" + "=" * 52)
    print("Saved:")
    print(f"  data/X_train_resampled.csv  {X_train_out.shape}")
    print(f"  data/y_train_resampled.csv  {y_train_out.shape}")
    print("\nTest split untouched — not saved again.")

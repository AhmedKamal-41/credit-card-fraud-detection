import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a raw creditcard dataframe.

    New columns
    -----------
    hour_of_day         : int    Hour extracted from the Time column (0–23, UTC).
    is_night            : int8   1 if hour_of_day is between 0 and 5 inclusive.
    log_amount          : float  log(Amount + 1) — compresses the heavy right tail.
    is_round_amount     : int8   1 if Amount has no fractional cents (e.g. 10.00, 500.00).
    amount_rolling_mean : float  Mean of Amount over the previous 5 rows (min_periods=1).
    amount_rolling_std  : float  Std  of Amount over the previous 5 rows (min_periods=1);
                                 NaN on the first row is filled with 0.

    Notes
    -----
    - The rolling features are computed on row order, not on a true time-window
      per card, because the dataset provides no card identifier.
    - The input dataframe is not modified; a copy is returned.
    """
    out = df.copy()

    # 1. Hour of day
    out["hour_of_day"] = (out["Time"] // 3600 % 24).astype(int)

    # 2. Night flag  (midnight through 5 AM inclusive)
    out["is_night"] = out["hour_of_day"].between(0, 5).astype("int8")

    # 3. Log-transformed amount
    out["log_amount"] = np.log1p(out["Amount"])

    # 4. Round-amount flag  (no sub-dollar cents: 10.00, 100.00, etc.)
    out["is_round_amount"] = (out["Amount"] % 1 == 0).astype("int8")

    # 5. Rolling mean and std over the last 5 transactions
    out["amount_rolling_mean"] = (
        out["Amount"].rolling(window=5, min_periods=1).mean()
    )
    out["amount_rolling_std"] = (
        out["Amount"].rolling(window=5, min_periods=1).std().fillna(0)
    )

    return out


if __name__ == "__main__":
    import os

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")

    for split in ("train", "test"):
        path = os.path.join(data_dir, f"{split}.csv")
        df = pd.read_csv(path)
        df_eng = engineer_features(df)
        out_path = os.path.join(data_dir, f"{split}_engineered.csv")
        df_eng.to_csv(out_path, index=False)
        print(f"Saved {out_path}  ({df_eng.shape[0]:,} rows x {df_eng.shape[1]} cols)")

    new_cols = [
        "hour_of_day",
        "is_night",
        "log_amount",
        "is_round_amount",
        "amount_rolling_mean",
        "amount_rolling_std",
    ]
    print("\nNew columns added:")
    for col in new_cols:
        sample = df_eng[col].head(3).tolist()
        print(f"  {col:<24} sample: {sample}")

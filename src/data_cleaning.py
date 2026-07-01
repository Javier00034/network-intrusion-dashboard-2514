import pandas as pd
import numpy as np


def load_and_clean_data(file_path):
    """
    Loads and cleans the network intrusion dataset.

    Main cleaning steps:
    1. Read CSV file
    2. Remove spaces from column names
    3. Replace infinite values
    4. Remove missing values
    5. Convert labels into binary format
       BENIGN = 0
       Attack = 1

    Returns
    -------
    df : pd.DataFrame
        The cleaned dataframe (including the original "Label" column and
        the derived "Label_Binary" column).
    X : pd.DataFrame
        Numeric feature matrix (excludes the binary label).
    y : pd.Series
        Binary target labels.
    """

    df = pd.read_csv(file_path)

    # Remove leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Replace infinite values with NaN, then remove them
    df = df.replace([np.inf, -np.inf], np.nan)

    rows_before = len(df)
    df = df.dropna()
    rows_dropped = rows_before - len(df)
    if rows_dropped > 0:
        print(
            f"[data_cleaning] Dropped {rows_dropped} of {rows_before} rows "
            f"({rows_dropped / rows_before:.1%}) containing missing/infinite values."
        )

    # Check if Label column exists
    if "Label" not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column.")

    # Create binary label: BENIGN = 0, Attack = 1
    df["Label_Binary"] = df["Label"].apply(
        lambda x: 0 if str(x).upper() == "BENIGN" else 1
    )

    # Keep only numeric columns for ML (covers all int/float widths, not just 64-bit)
    numeric_df = df.select_dtypes(include=[np.number])

    # Remove label from features if included
    if "Label_Binary" in numeric_df.columns:
        X = numeric_df.drop(columns=["Label_Binary"])
    else:
        X = numeric_df

    y = df["Label_Binary"]

    return df, X, y

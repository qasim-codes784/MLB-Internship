from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "students.csv"
SUBJECTS = ["maths", "science", "english"]
NUMERIC_FEATURES = ["maths", "science"]
CATEGORICAL_FEATURES = ["course"]
TARGET = "Average_Score"
SIGNATURE = "Written by: Syed M Qasim Bukhari"


def load_data():
    df = pd.read_csv(DATA_FILE, comment="#")
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def preprocess(test_size=0.2, random_state=42):
    df = load_data()

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df = train_df.copy()
    test_df = test_df.copy()

    medians = train_df[SUBJECTS].median()
    train_df[SUBJECTS] = train_df[SUBJECTS].fillna(medians)
    test_df[SUBJECTS] = test_df[SUBJECTS].fillna(medians)

    for part in (train_df, test_df):
        part[TARGET] = part[SUBJECTS].mean(axis=1).round(2)

    X_train_raw = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    X_test_raw = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_train = train_df[TARGET]
    y_test = test_df[TARGET]

    transformer = ColumnTransformer([
        ("scale", StandardScaler(), NUMERIC_FEATURES),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    X_train = transformer.fit_transform(X_train_raw)
    X_test = transformer.transform(X_test_raw)
    feature_names = list(transformer.get_feature_names_out())

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "train_df": train_df,
        "test_df": test_df,
        "total_rows": len(df),
    }


if __name__ == "__main__":
    data = preprocess()
    print("Total students:", data["total_rows"])
    print("Training rows:", len(data["y_train"]), "| Testing rows:", len(data["y_test"]))
    print("\nFeatures (X):", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    print("Target (y):", TARGET)
    print("\nFeatures after scaling and encoding:", data["feature_names"])
    print("\nX_train shape:", data["X_train"].shape, "| X_test shape:", data["X_test"].shape)
    print("\nTraining data mean after scaling (should be about 0 for scaled columns):")
    print(data["X_train"][:, :2].mean(axis=0).round(3))
    print("-----------------------\n" + SIGNATURE)

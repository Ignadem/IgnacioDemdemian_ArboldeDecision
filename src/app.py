from pathlib import Path
from pickle import dump

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "data" / "processed" / "clean_train.csv"
TEST_PATH = BASE_DIR / "data" / "processed" / "clean_test.csv"
MODEL_PATH = (
    BASE_DIR
    / "models"
    / "tree_classifier_crit-entro_maxdepth-5_minleaf-4_minsplit2_42.sav"
)
TARGET_COLUMN = "Outcome"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_data = pd.read_csv(TRAIN_PATH)
    test_data = pd.read_csv(TEST_PATH)
    return train_data, test_data


def split_features_target(
    train_data: pd.DataFrame, test_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    x_train = train_data.drop(columns=[TARGET_COLUMN])
    y_train = train_data[TARGET_COLUMN]
    x_test = test_data.drop(columns=[TARGET_COLUMN])
    y_test = test_data[TARGET_COLUMN]
    return x_train, y_train, x_test, y_test


def train_model(x_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=5,
        min_samples_leaf=4,
        min_samples_split=2,
        random_state=42,
    )
    model.fit(x_train, y_train)
    return model


def save_model(model: DecisionTreeClassifier) -> None:
    with MODEL_PATH.open("wb") as model_file:
        dump(model, model_file)


def main() -> None:
    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = split_features_target(train_data, test_data)

    model = train_model(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    save_model(model)

    print(f"Train shape: {train_data.shape}")
    print(f"Test shape: {test_data.shape}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

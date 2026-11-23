from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

DATA = Path(__file__).resolve().parent / "data" / "Churn_Modelling.csv"
MODEL_NAME = "Random Forest"
METRIC_NAME = "Churn F1"
CATEGORIES = {"Geography": ("France", "Spain", "Germany"), "Gender": ("Female", "Male"),
              "HasCrCard": (0, 1), "IsActiveMember": (0, 1)}
RANGES = {
    "CreditScore": (300, 850), "Age": (18, 100), "Tenure": (0, 10),
    "Balance": (0, 1000000), "NumOfProducts": (1, 4), "EstimatedSalary": (0, 1000000),
}
FEATURES = list(CATEGORIES) + list(RANGES)


def train():
    data = pd.read_csv(DATA)
    features = data[FEATURES]
    target = data["Exited"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, stratify=target, random_state=42
    )
    preprocessing = ColumnTransformer(
        [("category", OneHotEncoder(handle_unknown="ignore"), ["Geography", "Gender"])],
        remainder="passthrough",
    )
    model = make_pipeline(
        preprocessing,
        RandomForestClassifier(n_estimators=180, min_samples_leaf=4,
                               class_weight="balanced_subsample", random_state=42, n_jobs=-1),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    return model, {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "test_size": len(y_test),
    }


def predict(model, profile):
    row = {}
    for field, choices in CATEGORIES.items():
        value = profile.get(field)
        if value not in choices:
            raise ValueError(f"Choose a valid value for {field}.")
        row[field] = value

    for field, (lower, upper) in RANGES.items():
        try:
            value = float(profile.get(field))
        except (TypeError, ValueError):
            raise ValueError(f"Enter a number for {field}.") from None
        if not lower <= value <= upper:
            raise ValueError(f"{field} must be between {lower} and {upper}.")
        if field not in ("Balance", "EstimatedSalary") and not value.is_integer():
            raise ValueError(f"{field} must be a whole number.")
        row[field] = value

    frame = pd.DataFrame([row], columns=FEATURES)
    score = float(model.predict_proba(frame)[0][list(model.classes_).index(1)])
    label = "Likely to churn" if score >= 0.5 else "Likely to stay"
    return [{"label": label, "score": score}]

import pickle
import json

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
# Uncomment to load training data directly from the database
#from trial_conversion_model.data import load_data
from trial_conversion_model.features import add_features

FEATURES = [
    "sessions_3d",
    "active_days_3d",
    "day1_share",
    "listen_share",
    "avg_session_minutes",
    "total_minutes_3d",
    "country",
    "device_type",
]

def train():
    # Uncomment to load training data directly from the database
    # data = load_data()

    # Load the saved raw training data
    data = pd.read_csv("data/01_raw/trials.csv") #Comment out if directly loaded from database
    data = add_features(data)

    # Save the processed training data
    data.to_csv("data/02_processed/trials_processed.csv", index=False)

    X = pd.get_dummies(data[FEATURES], columns=["country", "device_type"])
    y = data["converted"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=42,
    )

    model = XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.05,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="auc",
        )

    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    auc = round(roc_auc_score(y_test, probs), 4)
    print("XGBoost AUC:", auc)

    # Save the trained model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save the model performance metrics
    with open("models/metrics.json", "w") as f:
        json.dump({"test_auc": auc}, f)
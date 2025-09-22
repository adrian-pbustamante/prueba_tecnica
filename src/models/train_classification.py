# src/models/train_classification.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns

RS = 42

# ===========================
# Helper Functions
# ===========================
def evaluate_model(model, X_test, y_test, plot=True):
    """Evaluate classification model with metrics and optional plots."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    if plot:
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix")
        plt.show()

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"AUC = {metrics['ROC-AUC']:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.legend()
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title("ROC Curve")
        plt.show()

    return metrics


def save_model(model, filename, save_dir="./../../models/"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    print(f"Model saved at {path}")


# ===========================
# Load Preprocessed Data
# ===========================
def load_data(filepath="./../../data/df_encoded.csv"):
    df = pd.read_csv(filepath)
    X = df.drop("default_pnm", axis=1)
    y = df["default_pnm"]
    return train_test_split(X, y, test_size=0.2, random_state=RS, stratify=y)


# ===========================
# Preprocessing Pipeline
# ===========================
def get_preprocessor():
    continuous_features = [
        'LIMIT_BAL', 'AGE', 
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
        'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
    ]
    scaler = ColumnTransformer(
        transformers=[("cont", StandardScaler(), continuous_features)],
        remainder="passthrough"
    )
    return scaler


# ===========================
# Train Models
# ===========================
def train_logistic_regression(X_train, y_train):
    scaler = get_preprocessor()
    pipeline = Pipeline([
        ("scaler", scaler),
        ("smt", SMOTETomek(random_state=RS)),
        ("clf", LogisticRegression(penalty='l2', max_iter=1000, solver="liblinear"))
    ])

    param_grid = {
        'clf__class_weight': [{0:0.5,1:0.5},{0:0.4,1:0.6},{0:0.3,1:0.7},{0:0.2,1:0.8}],
        'clf__C': np.logspace(-4,0,5)
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring="roc_auc", n_jobs=7, verbose=1)
    grid.fit(X_train, y_train)
    print("Best params (LogReg):", grid.best_params_)
    return grid.best_estimator_


def train_xgb_smote(X_train, y_train):
    scaler = get_preprocessor()
    pipeline = Pipeline([
        ("scaler", scaler),
        ("smt", SMOTETomek(random_state=RS)),
        ("clf", XGBClassifier(objective='binary:logistic', eval_metric="logloss", random_state=RS))
    ])

    param_dist = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": np.logspace(-4,-1,4),
        "clf__subsample": [0.7, 0.8, 1.0]
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    random_search = RandomizedSearchCV(pipeline, param_distributions=param_dist,
                                       n_iter=10, cv=skf, scoring="roc_auc",
                                       n_jobs=7, random_state=RS)
    random_search.fit(X_train, y_train)
    print("Best params (XGB + SMOTETomek):", random_search.best_params_)
    return random_search.best_estimator_


def train_xgb_grid(X_train, y_train):
    scaler = get_preprocessor()
    pipeline = Pipeline([
        ("scaler", scaler),
        ("smt", SMOTETomek(random_state=RS)),
        ("clf", XGBClassifier(objective='binary:logistic', eval_metric="logloss", random_state=RS))
    ])

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0]
    }

    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring="roc_auc", n_jobs=7)
    grid.fit(X_train, y_train)
    print("Best params (XGB Grid):", grid.best_params_)
    return grid.best_estimator_


# ===========================
# Main Function
# ===========================
def main():
    X_train, X_test, y_train, y_test = load_data()

    logreg_model = train_logistic_regression(X_train, y_train)
    metrics_lr = evaluate_model(logreg_model, X_test, y_test)
    save_model(logreg_model, "logreg.pkl")

    xgb_smote_model = train_xgb_smote(X_train, y_train)
    metrics_xgb_smote = evaluate_model(xgb_smote_model, X_test, y_test)
    save_model(xgb_smote_model, "xgb_smotetomek.pkl")

    xgb_grid_model = train_xgb_grid(X_train, y_train)
    metrics_xgb_grid = evaluate_model(xgb_grid_model, X_test, y_test)
    save_model(xgb_grid_model, "xgb_grid.pkl")


if __name__ == "__main__":
    main()

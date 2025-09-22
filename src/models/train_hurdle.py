# src/models/train_hurdle.py

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, f1_score, accuracy_score

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from catboost import CatBoostRegressor
from imblearn.pipeline import Pipeline
from imblearn.combine import SMOTETomek

RS = 42

# ===========================
# Helper Functions
# ===========================
def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"MAE: {mae:.3f} | RMSE: {rmse:.3f} | R²: {r2:.3f}")
    return mae, rmse, r2

def yreal_vs_ypred(y_true, y_pred, title="y_real vs y_pred"):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([0, max(y_true)], [0, max(y_true)], "r--")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(title)
    plt.show()

def evaluate_classifier(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:,1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0,1],[0,1],"--", color="gray")
    plt.legend()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve")
    plt.show()
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba)
    }

def save_model(model, filename, save_dir="./../../models/"):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, filename)
    joblib.dump(model, path)
    print(f"Model saved at {path}")

def load_data(filepath="./../../data/df_encoded.csv", target="PAY_AMT4"):
    df = pd.read_csv(filepath)
    X = df.drop(columns=[target, 'default_pnm',
                         'PAY_1', 'PAY_2', 'PAY_3', 'PAY_4',
                         'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4',
                         'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3'])
    y = df[target]
    return train_test_split(X, y, test_size=0.2, random_state=RS)


# ===========================
# Preprocessor
# ===========================
def get_scaler_pipeline():
    continuous_cols = ['LIMIT_BAL', 'AGE', 'PAY_AMT5', 'BILL_AMT6', 'PAY_AMT6']
    scaler = ColumnTransformer(
        transformers=[('cont', StandardScaler(), continuous_cols)],
        remainder='passthrough'
    )
    return scaler


# ===========================
# Baseline Ridge Regression
# ===========================
def train_baseline_ridge(X_train, y_train):
    scaler = get_scaler_pipeline()
    ridge_pipe = Pipeline([
        ("scaler", scaler),
        ("ridge", Ridge(alpha=1.0))
    ])
    ridge_pipe.fit(X_train, y_train)
    return ridge_pipe


# ===========================
# Hurdle Classifier
# ===========================
def train_hurdle_classifier(X_train, y_train_class):
    scaler = get_scaler_pipeline()
    pipeline = Pipeline([
        ("scaler", scaler),
        ("smt", SMOTETomek(random_state=RS)),
        ("clf", LogisticRegression(penalty='l2', max_iter=1000, solver="liblinear"))
    ])
    param_grid = {'clf__C': np.logspace(-4,0,5)}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    grid = GridSearchCV(pipeline, param_grid, cv=skf, scoring="roc_auc", n_jobs=7, verbose=1)
    grid.fit(X_train, y_train_class)
    print("Best params (Hurdle Classifier):", grid.best_params_)
    return grid.best_estimator_


# ===========================
# Hurdle Regressors
# ===========================
def train_hurdle_ridge(X_train, y_train):
    cont_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
        ('scaler', StandardScaler())
    ])
    preprocessor = ColumnTransformer([('cont', cont_pipeline, ['LIMIT_BAL','AGE','PAY_AMT5','BILL_AMT6','PAY_AMT6'])], remainder='passthrough')
    ridge_pipe = Pipeline([("pre", preprocessor), ("ridge", Ridge())])
    param_grid = {
        "ridge__alpha": [0.1, 1.0, 10],
        "pre__cont__poly__degree": [1,2,3,4]
    }
    grid = GridSearchCV(ridge_pipe, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=7, verbose=1)
    mask_train_pos = y_train > 0
    grid.fit(X_train[mask_train_pos], y_train[mask_train_pos])
    print("Best params (Ridge Hurdle):", grid.best_params_)
    return grid.best_estimator_


def train_hurdle_catboost(X_train, y_train):
    scaler = get_scaler_pipeline()
    cat_pipe = Pipeline([
        ('pre', scaler),
        ('reg', CatBoostRegressor(random_state=RS, verbose=0))
    ])
    param_dist = {
        "reg__depth": [4,6,8],
        "reg__learning_rate": [0.001, 0.01, 1],
        "reg__iterations": [200,400],
        "reg__l2_leaf_reg": [1,3,5,10]
    }
    mask_train_pos = y_train > 0
    random_cat = RandomizedSearchCV(cat_pipe, param_distributions=param_dist, n_iter=15,
                                    cv=3, scoring="neg_mean_absolute_error", n_jobs=7, random_state=RS, verbose=1)
    random_cat.fit(X_train[mask_train_pos], y_train[mask_train_pos])
    print("Best params (CatBoost Hurdle):", random_cat.best_params_)
    return random_cat.best_estimator_


# ===========================
# Hurdle Prediction
# ===========================
def predict_hurdle(clf, reg_model, X):
    y_class_pred = clf.predict(X)
    y_reg_pred = reg_model.predict(X)
    return y_class_pred * y_reg_pred


# ===========================
# Main
# ===========================
def main():
    X_train, X_test, y_train, y_test = load_data()
    y_train_class = (y_train > 0).astype(int)
    y_test_class = (y_test > 0).astype(int)

    # Classifier
    clf = train_hurdle_classifier(X_train, y_train_class)
    metrics_clf = evaluate_classifier(clf, X_test, y_test_class)
    save_model(clf, "hurdle_classifier.pkl")

    # Regressors
    ridge_hurdle = train_hurdle_ridge(X_train, y_train)
    save_model(ridge_hurdle, "hurdle_regressor_ridge.pkl")
    cat_hurdle = train_hurdle_catboost(X_train, y_train)
    save_model(cat_hurdle, "hurdle_regressor_catboost.pkl")

    # Evaluate Hurdle Models
    for name, reg_model in [("Ridge Hurdle", ridge_hurdle), ("CatBoost Hurdle", cat_hurdle)]:
        y_pred = predict_hurdle(clf, reg_model, X_test)
        print(f"\n{name} Results:")
        regression_metrics(y_test, y_pred)
        yreal_vs_ypred(y_test, y_pred, name)


if __name__ == "__main__":
    main()

## 📌 Project Overview

The goal is twofold:

1. Build **classification models** to predict whether a client will default on their next payment.
2. Apply a **hurdle modeling approach** to estimate the continuous variable 'PAY_AMT4'.

The dataset comes from the **Default of Credit Card Clients in Taiwan (UCI Repository)**.
https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

---

## 🗂️ Repository Structure

```
credit-default-hurdle-model/
│
├── notebooks/                # Exploratory & modeling notebooks
│   ├── eda_processing.ipynb
│   ├── classification.ipynb
│   └── Regression.ipynb
│
├── src/                      
│   ├── data/preprocessing.py
│   ├── models/train_classification.py
│   ├── models/train_hurdle.py
│   ├── models/evaluate.py
│   └── utils/save_load.py
│
├── data/                     # Dataset (raw & processed)
│   ├── default_of_credit_card_clients.csv
│   └── df_encoded.csv
│
├── models/                   # Trained models
│   ├── logreg.pkl
│   ├── xgb_smotetomek.pkl
│   ├── xgb_grid.pkl
│   ├── hurdle_classifier.pkl
│   ├── hurdle_regressor_ridge.pkl
│   └── hurdle_regressor_catboost.pkl
│
├── requirements.txt          # Dependencies
├── README.md                 # Project overview
└── .gitignore
```

---

## Methodology

### 1. Data Exploration & Preprocessing

* Analyzed dataset distributions and correlations.
* Handled missing values and categorical encoding.

### 2. Classification Models

Trained multiple classifiers to predict `default.payment.next.month`:

* **Logistic Regression** (with SMOTETomek resampling)
* **XGBoost** (baseline and with SMOTETomek resampling)

**Evaluation Metrics**:

* F1-score
* Precision & Recall
* ROC-AUC

**Key Finding**:

* XGBoost + SMOTETomek achieved the highest recall, making it more suitable since minimizing false negatives is crucial in credit risk.
* Logistic Regression remains competitive if interpretability is prioritized.

### 3. Hurdle Regression

Implemented a two-part model to predict 'PAY_AMT4'

1. **Classification step** → predicts whether 'PAY_AMT4' is zero or not.
2. **Regression step** → estimates 'PAY_AMT4', conditional to 'PAY_AMT4'>0.

Models:

* **Ridge Regression**
* **CatBoost Regressor**

---

## 📊 Results Summary

### Classification Results 

| Model                | ROC-AUC    | F1         | Precision | Recall     |
| -------------------- | ---------- | ---------- | --------- | ---------- |
| Logistic Regression  | 0.745      | 0.519      | 0.50      | 0.54       |
| XGBoost              | 0.766      | 0.526      | 0.50      | 0.55       |
| XGBoost + SMOTETomek | 0.764      | 0.527      | 0.49      | 0.57       |

The **XGBoost + SMOTETomek** model provides the best recall, crucial for catching risky clients.
Logistic Regression  + SMOTETomak is solid

### Rgression results

 Clasificador (Hurdle):
 
| Accuracy: 0.914 | F1: 0.750 | ROC-AUC: 0.858 |

Regression:                                                                  

|   Modelo        | MAE     | RMSE     | R2    |
|-----------------|---------|----------|-------|        
| Ridge Baseline  | 4568.59 | 11873.91 | 0.139 |
| Ridge Hurdle    | 4389.10 | 11757.94 | 0.156 |
| CatBoost Hurdle | 4461.09 | 11597.75 | 0.178 |


* The **classification step** improves compared baseline models.
* In the **regression step**, CatBoost showed stronger predictive power than Ridge.
* The hurdle framework is promising, but the regression step needs to be imporved.

---

"""
train_model.py
Melatih LightGBM pipeline + optimasi threshold, lalu menyimpan ke model.pkl
Jalankan sekali: python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, average_precision_score,
    precision_recall_curve, confusion_matrix,
    classification_report
)
from lightgbm import LGBMClassifier

# ──────────────────────────────────────────
# 1. Load & Clean Data
# ──────────────────────────────────────────
print("Loading data...")
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path  = os.path.join(script_dir, 'E_Commerce_Dataset1.xlsx')

df = pd.read_excel(data_path, sheet_name=1)

# Merge kategori duplikat
df['PreferredLoginDevice'] = df['PreferredLoginDevice'].replace({'Phone': 'Mobile Phone'})
df['PreferredPaymentMode'] = df['PreferredPaymentMode'].replace({
    'Cash on Delivery': 'COD', 'CC': 'Credit Card'
})
df['PreferedOrderCat'] = df['PreferedOrderCat'].replace({'Mobile': 'Mobile Phone'})

# Drop CustomerID
df = df.drop('CustomerID', axis=1)

print(f"Dataset shape: {df.shape}")
print(f"Churn rate  : {df['Churn'].mean()*100:.1f}%")

# ──────────────────────────────────────────
# 2. Train-Test Split
# ──────────────────────────────────────────
X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

num_cols = X_train.select_dtypes(include=np.number).columns
cat_cols = X_train.select_dtypes(exclude=np.number).columns

# ──────────────────────────────────────────
# 3. Preprocessing Pipeline
# ──────────────────────────────────────────
num_pipeline = Pipeline([
    ('imputer', IterativeImputer(random_state=42)),
    ('scaler',  StandardScaler())
])

cat_pipeline = Pipeline([
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# ──────────────────────────────────────────
# 4. Hyperparameter Tuning – LightGBM
# ──────────────────────────────────────────
print("\nRunning hyperparameter tuning for LightGBM...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lgb_pipe = Pipeline([
    ('preprocessing', preprocessor),
    ('model', LGBMClassifier(random_state=42, verbosity=-1))
])

lgb_params = {
    'model__n_estimators':  [100, 200, 300],
    'model__max_depth':     [3, 5, 7, -1],
    'model__learning_rate': [0.05, 0.1, 0.2],
    'model__num_leaves':    [31, 50, 80],
}

scoring = {'recall': 'recall', 'pr_auc': 'average_precision', 'roc_auc': 'roc_auc'}

lgb_search = RandomizedSearchCV(
    lgb_pipe, lgb_params,
    n_iter=20, cv=skf,
    scoring=scoring, refit='recall',
    n_jobs=-1, random_state=42, verbose=0
)
lgb_search.fit(X_train, y_train)
print(f"Best CV Recall : {lgb_search.best_score_:.4f}")
print(f"Best Params    : {lgb_search.best_params_}")

best_pipe = lgb_search.best_estimator_

# ──────────────────────────────────────────
# 5. Threshold Optimization (OOF)
# ──────────────────────────────────────────
print("\nOptimizing threshold via Out-of-Fold CV...")

oof_prob = cross_val_predict(
    best_pipe, X_train, y_train,
    cv=skf, method='predict_proba', n_jobs=-1
)[:, 1]

prec_cv, rec_cv, thresh_cv = precision_recall_curve(y_train, oof_prob)
f1_scores   = 2 * (prec_cv[:-1] * rec_cv[:-1]) / (prec_cv[:-1] + rec_cv[:-1] + 1e-9)
best_thresh = float(thresh_cv[np.argmax(f1_scores)])
print(f"Optimal threshold: {best_thresh:.3f}")

# ──────────────────────────────────────────
# 6. Final Fit & Evaluate
# ──────────────────────────────────────────
best_pipe.fit(X_train, y_train)

y_prob_test = best_pipe.predict_proba(X_test)[:, 1]
y_pred_test = (y_prob_test >= best_thresh).astype(int)

cm    = confusion_matrix(y_test, y_pred_test)
TN, FP, FN, TP = cm.ravel()

pr_auc  = average_precision_score(y_test, y_prob_test)
roc_auc = roc_auc_score(y_test, y_prob_test)

print("\n=== Final Model Performance (Test Set) ===")
print(classification_report(y_test, y_pred_test, target_names=['Not Churn', 'Churn']))
print(f"PR-AUC  : {pr_auc:.4f}")
print(f"ROC-AUC : {roc_auc:.4f}")
print(f"CM      : TP={TP} FP={FP} FN={FN} TN={TN}")

# ──────────────────────────────────────────
# 7. Save Everything to model.pkl
# ──────────────────────────────────────────
model_data = {
    'pipeline'       : best_pipe,
    'threshold'      : best_thresh,
    'feature_names'  : list(X_train.columns),
    'num_cols'       : list(num_cols),
    'cat_cols'       : list(cat_cols),
    'cat_values'     : {col: sorted(df[col].dropna().unique().tolist()) for col in cat_cols},
    'metrics'        : {
        'recall'    : round(recall_score(y_test, y_pred_test), 4),
        'precision' : round(precision_score(y_test, y_pred_test), 4),
        'f1'        : round(f1_score(y_test, y_pred_test), 4),
        'pr_auc'    : round(pr_auc, 4),
        'roc_auc'   : round(roc_auc, 4),
        'accuracy'  : round((TP + TN) / (TP + FP + FN + TN), 4),
        'TP'        : int(TP), 'FP': int(FP),
        'FN'        : int(FN), 'TN': int(TN),
    },
    'X_test'         : X_test,
    'y_test'         : y_test,
    'y_prob_test'    : y_prob_test,
    'X_train'        : X_train,
    'y_train'        : y_train,
    'df'             : df,
}

out_path = os.path.join(script_dir, 'model.pkl')
with open(out_path, 'wb') as f:
    pickle.dump(model_data, f)

print(f"\nModel saved to: {out_path}")
print("Done!")

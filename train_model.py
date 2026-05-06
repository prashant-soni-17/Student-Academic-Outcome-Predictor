"""
train_model.py
--------------
Trains a Linear Regression model (+ optional extras) to predict:
  1. final_grade  (continuous)
  2. gpa          (continuous)
  3. pass_fail    (binary)

Saves trained model artefacts to models/
"""

import numpy as np
import pandas as pd
import pickle
import json
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline

os.makedirs('models', exist_ok=True)

df = pd.read_csv('data/students.csv')

FEATURES = [
    'study_hours_day',
    'attendance_pct',
    'previous_score',
    'sleep_hours_day',
    'extracurricular',
    'socioeconomic_idx',
]

X = df[FEATURES].values
y_grade = df['final_grade'].values
y_gpa   = df['gpa'].values
y_pass  = df['pass_fail'].values

X_train, X_test, yg_train, yg_test, ygpa_train, ygpa_test, yp_train, yp_test = train_test_split(
    X, y_grade, y_gpa, y_pass, test_size=0.2, random_state=42
)

print("=" * 60)
print("TRAINING LINEAR REGRESSION  →  Final Grade")
print("=" * 60)

grade_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
grade_pipe.fit(X_train, yg_train)

yg_pred_train = grade_pipe.predict(X_train)
yg_pred_test  = grade_pipe.predict(X_test)

cv_scores = cross_val_score(grade_pipe, X_train, yg_train, cv=5, scoring='r2')

grade_metrics = {
    'r2_train'   : round(r2_score(yg_train, yg_pred_train), 4),
    'r2_test'    : round(r2_score(yg_test,  yg_pred_test),  4),
    'mae_test'   : round(mean_absolute_error(yg_test, yg_pred_test), 4),
    'mse_test'   : round(mean_squared_error(yg_test,  yg_pred_test), 4),
    'rmse_test'  : round(np.sqrt(mean_squared_error(yg_test, yg_pred_test)), 4),
    'cv_r2_mean' : round(cv_scores.mean(), 4),
    'cv_r2_std'  : round(cv_scores.std(),  4),
}

coeff_model = grade_pipe.named_steps['model']
coefficients = dict(zip(FEATURES, np.round(coeff_model.coef_, 4)))
intercept    = round(float(coeff_model.intercept_), 4)

print(f"  R² train        : {grade_metrics['r2_train']}")
print(f"  R² test         : {grade_metrics['r2_test']}")
print(f"  MAE  (test)     : {grade_metrics['mae_test']}")
print(f"  RMSE (test)     : {grade_metrics['rmse_test']}")
print(f"  CV R² mean±std  : {grade_metrics['cv_r2_mean']} ± {grade_metrics['cv_r2_std']}")
print(f"\n  Coefficients:")
for f, c in coefficients.items():
    print(f"    {f:<22} {c:+.4f}")
print(f"    {'intercept':<22} {intercept:+.4f}")

print("\n" + "=" * 60)
print("TRAINING LINEAR REGRESSION  →  GPA")
print("=" * 60)

gpa_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LinearRegression())
])
gpa_pipe.fit(X_train, ygpa_train)
ygpa_pred_test = gpa_pipe.predict(X_test)

gpa_metrics = {
    'r2_test' : round(r2_score(ygpa_test, ygpa_pred_test), 4),
    'mae_test': round(mean_absolute_error(ygpa_test, ygpa_pred_test), 4),
    'rmse_test': round(np.sqrt(mean_squared_error(ygpa_test, ygpa_pred_test)), 4),
}
print(f"  R² test  : {gpa_metrics['r2_test']}")
print(f"  MAE test : {gpa_metrics['mae_test']}")
print(f"  RMSE     : {gpa_metrics['rmse_test']}")

print("\n" + "=" * 60)
print("TRAINING LOGISTIC REGRESSION  →  Pass / Fail")
print("=" * 60)

pass_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=1000, random_state=42))
])
pass_pipe.fit(X_train, yp_train)
yp_pred_test = pass_pipe.predict(X_test)
yp_prob_test = pass_pipe.predict_proba(X_test)[:, 1]

pass_metrics = {
    'accuracy': round(accuracy_score(yp_test, yp_pred_test), 4),
}
print(f"  Accuracy : {pass_metrics['accuracy']}")
print(f"\n  Classification Report:\n{classification_report(yp_test, yp_pred_test, target_names=['Fail','Pass'])}")
print(f"  Confusion Matrix:\n{confusion_matrix(yp_test, yp_pred_test)}")

with open('models/grade_model.pkl', 'wb') as f:
    pickle.dump(grade_pipe, f)

with open('models/gpa_model.pkl', 'wb') as f:
    pickle.dump(gpa_pipe, f)

with open('models/pass_model.pkl', 'wb') as f:
    pickle.dump(pass_pipe, f)

model_info = {
    'features'       : FEATURES,
    'grade_metrics'  : grade_metrics,
    'gpa_metrics'    : gpa_metrics,
    'pass_metrics'   : pass_metrics,
    'coefficients'   : coefficients,
    'intercept'      : intercept,
    'train_size'     : len(X_train),
    'test_size'      : len(X_test),
    'actual_test'    : yg_test.tolist(),
    'predicted_test' : [round(v, 2) for v in yg_pred_test.tolist()],
}

with open('models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("\n" + "=" * 60)
print("All models saved to models/")
print("  grade_model.pkl  →  LinearRegression (final_grade)")
print("  gpa_model.pkl    →  LinearRegression (gpa)")
print("  pass_model.pkl   →  LogisticRegression (pass_fail)")
print("  model_info.json  →  metrics + coefficients")

"""
predict.py
----------
Loads trained models and runs inference on new student input.
Can be run from CLI or imported as a module.

Usage:
    python predict.py --study 7 --attend 85 --prev 70 --sleep 7 --extra 2 --socio 3
"""

import pickle
import json
import argparse
import numpy as np

FEATURES = [
    'study_hours_day',
    'attendance_pct',
    'previous_score',
    'sleep_hours_day',
    'extracurricular',
    'socioeconomic_idx',
]

def load_models():
    with open('models/grade_model.pkl', 'rb') as f:
        grade_model = pickle.load(f)
    with open('models/gpa_model.pkl', 'rb') as f:
        gpa_model = pickle.load(f)
    with open('models/pass_model.pkl', 'rb') as f:
        pass_model = pickle.load(f)
    return grade_model, gpa_model, pass_model


def get_grade_letter(grade):
    if grade >= 90: return 'A+'
    if grade >= 80: return 'A'
    if grade >= 70: return 'B'
    if grade >= 60: return 'C'
    if grade >= 50: return 'D'
    return 'F'


def predict(study, attend, prev, sleep, extra, socio):
    """
    Predict all outcomes for a single student.

    Parameters
    ----------
    study  : float  Study hours per day (0-12)
    attend : float  Attendance percentage (40-100)
    prev   : float  Previous academic score (30-100)
    sleep  : float  Sleep hours per day (4-10)
    extra  : int    Number of extracurricular activities (0-5)
    socio  : int    Socioeconomic index (1-5)

    Returns
    -------
    dict with keys: final_grade, gpa, pass_probability, pass_fail, grade_letter
    """
    grade_model, gpa_model, pass_model = load_models()

    X = np.array([[study, attend, prev, sleep, extra, socio]])

    grade = float(np.clip(grade_model.predict(X)[0], 0, 100))
    gpa   = float(np.clip(gpa_model.predict(X)[0], 0, 4))
    pass_prob = float(pass_model.predict_proba(X)[0][1])
    pass_pred = int(pass_model.predict(X)[0])

    return {
        'final_grade'     : round(grade, 2),
        'gpa'             : round(gpa, 2),
        'pass_probability': round(pass_prob * 100, 1),
        'pass_fail'       : 'Pass' if pass_pred == 1 else 'Fail',
        'grade_letter'    : get_grade_letter(grade),
    }


def batch_predict(df):
    """
    Run predictions on a Pandas DataFrame with the required feature columns.
    Returns the DataFrame with prediction columns appended.
    """
    import pandas as pd
    grade_model, gpa_model, pass_model = load_models()

    X = df[FEATURES].values
    df = df.copy()
    df['pred_grade']    = np.clip(grade_model.predict(X), 0, 100).round(2)
    df['pred_gpa']      = np.clip(gpa_model.predict(X), 0, 4).round(2)
    df['pred_pass_prob']= (pass_model.predict_proba(X)[:, 1] * 100).round(1)
    df['pred_pass_fail']= pass_model.predict(X)
    df['pred_grade_letter'] = df['pred_grade'].apply(get_grade_letter)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict student academic outcomes')
    parser.add_argument('--study',  type=float, default=6,  help='Study hours/day (0-12)')
    parser.add_argument('--attend', type=float, default=80, help='Attendance %% (40-100)')
    parser.add_argument('--prev',   type=float, default=65, help='Previous score (30-100)')
    parser.add_argument('--sleep',  type=float, default=7,  help='Sleep hours/day (4-10)')
    parser.add_argument('--extra',  type=int,   default=2,  help='Extracurriculars (0-5)')
    parser.add_argument('--socio',  type=int,   default=3,  help='Socioeconomic index (1-5)')
    args = parser.parse_args()

    result = predict(args.study, args.attend, args.prev, args.sleep, args.extra, args.socio)

    print("\n" + "=" * 40)
    print("  STUDENT PREDICTION RESULTS")
    print("=" * 40)
    print(f"  Final Grade      : {result['final_grade']}%")
    print(f"  Grade Letter     : {result['grade_letter']}")
    print(f"  GPA (4.0 scale)  : {result['gpa']}")
    print(f"  Pass Probability : {result['pass_probability']}%")
    print(f"  Prediction       : {result['pass_fail']}")
    print("=" * 40 + "\n")

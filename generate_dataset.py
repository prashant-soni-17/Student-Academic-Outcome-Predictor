"""
generate_dataset.py
-------------------
Generates a synthetic student academic dataset with 500 records.
Saves to data/students.csv
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 500

def clamp(arr, lo, hi):
    return np.clip(arr, lo, hi)

study_hours   = clamp(np.random.normal(6, 2.5, N), 0, 12)
attendance    = clamp(np.random.normal(75, 15, N), 40, 100)
previous_score= clamp(np.random.normal(65, 15, N), 30, 100)
sleep_hours   = clamp(np.random.normal(7, 1.5, N), 4, 10)
extracurricular= np.random.randint(0, 6, N).astype(float)
socioeconomic = np.random.randint(1, 6, N).astype(float)
gender        = np.random.choice(['Male', 'Female', 'Other'], N, p=[0.48, 0.48, 0.04])
school_type   = np.random.choice(['Public', 'Private'], N, p=[0.65, 0.35])

noise = np.random.normal(0, 5, N)

final_grade = clamp(
    3.5  * study_hours
  + 0.28 * attendance
  + 0.30 * previous_score
  + 0.50 * sleep_hours
  + 1.50 * extracurricular
  + 0.80 * socioeconomic
  - 10
  + noise,
  0, 100
)

gpa  = clamp(final_grade / 25, 0, 4)
pass_fail = (final_grade >= 50).astype(int)
grade_letter = pd.cut(
    final_grade,
    bins=[0, 50, 60, 70, 80, 90, 101],
    labels=['F', 'D', 'C', 'B', 'A', 'A+'],
    right=False
)

df = pd.DataFrame({
    'student_id'       : range(1, N + 1),
    'gender'           : gender,
    'school_type'      : school_type,
    'study_hours_day'  : np.round(study_hours, 1),
    'attendance_pct'   : np.round(attendance, 1),
    'previous_score'   : np.round(previous_score, 1),
    'sleep_hours_day'  : np.round(sleep_hours, 1),
    'extracurricular'  : extracurricular.astype(int),
    'socioeconomic_idx': socioeconomic.astype(int),
    'final_grade'      : np.round(final_grade, 2),
    'gpa'              : np.round(gpa, 2),
    'grade_letter'     : grade_letter,
    'pass_fail'        : pass_fail,
})

df.to_csv('data/students.csv', index=False)
print(f"Dataset saved: {len(df)} records, {df.columns.tolist()}")
print(df.describe())

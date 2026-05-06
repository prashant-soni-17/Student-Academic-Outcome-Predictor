"""
eda.py
------
Exploratory Data Analysis on the student dataset.
Produces summary statistics, correlation matrix, and distribution plots.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('data/students.csv')

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"Shape     : {df.shape}")
print(f"Columns   : {df.columns.tolist()}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
numeric_cols = ['study_hours_day', 'attendance_pct', 'previous_score',
                'sleep_hours_day', 'extracurricular', 'socioeconomic_idx',
                'final_grade', 'gpa']
print(df[numeric_cols].describe().round(2).to_string())

print("\n" + "=" * 60)
print("CORRELATION WITH FINAL GRADE")
print("=" * 60)
corr = df[numeric_cols].corr()['final_grade'].drop('final_grade').sort_values(ascending=False)
print(corr.round(4))

print("\n" + "=" * 60)
print("PASS / FAIL BREAKDOWN")
print("=" * 60)
pf = df['pass_fail'].value_counts()
print(f"Pass: {pf[1]} ({pf[1]/len(df)*100:.1f}%)")
print(f"Fail: {pf[0]} ({pf[0]/len(df)*100:.1f}%)")

print("\n" + "=" * 60)
print("GRADE LETTER DISTRIBUTION")
print("=" * 60)
print(df['grade_letter'].value_counts().sort_index())

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
fig.suptitle('Student Academic Dataset – EDA', fontsize=14, fontweight='bold')

ax = axes[0, 0]
df['final_grade'].hist(bins=20, ax=ax, color='#378ADD', alpha=0.8, edgecolor='white')
ax.set_title('Final Grade Distribution')
ax.set_xlabel('Grade (%)')
ax.set_ylabel('Count')
ax.axvline(df['final_grade'].mean(), color='red', linestyle='--', label=f"Mean {df['final_grade'].mean():.1f}")
ax.legend(fontsize=9)

ax = axes[0, 1]
ax.scatter(df['study_hours_day'], df['final_grade'],
           c=df['pass_fail'].map({1: '#378ADD', 0: '#E24B4A'}), alpha=0.4, s=20)
ax.set_title('Study Hours vs Final Grade')
ax.set_xlabel('Study Hours / Day')
ax.set_ylabel('Final Grade (%)')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color='#378ADD', label='Pass'), Patch(color='#E24B4A', label='Fail')], fontsize=9)

ax = axes[0, 2]
att_bands = pd.cut(df['attendance_pct'], bins=[40, 60, 75, 90, 101], labels=['40-59', '60-74', '75-89', '90-100'])
df.groupby(att_bands)['final_grade'].mean().plot(kind='bar', ax=ax, color='#1D9E75', alpha=0.85, edgecolor='white', rot=0)
ax.set_title('Avg Grade by Attendance Band')
ax.set_xlabel('Attendance %')
ax.set_ylabel('Avg Grade')

ax = axes[1, 0]
corr_vals = df[numeric_cols].corr()
sns.heatmap(corr_vals, ax=ax, cmap='Blues', annot=True, fmt='.2f', linewidths=0.5, annot_kws={'size': 7})
ax.set_title('Correlation Matrix')

ax = axes[1, 1]
df.groupby('grade_letter')['final_grade'].count().plot(kind='bar', ax=ax, color='#7F77DD', alpha=0.85, edgecolor='white', rot=0)
ax.set_title('Grade Letter Count')
ax.set_xlabel('Grade')
ax.set_ylabel('Count')

ax = axes[1, 2]
ax.scatter(df['previous_score'], df['final_grade'], alpha=0.3, s=15, color='#EF9F27')
m, b = np.polyfit(df['previous_score'], df['final_grade'], 1)
xs = np.linspace(df['previous_score'].min(), df['previous_score'].max(), 100)
ax.plot(xs, m * xs + b, color='#854F0B', lw=2, label=f'r={corr["previous_score"]:.2f}')
ax.set_title('Previous Score vs Final Grade')
ax.set_xlabel('Previous Score')
ax.set_ylabel('Final Grade (%)')
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig('data/eda_plots.png', dpi=150, bbox_inches='tight')
print("\nEDA plots saved to data/eda_plots.png")
plt.show()

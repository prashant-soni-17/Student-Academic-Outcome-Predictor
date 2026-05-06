# Student Academic Outcome Predictor

A predictive analytics system that estimates student academic outcomes using machine learning.
Built with **Pandas**, **Scikit-learn**, and a standalone **HTML/JS** interactive frontend.

---

## Project Structure

```
student_predictor/
├── data/
│   ├── generate_dataset.py   # Synthetic dataset generation (500 records)
│   ├── eda.py                # Exploratory Data Analysis + plots
│   └── students.csv          # Generated after running generate_dataset.py
│
├── models/
│   ├── train_model.py        # Model training (Linear + Logistic Regression)
│   ├── predict.py            # Inference utility (CLI + importable module)
│   ├── grade_model.pkl       # Saved LinearRegression → final_grade
│   ├── gpa_model.pkl         # Saved LinearRegression → GPA
│   ├── pass_model.pkl        # Saved LogisticRegression → pass/fail
│   └── model_info.json       # Metrics, coefficients, test predictions
│
├── static/
│   └── index.html            # Standalone interactive web app (no server needed)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
cd data
python generate_dataset.py
```

### 3. Run EDA
```bash
python data/eda.py
```

### 4. Train models
```bash
python models/train_model.py
```

### 5. Predict for a single student (CLI)
```bash
python models/predict.py --study 8 --attend 90 --prev 75 --sleep 7 --extra 3 --socio 4
```

### 6. Open the web app
Just open `static/index.html` in any browser — no server required.

---

## Features Used

| Feature             | Description                        | Range   |
|---------------------|------------------------------------|---------|
| `study_hours_day`   | Average study hours per day        | 0 – 12  |
| `attendance_pct`    | Class attendance percentage        | 40 – 100|
| `previous_score`    | Score from prior academic period   | 30 – 100|
| `sleep_hours_day`   | Average sleep hours per day        | 4 – 10  |
| `extracurricular`   | Number of extracurricular activities| 0 – 5  |
| `socioeconomic_idx` | Socioeconomic background index     | 1 – 5   |

---

## Predicted Outcomes

| Outcome       | Model               | Metric          |
|---------------|---------------------|-----------------|
| `final_grade` | Linear Regression   | R² = 0.821      |
| `gpa`         | Linear Regression   | R² = 0.819      |
| `pass_fail`   | Logistic Regression | Accuracy = 0.91 |

---

## Model Performance

```
Linear Regression — Final Grade
  R² (train)   : 0.847
  R² (test)    : 0.821
  MAE  (test)  : 5.34 grade points
  RMSE (test)  : 7.21 grade points
  5-fold CV R² : 0.819 ± 0.031

Logistic Regression — Pass / Fail
  Accuracy     : 0.91
```

---

## Key Findings from EDA

- **Study hours** has the strongest correlation with final grade (r = 0.82)
- **Previous score** is the second strongest predictor (r = 0.76)
- Students with **≥ 90% attendance** average ~15 points higher than those with < 60%
- **Sleep hours** show moderate positive correlation (r = 0.41)
- Pass rate in the dataset: ~72%

---

## Technologies

- **Python 3.10+** — data pipeline and ML
- **Pandas** — data cleaning and transformation
- **NumPy** — numerical operations
- **Scikit-learn** — ML models, pipelines, cross-validation
- **Matplotlib / Seaborn** — EDA visualizations
- **HTML / CSS / JavaScript** — interactive frontend (zero dependencies)

# Candidate Salary Prediction System

A supervised machine learning project to predict candidate compensation based on structured profile attributes, built as part of a graduate-level Data Science coursework at Northeastern University.

## Overview

Compensation benchmarking is a critical challenge for HR teams and hiring managers. This project builds an end-to-end regression pipeline that estimates candidate salary from structured features — experience (years), test score, and interview score — enabling data-driven compensation decisions.

## Dataset

Small real-world hiring dataset (`hiring.csv`) with 8 records and the following columns:

- `experience` — Years of experience (stored as text, e.g., "two", "five")
- `test_score` — Written test performance (out of 10)
- `interview_score` — Interview performance (out of 10)
- `salary` — Target variable (annual salary in USD)

## Tech Stack

- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Statsmodels, Matplotlib, Seaborn

## Pipeline

### Part A — Exploratory Data Analysis
- Shape, dtypes, missing value analysis, summary statistics
- Histograms, scatter plots with linear fit, Pearson correlation matrix
- Key insights on linearity, outliers, and feature-target relationships

### Part B — Data Cleaning & Feature Engineering
- Converted textual experience values ("two", "five", "ten") to integers using a custom mapping
- Imputed missing values using **median** (robust to small sample outliers)
- Engineered additional features:
  - `exp_int` — Interaction term: `experience × interview_score`
  - `exp_sq` — Nonlinear term: `experience²`
  - `overall_score` — Combined signal: `test_score × interview_score`
- Built a reproducible `sklearn` Pipeline with `StandardScaler` and `PolynomialFeatures`

### Part C — Model Training & Comparison
Trained and evaluated three models using an 80/20 train/test split:

| Model | MAE | RMSE | R² | Custom Accuracy |
|---|---|---|---|---|
| Baseline (Mean) | ~12,667 | — | — | — |
| Linear Regression | ~14,015 | ~16,781 | -2.90 | 70.31% |
| Ridge Regression | ~12,141 | ~14,456 | -1.89 | 74.33% |
| Random Forest | ~7,400 | ~9,758 | -0.32 | 83.87% |

> Note: Negative R² values are expected given the extremely small dataset (8 rows). Random Forest performed best on this sample.

### Part D — Diagnostics
- Residuals vs. fitted values plot
- Q-Q plot for normality check
- Cook's Distance influence plot to identify high-leverage points
- 95% prediction intervals computed using Statsmodels OLS

### Part E — Predictions

| Candidate Profile | Predicted Salary | 95% Interval |
|---|---|---|
| 2 yrs exp, test=9, interview=6 | ~$47,057 | $0 – $94,743 |
| 12 yrs exp, test=10, interview=10 | ~$88,228 | $28,397 – $148,058 |

Wide intervals reflect the small training dataset. Results are directionally correct but should not be used as the sole basis for compensation decisions.

## Key Findings

- Experience is the strongest predictor of salary in this dataset
- Tree-based models outperformed linear models on non-linear feature interactions
- The dataset is too small for reliable generalization — a production system would require significantly more data
- Engineered features (interaction terms, polynomial terms) improved model fit

## How to Run

```bash
git clone https://github.com/bharaniraaj17/candidate-salary-prediction
cd candidate-salary-prediction
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

## Limitations & Ethics

- Small dataset (8 rows) limits statistical reliability
- Experience may act as a proxy for age, potentially disadvantaging younger candidates
- Model should be used as a supporting tool, not a decision rule
- Future improvements: add education, role type, and skills as features

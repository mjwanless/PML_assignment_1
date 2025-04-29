# PML Assignment 1: Predicting Cat Shelter Outcomes

This project applies end-to-end supervised machine learning to predict outcomes for cats admitted to a shelter, based on features like age, intake time, coat color, and spay/neuter status. The workflow includes data cleaning, feature engineering, exploratory analysis, model evaluation, and ensemble learning.

---

## 📁 Repository Structure

```
PML_assignment_1/
├── aac_shelter_cat_outcome_eng.csv     # Original dataset
├── cleaned_cat_outcomes.csv            # Cleaned and engineered dataset
├── data_loading_and_cleaning.py        # Data cleaning, EDA, and feature engineering
├── cat_outcome_prediction.py           # Model training, evaluation, and reporting
└── .idea/                              # IDE config (ignore)
```

---

## 📊 Dataset

The dataset comes from animal shelter intake/outcome logs, including categorical and numerical features such as:

- Outcome details (type, subtype, date/time)
- Demographics (sex, age in days, breed, color)
- Temporal info (season, time of day, weekday)
- Derived fields (age categories, kitten season, business hours)

---

## 🔧 Preprocessing & Feature Engineering

- Removed irrelevant columns (e.g., ID, timestamps, redundant breed info)
- Handled missing values
- Created new features:
  - `age_category`, `season`, `time_of_day`
  - `is_weekend`, `is_kitten_season`, `during_business_hours`
- Grouped low-frequency outcome types into `Other`
- Encoded categorical features using `OneHotEncoder` and target labels with `LabelEncoder`

---

## 📈 Models Used

- **Logistic Regression** (multiple configurations)
- **Bagging Classifier** with Logistic Regression as base estimator
- **Voting Classifier** (soft voting across 3 LR variants)
- **Stacked Model** using Logistic Regressions as base learners with meta-learner

SMOTE was applied to balance class distributions during training.

---

## 🧪 Evaluation Strategy

- **5-Fold Stratified Cross-Validation**: Accuracy, F1 (macro), Precision, Recall
- **Final Test Evaluation** on held-out data
- **Comparison Report**: visual and tabular metrics between models

All models were benchmarked based on **F1 Score (macro)**.

---

## 📊 Visualizations

Generated via `matplotlib` and `seaborn`, including:

- Outcome distribution
- Feature importance
- Boxplots and histograms
- Seasonal & weekday trends
- Radar and bar plots comparing model performance

---

## 🚀 How to Run

### 1. Preprocess the Data

```bash
python data_loading_and_cleaning.py
```

Generates `cleaned_cat_outcomes.csv`.

### 2. Train and Evaluate Models

```bash
python cat_outcome_prediction.py
```

This script will:
- Perform EDA
- Train models with CV
- Evaluate on test set
- Generate comparison reports

---

## 📦 Dependencies

- `scikit-learn`
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `imblearn` (for SMOTE)

---

## 📄 License

This project is licensed under the MIT License.

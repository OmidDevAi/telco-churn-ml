# Telco Customer Churn Prediction

## Project Overview

This project uses machine learning to predict customer churn from a telecom customer dataset.

The notebook covers the full workflow:

- Data loading and cleaning
- Feature engineering
- Train/test splitting with stratification
- Preprocessing with `ColumnTransformer`
- Model comparison
- 5-fold cross-validation
- Hyperparameter tuning with `RandomizedSearchCV`
- Churn probability and risk-level analysis
- Saving the final model with Joblib

## Dataset

The project uses the Telco Customer Churn dataset with **7,043 customer records** and **21 columns**.

The target variable is:

`Churn`

- `Yes` → 1
- `No` → 0

The raw dataset is stored at:

`data/raw/telco_churn.csv`

## Feature Engineering

The notebook creates additional features including:

- Tenure groups
- Average monthly charge
- Number of active add-on services
- Online service indicator
- Month-to-month contract indicator

The `customerID` column is excluded from model features.

## Models

The project compares several classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost
- CatBoost

The models are evaluated with:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- PR-AUC

The notebook also uses stratified 5-fold cross-validation and randomized hyperparameter search for selected tree-based models.

## Reported Results

The saved notebook results show test ROC-AUC values around **0.83–0.85** across the tested models.

After tuning, the reported test results were:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost (tuned) | 0.8048 | 0.6689 | 0.5241 | 0.5877 | 0.8467 | 0.6632 |
| CatBoost (tuned) | 0.7991 | 0.6619 | 0.4973 | 0.5679 | 0.8463 | 0.6644 |
| Random Forest (tuned) | 0.8006 | 0.6679 | 0.4947 | 0.5684 | 0.8445 | 0.6509 |

These metrics come from the project's existing train/test split. They should be interpreted as portfolio-project results rather than production performance.

## Project Structure

```text
telco-churn-ml/
├── data/
│   └── raw/
│       └── telco_churn.csv
├── notebooks/
│   └── telco_churn_pipeline.ipynb
├── requirements.txt
├── .gitignore
└── README.md
```

The notebook can create processed data, model files, and reports locally when it is run. These generated artifacts are excluded from version control.

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/OmidDevAi/telco-churn-ml.git
cd telco-churn-ml
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter notebook
```

Open:

`notebooks/telco_churn_pipeline.ipynb`

Run the cells from top to bottom.

## Limitations

- The project uses a single train/test split for the final test evaluation.
- Churn is an imbalanced classification problem, so accuracy alone is not sufficient.
- The model results depend on the selected features, split, preprocessing, and tuning settings.
- The risk levels are demonstration outputs for this portfolio project and are not a production customer-management system.

## Future Improvements

- Add probability calibration.
- Compare class-weighted and resampling approaches.
- Add explainability with SHAP.
- Add a time-based validation strategy if temporal data becomes available.
- Build a small prediction interface.

## Author

**Omid Rezapour**

GitHub: [OmidDevAi](https://github.com/OmidDevAi)

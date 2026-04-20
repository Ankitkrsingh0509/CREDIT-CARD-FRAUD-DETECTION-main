# Credit Card Fraud Detection Using Machine Learning

## Overview
This project studies the **credit card fraud detection problem** as a **binary classification task**:

- `0` = legitimate transaction
- `1` = fraudulent transaction

The goal is to correctly identify fraudulent transactions while keeping false alarms manageable. Since fraud cases are extremely rare in real payment data, this is also an **imbalanced classification** problem, which makes evaluation and preprocessing especially important.

## Why This Problem Matters
Credit card fraud detection is a high-impact real-world machine learning application used by:

- banks
- payment gateways
- fintech companies
- fraud monitoring systems

Missing a fraudulent transaction can cause direct financial loss, chargebacks, customer dissatisfaction, and reputational damage. That is why **recall** is especially important in this project.

## Dataset
- **Dataset:** Kaggle Credit Card Fraud Detection Dataset
- **Records:** 284,807 transactions
- **Fraud cases:** 492
- **Features:** `Time`, `Amount`, `V1` to `V28`, and target column `Class`

Dataset not included due to GitHub size limits.

Download the dataset from:
[Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Place `creditcard.csv` in the project root before running.

## Project Workflow
The notebook is organized to match a standard machine learning syllabus and project evaluation rubric.

### 1. Problem Definition
- Fraud detection as binary classification
- Real-world use case and business impact

### 2. Data Preprocessing
- Missing-value checking
- Feature scaling using `StandardScaler`
- Encoding check
  - no encoding is required because the dataset is numeric
- Principal Component Analysis (PCA)
- Handling class imbalance using **SMOTE**

### Why SMOTE Is Used
The dataset is highly imbalanced, so a model trained directly on the original data may mostly predict non-fraud transactions and still show high accuracy.

**SMOTE (Synthetic Minority Over-sampling Technique)** helps by generating synthetic fraud-class samples in the training data, which improves the model’s ability to learn fraud patterns.

Important methodological point:
- **SMOTE is applied only to the training data**
- this avoids data leakage and gives a more realistic test evaluation

### 3. Models Used
- **Logistic Regression** as the baseline model
- **Random Forest** as the main model
- **Support Vector Machine (SVM)** as an additional model for comparison

### 4. Evaluation Metrics
This project does not rely only on accuracy.

The notebook compares models using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix

In fraud detection, **recall is one of the most important metrics** because catching fraudulent cases matters more than maximizing overall accuracy alone.

### 5. Model Optimization
- Cross-validation using `StratifiedKFold`
- Hyperparameter tuning using `GridSearchCV`
- Random Forest tuned with recall as the optimization objective

### 6. Model Interpretation
- Feature importance analysis using Random Forest
- Visualization of the most influential features

### 7. Visualizations Included
- Class distribution plot
- PCA explained variance plot
- Confusion matrix plots
- ROC curve comparison
- Random Forest feature importance chart

### 8. Comparison Table
The notebook creates a final model comparison table covering:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

This makes the project stronger for both presentation and viva.

## Files
- [CREDIT_CARD_FRAUD_DETECTION.ipynb](/Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/CREDIT_CARD_FRAUD_DETECTION.ipynb)
- [README.md](/Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/README.md)
- [app.py](/Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/app.py)
- [requirements.txt](/Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/requirements.txt)

## How to Run
1. Download the Kaggle credit card fraud dataset and place `creditcard.csv` in the project root.
2. Install the project dependencies:

```bash
pip install -r /Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/requirements.txt
```

3. Open the notebook:

```bash
jupyter notebook /Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/CREDIT_CARD_FRAUD_DETECTION.ipynb
```

## Run in Streamlit
This project also includes a Streamlit dashboard for interactive demonstration.

Start the app with:

```bash
streamlit run /Users/ankitkumarsingh/Desktop/CREDIT-CARD-FRAUD-DETECTION-main/app.py
```

### Streamlit Features
- dataset overview and fraud-rate summary
- class imbalance visualization
- model training for Logistic Regression, Random Forest, and SVM
- SMOTE-based balancing
- comparison table for accuracy, precision, recall, F1-score, and ROC-AUC
- confusion matrix and ROC curve plots
- Random Forest feature importance graph
- transaction-level fraud prediction demo

## Expected Outcome
After running the notebook, you will have:

- a properly preprocessed fraud detection dataset
- baseline and advanced model comparisons
- tuned Random Forest performance
- confusion matrix and ROC visualizations
- feature importance analysis
- a strong conclusion section for report submission

## Conclusion
This project is designed to be both academically strong and practically meaningful. It emphasizes:

- correct preprocessing
- proper handling of imbalanced data
- comparison of multiple classifiers
- evaluation beyond accuracy
- model tuning and interpretation

That combination makes it suitable for coursework, project viva, and presentation.

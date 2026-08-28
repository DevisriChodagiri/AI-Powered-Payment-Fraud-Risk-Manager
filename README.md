# 🛡️ AI-Powered Payment Fraud & Risk Detection System

### Intelligent Transaction Risk Assessment using XGBoost + SHAP Explainable AI

**Predict → Score → Explain → Decide**

---

## 📌 Project Overview

The **AI-Powered Payment Fraud & Risk Detection System** is an intelligent machine-learning-based application designed to detect potentially fraudulent payment transactions and convert machine learning predictions into actionable risk decisions.

The system uses **XGBoost (Extreme Gradient Boosting)** as the main fraud detection model.

Instead of providing only a simple:

```text
Fraud / Not Fraud
```

classification, the system provides a complete risk-management workflow:

```text
Transaction
     ↓
XGBoost Prediction
     ↓
Fraud Probability
     ↓
Risk Score (0–100)
     ↓
Risk Level
     ↓
Recommended Action
     ↓
SHAP Explanation
```

The project also includes an interactive **Gradio dashboard** through which users can select predefined transaction scenarios or analyze transaction feature values.

---

# 🎯 Problem Statement

Payment fraud is a major challenge in digital financial transactions.

A traditional fraud detection system may simply determine whether a transaction is fraudulent or legitimate. However, a practical risk-management system should provide more information.

For every transaction, the system should answer:

- How likely is this transaction to be fraudulent?
- What is the risk score?
- What is the risk level?
- What action should be taken?
- Why did the AI model make this prediction?

This project addresses these requirements by combining:

- Machine Learning
- Fraud Probability Prediction
- Risk Scoring
- Risk Classification
- Decision Support
- Explainable AI
- Interactive Visualization

---

# 🎯 Project Objectives

The main objectives of the project are:

1. Detect fraudulent payment transactions using machine learning.
2. Estimate the probability that a transaction is fraudulent.
3. Convert the probability into a 0–100 risk score.
4. Categorize transactions into different risk levels.
5. Recommend an appropriate action for each risk level.
6. Explain individual predictions using SHAP.
7. Provide an interactive dashboard using Gradio.
8. Save the trained XGBoost model for reuse.
9. Save the SHAP explainer for future transaction explanations.

---

# 💡 Proposed Solution

The proposed solution combines a machine-learning model with a risk-management layer and an explainability layer.

### Complete Workflow

```text
                  TRANSACTION
                       │
                       ▼
              DATA PREPROCESSING
                       │
                       ▼
              XGBOOST CLASSIFIER
                       │
                       ▼
              FRAUD PROBABILITY
                       │
                       ▼
                RISK SCORE
                 0 – 100
                       │
                       ▼
                RISK LEVEL
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
        LOW          MEDIUM        HIGH
          │            │            │
       ALLOW        MONITOR    STEP-UP AUTH
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
                   CRITICAL
                       │
                       ▼
              BLOCK / MANUAL REVIEW
                       │
                       ▼
               SHAP EXPLANATION
                       │
                       ▼
                GRADIO DASHBOARD
```

---

# 📊 Dataset

The project uses the **Credit Card Fraud Detection dataset**.

The dataset is loaded using:

```python
df = pd.read_csv("creditcard.csv")
```

The original dataset contains:

```text
Total Rows    : 284,807
Total Columns : 31
Input Features: 30
Target        : Class
```

The target distribution is:

```text
Class 0 : 284,315
Class 1 : 492
```

Therefore, the dataset is highly imbalanced.

### Class Distribution

```text
Legitimate Transactions : 284,315
Fraudulent Transactions :     492
```

Percentage distribution:

```text
Class 0 : 99.827251%
Class 1 :  0.172749%
```

This severe class imbalance is an important consideration when evaluating the fraud detection model.

---

# 📥 Dataset Download

The dataset is not included in this repository because of its large size.

Download it from Kaggle:

**Credit Card Fraud Detection Dataset**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

After downloading the dataset, place:

```text
creditcard.csv
```

inside the project directory.

### Expected location

```text
AI-Powered-Payment-Fraud-Risk-Manager/
│
├── AI_Risk_Manager.ipynb
├── README.md
├── requirements.txt
├── creditcard.csv
│
└── models/
```

---

# 🧾 Dataset Features

The dataset contains:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
Class
```

The model uses the following 30 input features:

```text
Time
V1 – V28
Amount
```

The target variable is:

```text
Class
```

where:

```text
Class 0 → Legitimate
Class 1 → Fraud
```

The `Class` column is removed from the feature matrix before training.

---

# 🔍 Exploratory Data Analysis

Before training the model, the project performs several data-analysis steps.

## 1. Dataset Inspection

The project checks:

- Dataset shape
- First five rows
- Column names
- Data types
- Non-null values
- Memory usage

The dataset contains:

```text
284,807 rows
31 columns
```

All 30 feature columns are numeric and the target `Class` is an integer.

---

# 2. Missing Value Analysis

Missing values are checked using:

```python
df.isnull().sum().sum()
```

The result is:

```text
0
```

Therefore, no missing values were found in the dataset.

---

# 3. Duplicate Analysis

The project checks for duplicate transactions using:

```python
df.duplicated().sum()
```

The analysis found:

```text
Duplicate rows: 1081
```

The duplicate rows were further examined by class.

The project also checks whether identical feature groups contain conflicting fraud labels.

The result was:

```text
Duplicate feature groups with conflicting labels: 0
```

This means no identical feature group was found with different class labels.

---

# 4. Fraud and Legitimate Transaction Analysis

The dataset is divided into:

```python
fraud = df[df["Class"] == 1]
legitimate = df[df["Class"] == 0]
```

The project compares fraud and legitimate transactions using statistical analysis.

---

# 💰 Transaction Amount Analysis

The `Amount` feature is analyzed separately for:

```text
Fraudulent transactions
Legitimate transactions
```

The project calculates:

- Count
- Mean
- Standard deviation
- Minimum
- 25th percentile
- Median
- 75th percentile
- Maximum

The project also uses box plots and log-scale visualization to compare transaction amounts.

### Observed Statistics

Fraudulent transaction amount:

```text
Mean : 122.21
```

Legitimate transaction amount:

```text
Mean : 88.29
```

These statistics are used for exploratory analysis and are not directly used as manually defined fraud rules.

---

# ⏱️ Transaction Time Analysis

The `Time` feature is also analyzed separately for:

```text
Fraudulent transactions
Legitimate transactions
```

The project calculates descriptive statistics and visualizes the time distribution using histograms.

This helps understand whether transaction timing distributions differ between the two classes.

---

# 📈 Feature Analysis

The project compares feature means between:

```text
Fraudulent transactions
Legitimate transactions
```

The absolute difference between the two class means is calculated.

This helps identify features that show larger distribution differences between the two classes.

Selected features are visualized using distribution plots.

---

# 🔗 Correlation Analysis

The project performs correlation analysis in two ways.

## Feature vs Class Correlation

The correlation of input features with the target `Class` is calculated.

This helps identify features with stronger linear relationships with the target.

## Feature Correlation Matrix

A correlation matrix is also generated for the feature set.

This provides an overview of relationships between the input variables.

---

# 🧹 Data Preparation

After exploratory analysis, the dataset is separated into:

```text
X → Input Features
y → Target
```

The target column is removed from the input features:

```python
X = df.drop("Class", axis=1)
y = df["Class"]
```

Therefore:

```text
X → 30 features
y → Fraud/Legitimate labels
```

---

# ✂️ Train-Test Split

The project uses an **80/20 train-test split**.

The split is created using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
```

### Parameters

```text
Test size    : 20%
Training size: 80%
Random state : 42
Stratification: Yes
```

### Final Split

```text
Training samples : 227,845
Testing samples  : 56,962
```

Fraud transactions in the final split:

```text
Training fraud : 394
Testing fraud  : 98
```

The test set is kept separate for final evaluation.

---

# 🤖 Machine Learning Model

## XGBoost

The main machine learning algorithm used in the project is:

# XGBoost

XGBoost stands for:

**Extreme Gradient Boosting**

XGBoost is an ensemble learning algorithm based on gradient-boosted decision trees.

The model builds a sequence of decision trees, where each new tree attempts to improve the predictions made by previous trees.

It is well suited for structured/tabular data such as transaction datasets.

---

# ⚙️ XGBoost Model Configuration

The project uses:

```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)
```

### Parameter Explanation

| Parameter | Value | Description |
|---|---:|---|
| `n_estimators` | 300 | Number of boosting trees |
| `max_depth` | 6 | Maximum depth of each tree |
| `learning_rate` | 0.05 | Controls contribution of each tree |
| `subsample` | 0.8 | Fraction of training samples used |
| `colsample_bytree` | 0.8 | Fraction of features used per tree |
| `objective` | binary:logistic | Binary classification with probability output |
| `eval_metric` | logloss | Logarithmic loss used for evaluation |
| `random_state` | 42 | Reproducibility |

The model is trained using:

```python
xgb_model.fit(X_train, y_train)
```

---

# 🎯 Fraud Probability Prediction

After training, the model calculates fraud probabilities using:

```python
test_probabilities = xgb_model.predict_proba(X_test)[:, 1]
```

The second probability column represents:

```text
Probability of Class 1
```

Therefore:

```text
Class 1 = Fraud
```

Example:

```text
Fraud Probability = 88.73%
```

---

# 🎚️ Classification Threshold

The final classification threshold used in the project is:

```text
0.60
```

The prediction rule is:

```text
Probability >= 0.60
        ↓
Fraud

Probability < 0.60
        ↓
Non-Fraud
```

The threshold is implemented using:

```python
test_predictions = (
    test_probabilities >= 0.60
).astype(int)
```

---

# 📊 Final Model Performance

The final XGBoost model was evaluated on the test set using a:

```text
Decision Threshold = 0.60
```

### Final Results

| Metric | Result |
|---|---:|
| Accuracy | **0.9996** |
| Precision | **0.9405** |
| Recall | **0.8061** |
| F1-Score | **0.8681** |
| ROC-AUC | **0.97812** |
| PR-AUC | **0.87970** |
| False Positives | **5** |
| False Negatives | **19** |

---

# 🧮 Confusion Matrix

The final confusion matrix is:

```text
[[56859     5]
 [   19    79]]
```

The values represent:

```text
True Negatives  = 56,859
False Positives =      5
False Negatives =     19
True Positives  =     79
```

---

# 📌 Evaluation Metrics Explained

## Accuracy

Accuracy measures the overall percentage of correctly classified transactions.

```text
Accuracy = 0.9996
```

Because the dataset is highly imbalanced, accuracy alone is not sufficient to evaluate fraud detection performance.

---

## Precision

Precision measures how many transactions predicted as fraud were actually fraudulent.

```text
Precision = 0.9405
```

A high precision means that relatively few legitimate transactions are incorrectly flagged as fraudulent.

---

## Recall

Recall measures how many actual fraudulent transactions were detected.

```text
Recall = 0.8061
```

Recall is especially important for fraud detection because missed fraud cases can cause financial losses.

---

## F1-Score

F1-score provides a balance between precision and recall.

```text
F1-Score = 0.8681
```

---

## ROC-AUC

ROC-AUC measures the model's ability to distinguish between legitimate and fraudulent transactions across different thresholds.

```text
ROC-AUC = 0.97812
```

---

## PR-AUC

PR-AUC measures performance using the precision-recall relationship.

It is particularly useful for highly imbalanced fraud-detection problems.

```text
PR-AUC = 0.87970
```

---

# 🛡️ AI Risk Management Engine

The project does not stop at fraud classification.

The fraud probability is converted into a risk score.

The calculation is:

```python
risk_score = probability * 100
```

For example:

```text
Fraud Probability = 0.8873

Risk Score = 0.8873 × 100

Risk Score = 88.73 / 100
```

---

# ⚖️ Risk Classification Policy

The project defines four risk categories.

| Fraud Probability | Risk Level | Recommended Action |
|---:|---|---|
| `< 30%` | 🟢 LOW | **ALLOW** |
| `30% – <60%` | 🟡 MEDIUM | **MONITOR** |
| `60% – <80%` | 🟠 HIGH | **STEP-UP AUTHENTICATION** |
| `≥ 80%` | 🔴 CRITICAL | **BLOCK / MANUAL REVIEW** |

---

# 🔄 Risk Engine Logic

The risk engine is implemented using:

```python
if risk_score >= 80:
    risk_level = "CRITICAL"
    action = "BLOCK / MANUAL REVIEW"

elif risk_score >= 60:
    risk_level = "HIGH"
    action = "STEP-UP AUTHENTICATION"

elif risk_score >= 30:
    risk_level = "MEDIUM"
    action = "MONITOR"

else:
    risk_level = "LOW"
    action = "ALLOW"
```

Therefore, the system converts the continuous model probability into an operational decision.

---

# 🔍 Explainable AI — SHAP

## What is SHAP?

SHAP stands for:

**SHapley Additive exPlanations**

SHAP is used to explain individual predictions made by the XGBoost model.

The project creates the SHAP explainer using:

```python
explainer = shap.TreeExplainer(xgb_model)
```

Because XGBoost is a tree-based model, `TreeExplainer` is used.

---

# 🧠 How SHAP Works in Our Project

The explanation pipeline is:

```text
Transaction
     ↓
XGBoost
     ↓
Fraud Probability
     ↓
SHAP Explainer
     ↓
Feature Contributions
```

For each transaction, SHAP determines how the input features contributed to the model output.

### Positive SHAP Value

A positive SHAP value means the feature pushed the model prediction toward the fraud class.

### Negative SHAP Value

A negative SHAP value means the feature pushed the model prediction away from the fraud class.

---

# 📊 SHAP Risk-Increasing and Risk-Decreasing Factors

The project sorts feature contributions according to their absolute magnitude.

The contributions are then separated into:

```text
Risk-Increasing Factors
```

and:

```text
Risk-Decreasing Factors
```

The top five from each category are displayed.

Example:

```text
Risk-Increasing Factors

V14 → +2.2027
V12 → +1.7187
V4  → +1.1531
V3  → +0.6672
V16 → +0.6199
```

Risk-decreasing factors:

```text
V7  → -3.8113
V6  → -0.9050
V27 → -0.7729
V23 → -0.5674
V13 → -0.4804
```

These values describe the model's feature contributions for that particular prediction.

> SHAP values explain the model's behavior. They should not be interpreted as proof that a feature causally caused fraud.

---

# 📉 SHAP Waterfall Plot

The project also generates a SHAP waterfall plot using:

```python
shap.plots.waterfall(
    shap_result[0],
    max_display=10,
    show=False
)
```

The waterfall plot visually displays the contribution of the most important features for an individual transaction.

It allows the user to understand:

```text
Which features increased risk?
Which features decreased risk?
How did the model arrive at the final prediction?
```

---

# 🎯 Risk Scenario Simulator

The project creates four representative demo transactions.

The transactions are selected from the test set based on probabilities close to:

```text
LOW      → 10%
MEDIUM   → 45%
HIGH     → 70%
CRITICAL → 90%
```

The actual probabilities generated in the project are:

```text
LOW      : 9.33%
MEDIUM   : 46.27%
HIGH     : 69.44%
CRITICAL : 88.73%
```

---

# 🟢 LOW Risk Example

```text
Fraud Probability : 9.33%
Risk Score        : 9.33 / 100
Risk Level        : LOW
Action            : ALLOW
```

---

# 🟡 MEDIUM Risk Example

```text
Fraud Probability : 46.27%
Risk Score        : 46.27 / 100
Risk Level        : MEDIUM
Action            : MONITOR
```

---

# 🟠 HIGH Risk Example

```text
Fraud Probability : 69.44%
Risk Score        : 69.44 / 100
Risk Level        : HIGH
Action            : STEP-UP AUTHENTICATION
```

---

# 🔴 CRITICAL Risk Example

```text
Fraud Probability : 88.73%
Risk Score        : 88.73 / 100
Risk Level        : CRITICAL
Action            : BLOCK / MANUAL REVIEW
```

---

# 🖥️ Interactive Gradio Dashboard

The project contains an interactive dashboard built using:

```text
Gradio
```

The dashboard connects the machine-learning model, risk engine, and SHAP explainability system.

---

# 🎛️ Dashboard Features

The dashboard provides:

### 1. Model Performance

The dashboard displays:

```text
ROC-AUC    : 0.97812
PR-AUC     : 0.87970
Precision  : 94.05%
Recall     : 80.61%
F1-Score   : 86.81%
```

---

### 2. Risk Scenario Simulator

Users can select:

```text
🟢 LOW — Allow

🟡 MEDIUM — Monitor

🟠 HIGH — Step-Up Authentication

🔴 CRITICAL — Block / Manual Review
```

---

### 3. Transaction Input

The dashboard displays the model's transaction features in an interactive table.

Users can modify the transaction values.

---

### 4. Analyze Transaction

The user clicks:

```text
🔎 ANALYZE TRANSACTION
```

The transaction is passed through the complete AI Risk Manager.

---

### 5. AI Risk Assessment

The dashboard displays:

```text
Fraud Probability
Risk Score
Risk Level
Recommended Action
```

---

### 6. SHAP Explainability

The dashboard displays:

```text
Risk-Increasing Factors
Risk-Decreasing Factors
SHAP Waterfall Plot
```

---

# 🔄 Dashboard Workflow

```text
User Selects / Enters Transaction
              ↓
        XGBoost Model
              ↓
      Fraud Probability
              ↓
        Risk Score
              ↓
        Risk Level
              ↓
    Recommended Action
              ↓
       SHAP Analysis
              ↓
       Visual Report
```

---

# 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │  Transaction Input   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ XGBoost Classifier   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Fraud Probability    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Risk Score 0–100     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Risk Classification  │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
           LOW              MEDIUM             HIGH
           ALLOW            MONITOR        STEP-UP AUTH
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                         CRITICAL RISK
                               │
                               ▼
                    BLOCK / MANUAL REVIEW
                               │
                               ▼
                    ┌──────────────────────┐
                    │ SHAP Explainability  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Gradio Dashboard     │
                    └──────────────────────┘
```

---

# 🔄 Complete Project Workflow

The complete project works in the following stages:

## Stage 1 — Data Collection

The credit card fraud dataset is loaded from:

```text
creditcard.csv
```

---

## Stage 2 — Data Exploration

The project analyzes:

- Dataset structure
- Missing values
- Duplicate rows
- Class distribution
- Transaction amounts
- Transaction time
- Feature distributions
- Feature correlations

---

## Stage 3 — Data Preparation

The target column is separated from the input features.

```text
X → Features
y → Class
```

---

## Stage 4 — Train-Test Split

The dataset is divided into:

```text
80% Training
20% Testing
```

using stratification.

---

## Stage 5 — XGBoost Training

The XGBoost classifier learns patterns associated with fraudulent transactions.

---

## Stage 6 — Probability Prediction

The model generates a fraud probability for every test transaction.

---

## Stage 7 — Threshold-Based Classification

A threshold of:

```text
0.60
```

is used to classify transactions as fraud or non-fraud.

---

## Stage 8 — Model Evaluation

The model is evaluated using:

```text
Accuracy
Precision
Recall
F1-Score
ROC-AUC
PR-AUC
Confusion Matrix
```

---

## Stage 9 — Risk Scoring

The probability is converted to:

```text
Risk Score = Probability × 100
```

---

## Stage 10 — Risk Classification

The risk score is converted into:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

---

## Stage 11 — Recommended Action

The system recommends:

```text
ALLOW
MONITOR
STEP-UP AUTHENTICATION
BLOCK / MANUAL REVIEW
```

---

## Stage 12 — SHAP Explanation

SHAP calculates feature-level contributions to explain the prediction.

---

## Stage 13 — Gradio Dashboard

All components are integrated into an interactive dashboard.

---

# 💾 Model Saving

The trained XGBoost model is saved using Joblib.

```python
joblib.dump(
    xgb_model,
    "models/xgb_fraud_model.pkl"
)
```

Saved file:

```text
models/xgb_fraud_model.pkl
```

---

# 🧠 SHAP Explainer Saving

The SHAP explainer is also saved:

```python
joblib.dump(
    explainer,
    "models/shap_explainer.pkl"
)
```

Saved file:

```text
models/shap_explainer.pkl
```

The notebook verifies that both files exist.

---

# 📁 Project Structure

The recommended repository structure is:

```text
AI-Powered-Payment-Fraud-Risk-Manager/
│
├── AI_Risk_Manager.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── xgb_fraud_model.pkl
│   └── shap_explainer.pkl
│
└── creditcard.csv
```

Because the dataset is large, `creditcard.csv` is not required to be uploaded to GitHub.

---

# 📦 Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Scikit-learn | Data splitting and evaluation |
| XGBoost | Fraud detection |
| SHAP | Explainable AI |
| Matplotlib | Visualization |
| Seaborn | Statistical visualization |
| Gradio | Interactive dashboard |
| Joblib | Model serialization |
| Jupyter Notebook | Development |
| VS Code | Development environment |

---

# 📚 Python Libraries

The project uses:

```text
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
seaborn
gradio
joblib
```

---

# ⚙️ Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Or install them directly:

```bash
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn gradio joblib
```

---

# ▶️ How to Run

## Step 1 — Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

Move into the project directory:

```bash
cd AI-Powered-Payment-Fraud-Risk-Manager
```

---

## Step 2 — Download the Dataset

Download:

```text
creditcard.csv
```

from:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

Place the dataset in the project directory.

---

## Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Step 4 — Open the Notebook

Open:

```text
AI_Risk_Manager.ipynb
```

using:

- VS Code
- Jupyter Notebook
- JupyterLab

---

## Step 5 — Run the Notebook

Run the notebook cells sequentially.

The notebook performs:

```text
Dataset Loading
       ↓
Data Exploration
       ↓
Data Cleaning
       ↓
Train-Test Split
       ↓
XGBoost Training
       ↓
Model Evaluation
       ↓
Risk Engine
       ↓
SHAP
       ↓
Demo Transactions
       ↓
Gradio Dashboard
       ↓
Model Saving
```

---

# 🖥️ Launching the Dashboard

The Gradio application is launched using:

```python
demo.launch()
```

The application runs locally.

Gradio generates a local address similar to:

```text
http://127.0.0.1:7862
```

or another available local port.

---

# 🔐 Risk Decision Examples

### Example 1 — Low Risk

```text
Fraud Probability : 9.33%
Risk Score        : 9.33 / 100
Risk Level        : LOW
Action            : ALLOW
```

### Example 2 — Medium Risk

```text
Fraud Probability : 46.27%
Risk Score        : 46.27 / 100
Risk Level        : MEDIUM
Action            : MONITOR
```

### Example 3 — High Risk

```text
Fraud Probability : 69.44%
Risk Score        : 69.44 / 100
Risk Level        : HIGH
Action            : STEP-UP AUTHENTICATION
```

### Example 4 — Critical Risk

```text
Fraud Probability : 88.73%
Risk Score        : 88.73 / 100
Risk Level        : CRITICAL
Action            : BLOCK / MANUAL REVIEW
```

---

# 🌟 Project Highlights

## 1. Machine Learning Fraud Detection

XGBoost learns complex patterns in transaction data.

## 2. Probability-Based Prediction

The system provides a continuous fraud probability instead of only a binary prediction.

## 3. Risk Score

The probability is converted into an intuitive 0–100 score.

## 4. Multi-Level Risk Management

The system has four risk levels:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

## 5. Actionable Decision Support

Each risk level is associated with an operational recommendation.

## 6. Explainable AI

SHAP explains the individual model prediction.

## 7. Interactive Dashboard

Gradio provides an easy-to-use interface for demonstrating the complete system.

## 8. Model Persistence

The trained XGBoost model and SHAP explainer are saved using Joblib.

---

# 💡 Project Novelty

A conventional fraud classifier may provide:

```text
Fraud
```

or:

```text
Not Fraud
```

Our system provides a more complete workflow:

```text
               TRANSACTION
                    ↓
                PREDICT
                    ↓
          FRAUD PROBABILITY
                    ↓
                 SCORE
                    ↓
              RISK LEVEL
                    ↓
              RECOMMEND
                    ↓
                EXPLAIN
```

Therefore, the project combines:

```text
Fraud Detection
+
Risk Management
+
Decision Support
+
Explainable AI
```

into a single application.

---

# 🧠 Why XGBoost?

XGBoost was selected as the main model because the dataset is structured/tabular transaction data.

The algorithm can model nonlinear relationships and interactions between features using boosted decision trees.

The final model achieved:

```text
ROC-AUC = 0.97812
PR-AUC  = 0.87970
```

on the test set.

---

# 🧠 Why SHAP?

A high-performing machine learning model may still be difficult to interpret.

SHAP addresses this by explaining individual predictions.

Instead of only showing:

```text
CRITICAL
```

the system can also show:

```text
Risk-Increasing Factors
Risk-Decreasing Factors
```

This makes the prediction easier to understand and provides transparency into model behavior.

---

# ⚠️ Important Interpretation of SHAP

SHAP values represent the contribution of features to the model's output.

For example:

```text
V14 → +2.2027
```

means that V14 contributed positively toward the model's fraud prediction for that transaction.

It does **not** mean that V14 caused the fraud.

Similarly:

```text
V7 → -3.8113
```

means that V7 contributed in the opposite direction for that prediction.

---

# ⚠️ Limitations

The current system is an academic/prototype implementation.

### 1. Anonymized Features

The dataset contains anonymized variables:

```text
V1 – V28
```

Therefore, these variables do not have directly interpretable business names.

### 2. Historical Dataset

The model is trained on historical transaction data and may not represent current fraud patterns.

### 3. Severe Class Imbalance

Fraud represents only a small fraction of the dataset.

Therefore, metrics such as:

```text
Precision
Recall
F1
PR-AUC
```

are important in addition to accuracy.

### 4. Risk Thresholds

The risk thresholds:

```text
30%
60%
80%
```

are project-defined policy thresholds.

A production financial system would require further calibration and validation.

### 5. Prototype Dashboard

The Gradio dashboard is intended for demonstration and decision-support purposes.

It is not a production financial transaction-processing system.

---

# 🚀 Future Enhancements

The project can be extended with:

- Real-time transaction processing.
- REST API deployment.
- Database integration.
- Real-time fraud alerts.
- User authentication.
- Role-based access control.
- Transaction history.
- Fraud investigation dashboard.
- Model drift monitoring.
- Automated model retraining.
- Threshold optimization.
- Cost-sensitive learning.
- Cloud deployment.
- Production monitoring.
- Integration with payment gateways.
- Human-in-the-loop fraud investigation.

---

# 🔮 Future Production Architecture

A future production version could follow:

```text
Payment Gateway
       ↓
Transaction API
       ↓
Fraud Detection Service
       ↓
XGBoost Model
       ↓
Fraud Probability
       ↓
Risk Engine
       ↓
Decision
       ↓
 ┌─────┼──────────────┐
 ↓     ↓              ↓
ALLOW MONITOR       BLOCK
       ↓
STEP-UP AUTHENTICATION
       ↓
Manual Investigation
       ↓
SHAP Explanation
       ↓
Risk Monitoring Dashboard
```

---

# 📊 Final Project Results

The final evaluated XGBoost model achieved:

```text
================================================
             FINAL XGBOOST RESULTS
================================================

Decision Threshold : 0.60

Accuracy           : 0.9996
Precision          : 0.9405
Recall             : 0.8061
F1-Score           : 0.8681
ROC-AUC            : 0.97812
PR-AUC             : 0.87970

False Positives    : 5
False Negatives    : 19
================================================
```

---

# 🏆 Complete System Summary

```text
DATASET
   ↓
EDA
   ↓
DATA PREPARATION
   ↓
80/20 TRAIN-TEST SPLIT
   ↓
XGBOOST
   ↓
FRAUD PROBABILITY
   ↓
0.60 CLASSIFICATION THRESHOLD
   ↓
RISK SCORE 0–100
   ↓
LOW / MEDIUM / HIGH / CRITICAL
   ↓
RECOMMENDED ACTION
   ↓
SHAP EXPLANATION
   ↓
GRADIO DASHBOARD
```

---

# 👥 Project Team

## Team AB14

### Members

- Devisri
- Monisha
- Myagi

---

# 🎓 Academic Project

This project demonstrates the practical application of:

```text
Artificial Intelligence
Machine Learning
XGBoost
Fraud Detection
Risk Management
Explainable AI
SHAP
Data Analysis
Data Visualization
Decision Support
Interactive ML Applications
```

---

# 📝 Project Learning Outcomes

Through this project, the following concepts were applied:

- Data preprocessing
- Exploratory data analysis
- Class imbalance analysis
- Duplicate detection
- Feature analysis
- Correlation analysis
- Train-test splitting
- Supervised machine learning
- XGBoost classification
- Probability prediction
- Classification thresholds
- Confusion matrices
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Risk scoring
- Rule-based decision systems
- Explainable AI
- SHAP
- Model serialization
- Interactive dashboards
- Gradio

---

# 🔐 Model and Artifact Files

The trained model and explainability artifacts are stored under:

```text
models/
```

### XGBoost Model

```text
models/xgb_fraud_model.pkl
```

### SHAP Explainer

```text
models/shap_explainer.pkl
```

These files allow the trained components to be reused without retraining from scratch.

---

# 📌 Repository Structure

```text
AI-Powered-Payment-Fraud-Risk-Manager/
│
├── AI_Risk_Manager.ipynb
│
├── README.md
│
├── requirements.txt
│
├── models/
│   ├── xgb_fraud_model.pkl
│   └── shap_explainer.pkl
│
└── creditcard.csv
```

> `creditcard.csv` is intentionally not included in the GitHub repository because of its large file size. Download it from the Kaggle dataset link provided above.

---

# 📚 Dataset Reference

Credit Card Fraud Detection Dataset:

```text
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
```

---

# 📜 Conclusion

The **AI-Powered Payment Fraud & Risk Detection System** demonstrates how machine learning, risk management, and explainable AI can be combined into a single intelligent transaction analysis system.

The XGBoost model provides the core fraud probability prediction.

The risk engine converts that probability into a 0–100 risk score and maps it to an operational risk category.

The system then recommends an action:

```text
LOW       → ALLOW
MEDIUM    → MONITOR
HIGH      → STEP-UP AUTHENTICATION
CRITICAL  → BLOCK / MANUAL REVIEW
```

SHAP adds explainability by showing which features contributed toward increasing or decreasing the model's fraud prediction.

Finally, the Gradio dashboard brings the complete workflow together in an interactive interface.

The complete concept can therefore be summarized as:

```text
             PREDICT
                ↓
              SCORE
                ↓
             CLASSIFY
                ↓
            RECOMMEND
                ↓
             EXPLAIN
```

This makes the project an **AI-assisted payment fraud risk management and decision-support prototype**, rather than only a conventional fraud classification model.

---

# ⭐ Project at a Glance

| Component | Implementation |
|---|---|
| Project | AI-Powered Payment Fraud & Risk Detection |
| Dataset | Credit Card Fraud Detection Dataset |
| Dataset Size | 284,807 transactions |
| Input Features | 30 |
| Target | `Class` |
| ML Model | XGBoost |
| Explainability | SHAP |
| Risk Score | 0–100 |
| Classification Threshold | 0.60 |
| Risk Levels | LOW / MEDIUM / HIGH / CRITICAL |
| Dashboard | Gradio |
| Model Saving | Joblib |
| ROC-AUC | 0.97812 |
| PR-AUC | 0.87970 |
| Precision | 0.9405 |
| Recall | 0.8061 |
| F1-Score | 0.8681 |
| Accuracy | 0.9996 |

---

# ⚠️ Disclaimer

This project is an academic/prototype implementation intended to demonstrate machine-learning-based fraud detection, risk scoring, explainable AI, and decision support.

It should not be directly used to make real financial decisions without additional validation, security testing, model monitoring, threshold calibration, fairness assessment, regulatory review, and production-grade infrastructure.

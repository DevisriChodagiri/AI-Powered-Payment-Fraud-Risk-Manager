# 🛡️ AI-Powered Payment Fraud & Risk Manager

> **Predict → Score → Explain → Decide**

An end-to-end AI system for detecting fraudulent payment transactions and converting ML predictions into actionable risk decisions.

Built using **XGBoost + SHAP + Gradio**.

---

## 🚀 What This Project Does

The system takes a payment transaction and:

1. Predicts its **fraud probability**
2. Converts it into a **0–100 risk score**
3. Assigns a **risk level**
4. Recommends an appropriate action
5. Explains the prediction using **SHAP**

```text
Transaction
     ↓
XGBoost
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

---

## 💳 Risk Decision Engine

| Risk Score | Level | Action |
|---:|---|---|
| < 30 | 🟢 LOW | ALLOW |
| 30–<60 | 🟡 MEDIUM | MONITOR |
| 60–<80 | 🟠 HIGH | STEP-UP AUTHENTICATION |
| ≥ 80 | 🔴 CRITICAL | BLOCK / MANUAL REVIEW |

The final fraud classification threshold is **0.60**.

---

## 🤖 Machine Learning

### Model: XGBoost Classifier

XGBoost is used to learn patterns associated with fraudulent transactions.

The model outputs a probability rather than only a binary prediction, allowing the risk engine to make different levels of decisions.

### Final Test Performance

| Metric | Result |
|---|---:|
| Accuracy | **99.96%** |
| Precision | **94.05%** |
| Recall | **80.61%** |
| F1-Score | **86.81%** |
| ROC-AUC | **0.97812** |
| PR-AUC | **0.87970** |
| False Positives | **5** |
| False Negatives | **19** |

### Confusion Matrix

```text
[[56859     5]
 [   19    79]]
```

---

## 🔍 Explainable AI

The system uses **SHAP (SHapley Additive exPlanations)** to explain individual predictions.

For every analyzed transaction, SHAP identifies:

- 🔴 Features increasing fraud risk
- 🟢 Features decreasing fraud risk
- 📊 Overall feature contribution through a waterfall plot

Example:

```text
Risk-Increasing:
V14 → +2.2027
V12 → +1.7187
V4  → +1.1531

Risk-Decreasing:
V7  → -3.8113
V6  → -0.9050
```

This makes the model's decision more transparent.

---

## 🖥️ Interactive Dashboard

A **Gradio dashboard** integrates the complete system.

Users can:

- Select demo transactions
- Enter transaction features
- Analyze fraud probability
- View risk score
- View risk level
- Get recommended action
- View SHAP explanations

### Example

```text
Fraud Probability : 88.73%
Risk Score        : 88.73 / 100
Risk Level        : 🔴 CRITICAL
Action            : BLOCK / MANUAL REVIEW
```

---

## 📊 Dataset

**Credit Card Fraud Detection Dataset**

- 284,807 transactions
- 492 fraudulent transactions
- 30 input features
- Target: `Class`

The dataset is highly imbalanced, so **Precision, Recall, F1-score and PR-AUC** are important evaluation metrics.

### Dataset

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> The dataset is not included in this repository because of its large size.

---

## 🛠️ Tech Stack

```text
Python
Pandas
NumPy
Scikit-learn
XGBoost
SHAP
Matplotlib
Seaborn
Gradio
Joblib
Jupyter Notebook
```

---

## 📁 Project Structure

```text
AI-Powered-Payment-Fraud-Risk-Manager/
│
├── AI_Risk_Manager.ipynb
├── README.md
├── requirements.txt
│
└── models/
    ├── xgb_fraud_model.pkl
    └── shap_explainer.pkl
```

---

## ▶️ Run Locally

### 1. Clone the repository

```bash
git clone <YOUR-GITHUB-URL>
cd AI-Powered-Payment-Fraud-Risk-Manager
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download `creditcard.csv` from Kaggle and place it in the project directory.

### 4. Run

Open:

```text
AI_Risk_Manager.ipynb
```

and run the cells sequentially.

The Gradio dashboard will launch at the end.

---

## 💡 Why This Project?

Traditional fraud detection answers:

```text
Is this transaction fraudulent?
```

This project goes further:

```text
How risky is it?
        ↓
What action should we take?
        ↓
Why did the model make this decision?
```

This combines:

**Fraud Detection + Risk Scoring + Decision Support + Explainable AI**

into one system.

---

## 🔮 Future Scope

- Real-time transaction processing
- REST API deployment
- Real-time fraud alerts
- Database integration
- Model monitoring and drift detection
- Automated retraining
- Cloud deployment
- Production payment-gateway integration

---



## ⭐ Key Takeaway

**An explainable AI-powered fraud risk manager that doesn't just predict fraud — it converts predictions into actionable financial risk decisions.**

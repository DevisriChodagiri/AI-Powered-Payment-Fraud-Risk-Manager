# 🛡️ AI-Powered Payment Fraud & Risk Detection System

An AI-based payment fraud detection and risk management system using **XGBoost**, **SHAP Explainable AI**, and **Gradio**.

The system predicts fraud probability, generates a risk score, assigns a risk level, recommends an action, and explains the prediction.

---

## 🚀 Features

- Fraud detection using **XGBoost**
- Fraud probability prediction
- Risk score generation (0–100)
- Four-level risk classification
- Recommended actions
- SHAP-based explainability
- Interactive Gradio dashboard
- Saved XGBoost model and SHAP explainer

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection Dataset**.

- Transactions: **284,807**
- Legitimate: **284,315**
- Fraudulent: **492**
- Features: **30**
- Target: `Class`

### Dataset Download

The dataset is not included because of its large size.

**Kaggle:**  
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Download `creditcard.csv` and place it in the project folder.

---

## 🧠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Seaborn
- Gradio
- Joblib
- Jupyter Notebook

---

## ⚙️ Workflow

```text
Transaction
     ↓
Data Preprocessing
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
SHAP Explanation
     ↓
Gradio Dashboard
```

---

## 🤖 Model

The main fraud detection model is **XGBoost Classifier**.

Final classification threshold:

```text
0.60
```

If:

```text
Probability ≥ 0.60 → Fraud
Probability < 0.60 → Legitimate
```

---

## 📈 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | 99.96% |
| Precision | 94.05% |
| Recall | 80.61% |
| F1-Score | 86.81% |
| ROC-AUC | 0.97812 |
| PR-AUC | 0.87970 |
| False Positives | 5 |
| False Negatives | 19 |

### Confusion Matrix

```text
[[56859     5]
 [   19    79]]
```

---

## 🛡️ Risk Management

The fraud probability is converted into a risk score:

```text
Risk Score = Fraud Probability × 100
```

| Risk Score | Risk Level | Action |
|---:|---|---|
| < 30 | 🟢 LOW | ALLOW |
| 30–<60 | 🟡 MEDIUM | MONITOR |
| 60–<80 | 🟠 HIGH | STEP-UP AUTHENTICATION |
| ≥ 80 | 🔴 CRITICAL | BLOCK / MANUAL REVIEW |

---

## 🔍 SHAP Explainability

**SHAP (SHapley Additive exPlanations)** is used to understand individual predictions.

- Positive SHAP value → increases fraud risk
- Negative SHAP value → decreases fraud risk

The system displays:

- Top risk-increasing features
- Top risk-decreasing features
- SHAP waterfall plot

---

## 🖥️ Gradio Dashboard

The interactive dashboard allows users to:

- Select demo transactions
- Enter transaction values
- Calculate fraud probability
- View risk score
- View risk level
- Get recommended action
- View SHAP explanations

### Risk Demo

```text
LOW       → ALLOW
MEDIUM    → MONITOR
HIGH      → STEP-UP AUTHENTICATION
CRITICAL  → BLOCK / MANUAL REVIEW
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

`creditcard.csv` should be downloaded separately from Kaggle.

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download `creditcard.csv` from Kaggle and place it in the project folder.

### 3. Open the notebook

```text
AI_Risk_Manager.ipynb
```

### 4. Run the notebook

Run the cells sequentially.

The Gradio dashboard will launch automatically.

---

## 💡 Project Highlights

**Predict → Score → Explain → Decide**

The system combines:

```text
Machine Learning
+
Risk Management
+
Explainable AI
+
Interactive Dashboard
```

to provide an intelligent payment fraud risk assessment system.

---


## ⚠️ Disclaimer

This is a  project for demonstrating fraud detection, risk scoring, and explainable AI. It is not intended for direct production financial decision-making.

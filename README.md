# 📉 Customer Churn Prediction Using Decision Tree

A machine learning project that predicts customer churn for a telecom company using a Decision Tree Classifier. The goal is to identify customers likely to leave, enabling proactive retention strategies.

---

## 📁 Dataset

- **Source:** [Telco Customer Churn – IBM Sample Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Records:** 7,043 customers
- **Features:** 21 columns including demographics, account info, and services subscribed

---

## 🎯 Objective

Predict whether a customer will churn (`Yes`) or stay (`No`) based on their account and service attributes.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Scikit-learn | ML model & evaluation |
| Matplotlib | Visualization |

---

## ⚙️ Project Workflow

### 1. Data Preprocessing
- Converted `TotalCharges` from string to numeric
- Found and filled **11 null values** in `TotalCharges` using median imputation
- Encoded target variable `Churn` → `Yes: 1`, `No: 0`
- Replaced redundant categories: `'No internet service'` and `'No phone service'` → `'No'`
- Dropped non-informative column: `customerID`

### 2. Feature Selection
Selected the following features as model inputs:
- `Contract`
- `MonthlyCharges`
- `InternetService`
- `tenure`
- `PaymentMethod`

### 3. Encoding & Splitting
- Applied **One-Hot Encoding** using `pd.get_dummies(drop_first=True)`
- Split data: **70% training / 30% testing** with `random_state=42`

### 4. Model Training
```python
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
```

### 5. Evaluation
```
Accuracy: 72.17%

Classification Report:
              precision    recall    f1-score   support
           0       0.80      0.82      0.81      1539
           1       0.49      0.47      0.48       574
    accuracy                           0.72      2113
   macro avg       0.65      0.64      0.64      2113
weighted avg       0.72      0.72      0.72      2113
```

---

## 📊 Results

| Metric | Non-Churn (0) | Churn (1) |
|--------|--------------|-----------|
| Precision | 0.80 | 0.49 |
| Recall | 0.82 | 0.47 |
| F1-Score | 0.81 | 0.48 |

**Confusion Matrix:**
- True Negatives: 1256 | False Positives: 283
- False Negatives: 305 | True Positives: 269

---

## 🔍 Feature Importance

| Feature | Importance |
|---------|-----------|
| MonthlyCharges | 44.17% |
| tenure | 33.80% |
| InternetService_Fiber optic | 10.47% |
| PaymentMethod_Electronic check | 2.91% |
| PaymentMethod_Credit card (automatic) | 2.21% |

> Customers with **high monthly charges**, **shorter tenure**, and **Fiber optic + electronic check** payment are most likely to churn.

---

## ⚠️ Limitations

- Churn recall is only **47%**, meaning more than half of actual churners are missed
- A basic Decision Tree tends to overfit without pruning
- Only 5 of 20 available features were used for training

---

## 🚀 Future Improvements

- Use **Random Forest** or **XGBoost** for better recall and generalization
- Apply **SMOTE** or class weighting to handle class imbalance
- Include all relevant features and perform proper feature selection
- Tune hyperparameters using **GridSearchCV**
- Set a custom decision threshold to optimize for recall

---


---

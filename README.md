# 💳 Credit Card Default Prediction

## 📌 Project Overview
This project predicts the likelihood of credit card customers defaulting on their payments next month.  
Dataset used: [UCI Credit Card Default Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

The goal is to help financial institutions identify high-risk customers and reduce credit risk.

---

## 📊 Workflow
1. **EDA & Visualization**
   - Distribution of Age, Gender, Education, Marriage, Credit Limit.
   - Correlation between demographic and financial features with default.
   - Pairplot and Heatmap analysis.

2. **Data Preprocessing**
   - Removed duplicates & irrelevant columns.
   - Scaled numerical features.
   - One-hot encoded categorical features.
   - Train-test split with stratification.

3. **Modeling**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - Support Vector Classifier
   - Neural Network (Keras Sequential)

4. **Evaluation Metrics**
   - Accuracy, Precision, Recall, F1-score
   - ROC Curve, Precision-Recall Curve, Confusion Matrix
   - AUC Score
   - Feature Importance (RandomForest / XGBoost)

---

## 🚀 Results
| Model                | Accuracy | Precision | Recall | F1-score | AUC  |
|-----------------------|----------|-----------|--------|----------|------|
| Logistic Regression   | 0.78     | 0.74      | 0.66   | 0.70     | 0.81 |
| Random Forest         | 0.82     | 0.77      | 0.71   | 0.74     | 0.85 |
| XGBoost               | 0.83     | 0.78      | 0.72   | 0.75     | 0.86 |
| Neural Network (Keras)| 0.81     | 0.76      | 0.70   | 0.73     | 0.84 |

---

## 📈 Business Insights
- Customers with **low credit limit** and **delayed payments (PAY_1, PAY_2)** are more likely to default.  
- Younger customers show slightly higher risk compared to older customers.  
- Education and marital status also have noticeable influence.

---

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Seaborn, Matplotlib, Plotly)
- Scikit-learn
- XGBoost
- TensorFlow / Keras
- Imbalanced-learn (SMOTE)

---

## 📂 How to Run
```bash
git clone https://github.com/ahmedhub2005/credit-card-default.git
cd credit-card-default
pip install -r requirements.txt
jupyter notebook credit_card_default.ipynb

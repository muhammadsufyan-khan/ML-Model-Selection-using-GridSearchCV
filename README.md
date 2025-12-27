# 📊 Machine Learning Model Comparison using GridSearchCV

## 📌 Objective
The objective of this project is to **build, compare, and evaluate multiple machine learning models** using a **clean, production-ready Scikit-learn Pipeline**.  
The task focuses on selecting the **best-performing model** based on **F1-score**, which is especially important for **imbalanced classification problems**.

---

## 🧠 Why This Project?
In real-world machine learning applications:
- Data preprocessing must be reusable and consistent
- Multiple models need fair comparison
- Hyperparameters should be optimized automatically
- Model selection must rely on strong evaluation metrics (not accuracy alone)

This project solves all of the above using **Pipeline + GridSearchCV**.

---

## 🧪 Dataset
- Tabular classification dataset
- Train/Test split applied
- Preprocessing includes:
  - Scaling numerical features
  - Encoding categorical features
  - Handling missing values

---

## ⚙️ Methodology

### 1️⃣ Data Preprocessing
- Numerical features scaled using `StandardScaler`
- Categorical features encoded using `OneHotEncoder`
- All preprocessing steps handled inside a **ColumnTransformer**

### 2️⃣ Pipeline Construction
A single reusable pipeline was built containing:
- Preprocessing step
- Classifier (Logistic Regression or Random Forest)

This ensures **no data leakage** and clean experimentation.

### 3️⃣ Model Training & Hyperparameter Tuning
`GridSearchCV` was used to:
- Compare multiple models
- Tune hyperparameters automatically
- Use **3-fold cross-validation**
- Select the best model based on **F1-score**

### 4️⃣ Models Compared
- Logistic Regression
- Random Forest Classifier

---

## 🧠 Final Pipeline (Core Idea)

```python
Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", model)
])
```

## 🔍 Hyperparameter Search Space

Logistic Regression

C: [0.1, 1, 10]

Random Forest

n_estimators: [100, 200]

max_depth: [None, 10, 20]

Total combinations evaluated: 9
Total model fits (3-fold CV): 27



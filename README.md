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

## 🏆 Best Model Selected
```
RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    random_state=42
)
```
## 📊 Evaluation Results
```
| Metric   | Score      |
| -------- | ---------- |
| Accuracy | **86.62%** |
| F1 Score | **68.24%** |
```
## 📋 Classification Report
```
| Class            | Precision | Recall | F1-score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| 0                | 0.88      | 0.95   | 0.92     | 7431    |
| 1                | 0.79      | 0.60   | 0.68     | 2338    |
| **Accuracy**     |           |        | **0.87** | 9769    |
| **Macro Avg**    | 0.84      | 0.78   | 0.80     | 9769    |
| **Weighted Avg** | 0.86      | 0.87   | 0.86     | 9769    |
```
## 🧠 Key Insights

Random Forest significantly outperformed Logistic Regression

High precision and recall for majority class

Reasonable performance on minority class despite imbalance

F1-score used instead of accuracy for fair evaluation

Pipeline ensures scalability and production-readiness

## 🚀 How to Use This Project
1️⃣ Clone the Repository
```
git clone https://github.com/your-username/ml-model-comparison-gridsearch.git
cd ml-model-comparison-gridsearch
```
2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
3️⃣ Run the Notebook
```
jupyter notebook notebooks/model_selection.py
```
OR run the script:
```
python src/model_selection.py
```
## 🧰 Tools & Libraries Used

Python

Pandas

NumPy

Scikit-learn

Jupyter Notebook

## 🏁 Final Conclusion

This project demonstrates a professional machine learning workflow using:

Pipelines

Automated hyperparameter tuning

Proper evaluation metrics

Clean and reusable code

It reflects industry-level ML practices and is suitable for:

Academic submission

Portfolio projects

Entry-level ML/AI roles


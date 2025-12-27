# ============================================================
# FIXED GRID SEARCH (MULTI-MODEL SAFE VERSION)
# ============================================================

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# -----------------------------
# PIPELINE
# -----------------------------
pipeline = Pipeline([
    ("preprocessing", preprocessor),   # already defined
    ("classifier", LogisticRegression())
])

# -----------------------------
# PARAMETER GRID
# -----------------------------
param_grid = [
    {
        "classifier": [LogisticRegression(max_iter=1000)],
        "classifier__C": [0.1, 1, 10]
    },
    {
        "classifier": [RandomForestClassifier(random_state=42)],
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20]
    }
]

# -----------------------------
# GRID SEARCH
# -----------------------------
grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1",
    cv=3,
    n_jobs=-1,
    verbose=2
)

# -----------------------------
# TRAIN MODEL
# -----------------------------
grid.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

print("✅ Best Model Parameters:", grid.best_params_)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

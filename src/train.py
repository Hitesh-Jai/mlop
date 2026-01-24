from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

from skopt import BayesSearchCV
from skopt.space import Integer, Real

# -------------------------
# Load data
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "loan_risk_data.csv"

df = pd.read_csv(DATA_PATH)

# -------------------------
# Features & Target
# -------------------------
X = df.drop("RiskCategory", axis=1)
y = df["RiskCategory"]

# -------------------------
# Column types
# -------------------------
categorical_features = [
    "EmploymentType",
    "ResidenceType",
    "PreviousDefault"
]

numerical_features = [
    "Age",
    "Income",
    "CreditScore",
    "LoanAmount",
    "LoanTerm"
]

# -------------------------
# Preprocessing
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

# -------------------------
# Model
# -------------------------
model = RandomForestClassifier(random_state=42)

# -------------------------
# Pipeline
# -------------------------
pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("classifier", model)
    ]
)

# -------------------------
# Bayesian search space
# -------------------------
search_space = {
    "classifier__n_estimators": Integer(100, 500),
    "classifier__max_depth": Integer(3, 20),
    "classifier__min_samples_split": Integer(2, 20),
    "classifier__min_samples_leaf": Integer(1, 10),
    "classifier__max_features": Real(0.3, 1.0)
}


# -------------------------
# Bayesian Search
# -------------------------
bayes_search = BayesSearchCV(
    estimator=pipeline,
    search_spaces=search_space,
    n_iter=30,              # number of intelligent trials
    cv=5,
    scoring="accuracy",
    random_state=42,
    n_jobs=-1
)


# -------------------------
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# -------------------------
# Fit Bayesian search
# -------------------------
bayes_search.fit(X_train, y_train)

# -------------------------
# Results
# -------------------------
best_model = bayes_search.best_estimator_

print("Best parameters:")
print(bayes_search.best_params_)

y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# -------------------------
# Train
# -------------------------
pipeline.fit(X_train, y_train)

# -------------------------
# Evaluate
# -------------------------
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


import joblib

MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

joblib.dump(pipeline, MODEL_DIR / "loan_model.joblib")

from pathlib import Path
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

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
model = LogisticRegression(max_iter=1000)

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
# Train-test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

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

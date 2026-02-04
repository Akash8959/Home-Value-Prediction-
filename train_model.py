import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import os
import json
import math

# ===== CONFIG =====
DATA_PATH = "data/housing.csv"
MODEL_PATH = "model/home_value_model.pkl"
METRICS_PATH = "model/model_metrics.json"
TARGET_COLUMN = "median_house_value"

# ===== LOAD DATA =====
print("📂 Loading dataset...")
data = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")

# ===== SPLIT =====
X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

numeric_features = ['longitude', 'latitude', 'housing_median_age',
                    'total_rooms', 'total_bedrooms', 'population',
                    'households', 'median_income']
categorical_features = ['ocean_proximity']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=200, random_state=42))
])

# ===== SPLIT DATA =====
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ===== TRAIN =====
print("🚀 Training model...")
model.fit(X_train, y_train)

# ===== EVALUATE =====
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# --------------- CONFUSION MATRIX FOR REGRESSION ---------------
# Convert continuous values to categorical bins for visualization
def categorize(value):
    if value < 100000:
        return "Low"
    elif value < 200000:
        return "Medium"
    elif value < 300000:
        return "High"
    else:
        return "Very High"

y_test_cat = [categorize(v) for v in y_test]
y_pred_cat = [categorize(v) for v in y_pred]

# Compute confusion matrix
cm = confusion_matrix(y_test_cat, y_pred_cat, labels=["Low", "Medium", "High", "Very High"])

# Display and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Low", "Medium", "High", "Very High"])
disp.plot(cmap="Blues", xticks_rotation=45)

plt.title("Confusion Matrix (Categorized Home Value Prediction)")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
plt.close()

print("✅ Confusion matrix saved to model/confusion_matrix.png")

# ---- version-safe RMSE ----
try:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
except TypeError:
    rmse = math.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 Model Evaluation:")
print(f"R²: {r2:.3f}")
print(f"MAE: {mae:.3f}")
print(f"RMSE: {rmse:.3f}")

# ===== SAVE MODEL =====
os.makedirs("model", exist_ok=True)
joblib.dump(model, MODEL_PATH)

# ===== FEATURE IMPORTANCE =====
rf_model = model.named_steps["regressor"]
encoder = model.named_steps["preprocessor"].transformers_[1][1].named_steps['encoder']
encoded_features = numeric_features + list(encoder.get_feature_names_out(categorical_features))

importances = rf_model.feature_importances_
feature_importance = sorted(zip(encoded_features, importances), key=lambda x: x[1], reverse=True)

# ===== SAVE METRICS =====
metrics = {
    "r2": r2,
    "mae": mae,
    "rmse": rmse,
    "feature_importance": feature_importance
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"\n✅ Model and metrics saved to {MODEL_PATH} and {METRICS_PATH}")

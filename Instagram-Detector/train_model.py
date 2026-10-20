"""
train_model.py
---------------
Instagram Fake-Spammer-Genuine Account Detection
End-to-End Training Script (Cleaned & Deployable)
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# =========================
# 1. Paths
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# =========================
# 2. Load Data
# =========================
print("Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# =========================
# 3. Data Cleaning
# =========================
train_df = train_df.drop_duplicates()
train_df = train_df.fillna(train_df.median(numeric_only=True))
train_df.columns = [c.strip().replace('#', 'num_').replace('/', '_') for c in train_df.columns]

# Remove non-numeric or interval-type columns
for col in train_df.columns:
    if not np.issubdtype(train_df[col].dtype, np.number):
        train_df[col] = train_df[col].astype('category').cat.codes

# =========================
# 4. Feature Engineering
# =========================
if "num_followers" in train_df.columns and "num_follows" in train_df.columns:
    train_df['followers_to_following'] = train_df['num_followers'] / (train_df['num_follows'] + 1)

if "num_posts" in train_df.columns and "num_followers" in train_df.columns:
    train_df['posts_per_follower'] = train_df['num_posts'] / (train_df['num_followers'] + 1)

train_df = train_df.replace([np.inf, -np.inf], 0)

# =========================
# 5. Split Data
# =========================
target = 'fake'  # ensure this matches your CSV
if target not in train_df.columns:
    raise ValueError(f"Target column '{target}' not found in train.csv")

X = train_df.drop(columns=[target])
y = train_df[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Ensure numeric
for df in [X_train, X_valid]:
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype('category').cat.codes
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0, inplace=True)

# =========================
# 6. Model Training
# =========================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_valid)
    acc = accuracy_score(y_valid, preds)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_valid, preds))

best_model_name = max(results, key=results.get)
print(f"\n✅ Best Model: {best_model_name} (Accuracy: {results[best_model_name]:.4f})")

# =========================
# 7. Retrain on Full Data
# =========================
# Convert all to numeric (for safety)
for col in X.columns:
    if not np.issubdtype(X[col].dtype, np.number):
        X[col] = X[col].astype('category').cat.codes

X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

best_model = models[best_model_name]
final_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', best_model)
])
print("\nRetraining best model on full dataset...")
final_pipe.fit(X, y)
print("✅ Final model trained successfully.")

# =========================
# 8. Evaluate on Test Data
# =========================
test_df.columns = [c.strip().replace('#', 'num_').replace('/', '_') for c in test_df.columns]
for col in test_df.columns:
    if not np.issubdtype(test_df[col].dtype, np.number):
        test_df[col] = test_df[col].astype('category').cat.codes
test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

missing_cols = set(X.columns) - set(test_df.columns)
for c in missing_cols:
    test_df[c] = 0

X_test = test_df[X.columns]
test_preds = final_pipe.predict(X_test)
test_df['predicted_fake'] = test_preds

print("\nSample test predictions:")
print(test_df[['predicted_fake']].head())

# =========================
# 9. Save Model
# =========================
model_path = os.path.join(NOTEBOOK_DIR, "best_instagram_fake_spammer_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(final_pipe, f)

print(f"\n✅ Model saved at: {model_path}")

# =========================
# 10. Confusion Matrix
# =========================
cm = confusion_matrix(y_valid, final_pipe.predict(X_valid))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix ({best_model_name})")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

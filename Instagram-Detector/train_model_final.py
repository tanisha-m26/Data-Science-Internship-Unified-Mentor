"""
train_model_final.py
---------------
Instagram Fake-Spammer-Genuine Account Detection
Full Analytics + Model Training with GridSearchCV + Visualizations
"""

import os
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ==========================================================
# 1. Folder Setup
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
NOTEBOOK_DIR = os.path.join(BASE_DIR, "notebooks")

os.makedirs(RESULTS_DIR, exist_ok=True)

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")

# ==========================================================
# 2. Load Data
# ==========================================================
print("📥 Loading data...")
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# ==========================================================
# 3. Data Cleaning & Preprocessing
# ==========================================================
train_df = train_df.drop_duplicates()
train_df.fillna(train_df.median(numeric_only=True), inplace=True)
train_df.columns = [c.strip().replace('#', 'num_').replace('/', '_') for c in train_df.columns]

# Convert categorical to numeric
for col in train_df.columns:
    if not np.issubdtype(train_df[col].dtype, np.number):
        train_df[col] = train_df[col].astype('category').cat.codes

# Feature engineering
if "num_followers" in train_df.columns and "num_follows" in train_df.columns:
    train_df['followers_to_following'] = train_df['num_followers'] / (train_df['num_follows'] + 1)

if "num_posts" in train_df.columns and "num_followers" in train_df.columns:
    train_df['posts_per_follower'] = train_df['num_posts'] / (train_df['num_followers'] + 1)

train_df = train_df.replace([np.inf, -np.inf], 0)

# ==========================================================
# 4. Target & Split
# ==========================================================
target = 'fake'
if target not in train_df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset!")

X = train_df.drop(columns=[target])
y = train_df[target]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ==========================================================
# 5. EDA Visualizations
# ==========================================================
plt.style.use('seaborn-v0_8-darkgrid')

# Fake vs Genuine Distribution
plt.figure(figsize=(6,4))
sns.countplot(x=target, data=train_df, palette='coolwarm')
plt.title("Fake vs Genuine Account Distribution")
plt.savefig(os.path.join(RESULTS_DIR, "01_fake_vs_genuine.png"))
plt.close()

# Follower vs Following
if 'num_followers' in X.columns and 'num_follows' in X.columns:
    plt.figure(figsize=(6,4))
    sns.scatterplot(x='num_followers', y='num_follows', hue=target, data=train_df, palette='viridis', alpha=0.7)
    plt.title("Follower vs Following Comparison")
    plt.savefig(os.path.join(RESULTS_DIR, "02_follower_vs_following.png"))
    plt.close()

# Correlation Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(train_df.corr(), annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig(os.path.join(RESULTS_DIR, "03_correlation_heatmap.png"))
plt.close()

# ==========================================================
# 6. Model Training + Hyperparameter Tuning
# ==========================================================
models = {
    "Random Forest": (RandomForestClassifier(random_state=42), {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [5, 10, 15]
    }),
    "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
        'clf__n_estimators': [100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2]
    }),
    "Logistic Regression": (LogisticRegression(max_iter=1000), {
        'clf__C': [0.1, 1, 10]
    })
}

best_model_name = None
best_score = 0
best_pipe = None
results_summary = []

print("\n🔍 Performing GridSearchCV for models...")

for name, (model, params) in models.items():
    print(f"\nTuning {name}...")
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', model)
    ])

    grid = GridSearchCV(pipe, param_grid=params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    preds = grid.predict(X_valid)

    acc = accuracy_score(y_valid, preds)
    prec = precision_score(y_valid, preds, zero_division=0)
    rec = recall_score(y_valid, preds, zero_division=0)
    f1 = f1_score(y_valid, preds, zero_division=0)

    results_summary.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

    print(f"Best Params for {name}: {grid.best_params_}")
    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}")

    if acc > best_score:
        best_score = acc
        best_model_name = name
        best_pipe = grid.best_estimator_

# Save summary
results_df = pd.DataFrame(results_summary)
results_df.to_csv(os.path.join(RESULTS_DIR, "model_metrics.csv"), index=False)

print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_score:.4f})")

# ==========================================================
# 7. Evaluation & Visuals
# ==========================================================
y_pred = best_pipe.predict(X_valid)

# Metrics
acc = accuracy_score(y_valid, y_pred)
prec = precision_score(y_valid, y_pred, zero_division=0)
rec = recall_score(y_valid, y_pred, zero_division=0)
f1 = f1_score(y_valid, y_pred, zero_division=0)

with open(os.path.join(RESULTS_DIR, "evaluation_report.txt"), "w") as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {acc:.4f}\nPrecision: {prec:.4f}\nRecall: {rec:.4f}\nF1-Score: {f1:.4f}\n\n")
    f.write(classification_report(y_valid, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_valid, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title(f"Confusion Matrix ({best_model_name})")
plt.savefig(os.path.join(RESULTS_DIR, "04_confusion_matrix.png"))
plt.close()

# Feature Importance
if hasattr(best_pipe.named_steps['clf'], 'feature_importances_'):
    importances = best_pipe.named_steps['clf'].feature_importances_
    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)

    plt.figure(figsize=(8,5))
    sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='mako')
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "05_feature_importance.png"))
    plt.close()

# ==========================================================
# 8. Save Final Model
# ==========================================================
model_path = os.path.join(NOTEBOOK_DIR, "best_instagram_fake_spammer_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(best_pipe, f)
print(f"\n✅ Final tuned model saved at: {model_path}")

print("\nAll plots, metrics, and results are saved in the 'results' folder.")

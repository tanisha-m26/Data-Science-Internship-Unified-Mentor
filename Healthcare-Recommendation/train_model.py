import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import pickle  # use pickle instead of joblib

# --------------------- Load dataset ---------------------
df = pd.read_csv("D:\\Personalized-healthcare-recommendation\\blood.csv")

X = df[['Recency', 'Frequency', 'Monetary', 'Time']]
y = df["Class"]

# --------------------- Train/Test Split ---------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------- Preprocessing & Pipeline ---------------------
numerical_features = ['Recency', 'Frequency', 'Monetary', 'Time']

preprocessor = ColumnTransformer(
    transformers=[('num', Pipeline([('scaler', StandardScaler())]), numerical_features)]
)

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Train the pipeline on the full dataset
model_pipeline.fit(X, y)

# --------------------- Save the trained pipeline ---------------------
with open("D:\\Personalized-healthcare-recommendation\\healthcare_model.pkl", "wb") as f:
    pickle.dump(model_pipeline, f)

print("✅ Model pipeline saved as healthcare_model.pkl using pickle")

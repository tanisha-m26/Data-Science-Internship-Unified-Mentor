"""
streamlit_app.py
-----------------
Interactive Streamlit dashboard for Instagram Fake-Spammer-Genuine Account Detection
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import StandardScaler

# ==========================================================
# 1. Paths & Setup
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "data")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")

MODEL_PATH = os.path.join(NOTEBOOKS_DIR, "best_instagram_fake_spammer_model.pkl")
METRICS_PATH = os.path.join(RESULTS_DIR, "model_metrics.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "evaluation_report.txt")

st.set_page_config(
    page_title="Instagram Fake-Spammer Detector Dashboard",
    layout="wide",
    page_icon="📊"
)

# ==========================================================
# 2. Helper Functions
# ==========================================================
@st.cache_data
def load_data():
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return train, test

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

def make_numeric(df):
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].astype('category').cat.codes
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)

# ==========================================================
# 3. Sidebar Navigation
# ==========================================================
st.sidebar.title("📁 Navigation")
page = st.sidebar.radio("Select Page:", [
    "📈 Dashboard Overview",
    "📊 Data Visualizations",
    "🤖 Model Metrics",
    "🔍 Predict New Data"
])

# ==========================================================
# 4. Load Base Data & Model
# ==========================================================
train_df, test_df = load_data()
model = load_model()

# Preprocessing
train_df = make_numeric(train_df)
if "fake" in train_df.columns:
    target = train_df["fake"]
    features = train_df.drop(columns=["fake"])
else:
    target = None
    features = train_df

# ==========================================================
# 5. Dashboard Overview
# ==========================================================
if page == "📈 Dashboard Overview":
    st.title("📸 Instagram Fake-Spammer-Genuine Detector")
    st.write("""
    This dashboard provides data analytics and model predictions for identifying fake or spammer accounts on Instagram.
    The model was trained using multiple algorithms (Logistic Regression, Random Forest, Gradient Boosting) 
    with **GridSearchCV** optimization for best performance.
    """)

    c1, c2, c3 = st.columns(3)
    if target is not None:
        total_fake = (target == 1).sum()
        total_genuine = (target == 0).sum()
        c1.metric("Total Fake Accounts", f"{total_fake}")
        c2.metric("Total Genuine Accounts", f"{total_genuine}")
        c3.metric("Dataset Size", f"{len(train_df)} records")

    st.subheader("📊 Key Results Overview")
    metrics_file = pd.read_csv(METRICS_PATH)
    st.dataframe(metrics_file, use_container_width=True)

    st.write("---")
    st.image(os.path.join(RESULTS_DIR, "04_confusion_matrix.png"), caption="Confusion Matrix", use_container_width=True)

# ==========================================================
# 6. Data Visualizations
# ==========================================================
elif page == "📊 Data Visualizations":
    st.title("📉 Data Visualizations")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(RESULTS_DIR, "01_fake_vs_genuine.png"), caption="Fake vs Genuine Account Distribution", use_container_width=True)
    with col2:
        if os.path.exists(os.path.join(RESULTS_DIR, "02_follower_vs_following.png")):
            st.image(os.path.join(RESULTS_DIR, "02_follower_vs_following.png"), caption="Follower vs Following Comparison", use_container_width=True)

    st.image(os.path.join(RESULTS_DIR, "03_correlation_heatmap.png"), caption="Feature Correlation Heatmap", use_container_width=True)
    if os.path.exists(os.path.join(RESULTS_DIR, "05_feature_importance.png")):
        st.image(os.path.join(RESULTS_DIR, "05_feature_importance.png"), caption="Feature Importance", use_container_width=True)

# ==========================================================
# 7. Model Metrics
# ==========================================================
elif page == "🤖 Model Metrics":
    st.title("🤖 Model Performance Metrics")

    if os.path.exists(METRICS_PATH):
        metrics_df = pd.read_csv(METRICS_PATH)
        st.dataframe(metrics_df, use_container_width=True)

        st.write("📄 Detailed Classification Report:")
        if os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r") as f:
                report_text = f.read()
            st.text(report_text)
    else:
        st.warning("Metrics not found. Please train the model first.")

# ==========================================================
# 8. Prediction Interface (Updated)
# ==========================================================
elif page == "🔍 Predict New Data":
    st.title("🔍 Predict Fake or Genuine Accounts")

    st.write("""
    Upload a CSV file containing Instagram account features, or input values manually to predict if an account is **Fake (1)** or **Genuine (0)**.
    """)

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        user_df = pd.read_csv(uploaded_file)

        # ---------------------------
        # 1. Rename columns to match training features
        # ---------------------------
        rename_dict = {
            "nums/length username": "nums_length username",
            "nums/length fullname": "nums_length fullname",
            "#posts": "num_posts",
            "#followers": "num_followers",
            "#follows": "num_follows",
            "profile pic": "profile pic",
            # Add other mappings if needed
        }
        user_df = user_df.rename(columns=rename_dict)

        # ---------------------------
        # 2. Create derived features if used in training
        # ---------------------------
        if "num_posts" in user_df.columns and "num_followers" in user_df.columns:
            user_df["posts_per_follower"] = user_df["num_posts"] / (user_df["num_followers"] + 1)

        # ---------------------------
        # 3. Convert non-numeric columns to numeric
        # ---------------------------
        user_df = make_numeric(user_df)

        # ---------------------------
        # 4. Keep only features the model expects
        # ---------------------------
        trained_features = model.feature_names_in_
        missing_features = [f for f in trained_features if f not in user_df.columns]
        for f in missing_features:
            user_df[f] = 0  # fill missing columns with 0

        user_df = user_df[trained_features]

        # ---------------------------
        # 5. Display and predict
        # ---------------------------
        st.write("Preview of uploaded data (after processing):")
        st.dataframe(user_df.head(), use_container_width=True)

        predictions = model.predict(user_df)
        user_df["Predicted_Fake"] = predictions

        st.success("✅ Predictions complete!")
        st.dataframe(user_df, use_container_width=True)

        # Download option
        csv = user_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "⬇️ Download Predictions",
            data=csv,
            file_name="predicted_accounts.csv",
            mime="text/csv"
        )
    else:
        st.info("Please upload a CSV file to start predictions.")
# ==========================================================
# End of File
# ==========================================================

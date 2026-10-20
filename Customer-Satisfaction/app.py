# app.py - Unified Customer Satisfaction Dashboard (XGBoost, no pickle)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
import warnings
warnings.filterwarnings("ignore")

import os
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="📊 Customer Satisfaction Dashboard", layout="wide")
st.title("📊 Customer Satisfaction Prediction & Monitoring Dashboard")

# -----------------------------
# Sidebar - Data Input & Controls
# -----------------------------
st.sidebar.header("📂 Data Input & Controls")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file (customer support tickets)", type=["csv"])
retrain_button = st.sidebar.button("🔁 Retrain model on current dataset")

@st.cache_data


def load_data(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    else:
        # fallback to default dataset inside repo
        return pd.read_csv(os.path.join("data", "D:\\Customer-Satisfaction-dashboard\\data\\customer_support_tickets.csv"))


uploaded_file = st.file_uploader("📂 Upload Customer Support Tickets CSV", type=["csv"])
df_raw = load_data(uploaded_file)

if df_raw is None:
    st.warning("⚠️ Please upload a dataset to proceed.")
    st.stop()

# Upload feedback
if uploaded_file:
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
else:
    st.sidebar.info("Using default dataset (no upload provided).")

st.sidebar.markdown(f"**Rows:** {df_raw.shape[0]}  —  **Cols:** {df_raw.shape[1]}")
st.write("### Sample of loaded data")
st.dataframe(df_raw.head())

# -----------------------------
# Utilities: safe encoder transform
# -----------------------------
def safe_label_transform(le: LabelEncoder, val):
    """Transform with LabelEncoder; if unseen value, return mode (0)"""
    try:
        return int(le.transform([str(val)])[0])
    except Exception:
        try:
            return int(np.argmax(np.bincount(le.transform(le.classes_))))
        except Exception:
            return 0

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess(df: pd.DataFrame, fit_encoders=False, encoders=None, fit_scaler=False, scaler=None):
    df = df.copy()
    drop_piis = ["Ticket ID", "Customer Name", "Customer Email"]
    for c in drop_piis:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    if "Date of Purchase" in df.columns:
        df["Date of Purchase"] = pd.to_datetime(df["Date of Purchase"], errors="coerce", dayfirst=True)
        df["days_since_purchase"] = (pd.to_datetime("today") - df["Date of Purchase"]).dt.days
        df.drop(columns=["Date of Purchase"], inplace=True)

    if "First Response Time" in df.columns:
        try:
            df["first_response_ts"] = pd.to_datetime(df["First Response Time"], errors="coerce")
            df["first_response_hours"] = (df["first_response_ts"] - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")
            df.drop(columns=["First Response Time", "first_response_ts"], inplace=True, errors=True)
        except Exception:
            df["first_response_hours"] = pd.to_numeric(df["First Response Time"], errors="coerce")

    if "Time to Resolution" in df.columns:
        try:
            df["time_to_resolution_td"] = pd.to_timedelta(df["Time to Resolution"], errors="coerce")
            df["time_to_resolution_hours"] = df["time_to_resolution_td"].dt.total_seconds() / 3600
            df.drop(columns=["Time to Resolution", "time_to_resolution_td"], inplace=True, errors=True)
        except Exception:
            df["time_to_resolution_hours"] = pd.to_numeric(df["Time to Resolution"], errors="coerce")

    text_drop = ["Ticket Description", "Ticket Subject", "Resolution"]
    for c in text_drop:
        if c in df.columns:
            df[c] = df[c].astype(str).fillna("")

    categorical_cols = []
    for col in df.columns:
        if df[col].dtype == "object" and col not in text_drop:
            categorical_cols.append(col)

    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if fit_encoders:
            le = LabelEncoder()
            df[col] = df[col].astype(str).fillna("Unknown")
            le.fit(df[col])
            encoders[col] = le
            df[col] = le.transform(df[col])
        else:
            if col in encoders:
                le = encoders[col]
                df[col] = df[col].astype(str).fillna("Unknown")
                df[col] = df[col].apply(lambda v: safe_label_transform(le, v))
            else:
                le = LabelEncoder()
                df[col] = df[col].astype(str).fillna("Unknown")
                le.fit(df[col])
                encoders[col] = le
                df[col] = le.transform(df[col])

    y = None
    if "Customer Satisfaction Rating" in df.columns:
        y = df["Customer Satisfaction Rating"].copy()
        mask = y.notnull()
        df = df[mask]
        y = y[mask].astype(int)
        y = y - 1

    exclude_cols = ["Ticket Description", "Ticket Subject", "Resolution", "Customer Satisfaction Rating"]
    X = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors="ignore")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[numeric_cols]
    X = X.fillna(0)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
    else:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)

    return X_scaled, y, encoders, scaler

# -----------------------------
# Initial training
# -----------------------------
encoders, scaler, model = {}, None, None
X, y, encoders, scaler = None, None, None, None

if "Customer Satisfaction Rating" in df_raw.columns and df_raw["Customer Satisfaction Rating"].notnull().any():
    X_train_full, y_full, encoders, scaler = preprocess(df_raw, fit_encoders=True, encoders=None, fit_scaler=True, scaler=None)
    X_train, X_test, y_train, y_test = train_test_split(X_train_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
    model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42,
                          n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    base_X_columns = X_train.columns.tolist()
else:
    st.warning("Dataset does not contain 'Customer Satisfaction Rating' or has no labeled rows - model will not be trained.")
    base_X_columns = []

if retrain_button:
    if "Customer Satisfaction Rating" in df_raw.columns and df_raw["Customer Satisfaction Rating"].notnull().any():
        st.sidebar.info("Retraining model on loaded dataset...")
        X_full, y_full, encoders, scaler = preprocess(df_raw, fit_encoders=True, encoders=None, fit_scaler=True, scaler=None)
        X_train, X_test, y_train, y_test = train_test_split(X_full, y_full, test_size=0.3, random_state=42, stratify=y_full)
        model = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42,
                              n_estimators=300, max_depth=6, learning_rate=0.05, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        base_X_columns = X_train.columns.tolist()
        st.sidebar.success("Retrain complete ✅")
    else:
        st.sidebar.error("Cannot retrain: no labeled target in uploaded dataset.")

# Save dropdown options
dropdown_options = {}
if encoders:
    for col, le in encoders.items():
        dropdown_options[col] = list(le.classes_)

# -----------------------------
# Sidebar Navigation
# -----------------------------
st.sidebar.title("🔍 Navigation")
section = st.sidebar.radio("Go to:",
                           ["Prediction", "Model Metrics", "Insights & Visuals", "Explainability", "Business Monitoring"])

# -----------------------------
# Align & scale input
# -----------------------------
def align_and_scale_input(df_input: pd.DataFrame, encoders: dict, scaler: StandardScaler, base_cols: list):
    df = df_input.copy()
    for c in ["Ticket ID", "Customer Name", "Customer Email", "Date of Purchase"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True, errors=True)

    if "Date of Purchase" in df_input.columns:
        df["days_since_purchase"] = (pd.to_datetime("today") - pd.to_datetime(df_input["Date of Purchase"], errors="coerce", dayfirst=True)).dt.days

    if "First Response Time" in df_input.columns:
        try:
            df["first_response_hours"] = (pd.to_datetime(df_input["First Response Time"], errors="coerce") - pd.Timestamp("1970-01-01")) // pd.Timedelta("1h")
        except Exception:
            df["first_response_hours"] = pd.to_numeric(df_input["First Response Time"], errors="coerce")

    if "Time to Resolution" in df_input.columns:
        try:
            df["time_to_resolution_hours"] = pd.to_timedelta(df_input["Time to Resolution"], errors="coerce").dt.total_seconds() / 3600
        except Exception:
            df["time_to_resolution_hours"] = pd.to_numeric(df_input["Time to Resolution"], errors="coerce")

    for col, le in encoders.items():
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("Unknown").apply(lambda v: safe_label_transform(le, v))
        else:
            df[col] = 0

    df = df.select_dtypes(include=[np.number]).fillna(0)

    for c in base_cols:
        if c not in df.columns:
            df[c] = 0

    df = df[base_cols]
    scaled = scaler.transform(df)
    return scaled

# -----------------------------
# Section 1: Prediction
# -----------------------------
if section == "Prediction":
    st.header("🔮 Predict Customer Satisfaction")

    st.subheader("1) Upload ticket CSV for batch prediction")
    pred_file = st.file_uploader("Upload CSV for prediction (optional)", type=["csv"], key="pred_csv")
    if pred_file and model is not None:
        pred_df = pd.read_csv(pred_file)
        try:
            preds_scaled = align_and_scale_input(pred_df, encoders, scaler, base_X_columns)
            preds = model.predict(preds_scaled) + 1
            pred_df["Predicted_Satisfaction"] = preds
            st.success("✅ Predictions complete")
            st.dataframe(pred_df.head(10))
            st.download_button("⬇ Download Predictions", pred_df.to_csv(index=False), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    elif pred_file and model is None:
        st.warning("Model not trained - cannot predict. Please load a dataset with labels or retrain.")

    st.markdown("---")
    st.subheader("2) Or enter a single ticket manually")
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Customer Age", min_value=0, max_value=120, value=30)

        if "Customer Gender" in dropdown_options:
            gender_val = st.selectbox("Customer Gender", dropdown_options["Customer Gender"])
        else:
            gender_val = st.text_input("Customer Gender (text)", value="Female")

        if "Product Purchased" in dropdown_options:
            product_val = st.selectbox("Product Purchased", dropdown_options["Product Purchased"])
        else:
            product_val = st.text_input("Product Purchased", value="GoPro Hero")

        if "Ticket Type" in dropdown_options:
            ticket_type_val = st.selectbox("Ticket Type", dropdown_options["Ticket Type"])
        else:
            ticket_type_val = st.text_input("Ticket Type", value="Technical issue")

        if "Ticket Priority" in dropdown_options:
            priority_val = st.selectbox("Ticket Priority", dropdown_options["Ticket Priority"])
        else:
            priority_val = st.text_input("Ticket Priority", value="Low")

    with col2:
        if "Ticket Channel" in dropdown_options:
            channel_val = st.selectbox("Ticket Channel", dropdown_options["Ticket Channel"])
        else:
            channel_val = st.text_input("Ticket Channel", value="Email")

        if "Ticket Status" in dropdown_options:
            status_val = st.selectbox("Ticket Status", dropdown_options["Ticket Status"])
        else:
            status_val = st.text_input("Ticket Status", value="Closed")

        first_response_val = st.text_input("First Response Time (hrs or timestamp)", value="5")
        time_to_resolution_val = st.text_input("Time to Resolution (hrs or duration)", value="24")

    if st.button("Predict this ticket"):
        if model is None:
            st.error("Model not trained. Please load dataset with labels or retrain.")
        else:
            manual_df = pd.DataFrame([{
                "Customer Age": age,
                "Customer Gender": gender_val,
                "Product Purchased": product_val,
                "Ticket Type": ticket_type_val,
                "Ticket Priority": priority_val,
                "Ticket Channel": channel_val,
                "Ticket Status": status_val,
                "First Response Time": first_response_val,
                "Time to Resolution": time_to_resolution_val
            }])
            try:
                X_manual = align_and_scale_input(manual_df, encoders, scaler, base_X_columns)
                pred = model.predict(X_manual)[0] + 1
                stars = "⭐" * pred + "☆" * (5 - pred)
                st.success(f"Predicted Satisfaction Rating: {pred}  ({stars})")
            except Exception as e:
                st.error(f"Manual prediction failed: {e}")

# -----------------------------
# Section 2: Model Metrics
# -----------------------------
elif section == "Model Metrics":
    st.header("📈 Model Evaluation")
    if model is None:
        st.warning("Model not trained yet.")
    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{acc*100:.2f}%")
        st.subheader("Classification Report (labels shown as original 1..5)")
        st.text(classification_report(y_test + 1, y_pred + 1))
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test + 1, y_pred + 1)
        fig, ax = plt.subplots(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

# -----------------------------
# Section 3: Insights & Visuals
# -----------------------------
elif section == "Insights & Visuals":
    st.header("📊 Insights & Visualizations")
    df_display = df_raw.copy()
    if "Customer Satisfaction Rating" in df_display.columns:
        st.subheader("Satisfaction Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x=df_display["Customer Satisfaction Rating"], ax=ax, palette="viridis")
        ax.set_xlabel("Satisfaction Rating")
        st.pyplot(fig)

    if "Ticket Type" in df_display.columns:
        st.subheader("Ticket Type Distribution")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.countplot(y="Ticket Type", data=df_display, order=df_display["Ticket Type"].value_counts().index, ax=ax)
        st.pyplot(fig)

    if {"Ticket Priority", "Customer Satisfaction Rating"}.issubset(df_display.columns):
        st.subheader("Priority vs Satisfaction")
        fig, ax = plt.subplots(figsize=(8,4))
        sns.boxplot(x="Ticket Priority", y="Customer Satisfaction Rating", data=df_display, ax=ax)
        st.pyplot(fig)

    if {"Ticket Channel", "Customer Satisfaction Rating"}.issubset(df_display.columns):
        st.subheader("Channel-wise Average Satisfaction")
        channel_avg = df_display.groupby("Ticket Channel")["Customer Satisfaction Rating"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8,4))
        channel_avg.plot(kind="barh", ax=ax)
        st.pyplot(fig)

    if "Customer Age" in df_display.columns:
        st.subheader("Age Group vs Avg Satisfaction")
        bins = [0, 30, 40, 50, 60, 120]
        labels = ["<30", "30-39", "40-49", "50-59", "60+"]
        df_display["age_group"] = pd.cut(df_display["Customer Age"], bins=bins, labels=labels)
        ag = df_display.groupby("age_group")["Customer Satisfaction Rating"].mean()
        fig, ax = plt.subplots(figsize=(7,4))
        ag.plot(kind="bar", ax=ax, color="orange")
        st.pyplot(fig)

# -----------------------------
# Section 4: Explainability (SHAP)
# -----------------------------
elif section == "Explainability":
    st.header("🤖 Model Explainability (SHAP)")

    if model is None:
        st.warning("⚠️ No trained model available.")
    else:
        # sample test data for speed (limit to 50 for stability)
        sample_size = min(50, X_test.shape[0])
        sample_X = pd.DataFrame(X_test, columns=base_X_columns).sample(sample_size, random_state=42)

        try:
            # Always use TreeExplainer for XGBoost
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample_X, check_additivity=False)

            # -----------------------------
            # Global SHAP Summary
            # -----------------------------
            st.subheader("🌍 Global SHAP Summary")
            fig, ax = plt.subplots(figsize=(10, 4))

            if isinstance(shap_values, list):  # Multiclass
                mean_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
                shap.summary_plot(mean_shap, sample_X, plot_type="bar", show=False)
            else:  # Binary/Regression
                shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)

            st.pyplot(fig)
            plt.close(fig)

            

        except Exception as e:
            st.error(f"❌ SHAP failed completely: {e}")


# -----------------------------
# Section 5: Business Monitoring
# -----------------------------
elif section == "Business Monitoring":
    st.header("📊 Business Monitoring Dashboard")
    if "Customer Satisfaction Rating" in df_raw.columns:
        rolling = df_raw["Customer Satisfaction Rating"].rolling(50, min_periods=1).mean()
        fig, ax = plt.subplots(figsize=(8,4))
        rolling.plot(ax=ax)
        ax.set_title("Rolling Average of Satisfaction (window=50)")
        st.pyplot(fig)
    if {"Ticket Priority", "Customer Satisfaction Rating"}.issubset(df_raw.columns):
        st.subheader("Priority vs Avg Satisfaction (Business metric)")
        avg = df_raw.groupby("Ticket Priority")["Customer Satisfaction Rating"].mean()
        st.bar_chart(avg)
    if "Ticket Channel" in df_raw.columns:
        st.subheader("Channel share of tickets")
        st.bar_chart(df_raw["Ticket Channel"].value_counts())


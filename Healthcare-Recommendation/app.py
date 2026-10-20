# --------------------- Streamlit App for Personalized Healthcare Recommendations --------------------- 

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import os
import pickle

# --------------------- Paths ---------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "blood.csv")
LOGO_PATH = os.path.join(BASE_DIR, "static", "images", "logo.png")
MODEL_PATH = os.path.join(BASE_DIR, "models", "healthcare_model.pkl")

# --------------------- Load Data ---------------------
data = pd.read_csv(DATA_PATH)

# --------------------- Train or Load Model ---------------------
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model_pipeline = pickle.load(f)
else:
    X = data[['Recency', 'Frequency', 'Monetary', 'Time']]
    y = data['Class']

    numerical_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('num', numerical_pipeline, X.columns)])

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
    ])
    model_pipeline.fit(X, y)

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model_pipeline, f)

# --------------------- Streamlit Page Config ---------------------
st.set_page_config(
    page_title="Healthcare Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------- Sidebar ---------------------
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, use_container_width=True)

st.sidebar.title("📊 Dashboard Options")
page = st.sidebar.radio("Navigate", ["Home", "Feature Importance", "Model Metrics", "Live Prediction"])

# --------------------- Sidebar: Real-time Metrics ---------------------
X = data[['Recency', 'Frequency', 'Monetary', 'Time']]
y = data['Class']
y_pred = model_pipeline.predict(X)
accuracy = accuracy_score(y, y_pred)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Real-time Metrics")
st.sidebar.metric("Accuracy", f"{accuracy*100:.2f}%")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y, model_pipeline.predict_proba(X)[:, 1])
roc_auc = auc(fpr, tpr)

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve'))
fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Random', line=dict(dash='dash')))
fig_roc.update_layout(title=f'ROC Curve (AUC = {roc_auc:.2f})', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate', height=300)
st.sidebar.plotly_chart(fig_roc, use_container_width=True)

# Live Feature Importance
importances = model_pipeline.named_steps['classifier'].feature_importances_
feat_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)
fig_feat = go.Figure(go.Bar(x=feat_importance_df['Importance'], y=feat_importance_df['Feature'], orientation='h'))
fig_feat.update_layout(title="Feature Importance", height=300)
st.sidebar.plotly_chart(fig_feat, use_container_width=True)

# --------------------- Home Page ---------------------
if page == "Home":
    st.title("🏥 Personalized Healthcare Recommendation System")
    st.markdown("### Explore your health data and predictions in real time")

    # Dataset preview
    st.subheader("Dataset Preview")
    st.dataframe(data.head())

    # Histograms
    st.subheader("Recency Distribution")
    fig = px.histogram(data, x="Recency", nbins=20, color='Class', title="Recency Distribution")
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plot
    st.subheader("Frequency vs Monetary Spend")
    fig2 = px.scatter(data, x="Frequency", y="Monetary", color="Class", size='Time',
                      title="Frequency vs Monetary with Time Size")
    st.plotly_chart(fig2, use_container_width=True)

# --------------------- Feature Importance ---------------------
elif page == "Feature Importance":
    st.title("🔍 Feature Importance")
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x=importances, y=X.columns, ax=ax, palette="viridis")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

# --------------------- Model Metrics ---------------------
elif page == "Model Metrics":
    st.title("📈 Model Performance Metrics")
    st.subheader("Classification Report")
    st.text(classification_report(y, y_pred))

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

# --------------------- Live Prediction ---------------------
elif page == "Live Prediction":
    st.title("⚡ Live Health Risk Prediction")
    st.subheader("Enter patient data to get a personalized recommendation.")

    recency = st.number_input("Recency (days since last visit)", min_value=0, max_value=365, value=60)
    frequency = st.number_input("Frequency (visits)", min_value=0, max_value=50, value=5)
    monetary = st.number_input("Monetary (medical spend)", min_value=0, max_value=5000, value=1000)
    time = st.number_input("Time (months as patient)", min_value=0, max_value=120, value=36)

    if st.button("Get Recommendation"):
        patient_df = pd.DataFrame([[recency, frequency, monetary, time]],
                                  columns=['Recency','Frequency','Monetary','Time'])
        prediction = model_pipeline.predict(patient_df)[0]
        result_map = {0: "✅ No immediate action needed",
                      1: "⚠️ Regular monitoring or intervention required"}
        st.success(f"Recommendation: {result_map[prediction]}")
        st.info("Note: This is a simplified example. Validate your model for real-world use.")

# --------------------- Dataset Overview ---------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Overview")
st.sidebar.dataframe(data)

st.subheader("Dataset Histogram Overview")
fig3 = px.histogram(data, x='Recency', color='Class', barmode='overlay', title="Recency by Class")
st.plotly_chart(fig3, use_container_width=True)


#30,1,500,24 ==default values
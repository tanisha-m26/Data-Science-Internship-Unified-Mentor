🧠 ML Prediction & Explainability Dashboard

This is a Streamlit web app for training, evaluating, and interpreting XGBoost-based ML models.
It integrates real-time prediction, explainability with SHAP, and visualization in one place.

🚀 Features

📊 Data Upload & Preprocessing (CSV support, train/test split)

🤖 Model Training (XGBoost classifier)

📈 Performance Evaluation (accuracy, confusion matrix, classification report)

🔍 Explainability with SHAP

Global SHAP summary plots

Single prediction explanations with Force Plot

🎨 Interactive UI built with Streamlit

🔗 Live Deployment

You can access the live app here: ML Explainability Dashboard

⚡ Run Locally

Clone the repository:

git clone https://github.com/tanisha-m26/Customer-Satisfaction-dashboard
cd ml-explainability-app


Install dependencies:

pip install -r requirements.txt


Run the app:

streamlit run app.py

📦 Requirements

Main Python libraries used:

streamlit

xgboost

shap

matplotlib

pandas

numpy

Install them all via:

pip install -r requirements.txt

📂 Project Structure
ml-explainability-app/
│── app.py               # Main Streamlit app
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
│── data/                # (Optional) Example datasets

✨ Example Screenshots

Global SHAP Summary (feature importance)

Single Prediction Force Plot (local interpretability)

📜 License

This project is open-source under the MIT License.
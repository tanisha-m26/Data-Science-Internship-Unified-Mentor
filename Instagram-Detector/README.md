# Instagram Fake-Spammer-Genuine Account Detector

This project is a **Streamlit-based interactive dashboard** for detecting fake, spammer, or genuine Instagram accounts using a trained machine learning model.  
It provides **data visualization, model metrics, and prediction capabilities** for new Instagram account data.

---

## **Folder Structure**

Instagram-fake-spammer-genuine-accounts/
-|
-├── data/
-│ ├── train.csv # Training dataset
-│ └── test.csv # Test dataset
-|
-├── notebooks/
-│ └── best_instagram_fake_spammer_model.pkl # Trained ML model
-│ └── Figure_1.png
-│ └── git_bash_output(train_model_final.py)
-│ └── git_bash_output(train_model.py)
-│ └── instagram_fake_spammer_project.ipynb
-│ └── best_instagram_fake_spammer_model.pkl # Trained ML model
-|
-├── results/
-│ ├── 01_fake_vs_genuine.png
-│ ├── 02_follower_vs_following.png
-│ ├── 03_correlation_heatmap.png
-│ ├── 04_confusion_matrix.png
-│ ├── 05_feature_importance.png
-│ ├── model_metrics.csv
-│ └── evaluation_report.txt
-|
-├── .gitignore
-|
-├── app.py # Main Streamlit application
-|
-├── README.md # Project documentation
-|
-├── LICENSE
-|
-├── train_model.py
-|
-├── train_model_final.py
-└── requirements.txt # Python dependencies


---

## **Features**

1. **Dashboard Overview**  
   - Displays total fake and genuine accounts in the dataset.
   - Provides dataset size and key statistics.

2. **Data Visualizations**  
   - Fake vs genuine account distribution.
   - Follower vs following comparisons.
   - Feature correlation heatmap.
   - Feature importance visualization.

3. **Model Metrics**  
   - Displays model performance metrics including accuracy, precision, recall, F1-score.
   - Shows confusion matrix and detailed classification report.

4. **Prediction Interface**  
   - Upload a CSV file containing Instagram account features.
   - Automatically preprocesses the data, including:
     - Converting categorical features to numeric.
     - Calculating derived features like `posts_per_follower`.
     - Aligning feature columns with the trained model.
   - Predicts if accounts are Fake (1) or Genuine (0).
   - Allows downloading the prediction results as a CSV file.

---

## **How to Run**

1. **Clone the repository**:

```bash
git clone <repository_url>
cd Instagram-Fake-Spammer-Detector
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run the Streamlit app**:
```bash
streamlit run streamlit_app.py
```

4. **Use the App**:
```bash
Navigate between pages using the sidebar:

    📈 Dashboard Overview

    📊 Data Visualizations

    🤖 Model Metrics

    🔍 Predict New Data

Upload a CSV file with account features to get predictions.
```

Data Requirements
CSV files for training, testing, or prediction must include the following columns (or their equivalents):

profile pic
nums/length username
fullname words
nums/length fullname
name==username
description length
external URL
private
#posts
#followers
#follows
fake

Non-numeric columns are automatically converted during preprocessing.


Missing features are filled with 0 for prediction purposes.


Derived features like posts_per_follower are automatically calculated.


Dependencies
Python libraries required (specified in requirements.txt):
streamlit,

pandas,

numpy,

scikit-learn,

matplotlib,

seaborn,

Pillow.

Notes
The trained model is stored in notebooks/best_instagram_fake_spammer_model.pkl.


All visualizations and model metrics are stored in the results/ folder.


Ensure uploaded CSV files have consistent column names with the training dataset for correct predictions.


This app can handle minor differences in uploaded CSVs by renaming and adding missing columns automatically.

Author

Developed with ❤️ using Python, Streamlit, Scikit-learn, Seaborn, and Matplotlib.
















import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df: pd.DataFrame):
    """
    Preprocess dataset for ML model training
    - Encode categorical variables
    - Split into train/test
    """

    df = df.copy()

    # Handle missing satisfaction ratings (drop rows without label)
    df = df.dropna(subset=["customer_satisfaction_rating"])

    # Encode categorical variables
    categorical_cols = [
        "customer_gender", "product_purchased", "ticket_type",
        "ticket_priority", "ticket_channel", "ticket_status"
    ]
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Feature engineering: resolution time in hours
    if "first_response_time" in df.columns and "time_to_resolution" in df.columns:
        try:
            df["first_response_time"] = pd.to_datetime(df["first_response_time"], errors="coerce")
            df["time_to_resolution"] = pd.to_datetime(df["time_to_resolution"], errors="coerce")
            df["resolution_hours"] = (df["time_to_resolution"] - df["first_response_time"]).dt.total_seconds() / 3600
        except Exception:
            df["resolution_hours"] = None

    # Features and target
    X = df.drop(columns=[
        "customer_satisfaction_rating", "ticket_id", "customer_name", "customer_email",
        "ticket_subject", "ticket_description", "resolution", "date_of_purchase"
    ], errors="ignore")

    y = df["customer_satisfaction_rating"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    # Example usage
    df = pd.read_csv("../data/raw/customer_data.csv")
    processed_df, encoders = preprocess_data(df)
    print(processed_df.head())
    print(processed_df.info())
    print(processed_df.describe())
    print(processed_df.columns)

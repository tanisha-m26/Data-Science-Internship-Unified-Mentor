import pickle
from sklearn.ensemble import RandomForestClassifier
from data_load import load_data
from preprocess import preprocess_data

def train_model(data_path: str, model_path: str):
    """
    Train RandomForestClassifier and save using pickle
    """
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    # Save model + encoders
    with open(model_path, "wb") as f:
        pickle.dump({"model": model, "encoders": encoders, "features": list(X_train.columns)}, f)

    print(f"✅ Model trained and saved to {model_path}")


if __name__ == "__main__":
    train_model("../data/customer_tickets.csv", "../models/model.pkl")

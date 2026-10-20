import pickle
from sklearn.metrics import accuracy_score, classification_report
from data_load import load_data
from preprocess import preprocess_data

def evaluate_model(data_path: str, model_path: str):
    """
    Load trained model and evaluate performance on test set
    """
    df = load_data(data_path)
    X_train, X_test, y_train, y_test, encoders = preprocess_data(df)

    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]

    y_pred = model.predict(X_test)

    print("✅ Evaluation Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))


if __name__ == "__main__":
    evaluate_model("../data/customer_tickets.csv", "../models/model.pkl")

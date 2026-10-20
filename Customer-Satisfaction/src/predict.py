import pickle
import pandas as pd

def predict_single(input_dict: dict, model_path: str) -> int:
    """
    Predict satisfaction rating for a single ticket
    """
    with open(model_path, "rb") as f:
        saved = pickle.load(f)
    model = saved["model"]
    encoders = saved["encoders"]
    features = saved["features"]

    # Create DataFrame
    df = pd.DataFrame([input_dict])

    # Encode categorical inputs
    for col, encoder in encoders.items():
        if col in df.columns:
            df[col] = encoder.transform(df[col].astype(str))

    # Align columns
    df = df.reindex(columns=features, fill_value=0)

    prediction = model.predict(df)
    return int(prediction[0])


if __name__ == "__main__":
    sample_input = {
        "customer_age": 32,
        "customer_gender": "Female",
        "product_purchased": "GoPro Hero",
        "ticket_type": "Technical issue",
        "ticket_priority": "Critical",
        "ticket_channel": "Social media",
        "ticket_status": "Closed",
        "first_response_time": "2023-06-01 12:15",
        "time_to_resolution": "2023-06-01 18:05"
    }
    result = predict_single(sample_input, "../models/model.pkl")
    print("Predicted Satisfaction Rating:", result)
    print("✅ Prediction complete")
    
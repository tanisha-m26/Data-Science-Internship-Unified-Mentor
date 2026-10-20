import pandas as pd
import yaml

def load_config(path="../config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    # Ensure consistent column names
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

    

if __name__ == "__main__":
    config = load_config()
    df = load_data(config["data"]["raw_csv"])
    df = load_data("../data/customer_tickets.csv")
    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.columns)



    
"""Use the project's main functionalities using real world data"""
import requests
import json
import argparse
import sys
import pprint

LATITUDE = 37.759586
LONGITUDE = 126.777767
ALTITUDE = 38.0

train_data_path = "datasets/training_input.json"
url = "http://localhost:8010/"

panel_metadata = {
    "inverter_id": '1',
    "plant_id": '1',
    "latitude": LATITUDE,
    "longitude": LONGITUDE,
    "altitude": ALTITUDE
}

def train():
    with open(train_data_path, 'r') as f:
        train_data = json.load(f)
    
    train_api_input = {
        "panel_metadata": panel_metadata,
        "panel_output":train_data
    }
    endpoint = url + "train"
    response = requests.post(endpoint, json=train_api_input)
    if response.status_code == 200:
        print(f"Train ended successfully")
    else:
        print("There was an error in training")
    return 0

def predict(days: int):
    panel_metadata['predict_days'] = days
    endpoint = url + "predict"
    response = requests.post(endpoint, json=panel_metadata)
    if response.status_code == 200:
        print(f"Predict ended successfully")
        pprint.pprint(f"Predictions: {response.json()}")
        return response.json()
    else:
        print("There was an error during prediction")

def main():
    parser = argparse.ArgumentParser(description="Train and run predictions on solar prediction AI")
    
    parser.add_argument("--train", action="store_true", help="Train the model")

    parser.add_argument("--predict", action="store_true", help="Make predictions with the model")

    parser.add_argument(
        "--days", help="Number of days to make predictions for, starting from tomorrow's date"
    )

    args = parser.parse_args()
    if args.train:
        train()
    elif args.predict:
        if not args.days:
            raise ValueError("The argument --days is required for prediction")
        predict(args.days)


if __name__ == "__main__":
    sys.exit(main())
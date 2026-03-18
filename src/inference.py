import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# root dir
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "iris_rf_model.pkl"
FEATURE_NAMES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

# load model
try:
    model = joblib.load(MODEL_PATH)
    print("Load model successfully!")
except Exception as e:
    print(f"ERROR: {e}")
    model = None

# mapping class id to class name
TARGET_NAMES = {0: "setosa", 1: "versicolor", 2: "virginica"}


def predict_iris(features):

    if model is None:
        return {"ERROR": "Model is not ready."}

    try:
        input_df = pd.DataFrame([features], columns=FEATURE_NAMES)

        # predicting
        prediction = model.predict(input_df)

        return {
            "class_id": int(prediction[0]),
            "species": TARGET_NAMES.get(int(prediction[0])),
        }

    except Exception as e:
        return {"ERROR": f"Error during prediction: {str(e)}"}

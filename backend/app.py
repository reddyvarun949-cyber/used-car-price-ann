"""Flask API for the Used Car Price (ANN) project.

Run from the project root:
    python backend/app.py
The API starts on http://localhost:5000

Endpoints
    GET  /api/health    -> is the API up, and which neural network is being used?
    GET  /api/options   -> allowed values and ranges (the frontend builds its form from these)
    POST /api/predict   -> JSON with listing details -> ANN price estimate + gradient boosting estimate

Which neural network is used?
    1. models/car_price_ann.keras (Keras) if that file exists AND TensorFlow is installed,
    2. otherwise models/car_price_mlp_sklearn.joblib (scikit-learn MLPRegressor, always available).
    Set CAR_MODEL=sklearn to force the scikit-learn model.

All models are trained in notebook/Used_Car_Price_ANN.ipynb and loaded ONCE when the server starts.
We never retrain on a request.
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.json")
MLP_PATH = os.path.join(MODELS_DIR, "car_price_mlp_sklearn.joblib")
GB_PATH = os.path.join(MODELS_DIR, "car_price_gb.joblib")
KERAS_PATH = os.path.join(MODELS_DIR, "car_price_ann.keras")
PREPROCESSOR_PATH = os.path.join(MODELS_DIR, "preprocessor.joblib")

# Which website may call this API from a browser (the Vite dev server by default).
ALLOWED_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")

app = Flask(__name__)

# ---- load everything once at start-up ----
for path in (INFO_PATH, MLP_PATH, GB_PATH):
    if not os.path.exists(path):
        raise SystemExit("Model files not found. Run notebook/Used_Car_Price_ANN.ipynb first to create models/.")
with open(INFO_PATH) as f:
    info = json.load(f)

NUMERIC = info["numeric_features"]           # car_age, km_driven, owners, mileage_kmpl, engine_cc
CATEGORICAL = info["categorical_features"]   # brand, model_segment, fuel, transmission, city, seller_type
FEATURES = NUMERIC + CATEGORICAL


def load_ann():
    """Return (predict_function, model_name). predict_function takes a DataFrame and returns prices in lakhs."""
    if os.environ.get("CAR_MODEL", "auto") != "sklearn":
        if not os.path.exists(KERAS_PATH):
            print(f"[startup] {os.path.relpath(KERAS_PATH, BASE_DIR)} not found "
                  "(run the Keras cells of the notebook to create it). Using the scikit-learn MLP.")
        else:
            try:
                from tensorflow import keras  # imported only here: TensorFlow is optional
                keras_model = keras.models.load_model(KERAS_PATH)
                preprocessor = joblib.load(PREPROCESSOR_PATH)
                print("[startup] Using the Keras ANN from models/car_price_ann.keras")

                def keras_predict(frame):
                    ready = preprocessor.transform(frame).astype("float32")
                    log_price = keras_model.predict(ready, verbose=0).ravel()
                    return np.expm1(log_price)               # the network predicts log(1 + price)
                return keras_predict, info["models"]["ann_keras"]["name"]
            except ImportError:
                print("[startup] models/car_price_ann.keras exists but TensorFlow is not installed. "
                      "Using the scikit-learn MLP.")

    mlp = joblib.load(MLP_PATH)                 # whole pipeline: preprocessing + MLP + log/exp of the target
    print("[startup] Using the scikit-learn MLP from models/car_price_mlp_sklearn.joblib")
    return mlp.predict, info["models"]["ann_sklearn"]["name"]


ann_predict, ANN_NAME = load_ann()
gb_model = joblib.load(GB_PATH)
GB_NAME = info["models"]["gradient_boosting"]["name"]
# Error size used for the likely range. The Keras model was not evaluated in this build,
# so we use the scikit-learn ANN's test error for both (they are the same kind of model).
ANN_MAPE = info["models"]["ann_sklearn"]["test_metrics"]["MAPE_pct"]


@app.after_request
def add_cors_headers(response):
    """Allow the React app (a different port = a different 'origin') to call this API."""
    response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


def is_empty(value):
    return value is None or (isinstance(value, str) and value.strip() == "")


def read_number(payload, field, low, high, errors, whole=False):
    """Read one number from the request. Adds a message to errors and returns None if it is not valid."""
    value = payload.get(field)
    if is_empty(value):
        errors[field] = "This field is required."
        return None
    if isinstance(value, bool):
        errors[field] = "Must be a number."
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        errors[field] = "Must be a number."
        return None
    if math.isnan(number) or math.isinf(number):
        errors[field] = "Must be a real number."
    elif whole and not number.is_integer():
        errors[field] = "Must be a whole number."
    elif not (low <= number <= high):
        errors[field] = f"Must be between {low:g} and {high:g} (the range the model was trained on)."
    else:
        return number
    return None


def validate(payload):
    """Return (feature_row, errors). errors is a dict: field -> message a person can understand."""
    errors, row = {}, {}
    if not isinstance(payload, dict):
        return None, {"body": "Send a JSON object with the car details."}

    # --- categories (brand, segment, fuel, ...) ---
    for field in CATEGORICAL:
        allowed = info["categories"][field]
        value = payload.get(field)
        if is_empty(value):
            errors[field] = "This field is required."
            continue
        text = info["city_aliases"].get(str(value).strip(), str(value).strip()) if field == "city" else str(value).strip()
        if text not in allowed:
            errors[field] = f"Unknown value '{value}'. Choose one of: {', '.join(allowed)}."
        else:
            row[field] = text
    is_electric = row.get("fuel") == "Electric"

    # --- year -> car_age ---
    first_year, last_year = info["year_range"]
    year = read_number(payload, "year", first_year, last_year, errors, whole=True)
    if year is not None:
        if is_electric and year < info["electric_min_year"]:
            errors["year"] = (f"Electric cars in our data start from {info['electric_min_year']}. "
                              "Please check the year or the fuel type.")
        else:
            row["car_age"] = info["data_year"] - int(year)

    # --- other numbers ---
    ranges = info["numeric_ranges"]
    km = read_number(payload, "km_driven", *ranges["km_driven"], errors)
    if km is not None:
        row["km_driven"] = km
    owners = read_number(payload, "owners", *ranges["owners"], errors, whole=True)
    if owners is not None:
        row["owners"] = owners

    # Mileage and engine size do not exist for electric cars; for other fuels mileage is required.
    if is_electric:
        for field in ("mileage_kmpl", "engine_cc"):
            if not is_empty(payload.get(field)):
                errors[field] = "Electric cars have no fuel mileage or engine size. Leave this empty."
            row[field] = np.nan          # the pipeline fills it and flags it as missing, as in training
    else:
        mileage = read_number(payload, "mileage_kmpl", *ranges["mileage_kmpl"], errors)
        if mileage is not None:
            row["mileage_kmpl"] = mileage
        if is_empty(payload.get("engine_cc")):
            row["engine_cc"] = np.nan    # optional: unknown engine size is filled with the training median
        else:
            engine = read_number(payload, "engine_cc", *ranges["engine_cc"], errors)
            if engine is not None:
                row["engine_cc"] = engine
    return row, errors


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "ann_model": ANN_NAME, "comparison_model": GB_NAME,
                    "using_fallback": ANN_NAME == info["models"]["ann_sklearn"]["name"]})


@app.get("/api/options")
def options():
    return jsonify({
        "categories": info["categories"],
        "numeric_ranges": {
            "year": info["year_range"],
            "km_driven": info["numeric_ranges"]["km_driven"],
            "owners": info["numeric_ranges"]["owners"],
            "mileage_kmpl": info["numeric_ranges"]["mileage_kmpl"],
            "engine_cc": info["numeric_ranges"]["engine_cc"],
        },
        "electric_min_year": info["electric_min_year"],
        "ann_model": ANN_NAME,
        "comparison_model": GB_NAME,
        "test_metrics": {"ann": info["models"]["ann_sklearn"]["test_metrics"],
                         "gradient_boosting": info["models"]["gradient_boosting"]["test_metrics"]},
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":           # browser "preflight" check before a POST
        return ("", 204)
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be JSON (Content-Type: application/json)."}), 400

    row, errors = validate(payload)
    if errors:
        return jsonify({"error": "Some inputs are invalid.", "fields": errors}), 422

    features = pd.DataFrame([row], columns=FEATURES)
    ann_price = float(ann_predict(features)[0])
    gb_price = float(gb_model.predict(features)[0])
    share = ANN_MAPE / 100
    return jsonify({
        "predicted_price_lakhs": round(ann_price, 2),
        "likely_range_lakhs": [round(ann_price * (1 - share), 2), round(ann_price * (1 + share), 2)],
        "model": ANN_NAME,
        "comparison": {"model": GB_NAME, "predicted_price_lakhs": round(gb_price, 2)},
        "difference_pct": round(abs(ann_price - gb_price) / ann_price * 100, 1),
        "note": (f"On unseen test cars the neural network was off by about {ANN_MAPE:.0f}% on average, "
                 "so the range is the estimate ± that percentage."),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")

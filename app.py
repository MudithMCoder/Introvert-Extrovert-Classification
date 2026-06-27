import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

# Column definitions
num_cols = [
    "Time_spent_Alone",
    "Social_event_attendance",
    "Going_outside",
    "Friends_circle_size",
    "Post_frequency"
]
cat_cols = [
    "Stage_fear",
    "Drained_after_socializing"
]
all_cols = [
    "Time_spent_Alone",
    "Stage_fear",
    "Social_event_attendance",
    "Going_outside",
    "Drained_after_socializing",
    "Friends_circle_size",
    "Post_frequency"
]

#  Load all artifacts
def load_artifact(path):
    with open(path, "rb") as f:
        return pickle.load(f)

num_imputer      = load_artifact("artifacts/num_imputer.pkl")
cat_imputer      = load_artifact("artifacts/cat_imputer.pkl")
feature_encoders = load_artifact("artifacts/feature_encoders.pkl")
target_encoder   = load_artifact("artifacts/target_encoder.pkl")
model            = load_artifact("artifacts/model.pkl")

print("All artifacts loaded.")


#  Preprocessing pipeline 
def preprocess(raw: dict) -> np.ndarray:
    df_input = pd.DataFrame([raw], columns=all_cols)

    df_input[num_cols] = num_imputer.transform(df_input[num_cols])
    df_input[cat_cols] = cat_imputer.transform(df_input[cat_cols])

    for col in cat_cols:
        le = feature_encoders[col]
        try:
            df_input[col] = le.transform(df_input[col])
        except ValueError as e:
            raise ValueError(
                f"Unseen value in '{col}'. "
                f"Accepted values: {list(le.classes_)}. Error: {e}"
            )

    return df_input[all_cols].values


#  Routes 

@app.route("/", methods=["GET"])
def home():
    """Serve the frontend HTML page."""
    return send_from_directory(".", "templates/index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(force=True, silent=True)

    if body is None:
        return jsonify({"error": "Request body must be valid JSON"}), 400

    try:
        X = preprocess(body)

        pred_encoded  = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        confidence    = float(probabilities.max())

        pred_label = target_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            "prediction":  pred_label,
            "confidence":  round(confidence, 4),
            "all_classes": list(target_encoder.classes_)
        })

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 422

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")  # Ensure this file exists in your deployment

# Route for home page (renders HTML form)
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure correct content type
        if request.content_type != "application/json":
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        # Get JSON input from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Convert JSON to DataFrame
        df = pd.DataFrame([data])

        # Define the expected input columns for the model
        expected_columns = [
            "Rooms", "Distance", "Landsize", "BuildingArea", "YearBuilt"
        ]

        # Ensure only expected columns are sent to the model
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Make prediction using PyCaret
        prediction = predict_model(model, data=df)

        # Extract predicted price
        predicted_value = prediction["Label"].tolist()[0]

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run Flask app (for local testing)
if __name__ == "__main__":
    app.run(debug=True)

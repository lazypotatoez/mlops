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

        # Model expects these features (modify this list to match your trained model)
        expected_columns = [
            "Suburb", "Address", "Rooms", "Type", "Method", "Seller", "Date",
            "Distance", "Postcode", "Bedroom2", "Bathroom", "Car", "Landsize",
            "BuildingArea", "YearBuilt", "CouncilArea", "Lattitude", "Longtitude",
            "Region", "Propertycount"
        ]

        # Fill missing categorical fields with placeholder values
        df["Suburb"] = "Unknown"
        df["Address"] = "Unknown"
        df["Type"] = "h"  # Default to "house"
        df["Method"] = "S"  # Default to "Sold"
        df["Seller"] = "GenericAgent"
        df["Date"] = "2025-01-01"
        df["Postcode"] = 3000
        df["Bedroom2"] = df["Rooms"]  # Use same value as Rooms
        df["Bathroom"] = 1
        df["Car"] = 1
        df["CouncilArea"] = "Generic Council"
        df["Lattitude"] = -37.8136  # Default Melbourne latitude
        df["Longtitude"] = 144.9631  # Default Melbourne longitude
        df["Region"] = "Northern Metropolitan"
        df["Propertycount"] = 5000

        # Ensure dataframe has the required columns
        df = df[expected_columns]

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

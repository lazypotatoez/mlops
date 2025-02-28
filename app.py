from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")  # Ensure your model file exists

# Expected feature columns for the model
EXPECTED_COLUMNS = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date',
                    'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                    'BuildingArea', 'YearBuilt', 'CouncilArea', 'Latitude', 'Longitude',
                    'Region', 'Propertycount']

# Default values for missing columns
DEFAULT_VALUES = {
    "Suburb": "Unknown",
    "Address": "Unknown",
    "Type": "h",
    "Method": "S",
    "Seller": "Agent",
    "Date": "2025-01-01",
    "Postcode": 3000,
    "Bedroom2": 3,
    "Bathroom": 1,
    "Car": 1,
    "CouncilArea": "Unknown",
    "Latitude": -37.8136,
    "Longitude": 144.9631,
    "Region": "Metropolitan",
    "Propertycount": 5000
}

@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Fill in missing values
        for key, default in DEFAULT_VALUES.items():
            if key not in data:
                data[key] = default  # Assign default value if missing

        df = pd.DataFrame([data])

        # Ensure correct feature columns
        df = df.reindex(columns=EXPECTED_COLUMNS)

        # Make prediction
        prediction = predict_model(model, data=df)
        predicted_value = prediction["Label"].tolist()

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

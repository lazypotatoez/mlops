from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure correct content type
        if request.content_type != "application/json":
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Extract only relevant input fields
        input_features = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
        df = pd.DataFrame([data])

        # Define default values for missing categorical fields
        default_values = {
            "Suburb": "Melbourne",
            "Address": "123 Main St",
            "Type": "h",
            "Method": "S",
            "Seller": "RealEstateAgent",
            "Date": "2025-01-01",
            "Postcode": 3000,
            "Bedroom2": data.get("Rooms", 2),  # Use 'Rooms' as fallback
            "Bathroom": 1,
            "Car": 1,
            "CouncilArea": "Melbourne City",
            "Lattitude": -37.8136,
            "Longtitude": 144.9631,
            "Region": "Northern Metropolitan",
            "Propertycount": 5000
        }

        # Fill missing fields with default values
        for key, value in default_values.items():
            if key not in df.columns:
                df[key] = value

        # Ensure numeric columns are floats
        numeric_columns = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt', 'Bedroom2', 'Bathroom', 'Car', 'Postcode', 'Lattitude', 'Longtitude', 'Propertycount']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Ensure categorical fields are strings
        categorical_columns = ["Suburb", "Address", "Type", "Method", "Seller", "CouncilArea", "Region"]
        df[categorical_columns] = df[categorical_columns].astype(str)

        # Order the dataframe columns to match model expectations
        expected_columns = [
            'Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
            'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 
            'Region', 'Propertycount'
        ]
        df = df.reindex(columns=expected_columns, fill_value=0)

        # Make prediction
        prediction = predict_model(model, data=df)
        predicted_value = prediction["Label"].tolist()[0]  # Extract price

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

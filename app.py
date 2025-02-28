from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")  # Ensure this file exists

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure request contains JSON
        if not request.is_json:
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        # Parse JSON input
        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Expected feature names
        expected_columns = [
            'Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
            'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 
            'Region', 'Propertycount'
        ]

        # Ensure correct column order
        df = df.reindex(columns=expected_columns, fill_value=0)

        # **Ensure categorical data is processed correctly**
        df = df.astype(str)  # Convert all columns to string to avoid NaN errors

        # Predict using the PyCaret model
        prediction = predict_model(model, data=df)
        predicted_price = prediction["Label"].tolist()[0]  # Extract price

        return jsonify({"predicted_price": predicted_price})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

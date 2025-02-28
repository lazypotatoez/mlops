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

        # Convert to DataFrame
        df = pd.DataFrame([data])

        # Ensure numeric columns are properly formatted
        numeric_columns = ['Rooms', 'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize',
                           'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude', 'Propertycount']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

        # Expected model input features
        expected_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
                            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                            'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 
                            'Region', 'Propertycount']

        # Reindex to match model’s expected columns
        df = df.reindex(columns=expected_columns, fill_value="Unknown")

        # Make prediction
        prediction = predict_model(model, data=df)
        predicted_value = prediction["Label"].tolist()[0]

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

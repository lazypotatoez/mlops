from flask import Flask, render_template, request, jsonify
import pandas as pd
from pycaret.regression import load_model, predict_model

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = load_model("best_pycaret_model")  # Ensure your model file exists

# Route for home page
@app.route("/")
def home():
    return render_template("index.html")

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.content_type != "application/json":
            return jsonify({"error": "Unsupported Media Type: Content-Type must be application/json"}), 415

        data = request.get_json()
        if not data:
            return jsonify({"error": "Invalid JSON data"}), 400

        df = pd.DataFrame([data])

        # Expected model input features
        expected_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
                            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                            'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 
                            'Region', 'Propertycount']

        df = df.reindex(columns=expected_columns, fill_value=0)

        # Use predict_model() for PyCaret predictions
        prediction = predict_model(model, data=df)
        predicted_value = prediction["Label"].tolist()

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})


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
        # Receive input data as JSON
        data = request.get_json()

        # Convert JSON input into a DataFrame
        df = pd.DataFrame([data])

        # Print incoming columns (for debugging)
        print("Received data columns:", df.columns.tolist())  

        # Expected model input features (check your trained model)
        expected_columns = ['Suburb', 'Address', 'Rooms', 'Type', 'Method', 'Seller', 'Date', 
                            'Distance', 'Postcode', 'Bedroom2', 'Bathroom', 'Car', 'Landsize', 
                            'BuildingArea', 'YearBuilt', 'CouncilArea', 'Lattitude', 'Longtitude', 
                            'Region', 'Propertycount']

        # Ensure the input matches expected columns
        df = df.reindex(columns=expected_columns, fill_value=0)  # Fill missing values with 0

        # Use predict_model() for PyCaret predictions
        prediction = predict_model(model, data=df)

        # Extract predicted value
        predicted_value = prediction["Label"].tolist()

        return jsonify({"prediction": predicted_value})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

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
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = {
            "Rooms": [float(request.form["Rooms"])],
            "Distance": [float(request.form["Distance"])],
            "Landsize": [float(request.form["Landsize"])],
            "BuildingArea": [float(request.form["BuildingArea"])],
            "YearBuilt": [float(request.form["YearBuilt"])]
        }

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(data)

        # Make predictions
        predictions = predict_model(model, data=df)

        # Return result
        return jsonify({"Predicted Price": float(predictions["Label"][0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

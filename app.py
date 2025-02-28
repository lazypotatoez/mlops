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
        # Collect data from the HTML form
        rooms = int(request.form.get("rooms"))
        distance = float(request.form.get("distance"))
        landsize = float(request.form.get("landsize"))
        building_area = float(request.form.get("building_area"))
        year_built = int(request.form.get("year_built"))

        # Create a DataFrame with only the required features
        df = pd.DataFrame([{
            "Rooms": rooms,
            "Distance": distance,
            "Landsize": landsize,
            "BuildingArea": building_area,
            "YearBuilt": year_built
        }])

        # Use PyCaret model for prediction
        prediction = predict_model(model, data=df)
        predicted_value = prediction["Label"].iloc[0]  # Get first prediction

        return render_template("index.html", prediction=predicted_value)

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

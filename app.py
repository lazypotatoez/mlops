from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
from pycaret.regression import load_model

# ✅ Load the trained PyCaret model correctly
model_path = "house_price_model_tuned"
model = load_model(model_path)  # Corrected model loading

# Initialize Flask app
app = Flask(__name__)

# Define the home route
@app.route("/")
def home():
    return render_template("index.html")

# Define the prediction route
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        input_data = {
            "Rooms": int(request.form["Rooms"]),
            "Bathroom": float(request.form["Bathroom"]),
            "Car": float(request.form["Car"]),
            "Landsize": float(request.form["Landsize"]),
            "BuildingArea": float(request.form["BuildingArea"]),
            "YearBuilt": float(request.form["YearBuilt"]),
            "Distance": float(request.form["Distance"]),
            "Lattitude": float(request.form["Lattitude"]),
            "Longtitude": float(request.form["Longtitude"]),
            "Type": request.form["Type"],
            "Region": request.form["Region"]
        }

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # ✅ Use PyCaret's `predict_model()` instead of `.predict()`
        prediction = predict_model(model, data=input_df)["Label"][0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ${round(prediction, 2)}")

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)

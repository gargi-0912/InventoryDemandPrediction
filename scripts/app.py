
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
from flask import Flask, request, jsonify

# Flask App Initialization
app = Flask(__name__)

# Step 1: Load the Trained Model
model_path = r'C:\Users\gargi\InventoryDemandPrediction\scripts\model.pkl'
model = joblib.load(model_path)

@app.route('/')
def home():
    return "Inventory Demand Prediction API is running!"

# Step 2: Prediction API Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input data from the POST request
        data = request.json
        day_of_week = data.get('day_of_week')
        is_holiday = data.get('is_holiday')
        is_festival = data.get('is_festival')
        prev_demand = data.get('prev_demand')

        # Validate input data
        if day_of_week is None or is_holiday is None or is_festival is None or prev_demand is None:
            return jsonify({'error': 'Missing input fields!'}), 400

        # Prepare the input for the model
        input_data = pd.DataFrame({
            'day_of_week': [day_of_week],
            'is_holiday': [is_holiday],
            'is_festival': [is_festival],
            'prev_demand': [prev_demand]
        })

        # Predict demand
        predicted_demand = model.predict(input_data)[0]
        return jsonify({'predicted_demand': predicted_demand}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask App
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


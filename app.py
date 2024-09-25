from flask import Flask, request, render_template, jsonify
import numpy as np
from Linear_regression import Linear_regression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

app = Flask(__name__)

# Load and train the model
housing = fetch_california_housing()
X = housing.data
y = housing.target

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = Linear_regression()
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form

        required_fields = ['Median_Income', 'House_Age', 'Avg_Number_of_Rooms', 'Avg_Number_of_Bedrooms',
                           'Population', 'Avg_Occupancy', 'Latitude', 'Longitude']

        features = []
        for field in required_fields:
            value = form_data.get(field)
            if not value:
                return jsonify({"error": f"{field} is required"}), 400
            features.append(float(value))

        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)[0] * 100000
        prediction_rf = model.predict_rf(features)[0] * 100000

        return jsonify({
            "prediction": f"{prediction:,.2f}$",
            "prediction_rf": f"{prediction_rf:,.2f}$"
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)

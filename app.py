from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the dataset
data = pd.read_excel("sensor.xlsx")

# Encode categorical variables
label_encoder = LabelEncoder()
data['Grain'] = label_encoder.fit_transform(data['Grain'])

# Split the data into features and target variable
X = data[['Grain', 'Moisture']]
y = data['Price']

# Initialize the RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    grain = request.form['grain']
    moisture = float(request.form['moisture'])

    # Print the encoded categories
    print("Encoded categories:", label_encoder.classes_)

    # Convert grain to encoded value
    grain_encoded = label_encoder.transform([grain])[0]

    # Debugging prints
    print("Grain encoded:", grain_encoded)
    print("Moisture level:", moisture)

    # Predict the price
    price = model.predict([[grain_encoded, moisture]])

    # Return JSON response
    return jsonify({
        'grain': grain,
        'moisture': moisture,
        'price': price[0]
    })

# if __name__ == '__main__':
#     app.run(debug=True,port=8080)

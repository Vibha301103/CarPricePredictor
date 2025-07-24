from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('LinearRegression.pkl', 'rb'))

# Load the cleaned car dataset
car = pd.read_csv("Cleaned Car.csv")

@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    years = sorted(car["year"].unique(), reverse=True)
    fuel_types = sorted(car["fuel_type"].unique())
    companies.insert(0, "Select Company")
    return render_template('index.html', companies=companies, car_models=car_models, years=years, fuel_types=fuel_types)

@app.route('/get_car_model', methods=['POST'])
def get_car_model():
    company = request.form.get('company')
    models = sorted(car[car['company'] == company]['name'].unique())
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('name')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    # Ensure the order of features matches model training
    input_data = pd.DataFrame([[car_model, company, year, kms_driven, fuel_type]],
                              columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'])

    prediction = model.predict(input_data)
    return str(np.round(prediction[0], 2))

if __name__ == "__main__":
    app.run(debug=True)

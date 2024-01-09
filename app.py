from flask import Flask, render_template, request
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# WSGI Application
app = Flask(__name__)

model = pickle.load(open('rfr_model.pkl', 'rb'))
standard_to = StandardScaler()

@app.route('/', methods = ['GET'])
def home():
    return render_template("prediction.html")

# Form inputs
@app.route('/predict', methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        params = []
        Year = int(request.form['Year'])
        Present_price = float(request.form['PresentPrice'])
        Kms = int(request.form['Kilometers'])
        owners = int(request.form['Owners'])

        fuel_type_petrol = request.form['FuelTypePetrol']
        if fuel_type_petrol == 'Petrol':
            Fuel_Type_Petrol = 1
            Fuel_Type_Diesel = 0
        else:
            Fuel_Type_Petrol = 0
            Fuel_Type_Diesel = 1

        Year = 2020 - Year

        Seller_Type_Individual = request.form['SellerType']
        if Seller_Type_Individual == 'Individual':
            Seller_Type_Individual = 1
        else:
            Seller_Type_Individual = 0

        Transmission_Manual = request.form['TransmissionManual']
        if Transmission_Manual == 'Manual':
            Transmission_Manual = 1
        else :
            Transmission_Manual = 0

        prediction = model.predict([[Present_price, Kms, owners,
                                    Year, Fuel_Type_Diesel,Fuel_Type_Petrol, Seller_Type_Individual,
                                    Transmission_Manual]])

        output = round(prediction[0], 2)
        if output < 0:
            return render_template('prediction.html', prediction_text="Sorry you cannot sell this car")
        else:
            return render_template('prediction.html', prediction_text="You Can Sell The Car at {} lakh.".format(output))
    else:
        return render_template('prediction.html')

if __name__ == '__main__':
    app.run(debug = True)

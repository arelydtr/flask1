from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo
model = joblib.load('model/pryhouse.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        try:
            # Obtener los datos del formulario
            bedrooms = float(request.form['bedrooms'])
            bathrooms = float(request.form['bathrooms'])
            sqft_living = float(request.form['sqft_living'])
            sqft_lot = float(request.form['sqft_lot'])
            waterfront = float(request.form['waterfront'])
            view = float(request.form['view'])
            condition = float(request.form['condition'])
            grade = float(request.form['grade'])
            sqft_above = float(request.form['sqft_above'])
            sqft_basement = float(request.form['sqft_basement'])
            yr_built = float(request.form['yr_built'])
            yr_renovated = float(request.form['yr_renovated'])
            zipcode = float(request.form['zipcode'])
            lat = float(request.form['lat'])
            long = float(request.form['long'])
            sqft_living15 = float(request.form['sqft_living15'])
            sqft_lot15 = float(request.form['sqft_lot15'])

            # Crear un array numpy con los datos
            input_data = np.array([[bedrooms, bathrooms, sqft_living, sqft_lot, waterfront, view, condition, grade, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, lat, long, sqft_living15, sqft_lot15]])

            # Realizar la predicción
            prediction = model.predict(input_data)
            result = prediction[0]

            prediction_text = f'Precio de la casa: ${result:.2f}'

        except ValueError:
            prediction_text = 'Error: Por favor, ingrese valores válidos.'

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)

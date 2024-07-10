from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('model_forest.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [
        data['sqft_living'], data['view'], data['grade'], 
        data['lat'], data['long'], data['sqft_living15']
    ]
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

@app.route('/metrics', methods=['GET'])
def metrics():
    # Cargar datos para el cálculo de métricas
    url = "https://raw.githubusercontent.com/arelydtr/PryectoMes2/main/kc_house.csv"
    data = pd.read_csv(url)
    data['date'] = pd.to_datetime(data['date']).astype('int64') // 10**9  # Convertir fecha a segundos desde la época

    X = data.drop(['price', 'id'], axis=1)
    Y = data['price']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.2, random_state=42)
    
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    return jsonify({
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R^2': r2
    })

if __name__ == '__main__':
    app.run(debug=True)

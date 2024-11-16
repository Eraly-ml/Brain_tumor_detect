import pickle
from flask import Flask, request, jsonify
import numpy as np

# Загружаем модель
with open('model/brain_tumor_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Инициализация Flask приложения
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # Получаем данные в формате JSON
    features = np.array(data["features"]).reshape(1, -1)  # Преобразуем данные в массив
    prediction = model.predict(features)  # Получаем предсказание
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)

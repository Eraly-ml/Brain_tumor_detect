from fastapi import FastAPI
from pydantic import BaseModel
import gdown
import pickle
import numpy as np
import os

app = FastAPI()

# URL модели на Google Drive
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1Cgx6JM18e3RZ0ADNqyBuSgiauvkIIle5'

# Путь для сохранения загруженной модели
model_path = "model/brain_tumor_model.pkl"

# Функция для загрузки модели из Google Drive
def download_model():
    if not os.path.exists(model_path):
        gdown.download(MODEL_URL, model_path, quiet=False)
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# Загружаем модель при старте приложения
model = download_model()

# Модель для входных данных
class PredictionRequest(BaseModel):
    features: list

@app.post("/predict")
async def predict(request: PredictionRequest):
    features = np.array(request.features).reshape(1, -1)  # Преобразуем данные в массив
    prediction = model.predict(features)  # Получаем предсказание
    return {"prediction": prediction[0]}


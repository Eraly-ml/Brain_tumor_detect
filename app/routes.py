from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import gdown
import torch
from fastai.vision.all import load_learner
from io import BytesIO
from PIL import Image
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
    return load_learner(model_path)

# Загружаем модель при старте приложения
learner = download_model()

# Модель для входных данных
class PredictionRequest(BaseModel):
    file: str  # Здесь предполагается, что файл будет передаваться

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Чтение изображения из файла
    image_data = await file.read()
    image = Image.open(BytesIO(image_data))

    # Получаем предсказание с использованием FastAI
    pred_class, pred_idx, outputs = learner.predict(image)

    # Вероятность для класса "Tumor"
    prob = torch.softmax(outputs, dim=0)[pred_idx].item()

    # Интерпретация предсказания
    prediction_label = "Tumor detected" if pred_class == 1 else "No tumor"
    
    return {
        "Prediction": prediction_label,
        "Probability": round(prob, 4)
    }

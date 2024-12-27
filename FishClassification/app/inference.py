# app/inference.py
from app.utils import *
from app.model import load_fish_model


def classify_fish(image_url):
    image_array = process_single_image(image_url)
    fish_classification_model = load_fish_model()
    pred_class = fish_classification_model.predict(image_array)  # Dự đoán loại cá từ các đặc trưng
    ocean_creature = process_label(pred_class)
    return ocean_creature

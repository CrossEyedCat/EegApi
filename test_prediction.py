import socket
import json
import joblib
import numpy as np

# 1. Загрузка обученной модели, scaler и энкодера меток
model = joblib.load('trained_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# 2. Параметры UDP
UDP_IP = '0.0.0.0'  # слушаем на всех интерфейсах
UDP_PORT = 2000  # порт для приема данных

# Создаем UDP-сокет
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
print("Режим прогнозирования запущен. Ожидание данных ЭЭГ...")

# Определяем список используемых признаков (должен соответствовать обучающему этапу)
feature_columns = ['EEG', 'd1', 't1', 't2', 'a1', 'a2', 'b1', 'b2', 'b3', 'g1']

while True:
    try:
        # Получение UDP-пакета
        data, addr = sock.recvfrom(1024)
        decoded = data.decode('utf-8')
        json_data = json.loads(decoded)

        # Извлечение признаков
        features = [json_data.get(col, 0) for col in feature_columns]
        X_sample = np.array(features).reshape(1, -1)  # преобразуем в массив с формой (1, число признаков)

        # Нормализация данных (используем scaler, обученный на тренировочных данных)
        X_scaled = scaler.transform(X_sample)

        # Прогнозирование класса
        y_pred = model.predict(X_scaled)

        predicted_label = label_encoder.inverse_transform(y_pred)[0]

        # Вывод результата
        print(f"Прогнозированное действие: {predicted_label}; {y_pred}")

    except Exception as e:
        print("Ошибка при прогнозировании:", e)
        continue
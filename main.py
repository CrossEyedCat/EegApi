import socket
import json
import csv
import time
import os

# Параметры UDP-соединения
UDP_IP = '0.0.0.0'  # слушаем на всех интерфейсах
UDP_PORT = 2000  # порт для приема данных

# Создаем и настраиваем UDP-сокет
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
sock.bind((UDP_IP, UDP_PORT))
print("Программа сбора данных ЭЭГ. Ожидание данных с устройства...")

# Путь к CSV-файлу и заголовок
csv_file = "eeg_data.csv"
header = ["EEG", "cum_time", "d1", "t1", "t2", "a1", "a2", "b1", "b2", "b3", "g1", "phase"]

# Если файл не существует, записываем заголовок
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    csv_writer = csv.writer(f)
    if not file_exists:
        csv_writer.writerow(header)

# Определяем фазы эксперимента
phases = [
    {"instruction": "Ни о чем не думать", "label": "none", "duration": 20},
    {"instruction": "Думать о нажатии правой кнопки", "label": "right", "duration": 20},
    {"instruction": "Думать о нажатии левой кнопки", "label": "left", "duration": 20},
]

# Цикл по фазам
for phase in phases:
    print(f"\nФаза: {phase['instruction']}")
    print("Приготовьтесь...")
    time.sleep(2)  # Краткий интервал, чтобы пользователь подготовился
    print("Старт!")

    phase_start = time.time()
    cumulative_time = 0.0  # Сбрасываем накопленное время для текущей фазы

    # Открываем CSV для дозаписи данных
    with open(csv_file, "a", newline="") as f:
        csv_writer = csv.writer(f)
        while (time.time() - phase_start) < phase['duration']:
            try:
                # Прием UDP-пакета
                data, addr = sock.recvfrom(1024)
                decoded = data.decode('utf-8')
                json_data = json.loads(decoded)

                # Извлекаем интервальное время и значение ЭЭГ
                dt = json_data.get("d", 0)
                eeg_val = json_data.get("E", 0)
                cumulative_time += dt

                # Извлекаем спектральные диапазоны: d1, t1, t2, a1, a2, b1, b2, b3, g1
                bands = [json_data.get(band, 0) for band in ['d1', 't1', 't2', 'a1', 'a2', 'b1', 'b2', 'b3', 'g1']]

                # Формируем строку для CSV: EEG, накопленное время, диапазоны и метка фазы
                row = [eeg_val, cumulative_time] + bands + [phase["label"]]
                csv_writer.writerow(row)

                print(f"EEG= {eeg_val} time= {cumulative_time} Bands= {bands}")
            except Exception as e:
                print("Ошибка при обработке данных:", e)
                continue

    print(f"Фаза '{phase['instruction']}' завершена. Отдых 5 секунд.")
    time.sleep(5)

print("Сбор данных завершен. Данные сохранены в", csv_file)
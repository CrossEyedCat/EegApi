from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import socket, json, csv, time, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

app = FastAPI(
    title="EEG Data Collection & ML API",
    description="API for EEG data collection, model training and real-time prediction",
    version="1.2"
)

# === Default configuration ===
DEFAULT_UDP_IP = '0.0.0.0'
DEFAULT_UDP_PORT = 2000
DEFAULT_CSV_FILE = 'eeg_data.csv'
MODEL_FILE = 'trained_model.pkl'
SCALER_FILE = 'scaler.pkl'
ENCODER_FILE = 'label_encoder.pkl'
HEADER = ["EEG", "cum_time", "d1", "t1", "t2", "a1", "a2", "b1", "b2", "b3", "g1", "phase"]

# === Pydantic models ===
class PhaseConfig(BaseModel):
    instruction: str
    label: str
    duration: float  # seconds

class CollectRequest(BaseModel):
    phases: list[PhaseConfig]
    udp_ip: str = DEFAULT_UDP_IP
    udp_port: int = DEFAULT_UDP_PORT
    csv_file: str = DEFAULT_CSV_FILE

class TrainResponse(BaseModel):
    accuracy: float
    report: str

class PredictResponse(BaseModel):
    predicted_label: str
    encoded: list[int]

# === Helper functions ===
def init_csv(csv_file: str):
    if not os.path.isfile(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(HEADER)


def collect_data(phases: list[PhaseConfig], udp_ip: str, udp_port: int, csv_file: str):
    init_csv(csv_file)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind((udp_ip, udp_port))

    for phase in phases:
        print(f"Starting phase: {phase.instruction} (label={phase.label}) for {phase.duration}s")
        time.sleep(2)
        start_t = time.time()
        cum_time = 0.0
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            while time.time() - start_t < phase.duration:
                data, _ = sock.recvfrom(1024)
                j = json.loads(data.decode('utf-8'))
                dt = j.get('d', 0)
                eeg = j.get('E', 0)
                cum_time += dt
                bands = [j.get(b, 0) for b in ['d1','t1','t2','a1','a2','b1','b2','b3','g1']]
                writer.writerow([eeg, cum_time, *bands, phase.label])
        print(f"Completed phase: {phase.label}")
        time.sleep(5)
    sock.close()
    print("Data collection complete.")

# === API endpoints ===
@app.post("/start-collection")
def start_collection(req: CollectRequest, background_tasks: BackgroundTasks):
    """
    Запуск сбора данных с параметрами:
    - phases: список конфигураций фаз
    - udp_ip, udp_port: адрес для приема UDP
    - csv_file: путь для сохранения CSV
    """
    background_tasks.add_task(collect_data, req.phases, req.udp_ip, req.udp_port, req.csv_file)
    return {"status": "collection started", "udp_ip": req.udp_ip, "udp_port": req.udp_port, "csv_file": req.csv_file}

@app.post("/train-model", response_model=TrainResponse)
def train_model():
    """
    Обучает RandomForest на собранных данных и сохраняет артефакты.
    """
    if not os.path.isfile(DEFAULT_CSV_FILE):
        raise HTTPException(status_code=400, detail="CSV data file not found. Collect data first.")
    df = pd.read_csv(DEFAULT_CSV_FILE)
    features = ['EEG','d1','t1','t2','a1','a2','b1','b2','b3','g1']
    X = df[features]
    y = df['phase']

    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_enc, test_size=0.15, random_state=42, stratify=y_enc
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    return {"accuracy": round(acc*100,2), "report": report}

@app.get("/predict", response_model=PredictResponse)
def predict_live():
    """
    Подключается к UDP-порту ЭЭГ, считывает один пакет и возвращает предсказание.
    """
    # Проверяем модель
    if not os.path.isfile(MODEL_FILE):
        raise HTTPException(status_code=400, detail="Model not found. Train model first.")
    # Получаем один пакет данных
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((DEFAULT_UDP_IP, DEFAULT_UDP_PORT))
    sock.settimeout(5.0)
    try:
        data, _ = sock.recvfrom(1024)
    except socket.timeout:
        raise HTTPException(status_code=408, detail="Timeout waiting for EEG data")
    finally:
        sock.close()

    # Разбор JSON
    j = json.loads(data.decode('utf-8'))
    features = [j.get('E',0)] + [j.get(b,0) for b in ['d1','t1','t2','a1','a2','b1','b2','b3','g1']]

    # Загрузка моделей
    clf = joblib.load(MODEL_FILE)
    scaler = joblib.load(SCALER_FILE)
    le = joblib.load(ENCODER_FILE)

    # Предсказание
    X = scaler.transform([features])
    y_pred = clf.predict(X)
    label = le.inverse_transform(y_pred)[0]

    return {"predicted_label": label, "encoded": y_pred.tolist()}

# Запуск: uvicorn api:app --reload

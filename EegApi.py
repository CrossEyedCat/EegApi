from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
import socket, json, time, os
import cupynumeric as cnp
from sqlmodel import Field, SQLModel, create_engine, Session, select
# CUDA-ускоренные ML-зависимости
import cudf
from cuml.preprocessing import StandardScaler as cuStandardScaler
from cuml.ensemble import RandomForestClassifier as cuRFClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# === Database setup ===
DATABASE_URL = "sqlite:///eeg_data.db"
engine = create_engine(DATABASE_URL, echo=False)

class EEGData(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    eeg: float
    cum_time: float
    d1: float
    t1: float
    t2: float
    a1: float
    a2: float
    b1: float
    b2: float
    b3: float
    g1: float
    phase: str

SQLModel.metadata.create_all(engine)

# === FastAPI init ===
app = FastAPI(
    title="EEG Data Collection & ML API (GPU + DB)",
    description="API using cuPyNumeric, cuDF, cuML for GPU acceleration and SQLite DB",
    version="2.1"
)

# === Default configuration ===
DEFAULT_UDP_IP = '0.0.0.0'
DEFAULT_UDP_PORT = 2000
MODEL_FILE = 'trained_model_gpu.pkl'
SCALER_FILE = 'scaler_gpu.pkl'
ENCODER_FILE = 'label_encoder_gpu.pkl'
FEATURES = ['EEG','d1','t1','t2','a1','a2','b1','b2','b3','g1']

# === Pydantic models ===
class PhaseConfig(BaseModel):
    instruction: str
    label: str
    duration: float  # seconds

class CollectRequest(BaseModel):
    phases: list[PhaseConfig]
    udp_ip: str = DEFAULT_UDP_IP
    udp_port: int = DEFAULT_UDP_PORT

class TrainResponse(BaseModel):
    accuracy: float
    report: str

class PredictResponse(BaseModel):
    predicted_label: str
    encoded: list[int]

# === Data collection ===
def collect_data(phases: list[PhaseConfig], udp_ip: str, udp_port: int):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.bind((udp_ip, udp_port))

    for phase in phases:
        print(f"Starting phase: {phase.instruction} (label={phase.label}) for {phase.duration}s")
        time.sleep(2)
        start_t = time.time()
        cum_time = 0.0
        with Session(engine) as session:
            while time.time() - start_t < phase.duration:
                data, _ = sock.recvfrom(1024)
                j = json.loads(data.decode('utf-8'))
                dt = j.get('d', 0)
                eeg = j.get('E', 0)
                cum_time += dt
                bands = {b: j.get(b, 0) for b in FEATURES[1:]}
                record = EEGData(
                    eeg=eeg,
                    cum_time=cum_time,
                    **bands,
                    phase=phase.label
                )
                session.add(record)
            session.commit()
        print(f"Completed phase: {phase.label}")
        time.sleep(5)
    sock.close()
    print("Data collection complete.")

# === API endpoints ===
@app.post("/start-collection")
def start_collection(req: CollectRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(collect_data, req.phases, req.udp_ip, req.udp_port)
    return {"status": "collection started", "udp_ip": req.udp_ip, "udp_port": req.udp_port}

@app.post("/train-model", response_model=TrainResponse)
def train_model():
    # Load from DB
    with Session(engine) as session:
        results = session.exec(select(EEGData)).all()
    if not results:
        raise HTTPException(status_code=400, detail="No data in database. Collect first.")

    # Создаем cuDF DataFrame для GPU
    df = cudf.DataFrame([r.dict() for r in results])
    X_gpu = df[FEATURES].to_cupy()
    phases_cpu = df['phase'].to_pandas()

    # Label encoding на CPU
    le = LabelEncoder()
    y_enc = le.fit_transform(phases_cpu)

    # Массив меток на GPU
    y_gpu = cnp.array(y_enc)

    # Масштабирование на GPU
    scaler = cuStandardScaler()
    X_scaled_gpu = scaler.fit_transform(X_gpu)

    # Обучение RandomForest на GPU
    clf = cuRFClassifier(n_estimators=100, random_state=42)
    clf.fit(X_scaled_gpu, y_gpu)

    # Оценка: переносим тестовые на CPU
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    X_cpu = cnp.asnumpy(X_scaled_gpu)
    y_cpu = y_enc
    X_train, X_test, y_train, y_test = train_test_split(
        X_cpu, y_cpu, test_size=0.15, random_state=42, stratify=y_cpu
    )
    y_pred = clf.predict(cnp.array(X_test)).get()  # перевод результата на CPU

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_)

    # Сохраняем артефакты
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    joblib.dump(le, ENCODER_FILE)

    return {"accuracy": round(acc * 100, 2), "report": report}

@app.get("/predict", response_model=PredictResponse)
def predict_live():
    # Получаем один UDP-пакет
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((DEFAULT_UDP_IP, DEFAULT_UDP_PORT))
    sock.settimeout(5.0)
    try:
        data, _ = sock.recvfrom(1024)
    except socket.timeout:
        raise HTTPException(status_code=408, detail="Timeout waiting for EEG data")
    finally:
        sock.close()

    j = json.loads(data.decode('utf-8'))
    feats = [j.get('E',0)] + [j.get(b,0) for b in FEATURES[1:]]

    # GPU-преобразование
    arr_gpu = cnp.array(feats).reshape(1, -1)
    scaler = joblib.load(SCALER_FILE)
    X_scaled = scaler.transform(arr_gpu)

    clf = joblib.load(MODEL_FILE)
    y_gpu_pred = clf.predict(X_scaled)
    y_cpu = y_gpu_pred.get()

    le = joblib.load(ENCODER_FILE)
    label = le.inverse_transform(y_cpu)[0]

    return {"predicted_label": label, "encoded": y_cpu.tolist()}

# Запуск: uvicorn api_cupynumeric_db:app --reload

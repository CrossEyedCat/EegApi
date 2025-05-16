# EEG Data Collection & ML API (GPU + DB)

This project provides a **FastAPI** service for collecting EEG data over UDP, storing it in an SQLite database, and performing **GPU-accelerated** machine learning using NVIDIA RAPIDS libraries (**cuPyNumeric**, **cuDF**, **cuML**).

## Features

* **Data Collection**: Receive real-time EEG data via UDP and save to SQLite with SQLModel.
* **GPU Processing**: Move data into GPU arrays (`cupynumeric`) and DataFrames (`cudf`).
* **GPU ML**: Scale features and train a RandomForest classifier on GPU (`cuml`).
* **Live Prediction**: Fetch a single EEG packet and predict phase in real time.

## Requirements

* Python 3.10+
* NVIDIA GPU with CUDA toolkit installed
* Dependencies (see `requirements.txt`):

  ```text
  fastapi
  uvicorn
  pydantic
  sqlmodel
  sqlite
  cupynumeric
  cudf
  cuml
  scikit-learn
  joblib
  ```

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/CrossEyedCat/EegApi.git
   cd eeg-gpu-ml-api
   ```
2. Create a Python venv and activate:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
4. Ensure CUDA runtime and RAPIDS are set up per [RAPIDS Installation Guide](https://docs.nvidia.com/cudf/).

## Docker Compose

Для быстрого запуска вместе с SQLite можно использовать `docker-compose`:

```yaml
version: '3.8'
services:
  api:
    image: python:3.10-slim
    container_name: eeg_api
    working_dir: /app
    volumes:
      - ./:/app
    command: uvicorn api_cupynumeric_db:app --host 0.0.0.0 --port 8000 --reload
    environment:
      - DATABASE_URL=sqlite:///./eeg_data.db
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    runtime: nvidia

  # Опционально: pgadmin или другие сервисы

volumes:
  db_data:
    driver: local
```

Запуск:

```bash
docker-compose up -d
```

## Usage

### Run the API

```bash
uvicorn api_cupynumeric_db:app --reload
```

The service will be available at `http://127.0.0.1:8000`.

### Endpoints

* `POST /start-collection`
  Start background data collection.
  **Body**:

  ```json
  {
    "phases": [
      {"instruction": "Relax", "label": "rest", "duration": 10.0},
      {"instruction": "Focus", "label": "task", "duration": 15.0}
    ],
    "udp_ip": "0.0.0.0",
    "udp_port": 2000
  }
  ```

* `POST /train-model`
  Train GPU-based RandomForest on collected data.
  **Response**:

  ```json
  {"accuracy": 92.5, "report": "..."}
  ```

* `GET /predict`
  Read one EEG UDP packet and return predicted phase.
  **Response**:

  ```json
  {"predicted_label": "rest", "encoded": [0]}
  ```

## Project Structure

```
├── api_cupynumeric_db.py        # Main FastAPI application
├── eeg_data.db                  # SQLite database (auto-created)
├── requirements.txt             # Python dependencies
├── trained_model_gpu.pkl        # Saved GPU-trained model
├── scaler_gpu.pkl               # Saved GPU scaler
└── label_encoder_gpu.pkl        # Saved label encoder
```

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License.

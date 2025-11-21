from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import os
import logging

app = Flask(__name__)

# -----------------------------
# Logging Configuration
# -----------------------------
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG level shows everything
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler()  # prints to console (Render logs)
    ]
)

# -----------------------------
# Paths (Render-safe)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_FILE = BASE_DIR / 'dataset.csv'

# -----------------------------
# Global Variables
# -----------------------------
scaler = None
pca = None
le = None
model = None
traffic_data = None

# -----------------------------
# Load ML Resources
# -----------------------------
def load_resources():
    global scaler, pca, le, model, traffic_data
    logging.info("Loading Defense System...")

    try:
        logging.info(f"Loading scaler from {MODELS_DIR / 'scaler.joblib'}")
        scaler = joblib.load(MODELS_DIR / 'scaler.joblib')

        logging.info(f"Loading PCA from {MODELS_DIR / 'pca.joblib'}")
        pca = joblib.load(MODELS_DIR / 'pca.joblib')

        logging.info(f"Loading label encoder from {MODELS_DIR / 'label_encoder.joblib'}")
        le = joblib.load(MODELS_DIR / 'label_encoder.joblib')

        logging.info(f"Loading Random Forest model from {MODELS_DIR / 'Random_Forest.joblib'}")
        model = joblib.load(MODELS_DIR / 'Random_Forest.joblib')

        logging.info(f"Loading dataset from {DATA_FILE}")
        traffic_data = pd.read_csv(DATA_FILE, low_memory=False)
        logging.info(f"Dataset loaded: {traffic_data.shape[0]} rows, {traffic_data.shape[1]} columns")

        # Clean column names
        traffic_data.columns = (
            traffic_data.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )
        traffic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        traffic_data.dropna(inplace=True)
        logging.info(f"Columns after cleaning: {list(traffic_data.columns)}")
        logging.info("✅ System Loaded Successfully!")

    except FileNotFoundError as e:
        logging.error(f"Missing file: {e}", exc_info=True)
    except Exception as e:
        logging.error(f"Error loading resources: {e}", exc_info=True)

# -----------------------------
# Routes
# -----------------------------
@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/get_traffic')
def get_traffic():
    global scaler, pca, le, model, traffic_data

    if model is None or traffic_data is None:
        logging.error("System not initialized. Model or dataset is None.")
        return jsonify({'error': 'System not initialized'}), 500

    try:
        logging.debug(f"Sampling packet from dataset of shape {traffic_data.shape}")
        packet_row = traffic_data.sample(1)

        # Drop label if present
        features_raw = packet_row.drop(columns=['label']) if 'label' in packet_row else packet_row
        features_numeric = features_raw.select_dtypes(include=['number'])
        logging.debug(f"Numeric features: {list(features_numeric.columns)}")

        # Prediction
        start = time.perf_counter()
        scaled = scaler.transform(features_numeric)
        reduced = pca.transform(scaled)

        pred_idx = model.predict(reduced)[0]
        prediction_label = le.inverse_transform([pred_idx])[0]
        confidence = np.max(model.predict_proba(reduced))

        latency_ms = (time.perf_counter() - start) * 1000

        # Fake IP + protocol for demo
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        protocol = "TCP" if random.random() > 0.5 else "UDP"

        logging.debug(f"Prediction: {prediction_label}, Confidence: {confidence:.2f}")
        return jsonify({
            'src_ip': src_ip,
            'protocol': protocol,
            'prediction': prediction_label,
            'is_threat': prediction_label != "BENIGN",
            'confidence': f"{confidence*100:.2f}%",
            'latency': f"{latency_ms:.3f} ms",
            'timestamp': time.strftime("%H:%M:%S")
        })

    except Exception as e:
        logging.error(f"Error processing packet: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

# -----------------------------
# Global Exception Handler
# -----------------------------
@app.errorhandler(Exception)
def handle_exception(e):
    logging.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': str(e)}), 500

# -----------------------------
# Main (Render-safe)
# -----------------------------
if __name__ == '__main__':
    load_resources()
    port = int(os.environ.get("PORT", 3000))  # Render assigns PORT
    app.run(host="0.0.0.0", port=port)

from flask import Flask, render_template, jsonify
import joblib
import pandas as pd
import numpy as np
import random
import time
from pathlib import Path
import os

app = Flask(__name__)

# -----------------------------
#  PATHS (RENDER SAFE)
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
DATA_FILE = BASE_DIR / 'dataset.csv'

# Global variables
scaler = None
pca = None
le = None
model = None
traffic_data = None


# -----------------------------
#  LOADING ML RESOURCES
# -----------------------------
def load_resources():
    global scaler, pca, le, model, traffic_data
    print("Loading Defense System...")

    try:
        # Load model + preprocessors
        scaler = joblib.load(MODELS_DIR / 'scaler.joblib')
        pca = joblib.load(MODELS_DIR / 'pca.joblib')
        le = joblib.load(MODELS_DIR / 'label_encoder.joblib')
        model = joblib.load(MODELS_DIR / 'Random_Forest.joblib')

        # Load Dataset
        traffic_data = pd.read_csv(DATA_FILE, low_memory=False)

        # Clean column names
        traffic_data.columns = (
            traffic_data.columns
            .str.strip()
            .str.lower()
            .str.replace(' ', '_')
        )

        # Clean values
        traffic_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        traffic_data.dropna(inplace=True)

        print("✅ System Loaded Successfully!")

    except FileNotFoundError as e:
        print(f"❌ Missing File: {e}")
        print("Make sure dataset.csv and models/ are uploaded to the repo root.")


# -----------------------------
#  ROUTES
# -----------------------------
@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/get_traffic')
def get_traffic():
    global scaler, pca, le, model, traffic_data

    if model is None or traffic_data is None:
        return jsonify({'error': 'System not initialized'}), 500

    try:
        # Random packet
        packet_row = traffic_data.sample(1)

        # Drop label if present
        if 'label' in packet_row:
            features_raw = packet_row.drop(columns=['label'])
        else:
            features_raw = packet_row

        # Select only numeric
        features_numeric = features_raw.select_dtypes(include=['number'])

        # Predict
        start = time.perf_counter()
        scaled = scaler.transform(features_numeric)
        reduced = pca.transform(scaled)

        pred_idx = model.predict(reduced)[0]
        prediction_label = le.inverse_transform([pred_idx])[0]
        confidence = np.max(model.predict_proba(reduced))

        latency_ms = (time.perf_counter() - start) * 1000

        # Fake IP + protocol for dashboard display
        src_ip = f"192.168.1.{random.randint(2, 254)}"
        protocol = "TCP" if random.random() > 0.5 else "UDP"

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
        print(f"⚠️ Error: {e}")
        return jsonify({'error': str(e)}), 500


# -----------------------------
#  MAIN ENTRY (RENDER SAFE)
# -----------------------------
if __name__ == '__main__':
    load_resources()
    port = int(os.environ.get("PORT", 3000))  # Render assigns PORT
    app.run(host="0.0.0.0", port=port)

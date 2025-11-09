import os
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# === RUTA ABSOLUTA CORRECTA AL CSV ===
# Calcula la ruta base del proyecto (3 niveles arriba desde este archivo)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "crypto_dataset_final.csv")

# Carga el dataset
df = pd.read_csv(DATA_PATH)

# Escalador global (asumiendo que lo usas para normalizar precios)
scaler = MinMaxScaler(feature_range=(0, 1))

# === CARGA DEL MODELO ===
MODEL_PATH = os.path.join(BASE_DIR, "models", "crypto_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# === FUNCIÓN: Predicción de precios ===
def predict_price(crypto_name: str):
    # Verifica si la cripto existe en el dataset
    if crypto_name not in df["name"].unique():
        raise ValueError(f"La criptomoneda '{crypto_name}' no existe en el dataset.")

    # Filtra datos históricos de esa criptomoneda
    data = df[df["name"] == crypto_name].sort_values(by="date")

    # Prepara datos de entrada
    prices = data["price"].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(prices)

    # Toma los últimos 60 valores para predecir el siguiente
    last_60 = scaled_data[-60:]
    X_test = np.array([last_60])
    prediction = model.predict(X_test)

    # Desescala el valor predicho
    predicted_price = scaler.inverse_transform(prediction)[0][0]
    return float(predicted_price)

# === FUNCIÓN: Obtener historial reciente ===
def get_history(crypto_name: str, limit: int = 30):
    if crypto_name not in df["name"].unique():
        raise ValueError(f"La criptomoneda '{crypto_name}' no existe en el dataset.")
    data = df[df["name"] == crypto_name].sort_values(by="date", ascending=False).head(limit)
    return data[["date", "price"]].to_dict(orient="records")

# === FUNCIÓN: Recomendaciones simples ===
def get_recommendations():
    latest_data = df.sort_values(by="date", ascending=False).groupby("name").head(1)
    top_cryptos = latest_data.sort_values(by="price", ascending=False).head(5)
    return top_cryptos[["name", "price"]].to_dict(orient="records")

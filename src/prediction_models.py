# src/prediction_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
import os

# ==========================================================
# 1️⃣ Configuración inicial
# ==========================================================
print("📥 Cargando dataset...")
df = pd.read_csv("data/crypto_dataset_final.csv")
df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")

# Crear carpeta de salida si no existe
os.makedirs("output", exist_ok=True)

# Identificar las monedas únicas
cryptos = df["symbol_id"].unique()
print(f"🔍 Monedas encontradas: {cryptos}")

# ==========================================================
# 2️⃣ Función para crear secuencias para LSTM / GRU
# ==========================================================
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

# ==========================================================
# 3️⃣ Entrenar y predecir por cada moneda
# ==========================================================
for symbol in cryptos:
    print(f"\n🚀 Procesando symbol_id = {symbol}...")
    crypto_df = df[df["symbol_id"] == symbol].sort_values("date")

    # Usamos la columna 'close'
    prices = crypto_df["close"].values.reshape(-1, 1)

    # Normalizar para redes neuronales
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)

    time_steps = 10
    X, y = create_sequences(scaled_prices, time_steps)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # ======================================================
    # 🔹 Modelo ARIMA
    # ======================================================
    print("🧩 Entrenando modelo ARIMA...")
    try:
        arima_model = ARIMA(prices, order=(5, 1, 2))
        arima_result = arima_model.fit()
        forecast_arima = arima_result.forecast(steps=30)

        plt.figure(figsize=(10, 5))
        plt.plot(crypto_df["date"], prices, label="Real")
        plt.plot(pd.date_range(crypto_df["date"].iloc[-1], periods=31, freq="D")[1:], forecast_arima, label="ARIMA Forecast", color="orange")
        plt.title(f"Predicción de precios {symbol} - ARIMA")
        plt.legend()
        plt.savefig(f"output/{symbol}_forecast_arima.png")
        plt.close()
    except Exception as e:
        print(f"⚠️ Error en ARIMA para {symbol}: {e}")

    # ======================================================
    # 🔹 Modelo LSTM
    # ======================================================
    print("🧠 Entrenando modelo LSTM...")
    lstm_model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    predicted_lstm = lstm_model.predict(X[-30:])
    predicted_lstm = scaler.inverse_transform(predicted_lstm)

    plt.figure(figsize=(10, 5))
    plt.plot(crypto_df["date"], prices, label="Real")
    plt.plot(crypto_df["date"].iloc[-30:], predicted_lstm, label="LSTM Forecast", color="red")
    plt.title(f"Predicción de precios {symbol} - LSTM")
    plt.legend()
    plt.savefig(f"output/{symbol}_forecast_lstm.png")
    plt.close()

    # ======================================================
    # 🔹 Modelo GRU
    # ======================================================
    print("🧩 Entrenando modelo GRU...")
    gru_model = Sequential([
        GRU(50, return_sequences=False, input_shape=(time_steps, 1)),
        Dense(1)
    ])
    gru_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    gru_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

    predicted_gru = gru_model.predict(X[-30:])
    predicted_gru = scaler.inverse_transform(predicted_gru)

    plt.figure(figsize=(10, 5))
    plt.plot(crypto_df["date"], prices, label="Real")
    plt.plot(crypto_df["date"].iloc[-30:], predicted_gru, label="GRU Forecast", color="green")
    plt.title(f"Predicción de precios {symbol} - GRU")
    plt.legend()
    plt.savefig(f"output/{symbol}_forecast_gru.png")
    plt.close()

    print(f"✅ Modelos de {symbol} entrenados y gráficos guardados en /output/")

print("\n🎯 Proceso completo para todas las monedas finalizado.")

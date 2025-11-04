# src/prediction_models.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from sklearn.preprocessing import MinMaxScaler

# ==========================================================
# 1️⃣ Cargar y preparar datos
# ==========================================================
print("📥 Cargando dataset...")
df = pd.read_csv("../data/crypto_dataset_final.csv")
df["date"] = pd.to_datetime(df["date"])

# Filtramos Bitcoin (id == 0)
btc = df[df["id"] == 0].sort_values("date")
prices = btc["close"].values.reshape(-1, 1)

# Normalizamos para LSTM y GRU
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(prices)

# ==========================================================
# 2️⃣ Crear datos de entrenamiento para redes neuronales
# ==========================================================
def create_sequences(data, time_steps=10):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:(i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)

time_steps = 10
X, y = create_sequences(scaled_prices, time_steps)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ==========================================================
# 3️⃣ Modelo ARIMA
# ==========================================================
print("🔹 Entrenando modelo ARIMA...")
arima_model = ARIMA(prices, order=(5, 1, 2))
arima_result = arima_model.fit()

# Predicción ARIMA
forecast_arima = arima_result.forecast(steps=30)
plt.figure(figsize=(10, 5))
plt.plot(btc["date"], prices, label="Real")
plt.plot(pd.date_range(btc["date"].iloc[-1], periods=31, freq="D")[1:], forecast_arima, label="ARIMA Forecast", color="orange")
plt.title("Predicción de precios Bitcoin - ARIMA")
plt.legend()
plt.savefig("output/forecast_arima.png")
plt.close()

# ==========================================================
# 4️⃣ Modelo LSTM
# ==========================================================
print("🔹 Entrenando modelo LSTM...")
lstm_model = Sequential([
    LSTM(50, return_sequences=False, input_shape=(time_steps, 1)),
    Dense(1)
])
lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
lstm_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Predicción LSTM
predicted_lstm = lstm_model.predict(X[-30:])
predicted_lstm = scaler.inverse_transform(predicted_lstm)

plt.figure(figsize=(10, 5))
plt.plot(btc["date"], prices, label="Real")
plt.plot(btc["date"].iloc[-30:], predicted_lstm, label="LSTM Forecast", color="red")
plt.title("Predicción de precios Bitcoin - LSTM")
plt.legend()
plt.savefig("output/forecast_lstm.png")
plt.close()

# ==========================================================
# 5️⃣ Modelo GRU
# ==========================================================
print("🔹 Entrenando modelo GRU...")
gru_model = Sequential([
    GRU(50, return_sequences=False, input_shape=(time_steps, 1)),
    Dense(1)
])
gru_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
gru_model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# Predicción GRU
predicted_gru = gru_model.predict(X[-30:])
predicted_gru = scaler.inverse_transform(predicted_gru)

plt.figure(figsize=(10, 5))
plt.plot(btc["date"], prices, label="Real")
plt.plot(btc["date"].iloc[-30:], predicted_gru, label="GRU Forecast", color="green")
plt.title("Predicción de precios Bitcoin - GRU")
plt.legend()
plt.savefig("output/forecast_gru.png")
plt.close()

print("✅ Modelos entrenados y gráficos guardados en /output/")

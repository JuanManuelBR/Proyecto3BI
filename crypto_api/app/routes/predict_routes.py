from fastapi import APIRouter, HTTPException
from tensorflow import keras
import numpy as np
import pandas as pd
import os

router = APIRouter(
    prefix="/predict",
    tags=["Predicciones"]
)

# ==============================
# 📂 CARGA DEL MODELO KERAS
# ==============================
model = None

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # /crypto_api/app/routes
    MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "0_gru.keras")
    MODEL_PATH = os.path.normpath(MODEL_PATH)

    print(f"Intentando cargar modelo desde: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)
    print("✅ Modelo Keras cargado correctamente.")
except Exception as e:
    print(f"⚠️ Error al cargar el modelo Keras: {e}")

# ==============================
# 📦 HISTORIAL EN MEMORIA
# ==============================
PREDICTION_HISTORY = []

# ==============================
# 📈 ENDPOINT DE PREDICCIÓN
# ==============================
@router.post("/")
def predict(data: dict):
    """
    🧠 Realiza una predicción con el modelo GRU cargado.
    Espera un JSON con el formato:
    {
        "values": [0.1, 0.2, 0.3, 0.4]
    }
    Devuelve la predicción en formato JSON.
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modelo no cargado correctamente.")

    try:
        values = data.get("values")
        if not values:
            raise ValueError("No se proporcionaron datos para la predicción.")

        # Convertir a array numpy
        X = np.array(values).reshape(1, len(values), 1)  # (batch, timesteps, features)
        prediction = model.predict(X)

        # Guardar en historial
        entry = {
            "input": values,
            "prediction": prediction.tolist()
        }
        PREDICTION_HISTORY.append(entry)

        return {"prediction": prediction.tolist()}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al realizar la predicción: {e}")


# ==============================
# 🕓 HISTORIAL DE PREDICCIONES
# ==============================
@router.get("/history")
def history():
    """
    📜 Devuelve las predicciones previas guardadas en memoria.
    """
    try:
        if not PREDICTION_HISTORY:
            return {"message": "No hay predicciones registradas aún."}
        return {"history": PREDICTION_HISTORY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 💹 RECOMENDACIONES REALES DESDE CSV
# ==============================
@router.get("/recommendations")
def recommendations():
    """
    📊 Devuelve recomendaciones REALES basadas en datos históricos de
    Bitcoin, Ethereum y Dogecoin desde crypto_api/data/crypto_dataset_final.csv.

    La lógica usa la variación diaria del precio de cierre para generar:
    - 📈 "Comprar" si subió más de 1%
    - 📉 "Vender" si bajó más de 1%
    - ⚖️ "Mantener" si el cambio fue leve
    """
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        CSV_PATH = ("C:\\Users\\juanm\\Documents\\GitHub\\Proyecto3BI\\crypto_api\\data\\crypto_dataset_final.csv")
        CSV_PATH = os.path.normpath(CSV_PATH)

        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError("No se encontró el archivo CSV con los datos reales.")

        df = pd.read_csv(CSV_PATH)
        if df.empty:
            raise ValueError("El archivo CSV está vacío o mal formateado.")

        SYMBOLS = {
            0: "Bitcoin",
            1: "Ethereum",
            2: "Dogecoin"
        }

        recomendaciones = []

        for symbol_id, nombre in SYMBOLS.items():
            data = df[df["symbol_id"] == symbol_id].sort_values("date", ascending=False)

            if data.empty:
                continue

            ultimo = data.iloc[0]
            anterior = data.iloc[1] if len(data) > 1 else None

            variacion = None
            if anterior is not None:
                variacion = ((ultimo["close"] - anterior["close"]) / anterior["close"]) * 100

            if variacion is not None:
                if variacion > 1:
                    recomendacion = "Comprar 📈"
                elif variacion < -1:
                    recomendacion = "Vender 📉"
                else:
                    recomendacion = "Mantener ⚖️"
            else:
                recomendacion = "Sin datos suficientes"

            recomendaciones.append({
                "nombre": nombre,
                "fecha": str(ultimo["date"]),
                "precio_actual": round(float(ultimo["close"]), 2),
                "precio_maximo": round(float(ultimo["high"]), 2),
                "precio_minimo": round(float(ultimo["low"]), 2),
                "volumen": round(float(ultimo["volume"]), 2),
                "variacion_24h": round(variacion, 2) if variacion is not None else None,
                "recomendacion": recomendacion
            })

        if not recomendaciones:
            raise HTTPException(status_code=404, detail="No se encontraron datos en el CSV.")

        return {
            "status": "success",
            "data": recomendaciones
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar recomendaciones: {e}")

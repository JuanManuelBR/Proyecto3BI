from fastapi import APIRouter, HTTPException
from tensorflow import keras
import numpy as np
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
    # Construir ruta absoluta segura al modelo
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
# 📈 PREDICCIÓN
# ==============================
@router.post("/")
def predict(data: dict):
    """
    Recibe datos de entrada y devuelve una predicción usando el modelo cargado.
    Espera un JSON con los datos de entrada, por ejemplo:
    {
        "values": [0.1, 0.2, 0.3, 0.4]
    }
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
# 💡 RECOMENDACIONES
# ==============================
@router.get("/recommendations")
def recommendations():
    """
    Devuelve recomendaciones simuladas de criptomonedas.
    """
    try:
        data = {
            "BTC": 0.82,
            "ETH": 0.75,
            "ADA": 0.68,
            "SOL": 0.71,
            "XRP": 0.65
        }

        sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=True))
        best = list(sorted_data.keys())[:3]

        return {
            "recommendations": best,
            "performance_scores": sorted_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# 🕓 HISTORIAL DE PREDICCIONES
# ==============================
@router.get("/history")
def history():
    """
    Devuelve las predicciones previas guardadas en memoria.
    """
    try:
        if not PREDICTION_HISTORY:
            return {"message": "No hay predicciones registradas aún."}
        return {"history": PREDICTION_HISTORY}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

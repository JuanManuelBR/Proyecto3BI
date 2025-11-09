from fastapi import FastAPI
from app.routes.predict_routes import router as predict_router

app = FastAPI(
    title="Crypto Recommender API",
    description="API para predicciones y recomendaciones de criptomonedas (ARIMA, LSTM, GRU).",
    version="1.0.0"
)

# Registrar rutas
app.include_router(predict_router)

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API funcionando correctamente 🚀"}

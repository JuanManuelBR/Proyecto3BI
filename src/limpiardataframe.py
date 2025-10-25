import pandas as pd
import yfinance as yf
import os

# 1️⃣ Descargar datos de Bitcoin y Ethereum
symbols = ['BTC-USD', 'ETH-USD']
api_data = pd.DataFrame()

for symbol in symbols:
    print(f"📥 Descargando {symbol}...")
    data = yf.download(symbol, period="3mo", interval="1d")

    # Aplanar el MultiIndex de columnas (por seguridad)
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

    data['symbol'] = symbol
    api_data = pd.concat([api_data, data])

# Estandarizar nombres
api_data.reset_index(inplace=True)
api_data.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

print("✅ Datos desde API listos:", api_data.shape)
print(api_data.head(), "\n")

# 2️⃣ Leer archivo CSV local de Dogecoin
# (Guarda el archivo en la misma carpeta del script, o cambia la ruta)
doge_csv_path = "coin_Dogecoin.csv"

if not os.path.exists(doge_csv_path):
    print("⚠️ No se encontró el archivo 'coin_Dogecoin.csv'. Por favor, colócalo junto a este script.")
    exit()

doge_csv = pd.read_csv(doge_csv_path)

# Estandarizar columnas
doge_csv.rename(columns={
    'Date': 'date',
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume'
}, inplace=True)

# Convertir fecha y agregar símbolo
doge_csv['date'] = pd.to_datetime(doge_csv['date'], errors='coerce')
doge_csv['symbol'] = 'DOGE-USD'

# Mantener solo columnas necesarias
doge_csv = doge_csv[['date', 'open', 'high', 'low', 'close', 'volume', 'symbol']]

print("✅ Datos Dogecoin listos:", doge_csv.shape)

# 3️⃣ Combinar datasets
combined = pd.concat([api_data, doge_csv], ignore_index=True)

# 4️⃣ Limpiar duplicados y ordenar
combined.drop_duplicates(subset=['date', 'symbol'], keep='last', inplace=True)
combined.sort_values(by=['symbol', 'date'], inplace=True)

# 5️⃣ Guardar resultado final
output_path = "crypto_dataset_final.csv"
combined.to_csv(output_path, index=False)
print(f"💾 Archivo guardado como {output_path}")

# 6️⃣ Mostrar primeras filas
print("\n📊 Primeras filas del dataset combinado:")
print(combined.head())

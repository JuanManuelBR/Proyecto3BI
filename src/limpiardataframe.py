import pandas as pd
import yfinance as yf
import os

# Definir las criptomonedas que se van a descargar de la API yfinance (Bitcoin y Ethereum)
symbols = ['BTC-USD', 'ETH-USD']

# Se crea una variable llamada api_data para almacenar los datos descargados
api_data = pd.DataFrame()

# Descargar datos de cada criptomoneda con un periodo más largo
for symbol in symbols:
    print(f"Descargando {symbol}...")
    data = yf.download(symbol, period="5y", interval="1d")  # Aumentar el periodo para tener más filas

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

print("Datos desde API listos:", api_data.shape)

# Recuperar los datos de Dogecoin desde un archivo .CSV local
doge_csv_path = "data/coin_Dogecoin.csv"

doge_csv = pd.read_csv(doge_csv_path)

# Estandarizar columnas al igual que Bitcoin y Ethereum
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

# Tomar la misma cantidad de datos para cada símbolo
target_rows = 2000 // 3  # aproximadamente 666 por criptomoneda

balanced_data = pd.DataFrame()
for symbol in ['BTC-USD', 'ETH-USD', 'DOGE-USD']:
    if symbol == 'DOGE-USD':
        df = doge_csv.copy()
    else:
        df = api_data[api_data['symbol'] == symbol].copy()

    # Ordenar por fecha y tomar muestras uniformemente distribuidas
    df = df.sort_values(by='date').reset_index(drop=True)
    if len(df) > target_rows:
        df = df.iloc[-target_rows:]  # tomar los últimos n registros


    balanced_data = pd.concat([balanced_data, df], ignore_index=True)

# Limpiar duplicados y ordenar
balanced_data.drop_duplicates(subset=['date', 'symbol'], keep='last', inplace=True)
balanced_data.sort_values(by=['symbol', 'date'], inplace=True)

# Guardar resultado final
output_path = "data/crypto_dataset_final.csv"
os.makedirs("data", exist_ok=True)
balanced_data.to_csv(output_path, index=False)
print(f"Archivo guardado como {output_path}")

# Mostrar resumen
print("\nTamaño final por símbolo:")
print(balanced_data['symbol'].value_counts())
print("\nPrimeras filas del dataset combinado:")
print(balanced_data.head())

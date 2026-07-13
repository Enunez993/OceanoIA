"""
OceanoIA - Módulo 2: RNN/LSTM — Predicción Oceanográfica
train.py

Entrena un modelo LSTM para predecir una variable oceanográfica
(oleaje, marea o SST) a partir de data/processed/oceano_merged.csv.

"""


#Importaciones
import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

ruta_base = os.path.abspath(os.path.dirname(__file__))
ruta_datos = os.path.join(ruta_base, "data", "processed", "oceano_merged.csv")
ruta_modelos = os.path.join(ruta_base, "models")

FEATURES_DISPONIBLES = [
    "Oleaje_m", "Periodo_Oleaje_s", "Direccion_Oleaje_deg", "Marea_m",
    "SST_Copernicus", "temp_aire_c", "viento_kmh", "presion_hpa",
    "humedad_pct", "nubosidad_octas", "Fase_Lunar",
]


def cargar_datos(ruta=ruta_datos):
    #Carga oceano_merged.csv indexado por Fecha, ordenado
    df = pd.read_csv(ruta, parse_dates=["Fecha"], index_col="Fecha")
    df = df.sort_index()
    return df


def preparar_features(df):
    #Devuelve solo las columnas de interés que sí existan en el CSV real
    features = [c for c in FEATURES_DISPONIBLES if c in df.columns]
    faltantes = df[features].isna().sum()
    if faltantes.sum() > 0:
        df[features] = df[features].interpolate(method="time")
    return df, features


def crear_secuencias(arr, window_in, window_out, target_idx):
    #window_in días de historia -> window_out días a predecir
    X, y = [], []
    for i in range(len(arr) - window_in - window_out):
        X.append(arr[i: i + window_in])
        y.append(arr[i + window_in: i + window_in + window_out, target_idx])
    return np.array(X), np.array(y)


def desescalar_target(y_scaled, scaler, target_idx, n_features):
    #Devuelve las predicciones de la escala 0-1 a las unidades reales
    dummy = np.zeros((y_scaled.shape[0] * y_scaled.shape[1], n_features))
    dummy[:, target_idx] = y_scaled.flatten()
    inv = scaler.inverse_transform(dummy)[:, target_idx]
    return inv.reshape(y_scaled.shape)


def construir_modelo(window_in, n_features, window_out):
    #Arquitectura fija del proyecto: LSTM(100) -> LSTM(50) -> Dropout(0.2) -> Dense
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(window_in, n_features)),
        LSTM(50),
        Dropout(0.2),
        Dense(window_out),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def evaluar(model, X_test, y_test, scaler, target_idx, n_features):
    #RMSE y MAE en unidades reales, más el baseline de persistencia
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred = desescalar_target(y_pred_scaled, scaler, target_idx, n_features)
    y_true = desescalar_target(y_test, scaler, target_idx, n_features)

    rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())

    # Baseline de persistencia: predicción = último valor conocido repetido
    window_out = y_test.shape[1]
    y_pred_base = np.repeat(X_test[:, -1, target_idx][:, None], window_out, axis=1)
    y_pred_base_real = desescalar_target(y_pred_base, scaler, target_idx, n_features)
    rmse_base = np.sqrt(mean_squared_error(y_true.flatten(), y_pred_base_real.flatten()))
    mae_base = mean_absolute_error(y_true.flatten(), y_pred_base_real.flatten())

    return {
        "rmse": rmse, "mae": mae,
        "rmse_baseline": rmse_base, "mae_baseline": mae_base,
    }


def entrenar(target="Oleaje_m", window_in=30, window_out=3, epochs=100, batch_size=16):
    print(f"=== Entrenando modelo LSTM — target: {target} ===\n")

    df = cargar_datos()
    df, features = preparar_features(df)

    if target not in features:
        raise ValueError(f"'{target}' no está en las columnas disponibles: {features}")

    target_idx = features.index(target)
    data = df[features].values

    # Split cronológico 80/20
    split_idx = int(len(data) * 0.8)
    train_raw, test_raw = data[:split_idx], data[split_idx:]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_raw)
    test_scaled = scaler.transform(test_raw)

    X_train, y_train = crear_secuencias(train_scaled, window_in, window_out, target_idx)
    X_test, y_test = crear_secuencias(test_scaled, window_in, window_out, target_idx)

    print(f"Train: {X_train.shape[0]} secuencias | Test: {X_test.shape[0]} secuencias")

    n_features = X_train.shape[2]
    model = construir_modelo(window_in, n_features, window_out)
    model.summary()

    early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1,
    )

    metricas = evaluar(model, X_test, y_test, scaler, target_idx, n_features)
    print(f"\nRMSE ({target}): {metricas['rmse']:.3f}  |  Baseline: {metricas['rmse_baseline']:.3f}")
    print(f"MAE  ({target}): {metricas['mae']:.3f}  |  Baseline: {metricas['mae_baseline']:.3f}")
    mejora = (1 - metricas["rmse"] / metricas["rmse_baseline"]) * 100
    print(f"Mejora sobre baseline: {mejora:.1f}% menos error (RMSE)")

    # Reentrenamiento final con TODOS los datos, para el modelo de producción
    print("\nReentrenando con el histórico completo...")
    full_scaled = scaler.fit_transform(data)
    X_full, y_full = crear_secuencias(full_scaled, window_in, window_out, target_idx)

    final_model = construir_modelo(window_in, n_features, window_out)
    final_model.fit(X_full, y_full, epochs=max(epochs // 2, 30), batch_size=batch_size, verbose=1)

    # Guardar modelo + scaler
    os.makedirs(ruta_modelos, exist_ok=True)
    ruta_modelo = os.path.join(ruta_modelos, f"lstm_{target.lower()}.keras")
    ruta_scaler = os.path.join(ruta_modelos, f"scaler_{target.lower()}.pkl")
    final_model.save(ruta_modelo)
    joblib.dump(scaler, ruta_scaler)

    print(f"\nModelo guardado en: {ruta_modelo}")
    print(f"Scaler guardado en: {ruta_scaler}")

    return final_model, scaler, metricas


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena el modelo LSTM de OceanoIA")
    parser.add_argument("--target", default="Oleaje_m",
                         help="Variable a predecir: Oleaje_m, Marea_m o SST_Copernicus")
    parser.add_argument("--window_in", type=int, default=30, help="Días de historia")
    parser.add_argument("--window_out", type=int, default=3, help="Días a predecir (72h = 3 días)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    entrenar(
        target=args.target,
        window_in=args.window_in,
        window_out=args.window_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )

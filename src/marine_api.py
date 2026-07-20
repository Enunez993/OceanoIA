"""
OceanoIA - Módulo 2: RNN/LSTM
Une los 3 datasets (Open-Meteo, Copernicus, Meteostat) en una sola tabla
diaria con TODAS las variables de interés:
    oleaje, marea, SST, viento, presión y fase lunar.

Se ejecuta DESPUÉS de haber generado los 3 CSV en data/raw/.
"""

import os
import pandas as pd


def fase_lunar(fechas):
    """
    Fase lunar como fracción [0,1): 0 = luna nueva, 0.5 = luna llena.
    Cálculo astronómico (ciclo sinódico de 29.53 días) desde una luna
    nueva de referencia. No usa ninguna API.
    """
    ref = pd.Timestamp("2000-01-06 18:14:00")
    ciclo = 29.53058867
    fechas = pd.to_datetime(fechas)
    try:
        fechas = fechas.tz_localize(None)   # quita zona horaria si la tuviera
    except TypeError:
        pass                                # ya era naive
    dias = (fechas - ref) / pd.Timedelta(days=1)
    return (dias % ciclo) / ciclo


def _a_diario(df):
    """Normaliza el índice a fecha diaria (sin zona horaria) y promedia."""
    df = df.copy()
    idx = pd.to_datetime(df.index)
    try:
        idx = idx.tz_localize(None)
    except TypeError:
        pass
    df.index = idx.normalize()
    num = df.select_dtypes("number")        # ignora columnas de texto (p.ej. 'fuente')
    return num.groupby(num.index).mean()


def unir_datasets():
    """
    Lee los 3 CSV de data/raw/, los unifica a nivel diario por fecha,
    agrega la fase lunar e interpola huecos. Guarda el resultado en
    data/processed/oceano_merged.csv, listo para el LSTM.
    """
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    raw = os.path.join(ruta_base, "data", "raw")
    proc = os.path.join(ruta_base, "data", "processed")
    os.makedirs(proc, exist_ok=True)

    def _load(nombre):
        ruta = os.path.join(raw, nombre)
        if not os.path.exists(ruta):
            print(f"  [aviso] falta {nombre}; se omite.")
            return None
        df = pd.read_csv(ruta, index_col=0)
        df.index = pd.to_datetime(df.index)
        return df

    d1 = _load("open_meteo_marine.csv")   # oleaje + marea
    d2 = _load("copernicus_sst.csv")      # SST
    d3 = _load("dataset3_meteostat.csv")    # <- VIENTO, PRESION

    partes = [_a_diario(d) for d in (d1, d2, d3) if d is not None]
    if not partes:
        print("No hay datasets para unir.")
        return None

    merged = partes[0].join(partes[1:], how="outer")
    merged.index.name = "Fecha"

    # Rellenar huecos por diferencia de frecuencias/cobertura
    merged = merged.interpolate(limit_direction="both")

    # Quitar precipitación (constante/rellenada; no es variable requerida y tiene un valor fijo)
    merged = merged.drop(columns=["precip_mm"], errors="ignore")

    # Fase lunar (variable de interés) — debe calcularse ANTES de guardar,
    # si no la columna queda solo en memoria y nunca llega al CSV.
    merged["Fase_Lunar"] = fase_lunar(merged.index)

    salida = os.path.join(proc, "oceano_merged.csv")
    merged.to_csv(salida)

    print(f"¡Tabla unificada guardada en: {salida}!")
    print(f"  Filas: {len(merged)} | Variables: {list(merged.columns)}")
    return merged


if __name__ == "__main__":
    print("=== UNIÓN DE DATASETS (tabla diaria para el LSTM) ===")
    df = unir_datasets()
    if df is not None:
        print("\nMuestra de la tabla final:")
        print(df.head())


import requests
import pandas as pd

# Coordenadas de cada zona (la app las importa como ZONAS_CR)
ZONAS_CR = {
    "golfo_nicoya":     (9.80, -84.80),
    "golfo_dulce":      (8.60, -83.30),
    "pacifico_norte":   (10.60, -85.70),
    "pacifico_central": (9.50, -84.30),
    "pacifico_sur":     (8.70, -83.50),
    "caribe_norte":     (10.50, -83.05),
    "caribe_sur":       (9.60, -82.70),
}


def get_zone_forecast(zone, days=3, **kwargs):
    """Pronóstico marino de una zona. Devuelve 'time' como COLUMNA
    y los nombres originales de Open-Meteo (los que usa Home.py)."""
    if zone not in ZONAS_CR:
        raise ValueError(f"Zona desconocida: {zone}")
    lat, lon = ZONAS_CR[zone]

    r = requests.get("https://marine-api.open-meteo.com/v1/marine", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wave_height", "wave_period", "wave_direction",
                   "sea_surface_temperature", "sea_level_height_msl"],
        "forecast_days": days,
        "timezone": "America/Costa_Rica",
    }, timeout=30)
    r.raise_for_status()

    df = pd.DataFrame(r.json()["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df
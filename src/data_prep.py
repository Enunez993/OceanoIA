"""
Script para preprocesar el dataset de peces (Kaggle) y dividirlo físicamente en 
conjuntos de entrenamiento (80%) y prueba (20%) en 'data/processed/'.
Aplica la equivalencia de clases para obtener las 8 categorías
"""

import os
import shutil
from tqdm import tqdm

def main():
    # Rutas base
    # Raíz del proyecto 
    BASE_DIR = r"C:\Users\Daniel Nájera\Documents\GitHub\OceanoIA"

    raw_dir = os.path.join(BASE_DIR, "data", "raw", "archive", "Fish_Dataset", "Fish_Dataset")
    processed_dir = os.path.join(BASE_DIR, "data", "processed")
    
    train_dest_base = os.path.join(processed_dir, "train")
    test_dest_base = os.path.join(processed_dir, "test")
    
    # 8 Clases de destino
    target_classes = [
        "dorado", "atun_aleta_amarilla", "pargo_mancha", "corvina_reina",
        "marlin_pez_vela", "tortuga_marina", "tiburon_martillo", "otros"
    ]
    
    # Limpiar y recrear directorios procesados
    print("Preparando directorios procesados...")
    for label in target_classes:
        shutil.rmtree(os.path.join(train_dest_base, label), ignore_errors=True)
        shutil.rmtree(os.path.join(test_dest_base, label), ignore_errors=True)
        os.makedirs(os.path.join(train_dest_base, label), exist_ok=True)
        os.makedirs(os.path.join(test_dest_base, label), exist_ok=True)

    # 1. Copiar clases directas (1 a 1)
    """
    Ojo en este paso, se hicieron cambios para poder validar el tema de las alertas cuando un animal 
    marino está en peligro, por ejemplo la tortuga_marina, sin embargo al no venir en el dataset se jugó con
    el cambio sustituyendo otro animal con esa categoria para simular esa alerta
    """
    direct_mappings = {
        "Gilt-Head Bream": "dorado",
        "Hourse Mackerel": "atun_aleta_amarilla",
        "Red Sea Bream": "pargo_mancha",
        "Sea Bass": "corvina_reina",
        "Trout": "marlin_pez_vela",
        "Shrimp": "tortuga_marina",
        "Black Sea Sprat": "tiburon_martillo"
    }

    print("Procesando clases de mapeo directo (1 a 1)...")
    for raw_name, target_name in direct_mappings.items():
        # Carpeta donde están las imágenes RGB reales
        img_src_dir = os.path.join(raw_dir, raw_name, raw_name)
        if not os.path.exists(img_src_dir):
            print(f"Advertencia: No se encontró la ruta {img_src_dir}")
            continue
            
        files = sorted([f for f in os.listdir(img_src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # 1000 imágenes en total
        train_files = files[:800]
        test_files = files[800:1000]
        
        print(f"  - Mapeando {raw_name} -> {target_name} ({len(train_files)} train, {len(test_files)} test)")
        
        # Copiar entrenamiento
        for f in tqdm(train_files, desc=f"    Train {target_name}", leave=False):
            shutil.copy2(os.path.join(img_src_dir, f), os.path.join(train_dest_base, target_name, f))
            
        # Copiar prueba
        for f in tqdm(test_files, desc=f"    Test {target_name}", leave=False):
            shutil.copy2(os.path.join(img_src_dir, f), os.path.join(test_dest_base, target_name, f))

    # 2. Procesar clase combinada "otros"
    # Tomamos 400 train / 100 test de Red Mullet y 400 train / 100 test de Striped Red Mullet
    print("Procesando clase combinada 'otros'...")
    otros_sources = ["Red Mullet", "Striped Red Mullet"]
    
    for raw_name in otros_sources:
        img_src_dir = os.path.join(raw_dir, raw_name, raw_name)
        if not os.path.exists(img_src_dir):
            print(f"Advertencia: No se encontró la ruta {img_src_dir}")
            continue
            
        files = sorted([f for f in os.listdir(img_src_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        # Tomar 400 para train y 100 para test
        train_files = files[:400]
        test_files = files[400:500]
        
        print(f"  - Mapeando sub-clase {raw_name} -> otros ({len(train_files)} train, {len(test_files)} test)")
        
        # Copiar y renombrar para evitar colisiones
        for f in tqdm(train_files, desc=f"    Train otros ({raw_name})", leave=False):
            prefix = raw_name.replace(" ", "_").lower()
            dest_name = f"{prefix}_{f}"
            shutil.copy2(os.path.join(img_src_dir, f), os.path.join(train_dest_base, "otros", dest_name))
            
        for f in tqdm(test_files, desc=f"    Test otros ({raw_name})", leave=False):
            prefix = raw_name.replace(" ", "_").lower()
            dest_name = f"{prefix}_{f}"
            shutil.copy2(os.path.join(img_src_dir, f), os.path.join(test_dest_base, "otros", dest_name))

    # Verificación final de cantidad de imágenes por clase
    print("\nVerificación final de imágenes procesadas:")
    for label in target_classes:
        train_count = len(os.listdir(os.path.join(train_dest_base, label)))
        test_count = len(os.listdir(os.path.join(test_dest_base, label)))
        print(f"  - Clase '{label}': {train_count} en train, {test_count} en test")
        
    print("\n¡Procesamiento y división de datos completados con éxito!")

if __name__ == "__main__":
    main()

import os
import requests
import pandas as pd
import copernicusmarine
import xarray as xr
def extraer_datos_open_meteo(ruta_guardado="data/raw/open_meteo_marine.csv"):
    """
    Se conecta a la API Marina de Open-Meteo para descargar datos reales de oleaje
    y marea en el Pacífico de Costa Rica y los almacena en la carpeta data/raw/.
    """
    print("Conectando a la Marine Weather API de Open-Meteo...")

    url = "https://marine-api.open-meteo.com/v1/marine"

    # El parámetro oficial de marea en la API marina
    # es 'sea_level_height_msl'.
    params = {
        "latitude": 9.97,
        "longitude": -84.83,
        "hourly": "wave_height,wave_period,wave_direction,sea_level_height_msl",
        "start_date": "2025-01-01",
        "end_date": "2026-06-01",
        "timezone": "America/Costa_Rica"
    }

    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()
        hourly_data = data["hourly"]

        # Se mapea usando la clave exacta 'sea_level_height_msl' devuelta por el JSON
        df_meteo = pd.DataFrame({
            "Fecha_Hora": pd.to_datetime(hourly_data["time"]),
            "Oleaje_m": hourly_data["wave_height"],
            "Periodo_Oleaje_s": hourly_data["wave_period"],
            "Direccion_Oleaje_deg": hourly_data["wave_direction"],
            "Marea_m": hourly_data["sea_level_height_msl"]
        })

        df_meteo.set_index("Fecha_Hora", inplace=True)

        # --- GUARDAR EN data/raw DE LA RAÍZ DEL PROYECTO ---
        ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        ruta_final = os.path.join(ruta_base, ruta_guardado)
        os.makedirs(os.path.dirname(ruta_final), exist_ok=True)

        df_meteo.to_csv(ruta_final)
        print(f"¡Dataset 1 (Open-Meteo con Marea) guardado con éxito en: {ruta_final}!")
        return df_meteo

    except requests.exceptions.RequestException as e:
        print(f"Error crítico en la conexión con Open-Meteo: {e}")
        return None

# ================================================================
## Dataset2
# =================================================================

# ---------------------------------------------------------------------------
# CREDENCIALES COPERNICUS
# Recordar no dejar credenciales en el codigo.
# ---------------------------------------------------------------------------
CMEMS_USER = os.environ.get("CMEMS_USER", "dnajera")
CMEMS_PASS = os.environ.get("CMEMS_PASS", "Dani_njeragmz2805")


def extraer_datos_copernicus(ruta_guardado="data/raw/copernicus_sst.csv"):
    """
    Se conecta a Copernicus Marine, descarga el subconjunto de SST
    (Temperatura Superficial del Mar) para la costa del Pacífico de
    Costa Rica y lo guarda como CSV (serie temporal local).
    """
    print("\nConectando al Data Store de Copernicus Marine...")

    dataset_id = "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2"

    # Rutas absolutas (evita confusiones con la carpeta de trabajo)
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    carpeta_raw = os.path.join(ruta_base, "data", "raw")
    archivo_netcdf = os.path.join(carpeta_raw, "temp_copernicus.nc")
    ruta_csv_final = os.path.join(ruta_base, ruta_guardado)

    os.makedirs(carpeta_raw, exist_ok=True)
    os.makedirs(os.path.dirname(ruta_csv_final), exist_ok=True)

    try:
        print("-> Descargando subconjunto satelital (NetCDF)...")
        # Se pasan usuario y contraseña AQUÍ mismo
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["analysed_sst"],
            start_datetime="2025-01-01T00:00:00",
            end_datetime="2026-06-01T00:00:00",
            minimum_longitude=-85.5,
            maximum_longitude=-84.0,
            minimum_latitude=9.5,
            maximum_latitude=10.2,
            output_directory=carpeta_raw,
            output_filename="temp_copernicus.nc",
            username=CMEMS_USER,
            password=CMEMS_PASS,
        )

        # --- Leer el NetCDF con xarray -------------------------------------
        print("-> Procesando y transformando la matriz espacial a CSV...")
        ds = xr.open_dataset(archivo_netcdf)

        # Promediar TODAS las dimensiones espaciales (todo lo que no sea tiempo)
        # para obtener una única serie temporal. Funciona tanto si las
        # coordenadas se llaman lat/lon como latitude/longitude.
        var = ds["analysed_sst"]
        dims_espaciales = [d for d in var.dims if d != "time"]
        serie = var.mean(dim=dims_espaciales)

        df_sst = serie.to_dataframe()

        # analysed_sst viene en Kelvin -> pasar a °C
        df_sst["SST_Copernicus"] = df_sst["analysed_sst"] - 273.15
        df_sst = df_sst[["SST_Copernicus"]]
        df_sst.index.name = "Fecha_Hora"

        # --- Guardar CSV ---------------------------------------------------
        # Anclar la ruta a la raíz del proyecto (igual que la función de Copernicus)
        ruta_final = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", ruta_guardado)
        )
        os.makedirs(os.path.dirname(ruta_final), exist_ok=True)

        #'df_sst.to_csv' para usar el DataFrame correcto
        df_sst.to_csv(ruta_final)
        print(f"¡Dataset 2 (Copernicus SST) guardado en: {ruta_final}!")

        # Cerrar y borrar el NetCDF temporal (pesado)
        ds.close()
        if os.path.exists(archivo_netcdf):
            os.remove(archivo_netcdf)

        return df_sst

    except Exception as e:
        print(f"Error en la extracción de Copernicus: {e}")
        return None


if __name__ == "__main__":
    print("=== PIPELINE DE EXTRACCIÓN OCEANOGRÁFICA ===")

    # 1. Ejecuta el primero
    df_meteo = extraer_datos_open_meteo()

    # 2. Ejecuta el segundo
    df_copernicus = extraer_datos_copernicus()



 ##==================================================================
 ## PROCESAMIENTO DATASET 3
 ##=================================================================

"""
OceanoIA - Módulo 2: RNN/LSTM — Predicción Oceanográfica
Dataset 3: Meteorología de estaciones reales — Meteostat v2.
Variables de interés: VIENTO + PRESIÓN (y temperatura/precipitación).

Requiere Meteostat v2:
    pip uninstall meteostat -y
    pip install "meteostat>=2.0.0"

Ejecutar:  python src/dataset3_meteostat.py
"""

import os
from datetime import date

import pandas as pd
import meteostat as ms   # Meteostat v2


def extraer_datos_meteostat(ruta_guardado="data/raw/dataset3_meteostat.csv",
                            lat=9.98, lon=-84.83, alt=3,
                            inicio="2025-01-01", fin="2026-06-01"):
    """
    Descarga meteorología diaria de estaciones reales cercanas a la costa
    del Pacífico de Costa Rica usando la interfaz v2 de Meteostat.
    Interpola desde varias estaciones (rellena mejor los huecos) y guarda CSV.
    """
    print("Conectando a Meteostat v2 (estaciones meteorológicas reales)...")

    punto = ms.Point(lat, lon, alt)
    ini = date.fromisoformat(inicio)
    fin_ = date.fromisoformat(fin)

    # Estaciones más cercanas
    estaciones = ms.stations.nearby(punto, limit=4)
    try:
        print("  Estaciones cercanas encontradas:")
        print(estaciones[["name", "country", "distance"]].head())
    except Exception:
        print(f"  Estaciones cercanas: {list(getattr(estaciones, 'index', []))[:4]}")

    # Serie diaria interpolada al punto
    ts = ms.daily(estaciones, ini, fin_)
    df = ms.interpolate(ts, punto).fetch()

    if df is None or df.empty:
        print("  [aviso] Meteostat no devolvió datos para ese punto/fecha.")
        return None

    # Mostrar los nombres reales de columnas que devuelve la v2
    print(f"  Columnas devueltas por Meteostat: {list(df.columns)}")

    # Renombrar (solo las que existan) a un esquema claro en español.
    # Nombres reales de Meteostat v2:
    #   temp °C | tmin/tmax °C | rhum % | prcp mm | wspd km/h | pres hPa | cldc octas
    mapa = {
        "temp": "temp_aire_c",
        "tmin": "temp_min_c",
        "tmax": "temp_max_c",
        "rhum": "humedad_pct",
        "prcp": "precip_mm",
        "wspd": "viento_kmh",
        "wdir": "viento_dir_deg",
        "pres": "presion_hpa",
        "cldc": "nubosidad_octas",
    }
    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})
    df.index.name = "Fecha"
    df["fuente"] = "Meteostat v2 (estaciones GHCN/ISD)"

    # Guardar en data/raw de la RAÍZ del proyecto
    ruta_base = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ruta_final = os.path.join(ruta_base, ruta_guardado)
    os.makedirs(os.path.dirname(ruta_final), exist_ok=True)

    df.to_csv(ruta_final)
    print(f"¡Dataset 3 (Meteostat) guardado con éxito en: {ruta_final}!")
    return df


def verificar(df, variables, nombre):
    """Confirma que las variables de interés existan y tengan datos reales."""
    print(f"\n--- Verificación {nombre} ---")
    if df is None or df.empty:
        print("  El DataFrame está vacío (revisá la conexión).")
        return
    for v in variables:
        if v not in df.columns:
            print(f"  FALTA la columna '{v}'")
        else:
            validos = df[v].notna().sum()
            estado = "OK " if validos > 0 else "VACÍO"
            print(f"  [{estado}] {v}: {validos}/{len(df)} valores válidos")


if __name__ == "__main__":
    print("=== DATASET 3: Meteorología de estaciones (Meteostat v2) ===")
    df3 = extraer_datos_meteostat()

    verificar(df3, ["viento_kmh", "presion_hpa"], "Dataset 3 (Meteostat)")

    if df3 is not None:
        print("\nMuestra de los datos reales de estación:")
        print(df3.head())
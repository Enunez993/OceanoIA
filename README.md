# 🌊 OceanoIA — Asistente Inteligente para Monitoreo Costero y Pesca Sostenible

> Proyecto académico del Colegio Universitario de Cartago (CUC) · Curso de Inteligencia Artificial

OceanoIA es un sistema inteligente que apoya a **INCOPESCA, MarViva, Guardacostas y pescadores artesanales** de Costa Rica integrando tres modelos de aprendizaje profundo en una sola plataforma:

| Módulo | Modelo | Objetivo |
|--------|--------|----------|
| 🐟 **Identificación de especies** | **CNN** | Clasificar peces capturados para verificar vedas, tallas mínimas y especies protegidas |
| 🌊 **Pronóstico oceanográfico** | **RNN / LSTM** | Predecir oleaje, viento, marea y SST en las próximas 24–72 h |
| 🎯 **Recomendación de pesca** | **ANN** | Sugerir acciones sostenibles (pescar, cambiar zona, regresar a puerto, etc.) |

---

## 📂 Estructura del proyecto

```
OceanoIA/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/                    # Datos crudos (imágenes, CSV, mseed...)
│   └── processed/              # Datos limpios listos para entrenar
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_CNN_Especies.ipynb
│   ├── 03_RNN_Oceanografia.ipynb
│   └── 04_ANN_Recomendacion.ipynb
├── src/
│   ├── data_prep.py            # Preprocesamiento general
│   ├── marine_api.py           # Cliente Open-Meteo Marine
│   └── train/
│       ├── cnn.py
│       ├── rnn.py
│       └── ann.py
├── models/                     # Modelos entrenados .h5 / .keras
├── api/
│   └── main.py                 # FastAPI: /predict/especie, /predict/oceano, /predict/accion
└── app/
    ├── Home.py                 # Frontend Streamlit
    └── assets/
```

---

## 🚀 Instalación rápida

### Requisitos previos
- Python 3.10+
- pip
- (Opcional) GPU con CUDA para acelerar el entrenamiento de la CNN

### Pasos

```bash
# 1. Clonar el repositorio
git clone https://github.com/Enunez993/OceanoIA.git
cd OceanoIA

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
# venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Lanzar Jupyter para los notebooks
jupyter notebook notebooks/
```

---

## 🧪 Uso

### Entrenar los modelos

```bash
# CNN — Identificación de especies
python src/train/cnn.py

# RNN — Pronóstico oceanográfico
python src/train/rnn.py

# ANN — Recomendación de pesca
python src/train/ann.py
```

Los modelos entrenados se guardan en `models/`.

### Lanzar la app demo

```bash
streamlit run app/Home.py
```

Abrirá `http://localhost:8501` con cuatro pestañas:
- 📷 Identificador de especies
- 🌊 Pronóstico oceánico
- 🗺️ Mapa de recomendaciones
- 📊 Dashboard combinado

### Lanzar la API REST (opcional)

```bash
uvicorn api.main:app --reload
```

Endpoints disponibles:
- `POST /predict/especie` (imagen)
- `POST /predict/oceano` (serie temporal)
- `POST /predict/accion` (features tabulares)

---

## 📊 Datasets utilizados

| Dataset | Uso | URL |
|---------|-----|-----|
| Large-Scale Fish Dataset | CNN | https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset |
| Open-Meteo Marine API | RNN (oleaje, marea) | https://open-meteo.com/en/docs/marine-weather-api |
| Copernicus Marine Service | RNN (SST) | https://data.marine.copernicus.eu/ |
| Meteostat | RNN (viento, presión) | https://meteostat.net/ |
| Sintético (sklearn) | ANN | Generado con `make_classification` + reglas INCOPESCA |
---
El sistema combina tres fuentes independientes y reales, más una variable
calculada localmente:

| Dataset | Fuente | Variables que aporta | Requiere credenciales |
|---------|--------|----------------------|-----------------------|
| 1 | **Open-Meteo Marine API** | Oleaje, periodo y dirección de ola, marea (nivel del mar) | No |
| 2 | **Copernicus Marine Service** | Temperatura superficial del mar (SST) | Sí (registro gratuito) |
| 3 | **Meteostat v2** (estaciones GHCN/ISD) | Viento, presión, temperatura del aire, humedad, nubosidad | No |
| — | Cálculo astronómico | Fase lunar | No |

Configuración de credenciales — Copernicus Marine

El Dataset 2 (SST) descarga datos del Copernicus Marine Service, que requiere
una cuenta gratuita. Registro: https://data.marine.copernicus.eu/register

Hay dos formas de autenticarse:

| Opción | Método | Comando | Ventaja |
|--------|--------|---------|---------|
| **A**  | Iniciar sesión una sola vez | `copernicusmarine login` | Guarda las credenciales de forma segura |
| **B** | Variables de entorno | `setx CMEMS_USER "tu_usuario"` <br> `setx CMEMS_PASS "tu_contraseña"` | El script las lee automáticamente si existen |


## 📈 Métricas objetivo

| Módulo | Métrica | Umbral |
|--------|---------|--------|
| CNN | Accuracy / F1-score | ≥ 90% |
| RNN | RMSE / MAE | Bajo (según escala) |
| ANN | Precision / Recall | ≥ 85% |

---

## 🏆 Rúbrica de evaluación

| Peso | Criterio |
|------|----------|
| 40 % | Modelo (métricas sobre datos reales) |
| 30 % | Producto (app funcional, demo en vivo) |
| 20 % | Documentación (informe técnico + README) |
| 10 % | Innovación (funcionalidades extra) |

---

## 🌟 Innovaciones implementadas

- ✅ Integración en tiempo real con **Open-Meteo Marine API**
- ✅ **Alerta automática** al detectar especies protegidas (tortugas, tiburones martillo, marlines)
- ✅ Mapa interactivo con **Folium** mostrando zonas pesqueras y Áreas Marinas Protegidas (AMP)
- 🔄 Modo offline (TensorFlow Lite) — *en progreso*
- 🔄 Bot Telegram/WhatsApp — *en progreso*

---

## 🤝 Equipo

- [Eduardo Núñez Morales] — *702080987@cuc.ac.cr*
- [Daniel Nájera Gómez] — *305620467@cuc.cr*
- [Stephanie Rodriguez Cortes] — *305630770@cuc.cr*

**Profesor:** Osvaldo González Chaves
**Curso:** Inteligencia Artificial 2026
**Fecha de entrega:** 20 de julio de 2026

---

## 📜 Licencia

Este proyecto es de uso académico. Datasets utilizados respetan sus licencias originales.

---

## 🌎 Impacto social

OceanoIA contribuye al **ODS 14 (Vida Submarina)** apoyando a **+14,000 pescadores artesanales** de Costa Rica y al monitoreo de los **+589,000 km²** de mar territorial nacional, alineado con la estrategia de **economía azul** del MINAE.

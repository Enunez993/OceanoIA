"""
Home.py
=======
App Streamlit demo de OceanoIA con cuatro pestañas:
  1. 📷 Identificador de especies (CNN)
  2. 🌊 Pronóstico oceánico (RNN)
  3. 🗺️ Mapa de recomendaciones (ANN + Folium)
  4. 📊 Dashboard combinado

Ejecutar:
    streamlit run app/Home.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import pandas as pd
import streamlit as st

# ===================== Config general =====================
st.set_page_config(
    page_title="OceanoIA · Pesca Sostenible",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ===================== Sidebar =====================
st.sidebar.image(
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Flag_of_Costa_Rica.svg/320px-Flag_of_Costa_Rica.svg.png",
    width=180,
)
st.sidebar.title("🌊 OceanoIA")
st.sidebar.markdown("**Monitoreo Costero y Pesca Sostenible en Costa Rica**")
st.sidebar.markdown("---")
st.sidebar.markdown("**Proyecto académico — CUC**")
st.sidebar.markdown("Curso de Inteligencia Artificial 2026")

# ===================== Header principal =====================
st.title("🌊 OceanoIA")
st.markdown(
    "#### Asistente Inteligente para **Monitoreo Costero y Pesca Sostenible**"
)
st.markdown("---")

# ===================== Tabs =====================
tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Identificar Especie",
    "🌊 Pronóstico Oceánico",
    "🗺️ Mapa de Recomendaciones",
    "📊 Dashboard",
])

# ===================== TAB 1: CNN Identificación =====================
with tab1:
    st.header("📷 Identificación de especies marinas (CNN)")
    st.markdown(
        "Sube una foto del pez capturado. El modelo identifica la especie y verifica "
        "si está en veda, es protegida o cumple talla mínima."
    )

    uploaded = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png"])
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            st.image(uploaded, caption="Imagen cargada", use_column_width=True)
        with col2:
            with st.spinner("Analizando..."):
                # ===== Modelo Real Cargado =====
                import tensorflow as tf
                from tensorflow.keras.preprocessing import image
                
                # Cargar el modelo
                model = tf.keras.models.load_model(ROOT / "models" / "cnn_especies.keras")
                
                # Cargar y preprocesar la imagen
                img = image.load_img(uploaded, target_size=(128, 128))
                arr = np.expand_dims(image.img_to_array(img) / 255.0, 0).astype("float32")
                
                # Predecir
                preds = model.predict(arr, verbose=0)[0]
                
                # Mapear a las clases con nombres legibles
                especies = [
                    "Atún aleta amarilla", "Corvina reina", "Dorado", "Marlín",
                    "Otros", "Pargo mancha", "Tiburón martillo", "Tortuga marina"
                ]
                idx = int(np.argmax(preds))

            st.success(f"**Especie:** {especies[idx]}")
            st.metric("Confianza", f"{preds[idx]*100:.1f}%")

            protegidas = ["Marlín", "Tortuga marina", "Tiburón martillo"]
            if especies[idx] in protegidas:
                st.error("🚨 **ALERTA — Especie protegida.** Devolver al mar inmediatamente.")
            else:
                st.info("✅ Especie no protegida. Verificar talla mínima y veda.")

            st.markdown("**Probabilidades por clase:**")
            df_probs = pd.DataFrame({"Especie": especies, "Probabilidad": preds})
            st.bar_chart(df_probs.set_index("Especie"))

# ===================== TAB 2: RNN Pronóstico (Con Modelo Real) =====================
with tab2:
    st.header("🌊 Pronóstico oceánico — próximas 72 horas")
    st.markdown("Predicción de oleaje usando red neuronal LSTM entrenada.")

    zona = st.selectbox("Selecciona una zona:", [
        "golfo_nicoya", "golfo_dulce", "pacifico_norte",
        "pacifico_central", "pacifico_sur", "caribe_norte", "caribe_sur",
    ])

    st.caption(
        "Modelo LSTM corriendo"
    )

    if st.button("Obtener pronóstico"):
        with st.spinner("Ejecutando modelo LSTM..."):
            try:
                import tensorflow as tf
                import joblib

                # Las MISMAS 11 variables del entrenamiento (src/train/rnn.py).
                # 'Fase_Lunar' es derivada: la calcula marine_api.fase_lunar y desde
                # ahora sí queda guardada en el CSV, así que el modelo usa las 11.
                FEATURES_DISPONIBLES = [
                    "Oleaje_m", "Periodo_Oleaje_s", "Direccion_Oleaje_deg", "Marea_m",
                    "SST_Copernicus", "temp_aire_c", "viento_kmh", "presion_hpa",
                    "humedad_pct", "nubosidad_octas", "Fase_Lunar",
                ]
                WINDOW_IN = 30   # días de historia que espera el modelo

                # 1. Cargar el histórico (mismo archivo y mismo orden que en el entrenamiento)
                df = pd.read_csv(
                    ROOT / "data" / "processed" / "oceano_merged.csv",
                    parse_dates=["Fecha"], index_col="Fecha",
                ).sort_index()

                features = [c for c in FEATURES_DISPONIBLES if c in df.columns]
                df[features] = df[features].interpolate(method="time")

                # 2. Cargar modelo y scaler
                modelo = tf.keras.models.load_model(ROOT / "models" / "lstm_oleaje_m.keras")
                scaler = joblib.load(ROOT / "models" / "scaler_oleaje_m.pkl")

                # 3. Escalar y tomar la última ventana de 30 días
                data_scaled = scaler.transform(df[features].values)
                input_data = data_scaled[-WINDOW_IN:].reshape(1, WINDOW_IN, len(features))

                # 4. Predicción (3 días = 72 h)
                pred_scaled = modelo.predict(input_data, verbose=0)

                # 5. Desescalar (misma lógica de desescalar_target en rnn.py)
                target_idx = features.index("Oleaje_m")
                dummy = np.zeros((pred_scaled.size, len(features)))
                dummy[:, target_idx] = pred_scaled.flatten()
                inv = scaler.inverse_transform(dummy)[:, target_idx]

                # 6. Mostrar resultados
                fechas_pred = pd.date_range(
                    df.index[-1] + pd.Timedelta(days=1), periods=len(inv), freq="D"
                )

                cols = st.columns(len(inv))
                for i, (col, valor) in enumerate(zip(cols, inv), start=1):
                    col.metric(f"Oleaje +{i*24}h", f"{valor:.2f} m")

                serie = pd.concat([
                    df["Oleaje_m"].tail(60).rename("Histórico"),
                    pd.Series(inv, index=fechas_pred, name="Pronóstico"),
                ], axis=1)
                st.line_chart(serie)

                st.dataframe(
                    pd.DataFrame({
                        "Fecha": fechas_pred.strftime("%d/%m/%Y"),
                        "Oleaje predicho (m)": np.round(inv, 2),
                    }),
                    hide_index=True,
                    use_container_width=True,
                )

            except FileNotFoundError as e:
                st.error(f"Falta un archivo requerido: {e}")
            except Exception as e:
                st.error(f"Error al ejecutar el modelo: {e}")
# ===================== TAB 3: ANN + Mapa =====================
with tab3:
    st.header("🗺️ Recomendación de pesca + mapa interactivo")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("**Ingresa las condiciones actuales:**")
        altura  = st.slider("Altura del oleaje (m)", 0.0, 5.0, 1.5, 0.1)
        viento  = st.slider("Viento (km/h)", 0, 80, 20)
        sst     = st.slider("Temperatura del mar (°C)", 22.0, 32.0, 28.0, 0.1)
        dist    = st.slider("Distancia a costa (km)", 1, 80, 20)
        especie = st.selectbox("Especie objetivo", [
            "dorado", "atun", "pargo", "corvina",
            "marlin", "tortuga_marina", "tiburon_martillo",
            "otro",
        ])
        veda    = st.checkbox("¿Hay veda activa?")
        amp     = st.checkbox("¿Es Área Marina Protegida?")

        if st.button("Obtener recomendación"):
            try:
                import joblib

                # Clases tal como se etiquetaron en src/train/ann.py
                CLASES = {
                    0: ("REGRESAR A PUERTO", "⚫", "Alerta meteorológica: oleaje o viento peligrosos."),
                    1: ("NO PESCAR", "🔴", "Veda activa, área marina protegida o especie restringida."),
                    2: ("PESCA CON PRECAUCIÓN", "🟡", "Condiciones aceptables con restricciones."),
                    3: ("PESCA RECOMENDADA", "🟢", "Condiciones óptimas para faenar."),
                }
                # La ANN recibe 'especie' como código numérico.
                # El 7 está reservado para especies protegidas: es el único valor que
                # dispara la regla `especie == 7` de src/train/ann.py (-> NO PESCAR).
                ESPECIE_COD = {
                    "dorado": 0, "atun": 1, "pargo": 2, "corvina": 3, "otro": 4,
                    "marlin": 7, "tortuga_marina": 7, "tiburon_martillo": 7,
                }

                modelo_ann = joblib.load(ROOT / "models" / "ann_clasificador_pesca.pkl")
                scaler_ann = joblib.load(ROOT / "models" / "scaler_ann_pesca.pkl")

                # Mismo orden de columnas que en el entrenamiento
                entrada = pd.DataFrame([{
                    "altura": altura,
                    "viento": float(viento),
                    "sst": sst,
                    "distancia": float(dist),
                    "especie": ESPECIE_COD[especie],
                    "veda": int(veda),
                    "amp": int(amp),
                }])

                entrada_scaled = scaler_ann.transform(entrada)
                pred = int(modelo_ann.predict(entrada_scaled)[0])
                probas = modelo_ann.predict_proba(entrada_scaled)[0]

                rec, color, msg = CLASES[pred]
                st.markdown(f"### {color} **{rec}**")
                st.info(msg)
                st.metric("Confianza del modelo", f"{probas[pred]*100:.1f}%")

                st.markdown("**Probabilidad por recomendación:**")
                st.bar_chart(pd.DataFrame(
                    {"Probabilidad": probas},
                    index=[CLASES[c][0] for c in modelo_ann.classes_],
                ))

            except FileNotFoundError:
                st.error(
                    "No se encontró el modelo ANN. Ejecuta primero: "
                    "`python src/train/ann.py` desde la raíz del proyecto."
                )
            except Exception as e:
                st.error(f"Error al ejecutar la ANN: {e}")

    with col2:
        try:
            import folium
            from streamlit_folium import st_folium
            from src.marine_api import ZONAS_CR

            m = folium.Map(location=[9.7, -84.0], zoom_start=7, tiles="CartoDB positron")
            for nombre, (lat, lon) in ZONAS_CR.items():
                folium.Marker(
                    [lat, lon],
                    popup=f"<b>{nombre.replace('_', ' ').title()}</b>",
                    icon=folium.Icon(color="blue", icon="anchor", prefix="fa"),
                ).add_to(m)
            st_folium(m, width=700, height=500)
        except ImportError:
            st.warning("Instala `folium` y `streamlit-folium` para ver el mapa.")

# ===================== TAB 4: Dashboard =====================
with tab4:
    st.header("📊 Dashboard integrado")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Especies catalogadas", "8")
    col2.metric("Zonas monitoreadas", "7")
    col3.metric("Pescadores beneficiados", "14,000+")
    col4.metric("Km² de mar territorial", "589,000")

    st.markdown("---")
    st.subheader("Datasets utilizados")
    st.table(pd.DataFrame({
        "Módulo": ["CNN", "RNN/LSTM", "ANN"],
        "Dataset principal": [
            "Large-Scale Fish Dataset (Kaggle)",
            "Open-Meteo Marine API + NOAA + IMN",
            "Sintético (sklearn) + reglas INCOPESCA",
        ],
        "Métrica":  ["Accuracy / F1 ≥ 90%", "RMSE / MAE", "Precision / Recall ≥ 85%"],
    }))

    st.markdown("---")
    st.markdown(
        "**Equipo:** [tu equipo]  ·  **Profesor:** [docente]  ·  "
        "**Curso:** Inteligencia Artificial 2026  ·  "
        "**Entrega:** 20 de julio 2026"
    )

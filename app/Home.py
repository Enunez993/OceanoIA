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

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
                # ===== Demo placeholder (descomentar cuando el modelo esté entrenado) =====
                # import tensorflow as tf
                # from tensorflow.keras.preprocessing import image
                # model = tf.keras.models.load_model("models/cnn_especies.keras")
                # img = image.load_img(uploaded, target_size=(128, 128))
                # arr = np.expand_dims(image.img_to_array(img) / 255.0, 0)
                # preds = model.predict(arr)[0]
                # ...

                # Demo simulada
                especies = ["Dorado", "Atún aleta amarilla", "Pargo mancha",
                            "Corvina reina", "Marlín", "Tortuga marina"]
                preds = np.random.dirichlet(np.ones(len(especies)))
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


# ===================== TAB 2: RNN Pronóstico =====================
with tab2:
    st.header("🌊 Pronóstico oceánico — próximas 72 horas")
    st.markdown("Predicción de oleaje, temperatura del mar y viento usando LSTM.")

    zonas = [
        "golfo_nicoya", "golfo_dulce", "pacifico_norte",
        "pacifico_central", "pacifico_sur",
        "caribe_norte", "caribe_sur",
    ]
    zona = st.selectbox("Selecciona una zona costera:", zonas)

    # Coordenadas aproximadas de cada zona costera de Costa Rica
    zonas_cord = {
        "golfo_nicoya":     (9.97, -84.83),
        "golfo_dulce":      (8.70, -83.30),
        "pacifico_norte":   (10.60, -85.70),
        "pacifico_central": (9.60, -84.70),
        "pacifico_sur":     (8.50, -83.50),
        "caribe_norte":     (10.50, -83.60),
        "caribe_sur":       (9.60, -82.80),
    }

    if st.button("Obtener pronóstico"):
        with st.spinner("Descargando datos de Open-Meteo Marine..."):
            try:
                import joblib
                import numpy as np
                import tensorflow as tf
                from rnn import cargar_datos, preparar_features, desescalar_target

                WINDOW_IN = 30
                DIAS_PRONOSTICO = 3
                TARGETS = ["Oleaje_m", "Marea_m", "SST_Copernicus"]

                df = cargar_datos()
                df, features = preparar_features(df)

                col1, col2, col3 = st.columns(3)

                col1.metric("Oleaje máx. 72h", f"{df['Oleaje_m'].max():.2f} m")
                col2.metric("SST promedio",     f"{df['SST_Copernicus'].mean():.1f} °C")
                col3.metric("Período promedio", f"{df['Periodo_Oleaje_s'].mean():.1f} s")

                st.line_chart(df[["Oleaje_m"]], use_container_width=True)
                st.line_chart(df[["SST_Copernicus"]], use_container_width=True)


                window_in = 30
                dias_p = 3
                targets = ["Oleaje_m", "Marea_m", "SST_Copernicus"]

                df_base, features = preparar_features(cargar_datos())

                lat, lon = zonas_cord[zona]
                fecha_fin = df_base.index[-1]
                fecha_ini = fecha_fin - pd.Timedelta(days=window_in + 5)

                resp = requests.get(
                    "https://marine-api.open-meteo.com/v1/marine",
                    params={
                        "latitude": lat,
                        "longitude": lon,
                        "hourly": "wave_height,wave_period,wave_direction,sea_level_height_msl",
                        "start_date": fecha_ini.date().isoformat(),
                        "end_date": fecha_fin.date().isoformat(),
                        "timezone": "America/Costa_Rica",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                hourly = resp.json()["hourly"]

                df_zona = pd.DataFrame({
                    "Fecha": pd.to_datetime(hourly["time"]),
                    "Oleaje_m": hourly["wave_height"],
                    "Periodo_Oleaje_s": hourly["wave_period"],
                    "Direccion_Oleaje_deg": hourly["wave_direction"],
                    "Marea_m": hourly["sea_level_height_msl"],
                }).set_index("Fecha")
                df_zona = df_zona.resample("D").mean().interpolate()


                df_final = df_zona.copy()
                for col in features:
                    if col not in df_final.columns:
                        df_final[col] = df_base[col].reindex(df_final.index, method="nearest")
                df_final = df_final[features].tail(window_in).interpolate()

                #Correr el modelo LSTM entrenado sobre la ventana de la zona
                if len(df_final) < window_in:
                    st.warning("No hay suficientes dias de datos para esta zona todavía.")
                else:
                    st.markdown(f"**Pronostico LSTM próximos {dias_p} días ({zona}):**")
                    resultados = {}

                    # Generar las fechas futuras correctas para el pronóstico
                    fechas_pred = pd.date_range(df_final.index[-1] + pd.Timedelta(days=1), periods=dias_p,
                                                freq="D")

                    for target in targets:
                        ruta_modelo = f"models/lstm_{target.lower()}.keras"
                        ruta_scaler = f"models/scaler_{target.lower()}.pkl"
                        if not (Path(ruta_modelo).exists() and Path(ruta_scaler).exists()):
                            continue

                        modelo = tf.keras.models.load_model(ruta_modelo)
                        scaler = joblib.load(ruta_scaler)
                        target_idx = features.index(target)

                        data_scaled = scaler.transform(df_final[features].values)
                        ultima_ventana = np.expand_dims(data_scaled[-window_in:], axis=0)

                        pred_scaled = modelo.predict(ultima_ventana, verbose=0)
                        pred = desescalar_target(pred_scaled, scaler, target_idx, len(features))[0]

                        # Guarda los valores correspondientes a los dias futuros
                        resultados[target] = pred[:dias_p]

                    if resultados:
                        # Crea el DataFrame exclusivo con las predicciones del futuro
                        df_pronostico = pd.DataFrame(resultados, index=fechas_pred).round(2)

                        #leen los valores máximos y promedios del PRONOSTICO FUTURO
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Oleaje máx. 72h", f"{df_pronostico['Oleaje_m'].max():.2f} m")
                        col2.metric("SST promedio", f"{df_pronostico['SST_Copernicus'].mean():.1f} °C")
                        # Agregamos la marea promedio del pronóstico para usar la tercera columna
                        col3.metric("Marea máx. 72h", f"{df_pronostico['Marea_m'].max():.2f} m")

                        #muestran las curvas de las próximas 72 horas predichas
                        st.line_chart(df_pronostico[["Oleaje_m"]], use_container_width=True)
                        st.line_chart(df_pronostico[["SST_Copernicus"]], use_container_width=True)

                        # Muestra la tabla con las predicciones calculadas
                        st.dataframe(df_pronostico)
                    else:
                        st.info("Todavía no hay modelos entrenados en /models.")


            except Exception as e:
                st.error(f"Error: {e}")



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
        especie = st.selectbox("Especie objetivo", ["dorado", "atun", "pargo", "corvina", "otro"])
        veda    = st.checkbox("¿Hay veda activa?")
        amp     = st.checkbox("¿Es Área Marina Protegida?")

        if st.button("Obtener recomendación"):
            # Lógica simplificada (espejo de las reglas; sustituir por modelo real)
            if altura > 2.5 or viento > 35:
                rec, color, msg = "REGRESAR A PUERTO", "⚫", "Alerta meteorológica."
            elif veda or amp:
                rec, color, msg = "NO PESCAR", "🔴", "Veda activa o área protegida."
            elif altura > 1.8 or viento > 25:
                rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Restricciones leves."
            elif dist > 60:
                rec, color, msg = "CAMBIAR ZONA", "🔵", "Hay zona alternativa más cercana."
            elif 26 <= sst <= 30 and altura < 1.5:
                rec, color, msg = "PESCA RECOMENDADA", "🟢", "Condiciones óptimas."
            else:
                rec, color, msg = "PESCA CON PRECAUCIÓN", "🟡", "Condiciones aceptables."

            st.markdown(f"### {color} **{rec}**")
            st.info(msg)

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
        "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
        "**Equipo:** [Eduardo Nuñez Morales,Daniel Najera Gomez, Stephanie Rodriguez Cortes]  ·  **Profesor:** [Osvaldo Gonzalez Chavez]  ·  <br>"
        "**Curso:** Inteligencia Artificial 2026  ·  "
        "**Entrega:** 13 de julio 2026"
        "</div>",
        unsafe_allow_html=True
    )
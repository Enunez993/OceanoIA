# train_ann.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
N_POOL = 200_000       # pool crudo; se sobre-genera para poder balancear
N_POR_CLASE = 2000     # muestras finales por cada una de las 4 clases


def entrenar_red_neuronal(retornar_datos=False):
    print("==================================================")
    print("iniciando el pipeline de entrenamiento de la ann")
    print("==================================================")

    # 1. generación del pool sintético con variables INDEPENDIENTES
    # (make_classification correlacionaba las features entre sí -viento/especie: -0.81-,
    #  algo sin sentido físico que impedía que las reglas se cumplieran en combinación)
    print("[1/5] generando pool sintético con variables independientes...")
    rng = np.random.default_rng(RANDOM_STATE)

    pool = pd.DataFrame({
        "altura":    rng.uniform(0.0, 5.0, N_POOL),      # rango del slider de la app
        "viento":    rng.uniform(0.0, 80.0, N_POOL),
        "sst":       rng.uniform(22.0, 32.0, N_POOL),
        "distancia": rng.uniform(1.0, 80.0, N_POOL),
        "especie":   rng.integers(0, 8, N_POOL),         # 8 clases de especie
        "veda":      rng.binomial(1, 0.25, N_POOL),
        "amp":       rng.binomial(1, 0.20, N_POOL),
    })

    # 2. aplicación de las reglas de negocio de incopesca (vectorizada)
    # np.select respeta el orden de prioridad, igual que la cadena if/elif original
    print("[2/5] aplicando etiquetado supervisado (reglas incopesca)...")
    condiciones = [
        (pool["altura"] > 2.5) | (pool["viento"] > 35),                    # 0 regresar a puerto
        (pool["veda"] == 1) | (pool["amp"] == 1) | (pool["especie"] == 7), # 1 no pescar
        (pool["altura"] > 1.8) | (pool["viento"] > 25),                    # 2 precaución
        (pool["sst"].between(26.0, 30.0)) & (pool["altura"] < 1.5),        # 3 recomendada
    ]
    pool["target"] = np.select(condiciones, [0, 1, 2, 3], default=2)

    # 3. balanceo por clase
    # Sin esto, "pesca recomendada" quedaba en 9 de 5000 muestras (0.18%) y el
    # modelo jamás la predecía, que es justo la recomendación más útil de la app.
    print("[3/5] balanceando el dataset por clase...")
    print(f"   pool crudo: {dict(pool.target.value_counts().sort_index())}")

    partes = [
        grupo.sample(n=min(N_POR_CLASE, len(grupo)), random_state=RANDOM_STATE)
        for _, grupo in pool.groupby("target")
    ]
    df = pd.concat(partes).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"   balanceado: {dict(df.target.value_counts().sort_index())}")

    # 4. preparación, escalado final y entrenamiento
    x = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    scaler_ann = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler_ann.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler_ann.transform(x_test), columns=x_test.columns)

    print("[4/5] entrenando el perceptrón multicapa (ann)...")
    ann_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=RANDOM_STATE
    )
    ann_model.fit(x_train_scaled, y_train)

    accuracy = ann_model.score(x_test_scaled, y_test)
    print(f" ¡modelo entrenado con éxito! precisión en test: {accuracy * 100:.2f}%")

    # Reporte por clase: con el dataset balanceado el accuracy global ya no
    # oculta el desempeño de "pesca recomendada"
    print(classification_report(
        y_test, ann_model.predict(x_test_scaled),
        target_names=["regresar a puerto", "no pescar", "precaución", "recomendada"],
    ))

    # 5. exportar
    print("[5/5] exportando archivos binarios (.pkl)...")
    ruta_modelos = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")), "models"
    )
    os.makedirs(ruta_modelos, exist_ok=True)
    joblib.dump(ann_model, os.path.join(ruta_modelos, "ann_clasificador_pesca.pkl"))
    joblib.dump(scaler_ann, os.path.join(ruta_modelos, "scaler_ann_pesca.pkl"))

    print("==================================================")
    print("proceso finalizado. archivos listos en /models")
    print("==================================================")

    if retornar_datos:
        return x_test_scaled, y_test


if __name__ == "__main__":
    entrenar_red_neuronal()




# train_ann.py
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def entrenar_red_neuronal(retornar_datos=False):
    print("==================================================")
    print("iniciando el pipeline de entrenamiento de la ann")
    print("==================================================")

    # 1. generación del dataset base con make_classification
    print("[1/5] generando matriz matemática sintética con sklearn...")
    x_raw, _ = make_classification(
        n_samples=5000,
        n_features=7,
        n_informative=5,
        n_redundant=2,
        random_state=42
    )

    # 2. escalamiento y transformación a rangos marinos de costa rica
    print("[2/5] calibrando variables según el entorno costarricense...")
    scaler_rangos = MinMaxScaler()
    x_scaled_steps = scaler_rangos.fit_transform(x_raw)

    altura_oleaje = x_scaled_steps[:, 0] * 5.0
    viento = x_scaled_steps[:, 1] * 80.0
    sst = 22.0 + (x_scaled_steps[:, 2] * 10.0)
    distancia = 1.0 + (x_scaled_steps[:, 3] * 79.0)
    especie = np.floor(x_scaled_steps[:, 4] * 7.99).astype(int)

    veda = (x_scaled_steps[:, 5] > 0.75).astype(int)
    amp = (x_scaled_steps[:, 6] > 0.80).astype(int)

    df = pd.DataFrame({
        "altura": altura_oleaje,
        "viento": viento,
        "sst": sst,
        "distancia": distancia,
        "especie": especie,
        "veda": veda,
        "amp": amp
    })

    # 3. aplicación de las reglas de negocio de incopesca
    print("[3/5] aplicando etiquetado supervisado (reglas incopesca)...")
    target = []
    for idx, row in df.iterrows():
        if row["altura"] > 2.5 or row["viento"] > 35:
            target.append(0)  # regresar a puerto
        elif row["veda"] == 1 or row["amp"] == 1 or row["especie"] == 7:
            target.append(1)  # no pescar
        elif row["altura"] > 1.8 or row["viento"] > 25:
            target.append(2)  # pesca con precaución
        elif 26.0 <= row["sst"] <= 30.0 and row["altura"] < 1.5:
            target.append(3)  # pesca recomendada
        else:
            target.append(2)

    df["target"] = target

    # 4. preparación, escalado final y entrenamiento
    x = df.drop(columns=["target"])
    y = df["target"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)

    scaler_ann = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler_ann.fit_transform(x_train), columns=x_train.columns)
    x_test_scaled = pd.DataFrame(scaler_ann.transform(x_test), columns=x_test.columns)

    print("[4/5] entrenando el perceptrón multicapa (ann)...")
    ann_model = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=500,
        random_state=42
    )
    ann_model.fit(x_train_scaled, y_train)

    accuracy = ann_model.score(x_test_scaled, y_test)
    print(f" ¡modelo entrenado con éxito! precisión en test: {accuracy * 100:.2f}%")

    # 5. exportar
    print("[5/5] exportando archivos binarios (.pkl)...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(ann_model, "models/ann_clasificador_pesca.pkl")
    joblib.dump(scaler_ann, "models/scaler_ann_pesca.pkl")

    print("==================================================")
    print("proceso finalizado. archivos listos en /models")
    print("==================================================")

    if retornar_datos:
        return x_test_scaled, y_test


if __name__ == "__main__":
    entrenar_red_neuronal()




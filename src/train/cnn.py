"""
Script para entrenar la red neuronal convolucional (CNN) de OceanoIA
para la clasificación de 8 especies marinas.
Guarda el modelo entrenado en 'models/cnn_especies.keras'.
"""

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

def main():
    # Directorios de datos
    train_path = r"C:\Users\usuario\Documents\GitHub\OceanoIA\data\processed\train"
    test_path = r"C:\Users\usuario\Documents\GitHub\OceanoIA\data\processed\test"
    models_dir = r"C:\Users\usuario\Documents\GitHub\OceanoIA\models"
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Configuración de hiperparámetros e imágenes
    image_shape = (128, 128, 3)
    batch_size = 32
    epochs = 20
    
    print("Inicializando Generadores de Imágenes...")
    # Generador con Aumento de Datos para Entrenamiento
    train_image_gen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.10,
        height_shift_range=0.10,
        rescale=1/255,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Generador de prueba (solo normaliza reescalando)
    test_image_gen = ImageDataGenerator(rescale=1/255)
    
    # Cargar los flujos de imágenes
    train_gen = train_image_gen.flow_from_directory(
        train_path,
        target_size=image_shape[:2],
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical'
    )
    
    test_gen = test_image_gen.flow_from_directory(
        test_path,
        target_size=image_shape[:2],
        color_mode='rgb',
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # Crucial para evaluación/matriz de confusión ordenada
    )
    
    print("\nClases detectadas e índices:")
    print(train_gen.class_indices)
    
    # 2. Definición del Modelo CNN (arquitectura del proyecto)
    print("\nConstruyendo arquitectura de la CNN...")
    model = Sequential()
    
    # Bloque 1
    model.add(Conv2D(filters=32, kernel_size=(3,3), input_shape=image_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Bloque 2
    model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Bloque 3
    model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Aplanado
    model.add(Flatten())
    
    # Capa Densa Totalmente Conectada
    model.add(Dense(256, activation='relu'))
    
    # Dropout (Evitar sobreajuste)
    model.add(Dropout(0.5))
    
    # Capa de Salida Softmax (8 clases)
    model.add(Dense(8, activation='softmax'))
    
    # Compilación del modelo
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 3. Entrenamiento con parada temprana
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    print("\nIniciando entrenamiento del modelo...")
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=test_gen,
        callbacks=[early_stop]
    )
    
    # 4. Guardar el modelo
    model_save_path = os.path.join(models_dir, "cnn_especies.keras")
    print(f"\nGuardando el modelo en {model_save_path}...")
    model.save(model_save_path)
    
    # 5. Generar gráficas de entrenamiento
    print("\nGenerando gráficas de curvas de aprendizaje...")
    losses = pd.DataFrame(model.history.history)
    
    plt.figure(figsize=(12, 4))
    
    # Pérdida
    plt.subplot(1, 2, 1)
    plt.plot(losses['loss'], label='Loss Entrenamiento')
    plt.plot(losses['val_loss'], label='Loss Validación')
    plt.title('Curva de Pérdida (Loss)')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Precisión
    plt.subplot(1, 2, 2)
    plt.plot(losses['accuracy'], label='Accuracy Entrenamiento')
    plt.plot(losses['val_accuracy'], label='Accuracy Validación')
    plt.title('Curva de Precisión (Accuracy)')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    chart_path = os.path.join(models_dir, "cnn_training_history.png")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"Gráficas de entrenamiento guardadas en {chart_path}")
    
    # 6. Evaluación del modelo
    print("\nEvaluando modelo en el conjunto de prueba...")
    eval_results = model.evaluate(test_gen)
    print(f"Pérdida en Test: {eval_results[0]:.4f}")
    print(f"Precisión (Accuracy) en Test: {eval_results[1]*100:.2f}%")
    
    # 7. Matriz de confusión y Reporte de Clasificación
    print("\nGenerando predicciones sobre el conjunto de test...")
    preds = model.predict(test_gen)
    pred_classes = np.argmax(preds, axis=-1)
    true_classes = test_gen.classes
    class_labels = list(test_gen.class_indices.keys())
    
    print("\nReporte de Clasificación:")
    report = classification_report(true_classes, pred_classes, target_names=class_labels)
    print(report)
    
    # Guardar reporte en un archivo de texto
    report_path = os.path.join(models_dir, "cnn_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("REPORTE DE CLASIFICACIÓN - CNN OCEANOIA\n")
        f.write("=======================================\n")
        f.write(report)
    print(f"Reporte de clasificación guardado en {report_path}")
    
    # Confusión
    cm = confusion_matrix(true_classes, pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Matriz de Confusión - Especies Marinas CNN')
    plt.ylabel('Clase Real')
    plt.xlabel('Predicción')
    
    cm_path = os.path.join(models_dir, "cnn_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Matriz de confusión guardada en {cm_path}")
    
    print("\n¡Entrenamiento y evaluación del modelo completados con éxito!")

if __name__ == "__main__":
    main()

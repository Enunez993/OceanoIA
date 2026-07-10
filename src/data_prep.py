"""
data_prep.py
============
Script para preprocesar el dataset de peces (Kaggle) y dividirlo físicamente en 
conjuntos de entrenamiento (80%) y prueba (20%) bajo 'data/processed/'.
Aplica la equivalencia de clases para obtener las 8 categorías deseadas de OceanoIA,
manteniendo un balance de clases perfecto (800 train, 200 test por clase).
"""

import os
import shutil
from tqdm import tqdm

def main():
    # Rutas base
    raw_dir = r"c:\BigData\Inteligencia Artificial Aplicada\OceanoIA\data\raw\archive\Fish_Dataset\Fish_Dataset"
    processed_dir = r"c:\BigData\Inteligencia Artificial Aplicada\OceanoIA\data\processed"
    
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

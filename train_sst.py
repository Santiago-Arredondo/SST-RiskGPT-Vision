#!/usr/bin/env python3
"""
Script optimizado para entrenar YOLO con clases específicas de SST
Incluye augmentación específica para escenarios de seguridad industrial
"""

import argparse
import yaml
from pathlib import Path
from ultralytics import YOLO
import torch

# Clases específicas SST basadas en normatividad colombiana
SST_CLASSES = [
    # Personas y EPP
    "person", "person_no_helmet", "person_no_vest", "person_incorrect_ppe",
    
    # Vehículos industriales (Res. 5018/2019)
    "forklift", "truck", "crane", "excavator", "loader",
    
    # Maquinaria (Res. 2400/1979)
    "machine", "conveyor", "press", "saw", "grinder", "lathe", "drill", "welder",
    
    # Elementos de seguridad
    "guard_missing", "emergency_stop", "safety_sign", "fire_extinguisher",
    
    # Riesgos locativos (GTC 45)
    "wet_floor", "spill", "cable", "obstacle", "hole", "damaged_floor",
    "broken_stair", "missing_handrail",
    
    # Almacenamiento (NTC 5227)
    "pallet", "shelf", "unstable_load", "overloaded_shelf",
    
    # Trabajo en alturas (Res. 4272/2021)
    "ladder", "scaffold", "harness", "edge_unprotected", "safety_net",
    
    # EPP individual
    "helmet", "safety_vest", "gloves", "safety_glasses", "safety_boots",
    
    # Herramientas
    "hand_tool", "power_tool", "damaged_tool",
    
    # Ergonomía oficina (Res. 4023/1997)
    "desk", "chair", "monitor", "keyboard", "laptop",
    
    # Iluminación
    "dark_area", "glare_source",
    
    # Señalización (NTC 1461)
    "warning_sign", "prohibition_sign", "mandatory_sign", "emergency_sign"
]

def create_sst_dataset_yaml(
    data_root: Path,
    output_path: Path,
    train_path: str = "images/train",
    val_path: str = "images/val",
    test_path: str = "images/test"
) -> None:
    """Crea archivo YAML para dataset SST"""
    
    yaml_content = {
        'path': str(data_root.absolute()),
        'train': train_path,
        'val': val_path,
        'test': test_path,
        'names': {i: name for i, name in enumerate(SST_CLASSES)},
        'nc': len(SST_CLASSES),
        
        # Metadatos SST
        'metadata': {
            'description': 'Dataset SST basado en normatividad colombiana',
            'version': '2.0',
            'regulations': [
                'Decreto 1072/2015',
                'Resolución 2400/1979',
                'Resolución 4272/2021',
                'GTC 45'
            ]
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
    
    print(f"✓ Dataset YAML creado: {output_path}")
    print(f"  Total de clases SST: {len(SST_CLASSES)}")

def get_training_hyperparams(scenario: str = "balanced") -> dict:
    """
    Retorna hiperparámetros optimizados para diferentes escenarios
    
    Args:
        scenario: 'speed' | 'balanced' | 'accuracy'
    """
    base_params = {
        'patience': 30,
        'save': True,
        'save_period': 10,
        'cache': True,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'workers': 8,
        'project': 'runs/sst',
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'verbose': True,
        'seed': 42,
        
        # Augmentación específica para SST
        'hsv_h': 0.015,  # Variación de tono
        'hsv_s': 0.7,    # Saturación
        'hsv_v': 0.4,    # Brillo
        'degrees': 10,    # Rotación máxima
        'translate': 0.1, # Traslación
        'scale': 0.3,     # Escala
        'shear': 5,       # Cizallamiento
        'flipud': 0.0,    # No voltear vertical (poco realista en SST)
        'fliplr': 0.5,    # Voltear horizontal sí es útil
        'mosaic': 1.0,    # Mosaic augmentation
        'mixup': 0.15,    # Mixup augmentation
        'copy_paste': 0.1, # Copy-paste augmentation para objetos pequeños
        
        # Específico para objetos pequeños (EPP, herramientas)
        'close_mosaic': 10,
    }
    
    scenarios = {
        'speed': {
            **base_params,
            'imgsz': 640,
            'batch': 16,
            'epochs': 100,
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
        },
        'balanced': {
            **base_params,
            'imgsz': 960,
            'batch': 8,
            'epochs': 200,
            'lr0': 0.01,
            'lrf': 0.001,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 5,
            'warmup_momentum': 0.8,
            'cos_lr': True,
        },
        'accuracy': {
            **base_params,
            'imgsz': 1280,
            'batch': 4,
            'epochs': 300,
            'lr0': 0.001,
            'lrf': 0.0001,
            'momentum': 0.937,
            'weight_decay': 0.001,
            'warmup_epochs': 10,
            'warmup_momentum': 0.8,
            'cos_lr': True,
            'label_smoothing': 0.1,
        }
    }
    
    return scenarios.get(scenario, scenarios['balanced'])

def train_sst_model(args):
    """Entrena modelo YOLO para detección SST"""
    
    # Preparar paths
    data_root = Path(args.data_root)
    dataset_yaml = Path(args.dataset_yaml)
    
    # Crear dataset YAML si no existe
    if not dataset_yaml.exists() or args.recreate_yaml:
        create_sst_dataset_yaml(data_root, dataset_yaml)
    
    # Verificar dataset
    for split in ['train', 'val']:
        split_path = data_root / 'images' / split
        if not split_path.exists():
            raise FileNotFoundError(f"No se encuentra {split_path}")
        
        img_count = len(list(split_path.glob('*.*')))
        print(f"  {split}: {img_count} imágenes")
    
    # Seleccionar modelo base
    model_size = args.model_size
    base_models = {
        'n': 'yolov8n.pt',  # Nano - más rápido
        's': 'yolov8s.pt',  # Small
        'm': 'yolov8m.pt',  # Medium
        'l': 'yolov8l.pt',  # Large
        'x': 'yolov8x.pt',  # Extra Large - más preciso
    }
    
    base_model = base_models.get(model_size, 'yolov8m.pt')
    
    # Si hay modelo pre-entrenado SST, usarlo
    if args.pretrained_sst and Path(args.pretrained_sst).exists():
        print(f"Usando modelo SST pre-entrenado: {args.pretrained_sst}")
        base_model = args.pretrained_sst
    
    # Inicializar modelo
    model = YOLO(base_model)
    
    # Obtener hiperparámetros
    hyperparams = get_training_hyperparams(args.scenario)
    
    # Ajustar batch size si se especifica
    if args.batch > 0:
        hyperparams['batch'] = args.batch
    
    # Ajustar epochs si se especifica
    if args.epochs > 0:
        hyperparams['epochs'] = args.epochs
    
    # Nombre del experimento
    hyperparams['name'] = f"sst_{model_size}_{args.scenario}_{hyperparams['imgsz']}"
    
    print("\n" + "="*50)
    print("CONFIGURACIÓN DE ENTRENAMIENTO SST")
    print("="*50)
    print(f"Modelo base: {base_model}")
    print(f"Escenario: {args.scenario}")
    print(f"Tamaño imagen: {hyperparams['imgsz']}")
    print(f"Batch size: {hyperparams['batch']}")
    print(f"Epochs: {hyperparams['epochs']}")
    print(f"Device: {hyperparams['device']}")
    print("="*50 + "\n")
    
    # Entrenar
    results = model.train(
        data=str(dataset_yaml),
        **hyperparams
    )
    
    # Validar modelo final
    print("\n" + "="*50)
    print("VALIDACIÓN FINAL")
    print("="*50)
    
    metrics = model.val(data=str(dataset_yaml))
    
    # Guardar mejor modelo con nombre descriptivo
    best_model_path = Path(hyperparams['project']) / hyperparams['name'] / 'weights' / 'best.pt'
    if best_model_path.exists():
        output_name = f"sst_model_{model_size}_{args.scenario}.pt"
        output_path = Path('models') / output_name
        output_path.parent.mkdir(exist_ok=True)
        
        import shutil
        shutil.copy(best_model_path, output_path)
        print(f"\n✓ Modelo guardado como: {output_path}")
        
        # Generar reporte
        generate_training_report(results, metrics, output_path.with_suffix('.md'))
    
    return results

def generate_training_report(results, metrics, output_path: Path):
    """Genera reporte de entrenamiento en Markdown"""
    
    report = f"""# Reporte de Entrenamiento - Modelo SST

## Información General
- **Fecha:** {Path.ctime(Path.cwd())}
- **Modelo:** YOLOv8 para SST
- **Clases:** {len(SST_CLASSES)} clases de seguridad industrial

## Métricas Finales
- **mAP@0.5:** {metrics.box.map50:.3f}
- **mAP@0.5:0.95:** {metrics.box.map:.3f}
- **Precisión:** {metrics.box.mp:.3f}
- **Recall:** {metrics.box.mr:.3f}

## Clases SST Entrenadas

### Detección de Personas y EPP
- person, person_no_helmet, person_no_vest

### Vehículos y Maquinaria
- forklift, truck, crane, machine, conveyor, press, saw

### Riesgos Locativos
- wet_floor, spill, cable, obstacle, damaged_floor

### Trabajo en Alturas
- ladder, scaffold, harness, edge_unprotected

## Normatividad Cubierta
- Decreto 1072/2015 - Sistema de Gestión SST
- Resolución 2400/1979 - Estatuto de Seguridad Industrial  
- Resolución 4272/2021 - Trabajo en Alturas
- GTC 45 - Identificación de Peligros

## Recomendaciones de Uso
1. Confianza mínima recomendada: 0.35 para detección general
2. Para EPP crítico (cascos, arnés): usar confianza 0.45
3. Validar con imágenes reales del sitio antes de producción
4. Re-entrenar cada 6 meses con nuevos datos

---
*Reporte generado automáticamente por train_sst_optimized.py*
"""
    
    output_path.write_text(report, encoding='utf-8')
    print(f"✓ Reporte guardado: {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Entrenar modelo YOLO optimizado para SST'
    )
    
    parser.add_argument(
        '--data-root',
        type=str,
        required=True,
        help='Ruta raíz del dataset (debe contener images/ y labels/)'
    )
    
    parser.add_argument(
        '--dataset-yaml',
        type=str,
        default='dataset_sst.yaml',
        help='Ruta del archivo YAML del dataset'
    )
    
    parser.add_argument(
        '--model-size',
        choices=['n', 's', 'm', 'l', 'x'],
        default='m',
        help='Tamaño del modelo YOLO (n=nano, x=extra large)'
    )
    
    parser.add_argument(
        '--scenario',
        choices=['speed', 'balanced', 'accuracy'],
        default='balanced',
        help='Escenario de optimización'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=-1,
        help='Número de epochs (sobrescribe el del escenario)'
    )
    
    parser.add_argument(
        '--batch',
        type=int,
        default=-1,
        help='Batch size (sobrescribe el del escenario)'
    )
    
    parser.add_argument(
        '--pretrained-sst',
        type=str,
        default=None,
        help='Ruta a modelo SST pre-entrenado para transfer learning'
    )
    
    parser.add_argument(
        '--recreate-yaml',
        action='store_true',
        help='Recrear archivo YAML del dataset'
    )
    
    args = parser.parse_args()
    
    # Entrenar
    train_sst_model(args)

if __name__ == '__main__':
    main()
from ultralytics import YOLO

# Ruta a tu modelo entrenado
model_path = "runs/detect/train4/weights/best.pt"

# Carga el modelo
model = YOLO(model_path)

# Imprime los nombres de las clases reconocidas por el modelo
print("Clases del modelo:")
for i, name in model.names.items():
    print(f"{i}: {name}")

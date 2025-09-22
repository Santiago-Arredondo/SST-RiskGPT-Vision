# tools/create_scaffold.py
import os

ROOT = r"D:/datasets/sst"  # <- ajusta si quieres otra ruta
SUBS = [
    "images/train", "images/val", "images/test",
    "labels/train", "labels/val", "labels/test"
]

def main():
    for s in SUBS:
        p = os.path.join(ROOT, s)
        os.makedirs(p, exist_ok=True)
        print("ok:", p)
    print("\nEstructura creada. Copia tus imágenes a images/train|val|test y sus .txt a labels/...")

if __name__ == "__main__":
    main()

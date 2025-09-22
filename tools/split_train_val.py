# tools/split_train_val.py
import os, glob, random, shutil

ROOT = r"D:/datasets/sst"
SEED = 42
VAL_PCT = 0.15
TEST_PCT = 0.05

def main():
    random.seed(SEED)
    src_images = glob.glob(os.path.join(ROOT, "images", "all", "*.*"))
    assert src_images, "No hay imágenes en images/all. Copia todo ahí antes de dividir."

    # limpia destinos
    for sp in ["train","val","test"]:
        for sub in ["images","labels"]:
            d = os.path.join(ROOT, sub, sp)
            os.makedirs(d, exist_ok=True)
            for f in glob.glob(os.path.join(d, "*")):
                os.remove(f)

    for ip in src_images:
        base = os.path.splitext(os.path.basename(ip))[0]
        lbl = os.path.join(ROOT, "labels", "all", base + ".txt")
        if not os.path.exists(lbl):
            print("[WARN] sin etiqueta:", base)
            continue
        r = random.random()
        if r < TEST_PCT:
            sp = "test"
        elif r < TEST_PCT + VAL_PCT:
            sp = "val"
        else:
            sp = "train"
        shutil.copy(ip, os.path.join(ROOT, "images", sp, os.path.basename(ip)))
        shutil.copy(lbl, os.path.join(ROOT, "labels", sp, os.path.basename(lbl)))
    print("Split hecho.")

if __name__ == "__main__":
    main()

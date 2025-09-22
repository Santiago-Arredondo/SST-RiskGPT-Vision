# tools/check_labels.py
import os, glob

DATA = r"D:/datasets/sst"   # raíz del dataset
NAMES = [
    "person","forklift","truck","car","excavator","bus","conveyor","machine","saw","press",
    "pallet","cable","spill","toolbox","chair","screen","phone","helmet",
    "ladder","scaffold","guardrail","harness","leading_edge","warning_sign"
]
NUM_CLASSES = len(NAMES)

def check_split(split):
    img_dir = os.path.join(DATA, "images", split)
    lbl_dir = os.path.join(DATA, "labels", split)
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    errs = 0
    for ip in imgs:
        base = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lp):
            print(f"[WARN] Falta label para {base}")
            continue
        with open(lp, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                parts = line.strip().split()
                if len(parts) != 5: 
                    print(f"[ERR] {base}.txt L{ln}: formato inválido")
                    errs += 1; continue
                cls, x, y, w, h = parts
                try:
                    cls = int(cls)
                    x = float(x); y = float(y); w = float(w); h = float(h)
                except:
                    print(f"[ERR] {base}.txt L{ln}: valores no numéricos"); errs += 1; continue
                if not (0 <= cls < NUM_CLASSES):
                    print(f"[ERR] {base}.txt L{ln}: clase fuera de rango ({cls})"); errs += 1
                for v, nm in [(x,"x"),(y,"y"),(w,"w"),(h,"h")]:
                    if not (0.0 < v < 1.0):
                        print(f"[ERR] {base}.txt L{ln}: {nm}={v} fuera de [0..1]"); errs += 1
                if w <= 0 or h <= 0:
                    print(f"[ERR] {base}.txt L{ln}: ancho/alto nulos"); errs += 1
    if errs == 0:
        print(f"[OK] {split}: sin errores")
    else:
        print(f"[FIN] {split}: {errs} errores")

if __name__ == "__main__":
    for sp in ["train","val","test"]:
        check_split(sp)

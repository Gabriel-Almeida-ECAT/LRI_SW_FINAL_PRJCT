import os
import torch
import cv2
import time
import yaml

from ultralytics import YOLO
from pathlib import Path

def createDsFolder():
    dataset_root = Path("../datasets/old_produtos")
    if not (dataset_root.exists() and dataset_root.is_dir()):
        for split in ["train", "valid"]:
            (dataset_root / f"{split}/images").mkdir(parents=True, exist_ok=True)
            (dataset_root / f"{split}/labels").mkdir(parents=True, exist_ok=True)
        print("Estrutura YOLO criada em:", dataset_root.resolve())

def getImgsFromVideo(video_path, label):
    split = "train"  # "valid" para validação
    img_dir = pathlib.Path(f"datasets/old_produtos/{split}/images")
    img_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível abrir a webcam.")

    while True:
        ok, f = cap.read()
        if not ok: break
        cv2.putText(f, "Tecle s para salvar frame; q sai", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("coleta", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'): break
        if k == ord('s'):
            p = img_dir / f"{int(time.time() * 1000)}.jpg"
            cv2.imwrite(str(p), f)
            print("Salvo:", p)

    cap.release()
    cv2.destroyAllWindows()

def createDsYaml():
    yaml_dict = {
        "path": "./datasets/old_produtos",
        "train": "train/images",
        "val": "valid/images",
        "names": ["philips", "sextavado", "pitao"]
    }
    with open("../old_produtos.yaml", "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
    print("Gerado old_produtos.yaml")

def train():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")  # PyTorch 2.x

    model = YOLO("yolov8n.pt")  # já é o menor

    results = model.train(
        data="old_produtos.yaml",
        imgsz=512,  # ↓ de 640 para 512 (ou 448)
        batch=8,  # ↓ de 16 para 8 (tente 4 se ainda der OOM)
        workers=0,  # notebook: evita múltiplos loaders
        device=0 if torch.cuda.is_available() else "cpu",
        epochs=30,
        cache=False,  # True usa RAM; não afeta VRAM
        amp=True,  # mixed precision (padrão True)
        rect=True,  # batches retangulares economizam memória
        optimizer="SGD",  # opcional; memória similar ao AdamW
        close_mosaic=5,  # estável no fim; não muda VRAM do forward
        patience=0  # sem early stop automático
    )
    print("Treino em:", results.save_dir)

def getBestModle():
    best = list((Path("../runs/detect")).glob("train*/weights/best.pt"))[-1]
    print("Usando:", best)
    model = YOLO(str(best))

if __name__ == '--main__':
    train()
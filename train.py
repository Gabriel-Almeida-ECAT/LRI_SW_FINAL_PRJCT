import os
import torch
import cv2
import yaml
import numpy as np
import shutil

from pathlib import Path
from ultralytics import YOLO
from sklearn.model_selection import train_test_split


IMG_SIZE_X = 720
IMG_SIZE_Y = 480

def copy_images_to_split(image_list, src_dir,  target_dir):
    for img_name in image_list:
        src = src_dir / img_name
        dst = target_dir / img_name
        shutil.copy2(src, dst)


def getDatasetFromVideos(videos_list, labels_list, num_photos, tra_val_tes_split=[0.7, 0.15, 0.15]):
    assert len(videos_list) != 0, "Nenhum vídeo fornecido"
    assert len(videos_list) == len(labels_list), "Deve haver um label por vídeo"
    assert abs(sum(tra_val_tes_split) - 1.0) < 1e-6, "As proporções devem somar 1.0"

    base_dir = Path("datasets/final_dataset")
    dir_all_photos = base_dir / "all_photos"
    dir_all_photos.mkdir(parents=True, exist_ok=True)

    dir_train_images = base_dir / "train" / "images"
    dir_train_labels = base_dir / "train" / "labels"
    dir_val_images = base_dir / "val" / "images"
    dir_val_labels = base_dir / "val" / "labels"
    dir_test_images = base_dir / "test" / "images"
    dir_test_labels = base_dir / "test" / "labels"

    for directory in [dir_train_images, dir_train_labels, dir_val_images,
                      dir_val_labels, dir_test_images, dir_test_labels]:
        directory.mkdir(parents=True, exist_ok=True)

    image_label_map = {}

    for video_path, label in zip(videos_list, labels_list):
        video_src = cv2.VideoCapture(video_path)
        if not video_src.isOpened():
            raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

        total_frames = int(video_src.get(cv2.CAP_PROP_FRAME_COUNT))

        if num_photos > total_frames:
            print(f"AVISO: {num_photos} fotos solicitadas, mas apenas {total_frames} frames disponíveis em {video_path}")
            actual_num_photos = total_frames
        else:
            actual_num_photos = num_photos

        photos_ind = np.linspace(0, total_frames+1, actual_num_photos, dtype=int)

        img_ind = 0
        frame_ind = 0

        while img_ind < len(photos_ind):
            if frame_ind == photos_ind[img_ind]:
                has_frame, crtFrame = video_src.read()

                if not has_frame:
                    break

                img_name = f"{Path(video_path).stem}_frame{frame_ind:06d}.jpg"
                img_path = dir_all_photos / img_name

                resized_img = cv2.resize(crtFrame, (IMG_SIZE_X, IMG_SIZE_Y))
                cv2.imwrite(str(img_path), resized_img)

                image_label_map[img_name] = label

                img_ind += 1
            else:
                has_frame = video_src.grab()

                if not has_frame:
                    break

            frame_ind += 1

        video_src.release()
        print(f"Extraídas {img_ind} imagens de {Path(video_path).name} (label: {label})")

        # Separação estratificada em train/val/test
    all_images = list(image_label_map.keys())
    all_labels = [image_label_map[img] for img in all_images]

    if len(all_images) == 0:
        raise RuntimeError("Nenhuma imagem foi extraída dos vídeos")

    # Primeira divisão: train vs (val + test)
    train_images, temp_images, train_labels, temp_labels = train_test_split(
        all_images,
        all_labels,
        test_size=(tra_val_tes_split[1] + tra_val_tes_split[2]),
        stratify=all_labels,
        random_state=42
    )

    # Segunda divisão: val vs test
    val_size_ratio = tra_val_tes_split[1] / (tra_val_tes_split[1] + tra_val_tes_split[2])
    val_images, test_images, val_labels, test_labels = train_test_split(
        temp_images,
        temp_labels,
        test_size=(1 - val_size_ratio),
        stratify=temp_labels,
        random_state=42
    )

    copy_images_to_split(train_images, dir_all_photos, dir_train_images)
    copy_images_to_split(val_images, dir_all_photos, dir_val_images)
    copy_images_to_split(test_images, dir_all_photos, dir_test_images)

    yaml_dict = {
        "path": ".\\" + str(base_dir),
        "train": "train/images",
        "val": "valid/images",
        "names": ["radiador", "deifeito"]
    }
    yaml_name = base_dir / "train_info.yaml"
    with open(str(yaml_name), "w") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)

    return None


def train():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")  # PyTorch 2.x

    model = YOLO("yolov8n.pt")  # já é o menor

    results = model.train(
        data="train_info.yaml",
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

if __name__ == '__main__':
    video_1 = r'datasets/VIDEO.mp4'
    video_2 = r'datasets/VIDEO2.mp4'

    getDatasetFromVideos(
        [video_1, video_2],
        ['SERP', 'DEFEITO'],
        num_photos=100,
        tra_val_tes_split=[0.7, 0.15, 0.15]
    )

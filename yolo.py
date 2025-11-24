import os
import torch
import cv2
import yaml
import numpy as np
import shutil
import sys

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


def train(dataset_path):
    os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision("medium")

    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "train_info.yaml"
    
    if not yaml_path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {yaml_path}")
    
    # Verificar diretórios necessários
    required_dirs = [
        dataset_path / "train" / "images",
        dataset_path / "train" / "labels",
        dataset_path / "val" / "images",
        dataset_path / "val" / "labels"
    ]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            raise FileNotFoundError(f"Diretório necessário não encontrado: {dir_path}")
        
        files = list(dir_path.glob("*"))
        if len(files) == 0:
            print(f"<AVISO> Diretório vazio: {dir_path}")
    
    train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
    train_labels = list((dataset_path / "train" / "labels").glob("*.txt"))
    val_images = list((dataset_path / "val" / "images").glob("*.jpg"))
    val_labels = list((dataset_path / "val" / "labels").glob("*.txt"))
    
    print("\n" + "="*60)
    print("# VERIFICAÇÃO DO DATASET")
    print(f"Treino  - Imagens: {len(train_images)} | Labels: {len(train_labels)}")
    print(f"Validação - Imagens: {len(val_images)} | Labels: {len(val_labels)}")
    
    if len(train_labels) == 0 or len(val_labels) == 0:
        print("\n<AVISO> Não há arquivos de anotação (.txt) nos diretórios 'labels'!")
        return None
    
    model = YOLO("yolo11x.pt")
    
    # Configurações de treinamento otimizadas
    print("\nIniciando treinamento \n")
    results = model.train(
        data=str(yaml_path),
        imgsz=512,              # Reduzido para economizar VRAM
        batch=16,               # Ajuste conforme sua GPU (4 se OOM)
        workers=0,              # Evita problemas em notebooks
        device= 0 if torch.cuda.is_available() else "cpu",
        epochs=30,
        cache=False,            # False para economizar RAM
        amp=True,               # Mixed precision training
        rect=True,              # Batches retangulares (economiza memória)
        optimizer="SGD",        # SGD ou AdamW
        close_mosaic=5,         # Desativa mosaic nos últimos 5 epochs
        patience=5,            # Early stopping após 10 epochs sem melhora
        save=True,              # Salvar checkpoints
        save_period=5,          # Salvar a cada 5 epochs
        project="runs/train",   # Diretório de saída
        name="yolo_custom",     # Nome do experimento
        exist_ok=True,          # Sobrescrever experimentos anteriores
        pretrained=True,        # Usar pesos pré-treinados
        verbose=True,           # Logs detalhados
        hsv_h=0.015,           # Variação de matiz
        hsv_s=0.7,             # Variação de saturação
        hsv_v=0.4,             # Variação de valor
        degrees=0.0,           # Rotação (0 se objetos têm orientação fixa)
        translate=0.1,         # Translação
        scale=0.0,             # Escala
        shear=0.4,             # Cisalhamento
        perspective=0.0,       # Perspectiva
        flipud=0.5,            # Flip vertical
        fliplr=0.5,            # Flip horizontal (50% de chance)
        mosaic=0.0,            # Mosaic augmentation
        mixup=0.0,             # Mixup augmentation
    )
    
    print("\n" + "="*60)
    print("# TREINAMENTO CONCLUÍDO")
    print(f"Resultados salvos em: {results.save_dir}")
    print(f"Melhor modelo: {results.save_dir}/weights/best.pt")
    print(f"Último modelo: {results.save_dir}/weights/last.pt")
    print("="*60)
    
    return results


def test_pytorch_gpu():
    print("=== PyTorch GPU Information ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} capabilities: {torch.cuda.get_device_capability(i)}")
            print(f"GPU {i} total memory: {torch.cuda.get_device_properties(i).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("No CUDA-compatible GPU found")

    return torch.cuda.is_available()



class yoloDefectDetectorClass():
    def __init__(self):
        self.model_path = Path("./runs/train/yolo_custom/weights/best.pt")
        self.model = YOLO(str(self.model_path))
        self.IMG_SIZE_X = 720
        self.IMG_SIZE_Y = 480

    def testModel(self, camera_id=0):
        cap = cv2.VideoCapture(camera_id)

        if not cap.isOpened():
            print(f"Error: Cannot open camera {camera_id}")
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            fps = 30

        frame_interval = 1 * fps
        frame_ind = 0
        cached_boxes = []  # Armazena as boxes da última detecção

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Cannot read frame")
                break

            resized_frame = cv2.resize(frame, (self.IMG_SIZE_X, self.IMG_SIZE_Y))

            # Executar predição apenas no intervalo especificado
            if frame_ind % frame_interval == 0:
                results = self.model(resized_frame)

                # Atualizar cache com as novas detecções
                cached_boxes = []
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = f"{results[0].names[cls]} {conf:.2f}"
                    cached_boxes.append((x1, y1, x2, y2, label))

            # Sempre desenhar as boxes cacheadas no frame atual
            display_frame = resized_frame.copy()
            for x1, y1, x2, y2, label in cached_boxes:
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imshow('YOLO Detection', display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_ind += 1

        cap.release()
        cv2.destroyAllWindows()

    def detectErrInFrame(self, frame):
        return self.model(frame)


if __name__ == '__main__':
    if False:
        video_1 = r'datasets/VIDEO.mp4'
        video_2 = r'datasets/VIDEO2.mp4'

        getDatasetFromVideos(
            [video_1, video_2],
            ['SERP', 'DEFEITO'],
            num_photos=100,
            tra_val_tes_split=[0.7, 0.15, 0.15]
        )

    #train("datasets/final_dataset")

    #test_pytorch_gpu()

    yoloDetector = yoloDefectDetectorClass()
    yoloDetector.testModel(camera_id=1)




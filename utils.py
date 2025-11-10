import os
from pickle import FRAME

import cv2

from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


class logHandlerClass():
    def __init__(self):
        self.logs_path = Path("./logs/")
        self.logs_path.mkdir(parents=True, exist_ok=True)

        self.crt_log_file = self.getCrtLogFile()

    def getCrtLogFile(self):
        logs = os.listdir(str(self.logs_path))
        logs = [file for file in logs if file.endswith('.txt')]

        crt_day = datetime.now().strftime("%Y%m%d")
        last_log_file = sorted(logs)[-1] if len(logs) > 0 else None

        if not logs:
            return self.createLogFile(crt_day)

        elif logs and Path(last_log_file).stem != crt_day:
            return self.createLogFile(crt_day)

        elif logs and Path(last_log_file).stem == crt_day:
            last_log_path = self.logs_path / Path(last_log_file)
            return last_log_path

    def createLogFile(self, crt_day):
        today_log_file = self.logs_path / f"{crt_day}.txt"

        with open(str(today_log_file), 'w') as file:
            pass

        return today_log_file

    def log(self, message):
        self.crt_log_file = self.getCrtLogFile()
        log_msg = f"{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")} {message}\n"
        print(log_msg)

        with open(self.crt_log_file, 'a') as file:
            file.write(log_msg)

class yoloDefectAnalysis():
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


if __name__ == "__main__":
    #logHandler = logHandlerClass()
    #logHandler.log("test 2")

    yoloDetector = yoloDefectAnalysis()
    yoloDetector.testModel(camera_id=1)





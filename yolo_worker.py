import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker
from yolo import yoloDefectDetectorClass
from utils import logHandlerClass

logHandler_worker = logHandlerClass()


class YoloWorkerThread(QThread):
    """
    Thread worker para executar detecção YOLO de forma assíncrona.
    Evita bloqueio do thread principal da GUI.
    """

    # Sinais para comunicação thread-safe com a GUI
    detection_completed = pyqtSignal(bool, dict)  # (tem_erro, resultado_completo)
    detection_error = pyqtSignal(str)  # mensagem de erro

    def __init__(self):
        super().__init__()

        self.yolo_detector = None
        self.frame_queue = []
        self.queue_mutex = QMutex()
        self.is_processing = False
        self.should_stop = False

        logHandler_worker.log("YoloWorkerThread(): Thread worker inicializada")

    def initialize_model(self):
        """Inicializa o modelo YOLO (deve ser chamado após thread iniciar)"""
        try:
            self.yolo_detector = yoloDefectDetectorClass()
            logHandler_worker.log("YoloWorkerThread(): Modelo YOLO carregado com sucesso")
        except Exception as e:
            logHandler_worker.log("YoloWorkerThread(): Erro ao carregar modelo YOLO")
            print(f"YoloWorkerThread(): Erro ao carregar modelo YOLO: {e}")
            self.detection_error.emit(f"YoloWorkerThread(): Erro ao carregar modelo YOLO: {e}")

    def add_frame_to_queue(self, frame):
        """
        Adiciona frame à fila de processamento (thread-safe).
        Mantém apenas o frame mais recente para evitar acúmulo.
        """
        with QMutexLocker(self.queue_mutex):
            # Mantém apenas o último frame (descarta frames antigos)
            self.frame_queue = [frame.copy()]

    def run(self):
        """Loop principal do thread worker"""
        self.initialize_model()

        if self.yolo_detector is None:
            logHandler_worker.log("YoloWorkerThread(): Falha na inicialização, thread encerrada")
            return

        logHandler_worker.log("YoloWorkerThread(): Thread worker em execução")

        while not self.should_stop:
            frame_to_process = None

            # Verifica se há frame na fila (thread-safe)
            with QMutexLocker(self.queue_mutex):
                if len(self.frame_queue) > 0:
                    frame_to_process = self.frame_queue.pop(0)

            if frame_to_process is not None:
                self.is_processing = True
                self.process_frame(frame_to_process)
                self.is_processing = False
            else:
                # Aguarda 100ms antes de verificar novamente
                self.msleep(100)

        logHandler_worker.log("YoloWorkerThread(): Thread worker finalizada")

    def process_frame(self, frame):
        """Processa um frame com detecção YOLO"""
        try:
            logHandler_worker.log("YoloWorkerThread(): Iniciando detecção YOLO")

            # Executa detecção YOLO
            detection_result = self.yolo_detector.detectErrInFrame(frame)

            # Extrai informações relevantes
            detections_dict = detection_result[0].names if len(detection_result) > 0 else {}
            boxes = detection_result[0].boxes if len(detection_result) > 0 else []

            # Verifica se há erro detectado
            has_error = False
            detected_classes = []

            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = detections_dict.get(cls_id, "unknown")
                confidence = float(box.conf[0])
                detected_classes.append({
                    'class': class_name,
                    'confidence': confidence,
                    'box': box.xyxy[0].tolist()
                })

                # Verifica se é um erro (ajuste conforme suas classes)
                if 'ERR' in class_name.upper() or 'DEFEITO' in class_name.upper():
                    has_error = True

            result_data = {
                'has_error': has_error,
                'detections': detected_classes,
                'num_detections': len(detected_classes)
            }

            logHandler_worker.log(
                f"YoloWorkerThread(): Detecção concluída - "
                f"Erro: {has_error}, Detecções: {len(detected_classes)}"
            )

            # Emite sinal com resultado
            self.detection_completed.emit(has_error, result_data)

        except Exception as e:
            error_msg = f"YoloWorkerThread(): Erro durante processamento: {e}"
            logHandler_worker.log(error_msg)
            self.detection_error.emit(error_msg)

    def stop(self):
        """Para o thread worker de forma segura"""
        logHandler_worker.log("YoloWorkerThread(): Solicitação de parada recebida")
        self.should_stop = True
        self.wait(5000)  # Aguarda até 5 segundos para finalizar

    def is_busy(self):
        """Verifica se o worker está processando"""
        return self.is_processing
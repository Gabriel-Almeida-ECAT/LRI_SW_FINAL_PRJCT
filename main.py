import sys
import time

from yolo_worker import YoloWorkerThread
from UI import IndustrialUI
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer
from plc import plcInterfaceClass
from utils import logHandlerClass
from enum import Enum


logHandler = logHandlerClass()
plcInterface = plcInterfaceClass()


class stateClass(Enum):
    INIT = 0
    WAITING_PART = 1
    PART_ANALYSIS = 2
    UPDATE_METRICS = 3
    DEFECT_PRCSS = 4


class RADIATOR_CHECK_SM:
    def __init__(self, ui_window):
        self.STATE = stateClass.INIT
        self.ui = ui_window

        # Timer para executar a state machine periodicamente
        self.sm_timer = QTimer()
        self.sm_timer.timeout.connect(self.run)
        self.sm_timer.start(100)  # Executa a cada 100ms

        # Inicializa worker thread para YOLO
        self.yolo_worker = YoloWorkerThread()
        self.yolo_worker.detection_completed.connect(self.on_detection_completed)
        self.yolo_worker.detection_error.connect(self.on_detection_error)
        self.yolo_worker.start()

        logHandler.log("RADIATOR_CHECK_SM(): Sistema inicializado")
        self.ui.set_sistema_rodando(True)
        self.frame_ind = 0

        # Controle de detecção
        self.detection_in_progress = False
        self.last_detection_result = None


    def run(self):
        match self.STATE:
            case stateClass.INIT:
                # Inicialização completa, vai para espera
                self.STATE = stateClass.WAITING_PART
                logHandler.log("RADIATOR_CHECK_SM(): Sistema pronto, aguardando peça")

                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.WAITING_PART:
                # Espera sinal do sensor de presença
                if plcInterface.readMemAddrs("M202"):
                    logHandler.log("RADIATOR_CHECK_SM(): serpentina detectada")

                    self.STATE = stateClass.PART_ANALYSIS

                return

            case stateClass.PART_ANALYSIS:
                # Pega o frame atual da UI
                frame = self.ui.get_current_frame()

                if frame is not None:
                    # Verifica se deve executar detecção (a cada 5 segundos)
                    if self.frame_ind % (5*30) == 0:
                        # Verifica se não há detecção em andamento
                        if not self.detection_in_progress and not self.yolo_worker.is_busy():
                            logHandler.log("RADIATOR_CHECK_SM(): Solicitando detecção YOLO assíncrona")

                            # Envia frame para processamento assíncrono
                            self.yolo_worker.add_frame_to_queue(frame)
                            self.detection_in_progress = True
                        else:
                            logHandler.log("RADIATOR_CHECK_SM(): Detecção anterior ainda em andamento")

                    self.frame_ind += 1

                # Verifica se há resultado de detecção disponível
                if self.last_detection_result is not None:
                    if self.last_detection_result['has_error']:
                        logHandler.log("RADIATOR_CHECK_SM(): Defeito detectado!")
                        self.STATE = stateClass.DEFECT_PRCSS
                        self.last_detection_result = None  # Limpa resultado
                        return

                    else:
                        plcInterface.setMbit("M204", 1)
                        time.sleep(0.01)
                        plcInterface.setMbit("M204", 0)

                self.STATE = stateClass.UPDATE_METRICS
                return

            case stateClass.UPDATE_METRICS:
                # Atualiza métricas na UI
                self.ui.increment_serpentinas()

                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.DEFECT_PRCSS:
                # Processa defeito
                self.ui.increment_erros()
                self.ui.set_erro_presente(True)

                plcInterface.setMbit("M203", 1)

                if plcInterface.readMemAddrs("M203") == 0:
                    self.STATE = stateClass.WAITING_PART
                return

    def on_detection_completed(self, has_error, result_data):
        """Callback quando detecção YOLO é concluída (thread-safe via signal)"""
        logHandler.log(
            f"RADIATOR_CHECK_SM(): Detecção concluída - "
            f"Erro: {has_error}, Detecções: {result_data['num_detections']}"
        )

        self.detection_in_progress = False
        self.last_detection_result = result_data

        # Log detalhado das detecções
        for detection in result_data['detections']:
            logHandler.log(
                f"  - Classe: {detection['class']}, "
                f"Confiança: {detection['confidence']:.2f}"
            )

    def on_detection_error(self, error_message):
        """Callback quando ocorre erro na detecção YOLO"""
        logHandler.log(f"RADIATOR_CHECK_SM(): Erro na detecção: {error_message}")
        self.detection_in_progress = False

    def on_frame_captured(self, frame):
        """Callback quando um novo frame é capturado"""
        # Você pode processar o frame aqui se necessário
        pass

    def cleanup(self):
        """Limpa recursos ao encerrar"""
        logHandler.log("RADIATOR_CHECK_SM(): Encerrando worker thread")
        self.yolo_worker.stop()


def main():
    app = QApplication(sys.argv)
    window = IndustrialUI()

    state_machine = RADIATOR_CHECK_SM(window)

    window.frame_captured.connect(state_machine.on_frame_captured)
    app.aboutToQuit.connect(state_machine.cleanup)
    window.show()

    sys.exit(app.exec_())

    del plcInterface

    #yoloDetector.testModel(camera_id=0)


if __name__ == '__main__':
    main()
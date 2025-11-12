import os
import sys
import datetime
import cv2

from ultralytics import YOLO
from yolo import yoloDefectDetectorClass
from pathlib import Path

from UI import IndustrialUI
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer

from utils import logHandlerClass
from enum import Enum


yoloDetector = yoloDefectDetectorClass()
logHandler = logHandlerClass()


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

        logHandler.log("RADIATOR_CHECK_SM(): Sistema inicializado")
        self.ui.set_sistema_rodando(True)
        self.frame_ind = 0

    def run(self):
        """Executa um ciclo da state machine"""
        match self.STATE:
            case stateClass.INIT:
                # Inicialização completa, vai para espera
                self.STATE = stateClass.WAITING_PART
                logHandler.log("RADIATOR_CHECK_SM(): Sistema pronto, aguardando peça")

                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.WAITING_PART:
                # Espera sinal do sensor de presença
                # Quando detectar, desliga esteira e vai para análise

                if True: #sensor serpentina
                    logHandler.log("RADIATOR_CHECK_SM(): serpentina detectada")

                self.STATE = stateClass.PART_ANALYSIS
                return

            case stateClass.PART_ANALYSIS:
                # Pega o frame atual da UI
                frame = self.ui.get_current_frame()

                if frame is not None:
                    if self.frame_ind % (5*30) == 0:
                        detection_result = yoloDetector.detectErrInFrame(frame)
                        detections_dict = detection_result.names

                        if 'ERR' in list(detections_dict.values()):
                            self.STATE = stateClass.DEFECT_PRCSS
                            return

                    self.frame_ind += 1

                self.STATE = stateClass.UPDATE_METRICS
                return

            case stateClass.UPDATE_METRICS:
                # Atualiza métricas na UI
                # self.ui.increment_serpentinas()

                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.DEFECT_PRCSS:
                # Processa defeito
                # self.ui.increment_erros()
                # self.ui.set_erro_presente(True)

                logHandler.log("RADIATOR_CHECK_SM(): serpentina detectada")

                self.STATE = stateClass.WAITING_PART
                return

    def on_frame_captured(self, frame):
        """Callback quando um novo frame é capturado"""
        # Você pode processar o frame aqui se necessário
        pass


def main():
    if True:
        app = QApplication(sys.argv)
        window = IndustrialUI()

        state_machine = RADIATOR_CHECK_SM(window)

        window.frame_captured.connect(state_machine.on_frame_captured)
        window.show()

        sys.exit(app.exec_())

    # yoloDetector.testModel(camera_id=0)

    return


if __name__ == '__main__':
    main()
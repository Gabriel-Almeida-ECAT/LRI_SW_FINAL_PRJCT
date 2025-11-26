import cv2
import sys

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
    def __init__(self, ui_window: IndustrialUI):
        self.STATE = stateClass.INIT
        self.ui = ui_window

        # Timer para executar a state machine periodicamente
        self.sm_timer = QTimer()
        self.sm_timer.timeout.connect(self.run)
        self.sm_timer.start(100)  # Executa a cada 100 ms

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
                logHandler.log("RADIATOR_CHECK_SM(): Sistema pronto, aguardando peça")
                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.WAITING_PART:
                # Leitura do sensor de presença (M202)
                m202 = plcInterface.readMemAddrs("M202")
                if m202 is None:
                    # Falha de comunicação; mantém estado de espera
                    logHandler.log(
                        "RADIATOR_CHECK_SM(): Falha ao ler M202 (sensor de serpentina)"
                    )
                    self.ui.set_sensor_ativo(False)
                    return

                self.ui.set_sensor_ativo(bool(m202))

                # Se sensor ativo, inicia análise da peça
                if m202 == 1:
                    logHandler.log("RADIATOR_CHECK_SM(): Serpentina detectada (M202=1)")
                    self.STATE = stateClass.PART_ANALYSIS

                return

            case stateClass.PART_ANALYSIS:
                # Pega o frame atual da UI
                frame = self.ui.get_current_frame()

                if frame is not None:
                    # Verifica se deve executar detecção (a cada ~5 s; 30 fps)
                    if self.frame_ind % (5 * 30) == 0:
                        # Verifica se não há detecção em andamento
                        if not self.detection_in_progress and not self.yolo_worker.is_busy():
                            logHandler.log(
                                "RADIATOR_CHECK_SM(): Solicitando detecção YOLO assíncrona"
                            )
                            # Envia frame para processamento assíncrono
                            self.yolo_worker.add_frame_to_queue(frame)
                            self.detection_in_progress = True
                        else:
                            logHandler.log(
                                "RADIATOR_CHECK_SM(): Detecção anterior ainda em andamento"
                            )

                    self.frame_ind += 1

                # Verifica se há resultado de detecção disponível
                if self.last_detection_result is not None:
                    if self.last_detection_result["has_error"]:
                        logHandler.log("RADIATOR_CHECK_SM(): Defeito detectado!")
                        self.STATE = stateClass.DEFECT_PRCSS
                        self.last_detection_result = None  # Limpa resultado
                        return
                    else:
                        # Sem defeito: gera pulso em M204 (por exemplo, avanço de esteira)
                        if plcInterface.setMbit("M204", 1):
                            # Usa QTimer.singleShot para não bloquear a GUI
                            QTimer.singleShot(
                                10,  # 10 ms
                                lambda: plcInterface.setMbit("M204", 0)
                            )
                        self.last_detection_result = None
                        self.STATE = stateClass.UPDATE_METRICS
                        return

                # Se ainda não há resultado, permanece em PART_ANALYSIS
                return

            case stateClass.UPDATE_METRICS:
                # Atualiza métricas na UI (peça boa)
                self.ui.increment_serpentinas()
                self.ui.set_erro_presente(False)
                logHandler.log("RADIATOR_CHECK_SM(): Métricas atualizadas (peça OK)")
                self.STATE = stateClass.WAITING_PART
                return

            case stateClass.DEFECT_PRCSS:
                # Processa defeito
                self.ui.increment_erros()
                self.ui.set_erro_presente(True)

                # Sinaliza defeito para o PLC (M203 = 1)
                if not plcInterface.setMbit("M203", 1):
                    logHandler.log(
                        "RADIATOR_CHECK_SM(): Falha ao escrever M203 (sinal de defeito)"
                    )
                    return

                # Aguarda PLC limpar M203 para voltar ao ciclo normal
                m203 = plcInterface.readMemAddrs("M203")
                if m203 is None:
                    logHandler.log(
                        "RADIATOR_CHECK_SM(): Falha ao ler M203 durante tratamento de defeito"
                    )
                    return

                if m203 == 0:
                    logHandler.log(
                        "RADIATOR_CHECK_SM(): PLC liberou M203, retornando para WAITING_PART"
                    )
                    self.ui.set_erro_presente(False)
                    self.STATE = stateClass.WAITING_PART

                return

    def on_detection_completed(self, has_error: bool, result_data: dict):
        """Callback quando detecção YOLO é concluída (thread-safe via signal)."""
        logHandler.log(
            f"RADIATOR_CHECK_SM(): Detecção concluída - "
            f"Erro: {has_error}, Detecções: {result_data['num_detections']}"
        )

        self.detection_in_progress = False
        self.last_detection_result = result_data

        # Log detalhado das detecções
        for detection in result_data["detections"]:
            logHandler.log(
                f"  - Classe: {detection['class']}, "
                f"Confiança: {detection['confidence']:.2f}"
            )

    def on_detection_error(self, error_message: str):
        """Callback quando ocorre erro na detecção YOLO."""
        logHandler.log(f"RADIATOR_CHECK_SM(): Erro na detecção: {error_message}")
        self.detection_in_progress = False

    def on_frame_captured(self, frame):
        """Callback quando um novo frame é capturado pela UI."""
        # Caso deseje realizar algum pré-processamento do frame
        pass

    def cleanup(self):
        """Limpa recursos ao encerrar."""
        logHandler.log("RADIATOR_CHECK_SM(): Encerrando worker thread")
        self.yolo_worker.stop()
        # Fecha conexão com PLC de forma explícita
        plcInterface.close()
        logHandler.log("RADIATOR_CHECK_SM(): Conexão com PLC encerrada")


def main():
    app = QApplication(sys.argv)
    window = IndustrialUI()

    state_machine = RADIATOR_CHECK_SM(window)

    window.frame_captured.connect(state_machine.on_frame_captured)
    app.aboutToQuit.connect(state_machine.cleanup)
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
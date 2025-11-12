import sys
import cv2
import numpy as np

from utils import logHandlerClass
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QSlider, QComboBox, QPushButton,
                             QGroupBox, QGridLayout, QFrame, QScrollArea, QCheckBox,
                             QSpinBox)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QMutex
from PyQt5.QtGui import QImage, QPixmap, QFont

logHandler_UI = logHandlerClass()

class IndustrialUI(QMainWindow):
    """Interface gráfica para sistema de detecção de serpentinas"""

    # Sinais para comunicação com state machine
    camera_changed = pyqtSignal(int)
    pdi_params_changed = pyqtSignal(dict)
    frame_captured = pyqtSignal(object)  # Emite o frame capturado

    def __init__(self):
        super().__init__()

        self.camera = None
        self.camera_index = 0
        self.current_frame = None

        self.params_mutex = QMutex()

        # Parâmetros PDI
        self.pdi_params = {
            'auto_focus': True,
            'focus': 0,
            'auto_exposure': True,
            'exposure': -6,
            'gain': 0,
            'brightness': 128,
            'contrast': 40,
            'gamma': 100,
            'sharpness': 200
        }

        # Dados de status
        self.total_serpentinas = 0
        self.total_erros = 0
        self.sistema_rodando = False
        self.sensor_ativo = False
        self.erro_presente = False

        self.init_ui()
        self.setup_timer()
        self.detect_cameras()

    def init_ui(self):
        self.setWindowTitle("Sistema de Detecção de Serpentinas - Robótica Industrial")
        self.showMaximized()
        self.setStyleSheet(self.get_stylesheet())

        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Layout principal (horizontal)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Painel esquerdo (menus)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)

        # Painel direito (imagem e indicadores)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)

    def detect_cameras(self):
        """Detecta câmeras disponíveis no sistema"""
        available_cameras = []

        # Testa até 10 índices de câmera
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)

            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        # Atualiza ComboBox apenas com câmeras disponíveis
        self.camera_combo.clear()
        if available_cameras:
            for idx in available_cameras:
                self.camera_combo.addItem(f"Câmera {idx}", idx)
            logHandler_UI.log(f"IndustrialUI(): Câmeras detectadas: {available_cameras}")

            # Abre a primeira câmera automaticamente
            if len(available_cameras) > 0:
                self.set_camera(available_cameras[0])
        else:
            self.camera_combo.addItem("Nenhuma câmera detectada", -1)
            logHandler_UI.log("IndustrialUI(): Nenhuma câmera detectada")

    def set_camera(self, index: int) -> bool:
        """Configura e abre a câmera com otimizações para Pcyes FHD-03"""
        self.camera_index = index
        if self.camera is not None:
            self.camera.release()

        self.camera = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.camera.isOpened():
            self.camera = cv2.VideoCapture(index)
            if not self.camera.isOpened():
                return False

        # Configuração específica para FHD-03
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.camera.set(cv2.CAP_PROP_FPS, 30)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        import time
        time.sleep(0.5)

        self.apply_pdi_params()

        logHandler_UI.log(f"IndustrialUI(): Câmera configurada: {int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ {int(self.camera.get(cv2.CAP_PROP_FPS))}fps")

        return True

    def apply_pdi_params(self):
        """Aplica parâmetros de PDI na câmera"""
        if self.camera is None or not self.camera.isOpened():
            return

        self.params_mutex.lock()
        params = self.pdi_params.copy()
        self.params_mutex.unlock()

        try:
            # AUTOFOCO
            if params.get("auto_focus", True):
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                logHandler_UI.log(f"IndustrialUI(): Autofoco ATIVADO (modo contínuo)")
            else:
                self.camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                focus_value = params.get("focus", 0)
                self.camera.set(cv2.CAP_PROP_FOCUS, focus_value)
                logHandler_UI.log(f"IndustrialUI(): Foco MANUAL: {focus_value}")

            # AUTO EXPOSIÇÃO
            if params.get("auto_exposure", True):
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                logHandler_UI.log(f"IndustrialUI(): Auto exposição ATIVADA")
            else:
                self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
                exposure_value = params.get("exposure", -6)
                self.camera.set(cv2.CAP_PROP_EXPOSURE, exposure_value)
                logHandler_UI.log(f"IndustrialUI(): Exposição MANUAL: {exposure_value}")

            # OUTROS PARÂMETROS
            self.camera.set(cv2.CAP_PROP_GAIN, params.get("gain", 0))
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, params.get("brightness", 128))
            self.camera.set(cv2.CAP_PROP_CONTRAST, params.get("contrast", 40))
            self.camera.set(cv2.CAP_PROP_GAMMA, params.get("gamma", 100))
            self.camera.set(cv2.CAP_PROP_SHARPNESS, params.get("sharpness", 200))

        except Exception as e:
            logHandler_UI.log(f"IndustrialUI(): Aviso ao aplicar parâmetros: {e}")

    def update_pdi_param(self, param: str, value):
        """Atualiza um parâmetro específico de PDI (thread-safe)"""
        self.params_mutex.lock()
        self.pdi_params[param] = value
        self.params_mutex.unlock()
        self.apply_pdi_params()

    def get_current_frame(self):
        """Retorna o frame atual capturado (para uso pela SM)"""
        return self.current_frame

    def create_left_panel(self):
        """Cria o painel esquerdo com menus"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        camera_group = self.create_camera_menu()
        layout.addWidget(camera_group)

        pdi_group = self.create_pdi_menu()
        layout.addWidget(pdi_group)

        info_group = self.create_info_menu()
        layout.addWidget(info_group)

        layout.addStretch()

        return panel

    def create_camera_menu(self):
        """Cria o menu de seleção de câmera"""
        group = QGroupBox("Câmera")
        group.setFont(QFont("Arial", 11, QFont.Bold))
        layout = QVBoxLayout()

        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)

        layout.addWidget(QLabel("Selecione a câmera:"))
        layout.addWidget(self.camera_combo)

        group.setLayout(layout)
        return group

    def create_pdi_menu(self):
        """Cria o menu de edição de imagem com parâmetros PDI (scrollable)"""
        group = QGroupBox("Paramêtros PDI")
        group.setFont(QFont("Arial", 11, QFont.Bold))
        group.setMaximumHeight(400)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        scroll_widget = QWidget()
        layout = QVBoxLayout(scroll_widget)
        layout.setSpacing(8)

        # AUTOFOCO
        autofocus_frame = QFrame()
        autofocus_frame.setFrameStyle(QFrame.StyledPanel)
        autofocus_layout = QVBoxLayout(autofocus_frame)

        self.autofocus_check = QCheckBox("Auto Foco")
        self.autofocus_check.setChecked(self.pdi_params['auto_focus'])
        self.autofocus_check.stateChanged.connect(self.on_autofocus_changed)
        autofocus_layout.addWidget(self.autofocus_check)

        focus_layout = QHBoxLayout()
        self.focus_label = QLabel(f"Foco Manual: {self.pdi_params['focus']}")
        self.focus_slider = QSlider(Qt.Horizontal)
        self.focus_slider.setMinimum(0)
        self.focus_slider.setMaximum(255)
        self.focus_slider.setValue(self.pdi_params['focus'])
        self.focus_slider.setEnabled(not self.pdi_params['auto_focus'])
        self.focus_slider.valueChanged.connect(self.on_focus_changed)
        focus_layout.addWidget(self.focus_label)
        focus_layout.addWidget(self.focus_slider)
        autofocus_layout.addLayout(focus_layout)

        layout.addWidget(autofocus_frame)

        # AUTO EXPOSIÇÃO
        autoexp_frame = QFrame()
        autoexp_frame.setFrameStyle(QFrame.StyledPanel)
        autoexp_layout = QVBoxLayout(autoexp_frame)

        self.autoexp_check = QCheckBox("Auto Exposição")
        self.autoexp_check.setChecked(self.pdi_params['auto_exposure'])
        self.autoexp_check.stateChanged.connect(self.on_autoexp_changed)
        autoexp_layout.addWidget(self.autoexp_check)

        exp_layout = QHBoxLayout()
        self.exp_label = QLabel(f"Exposição: {self.pdi_params['exposure']}")
        self.exp_slider = QSlider(Qt.Horizontal)
        self.exp_slider.setMinimum(-13)
        self.exp_slider.setMaximum(0)
        self.exp_slider.setValue(self.pdi_params['exposure'])
        self.exp_slider.setEnabled(not self.pdi_params['auto_exposure'])
        self.exp_slider.valueChanged.connect(self.on_exposure_changed)
        exp_layout.addWidget(self.exp_label)
        exp_layout.addWidget(self.exp_slider)
        autoexp_layout.addLayout(exp_layout)

        layout.addWidget(autoexp_frame)

        # OUTROS PARÂMETROS
        layout.addWidget(self.create_param_slider("Ganho", "gain", 0, 100, self.pdi_params['gain']))
        layout.addWidget(self.create_param_slider("Brilho", "brightness", 0, 255, self.pdi_params['brightness']))
        layout.addWidget(self.create_param_slider("Contraste", "contrast", 0, 100, self.pdi_params['contrast']))
        layout.addWidget(self.create_param_slider("Gamma", "gamma", 0, 200, self.pdi_params['gamma']))
        layout.addWidget(self.create_param_slider("Nitidez", "sharpness", 0, 255, self.pdi_params['sharpness']))

        reset_btn = QPushButton("Resetar Valores Padrão")
        reset_btn.clicked.connect(self.reset_pdi_values)
        layout.addWidget(reset_btn)

        scroll.setWidget(scroll_widget)

        main_layout = QVBoxLayout()
        main_layout.addWidget(scroll)
        group.setLayout(main_layout)

        return group

    def create_param_slider(self, label, param_name, min_val, max_val, init_val):
        """Cria um slider para um parâmetro PDI"""
        frame = QFrame()
        frame.setFrameStyle(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)

        title = QLabel(label)
        title.setFont(QFont("Arial", 10, QFont.Bold))
        layout.addWidget(title)

        slider_layout = QHBoxLayout()
        value_label = QLabel(f"{init_val}")
        value_label.setMinimumWidth(40)

        slider = QSlider(Qt.Horizontal)
        slider.setMinimum(min_val)
        slider.setMaximum(max_val)
        slider.setValue(init_val)
        slider.valueChanged.connect(
            lambda v: self.on_param_changed(param_name, v, value_label)
        )

        slider_layout.addWidget(value_label)
        slider_layout.addWidget(slider)
        layout.addLayout(slider_layout)

        return frame

    def create_info_menu(self):
        """Cria o menu de informações"""
        group = QGroupBox("Informações")
        group.setFont(QFont("Arial", 11, QFont.Bold))
        layout = QVBoxLayout()

        self.serpentinas_label = QLabel(f"Contagem total de serpentinas: {self.total_serpentinas}")
        self.serpentinas_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.serpentinas_label)

        self.erros_label = QLabel(f"Número erros: {self.total_erros}")
        self.erros_label.setFont(QFont("Arial", 10))
        layout.addWidget(self.erros_label)

        reset_btn = QPushButton("Resetar Contadores")
        reset_btn.clicked.connect(self.reset_counters)
        layout.addWidget(reset_btn)

        group.setLayout(layout)
        return group

    def create_right_panel(self):
        """Cria o painel direito com indicadores e imagem"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        indicators_group = self.create_indicators()
        layout.addWidget(indicators_group)

        self.image_label = QLabel("Aguardando Imagem da Webcam...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(1280, 720)
        self.image_label.setScaledContents(False)

        self.image_label.setStyleSheet("""
            QLabel {
                background-color: #2b2b2b;
                color: #ffffff;
                font-size: 18px;
                border: 2px solid #555555;
                border-radius: 5px;
            }
        """)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        return panel

    def create_indicators(self):
        """Cria os indicadores de status"""
        group = QGroupBox("Indicadores")
        group.setFont(QFont("Arial", 11, QFont.Bold))

        layout = QHBoxLayout()
        layout.setSpacing(30)
        layout.setContentsMargins(15, 15, 15, 15)

        # Sistema rodando
        sistema_layout = QVBoxLayout()
        sistema_layout.setAlignment(Qt.AlignCenter)
        sistema_label = QLabel("Sistema rodando")
        sistema_label.setFont(QFont("Arial", 30))
        sistema_label.setAlignment(Qt.AlignCenter)
        self.sistema_indicator = QLabel("●")
        self.sistema_indicator.setFont(QFont("Arial", 30))
        self.sistema_indicator.setAlignment(Qt.AlignCenter)
        sistema_layout.addWidget(sistema_label)
        sistema_layout.addWidget(self.sistema_indicator)
        layout.addLayout(sistema_layout)

        # Sensor de serpentina
        sensor_layout = QVBoxLayout()
        sensor_layout.setAlignment(Qt.AlignCenter)
        sensor_label = QLabel("Sensor de serpentina")
        sensor_label.setFont(QFont("Arial", 30))
        sensor_label.setAlignment(Qt.AlignCenter)
        self.sensor_indicator = QLabel("●")
        self.sensor_indicator.setFont(QFont("Arial", 30))
        self.sensor_indicator.setAlignment(Qt.AlignCenter)
        sensor_layout.addWidget(sensor_label)
        sensor_layout.addWidget(self.sensor_indicator)
        layout.addLayout(sensor_layout)

        # Erro presente
        erro_layout = QVBoxLayout()
        erro_layout.setAlignment(Qt.AlignCenter)
        erro_label = QLabel("Erro presente")
        erro_label.setFont(QFont("Arial", 30))
        erro_label.setAlignment(Qt.AlignCenter)
        self.erro_indicator = QLabel("●")
        self.erro_indicator.setFont(QFont("Arial", 30))
        self.erro_indicator.setAlignment(Qt.AlignCenter)
        erro_layout.addWidget(erro_label)
        erro_layout.addWidget(self.erro_indicator)
        layout.addLayout(erro_layout)

        layout.addStretch()

        group.setLayout(layout)
        group.setMinimumHeight(90)

        self.update_indicators()

        return group

    def setup_timer(self):
        """Configura o timer para atualização da imagem"""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(33)  # ~30 FPS

    def update_frame(self):
        """Atualiza o frame da câmera na interface"""
        if self.camera and self.camera.isOpened():
            ret, frame = self.camera.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_frame(frame)
                # Emite sinal com o frame para a SM
                self.frame_captured.emit(self.current_frame)

    def display_frame(self, frame):
        """Exibe o frame na interface mantendo proporção 3:2"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w

        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        scaled_pixmap = pixmap.scaled(self.image_label.size(),
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def on_camera_changed(self, index):
        """Callback quando a câmera é alterada"""
        camera_index = self.camera_combo.itemData(index)

        if camera_index == -1:
            logHandler_UI.log(f"IndustrialUI(): Nenhuma câmera disponível")
            return

        success = self.set_camera(camera_index)
        if success:
            logHandler_UI.log(f"IndustrialUI(): Câmera {camera_index} selecionada")
        else:
            logHandler_UI.log(f"IndustrialUI(): Erro ao selecionar câmera {camera_index}")

        self.camera_changed.emit(camera_index)

    def on_autofocus_changed(self, state):
        """Callback para mudança no autofoco"""
        auto_focus = (state == Qt.Checked)
        self.update_pdi_param('auto_focus', auto_focus)
        self.focus_slider.setEnabled(not auto_focus)

    def on_focus_changed(self, value):
        """Callback para mudança no foco manual"""
        self.focus_label.setText(f"Foco Manual: {value}")
        self.update_pdi_param('focus', value)

    def on_autoexp_changed(self, state):
        """Callback para mudança na auto exposição"""
        auto_exposure = (state == Qt.Checked)
        self.update_pdi_param('auto_exposure', auto_exposure)
        self.exp_slider.setEnabled(not auto_exposure)

    def on_exposure_changed(self, value):
        """Callback para mudança na exposição manual"""
        self.exp_label.setText(f"Exposição: {value}")
        self.update_pdi_param('exposure', value)

    def on_param_changed(self, param, value, label):
        """Callback genérico para mudança de parâmetro"""
        label.setText(f"{value}")
        self.update_pdi_param(param, value)

    def reset_pdi_values(self):
        """Reseta os valores PDI para padrão"""
        self.pdi_params = {
            'auto_focus': True,
            'focus': 0,
            'auto_exposure': True,
            'exposure': -6,
            'gain': 0,
            'brightness': 128,
            'contrast': 40,
            'gamma': 100,
            'sharpness': 200
        }
        self.apply_pdi_params()
        logHandler_UI.log(f"IndustrialUI(): Valores PDI resetados para padrão")

    def reset_counters(self):
        """Reseta os contadores de serpentinas e erros"""
        self.total_serpentinas = 0
        self.total_erros = 0
        self.update_info_display()

    def update_info_display(self):
        """Atualiza a exibição das informações"""
        self.serpentinas_label.setText(f"Contagem total de serpentinas: {self.total_serpentinas}")
        self.erros_label.setText(f"Número de erros: {self.total_erros}")

    def update_indicators(self):
        """Atualiza os indicadores visuais"""
        self.sistema_indicator.setStyleSheet(
            f"color: {'#00ff00' if self.sistema_rodando else '#ff0000'};"
        )
        self.sensor_indicator.setStyleSheet(
            f"color: {'#00ff00' if self.sensor_ativo else '#888888'};"
        )
        self.erro_indicator.setStyleSheet(
            f"color: {'#ff0000' if self.erro_presente else '#00ff00'};"
        )

    def set_sistema_rodando(self, status: bool):
        """Define o status do sistema"""
        self.sistema_rodando = status
        self.update_indicators()

    def set_sensor_ativo(self, status: bool):
        """Define o status do sensor"""
        self.sensor_ativo = status
        self.update_indicators()

    def set_erro_presente(self, status: bool):
        """Define o status de erro"""
        self.erro_presente = status
        self.update_indicators()

    def increment_serpentinas(self):
        """Incrementa o contador de serpentinas"""
        self.total_serpentinas += 1
        self.update_info_display()

    def increment_erros(self):
        """Incrementa o contador de erros"""
        self.total_erros += 1
        self.update_info_display()

    def closeEvent(self, event):
        """Libera a câmera ao fechar"""
        if self.camera:
            self.camera.release()
        event.accept()

    def get_stylesheet(self):
        """Retorna o stylesheet da aplicação"""
        return """
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #ffffff;
            }
            QGroupBox {
                border: 2px solid #555555;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #ffffff;
                font-size: 11px;
            }
            QComboBox {
                background-color: #2b2b2b;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 5px;
                color: #ffffff;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 5px solid #ffffff;
            }
            QSlider::groove:horizontal {
                border: 1px solid #555555;
                height: 8px;
                background: #2b2b2b;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #0078d7;
                border: 1px solid #555555;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QPushButton {
                background-color: #0078d7;
                color: #ffffff;
                border: none;
                padding: 8px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cc1;
            }
            QFrame {
                background-color: #252525;
                border-radius: 5px;
                padding: 5px;
            }
            QCheckBox {
                color: #ffffff;
                font-size: 11px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: #0078d7;
                border-radius: 6px;
            }
        """


def main():
    """Função principal para testar a UI"""
    app = QApplication(sys.argv)

    # Cria a interface (sem camera_controller para teste)
    window = IndustrialUI()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
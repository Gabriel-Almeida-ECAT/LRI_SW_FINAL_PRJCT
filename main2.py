import time
from datetime import datetime
from enum import Enum

import cv2
import pymcprotocol
from yolo import yoloDefectDetectorClass  # keep your existing YOLO class

# PLC Connection Settings
PLC_IP = "192.168.3.39"  # IP do PLC
PLC_PORT = 5010          # Porta padrão do MC Protocol (SLMP)
CAMERA_ID = 1

'''
M200 - Liga a esteira
M201 - Desliga a esteira
M202 - Sensor presenca
M203 - falha detectada
M204 - Parte ok
'''

# --------------------------------------------------------------------
# Simple logging
# --------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"{datetime.now():%Y-%m-%d_%Hh:%Mm:%Ss} {msg}")


# --------------------------------------------------------------------
# PLC interface (simplified)
# --------------------------------------------------------------------
class PLC:
    def __init__(self):
        self.client = pymcprotocol.Type3E()
        self.client.connect(PLC_IP, PLC_PORT)
        log(f"PLC connected to {PLC_IP}:{PLC_PORT}")

    def read_bit(self, addr: str):
        """Reads a single M-bit (e.g., 'M202'). Returns 0, 1 or None on error."""
        try:
            result = self.client.batchread_bitunits(addr, 1)
            if not result:
                log(f"PLC read empty result from {addr}")
                return None
            val = int(result[0])
            log(f"PLC read {addr} -> {val}")
            return val
        except Exception as e:
            log(f"PLC read error {addr}: {e}")
            return None

    def write_bit(self, addr: str, value) -> bool:
        """Writes a single M-bit (e.g., 'M203', 'M204'). Returns True/False."""
        try:
            v = int(bool(value))
            self.client.batchwrite_bitunits(addr, [v])
            log(f"PLC write {addr} = {v}")
            return True
        except Exception as e:
            log(f"PLC write error {addr}: {e}")
            return False

    def close(self):
        try:
            self.client.close()
            log("PLC connection closed")
        except Exception:
            pass


# --------------------------------------------------------------------
# YOLO detector wrapper (synchronous)
# --------------------------------------------------------------------
class YoloDetector:
    def __init__(self):
        log("Loading YOLO model...")
        self.model = yoloDefectDetectorClass()
        log("YOLO model loaded")

    def detect(self, frame):
        try:
            detection_result = self.model.detectErrInFrame(frame)
        except Exception as e:
            log(f"YOLO detection error: {e}")
            return {"has_error": False, "detections": [], "num_detections": 0}

        if not detection_result:
            return {"has_error": False, "detections": [], "num_detections": 0}

        det = detection_result[0]
        names = det.names  # dict: class_id -> class_name
        boxes = det.boxes  # iterable of boxes

        detections = []
        has_error = False

        for box in boxes:
            cls_id = int(box.cls[0])
            class_name = names.get(cls_id, "unknown")
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "confidence": conf,
                "box": xyxy,
            })

            # error if class name contains ERR or DEFEITO
            up = class_name.upper()
            if "ERR" in up or "DEFEITO" in up:
                has_error = True

        log(f"YOLO: has_error={has_error}, num_detections={len(detections)}")
        return {
            "has_error": has_error,
            "detections": detections,
            "num_detections": len(detections),
        }


def draw_detections(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        class_name = det["class"]
        conf = det["confidence"]

        # Caixa
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Texto
        label = f"{class_name} {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame



# --------------------------------------------------------------------
# State machine
# --------------------------------------------------------------------
class State(Enum):
    INIT = 0
    WAITING_PART = 1
    PART_ANALYSIS = 2
    UPDATE_METRICS = 3
    DEFECT_PRCSS = 4


class RadiatorCheckSM:

    def __init__(self, plc: PLC, detector: YoloDetector):
        self.plc = plc
        self.detector = detector
        self.state = State.INIT

        self.total_parts = 0
        self.total_errors = 0

        self.cap = cv2.VideoCapture(CAMERA_ID, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {CAMERA_ID}")
        log(f"Camera {CAMERA_ID} opened")

        self.last_detections = []

    def show_live_frame(self):
        frame = self._get_frame()
        if frame is None:
            return

        # desenhar somente as últimas detecções conhecidas
        frame_show = draw_detections(frame.copy(), self.last_detections)

        cv2.imshow("Inspeção (Live)", frame_show)

        # tecla para sair
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log("User requested exit (q)")
            self.state = None

    def _get_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            log("Camera read failed")
            return None
        return frame

    def step(self):
        self.show_live_frame()
        if self.state is None:
            return
        
        if self.state == State.INIT:
            log("SM: INIT -> system ready, waiting for part")
            self.state = State.WAITING_PART

        elif self.state == State.WAITING_PART:
            m202 = self.plc.read_bit("M202")
            if m202 is None:
                # communication error, stay here
                return
            if m202 == 1:
                log("SM: part detected (M202=1) -> PART_ANALYSIS")
                self.state = State.PART_ANALYSIS

        elif self.state == State.PART_ANALYSIS:
            frame = self._get_frame()
            if frame is None:
                return  # stay in same state, try again next step

            result = self.detector.detect(frame)

            # salvar boxes para mostrar no vídeo ao vivo
            self.last_detections = result["detections"]

            if result["has_error"]:
                log("SM: defect detected -> DEFECT_PRCSS")

                # Signal defect to PLC on M203 = 1
                if not self.plc.write_bit("M203", 1):
                    # if write failed, stay and try again later
                    return

                self.state = State.DEFECT_PRCSS
            else:
                self.plc.write_bit("M204", 1)
                    #time.sleep(0.01)  # short pulse, no strict timing needed
                self.state = State.UPDATE_METRICS

        elif self.state == State.UPDATE_METRICS:
            self.plc.write_bit("M204", 0)
            self.total_parts += 1
            log(f"SM: part OK, total_parts={self.total_parts} -> WAITING_PART")
            self.last_detections = []
            self.state = State.WAITING_PART

        elif self.state == State.DEFECT_PRCSS:
            self.total_errors += 1
            log(f"SM: processing defect, total_errors={self.total_errors}")

            # Wait until PLC clears M203 back to 0
            while True:
                m203 = self.plc.read_bit("M203")
                if m203 is None:
                    # comm error: break to avoid infinite loop
                    log("SM: error reading M203 during defect process")
                    break
                if m203 == 0:
                    log("SM: PLC cleared M203 -> WAITING_PART")
                    break
                time.sleep(0.1)

            self.last_detections = []
            self.state = State.WAITING_PART

    def cleanup(self):
        try:
            self.cap.release()
            log("Camera released")
        except Exception:
            pass
        self.plc.close()
        cv2.destroyAllWindows()


# --------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------
def main():
    plc = PLC()
    detector = YoloDetector()
    sm = RadiatorCheckSM(plc, detector)

    try:
        while True:
            sm.step()
            time.sleep(0.5)
    except KeyboardInterrupt:
        log("Stopping by user request")
    finally:
        sm.cleanup()


if __name__ == "__main__":
    main()
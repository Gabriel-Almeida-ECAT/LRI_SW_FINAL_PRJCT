import os
import cv2

from pathlib import Path
from datetime import datetime


# PLC Connection Settings
PLC_IP = "192.168.3.39"  # Change to your PLC's IP address
PLC_PORT = 5010          # Default port for MC Protocol (SLMP)


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
        log_msg = f"{datetime.now().strftime('%Y-%m-%d_%Hh:%Mm:%Ss')} {message}\n"
        print(log_msg)

        with open(self.crt_log_file, 'a') as file:
            file.write(log_msg)


def set_plc_bit(device, value):
    pymc3e = pymcprotocol.Type3E()

    try:
        # Connect to PLC
        pymc3e.connect(PLC_IP, PLC_PORT)
        print(f"Connected to PLC at {PLC_IP}:{PLC_PORT}")

        # Set the bit value
        pymc3e.batchwrite_bitunits(device, [value])
        print(f"Successfully set {device} to {value}")

        # Read back to verify
        result = pymc3e.batchread_bitunits(device, 1)
        print(f"Verification: {device} = {result[0]}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

    finally:
        pymc3e.close()
        print("Connection closed")


if __name__ == "__main__":
    #logHandler = logHandlerClass()
    #logHandler.log("test 2")

    set_plc_bit("M150", 0)

    pass





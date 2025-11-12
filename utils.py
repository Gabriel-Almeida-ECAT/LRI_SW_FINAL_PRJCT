import os
import cv2

#from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

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
        log_msg = f"{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')} {message}\n"
        print(log_msg)

        with open(self.crt_log_file, 'a') as file:
            file.write(log_msg)



if __name__ == "__main__":
    #logHandler = logHandlerClass()
    #logHandler.log("test 2")

    pass





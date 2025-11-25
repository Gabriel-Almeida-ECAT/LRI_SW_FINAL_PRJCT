import pymcprotocol

from utils import logHandlerClass

# PLC Connection Settings
PLC_IP = "192.168.3.39"  # Change to your PLC's IP address
PLC_PORT = 5010          # Default port for MC Protocol (SLMP)

logHandler_worker = logHandlerClass()

class plcInterfaceClass():
    def __init__(self):
        self.pymc3e = pymcprotocol.Type3E()

    def connect(self):
        try:
            self.pymc3e.connect(PLC_IP, PLC_PORT)
            logHandler_worker.log(f"plcInterfaceClass(): Connected to PLC at {PLC_IP}:{PLC_PORT}")

        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass(): Error: {e}")
            exit()
            return False

    def setMbit(self, device, value):
        try:
            self.pymc3e.batchwrite_bitunits(device, [value])
            result = self.pymc3e.batchread_bitunits(device, 1)
            if result == value:
                logHandler_worker.log(f"plcInterfaceClass(): successfully set {device} to {value}")

            return True

        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass(): Error: {e}")
            return False

    def readMemAddrs(self, addr):
        try:
            result = self.pymc3e.batchread_bitunits(addr, 1)

            return True

        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass(): Error: {e}")
            return False

    def __del__(self):
        self.pymc3e.close()
        logHandler_worker.log("plcInterfaceClass(): Connection closed")


if __name__ == "__main__":
    plcInterface = plcInterfaceClass()
    plcInterface.set_plc_bit("M150", 0)

    pass
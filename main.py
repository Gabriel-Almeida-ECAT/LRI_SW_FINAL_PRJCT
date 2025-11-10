import os
import datetime

from utils import logHandlerClass
from enum import Enum

logHandler = logHandlerClass()

class stateClass(Enum):
    INIT = 0
    WAITING_PART = 1
    PART_ANALYSIS = 2
    UPDATE_METRICS = 3
    DEFECT_PRCSS = 4

STATE = stateClass.INIT

def RADIATOR_CHECK_SM():
    match STATE:
        case stateClass.INIT:
            logHandler.log("RADIATOR_CHECK_SM(): Initiating system")

        case stateClass.WAITING_PART:
            # espera sinal do sensor de presenca de que a peca esta no local correto
            # desliga a esteira
            # segue pra analise de visao
            pass

        case stateClass.PART_ANALYSIS:

            pass

        case stateClass.UPDATE_METRICS:
            pass

        case stateClass.DEFECT_PRCSS:
            pass

def main():
    RADIATOR_CHECK_SM()

if __name__ == '__main__':
    main()
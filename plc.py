import os
import pymcprotocol

from utils import logHandlerClass

# PLC Connection Settings
PLC_IP = "192.168.3.39"  # IP do PLC
PLC_PORT = 5010          # Porta padrão do MC Protocol (SLMP)

logHandler_worker = logHandlerClass()


class plcInterfaceClass:
    """
    Interface de alto nível para o PLC Mitsubishi via pymcprotocol.

    Esta implementação é segura para uso em um cenário com múltiplos processos:
    - Não mantém uma conexão fixa por processo filho.
    - A cada uso, garante que a conexão pertence ao PID atual (reabre se necessário).
    """

    def __init__(self, ip: str = PLC_IP, port: int = PLC_PORT):
        self.ip = ip
        self.port = port
        self._pymc3e = None   # instância de Type3E
        self._pid = None      # PID no qual a conexão foi aberta

    # ------------------------------------------------------------------
    # Gestão da conexão (segura para multiprocessos)
    # ------------------------------------------------------------------
    def _ensure_connection(self) -> None:
        """
        Garante que exista uma conexão válida para o processo atual.

        Se:
          - ainda não houver conexão, OU
          - o objeto estiver sendo usado em um PID diferente daquele em que
            a conexão foi aberta,

        então uma nova conexão é criada para o PID atual.
        """
        current_pid = os.getpid()

        # Se não há conexão, ou se o PID mudou (objeto atravessou processos)
        if self._pymc3e is None or self._pid != current_pid:
            # Fecha conexão antiga (se existir) antes de reabrir
            if self._pymc3e is not None:
                try:
                    self._pymc3e.close()
                except Exception:
                    pass

            self._pymc3e = pymcprotocol.Type3E()
            self._pymc3e.connect(self.ip, self.port)
            self._pid = current_pid

            logHandler_worker.log(
                f"plcInterfaceClass._ensure_connection(): "
                f"Conectado ao PLC em {self.ip}:{self.port} (pid={current_pid})"
            )

    def connect(self) -> bool:
        """
        Abre (ou reabre) explicitamente a conexão para o processo atual.
        Normalmente não é necessário chamar manualmente, pois os métodos
        de leitura/escrita já chamam `_ensure_connection()`.
        """
        try:
            self._ensure_connection()
            return True
        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass.connect(): Erro ao conectar: {e}")
            return False

    def close(self) -> None:
        """Fecha a conexão atual com o PLC, se existir."""
        if self._pymc3e is not None:
            try:
                self._pymc3e.close()
                logHandler_worker.log("plcInterfaceClass.close(): Conexão encerrada")
            except Exception as e:
                logHandler_worker.log(f"plcInterfaceClass.close(): Erro ao fechar conexão: {e}")
            finally:
                self._pymc3e = None
                self._pid = None

    # ------------------------------------------------------------------
    # Operações de alto nível (bits M)
    # ------------------------------------------------------------------
    def setMbit(self, device: str, value) -> bool:
        """
        Escreve um bit (por ex. 'M203', 'M204') no PLC e verifica a escrita.

        Retorna:
            True  -> escrita confirmada
            False -> erro de comunicação ou falha na verificação
        """
        try:
            self._ensure_connection()

            v = int(bool(value))  # força valor 0/1
            self._pymc3e.batchwrite_bitunits(device, [v])

            # Lê de volta para confirmar
            result = self._pymc3e.batchread_bitunits(device, 1)
            if not result:
                logHandler_worker.log(
                    f"plcInterfaceClass.setMbit(): leitura vazia ao verificar {device}"
                )
                return False

            read_val = int(result[0])
            if read_val == v:
                logHandler_worker.log(
                    f"plcInterfaceClass.setMbit(): {device} definido como {v}"
                )
                return True
            else:
                logHandler_worker.log(
                    f"plcInterfaceClass.setMbit(): falha na verificação de {device}, "
                    f"esperado {v}, lido {read_val}"
                )
                return False

        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass.setMbit(): Erro: {e}")
            return False

    def readMemAddrs(self, addr: str):
        """
        Lê um bit (por ex. 'M202', 'M203').

        Retorna:
            0 ou 1  -> valor do bit
            None    -> em caso de erro (exceção, falha de comunicação)
        """
        try:
            self._ensure_connection()

            result = self._pymc3e.batchread_bitunits(addr, 1)
            if not result:
                logHandler_worker.log(
                    f"plcInterfaceClass.readMemAddrs(): leitura vazia em {addr}"
                )
                return None

            bit_val = int(result[0])
            logHandler_worker.log(
                f"plcInterfaceClass.readMemAddrs(): {addr} -> {bit_val}"
            )
            return bit_val

        except Exception as e:
            logHandler_worker.log(f"plcInterfaceClass.readMemAddrs(): Erro: {e}")
            return None

    # ------------------------------------------------------------------
    # Limpeza
    # ------------------------------------------------------------------
    def __del__(self):
        # Evita levantar exceção no GC caso algo dê errado
        try:
            self.close()
        except Exception:
            pass


if __name__ == "__main__":
    # Teste simples
    plc = plcInterfaceClass()
    ok = plc.setMbit("M150", 0)
    print(f"Escrita de teste em M150: {'OK' if ok else 'FALHOU'}")
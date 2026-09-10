"""
Utilidades para medir el consumo de recursos del sistema durante el benchmark.
"""
import logging
import platform
import sys
import time

import psutil

logger = logging.getLogger(__name__)


class SystemMetrics:
    """
    Mide el consumo de CPU y RAM del proceso actual durante el benchmark.

    Utiliza `psutil` para obtener el porcentaje de uso de CPU y la memoria
    RAM consumida por el proceso en ejecución. Está pensada para usarse
    dentro de bucles de alto rendimiento (un frame de video por iteración)
    sin introducir overhead significativo.
    """

    def __init__(self):
        """
        Inicializa el objeto y crea el handle al proceso actual.

        No realiza mediciones todavía; para eso debe llamarse a `start()`.

        Retorna:
            None
        """
        self._process = None
        try:
            self._process = psutil.Process()
            # Llamada "en vacío": psutil.cpu_percent necesita una primera
            # invocación de referencia antes de que los valores siguientes
            # sean significativos.
            self._process.cpu_percent(interval=None)
        except Exception as exc:
            logger.warning("No se pudo inicializar psutil.Process(): %s", exc)
            self._process = None

    def start(self):
        """
        Marca el inicio de una sesión de medición.

        Reinicia la referencia interna de `cpu_percent` para que las
        siguientes llamadas a `measure()` reflejen el consumo desde este
        punto en adelante.

        Retorna:
            None
        """
        self._start_time = time.time()
        if self._process is not None:
            try:
                self._process.cpu_percent(interval=None)
            except Exception as exc:
                logger.warning("No se pudo reiniciar la medición de CPU: %s", exc)

    def measure(self):
        """
        Toma una medición instantánea de CPU y RAM del proceso actual.

        Retorna:
            dict: {
                "cpu_percent": float,  # % de CPU usado por el proceso
                "ram_mb": float,       # RAM usada por el proceso, en MB
                "ram_percent": float,  # % de la RAM total del sistema
            }
            Si psutil falla, retorna valores en 0.0 en lugar de lanzar
            una excepción, para no interrumpir el benchmark en curso.
        """
        if self._process is None:
            return {"cpu_percent": 0.0, "ram_mb": 0.0, "ram_percent": 0.0}

        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / (1024 ** 2)
            ram_percent = self._process.memory_percent()
            return {
                "cpu_percent": float(cpu_percent),
                "ram_mb": float(ram_mb),
                "ram_percent": float(ram_percent),
            }
        except Exception as exc:
            logger.warning("Error midiendo métricas del sistema: %s", exc)
            return {"cpu_percent": 0.0, "ram_mb": 0.0, "ram_percent": 0.0}

    def get_system_info(self):
        """
        Obtiene información estática del sistema donde corre el benchmark.

        Retorna:
            dict: {
                "cpu_count": int,       # núcleos físicos disponibles
                "total_ram_gb": float,  # RAM total del sistema, en GB
                "platform": str,        # descripción del sistema operativo
                "python_version": str,  # versión de Python usada
            }
        """
        try:
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            return {
                "cpu_count": int(cpu_count),
                "total_ram_gb": round(float(total_ram_gb), 2),
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
            }
        except Exception as exc:
            logger.warning("Error obteniendo información del sistema: %s", exc)
            return {
                "cpu_count": 0,
                "total_ram_gb": 0.0,
                "platform": "desconocido",
                "python_version": sys.version.split()[0],
            }

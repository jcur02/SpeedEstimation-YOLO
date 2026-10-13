"""
Utilidades para medir el consumo de recursos del sistema durante el benchmark.
"""
import logging
import platform
import sys
import time

import psutil

logger = logging.getLogger(__name__)


def get_gpu_info():
    """
    Información de GPU/CUDA disponible en este equipo, para declarar las
    condiciones exactas de medición en el encabezado de la corrida y en el
    JSON de resumen.

    Importa `torch` de forma perezosa y protegida: si no está instalado o
    falla, degrada a valores `None` sin lanzar excepción.

    Retorna:
        dict: {
            "cuda_available": bool,
            "gpu_name": str | None,
            "torch_version": str | None,
            "cuda_version": str | None,
        }
    """
    info = {"cuda_available": False, "gpu_name": None, "torch_version": None, "cuda_version": None}
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if info["cuda_available"]:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda
    except Exception as exc:
        logger.warning("No se pudo obtener información de GPU/CUDA: %s", exc)
    return info


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
        Toma una medición instantánea de CPU, RAM y (si hay CUDA) memoria de
        GPU del proceso actual.

        Retorna:
            dict: {
                "cpu_percent": float,  # % de CPU usado por el proceso
                "ram_mb": float,       # RAM usada por el proceso, en MB
                "ram_percent": float,  # % de la RAM total del sistema
                "gpu_mem_allocated_mb": float | None,  # None si no hay CUDA
                "gpu_mem_peak_mb": float | None,       # None si no hay CUDA
            }
            Si psutil falla, retorna valores en 0.0/None en lugar de lanzar
            una excepción, para no interrumpir el benchmark en curso.
        """
        gpu_allocated_mb, gpu_peak_mb = self._measure_gpu()

        if self._process is None:
            return {
                "cpu_percent": 0.0, "ram_mb": 0.0, "ram_percent": 0.0,
                "gpu_mem_allocated_mb": gpu_allocated_mb, "gpu_mem_peak_mb": gpu_peak_mb,
            }

        try:
            cpu_percent = self._process.cpu_percent(interval=None)
            ram_mb = self._process.memory_info().rss / (1024 ** 2)
            ram_percent = self._process.memory_percent()
            return {
                "cpu_percent": float(cpu_percent),
                "ram_mb": float(ram_mb),
                "ram_percent": float(ram_percent),
                "gpu_mem_allocated_mb": gpu_allocated_mb,
                "gpu_mem_peak_mb": gpu_peak_mb,
            }
        except Exception as exc:
            logger.warning("Error midiendo métricas del sistema: %s", exc)
            return {
                "cpu_percent": 0.0, "ram_mb": 0.0, "ram_percent": 0.0,
                "gpu_mem_allocated_mb": gpu_allocated_mb, "gpu_mem_peak_mb": gpu_peak_mb,
            }

    def _measure_gpu(self):
        """
        Mide la memoria de GPU asignada actualmente y el pico acumulado
        desde el último `reset_gpu_peak()`, si hay CUDA disponible.

        Importa `torch` de forma perezosa y protegida.

        Retorna:
            tuple[float | None, float | None]: (memoria asignada en MB,
            pico en MB). Ambos `None` si no hay CUDA disponible o `torch`
            no está instalado.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return None, None
            allocated_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            return float(allocated_mb), float(peak_mb)
        except Exception as exc:
            logger.warning("Error midiendo memoria de GPU: %s", exc)
            return None, None

    def reset_gpu_peak(self):
        """
        Reinicia el contador de memoria pico de GPU. Llamar antes de medir
        cada modelo, para que el pico reportado corresponda solo a la fase
        medida y no arrastre el de un modelo evaluado antes.

        No-op si no hay CUDA disponible o `torch` no está instalado.

        Retorna:
            None
        """
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except Exception as exc:
            logger.warning("No se pudo reiniciar el pico de memoria de GPU: %s", exc)

    def get_system_info(self):
        """
        Obtiene información estática del sistema donde corre el benchmark,
        incluida la de GPU/CUDA/PyTorch (ver `get_gpu_info()`) — necesaria
        para que el JSON de resumen declare las condiciones exactas de
        medición.

        Retorna:
            dict: {
                "cpu_count": int,       # núcleos físicos disponibles
                "total_ram_gb": float,  # RAM total del sistema, en GB
                "platform": str,        # descripción del sistema operativo
                "python_version": str,  # versión de Python usada
                "cuda_available": bool,
                "gpu_name": str | None,
                "torch_version": str | None,
                "cuda_version": str | None,
            }
        """
        try:
            cpu_count = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 1
            total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            info = {
                "cpu_count": int(cpu_count),
                "total_ram_gb": round(float(total_ram_gb), 2),
                "platform": platform.platform(),
                "python_version": sys.version.split()[0],
            }
        except Exception as exc:
            logger.warning("Error obteniendo información del sistema: %s", exc)
            info = {
                "cpu_count": 0,
                "total_ram_gb": 0.0,
                "platform": "desconocido",
                "python_version": sys.version.split()[0],
            }

        info.update(get_gpu_info())
        return info

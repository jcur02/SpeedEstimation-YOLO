"""
Configuración compartida de logging para el benchmark.
"""
import logging
import os
import sys


def setup_logging(results_dir):
    """
    Configura el logging en nivel INFO, con salida a consola y a un
    archivo `benchmark.log` dentro de `results_dir`.

    Se usa tanto en el proceso principal (`scripts/run_benchmark.py`) como
    en cada worker por modelo (`src/detection/worker.py`), de forma que
    todos los logs de una misma corrida terminen en el mismo archivo.

    Parámetros:
        results_dir (str): directorio donde se guardará el log.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "benchmark.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

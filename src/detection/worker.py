"""
Worker de evaluación aislada: ejecuta UN solo modelo YOLO en un proceso propio.

Se invoca como subproceso desde `YOLOBenchmark.run()` (ver src/detection/benchmark.py)
para que la medición de RAM de cada modelo sea independiente de qué otros modelos
se evaluaron antes en la misma corrida del benchmark.

Uso:
    python -m src.detection.worker --config <config.yaml> --video <video.mp4> \
        --model-id <id> --weights <pesos.pt> --output <resultado.json>
"""
import argparse
import json
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.detection.benchmark import YOLOBenchmark
from src.detection.detector import VehicleDetector
from src.utils.logging_setup import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """
    Define y parsea los argumentos de línea de comandos del worker.

    Retorna:
        argparse.Namespace: argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Worker interno: evalúa un solo modelo YOLO.")
    parser.add_argument("--config", required=True, help="Ruta a un config.yaml ya resuelto.")
    parser.add_argument("--video", required=True, help="Ruta al video de prueba.")
    parser.add_argument("--model-id", required=True, help="Identificador del modelo a evaluar.")
    parser.add_argument("--weights", required=True, help="Nombre del archivo de pesos del modelo.")
    parser.add_argument("--output", required=True, help="Ruta donde escribir el resultado en JSON.")
    return parser.parse_args()


def main():
    """
    Punto de entrada del worker: carga un único modelo, procesa el video
    completo con él y escribe el resultado en `--output` como JSON.

    En caso de éxito escribe `{"frame_data": [...], "summary": {...}}`;
    si algo falla, escribe `{"error": "..."}` y termina con código 1, para
    que el proceso principal pueda seguir con el siguiente modelo.

    Retorna:
        None
    """
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    setup_logging(config["paths"]["results"])

    try:
        bench = YOLOBenchmark(config, args.video)
        detector = VehicleDetector(args.model_id, args.weights, config)
        print(f"  ✓ Modelo cargado ({detector.get_model_info()['parameters_M']:.1f} M parámetros)")

        frame_data = bench._process_video(detector)
        summary = bench._compute_summary(detector.get_model_info(), frame_data)

        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"frame_data": frame_data, "summary": summary}, f)

    except Exception as exc:
        logger.error("Error en el worker del modelo %s: %s", args.model_id, exc, exc_info=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump({"error": str(exc)}, f)
        sys.exit(1)


if __name__ == "__main__":
    main()

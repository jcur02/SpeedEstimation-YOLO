#!/usr/bin/env python3
"""
Script principal para ejecutar el benchmark de detección vehicular YOLO.

Uso:
    python scripts/run_benchmark.py --video data/videos/mi_video.mp4 --device cpu
"""
import argparse
import glob
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.benchmark import YOLOBenchmark
from src.utils.logging_setup import setup_logging
from src.utils.roi import load_roi, polygon_area_ratio

logger = logging.getLogger(__name__)


def parse_args():
    """
    Define y parsea los argumentos de línea de comandos del script.

    Retorna:
        argparse.Namespace: argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark de detección vehicular con variantes YOLO."
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Ruta al video de prueba. Si no se pasa, se busca automáticamente "
             "el primer .mp4 en data/videos/.",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml).",
    )
    parser.add_argument(
        "--models", type=str, default=None,
        help="Lista de model_ids separados por coma a evaluar (ej. yolov8n,yolo11n). "
             "Si no se pasa, se evalúan todos los del config.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Sobreescribe el número máximo de frames a procesar por video.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Sobreescribe el dispositivo de inferencia (cpu o cuda).",
    )
    parser.add_argument(
        "--roi", action="store_true",
        help="Activa el filtrado por región de interés (requiere haberla marcado con select_roi.py).",
    )
    parser.add_argument(
        "--roi-file", type=str, default=None,
        help="Ruta del archivo de ROI a usar. Default: roi.file del config.",
    )
    return parser.parse_args()


def find_video(videos_dir):
    """
    Busca automáticamente el primer archivo .mp4 disponible en un directorio.

    Parámetros:
        videos_dir (str): directorio donde buscar videos.

    Retorna:
        str | None: ruta al primer video encontrado (orden alfabético), o
        None si no hay ninguno.
    """
    candidates = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    return candidates[0] if candidates else None


def load_config(config_path):
    """
    Carga el archivo de configuración YAML del proyecto.

    Parámetros:
        config_path (str): ruta al archivo config.yaml.

    Retorna:
        dict: configuración cargada.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_overrides(config, args):
    """
    Aplica los overrides de línea de comandos sobre la configuración cargada.

    Parámetros:
        config (dict): configuración base cargada del YAML.
        args (argparse.Namespace): argumentos de línea de comandos.

    Retorna:
        dict: la misma configuración, modificada in-place con los overrides.
    """
    if args.max_frames is not None:
        config["benchmark"]["max_frames"] = args.max_frames

    if args.device is not None:
        config["benchmark"]["device"] = args.device

    if args.models is not None:
        requested_ids = [m.strip() for m in args.models.split(",") if m.strip()]
        available = config["detection"]["models"]
        filtered = [m for m in available if m["id"] in requested_ids]

        missing = set(requested_ids) - {m["id"] for m in filtered}
        if missing:
            print(f"⚠ Aviso: los siguientes model_ids no existen en el config y "
                  f"serán ignorados: {', '.join(sorted(missing))}")

        if not filtered:
            print("✗ Ninguno de los model_ids solicitados existe en el config. Abortando.")
            sys.exit(1)

        config["detection"]["models"] = filtered

    if args.roi:
        config.setdefault("roi", {})["enabled"] = True
    if args.roi_file is not None:
        config.setdefault("roi", {})["file"] = args.roi_file

    return config


def print_banner(config, video_path):
    """
    Imprime el encabezado con el resumen de configuración antes de iniciar
    el benchmark.

    Parámetros:
        config (dict): configuración final (con overrides aplicados).
        video_path (str): ruta al video de prueba a usar.

    Retorna:
        None
    """
    device = config["benchmark"]["device"]
    max_frames = config["benchmark"]["max_frames"]
    n_models = len(config["detection"]["models"])

    line = "=" * 64
    print(f"\n{line}")
    print("  Benchmark YOLO — Estimación de velocidad vehicular")
    print(f"  Video: {video_path}")
    print(f"  Dispositivo: {device} | Frames: {max_frames} | Modelos: {n_models}")

    roi_cfg = config.get("roi", {})
    if roi_cfg.get("enabled"):
        roi_file = roi_cfg.get("file", "config/roi.json")
        try:
            roi_data = load_roi(roi_file)
            area_pct = polygon_area_ratio(roi_data["roi_polygon"]) * 100
            print(f"  ROI: activa ({area_pct:.1f}% del frame) — {roi_file}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"  ROI: activa pero no se pudo cargar ({exc})")

    print(line)

    print("\nModelos a evaluar:")
    for m in config["detection"]["models"]:
        print(f"  - {m['id']}: {m.get('description', '')}")


def print_summary_table(summaries):
    """
    Imprime la tabla resumen final en consola con las métricas clave de
    cada modelo evaluado, ordenada por FPS medio descendente. Marca con
    *** el modelo con mejor FPS y con [!] los que superan 512 MB de RAM.

    Parámetros:
        summaries (list[dict]): resúmenes de todos los modelos evaluados.

    Retorna:
        None
    """
    if not summaries:
        print("\nNo hay resultados para mostrar (todos los modelos fallaron).")
        return

    ordered = sorted(summaries, key=lambda s: s["fps_mean"], reverse=True)
    best_fps_id = ordered[0]["model_id"]

    print("\n" + "═" * 22 + " RESULTADOS FINALES " + "═" * 22)
    print(f'{"Modelo":<12}│ {"FPS med":>7} │ {"FPS p5":>6} │ {"Inf.ms":>7} │ {"RAM MB":>8} │ {"CPU%":>6} │ {"Veh/f":>6}')
    print("─" * 12 + "┼" + "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 9 + "┼" + "─" * 10 + "┼" + "─" * 8 + "┼" + "─" * 7)

    for s in ordered:
        name = s["model_id"] + ("***" if s["model_id"] == best_fps_id else "")
        ram_str = f'{s["ram_mb_mean"]:.0f}' + ("[!]" if s["ram_mb_mean"] > 512 else "")

        print(
            f'{name:<12}│ {s["fps_mean"]:>7.1f} │ {s["fps_p5"]:>6.1f} │ '
            f'{s["inference_ms_mean"]:>7.1f} │ {ram_str:>8} │ '
            f'{s["cpu_percent_mean"]:>6.1f} │ {s["vehicles_per_frame_mean"]:>6.1f}'
        )

    print("─" * 66)
    print("*** Mejor FPS    [!] Supera 512 MB (límite típico RPi4)")


def main():
    """
    Punto de entrada principal: carga configuración, resuelve el video de
    prueba, pide confirmación al usuario y ejecuta el benchmark completo.

    Retorna:
        None
    """
    args = parse_args()
    config = load_config(args.config)

    setup_logging(config["paths"]["results"])
    config = apply_overrides(config, args)

    video_path = args.video or find_video(config["paths"]["videos"])
    if video_path is None:
        print(f"✗ No se especificó --video y no se encontró ningún .mp4 en "
              f"{config['paths']['videos']}")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"✗ El video especificado no existe: {video_path}")
        sys.exit(1)

    print_banner(config, video_path)

    answer = input("\n¿Continuar? [S/n]: ").strip().lower()
    if answer not in ("", "s", "si", "sí", "y", "yes"):
        print("Benchmark cancelado por el usuario.")
        sys.exit(0)

    logger.info("Iniciando benchmark sobre %s", video_path)
    try:
        benchmark = YOLOBenchmark(config, video_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"✗ Error inicializando el benchmark: {exc}")
        sys.exit(1)

    summaries = benchmark.run()
    print_summary_table(summaries)


if __name__ == "__main__":
    main()

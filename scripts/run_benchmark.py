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
from datetime import datetime

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.benchmark import YOLOBenchmark
from src.utils.logging_setup import setup_logging
from src.utils.metrics import get_gpu_info
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
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Repite la corrida completa N veces y reporta la variabilidad del fps_median "
             "entre repeticiones (distinto de la variabilidad entre frames dentro de una corrida).",
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


def print_banner(config, video_path, repeats=1):
    """
    Imprime el encabezado con el resumen de configuración antes de iniciar
    el benchmark, incluidas las condiciones exactas de medición (GPU,
    versión de PyTorch/CUDA) cuando el dispositivo es `cuda`.

    Parámetros:
        config (dict): configuración final (con overrides aplicados).
        video_path (str): ruta al video de prueba a usar.
        repeats (int): cantidad de repeticiones de la corrida completa.

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
    repeats_suffix = f" | Repeticiones: {repeats}" if repeats > 1 else ""
    print(f"  Dispositivo: {device} | Frames: {max_frames} | Modelos: {n_models}{repeats_suffix}")

    if device == "cuda":
        gpu_info = get_gpu_info()
        if gpu_info["cuda_available"]:
            print(f"  GPU: {gpu_info['gpu_name']} | PyTorch {gpu_info['torch_version']} | CUDA {gpu_info['cuda_version']}")
        else:
            print("  GPU: device=cuda configurado, pero CUDA no está disponible en este equipo.")

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

    El tiempo de inferencia se muestra como "mediana (media)": la mediana
    es la cifra principal por ser robusta a atípicos (Defecto 4); la media
    queda entre paréntesis como referencia. Si algún modelo tuvo frames
    atípicos (`outlier_frames > 0`), se advierte al final.

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
    print(f'{"Modelo":<12}│ {"FPS med":>7} │ {"FPS p5":>6} │ {"Inf.ms (mediana/media)":>23} │ {"RAM MB":>8} │ {"CPU%":>6} │ {"Veh/f":>6}')
    print("─" * 12 + "┼" + "─" * 9 + "┼" + "─" * 8 + "┼" + "─" * 25 + "┼" + "─" * 10 + "┼" + "─" * 8 + "┼" + "─" * 7)

    for s in ordered:
        name = s["model_id"] + ("***" if s["model_id"] == best_fps_id else "")
        ram_str = f'{s["ram_mb_mean"]:.0f}' + ("[!]" if s["ram_mb_mean"] > 512 else "")
        inf_median = s.get("inference_ms_median", s["inference_ms_mean"])
        inf_str = f'{inf_median:.1f} ({s["inference_ms_mean"]:.1f})'

        print(
            f'{name:<12}│ {s["fps_mean"]:>7.1f} │ {s["fps_p5"]:>6.1f} │ '
            f'{inf_str:>23} │ {ram_str:>8} │ '
            f'{s["cpu_percent_mean"]:>6.1f} │ {s["vehicles_per_frame_mean"]:>6.1f}'
        )

    print("─" * 82)
    print("*** Mejor FPS    [!] Supera 512 MB (límite típico RPi4)")

    outliers = [s for s in ordered if s.get("outlier_frames", 0) > 0]
    if outliers:
        print()
        for s in outliers:
            print(
                f"  ⚠ {s['model_id']}: {s['outlier_frames']} frame(s) atípico(s) "
                f"(inference_ms > 10× la mediana) — revisá si hubo carga externa en el equipo durante la corrida."
            )


def print_reproducibility_summary(all_repeats, benchmarks_dir):
    """
    Con `--repeats > 1`: calcula la desviación estándar de `fps_median`
    **entre repeticiones** (no entre frames dentro de una corrida) por
    modelo, la imprime, y guarda una tabla combinada de todas las
    repeticiones en un CSV con columna `repeat_idx`.

    Parámetros:
        all_repeats (list[list[dict]]): resúmenes por modelo, uno por
            repetición (cada elemento ya trae `repeat_idx` agregado).
        benchmarks_dir (str): directorio donde guardar la tabla combinada.

    Retorna:
        None
    """
    flat_rows = [row for rep in all_repeats for row in rep]
    if not flat_rows:
        return

    df = pd.DataFrame(flat_rows)

    print("\n" + "═" * 18 + " REPRODUCIBILIDAD ENTRE REPETICIONES " + "═" * 18)
    print(f'{"Modelo":<12}│ {"fps_median (std entre repeticiones)":>38}')
    print("─" * 12 + "┼" + "─" * 40)
    for model_id, group in df.groupby("model_id"):
        values = group["fps_median"].dropna().to_numpy(dtype=np.float64)
        std = float(np.std(values, ddof=0)) if len(values) > 1 else 0.0
        print(f'{model_id:<12}│ {std:>38.2f}')
    print("─" * 53)

    os.makedirs(benchmarks_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(benchmarks_dir, f"repeats_{timestamp}.csv")
    df.to_csv(out_path, index=False)
    print(f"\nTabla combinada de repeticiones (una fila por modelo y repetición): {out_path}")


def main():
    """
    Punto de entrada principal: carga configuración, resuelve el video de
    prueba, pide confirmación al usuario y ejecuta el benchmark completo
    (una vez, o `--repeats` veces para medir reproducibilidad).

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

    print_banner(config, video_path, args.repeats)

    answer = input("\n¿Continuar? [S/n]: ").strip().lower()
    if answer not in ("", "s", "si", "sí", "y", "yes"):
        print("Benchmark cancelado por el usuario.")
        sys.exit(0)

    logger.info("Iniciando benchmark sobre %s (%d repetición/es)", video_path, args.repeats)

    all_repeats = []
    for repeat_idx in range(args.repeats):
        if args.repeats > 1:
            print(f"\n{'=' * 64}\n  Repetición {repeat_idx + 1}/{args.repeats}\n{'=' * 64}")

        try:
            benchmark = YOLOBenchmark(config, video_path)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"✗ Error inicializando el benchmark: {exc}")
            sys.exit(1)

        summaries = benchmark.run()
        for s in summaries:
            s["repeat_idx"] = repeat_idx
        all_repeats.append(summaries)

    if not any(all_repeats):
        print_summary_table([])
        return

    print_summary_table(all_repeats[-1])

    if args.repeats > 1:
        print_reproducibility_summary(all_repeats, config["paths"].get("benchmarks", "results/benchmarks/"))


if __name__ == "__main__":
    main()

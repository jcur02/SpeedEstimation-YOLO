#!/usr/bin/env python3
"""
Benchmark comparativo de seguimiento: corre la matriz completa de modelos x
trackers (por defecto 5 modelos x 2 trackers = 10 corridas) sobre un mismo
video, y compara ByteTrack contra BoTSORT empíricamente.

Uso:
    python scripts/run_tracking_benchmark.py
    python scripts/run_tracking_benchmark.py --models yolov8n --max-frames 300
"""
import argparse
import glob
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tracking.tracker import VehicleTracker
from src.tracking.quality import TrackingQuality
from src.utils.metrics import SystemMetrics

logger = logging.getLogger(__name__)

_VALID_TRACKERS = ("bytetrack", "botsort")
# Estimación gruesa de ms/frame (detección + asociación) en CPU, solo para
# mostrar un tiempo total aproximado antes de confirmar. El valor real de
# cada corrida se ve en su propia barra de progreso.
_ASSUMED_MS_PER_FRAME = 150.0


def setup_tracking_logging(results_dir: str) -> None:
    """
    Configura el logging en nivel INFO para el benchmark de seguimiento, con
    salida a consola y a `tracking.log` dentro de `results_dir`, separado de
    `benchmark.log`, `preview.log` y `roi.log`.

    Parámetros:
        results_dir (str): directorio donde se guardará el log.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "tracking.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def parse_args() -> argparse.Namespace:
    """
    Define y parsea los argumentos de línea de comandos del script.

    Retorna:
        argparse.Namespace: argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark comparativo de seguimiento: modelos x trackers."
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
        help="Lista de model_ids separados por coma (ej. yolov8n,yolo11n). "
             "Si no se pasa, se evalúan todos los del config.",
    )
    parser.add_argument(
        "--trackers", type=str, default="bytetrack,botsort",
        help="Lista de trackers separados por coma a comparar (default: bytetrack,botsort).",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Sobreescribe el umbral de confianza.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Sobreescribe el dispositivo de inferencia (cpu o cuda).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limita cuántos frames procesar por corrida (para pruebas rápidas).",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Fuerza los FPS del video si los reporta mal (0 o negativos).",
    )
    parser.add_argument(
        "--no-save-tracks", action="store_true",
        help="No escribe los JSON de trayectorias en results/tracks/.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Carga el archivo de configuración YAML del proyecto.

    Retorna:
        dict: configuración cargada.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_video(videos_dir: str) -> Optional[str]:
    """
    Busca automáticamente el primer archivo .mp4 disponible en un directorio.

    Retorna:
        str | None: ruta al primer video encontrado (orden alfabético), o
        None si no hay ninguno.
    """
    candidates = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))
    return candidates[0] if candidates else None


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Aplica los overrides de línea de comandos sobre la configuración cargada.

    Retorna:
        dict: la misma configuración, modificada in-place.
    """
    if args.conf is not None:
        config["detection"]["confidence_threshold"] = args.conf
    if args.device is not None:
        config.setdefault("benchmark", {})["device"] = args.device
    return config


def resolve_models(config: dict, models_arg: Optional[str]) -> List[dict]:
    """
    Resuelve la lista de modelos a evaluar a partir de `--models`.

    Retorna:
        list[dict]: entradas de `detection.models` a evaluar.
    """
    available = config["detection"]["models"]
    if models_arg is None:
        return available

    requested = [m.strip() for m in models_arg.split(",") if m.strip()]
    filtered = [m for m in available if m["id"] in requested]

    missing = set(requested) - {m["id"] for m in filtered}
    if missing:
        print(f"⚠ Aviso: los siguientes model_ids no existen en el config y "
              f"serán ignorados: {', '.join(sorted(missing))}")

    if not filtered:
        print("✗ Ninguno de los model_ids solicitados existe en el config. Abortando.")
        sys.exit(1)

    return filtered


def resolve_trackers(trackers_arg: str, tracker_config_dir: str) -> List[str]:
    """
    Resuelve y valida la lista de trackers a comparar a partir de `--trackers`.

    Retorna:
        list[str]: nombres de tracker válidos y con su YAML presente.
    """
    requested = [t.strip() for t in trackers_arg.split(",") if t.strip()]

    invalid = [t for t in requested if t not in _VALID_TRACKERS]
    if invalid:
        print(f"✗ Tracker(s) no reconocido(s): {', '.join(invalid)}. "
              f"Válidos: {', '.join(_VALID_TRACKERS)}")
        sys.exit(1)

    for tracker_name in requested:
        path = os.path.join(tracker_config_dir, f"{tracker_name}.yaml")
        if not os.path.isfile(path):
            print(f"✗ No se encontró la configuración del tracker '{tracker_name}' en '{path}'.")
            sys.exit(1)

    if not requested:
        print("✗ No se especificó ningún tracker en --trackers.")
        sys.exit(1)

    return requested


def peek_video_info(video_path: str) -> Tuple[int, float, int, int]:
    """
    Abre brevemente un video para leer su metadata, sin procesar frames.

    Retorna:
        tuple[int, float, int, int]: (total_frames, fps, ancho, alto).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ No se pudo abrir el video: {video_path}")
        sys.exit(1)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return total_frames, fps, width, height


def find_latest_detection_summary(benchmarks_dir: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Busca el JSON de resumen de benchmark de detección pura más reciente en
    `results/benchmarks/`, para cuantificar el overhead del tracking.

    Retorna:
        tuple[dict | None, str | None]: (contenido del summary, ruta), o
        (None, None) si no hay ninguno o todos están corruptos.
    """
    paths = sorted(glob.glob(os.path.join(benchmarks_dir, "summary_*.json")), reverse=True)
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f), path
        except (OSError, json.JSONDecodeError):
            continue
    return None, None


def detection_only_stats(summary_data: Optional[dict], model_id: str) -> Optional[Dict[str, float]]:
    """
    Extrae `fps_mean`/`inference_ms_mean` de un modelo dentro de un summary
    de benchmark de detección pura.

    Retorna:
        dict | None: `{"fps_mean", "inference_ms_mean"}`, o None si no hay
        summary o el modelo no aparece en él.
    """
    if summary_data is None:
        return None
    for m in summary_data.get("models", []):
        if m.get("model_id") == model_id:
            return {"fps_mean": m.get("fps_mean"), "inference_ms_mean": m.get("inference_ms_mean")}
    return None


def print_banner(video_path: str, models: List[dict], trackers: List[str],
                  frames_to_process: int, source_fps: float) -> None:
    """
    Imprime la matriz a evaluar y una estimación gruesa del tiempo total.

    Retorna:
        None
    """
    n_combos = len(models) * len(trackers)
    estimated_sec = n_combos * frames_to_process * _ASSUMED_MS_PER_FRAME / 1000.0

    line = "=" * 64
    print(f"\n{line}")
    print("  Benchmark de seguimiento — YOLO + ByteTrack/BoTSORT")
    print(f"  Video: {video_path} ({source_fps:.1f} fps)")
    print(f"  Frames por corrida: {frames_to_process} | Combinaciones: {n_combos}")
    print(f"  Tiempo estimado: ~{estimated_sec / 60:.1f} min "
          f"(estimación gruesa, ~{_ASSUMED_MS_PER_FRAME:.0f} ms/frame asumidos; "
          f"el tiempo real se ve en la barra de progreso de cada corrida)")
    print(line)

    print("\nMatriz a evaluar:")
    for model_cfg in models:
        for tracker_name in trackers:
            print(f"  - {model_cfg['id']} + {tracker_name}")


def run_one_combination(
    model_cfg: dict, tracker_name: str, config: dict, video_path: str,
    frames_to_process: int, source_fps: float, width: int, height: int,
    tracks_dir: str, save_tracks: bool,
) -> Tuple[Dict[str, Any], VehicleTracker]:
    """
    Corre una combinación (modelo, tracker) completa sobre el video: recorre
    los frames con `tqdm`, mide tiempo por frame y muestrea métricas de
    sistema, finaliza el tracker y calcula sus métricas de calidad.

    Retorna:
        tuple[dict, VehicleTracker]: (fila de resultados para la tabla
        comparativa, el tracker ya finalizado — por si se necesita inspeccionar).
    """
    # Instancia nueva de VehicleTracker (y por lo tanto de YOLO) para esta
    # combinación exacta: ver la decisión de diseño C en VehicleTracker.
    tracker = VehicleTracker(model_cfg["id"], model_cfg["weights"], tracker_name, config, source_fps)
    tracker.source_video = video_path
    tracker.frame_size = (width, height)

    system_metrics = SystemMetrics()
    system_metrics.start()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    inference_times: List[float] = []
    system_samples: List[Dict[str, float]] = []

    try:
        with tqdm(total=frames_to_process, desc=f"{model_cfg['id']} + {tracker_name}", unit="frame") as progress:
            frame_idx = 0
            while frame_idx < frames_to_process:
                ret, frame = cap.read()
                if not ret:
                    break

                t0 = time.perf_counter()
                tracker.update(frame, frame_idx)
                t1 = time.perf_counter()

                inference_times.append((t1 - t0) * 1000.0)
                system_samples.append(system_metrics.measure())
                frame_idx += 1
                progress.update(1)
    finally:
        cap.release()

    tracker.finalize()

    tracks_path = ""
    if save_tracks:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        tracks_path = os.path.join(tracks_dir, f"{model_cfg['id']}_{tracker_name}_{video_name}.json")
        tracker.save_tracks(tracks_path)

    quality = TrackingQuality(tracker.tracks, (width, height), config).compute()

    inference_arr = np.array(inference_times) if inference_times else np.array([0.0])
    fps_arr = np.where(inference_arr > 0, 1000.0 / inference_arr, 0.0)
    ram_arr = np.array([s["ram_mb"] for s in system_samples]) if system_samples else np.array([0.0])
    cpu_arr = np.array([s["cpu_percent"] for s in system_samples]) if system_samples else np.array([0.0])

    row: Dict[str, Any] = {
        "model_id": model_cfg["id"],
        "tracker": tracker_name,
        "frames_processed": len(inference_times),
        "fps_mean": float(fps_arr.mean()),
        "fps_std": float(fps_arr.std(ddof=0)),
        "fps_p5": float(np.percentile(fps_arr, 5)) if len(fps_arr) else 0.0,
        "inference_ms_mean": float(inference_arr.mean()),
        "inference_ms_std": float(inference_arr.std(ddof=0)),
        "ram_mb_mean": float(ram_arr.mean()),
        "cpu_percent_mean": float(cpu_arr.mean()),
        "tracks_path": tracks_path,
    }
    row.update(quality)

    return row, tracker


def save_results(rows: List[dict], config: dict, video_path: str, timestamp: str, benchmarks_dir: str) -> Tuple[str, str]:
    """
    Guarda la tabla comparativa (CSV) y el resumen completo (JSON) del
    benchmark de seguimiento.

    Retorna:
        tuple[str, str]: (ruta del CSV, ruta del JSON).
    """
    csv_path = os.path.join(benchmarks_dir, f"tracking_comparison_{timestamp}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    json_path = os.path.join(benchmarks_dir, f"tracking_summary_{timestamp}.json")
    payload = {
        "video_path": video_path,
        "timestamp": timestamp,
        "config": config,
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("\nArchivos generados:")
    print(f"  \U0001F4CA {csv_path}")
    print(f"  \U0001F4CB {json_path}")
    logger.info("Resultados de seguimiento guardados en %s y %s", csv_path, json_path)

    return csv_path, json_path


def generate_plots(rows: List[dict], plots_dir: str, timestamp: str) -> None:
    """
    Genera las tres gráficas comparativas del benchmark de seguimiento.

    Retorna:
        None
    """
    sns.set_theme(style="whitegrid")
    df = pd.DataFrame(rows)
    df["combination"] = df["model_id"] + "+" + df["tracker"]

    # 1. FPS por modelo, barras agrupadas por tracker.
    models = sorted(df["model_id"].unique())
    trackers = sorted(df["tracker"].unique())
    x = np.arange(len(models))
    bar_width = 0.8 / max(1, len(trackers))

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, tracker_name in enumerate(trackers):
        sub = df[df["tracker"] == tracker_name].set_index("model_id").reindex(models)
        ax.bar(x + i * bar_width, sub["fps_mean"], bar_width, yerr=sub["fps_std"],
               capsize=3, label=tracker_name)
    ax.set_xticks(x + bar_width * (len(trackers) - 1) / 2)
    ax.set_xticklabels(models, rotation=20)
    ax.axhline(15, color="red", linestyle="--", label="Mínimo embebido (15 FPS)")
    ax.set_title("FPS de seguimiento por modelo y tracker")
    ax.set_xlabel("Modelo")
    ax.set_ylabel("FPS")
    ax.legend()
    fig.tight_layout()
    fps_path = os.path.join(plots_dir, "tracking_fps_comparison.png")
    fig.savefig(fps_path, dpi=150)
    plt.close(fig)

    # 2. Calidad: inestabilidad de identidad y fragmentación por combinación.
    order = df.sort_values("identity_instability")["combination"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(data=df, x="combination", y="identity_instability", order=order, ax=axes[0], color="indianred")
    axes[0].set_title("Inestabilidad de identidad (indicador indirecto, no MOTA/HOTA)")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("interior_births + interior_deaths / unique_ids")
    axes[0].tick_params(axis="x", rotation=45)
    sns.barplot(data=df, x="combination", y="fragment_ratio", order=order, ax=axes[1], color="steelblue")
    axes[1].set_title("Fracción de tracks fragmentados")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=45)
    fig.tight_layout()
    quality_path = os.path.join(plots_dir, "tracking_quality.png")
    fig.savefig(quality_path, dpi=150)
    plt.close(fig)

    # 3. Compromiso FPS vs. inestabilidad — la gráfica que justifica la
    #    elección final entre tracker/modelo.
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("husl", len(df))
    for (_, row), color in zip(df.iterrows(), palette):
        ax.scatter(row["identity_instability"], row["fps_mean"], s=120, color=color)
        ax.annotate(
            row["combination"], (row["identity_instability"], row["fps_mean"]),
            textcoords="offset points", xytext=(6, 6), fontsize=9,
        )
    ax.set_title("Compromiso: FPS vs. inestabilidad de identidad (indicador indirecto)")
    ax.set_xlabel("Inestabilidad de identidad (más bajo = mejor)")
    ax.set_ylabel("FPS")
    fig.tight_layout()
    tradeoff_path = os.path.join(plots_dir, "tracking_tradeoff.png")
    fig.savefig(tradeoff_path, dpi=150)
    plt.close(fig)

    print(f"  \U0001F4C8 {fps_path}")
    print(f"  \U0001F4C8 {quality_path}")
    print(f"  \U0001F4C8 {tradeoff_path}")
    logger.info("Gráficas de seguimiento guardadas en %s", plots_dir)


def print_summary_table(rows: List[dict]) -> None:
    """
    Imprime la tabla resumen final en consola, ordenada por FPS descendente,
    marcando el mejor FPS y el mejor compromiso FPS/estabilidad.

    Retorna:
        None
    """
    df = pd.DataFrame(rows).sort_values("fps_mean", ascending=False)
    best_fps_combo = f"{df.iloc[0]['model_id']}+{df.iloc[0]['tracker']}"

    median_instability = df["identity_instability"].median()
    candidates = df[df["identity_instability"] <= median_instability].sort_values("fps_mean", ascending=False)
    best_tradeoff_row = candidates.iloc[0] if not candidates.empty else df.iloc[0]
    best_tradeoff_combo = f"{best_tradeoff_row['model_id']}+{best_tradeoff_row['tracker']}"

    print("\n" + "═" * 20 + " RESULTADOS DE SEGUIMIENTO " + "═" * 20)
    print(f'{"Modelo+Tracker":<24}│ {"FPS":>7} │ {"IDs":>5} │ {"Frag.":>6} │ {"Inestab.":>8} │ {"Recta":>6}')
    print("─" * 24 + "┼" + "─" * 9 + "┼" + "─" * 7 + "┼" + "─" * 8 + "┼" + "─" * 10 + "┼" + "─" * 7)

    for _, r in df.iterrows():
        combo = f"{r['model_id']}+{r['tracker']}"
        marker = ("*" if combo == best_fps_combo else "") + ("+" if combo == best_tradeoff_combo else "")
        label = f"{combo}{marker}"
        print(
            f'{label:<24}│ {r["fps_mean"]:>7.1f} │ {int(r["unique_ids"]):>5} │ '
            f'{r["fragment_ratio"]:>6.2f} │ {r["identity_instability"]:>8.2f} │ {r["straightness_mean"]:>6.2f}'
        )

    print("─" * 68)
    print("*  Mejor FPS    +  Mejor compromiso FPS/estabilidad "
          "(mayor FPS entre los de inestabilidad <= mediana)")
    print("\nRecordatorio: 'Inestab.' es un indicador indirecto (ver src/tracking/quality.py), no MOTA/HOTA.")


def main() -> None:
    """
    Punto de entrada principal: resuelve la matriz modelo x tracker, pide
    confirmación, corre cada combinación y genera los resultados comparativos.

    Retorna:
        None
    """
    args = parse_args()
    config = load_config(args.config)

    setup_tracking_logging(config["paths"]["results"])
    config = apply_overrides(config, args)

    video_path = args.video or find_video(config["paths"]["videos"])
    if video_path is None:
        print(f"✗ No se especificó --video y no se encontró ningún .mp4 en "
              f"{config['paths']['videos']}")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"✗ El video especificado no existe: {video_path}")
        sys.exit(1)

    tracking_cfg = config.setdefault("tracking", {})
    tracker_config_dir = tracking_cfg.get("tracker_config_dir", "config/trackers/")
    tracks_dir = tracking_cfg.get("tracks_dir", "results/tracks/")
    os.makedirs(tracks_dir, exist_ok=True)

    models = resolve_models(config, args.models)
    trackers = resolve_trackers(args.trackers, tracker_config_dir)

    total_frames, source_fps, width, height = peek_video_info(video_path)
    if args.fps is not None:
        source_fps = args.fps
    if source_fps <= 0:
        print(f"✗ El video no reporta FPS válidos ({source_fps}). Pasalos manualmente con --fps.")
        sys.exit(1)

    frames_to_process = min(total_frames, args.max_frames) if args.max_frames else total_frames
    if frames_to_process <= 0:
        print("✗ El video no tiene frames para procesar.")
        sys.exit(1)

    print_banner(video_path, models, trackers, frames_to_process, source_fps)

    answer = input("\n¿Continuar? [S/n]: ").strip().lower()
    if answer not in ("", "s", "si", "sí", "y", "yes"):
        print("Benchmark de seguimiento cancelado por el usuario.")
        sys.exit(0)

    benchmarks_dir = config["paths"].get("benchmarks", "results/benchmarks/")
    plots_dir = config["paths"].get("plots", "results/plots/")
    os.makedirs(benchmarks_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    detection_summary, detection_summary_path = find_latest_detection_summary(benchmarks_dir)
    if detection_summary_path:
        print(f"\nUsando benchmark de detección pura como referencia de overhead: {detection_summary_path}")
    else:
        print("\n(No se encontró un benchmark de detección pura previo en "
              f"{benchmarks_dir}; se omite la columna de overhead sin fallar.)")

    logger.info("Iniciando benchmark de seguimiento sobre %s", video_path)

    rows: List[Dict[str, Any]] = []
    total = len(models) * len(trackers)
    idx = 0

    for model_cfg in models:
        for tracker_name in trackers:
            idx += 1
            print(f"\n[{idx}/{total}] Evaluando {model_cfg['id']} + {tracker_name}...")
            logger.info("[%d/%d] Evaluando %s + %s", idx, total, model_cfg["id"], tracker_name)

            try:
                row, _tracker = run_one_combination(
                    model_cfg, tracker_name, config, video_path, frames_to_process,
                    source_fps, width, height, tracks_dir, not args.no_save_tracks,
                )
            except Exception as exc:
                logger.error("Error evaluando %s + %s: %s", model_cfg["id"], tracker_name, exc, exc_info=True)
                print(f"  ✗ Error evaluando {model_cfg['id']} + {tracker_name}: {exc}")
                continue

            det_stats = detection_only_stats(detection_summary, model_cfg["id"])
            if det_stats is not None and det_stats.get("inference_ms_mean"):
                row["detection_only_fps_mean"] = det_stats["fps_mean"]
                row["detection_only_inference_ms_mean"] = det_stats["inference_ms_mean"]
                row["tracking_overhead_ms"] = row["inference_ms_mean"] - det_stats["inference_ms_mean"]
                row["tracking_overhead_pct"] = (
                    100.0 * row["tracking_overhead_ms"] / det_stats["inference_ms_mean"]
                )
            else:
                row["detection_only_fps_mean"] = None
                row["detection_only_inference_ms_mean"] = None
                row["tracking_overhead_ms"] = None
                row["tracking_overhead_pct"] = None

            rows.append(row)
            print(
                f"  ✓ {model_cfg['id']}+{tracker_name} completado: "
                f"{row['fps_mean']:.1f} FPS | {row['unique_ids']} IDs | "
                f"inestabilidad {row['identity_instability']:.2f}"
            )

    if not rows:
        print("\n✗ Ninguna combinación pudo evaluarse correctamente. No se generaron resultados.")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(rows, config, video_path, timestamp, benchmarks_dir)
    generate_plots(rows, plots_dir, timestamp)
    print_summary_table(rows)


if __name__ == "__main__":
    main()

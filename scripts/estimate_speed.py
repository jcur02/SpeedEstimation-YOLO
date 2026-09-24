#!/usr/bin/env python3
"""
Estimación de velocidad vehicular a partir de trayectorias ya calibradas.

Uso:
    python scripts/estimate_speed.py --self-test
    python scripts/estimate_speed.py
    python scripts/estimate_speed.py --tracks results/tracks/yolov8n_bytetrack_sitio_a_01.json
    python scripts/estimate_speed.py --all-estimators
    python scripts/estimate_speed.py --validate config/speed_reference.csv
"""
import argparse
import glob
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import cv2
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.homography import CalibratedPlane
from src.speed.estimator import SpeedEstimator, TrackSpeed
from src.speed.report import plot_estimator_comparison, plot_speed_histogram, plot_speed_profile, plot_trajectories_metric
from src.speed.synthetic import estimate_error_budget, run_suite
from src.speed.validation import SpeedValidation, interpret_bias_dispersion, plot_bland_altman, plot_error_vs_distance, plot_scatter_validation

logger = logging.getLogger(__name__)

_ESTIMATOR_NAMES = ("window", "endpoint", "cumulative")


def setup_speed_logging(results_dir: str) -> None:
    """
    Configura el logging en nivel INFO para la estimación de velocidad, con
    salida a consola y a `speed.log` dentro de `results_dir`.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "speed.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(description="Estimación de velocidad vehicular a partir de trayectorias calibradas.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Ruta al archivo de configuración.")
    parser.add_argument("--tracks", type=str, default=None, help="Archivo de trayectorias. Default: el más reciente de results/tracks/.")
    parser.add_argument("--calibration", type=str, default=None, help="Ruta a calibration.json. Default: calibration.file del config.")
    parser.add_argument("--estimator", type=str, default=None, choices=list(_ESTIMATOR_NAMES), help="Estimador a usar. Default: speed.estimator del config.")
    parser.add_argument("--window", type=float, default=None, help="Ancho de la ventana, en segundos. Default: speed.window_s del config.")
    parser.add_argument("--output", type=str, default=None, help="Prefijo de los archivos de salida. Default: automático en results/speed/.")
    parser.add_argument("--validate", type=str, default=None, help="CSV de referencia; activa el módulo de validación.")
    parser.add_argument("--self-test", action="store_true", help="Corre el banco sintético. No requiere calibración.")
    parser.add_argument("--all-estimators", action="store_true", help="Calcula los tres estimadores y genera la comparación.")
    parser.add_argument("--min-speed", type=float, default=None, help="Velocidad mínima (km/h) para considerar un track en movimiento. Default: speed.filters.min_speed_kmh.")
    parser.add_argument("--force", action="store_true", help="Continúa aunque la calidad de la calibración esté por debajo del umbral.")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Carga el archivo de configuración YAML del proyecto."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_latest_tracks(tracks_dir: str) -> Optional[str]:
    """
    Busca el archivo de trayectorias más reciente (por fecha de modificación)
    en `tracks_dir`.

    Retorna:
        str | None: ruta al archivo más reciente, o None si no hay ninguno.
    """
    candidates = glob.glob(os.path.join(tracks_dir, "*.json"))
    return max(candidates, key=os.path.getmtime) if candidates else None


def _print_box(title: str, sections: List[List[str]]) -> None:
    """
    Imprime un resumen en un cuadro de consola, con secciones separadas por
    líneas horizontales.

    Parámetros:
        title (str): título de la primera línea.
        sections (list[list[str]]): cada elemento es una lista de líneas de
            contenido; se separan entre sí con una línea horizontal.

    Retorna:
        None
    """
    width = 62

    def line(text: str = "") -> str:
        return "║" + text.ljust(width)[:width] + "║"

    print("\n╔" + "═" * width + "╗")
    print(line(f"  {title}"))
    for i, section in enumerate(sections):
        print("╠" + "═" * width + "╣")
        for content_line in section:
            print(line(f"  {content_line}"))
    print("╚" + "═" * width + "╝")


def _categorize_rejection(ts: TrackSpeed) -> str:
    """Clasifica la causa dominante de rechazo de un `TrackSpeed` inválido, a partir de sus `quality_flags`."""
    flags = " ".join(ts.quality_flags)
    if "fuera_zona_calibrada" in flags:
        return "fuera de zona calibrada"
    if "escala_excesiva" in flags:
        return "escala excesiva"
    if "salto_imposible" in flags or "id_switch_suspected" in flags:
        return "salto/cambio de ID"
    return "muy corta"


def _print_speed_summary(results: List[TrackSpeed], n_total_tracks: int, n_moving_tracks: int, min_speed_kmh: float) -> List[TrackSpeed]:
    """
    Imprime el resumen de la estimación (cuadro de consola) y retorna los
    resultados válidos que además superan `min_speed_kmh` (los usados para
    las estadísticas agregadas de velocidad "típica").
    """
    valid = [r for r in results if r.is_valid]
    rejected = [r for r in results if not r.is_valid]

    reasons: Dict[str, int] = {}
    for r in rejected:
        cat = _categorize_rejection(r)
        reasons[cat] = reasons.get(cat, 0) + 1
    reasons_str = ", ".join(f"{count} {cat}" for cat, count in reasons.items()) or "ninguna"

    counts_section = [
        f"Trayectorias:            {n_total_tracks:<4} ({n_moving_tracks} en movimiento)",
        f"Válidas tras filtrado:   {len(valid)}",
        f"Descartadas:             {len(rejected)}   ({reasons_str})",
    ]

    moving_results = [r for r in valid if r.speed_mean_kmh >= min_speed_kmh]
    if moving_results:
        speeds = sorted(r.speed_mean_kmh for r in moving_results)
        mean_v = sum(speeds) / len(speeds)
        median_v = speeds[len(speeds) // 2] if len(speeds) % 2 else (speeds[len(speeds) // 2 - 1] + speeds[len(speeds) // 2]) / 2.0
        speed_section = [
            f"Velocidad media:         {mean_v:.1f} km/h",
            f"Mediana:                 {median_v:.1f} km/h",
            f"Rango:                   {speeds[0]:.1f} – {speeds[-1]:.1f} km/h",
        ]
    else:
        speed_section = ["Sin tracks en movimiento por encima del mínimo configurado."]

    _print_box("Estimación de velocidad", [counts_section, speed_section])
    return moving_results


def run_self_test(config: dict) -> None:
    """
    Implementa `--self-test`: corre el banco sintético, imprime el criterio
    de aceptación con marca de cumplido o no por cada punto, y el
    presupuesto de error. No toca la calibración en absoluto.

    Retorna:
        None
    """
    out_dir = config.get("speed", {}).get("synthetic", {}).get("output_dir", "results/speed/synthetic/")
    print(f"Corriendo el banco de pruebas sintético (esto no usa ninguna calibración real)...\n  Salida: {out_dir}\n")

    result = run_suite(config, out_dir)

    print(f"\nResultados: {result['csv_path']}")
    for name, path in result["plot_paths"].items():
        print(f"  {name}: {path}")

    print("\nCriterios de aceptación:")
    all_passed = True
    for name, check in result["checks"].items():
        mark = "✓" if check["passed"] else "✗"
        all_passed = all_passed and check["passed"]
        print(f"  [{mark}] {name}: {check['detail']}")

    print("\nPresupuesto de error (condiciones representativas):")
    budget = estimate_error_budget(config)
    cond = budget["condiciones"]
    print(f"  Velocidad={cond['speed_kmh']:.0f} km/h, ruido={cond['noise_px']:.0f}px, tramo medio ({cond['zone_depth_m']:.0f} m de profundidad)")
    print(f"  Cuantización de píxeles:   {budget['cuantizacion_kmh']:.3f} km/h")
    print(f"  Ruido de detección:        {budget['ruido_deteccion_kmh']:.3f} km/h")
    print(f"  Geometría (long. vs trans.): {budget['geometria_kmh']:.3f} km/h")
    print(f"  Total, vista transversal:  {budget['total_transversal_kmh']:.3f} km/h")
    print(f"  Total, vista longitudinal: {budget['total_longitudinal_kmh']:.3f} km/h")

    if all_passed:
        print("\n✓ El estimador se comporta como se espera bajo condiciones controladas.")
    else:
        print("\n✗ Alguno de los criterios de aceptación no se cumplió — revisá src/speed/estimator.py antes de confiar en datos reales.")


def _read_reference_frame(video_path: str) -> Optional[Any]:
    """Intenta leer un frame medio de `video_path`, para `CalibratedPlane.check_frame()`. Retorna None si falla, sin lanzar."""
    if not video_path or not os.path.isfile(video_path):
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2 if total > 0 else 0)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    except cv2.error:
        return None


def _check_calibration_quality(plane: CalibratedPlane, config: dict, force: bool, warning_path: str) -> None:
    """
    Compara `plane.quality` contra los umbrales de la fase de calibración. Si
    la calidad es insuficiente, advierte de forma prominente; sin `--force`,
    aborta. Con `--force`, además dejar la advertencia por escrito en
    `warning_path`, para que sobreviva más allá de la consola.

    Retorna:
        None
    """
    thresholds = config.get("calibration", {}).get("thresholds", {})
    quality = plane.quality or {}
    reproj_mean = quality.get("reprojection_mean_m")
    loo = quality.get("leave_one_out", {}) or {}

    problems = []
    if reproj_mean is not None and reproj_mean > thresholds.get("reprojection_warn_m", 0.30):
        problems.append(f"error de reproyección medio ({reproj_mean:.2f} m) supera el umbral ({thresholds.get('reprojection_warn_m', 0.30):.2f} m)")
    if loo.get("available") and loo.get("mean_m", 0.0) > thresholds.get("loo_warn_m", 0.60):
        problems.append(f"validación cruzada ({loo['mean_m']:.2f} m) supera el umbral ({thresholds.get('loo_warn_m', 0.60):.2f} m)")

    if not problems:
        return

    message = (
        "La calidad de la calibración está por debajo del umbral aceptable: " + "; ".join(problems) + ". "
        "Las velocidades resultantes NO son confiables — corré scripts/calibrate.py de nuevo antes de "
        "sacar conclusiones de estos números."
    )
    print(f"\n⚠⚠⚠ {message}")
    logger.warning(message)

    if not force:
        print("\n✗ Deteniendo (usá --force para continuar de todas formas; la advertencia igual queda en la salida).")
        sys.exit(1)

    out_dir = os.path.dirname(warning_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(warning_path, "w", encoding="utf-8") as f:
        f.write(f"ADVERTENCIA: {message}\n")
    print(f"  (continuando por --force; advertencia guardada en {warning_path})")


def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()
    config = load_config(args.config)
    setup_speed_logging(config["paths"]["results"])

    if args.self_test:
        run_self_test(config)
        return

    speed_cfg = config.setdefault("speed", {})
    if args.window is not None:
        speed_cfg["window_s"] = args.window
    estimator_name = args.estimator or speed_cfg.get("estimator", "window")
    min_speed_kmh = args.min_speed if args.min_speed is not None else speed_cfg.get("filters", {}).get("min_speed_kmh", 3.0)

    tracks_path = args.tracks or find_latest_tracks(config.get("tracking", {}).get("tracks_dir", "results/tracks/"))
    if not tracks_path or not os.path.isfile(tracks_path):
        print(
            f"✗ No se encontró ningún archivo de trayectorias en "
            f"{config.get('tracking', {}).get('tracks_dir', 'results/tracks/')}. Generalo primero con "
            f"scripts/run_tracking_benchmark.py, o pasá --tracks."
        )
        sys.exit(1)

    calibration_path = args.calibration or config.get("calibration", {}).get("file", "config/calibration.json")
    try:
        plane = CalibratedPlane.load(calibration_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}")
        print("  Mientras tanto, podés validar el estimador sin calibración con: python scripts/estimate_speed.py --self-test")
        sys.exit(1)

    output_dir = speed_cfg.get("output_dir", "results/speed/")
    tracks_basename = os.path.splitext(os.path.basename(tracks_path))[0]
    output_prefix = args.output or os.path.join(output_dir, tracks_basename)
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    _check_calibration_quality(plane, config, args.force, f"{output_prefix}_calibration_warning.txt")

    estimator = SpeedEstimator(config, plane=plane)

    with open(tracks_path, "r", encoding="utf-8") as f:
        tracks_data = json.load(f)
    n_total_tracks = len(tracks_data.get("tracks", []))
    n_moving_tracks = sum(1 for t in tracks_data.get("tracks", []) if not t.get("is_stationary"))
    reference_frame = _read_reference_frame(tracks_data.get("source_video", ""))

    print(f"Trayectorias: {tracks_path}")
    print(f"Calibración:  {calibration_path}")

    try:
        if args.all_estimators:
            results_by_estimator: Dict[str, List[TrackSpeed]] = {}
            for name in _ESTIMATOR_NAMES:
                estimator_results = estimator.estimate_all(tracks_path, estimator=name, reference_frame=reference_frame)
                results_by_estimator[name] = estimator_results
                estimator.to_csv(estimator_results, f"{output_prefix}_{name}_speeds.csv")
                estimator.to_json(estimator_results, f"{output_prefix}_{name}_speeds.json")
        else:
            results = estimator.estimate_all(tracks_path, estimator=estimator_name, reference_frame=reference_frame)
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n✗ {exc}")
        sys.exit(1)

    if args.all_estimators:
        plot_estimator_comparison(results_by_estimator, f"{output_prefix}_estimator_comparison.png")
        results = results_by_estimator[estimator_name]
        csv_path, json_path = f"{output_prefix}_{estimator_name}_speeds.csv", f"{output_prefix}_{estimator_name}_speeds.json"
        print(f"\nSe calcularon los tres estimadores; comparación en {output_prefix}_estimator_comparison.png")
        print(f"Resumen principal según --estimator ({estimator_name}):")
    else:
        csv_path, json_path = f"{output_prefix}_speeds.csv", f"{output_prefix}_speeds.json"
        estimator.to_csv(results, csv_path)
        estimator.to_json(results, json_path)

    _print_speed_summary(results, n_total_tracks, n_moving_tracks, min_speed_kmh)

    plot_speed_histogram(results, f"{output_prefix}_histogram.png")
    plot_trajectories_metric(results, plane, f"{output_prefix}_trajectories.png")

    longest_valid = sorted((r for r in results if r.is_valid), key=lambda r: r.n_used, reverse=True)[:5]
    for r in longest_valid:
        plot_speed_profile(r, f"{output_prefix}_profile_track{r.track_id}.png")

    print("\nSalida:")
    print(f"  CSV:            {csv_path}")
    print(f"  JSON:           {json_path}")
    print(f"  Histograma:     {output_prefix}_histogram.png")
    print(f"  Trayectorias:   {output_prefix}_trajectories.png")
    if longest_valid:
        print(f"  Perfiles:       {output_prefix}_profile_track<id>.png ({len(longest_valid)} track(s) más largos)")

    if args.validate:
        try:
            reference = SpeedValidation.load_reference(args.validate)
        except (FileNotFoundError, ValueError) as exc:
            print(f"\n✗ {exc}")
            sys.exit(1)

        validation = SpeedValidation()
        pairs = validation.join(results, reference, tracks_data.get("source_video", tracks_path))

        if not pairs:
            print("\n✗ No se formó ningún par válido; no se pueden calcular métricas de validación.")
            return

        metrics = validation.metrics()
        interpretation = interpret_bias_dispersion(metrics)

        validation_section = [
            f"n = {metrics['n']}",
            f"MAE:   {metrics['mae_kmh']:.2f} km/h    MAPE: {metrics['mape_pct']:.1f}%",
            f"RMSE:  {metrics['rmse_kmh']:.2f} km/h    r:    {metrics['pearson_r']:.3f}",
            f"Sesgo: {metrics['bias_kmh']:+.2f} km/h   σ:    {metrics['std_error_kmh']:.2f} km/h",
        ]
        _print_box("Validación contra referencia", [validation_section])
        print(f"\n{interpretation}")

        val_out_dir = speed_cfg.get("validation", {}).get("output_dir", "results/speed/validation/")
        os.makedirs(val_out_dir, exist_ok=True)
        plot_scatter_validation(pairs, os.path.join(val_out_dir, f"{tracks_basename}_scatter_validation.png"))
        plot_bland_altman(pairs, os.path.join(val_out_dir, f"{tracks_basename}_bland_altman.png"))
        plot_error_vs_distance(pairs, os.path.join(val_out_dir, f"{tracks_basename}_error_vs_distance.png"))
        print(f"\nGráficas de validación en {val_out_dir}")


if __name__ == "__main__":
    main()

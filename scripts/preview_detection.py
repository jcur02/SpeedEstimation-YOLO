#!/usr/bin/env python3
"""
Vista previa interactiva de detección vehicular sobre un video.

Herramienta de validación cualitativa, independiente del benchmark
(`scripts/run_benchmark.py`): permite ver el video corriendo con las
bounding boxes dibujadas en tiempo real, o exportarlo anotado a un archivo,
para revisar visualmente que un modelo detecta bien.

Uso:
    python scripts/preview_detection.py --model yolov8n
    python scripts/preview_detection.py --export --max-frames 600
"""
import argparse
import glob
import logging
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.homography import CalibratedPlane
from src.detection.detector import VehicleDetector
from src.detection.visualizer import DetectionVisualizer
from src.speed.estimator import SpeedEstimator
from src.tracking.tracker import VehicleTracker

logger = logging.getLogger(__name__)


def setup_preview_logging(results_dir: str) -> None:
    """
    Configura el logging en nivel INFO para el modo de vista previa, con
    salida a consola y a `preview.log` dentro de `results_dir`.

    Se mantiene deliberadamente separado de `setup_logging` (usado por el
    benchmark, que escribe en `benchmark.log`) para no mezclar los logs de
    ambas herramientas.

    Parámetros:
        results_dir (str): directorio donde se guardará el log.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "preview.log")

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
        description="Vista previa interactiva de detección vehicular con YOLO."
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Ruta al video. Si se omite, se usa el primer .mp4 de data/videos/.",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="model_id a usar (debe existir en detection.models). "
             "Default: preview.default_model del config.",
    )
    parser.add_argument(
        "--compare", type=str, default=None,
        help="Segundo model_id para vista lado a lado.",
    )
    parser.add_argument(
        "--export", action="store_true",
        help="Exporta el resultado a un archivo de video en vez de abrir una ventana.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Ruta del archivo de salida en modo export. Default: automática en results/preview/.",
    )
    parser.add_argument(
        "--codec", type=str, default=None,
        help="Codec del VideoWriter para exportar. Default: preview.export_codec del config.",
    )
    parser.add_argument(
        "--conf", type=float, default=None,
        help="Umbral de confianza. Default: detection.confidence_threshold del config.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Dispositivo de inferencia. Default: benchmark.device del config.",
    )
    parser.add_argument(
        "--start-sec", type=float, default=0.0,
        help="Segundo del video donde empezar (default: 0).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Limita cuántos frames procesar. Default: el video completo.",
    )
    parser.add_argument(
        "--fps", type=int, default=None,
        help="Limita la tasa de reproducción. Default: preview.playback_fps del config.",
    )
    parser.add_argument(
        "--no-hud", action="store_true",
        help="Arranca con el panel de información (HUD) oculto.",
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
        "--track", action="store_true",
        help="Activa el modo seguimiento (IDs persistentes, estelas) en vez de solo detección.",
    )
    parser.add_argument(
        "--tracker", type=str, default=None, choices=["bytetrack", "botsort"],
        help="Tracker a usar con --track. Default: tracking.default_tracker del config.",
    )
    parser.add_argument(
        "--compare-tracker", type=str, default=None, choices=["bytetrack", "botsort"],
        help="Segundo tracker (mismo modelo) para vista lado a lado. Requiere --track.",
    )
    parser.add_argument(
        "--trail", type=int, default=None,
        help="Longitud de la estela en frames. Default: tracking.visualization.trail_length del config.",
    )
    parser.add_argument(
        "--no-trail", action="store_true",
        help="Arranca con las estelas ocultas.",
    )
    parser.add_argument(
        "--speed", action="store_true",
        help="Activa el overlay de velocidad en vivo (km/h junto al ID). Requiere --track y una "
             "calibración ya hecha con scripts/calibrate.py.",
    )
    parser.add_argument(
        "--calibration", type=str, default=None,
        help="Ruta a calibration.json para --speed. Default: calibration.file del config.",
    )
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Carga el archivo de configuración YAML del proyecto.

    Parámetros:
        config_path (str): ruta al archivo config.yaml.

    Retorna:
        dict: configuración cargada.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_video(videos_dir: str) -> "str | None":
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


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    """
    Aplica los overrides de línea de comandos sobre la configuración cargada,
    antes de construir ningún detector.

    Parámetros:
        config (dict): configuración base cargada del YAML.
        args (argparse.Namespace): argumentos de línea de comandos.

    Retorna:
        dict: la misma configuración, modificada in-place con los overrides.
    """
    if args.conf is not None:
        config["detection"]["confidence_threshold"] = args.conf

    if args.device is not None:
        config.setdefault("benchmark", {})["device"] = args.device

    if args.codec is not None:
        config["preview"]["export_codec"] = args.codec

    if args.fps is not None:
        config["preview"]["playback_fps"] = args.fps

    if args.roi:
        config.setdefault("roi", {})["enabled"] = True
    if args.roi_file is not None:
        config.setdefault("roi", {})["file"] = args.roi_file

    return config


def resolve_model(config: dict, model_id: "str | None", role: str) -> dict:
    """
    Valida que un model_id exista en `detection.models` y retorna su entrada
    de configuración completa.

    Parámetros:
        config (dict): configuración completa.
        model_id (str | None): model_id solicitado. Si es None, se usa
            `preview.default_model`.
        role (str): descripción del rol de este modelo, para los mensajes de
            error (ej. "principal" o "de comparación").

    Retorna:
        dict: entrada del modelo (`id`, `weights`, `description`).
    """
    available = config.get("detection", {}).get("models", [])
    valid_ids = [m["id"] for m in available]

    if model_id is None:
        model_id = config.get("preview", {}).get("default_model")

    matches = [m for m in available if m["id"] == model_id]
    if not matches:
        print(f"✗ El modelo {role} '{model_id}' no existe en config.detection.models.")
        print(f"  Modelos válidos: {', '.join(valid_ids)}")
        sys.exit(1)

    return matches[0]


def print_banner(video_path: str, visualizer: DetectionVisualizer, model_id: str,
                  conf: float, device: str, mode_desc: str, tracking_active: bool = False,
                  speed_active: bool = False) -> None:
    """
    Imprime el encabezado con la configuración efectiva antes de arrancar.

    Parámetros:
        video_path (str): ruta al video.
        visualizer (DetectionVisualizer): visualizador ya construido (para
            leer resolución, fps y cantidad de frames).
        model_id (str): modelo principal a usar.
        conf (float): umbral de confianza efectivo.
        device (str): dispositivo de inferencia efectivo.
        mode_desc (str): descripción del modo de ejecución elegido.
        tracking_active (bool): si es True, agrega la línea de controles
            propios del modo seguimiento (estelas, IDs, color).
        speed_active (bool): si es True, agrega el control del overlay de
            velocidad estimada (requiere `tracking_active`).

    Retorna:
        None
    """
    lines = [
        "Vista previa de detección — YOLO",
        f"Video: {video_path}  ({visualizer.width}x{visualizer.height}, "
        f"{visualizer.source_fps:.0f} fps, {visualizer.total_frames} f)",
        f"Modelo: {model_id} | conf: {conf:.2f} | device: {device}",
        f"Modo: {mode_desc}",
    ]
    box_width = max(len(line) for line in lines) + 2

    print("\n╔" + "═" * box_width + "╗")
    for line in lines:
        print("║ " + line.ljust(box_width - 1) + "║")
    print("╚" + "═" * box_width + "╝")

    print("\nControles: [espacio] pausa · [n] siguiente frame · [s] captura")
    print("           [b] cajas · [h] HUD · [+/-] velocidad de reproducción · [r] reiniciar · [q] salir")
    if tracking_active:
        extra = " · [v] velocidad estimada" if speed_active else ""
        print(f"           [t] estelas · [i] IDs · [c] color por ID/clase · [[/]] largo de estela{extra}\n")
    else:
        print()


def main() -> None:
    """
    Punto de entrada principal: carga configuración, resuelve video y
    modelo(s), construye el visualizador y ejecuta el modo correspondiente
    (ventana interactiva, exportación o comparación).

    A diferencia de `run_benchmark.py`, este script no pide confirmación al
    usuario: es una herramienta de uso rápido y repetitivo.

    Retorna:
        None
    """
    args = parse_args()
    config = load_config(args.config)

    setup_preview_logging(config["paths"]["results"])
    config = apply_overrides(config, args)

    video_path = args.video or find_video(config["paths"]["videos"])
    if video_path is None:
        print(f"✗ No se especificó --video y no se encontró ningún .mp4 en "
              f"{config['paths']['videos']}. Coloca ahí tu video de prueba o "
              f"indica la ruta con --video.")
        sys.exit(1)
    if not os.path.isfile(video_path):
        print(f"✗ El video especificado no existe: {video_path}")
        sys.exit(1)

    if args.track and args.compare:
        print("✗ No se puede combinar --compare (comparar modelos) con --track. "
              "Para comparar dos trackers del mismo modelo, usá --compare-tracker.")
        sys.exit(1)
    if args.compare_tracker and not args.track:
        print("✗ --compare-tracker requiere --track.")
        sys.exit(1)
    if args.speed and not args.track:
        print("✗ --speed requiere --track (la velocidad se calcula sobre la trayectoria de un ID persistente).")
        sys.exit(1)

    model_cfg = resolve_model(config, args.model, "principal")
    compare_cfg = resolve_model(config, args.compare, "de comparación") if args.compare else None

    try:
        logger.info("Cargando detector principal: %s", model_cfg["id"])
        detector = VehicleDetector(model_cfg["id"], model_cfg["weights"], config)

        detector_b = None
        if compare_cfg is not None:
            logger.info("Cargando detector de comparación: %s", compare_cfg["id"])
            detector_b = VehicleDetector(compare_cfg["id"], compare_cfg["weights"], config)

        visualizer = DetectionVisualizer(config, detector, video_path)

        tracker_obj = None
        tracker_b_obj = None
        if args.track:
            tracker_name = args.tracker or config.get("tracking", {}).get("default_tracker", "bytetrack")
            logger.info("Construyendo tracker principal: %s + %s", model_cfg["id"], tracker_name)
            tracker_obj = VehicleTracker(
                model_cfg["id"], model_cfg["weights"], tracker_name, config, visualizer.source_fps,
            )
            visualizer.tracker = tracker_obj

            if args.speed:
                calibration_path = args.calibration or config.get("calibration", {}).get("file", "config/calibration.json")
                try:
                    plane = CalibratedPlane.load(calibration_path)
                except (FileNotFoundError, ValueError) as exc:
                    print(f"✗ {exc}")
                    sys.exit(1)

                plane_w, plane_h = plane.frame_size
                if (plane_w, plane_h) != (visualizer.width, visualizer.height):
                    print(
                        f"✗ La calibración '{calibration_path}' fue ajustada para "
                        f"{plane_w}x{plane_h}, pero este video es {visualizer.width}x{visualizer.height}. "
                        f"Usarla igual daría un error de escala sistemático. Recalibrá para esta resolución."
                    )
                    sys.exit(1)

                logger.info("Calibración cargada para overlay de velocidad: %s", calibration_path)
                visualizer.plane = plane
                visualizer.speed_estimator = SpeedEstimator(config, plane=plane)

            if args.compare_tracker:
                logger.info(
                    "Construyendo tracker de comparación: %s + %s", model_cfg["id"], args.compare_tracker,
                )
                tracker_b_obj = VehicleTracker(
                    model_cfg["id"], model_cfg["weights"], args.compare_tracker, config, visualizer.source_fps,
                )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"✗ Error inicializando la vista previa: {exc}")
        sys.exit(1)

    try:
        visualizer.seek(args.start_sec)
    except ValueError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    visualizer.max_frames = args.max_frames
    if args.no_hud:
        visualizer.show_hud = False
    if args.trail is not None:
        visualizer.trail_length = args.trail
    if args.no_trail:
        visualizer.show_trails = False

    compare_requested = detector_b is not None
    compare_tracker_requested = tracker_b_obj is not None
    want_export = args.export
    if not want_export and visualizer._headless_check():
        print("⚠ No se detectó un display disponible (SSH sin X / entorno headless). "
              "Se cambia automáticamente a modo exportación.")
        want_export = True

    if compare_tracker_requested:
        mode_desc = f"comparación de trackers {args.tracker} vs {args.compare_tracker} — " + (
            "exportación a archivo" if want_export else "ventana interactiva"
        )
    elif compare_requested:
        mode_desc = f"comparación {model_cfg['id']} vs {compare_cfg['id']} — " + (
            "exportación a archivo" if want_export else "ventana interactiva"
        )
    elif args.track:
        speed_suffix = " + velocidad" if args.speed else ""
        mode_desc = f"seguimiento ({tracker_obj.tracker_name}{speed_suffix}) — " + (
            "exportación a archivo" if want_export else "ventana interactiva"
        )
    elif want_export:
        mode_desc = "exportación a archivo"
    else:
        mode_desc = "ventana interactiva"

    print_banner(
        video_path, visualizer, model_cfg["id"],
        config["detection"]["confidence_threshold"], config["benchmark"]["device"], mode_desc,
        tracking_active=args.track, speed_active=args.speed,
    )

    try:
        if compare_tracker_requested:
            visualizer.run_compare(tracker_b=tracker_b_obj, export=want_export, output_path=args.output)
        elif compare_requested:
            visualizer.run_compare(detector_b, export=want_export, output_path=args.output)
        elif want_export:
            visualizer.run_export(args.output)
        else:
            visualizer.run_live()
    except KeyboardInterrupt:
        print("\nVista previa interrumpida por el usuario (Ctrl+C). Liberando recursos...")
        visualizer.cap.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        sys.exit(0)


if __name__ == "__main__":
    main()

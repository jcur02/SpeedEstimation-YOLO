#!/usr/bin/env python3
"""
Herramienta interactiva para marcar la región de interés (ROI) de un sitio
de grabación. Se corre una sola vez por sitio; el resultado (`config/roi.json`
por defecto) se reutiliza en todas las corridas posteriores de ese sitio.

Uso:
    python scripts/select_roi.py --video "data/videos/sitio_a_*.mp4"
    python scripts/select_roi.py --show
"""
import argparse
import glob
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import VehicleDetector
from src.detection.heatmap import TrafficHeatmap
from src.utils.roi import ROIFilter, load_roi, md5_of_frame, polygon_area_ratio, reference_frame_path, save_roi

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ROI_COLOR = (0, 255, 255)
_HOMOGRAPHY_COLOR = (255, 0, 255)
_MAD_WARNING_THRESHOLD = 20.0


def setup_roi_logging(results_dir: str) -> None:
    """
    Configura el logging en nivel INFO para la herramienta de ROI, con
    salida a consola y a `roi.log` dentro de `results_dir`, separado de
    `benchmark.log` y `preview.log`.

    Parámetros:
        results_dir (str): directorio donde se guardará el log.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "roi.log")

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
        description="Marca interactivamente la región de interés (ROI) de un sitio de grabación."
    )
    parser.add_argument(
        "--video", type=str, nargs="+", default=None,
        help="Uno o varios videos (rutas o comodines). Default: todos los .mp4 de data/videos/.",
    )
    parser.add_argument(
        "--config", type=str, default="config/config.yaml",
        help="Ruta al archivo de configuración (default: config/config.yaml).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Dónde guardar el JSON de la ROI. Default: roi.file del config.",
    )
    parser.add_argument(
        "--no-heatmap", action="store_true",
        help="Salta el mapa de calor y marca sobre un frame limpio.",
    )
    parser.add_argument(
        "--frame-sec", type=float, default=None,
        help="Segundo del que se toma el frame de fondo. Default: el frame medio del primer video.",
    )
    parser.add_argument(
        "--sample-mode", type=str, default=None, choices=["uniform", "middle", "range"],
        help="Modo de muestreo del mapa de calor. Default: roi.heatmap.sample_mode del config.",
    )
    parser.add_argument(
        "--samples", type=int, default=None,
        help="Cuántos frames muestrear para el mapa de calor. Default: roi.heatmap.samples del config.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Modelo para acumular el mapa de calor. Default: roi.heatmap.model del config.",
    )
    parser.add_argument(
        "--device", type=str, default=None, choices=["cpu", "cuda"],
        help="Dispositivo de inferencia. Default: benchmark.device del config.",
    )
    parser.add_argument(
        "--with-homography", action="store_true",
        help="Tras el polígono, pide 4 puntos adicionales para la calibración futura.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Solo muestra la ROI ya guardada sobre el frame, sin editarla.",
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


def _is_headless() -> bool:
    """
    Determina si el entorno actual probablemente no tiene display disponible
    (por ejemplo, una sesión SSH a una Raspberry Pi sin X).

    Retorna:
        bool: True si no parece haber display disponible.
    """
    if os.name == "posix" and os.uname().sysname != "Darwin":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    try:
        cv2.namedWindow("_roi_headless_probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_roi_headless_probe")
        return False
    except cv2.error:
        return True


def resolve_videos(video_args: Optional[List[str]], videos_dir: str) -> List[str]:
    """
    Resuelve la lista de videos a usar, expandiendo comodines.

    Parámetros:
        video_args (list[str] | None): rutas o patrones pasados por
            `--video`. Si es None, se usan todos los `.mp4` de `videos_dir`.
        videos_dir (str): directorio por defecto donde buscar videos.

    Retorna:
        list[str]: rutas de video encontradas, ordenadas y sin duplicados.
    """
    if video_args:
        expanded = []
        for pattern in video_args:
            matches = sorted(glob.glob(pattern))
            if matches:
                expanded.extend(matches)
            elif os.path.isfile(pattern):
                expanded.append(pattern)
            else:
                print(f"✗ No se encontró ningún archivo para el patrón/ruta: '{pattern}'")
                sys.exit(1)
        videos = sorted(set(expanded))
    else:
        videos = sorted(glob.glob(os.path.join(videos_dir, "*.mp4")))

    if not videos:
        print(f"✗ No se encontró ningún video. Indicá --video o coloca archivos .mp4 en {videos_dir}.")
        sys.exit(1)

    return videos


def validate_same_resolution(videos: List[str]) -> Tuple[int, int]:
    """
    Verifica que todos los videos tengan la misma resolución (requisito para
    que la misma ROI normalizada les sirva a todos).

    Parámetros:
        videos (list[str]): rutas de video a validar.

    Retorna:
        tuple[int, int]: `(ancho, alto)` común a todos los videos.
    """
    sizes: Dict[str, Tuple[int, int]] = {}
    for path in videos:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            print(f"✗ No se pudo abrir el video: {path}")
            sys.exit(1)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        sizes[path] = (w, h)

    distinct = set(sizes.values())
    if len(distinct) > 1:
        print("✗ Los videos tienen resoluciones distintas; deben coincidir para usar la misma ROI:")
        for path, size in sizes.items():
            print(f"    {path}: {size[0]}x{size[1]}")
        sys.exit(1)

    width, height = next(iter(distinct))
    print(f"  {len(videos)} video(s) validados, resolución {width}x{height}.")
    return width, height


def get_background_frame(video_path: str, frame_sec: Optional[float]) -> np.ndarray:
    """
    Extrae un frame de fondo de un video, para marcar la ROI sobre él.

    Parámetros:
        video_path (str): ruta al video.
        frame_sec (float | None): segundo del que tomar el frame. Si es
            None, se usa el frame medio del video.

    Retorna:
        numpy.ndarray: el frame extraído (BGR).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ No se pudo abrir el video: {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    target = int(round(frame_sec * fps)) if frame_sec is not None else (total_frames // 2 if total_frames > 0 else 0)
    target = max(0, min(target, max(total_frames - 1, 0)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"✗ No se pudo leer ningún frame de: {video_path}")
        sys.exit(1)

    return frame


def resolve_model_entry(config: dict, model_id: str) -> dict:
    """
    Valida que un model_id exista en `detection.models` y retorna su entrada
    de configuración completa.

    Parámetros:
        config (dict): configuración completa.
        model_id (str): model_id solicitado.

    Retorna:
        dict: entrada del modelo (`id`, `weights`, `description`).
    """
    available = config.get("detection", {}).get("models", [])
    matches = [m for m in available if m["id"] == model_id]
    if not matches:
        valid = ", ".join(m["id"] for m in available)
        print(f"✗ El modelo '{model_id}' no existe en config.detection.models. Válidos: {valid}")
        sys.exit(1)
    return matches[0]


# ----------------------------------------------------------------------
# Geometría: autointersección de polígonos
# ----------------------------------------------------------------------

def _orientation(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> int:
    """Retorna 0 (colineales), 1 (horario) o 2 (antihorario) para a-b-c."""
    val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if abs(val) < 1e-9:
        return 0
    return 1 if val > 0 else 2


def _on_segment(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> bool:
    """Asumiendo a, b, c colineales, retorna si b está en el segmento a-c."""
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


def _segments_intersect(
    p1: Tuple[int, int], p2: Tuple[int, int], p3: Tuple[int, int], p4: Tuple[int, int]
) -> bool:
    """Retorna True si el segmento p1-p2 se cruza con el segmento p3-p4."""
    o1, o2 = _orientation(p1, p2, p3), _orientation(p1, p2, p4)
    o3, o4 = _orientation(p3, p4, p1), _orientation(p3, p4, p2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(p1, p3, p2):
        return True
    if o2 == 0 and _on_segment(p1, p4, p2):
        return True
    if o3 == 0 and _on_segment(p3, p1, p4):
        return True
    if o4 == 0 and _on_segment(p3, p2, p4):
        return True
    return False


def _polygon_self_intersects(points: List[Tuple[int, int]]) -> bool:
    """
    Verifica si un polígono cerrado se autointerseca, comparando cada par de
    aristas no adyacentes.

    Parámetros:
        points (list[tuple[int, int]]): vértices del polígono, en orden.

    Retorna:
        bool: True si alguna pareja de aristas no adyacentes se cruza.
    """
    n = len(points)
    if n < 4:
        return False
    edges = [(points[i], points[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue  # aristas adyacentes, comparten vértice por construcción
            if _segments_intersect(*edges[i], *edges[j]):
                return True
    return False


def _polygon_area_px(points: List[Tuple[int, int]], width: int, height: int) -> float:
    """Fracción del área del frame cubierta por un polígono en píxeles."""
    if len(points) < 3:
        return 0.0
    n = len(points)
    area2 = 0.0
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0 / (width * height)


def _draw_dashed_line(
    img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], color: Tuple[int, int, int],
    thickness: int = 2, dash_len: int = 10,
) -> None:
    """Dibuja una línea punteada entre dos puntos, in-place."""
    dist = int(np.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1]))
    if dist == 0:
        return
    dashes = max(1, dist // dash_len)
    for i in range(dashes):
        f1, f2 = i / dashes, (i + 0.5) / dashes
        sx, sy = int(pt1[0] + (pt2[0] - pt1[0]) * f1), int(pt1[1] + (pt2[1] - pt1[1]) * f1)
        ex, ey = int(pt1[0] + (pt2[0] - pt1[0]) * f2), int(pt1[1] + (pt2[1] - pt1[1]) * f2)
        cv2.line(img, (sx, sy), (ex, ey), color, thickness)


# ----------------------------------------------------------------------
# Marcado interactivo
# ----------------------------------------------------------------------

def _render_polygon_overlay(
    base_frame: np.ndarray, points: List[Tuple[int, int]], mouse_pos: Tuple[int, int],
    intersects: bool, width: int, height: int,
) -> np.ndarray:
    """
    Dibuja el estado actual del polígono en edición sobre el frame de fondo:
    vértices numerados, aristas, línea elástica, relleno translúcido,
    contador de vértices/área, y advertencia de autointersección si aplica.

    Retorna:
        numpy.ndarray: frame listo para mostrar.
    """
    frame = base_frame.copy()

    if len(points) >= 3:
        overlay = frame.copy()
        pts_arr = np.array(points, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_arr], _ROI_COLOR)
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    for i in range(len(points) - 1):
        cv2.line(frame, points[i], points[i + 1], _ROI_COLOR, 2)
    if len(points) >= 3:
        _draw_dashed_line(frame, points[-1], points[0], _ROI_COLOR, 2)

    if points:
        cv2.line(frame, points[-1], mouse_pos, (0, 200, 255), 1)

    for i, pt in enumerate(points, start=1):
        cv2.circle(frame, pt, 6, (0, 140, 255), -1)
        cv2.putText(frame, str(i), (pt[0] + 8, pt[1] - 8), _FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    area_pct = _polygon_area_px(points, width, height) * 100
    cv2.putText(frame, f"Vertices: {len(points)}  |  Area: {area_pct:.1f}%",
                (10, 30), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    if intersects:
        cv2.putText(frame, "¡El poligono se autointerseca!", (10, 60), _FONT, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    help_lines = [
        "[clic izq] agregar vertice   [clic der]/[z] deshacer   [c] reiniciar",
        "[t] alternar mapa de calor   [Enter] confirmar (min. 3 vertices)   [q/ESC] salir",
    ]
    y = height - 20 * len(help_lines) - 10
    for line in help_lines:
        cv2.putText(frame, line, (10, y), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20

    return frame


def _run_polygon_editor(
    background_frame: np.ndarray, heatmap_frame: Optional[np.ndarray], width: int, height: int, video_label: str,
) -> List[Tuple[int, int]]:
    """
    Corre el bucle interactivo de marcado del polígono de la ROI.

    Controles: clic izquierdo agrega un vértice, clic derecho / `z` deshace
    el último, `c` reinicia, `t` alterna el mapa de calor, Enter confirma
    (requiere >= 3 vértices y que el polígono no se autointerseque), `q`/ESC
    sale pidiendo confirmación en consola.

    Parámetros:
        background_frame (numpy.ndarray): frame de fondo limpio.
        heatmap_frame (numpy.ndarray | None): frame de fondo con el mapa de
            calor mezclado, o None si no se calculó/estaba vacío.
        width (int): ancho del frame, en píxeles.
        height (int): alto del frame, en píxeles.
        video_label (str): nombre descriptivo para el título de la ventana.

    Retorna:
        list[tuple[int, int]]: vértices confirmados, en píxeles.
    """
    points: List[Tuple[int, int]] = []
    mouse_pos = [width // 2, height // 2]
    show_heatmap = heatmap_frame is not None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    window_name = f"Marcar ROI - {video_label}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            base = heatmap_frame if (show_heatmap and heatmap_frame is not None) else background_frame
            intersects = _polygon_self_intersects(points)
            frame = _render_polygon_overlay(base, points, tuple(mouse_pos), intersects, width, height)
            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Ventana cerrada sin guardar.")
                sys.exit(0)

            if key in (ord('q'), 27):
                answer = input("\n¿Salir sin guardar la ROI? [s/N]: ").strip().lower()
                if answer in ("s", "si", "sí", "y", "yes"):
                    sys.exit(0)
            elif key == ord('z'):
                if points:
                    points.pop()
            elif key == ord('c'):
                points.clear()
            elif key == ord('t') and heatmap_frame is not None:
                show_heatmap = not show_heatmap
            elif key in (13, 10):
                if len(points) < 3:
                    print("  Necesitás al menos 3 vértices para confirmar.")
                    continue
                if intersects:
                    print("  El polígono se autointerseca; corregilo antes de confirmar.")
                    continue
                area_ratio = _polygon_area_px(points, width, height)
                if area_ratio < 0.02 or area_ratio > 0.95:
                    print(
                        f"  Advertencia: el área cubierta es {area_ratio * 100:.1f}% del frame — "
                        f"¿es correcto? (esto no bloquea, solo avisa)"
                    )
                return list(points)
    finally:
        cv2.destroyWindow(window_name)


def _run_homography_editor(
    background_frame: np.ndarray, roi_points: List[Tuple[int, int]], width: int, height: int,
) -> Optional[Dict[str, Any]]:
    """
    Corre una segunda ronda de marcado, pidiendo exactamente 4 puntos
    coplanares sobre el asfalto para la calibración de homografía futura.

    Parámetros:
        background_frame (numpy.ndarray): frame de fondo limpio.
        roi_points (list[tuple[int, int]]): polígono de la ROI ya confirmado,
            dibujado de fondo como referencia.
        width (int): ancho del frame.
        height (int): alto del frame.

    Retorna:
        dict | None: `{"image_points": [...norm...], "world_points": None,
        "notes": "..."}`, o None si el usuario cancela.
    """
    labels = ["A", "B", "C", "D"]
    points: List[Tuple[int, int]] = []
    mouse_pos = [width // 2, height // 2]

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        mouse_pos[0], mouse_pos[1] = x, y
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    window_name = "Puntos de homografía (4 esquinas sobre el asfalto)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    try:
        while True:
            frame = background_frame.copy()
            if len(roi_points) >= 3:
                cv2.polylines(frame, [np.array(roi_points, dtype=np.int32)], True, _ROI_COLOR, 1)

            for i, pt in enumerate(points):
                cv2.circle(frame, pt, 7, _HOMOGRAPHY_COLOR, -1)
                cv2.putText(frame, labels[i], (pt[0] + 8, pt[1] - 8), _FONT, 0.6, _HOMOGRAPHY_COLOR, 2, cv2.LINE_AA)

            cv2.putText(
                frame, f"Puntos de homografia: {len(points)}/4", (10, 30),
                _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )
            help_lines = [
                "Marca 4 puntos coplanares sobre el asfalto (esquinas de marcas",
                "viales o del cruce peatonal). [clic der]/[z] deshacer  [q/ESC] omitir",
            ]
            y = height - 20 * len(help_lines) - 10
            for line in help_lines:
                cv2.putText(frame, line, (10, y), _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                y += 20

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("  Homografía omitida (ventana cerrada).")
                return None

            if key in (ord('q'), 27):
                print("  Homografía omitida por el usuario.")
                return None
            if key == ord('z') and points:
                points.pop()
            if len(points) == 4:
                cv2.imshow(window_name, frame)
                cv2.waitKey(300)
                image_points = [[x / width, y / height] for x, y in points]
                return {
                    "image_points": image_points,
                    "world_points": None,
                    "notes": "Pendiente: completar world_points en la fase de calibración.",
                }
    finally:
        cv2.destroyWindow(window_name)


def _save_reference_thumbnail(roi_json_path: str, frame: np.ndarray) -> None:
    """
    Guarda el thumbnail en escala de grises usado para comparar (con
    tolerancia) si la cámara se movió, junto al JSON de la ROI.

    Parámetros:
        roi_json_path (str): ruta del archivo `roi.json`.
        frame (numpy.ndarray): frame de referencia usado al marcar la ROI.

    Retorna:
        None
    """
    thumb_path = reference_frame_path(roi_json_path)
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(thumb_path, gray)


def _run_show_mode(args: argparse.Namespace, config: dict, roi_path: str) -> None:
    """
    Implementa `--show`: carga la ROI existente, la dibuja sobre un frame de
    la escena y verifica (con tolerancia) si la cámara se movió desde que se
    calibró.

    Retorna:
        None
    """
    try:
        roi_data = load_roi(roi_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    if args.video:
        videos = resolve_videos(args.video, config["paths"]["videos"])
    else:
        candidates = [v for v in roi_data.get("source_videos", []) if os.path.isfile(v)]
        videos = candidates or resolve_videos(None, config["paths"]["videos"])

    width, height = validate_same_resolution(videos)
    frame = get_background_frame(videos[0], args.frame_sec)

    roi_filter = ROIFilter(roi_data, (width, height), config)
    annotated = roi_filter.draw(frame)
    area_pct = roi_filter.area_ratio() * 100
    cv2.putText(
        annotated, f"ROI: {area_pct:.1f}% del frame -- cualquier tecla para salir",
        (10, 30), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
    )

    stored_md5 = roi_data.get("reference_frame_md5")
    thumb_path = reference_frame_path(roi_path)
    current_md5 = md5_of_frame(frame)

    if not stored_md5 or not os.path.isfile(thumb_path):
        print("  No hay frame de referencia guardado; no se puede verificar si la cámara se movió.")
    elif current_md5 == stored_md5:
        print("  El frame actual coincide exactamente con el de referencia usado al marcar la ROI.")
    else:
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        current_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        ref_gray = cv2.imread(thumb_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        mad = float(np.mean(np.abs(current_gray - ref_gray)))
        if mad > _MAD_WARNING_THRESHOLD:
            print(
                f"  ⚠ El frame actual difiere bastante del de referencia (diferencia media: "
                f"{mad:.1f}/255). La cámara podría haberse movido: considerá volver a marcar la ROI."
            )
        else:
            print(f"  El frame actual es razonablemente similar al de referencia (diferencia media: {mad:.1f}/255).")

    if _is_headless():
        print("✗ No se detectó un display disponible para mostrar la ROI.")
        sys.exit(1)

    window_name = f"ROI guardada - {os.path.basename(videos[0])}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    """
    Punto de entrada principal: resuelve videos, calcula el mapa de calor
    (salvo `--no-heatmap`), corre el marcado interactivo del polígono (y
    opcionalmente los puntos de homografía), y guarda el resultado.

    Retorna:
        None
    """
    args = parse_args()
    config = load_config(args.config)
    setup_roi_logging(config["paths"]["results"])

    roi_cfg = config.setdefault("roi", {})
    heatmap_cfg = roi_cfg.setdefault("heatmap", {})
    output_path = args.output or roi_cfg.get("file", "config/roi.json")

    if args.show:
        _run_show_mode(args, config, output_path)
        return

    videos = resolve_videos(args.video, config["paths"]["videos"])
    width, height = validate_same_resolution(videos)
    print("Videos a usar:")
    for v in videos:
        print(f"  - {v}")

    background_frame = get_background_frame(videos[0], args.frame_sec)
    reference_md5 = md5_of_frame(background_frame)

    heatmap_frame = None
    heatmap_obj: Optional[TrafficHeatmap] = None
    heatmap_path = None

    if not args.no_heatmap:
        if args.samples is not None:
            heatmap_cfg["samples"] = args.samples
        if args.sample_mode is not None:
            heatmap_cfg["sample_mode"] = args.sample_mode
        if args.device is not None:
            config.setdefault("benchmark", {})["device"] = args.device

        model_id = args.model or heatmap_cfg.get("model", "yolov8n")
        model_entry = resolve_model_entry(config, model_id)

        print(f"\nCalculando mapa de calor con {model_id} sobre {len(videos)} video(s)...")
        detector = VehicleDetector(model_entry["id"], model_entry["weights"], config)
        heatmap_obj = TrafficHeatmap(config, detector)
        heatmap_obj.accumulate_many(videos)

        if heatmap_obj.is_empty():
            print(
                "\n⚠ No se detectó ningún vehículo en los frames muestreados para el mapa de calor.\n"
                "  Sugerencias: subí --samples, probá un modelo más preciso con --model, o agregá más videos.\n"
                "  Se continúa con el frame limpio para marcar la ROI a mano."
            )
        else:
            heatmap_frame = heatmap_obj.render(background_frame)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            heatmap_path = os.path.join("results", "preview", f"heatmap_{timestamp}.png")
            heatmap_obj.save(heatmap_path, background_frame)

    if _is_headless():
        print(
            "✗ No se detectó un display disponible. La ROI debe marcarse de forma interactiva en una "
            "máquina con interfaz gráfica. Copiá luego el archivo roi.json generado a la Raspberry Pi."
        )
        sys.exit(1)

    polygon_px = _run_polygon_editor(background_frame, heatmap_frame, width, height, os.path.basename(videos[0]))
    polygon_norm = [[x / width, y / height] for x, y in polygon_px]
    area_pct = polygon_area_ratio(polygon_norm) * 100

    homography_points = None
    if args.with_homography:
        homography_points = _run_homography_editor(background_frame, polygon_px, width, height)

    save_roi(output_path, polygon_norm, (width, height), videos, reference_md5, homography_points)
    _save_reference_thumbnail(output_path, background_frame)

    print(f"\n✓ ROI guardada en {output_path}")
    print(f"  Vértices: {len(polygon_px)}")
    print(f"  Área cubierta: {area_pct:.1f}% del frame")
    print(f"  Videos usados para el mapa: {heatmap_obj.stats()['videos_processed'] if heatmap_obj else 0}")
    print(f"  Detecciones acumuladas: {heatmap_obj.stats()['detections_total'] if heatmap_obj else 0}")
    if heatmap_path:
        print(f"  Mapa de calor: {heatmap_path}")
    if homography_points:
        print(f"  Puntos de homografía: {len(homography_points['image_points'])} (world_points pendientes)")

    print("\nPara usarla:")
    print("  python scripts/preview_detection.py --roi")
    print("  python scripts/run_benchmark.py --roi")


if __name__ == "__main__":
    main()

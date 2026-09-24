#!/usr/bin/env python3
"""
Calibración por homografía: obtiene la matriz que convierte píxeles de la
cámara a metros del mundo real, y cuantifica qué tan confiable es.

Esta fase **no calcula velocidades** — solo produce y verifica la
transformación de píxeles a metros. Una calibración silenciosamente mala
produce velocidades que parecen razonables pero están sistemáticamente
equivocadas.

Tres formas de obtener los puntos de correspondencia:
    A) Con imagen aérea (si hay MAPTILER_KEY configurada): marcado por pares
       cámara <-> aérea, con error de reproyección en vivo.
    B) Manual: clic sobre la cámara, lat/lon tecleada en consola.
    C) Desde un CSV versionado y reproducible (`--points`).

Uso:
    python scripts/calibrate.py --fetch-aerial --center 9.8574,-83.9112
    python scripts/calibrate.py
    python scripts/calibrate.py --points config/points.csv
    python scripts/calibrate.py --verify
    python scripts/calibrate.py --show
"""
import argparse
import csv
import glob
import logging
import os
import sys
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calibration.aerial import AerialImage
from src.calibration.geo import LocalENU
from src.calibration.homography import CalibratedPlane, HomographyCalibration
from src.calibration.report import render_birdseye, render_residuals, render_scale_map
from src.utils.roi import md5_of_frame

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_STATUS_RANK = {"ok": 0, "warn": 1, "reject": 2}


def setup_calibration_logging(results_dir: str) -> None:
    """
    Configura el logging en nivel INFO para la calibración, con salida a
    consola y a `calibration.log` dentro de `results_dir`.

    Parámetros:
        results_dir (str): directorio donde se guardará el log.

    Retorna:
        None
    """
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "calibration.log")

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
        description="Calibración por homografía: convierte píxeles de la cámara a metros del mundo real."
    )
    parser.add_argument("--video", type=str, default=None, help="Video del que se toma el frame de referencia. Default: primer .mp4 en data/videos/.")
    parser.add_argument("--frame-sec", type=float, default=None, help="Segundo del frame de referencia. Default: el frame medio del video.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Ruta al archivo de configuración (default: config/config.yaml).")
    parser.add_argument("--points", type=str, default=None, help="CSV de puntos ya preparado; salta el marcado interactivo.")
    parser.add_argument("--aerial", type=str, default=None, help="Imagen aérea ya descargada (con su .meta.json).")
    parser.add_argument("--fetch-aerial", action="store_true", help="Descarga la imagen aérea vía MapTiler.")
    parser.add_argument("--center", type=str, default=None, help="lat,lon del centro para la descarga aérea.")
    parser.add_argument("--zoom", type=int, default=20, help="Zoom de la descarga aérea (default: 20).")
    parser.add_argument("--method", type=str, default=None, choices=["least_squares", "ransac"], help="Método de ajuste. Default: calibration.fit_method del config.")
    parser.add_argument("--output", type=str, default=None, help="Dónde guardar el JSON. Default: calibration.file del config.")
    parser.add_argument("--verify", action="store_true", help="Modo verificación sobre una calibración existente.")
    parser.add_argument("--show", action="store_true", help="Muestra la calibración guardada y sus reportes.")
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


def find_video(videos_dir: str) -> Optional[str]:
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


def _is_headless() -> bool:
    """
    Determina si el entorno actual probablemente no tiene display disponible.

    Retorna:
        bool: True si no parece haber display disponible.
    """
    if os.name == "posix" and os.uname().sysname != "Darwin":
        if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
            return True
    try:
        cv2.namedWindow("_calibrate_headless_probe", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("_calibrate_headless_probe")
        return False
    except cv2.error:
        return True


def get_reference_frame(video_path: str, frame_sec: Optional[float]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Extrae el frame de referencia de un video, sobre el que se marcarán los
    puntos de calibración.

    Parámetros:
        video_path (str): ruta al video.
        frame_sec (float | None): segundo del que tomar el frame. Si es
            None, se usa el frame medio del video.

    Retorna:
        tuple[numpy.ndarray, tuple[int, int]]: `(frame, (ancho, alto))`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"✗ No se pudo abrir el video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
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

    return frame, (width, height)


# ----------------------------------------------------------------------
# Modo C: CSV
# ----------------------------------------------------------------------

def load_points_csv(path: str) -> List[Dict[str, Any]]:
    """
    Carga puntos de correspondencia desde un CSV con columnas
    `label,px,py,lat,lon,note`. Ignora líneas vacías o que empiezan con `#`.

    Es el modo que hace reproducible la calibración: el CSV se versiona en
    el repositorio.

    Parámetros:
        path (str): ruta al CSV.

    Retorna:
        list[dict]: puntos con `label`, `pixel`, `latlon`, `note`.
    """
    if not os.path.isfile(path):
        print(f"✗ No se encontró el CSV de puntos: '{path}'.")
        sys.exit(1)

    with open(path, "r", encoding="utf-8", newline="") as f:
        raw_lines = f.readlines()

    header: Optional[List[str]] = None
    data_rows: List[Tuple[int, str]] = []
    for line_num, raw in enumerate(raw_lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if header is None:
            header = next(csv.reader([raw]))
            continue
        data_rows.append((line_num, raw))

    required_cols = ["label", "px", "py", "lat", "lon"]
    if header is None or not all(col in header for col in required_cols):
        print(f"✗ El CSV '{path}' debe tener las columnas: label,px,py,lat,lon,note. Encontradas: {header}")
        sys.exit(1)

    points: List[Dict[str, Any]] = []
    for line_num, raw in data_rows:
        values = next(csv.reader([raw]))
        row = dict(zip(header, values))
        label = (row.get("label") or "").strip()
        if not label:
            continue

        lat_str, lon_str = (row.get("lat") or "").strip(), (row.get("lon") or "").strip()
        ok, msg = LocalENU.validate_precision(lat_str, lon_str)
        if not ok:
            print(f"✗ Línea {line_num} del CSV '{path}' (punto {label}): {msg}")
            sys.exit(1)

        try:
            px, py = float(row["px"]), float(row["py"])
        except (TypeError, ValueError, KeyError):
            print(f"✗ Línea {line_num} del CSV '{path}' (punto {label}): coordenadas de píxel inválidas.")
            sys.exit(1)

        points.append({
            "label": label,
            "pixel": [px, py],
            "latlon": [float(lat_str), float(lon_str)],
            "note": (row.get("note") or "").strip(),
        })

    if len(points) < 4:
        print(f"✗ El CSV '{path}' tiene {len(points)} punto(s) válido(s); se requieren al menos 4.")
        sys.exit(1)

    return points


# ----------------------------------------------------------------------
# Modo B: manual, sin imagen aérea
# ----------------------------------------------------------------------

def run_manual_mode(frame: np.ndarray, width: int, height: int) -> List[Dict[str, Any]]:
    """
    Marcado manual: clic sobre el frame de la cámara, luego la consola pide
    latitud y longitud de ese punto, validadas con `LocalENU.validate_precision`.

    Controles: `z` deshace el último punto, Enter confirma (mínimo 4
    puntos), `q`/ESC sale pidiendo confirmación.

    Parámetros:
        frame (numpy.ndarray): frame de la cámara sobre el que marcar.
        width, height (int): dimensiones del frame.

    Retorna:
        list[dict]: puntos con `label`, `pixel`, `latlon`, `note`.
    """
    points: List[Dict[str, Any]] = []
    pending_click: List[Optional[Tuple[int, int]]] = [None]

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click[0] = (x, y)

    window_name = "Marcar puntos de calibracion (clic sobre el asfalto)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nModo manual: hacé clic sobre un punto de referencia en la imagen; luego se te pedirá")
    print("su latitud y longitud en la consola.")
    print("[z] deshacer el último punto   [Enter] terminar (mínimo 4 puntos)   [q/ESC] salir\n")

    try:
        while True:
            disp = frame.copy()
            for p in points:
                x, y = p["pixel"]
                cv2.circle(disp, (int(x), int(y)), 6, (0, 140, 255), -1)
                cv2.putText(disp, p["label"], (int(x) + 8, int(y) - 8), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(disp, f"Puntos: {len(points)}", (10, 30), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow(window_name, disp)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Ventana cerrada sin guardar.")
                sys.exit(0)

            if key in (ord('q'), 27):
                answer = input("\n¿Salir sin guardar? [s/N]: ").strip().lower()
                if answer in ("s", "si", "sí", "y", "yes"):
                    sys.exit(0)
                continue
            if key == ord('z'):
                if points:
                    removed = points.pop()
                    print(f"  Deshecho: punto {removed['label']}.")
                continue
            if key in (13, 10):
                if len(points) < 4:
                    print("  Necesitás al menos 4 puntos para confirmar.")
                    continue
                break

            if pending_click[0] is not None:
                x, y = pending_click[0]
                pending_click[0] = None
                label = _LABELS[len(points)]
                print(f"\nPunto {label} marcado en píxel ({x}, {y}).")
                while True:
                    lat_str = input(f"  Latitud de {label}: ").strip()
                    lon_str = input(f"  Longitud de {label}: ").strip()
                    ok, msg = LocalENU.validate_precision(lat_str, lon_str)
                    print(f"  {msg}")
                    if ok:
                        break
                note = input(f"  Nota descriptiva para {label} (opcional, Enter para omitir): ").strip()
                points.append({
                    "label": label, "pixel": [float(x), float(y)],
                    "latlon": [float(lat_str), float(lon_str)], "note": note,
                })
    finally:
        cv2.destroyWindow(window_name)

    return points


# ----------------------------------------------------------------------
# Modo A: marcado con imagen aérea
# ----------------------------------------------------------------------

def _draw_magnifier(
    canvas: np.ndarray, mouse_pos: List[Any], cam_w: int, cam_scale: float, aer_scale: float,
    frame: np.ndarray, aerial_img: np.ndarray, zoom: int = 4, size: int = 140,
) -> None:
    """
    Dibuja una lupa con la zona bajo el cursor, ampliada, en la esquina
    superior derecha del canvas. Sin esto, la precisión del clic domina el
    error total de la calibración.
    """
    x_disp, y_disp, side = mouse_pos
    if side == "cam":
        src, sx, sy = frame, x_disp / cam_scale, y_disp / cam_scale
    else:
        src, sx, sy = aerial_img, (x_disp - cam_w) / aer_scale, y_disp / aer_scale

    half = max(size // (2 * zoom), 4)
    x0, y0 = int(sx - half), int(sy - half)
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    h, w = src.shape[:2]
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return

    crop = src[y0c:y1c, x0c:x1c]
    mag = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.line(mag, (size // 2 - 8, size // 2), (size // 2 + 8, size // 2), (0, 255, 0), 1)
    cv2.line(mag, (size // 2, size // 2 - 8), (size // 2, size // 2 + 8), (0, 255, 0), 1)
    cv2.rectangle(mag, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)

    px, py = canvas.shape[1] - size - 10, 10
    canvas[py:py + size, px:px + size] = mag


def run_aerial_mode(
    frame: np.ndarray, aerial: AerialImage, width: int, height: int,
) -> List[Dict[str, Any]]:
    """
    Marcado por pares: ventana única con la cámara a la izquierda y la
    imagen aérea a la derecha. Los clics se alternan por pares (cámara,
    luego aérea); cada par forma un punto de correspondencia.

    Desde el quinto par, recalcula y muestra en pantalla el error de
    reproyección medio actual, resaltando en rojo el par de peor residuo:
    si se marcó mal una esquina, se ve en el momento, no media hora después.

    Controles: `z` deshace, `n` anota el último par, Enter termina (mínimo
    4 pares), `q`/ESC sale pidiendo confirmación.

    Parámetros:
        frame (numpy.ndarray): frame de la cámara.
        aerial (AerialImage): imagen aérea georreferenciada.
        width, height (int): dimensiones del frame de la cámara.

    Retorna:
        list[dict]: puntos con `label`, `pixel`, `latlon`, `note`.
    """
    aerial_img = cv2.imread(aerial.image_path)
    if aerial_img is None:
        print(f"✗ No se pudo leer la imagen aérea: '{aerial.image_path}'")
        sys.exit(1)

    panel_h = min(height, aerial_img.shape[0], 900)
    cam_scale = panel_h / height
    aer_scale = panel_h / aerial_img.shape[0]
    cam_w = int(width * cam_scale)
    aer_w = int(aerial_img.shape[1] * aer_scale)

    cam_disp_base = cv2.resize(frame, (cam_w, panel_h))
    aer_disp_base = cv2.resize(aerial_img, (aer_w, panel_h))

    pairs: List[Dict[str, Any]] = []
    pending_cam: Optional[Tuple[float, float]] = None
    click_event: List[Optional[Tuple[str, int, int]]] = [None]
    mouse_pos: List[Any] = [0, 0, "cam"]
    calib_preview: Optional[Tuple[float, int]] = None

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        side = "cam" if x < cam_w else "aer"
        mouse_pos[0], mouse_pos[1], mouse_pos[2] = x, y, side
        if event == cv2.EVENT_LBUTTONDOWN:
            click_event[0] = (side, x if side == "cam" else x - cam_w, y)

    window_name = "Calibracion: camara (izq.) | imagen aerea (der.)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nModo aéreo: hacé clic alternando cámara -> aérea para formar cada par de puntos.")
    print("[z] deshacer   [n] anotar el último par   [Enter] terminar (mínimo 4 pares)   [q/ESC] salir\n")

    try:
        while True:
            canvas = np.zeros((panel_h, cam_w + aer_w, 3), dtype=np.uint8)
            canvas[:, :cam_w] = cam_disp_base
            canvas[:, cam_w:] = aer_disp_base
            cv2.line(canvas, (cam_w, 0), (cam_w, panel_h), (255, 255, 255), 1)

            worst_idx = calib_preview[1] if calib_preview else None
            for i, pair in enumerate(pairs):
                color = (0, 0, 255) if i == worst_idx else (0, 140, 255)
                cx, cy = pair["cam_px"]
                ax, ay = pair["aer_px"]
                cv2.circle(canvas, (int(cx * cam_scale), int(cy * cam_scale)), 6, color, -1)
                cv2.putText(canvas, pair["label"], (int(cx * cam_scale) + 8, int(cy * cam_scale) - 8), _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(canvas, (cam_w + int(ax * aer_scale), int(ay * aer_scale)), 6, color, -1)
                cv2.putText(canvas, pair["label"], (cam_w + int(ax * aer_scale) + 8, int(ay * aer_scale) - 8), _FONT, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

            if pending_cam is not None:
                cx, cy = pending_cam
                cv2.circle(canvas, (int(cx * cam_scale), int(cy * cam_scale)), 6, (0, 255, 255), -1)

            status = f"Pares: {len(pairs)}   Siguiente clic: {'AEREA' if pending_cam is not None else 'CAMARA'}"
            cv2.putText(canvas, status, (10, 25), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            if calib_preview:
                cv2.putText(canvas, f"Error medio actual: {calib_preview[0]:.2f} m", (10, 50), _FONT, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            _draw_magnifier(canvas, mouse_pos, cam_w, cam_scale, aer_scale, frame, aerial_img)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                print("Ventana cerrada sin guardar.")
                sys.exit(0)
            if key in (ord('q'), 27):
                answer = input("\n¿Salir sin guardar? [s/N]: ").strip().lower()
                if answer in ("s", "si", "sí", "y", "yes"):
                    sys.exit(0)
                continue
            if key == ord('z'):
                if pending_cam is not None:
                    pending_cam = None
                elif pairs:
                    removed = pairs.pop()
                    print(f"  Deshecho: par {removed['label']}.")
                    calib_preview = None
                continue
            if key == ord('n') and pairs:
                note = input(f"  Nota para el punto {pairs[-1]['label']}: ").strip()
                pairs[-1]["note"] = note
                continue
            if key in (13, 10):
                if len(pairs) < 4:
                    print("  Necesitás al menos 4 pares para confirmar.")
                    continue
                break

            if click_event[0] is not None:
                side, x_disp, y_disp = click_event[0]
                click_event[0] = None

                if side == "cam":
                    pending_cam = (x_disp / cam_scale, y_disp / cam_scale)
                elif side == "aer" and pending_cam is not None:
                    aer_px = (x_disp / aer_scale, y_disp / aer_scale)
                    lat, lon = aerial.pixel_to_latlon(*aer_px)
                    label = _LABELS[len(pairs)]
                    pairs.append({
                        "label": label, "cam_px": pending_cam, "aer_px": aer_px,
                        "latlon": [lat, lon], "note": "",
                    })
                    print(f"  Par {label}: cámara({pending_cam[0]:.0f},{pending_cam[1]:.0f}) -> "
                          f"lat/lon ({lat:.7f}, {lon:.7f})")
                    pending_cam = None

                    if len(pairs) >= 5:
                        try:
                            origin_lat, origin_lon = pairs[0]["latlon"]
                            enu = LocalENU(origin_lat, origin_lon)
                            trial_world = [enu.to_meters(*p["latlon"]) for p in pairs]
                            trial_px = [p["cam_px"] for p in pairs]
                            trial_calib = HomographyCalibration(trial_px, trial_world, (width, height))
                            trial_calib.fit(method="least_squares")
                            res = trial_calib.residuals()
                            calib_preview = (float(np.mean(res)), int(np.argmax(res)))
                        except ValueError:
                            calib_preview = None
    finally:
        cv2.destroyWindow(window_name)

    return [
        {"label": p["label"], "pixel": [p["cam_px"][0], p["cam_px"][1]], "latlon": p["latlon"], "note": p.get("note", "")}
        for p in pairs
    ]


# ----------------------------------------------------------------------
# Flujo común: ajuste, verificación, reportes, guardado
# ----------------------------------------------------------------------

def _classify(value: float, ok_th: float, warn_th: float) -> str:
    """Clasifica un valor contra dos umbrales: 'ok', 'warn' o 'reject'."""
    if value <= ok_th:
        return "ok"
    if value <= warn_th:
        return "warn"
    return "reject"


def _print_diagnosis(quality: Dict[str, Any], loo: Dict[str, Any], scale: Dict[str, Any], thresholds: Dict[str, Any]) -> None:
    """
    Imprime pistas de diagnóstico accionables según la combinación de
    resultados obtenida (ver guía completa en el README).
    """
    mean_status = _classify(quality["reprojection_mean_m"], thresholds["reprojection_ok_m"], thresholds["reprojection_warn_m"])
    loo_status = _classify(loo["mean_m"], thresholds["loo_ok_m"], thresholds["loo_warn_m"]) if loo.get("available") else None

    print()
    if loo_status is not None and mean_status == "ok" and loo_status in ("warn", "reject"):
        print("  ⚠ Diagnóstico: error de reproyección bajo pero validación cruzada alta -> la calibración")
        print("    está sobreajustada (overfitting) y no es confiable pese al buen ajuste sobre los mismos puntos.")

    median = quality["reprojection_median_m"]
    gap = quality["reprojection_max_m"] - median
    if gap > 2 * max(median, 0.01) and quality["reprojection_max_m"] > thresholds["reprojection_ok_m"]:
        print(f"  ⚠ Diagnóstico: el punto {quality['worst_point_label']} tiene un residuo muy superior al resto "
              f"({quality['worst_point_residual_m']:.2f} m vs. mediana {median:.2f} m).")
        print("    Revisá su coordenada (¿decimal mal copiado?) o si la esquina está mal emparejada entre las dos vistas.")
    elif mean_status in ("warn", "reject"):
        print("  ⚠ Diagnóstico: todos los residuos son altos y parecidos -> revisá si algún punto está fuera del")
        print("    plano de la calzada, o si el sistema de coordenadas está mal construido.")

    max_scale_cm = thresholds.get("max_scale_cm_per_px")
    if max_scale_cm and scale["max_m_per_px"] * 100 > max_scale_cm:
        print(f"  ⚠ Diagnóstico: la escala en la zona más lejana ({scale['max_m_per_px'] * 100:.1f} cm/px) supera "
              f"el umbral configurado ({max_scale_cm:.1f} cm/px): esa franja no sirve para medir velocidad.")
        print("    Acercá la zona de medición o reubicá la cámara.")


def _print_summary_box(quality: Dict[str, Any], loo: Dict[str, Any], scale: Dict[str, Any], thresholds: Dict[str, Any]) -> str:
    """Imprime el resumen final en un cuadro de consola. Retorna el estado global ('ok'/'warn'/'reject')."""
    mean_status = _classify(quality["reprojection_mean_m"], thresholds["reprojection_ok_m"], thresholds["reprojection_warn_m"])
    loo_status = _classify(loo["mean_m"], thresholds["loo_ok_m"], thresholds["loo_warn_m"]) if loo.get("available") else "ok"
    overall = max((mean_status, loo_status), key=lambda s: _STATUS_RANK[s])

    if overall == "ok":
        status_line = "✓ Calibración aceptable"
    elif overall == "warn":
        status_line = f"⚠ Revisar el punto {quality['worst_point_label']} antes de continuar"
    else:
        status_line = "✗ NO continuar a la fase de velocidad -- revisar puntos"

    width = 62

    def line(text: str = "") -> str:
        return "║" + text.ljust(width)[:width] + "║"

    print("\n╔" + "═" * width + "╗")
    print(line("  Calibración por homografía"))
    print("╠" + "═" * width + "╣")
    print(line(f"  Puntos utilizados:              {quality['n_points']}"))
    print(line(f"  Error de reproyección (media):  {quality['reprojection_mean_m']:.2f} m"))
    print(line(f"  Error de reproyección (máx):    {quality['reprojection_max_m']:.2f} m   <- punto {quality['worst_point_label']}"))
    if loo.get("available"):
        print(line(f"  Validación cruzada (media):     {loo['mean_m']:.2f} m"))
    else:
        print(line("  Validación cruzada:             no disponible (< 5 puntos)"))
    print(line(f"  Cobertura del casco convexo:    {quality['hull_area_ratio'] * 100:.1f}% del frame"))
    print(line(f"  Escala:  {scale['min_m_per_px'] * 100:.1f} cm/px (cerca)  ->  {scale['max_m_per_px'] * 100:.1f} cm/px (lejos)"))
    print("╠" + "═" * width + "╣")
    print(line(f"  {status_line}"))
    print("╚" + "═" * width + "╝")

    return overall


def run_common_pipeline(
    points: List[Dict[str, Any]], frame: np.ndarray, frame_size: Tuple[int, int],
    video_path: str, config: dict, args: argparse.Namespace, output_path: str,
    aerial_meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Flujo común a los tres modos de obtención de puntos: convierte a
    metros, ajusta la homografía, calcula calidad y validación cruzada,
    genera los reportes gráficos y guarda el JSON de calibración.

    Retorna:
        None
    """
    if len(points) < 4:
        print(f"✗ Se requieren al menos 4 puntos; se obtuvieron {len(points)}.")
        sys.exit(1)

    origin_lat, origin_lon = points[0]["latlon"]
    enu = LocalENU(origin_lat, origin_lon)
    for p in points:
        lat, lon = p["latlon"]
        p["world_m"] = list(enu.to_meters(lat, lon))

    labels = [p["label"] for p in points]
    n = len(points)
    print("\nDistancias entre puntos (chequeo de cordura, calculadas directamente de lat/lon):")
    for i in range(n):
        for j in range(i + 1, n):
            lat1, lon1 = points[i]["latlon"]
            lat2, lon2 = points[j]["latlon"]
            d = LocalENU.haversine(lat1, lon1, lat2, lon2)
            print(f"  {labels[i]}-{labels[j]}: {d:.2f} m")

    calib_cfg = config["calibration"]
    method = args.method or calib_cfg.get("fit_method", "least_squares")
    ransac_threshold = calib_cfg.get("ransac_threshold_m", 0.5)

    calib = HomographyCalibration([p["pixel"] for p in points], [p["world_m"] for p in points], frame_size)
    try:
        calib.fit(method=method, ransac_threshold_m=ransac_threshold)
    except ValueError as exc:
        print(f"\n✗ {exc}")
        sys.exit(1)

    for p, r in zip(points, calib.residuals()):
        p["residual_m"] = float(r)
        p["used_in_fit"] = True

    quality = calib.quality(labels=labels)
    loo = calib.leave_one_out()
    scale = calib.scale_map()
    thresholds = calib_cfg.get("thresholds", {})

    _print_diagnosis(quality, loo, scale, thresholds)

    out_dir = calib_cfg.get("output_dir", "results/calibration/")
    os.makedirs(out_dir, exist_ok=True)
    scale_map_path = os.path.join(out_dir, "scale_map.png")
    birdseye_path = os.path.join(out_dir, "birdseye.png")
    residuals_path = os.path.join(out_dir, "residuals.png")

    birdseye_cfg = calib_cfg.get("birdseye", {})
    render_scale_map(frame, calib, scale_map_path)
    render_birdseye(frame, calib, birdseye_path, px_per_meter=birdseye_cfg.get("px_per_meter", 20), margin_m=birdseye_cfg.get("margin_m", 5))
    render_residuals(calib, residuals_path, labels=labels)

    metadata = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_video": video_path,
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "reference_frame_md5": md5_of_frame(frame),
        "geo_model": "local_enu_wgs84",
        "origin": {"lat": origin_lat, "lon": origin_lon, "label": labels[0]},
        "points": [
            {
                "label": p["label"],
                "pixel": [float(p["pixel"][0]), float(p["pixel"][1])],
                "latlon": [float(p["latlon"][0]), float(p["latlon"][1])],
                "world_m": [float(p["world_m"][0]), float(p["world_m"][1])],
                "residual_m": p["residual_m"],
                "used_in_fit": p["used_in_fit"],
                "note": p.get("note", ""),
            }
            for p in points
        ],
        "quality": {**quality, "leave_one_out": loo},
        "scale": {
            "min_cm_per_px": scale["min_m_per_px"] * 100,
            "max_cm_per_px": scale["max_m_per_px"] * 100,
            "at_centroid_cm_per_px": scale["at_centroid_m_per_px"] * 100,
        },
    }
    if aerial_meta:
        metadata["aerial"] = aerial_meta

    calib.save(output_path, metadata)
    _print_summary_box(quality, loo, scale, thresholds)

    print(f"\n✓ Calibración guardada en {output_path}")
    print(f"  Mapa de escala:  {scale_map_path}")
    print(f"  Vista cenital:   {birdseye_path}")
    print(f"  Residuos:        {residuals_path}")
    print("\nRevisá la vista cenital: si los bordes de la calle salen rectos y paralelos, y el ancho del")
    print("carril se mantiene constante de cerca a lejos, la calibración es correcta. Si convergen, se")
    print("curvan o el carril se ensancha con la distancia, está mal.")


# ----------------------------------------------------------------------
# --verify / --show
# ----------------------------------------------------------------------

def run_verify_mode(args: argparse.Namespace, config: dict, calibration_path: str) -> None:
    """
    Carga una calibración existente y permite hacer clic en dos puntos
    sobre el frame de la cámara para obtener la distancia métrica que la
    homografía afirma entre ellos: la comprobación de escala.

    Retorna:
        None
    """
    try:
        plane = CalibratedPlane.load(calibration_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    if _is_headless():
        print("✗ No se detectó un display disponible. --verify requiere marcar puntos interactivamente.")
        sys.exit(1)

    video_path = args.video or find_video(config["paths"]["videos"])
    if not video_path:
        print(f"✗ No se encontró ningún video en {config['paths']['videos']}. Indicá --video.")
        sys.exit(1)

    frame, _ = get_reference_frame(video_path, args.frame_sec)
    plane.check_frame(frame)

    cam_h, cam_w = frame.shape[:2]
    calib_cfg = config.get("calibration", {})
    birdseye_cfg = calib_cfg.get("birdseye", {})

    tmp_birdseye = os.path.join(tempfile.gettempdir(), "calibrate_verify_birdseye.png")
    try:
        render_birdseye(frame, plane, tmp_birdseye, px_per_meter=birdseye_cfg.get("px_per_meter", 20), margin_m=birdseye_cfg.get("margin_m", 5))
        birdseye_img = cv2.imread(tmp_birdseye)
    finally:
        if os.path.isfile(tmp_birdseye):
            os.remove(tmp_birdseye)

    panel_h = min(cam_h, 900)
    cam_scale = panel_h / cam_h
    cam_disp_w = int(cam_w * cam_scale)
    cam_disp = cv2.resize(frame, (cam_disp_w, panel_h))

    if birdseye_img is not None:
        bird_scale = panel_h / birdseye_img.shape[0]
        bird_disp_w = int(birdseye_img.shape[1] * bird_scale)
        bird_disp = cv2.resize(birdseye_img, (bird_disp_w, panel_h))
    else:
        bird_disp_w = cam_disp_w
        bird_disp = np.zeros((panel_h, bird_disp_w, 3), dtype=np.uint8)

    clicks: List[Tuple[float, float]] = []
    click_event: List[Optional[Tuple[float, float]]] = [None]

    def on_mouse(event: int, x: int, y: int, flags: int, userdata: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and x < cam_disp_w:
            click_event[0] = (x / cam_scale, y / cam_scale)

    window_name = "Verificar calibracion: camara (izq.) | vista cenital (der.)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nModo verificación: hacé clic en dos puntos sobre la CÁMARA (panel izquierdo) para medir la")
    print("distancia entre ellos. Compará el resultado contra una medición con cinta en el sitio, o")
    print("contra una distancia leída en Google Maps entre puntos distintos a los usados en el ajuste.")
    print("[c] reiniciar el par actual   [q/ESC] salir\n")

    verif_rows: List[Dict[str, Any]] = []
    try:
        while True:
            canvas = np.zeros((panel_h, cam_disp_w + bird_disp_w, 3), dtype=np.uint8)
            canvas[:, :cam_disp_w] = cam_disp
            canvas[:, cam_disp_w:] = bird_disp
            cv2.line(canvas, (cam_disp_w, 0), (cam_disp_w, panel_h), (255, 255, 255), 1)

            for x, y in clicks:
                cv2.circle(canvas, (int(x * cam_scale), int(y * cam_scale)), 6, (0, 255, 255), -1)

            if len(clicks) == 2:
                world_pts = plane.to_world(clicks)
                distance = float(np.linalg.norm(world_pts[0] - world_pts[1]))
                cv2.putText(canvas, f"Distancia: {distance:.2f} m", (10, 30), _FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord('q'), 27):
                break
            if key == ord('c'):
                clicks.clear()
                continue

            if click_event[0] is not None:
                pt = click_event[0]
                click_event[0] = None
                if len(clicks) >= 2:
                    clicks.clear()
                clicks.append(pt)
                if len(clicks) == 2:
                    world_pts = plane.to_world(clicks)
                    distance = float(np.linalg.norm(world_pts[0] - world_pts[1]))
                    print(f"  Distancia medida: {distance:.2f} m  "
                          f"(píxeles: ({clicks[0][0]:.0f},{clicks[0][1]:.0f}) <-> ({clicks[1][0]:.0f},{clicks[1][1]:.0f}))")
                    verif_rows.append({
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "px1": f"{clicks[0][0]:.0f},{clicks[0][1]:.0f}",
                        "px2": f"{clicks[1][0]:.0f},{clicks[1][1]:.0f}",
                        "distance_m": round(distance, 2),
                    })
    finally:
        cv2.destroyWindow(window_name)

    if verif_rows:
        verif_csv = os.path.join(calib_cfg.get("output_dir", "results/calibration/"), "verificaciones.csv")
        os.makedirs(os.path.dirname(verif_csv), exist_ok=True)
        write_header = not os.path.isfile(verif_csv)
        with open(verif_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "px1", "px2", "distance_m"])
            if write_header:
                writer.writeheader()
            writer.writerows(verif_rows)
        print(f"\n{len(verif_rows)} medición(es) guardada(s) en {verif_csv}")


def run_show_mode(args: argparse.Namespace, config: dict, calibration_path: str) -> None:
    """
    Muestra el resumen de calidad de una calibración ya guardada y abre sus
    reportes gráficos, si existen.

    Retorna:
        None
    """
    try:
        plane = CalibratedPlane.load(calibration_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    q = plane.quality
    print(f"\nCalibración cargada desde {calibration_path}")
    print(f"  Frame de referencia: {plane.frame_size[0]}x{plane.frame_size[1]}")
    if q:
        print(f"  Puntos utilizados: {q.get('n_points')}")
        print(f"  Error de reproyección (media): {q.get('reprojection_mean_m', float('nan')):.3f} m")
        print(f"  Error de reproyección (máx):   {q.get('reprojection_max_m', float('nan')):.3f} m  <- punto {q.get('worst_point_label')}")
        loo = q.get("leave_one_out", {})
        if loo.get("available"):
            print(f"  Validación cruzada (media):    {loo.get('mean_m'):.3f} m")
        for w in q.get("warnings", []):
            print(f"  ⚠ {w}")

    out_dir = config.get("calibration", {}).get("output_dir", "results/calibration/")
    report_paths = {
        "Mapa de escala": os.path.join(out_dir, "scale_map.png"),
        "Vista cenital": os.path.join(out_dir, "birdseye.png"),
        "Residuos": os.path.join(out_dir, "residuals.png"),
    }
    existing = {name: p for name, p in report_paths.items() if os.path.isfile(p)}
    missing = [name for name in report_paths if name not in existing]
    if missing:
        print(f"\n  (No se encontraron todavía: {', '.join(missing)}.)")
    if not existing:
        return

    if _is_headless():
        print("\n  No hay display disponible; abrí manualmente estos archivos:")
        for name, p in existing.items():
            print(f"    {name}: {p}")
        return

    for name, p in existing.items():
        img = cv2.imread(p)
        if img is not None:
            cv2.imshow(name, img)
    print("\n  Mostrando reportes guardados -- cualquier tecla para salir.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def _resolve_aerial(args: argparse.Namespace, calib_cfg: dict) -> Optional[AerialImage]:
    """
    Resuelve qué `AerialImage` usar, si alguna: explícita (`--aerial`),
    recién descargada (`--fetch-aerial`), o la última ya cacheada en
    `calibration.aerial.cache_dir` si el modo aéreo está habilitado.

    Retorna:
        AerialImage | None
    """
    if args.aerial:
        try:
            return AerialImage.load(args.aerial)
        except (FileNotFoundError, ValueError) as exc:
            print(f"✗ {exc}")
            sys.exit(1)

    if args.fetch_aerial:
        if not args.center:
            print("✗ --fetch-aerial requiere --center lat,lon")
            sys.exit(1)
        try:
            lat_str, lon_str = args.center.split(",")
            center_lat, center_lon = float(lat_str.strip()), float(lon_str.strip())
        except ValueError:
            print(f"✗ --center inválido: '{args.center}'. Formato esperado: lat,lon (ej. 9.8574,-83.9112)")
            sys.exit(1)
        aerial_cfg = calib_cfg.get("aerial", {})
        api_key = os.environ.get(aerial_cfg.get("api_key_env", "MAPTILER_KEY"))
        cache_dir = aerial_cfg.get("cache_dir", "data/aerial/")
        os.makedirs(cache_dir, exist_ok=True)
        out_img_path = os.path.join(cache_dir, f"aerial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        size = aerial_cfg.get("size", [1024, 1024])
        return AerialImage.fetch(center_lat, center_lon, args.zoom, size[0], size[1], api_key, out_img_path)

    if calib_cfg.get("aerial", {}).get("enabled", True):
        cache_dir = calib_cfg.get("aerial", {}).get("cache_dir", "data/aerial/")
        existing = sorted(p for p in glob.glob(os.path.join(cache_dir, "*.png")) if os.path.isfile(f"{p}.meta.json"))
        if existing:
            try:
                aerial_obj = AerialImage.load(existing[-1])
                print(f"  Usando imagen aérea ya descargada: {existing[-1]} (pasá --fetch-aerial para actualizarla).")
                return aerial_obj
            except (FileNotFoundError, ValueError):
                return None

    return None


def main() -> None:
    """
    Punto de entrada principal: despacha a `--show`/`--verify`, o resuelve
    video, imagen aérea y puntos (CSV, aéreo o manual) y corre el flujo
    común de calibración.

    Retorna:
        None
    """
    args = parse_args()
    config = load_config(args.config)
    setup_calibration_logging(config["paths"]["results"])

    calib_cfg = config.setdefault("calibration", {})
    output_path = args.output or calib_cfg.get("file", "config/calibration.json")

    if args.show:
        run_show_mode(args, config, output_path)
        return
    if args.verify:
        run_verify_mode(args, config, output_path)
        return

    video_path = args.video or find_video(config["paths"]["videos"])
    if not video_path:
        print(f"✗ No se encontró ningún video en {config['paths']['videos']}. Indicá --video.")
        sys.exit(1)

    frame, frame_size = get_reference_frame(video_path, args.frame_sec)
    print(f"Video: {video_path}  ({frame_size[0]}x{frame_size[1]})")

    aerial_obj = None if args.points else _resolve_aerial(args, calib_cfg)
    points_mode = "csv" if args.points else ("aerial" if aerial_obj is not None else "manual")

    if points_mode != "csv" and _is_headless():
        print(
            "✗ No se detectó un display disponible. El marcado interactivo requiere una interfaz "
            "gráfica; preparalo en un CSV y usá --points archivo.csv en su lugar."
        )
        sys.exit(1)

    width, height = frame_size
    if points_mode == "csv":
        points = load_points_csv(args.points)
    elif points_mode == "aerial":
        points = run_aerial_mode(frame, aerial_obj, width, height)
    else:
        points = run_manual_mode(frame, width, height)

    aerial_meta = None
    if points_mode == "aerial":
        aerial_meta = {
            "source": calib_cfg.get("aerial", {}).get("provider", "maptiler"),
            "image": aerial_obj.image_path,
            "center": [aerial_obj.center_lat, aerial_obj.center_lon],
            "zoom": aerial_obj.zoom,
            "m_per_px": aerial_obj.meters_per_pixel(),
        }

    run_common_pipeline(points, frame, frame_size, video_path, config, args, output_path, aerial_meta)


if __name__ == "__main__":
    main()

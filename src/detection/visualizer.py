"""
Vista previa interactiva de detección vehicular sobre video.

Este módulo es una herramienta de **inspección visual**, independiente del
benchmark (`src/detection/benchmark.py`). Permite ver el video corriendo con
las bounding boxes dibujadas en tiempo real (o exportarlo a un archivo), para
validar cualitativamente que un modelo detecta bien: que no pierde vehículos
entre frames, que las cajas no parpadean, que no confunde clases y que el
umbral de confianza está bien elegido.

Importante: los FPS que se muestran aquí (`display_fps`) son solo
informativos en pantalla. Incluyen el costo de dibujar las cajas, el HUD y
renderizar la ventana (o codificar el video de salida), por lo que **no son
comparables** con los FPS de inferencia pura que mide `YOLOBenchmark`. No se
guardan en ningún archivo de resultados del benchmark.
"""
import logging
import os
import time
from collections import deque
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from src.detection.detector import VehicleDetector
from src.tracking.tracker import VehicleTracker
from src.utils.roi import ROIFilter, load_roi

logger = logging.getLogger(__name__)

# Colores BGR por clase de vehículo. Se duplican intencionalmente los mismos
# valores que usa `detector.py` (en vez de importarlos desde allí) para que
# este módulo no dependa de un símbolo privado de `detector.py` y para
# respetar la restricción de no tocar ese archivo.
_CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (255, 100, 0),
    "bus": (0, 165, 255),
    "motorcycle": (0, 255, 255),
}
_DEFAULT_COLOR = (200, 200, 200)
_VEHICLE_CLASS_NAMES = ["car", "truck", "bus", "motorcycle"]
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# Códigos de flecha derecha devueltos por cv2.waitKeyEx según plataforma y
# backend de GUI de OpenCV (Linux/GTK, Windows, macOS). La tecla 'n' es el
# atajo confiable y multiplataforma; estos códigos son un plus best-effort.
_RIGHT_ARROW_CODES = {83, 3, 2555904, 65363, 63235}


def _build_id_palette(n: int = 20) -> List[Tuple[int, int, int]]:
    """
    Genera una paleta de `n` colores BGR bien separados en tono (hues
    distribuidos uniformemente a saturación y valor máximos), para colorear
    por `track_id` en el modo seguimiento.

    Retorna:
        list[tuple[int, int, int]]: `n` colores BGR.
    """
    palette = []
    for i in range(n):
        hue = int(180 * i / n)  # OpenCV usa hue en [0, 179]
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        palette.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return palette


# Paleta fija de 20 colores por track_id. La decisión central del modo
# seguimiento: si un vehículo conserva su color a lo largo de la escena, el
# seguimiento es correcto; si cambia de color a media cuadra, se acaba de
# observar un cambio de identidad a simple vista. Con colores por clase
# (todos los carros verdes) ese fallo sería invisible.
_ID_PALETTE = _build_id_palette(20)


class DetectionVisualizer:
    """
    Reproduce, exporta o compara visualmente un video con las detecciones de
    uno o dos `VehicleDetector` dibujadas encima, para validación cualitativa.

    No escribe nunca en `results/benchmarks/`, `results/plots/` ni
    `results/frames/`: toda su salida va a `results/preview/`.
    """

    def __init__(
        self, config: Dict[str, Any], detector: VehicleDetector, video_path: str,
        roi_filter: Optional[ROIFilter] = None, tracker: Optional[VehicleTracker] = None,
    ) -> None:
        """
        Abre el video y prepara el estado interno del visualizador.

        Parámetros:
            config (dict): configuración completa cargada desde config.yaml.
            detector (VehicleDetector): instancia ya construida del detector
                a usar como modelo principal. Se usa siempre para el nombre
                del modelo, el umbral de confianza mostrado en el HUD, etc.,
                incluso en modo seguimiento.
            video_path (str): ruta al archivo de video a visualizar.
            roi_filter (ROIFilter | None): filtro de región de interés ya
                construido para la resolución de este video. Si es None pero
                `config.roi.enabled` es True, se construye uno automáticamente
                a partir de `config.roi.file` una vez que se conoce la
                resolución real del video (ver más abajo).
            tracker (VehicleTracker | None): si se pasa, el visualizador usa
                `tracker.update(frame, idx)` en vez de `detector.detect(frame)`
                para obtener las detecciones (con la clave adicional
                `"track_id"`), y activa el modo de visualización de
                seguimiento (color por ID, estelas, HUD extendido).

        Retorna:
            None

        Excepciones:
            FileNotFoundError: si `video_path` no existe en disco, o si la
                ROI está habilitada pero su archivo no existe.
            RuntimeError: si OpenCV no puede abrir el video (formato o
                codec no soportado).
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"No se encontró el video para la vista previa: '{video_path}'. "
                f"Verifica la ruta o coloca el archivo en data/videos/."
            )

        self.config = config
        self.detector = detector
        self.video_path = video_path

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"OpenCV no pudo abrir el video '{video_path}'. El archivo puede estar "
                f"corrupto o usar un códec no soportado por tu instalación de OpenCV."
            )

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        preview_cfg = config.get("preview", {})
        self.max_display_width = preview_cfg.get("max_display_width", 1280)
        self.scale_factor = (
            self.max_display_width / self.width if self.width > self.max_display_width else 1.0
        )
        self.display_width = max(1, int(round(self.width * self.scale_factor)))
        self.display_height = max(1, int(round(self.height * self.scale_factor)))

        self.show_hud = bool(preview_cfg.get("show_hud", True))
        self.show_confidence = bool(preview_cfg.get("show_confidence", True))
        self.box_thickness = int(preview_cfg.get("box_thickness", 2))
        self.font_scale = float(preview_cfg.get("font_scale", 0.5))
        self.playback_fps = int(preview_cfg.get("playback_fps", 0))
        self.export_codec = preview_cfg.get("export_codec", "mp4v")
        self.export_dir = preview_cfg.get("export_dir", "results/preview/")
        self.snapshots_dir = preview_cfg.get("snapshots_dir", "results/preview/snapshots/")

        roi_cfg = config.get("roi", {})
        self.roi_draw_overlay = bool(roi_cfg.get("draw_overlay", True))
        if roi_filter is not None:
            self.roi_filter: Optional[ROIFilter] = roi_filter
        elif roi_cfg.get("enabled"):
            roi_data = load_roi(roi_cfg.get("file", "config/roi.json"))
            self.roi_filter = ROIFilter(roi_data, (self.width, self.height), config)
            logger.info(
                "ROI activa para la vista previa (%.1f%% del frame)", self.roi_filter.area_ratio() * 100
            )
        else:
            self.roi_filter = None

        self.tracker = tracker
        tracking_cfg = config.get("tracking", {})
        viz_cfg = tracking_cfg.get("visualization", {})
        self.min_track_frames = int(tracking_cfg.get("min_track_frames", 10))
        self.trail_length = int(viz_cfg.get("trail_length", 40))
        self.trail_ttl_frames = int(viz_cfg.get("trail_ttl_frames", 15))
        self.color_by_id = bool(viz_cfg.get("color_by_id", True))
        self.show_ids = bool(viz_cfg.get("show_ids", True))
        self.show_trails = bool(viz_cfg.get("show_trails", True))
        self.show_lost_trails = bool(viz_cfg.get("show_lost_trails", True))
        self.id_font_scale = float(viz_cfg.get("id_font_scale", 0.6))
        # Estado de estelas del tracker principal (self.tracker). El modo de
        # comparación de trackers (run_compare con tracker_b) usa diccionarios
        # locales propios en vez de estos, para no mezclar los IDs de dos
        # trackers independientes en la misma estructura.
        self._trails: Dict[int, deque] = {}
        self._trail_last_seen: Dict[int, int] = {}

        # Controlados por el script CLI antes de correr; no forman parte del
        # prompt original pero son necesarios para soportar --start-sec y
        # --max-frames sin alterar la firma pedida del constructor.
        self.start_frame = 0
        self.max_frames: Optional[int] = None

        self._fps_deque: deque = deque(maxlen=preview_cfg.get("fps_window", 30))
        self._last_tick: Optional[float] = None

        os.makedirs(self.export_dir, exist_ok=True)
        os.makedirs(self.snapshots_dir, exist_ok=True)

        logger.info(
            "Visualizador listo: %s (%dx%d, %.1f fps, %d frames) | modelo=%s",
            video_path, self.width, self.height, self.source_fps,
            self.total_frames, getattr(detector, "model_id", "?"),
        )

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def seek(self, start_sec: float) -> None:
        """
        Posiciona el video en el segundo indicado antes de empezar a leer.

        Parámetros:
            start_sec (float): segundo del video donde comenzar. `0` deja el
                video en su posición inicial (no hace nada).

        Retorna:
            None

        Excepciones:
            ValueError: si el video es más corto que `start_sec`.
        """
        if start_sec <= 0:
            return

        start_frame = int(round(start_sec * self.source_fps))
        if self.total_frames > 0 and start_frame >= self.total_frames:
            duration_sec = self.total_frames / self.source_fps if self.source_fps else 0.0
            raise ValueError(
                f"--start-sec {start_sec:.1f}s está más allá del final del video "
                f"(duración aproximada: {duration_sec:.1f}s). Elige un valor menor."
            )

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.start_frame = start_frame
        logger.info("Video posicionado en el frame %d (segundo %.1f)", start_frame, start_sec)

    def _effective_total(self) -> int:
        """
        Calcula cuántos frames se van a procesar en total, considerando el
        punto de inicio (`start_frame`) y el límite (`max_frames`).

        Retorna:
            int: cantidad de frames a procesar. `0` si no se puede determinar
            (metadata de duración no confiable).
        """
        remaining = max(0, self.total_frames - self.start_frame)
        if self.max_frames is not None:
            return min(remaining, self.max_frames) if remaining > 0 else self.max_frames
        return remaining

    def _tick_fps(self) -> float:
        """
        Actualiza el promedio móvil de FPS de pantalla con una nueva marca
        de tiempo (se llama una vez por frame efectivamente mostrado).

        Retorna:
            float: FPS promedio sobre la ventana `fps_window` configurada.
        """
        now = time.perf_counter()
        if self._last_tick is not None:
            delta = now - self._last_tick
            if delta > 0:
                self._fps_deque.append(1.0 / delta)
        self._last_tick = now
        if not self._fps_deque:
            return 0.0
        return sum(self._fps_deque) / len(self._fps_deque)

    def _count_by_class(self, detections: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Cuenta detecciones por clase de vehículo.

        Parámetros:
            detections (list[dict]): detecciones retornadas por `detect()`.

        Retorna:
            dict: conteo por clase, con las 4 clases siempre presentes
            (aunque su valor sea 0).
        """
        counts = {name: 0 for name in _VEHICLE_CLASS_NAMES}
        for det in detections:
            if det["class_name"] in counts:
                counts[det["class_name"]] += 1
        return counts

    def _resize_for_display(self, frame: np.ndarray) -> np.ndarray:
        """
        Reescala un frame ya anotado a su tamaño de despliegue/exportación.

        La inferencia siempre corre sobre el frame original a resolución
        completa; este reescalado es solo para mostrar/exportar.

        Parámetros:
            frame (numpy.ndarray): frame anotado a resolución original.

        Retorna:
            numpy.ndarray: frame reescalado a `(display_width, display_height)`.
        """
        if self.scale_factor == 1.0:
            return frame
        return cv2.resize(
            frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA
        )

    @staticmethod
    def _scale_to_width(frame: np.ndarray, max_width: int) -> np.ndarray:
        """
        Reescala un frame para que su ancho no supere `max_width`, manteniendo
        proporción. Se usa para la vista combinada de `run_compare`, cuyo
        ancho (suma de dos videos) no se conoce hasta tener ambos frames.

        Parámetros:
            frame (numpy.ndarray): frame a reescalar.
            max_width (int): ancho máximo permitido.

        Retorna:
            numpy.ndarray: el mismo frame si ya entra en `max_width`, o una
            copia reescalada si no.
        """
        h, w = frame.shape[:2]
        if w <= max_width:
            return frame
        scale = max_width / w
        return cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    def _compute_wait_delay(self, inference_ms: float, paused: bool) -> int:
        """
        Calcula el delay en milisegundos para `cv2.waitKey`.

        Parámetros:
            inference_ms (float): tiempo que tomó la inferencia del frame
                actual, para descontarlo del delay cuando hay límite de FPS.
            paused (bool): si el video está en pausa (se usa un delay fijo
                bajo para no saturar la CPU con sondeo de teclado).

        Retorna:
            int: milisegundos a pasarle a `cv2.waitKey`, mínimo 1.
        """
        if paused:
            return 30
        if self.playback_fps > 0:
            return max(1, int(1000 / self.playback_fps - inference_ms))
        return 1

    def _save_snapshot(self, frame: np.ndarray, frame_idx: int) -> str:
        """
        Guarda una captura del frame anotado actual en `snapshots_dir`.

        Parámetros:
            frame (numpy.ndarray): frame anotado a guardar (resolución
                original, no la reescalada para display).
            frame_idx (int): índice del frame, usado en el nombre de archivo.

        Retorna:
            str: ruta absoluta/relativa del archivo PNG guardado.
        """
        filename = f"snap_{self.detector.model_id}_{frame_idx:06d}.png"
        path = os.path.join(self.snapshots_dir, filename)
        cv2.imwrite(path, frame)
        logger.info("Captura guardada: %s", path)
        return path

    def _headless_check(self) -> bool:
        """
        Determina si el entorno actual probablemente no tiene display
        disponible (por ejemplo, una sesión SSH a una Raspberry Pi sin X).

        Retorna:
            bool: True si no parece haber display disponible.
        """
        if os.name == "posix" and not os.uname().sysname == "Darwin":
            has_display = bool(os.environ.get("DISPLAY")) or bool(os.environ.get("WAYLAND_DISPLAY"))
            if not has_display:
                return True

        try:
            test_window = "_preview_headless_probe"
            cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
            cv2.destroyWindow(test_window)
            return False
        except cv2.error:
            return True

    # ------------------------------------------------------------------
    # Dibujado
    # ------------------------------------------------------------------

    def _draw_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibuja las bounding boxes de las detecciones sobre una copia del frame.

        Parámetros:
            frame (numpy.ndarray): imagen BGR original.
            detections (list[dict]): detecciones retornadas por `detect()`.

        Retorna:
            numpy.ndarray: copia del frame con las cajas y etiquetas
            dibujadas. El frame original no se modifica.
        """
        annotated = frame.copy()
        frame_h = annotated.shape[0]

        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
            color = _CLASS_COLORS.get(det["class_name"], _DEFAULT_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            if self.show_confidence:
                label = f'{det["class_name"]} {det["confidence"]:.2f}'
            else:
                label = det["class_name"]

            (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, self.font_scale, 1)
            pad = 4

            if y1 - text_h - baseline - pad < 0:
                # No cabe encima de la caja: se dibuja dentro, pegada al borde superior.
                label_top = y1
                label_bottom = min(frame_h, y1 + text_h + baseline + pad)
                text_y = y1 + text_h + 1
            else:
                label_top = y1 - text_h - baseline - pad
                label_bottom = y1
                text_y = y1 - baseline - 2

            cv2.rectangle(annotated, (x1, label_top), (x1 + text_w + pad * 2, label_bottom), color, -1)
            cv2.putText(
                annotated, label, (x1 + pad, text_y),
                _FONT, self.font_scale, (0, 0, 0), 1, cv2.LINE_AA,
            )

        return annotated

    def _draw_hud(self, frame: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Dibuja el panel de información (HUD) sobre una copia del frame.

        Parámetros:
            info (dict): datos a mostrar. Debe contener las claves
                `frame_idx`, `total_frames`, `model_id`, `inference_ms`,
                `display_fps`, `counts`, `total_vehicles`, `paused` y
                `conf_threshold`. Si incluye `roi_active=True`, además debe
                traer `roi_inside`/`roi_outside` y se agrega una línea extra
                con el desglose dentro/fuera de la ROI. Si incluye
                `tracker_name`, además debe traer `active_tracks`/`total_ids`
                y se agrega una línea extra con el estado del seguimiento.

        Retorna:
            numpy.ndarray: copia del frame con el HUD dibujado encima.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]
        pad = 10
        line_height = 20

        total_frames = max(info["total_frames"], 1)
        pct = int(round(100 * info["frame_idx"] / total_frames))

        parts = [f'{name} {n}' for name, n in info["counts"].items() if n > 0]
        detail = "  (" + " · ".join(parts) + ")" if parts else ""

        lines = [
            f'Modelo: {info["model_id"]} | conf: {info["conf_threshold"]:.2f}',
            f'Frame: {info["frame_idx"]}/{info["total_frames"]}  ({pct}%)',
            f'Inferencia: {info["inference_ms"]:.1f} ms | Display: {info["display_fps"]:.1f} FPS',
            f'Vehiculos: {info["total_vehicles"]}{detail}',
        ]

        if info.get("roi_active"):
            lines.append(f'ROI: {info["roi_inside"]} dentro · {info["roi_outside"]} fuera')

        if info.get("tracker_name"):
            lines.append(
                f'Tracker: {info["tracker_name"]} | Activos: {info["active_tracks"]} | '
                f'IDs totales: {info["total_ids"]}'
            )

        panel_w = min(w - 20, 460)
        panel_h = pad * 2 + line_height * len(lines)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (0, 0, 0), -1)
        annotated = cv2.addWeighted(overlay, 0.55, annotated, 0.45, 0)

        text_y = 10 + pad + 14
        for line in lines:
            cv2.putText(annotated, line, (10 + pad, text_y), _FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            text_y += line_height

        if info.get("paused"):
            pause_text = "|| PAUSA"
            (tw, th), _ = cv2.getTextSize(pause_text, _FONT, 1.0, 2)
            cv2.putText(annotated, pause_text, (w - tw - 20, th + 20), _FONT, 1.0, (0, 0, 255), 2, cv2.LINE_AA)

        # Barra de progreso en el borde inferior.
        progress = info["frame_idx"] / total_frames if total_frames else 0.0
        progress = min(max(progress, 0.0), 1.0)
        bar_y = h - 4
        cv2.line(annotated, (0, bar_y), (w, bar_y), (70, 70, 70), 4)
        cv2.line(annotated, (0, bar_y), (int(w * progress), bar_y), (0, 220, 0), 4)

        # Ayuda de controles, esquina inferior derecha.
        controls_text = "[espacio] pausa  [q] salir  [s] captura  [h] HUD"
        (tw, th), _ = cv2.getTextSize(controls_text, _FONT, 0.4, 1)
        cv2.putText(
            annotated, controls_text, (w - tw - 10, h - 12),
            _FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA,
        )

        return annotated

    def _label_compare_half(
        self, frame: np.ndarray, model_id: str, inference_ms: float, vehicle_count: int,
        outside_count: Optional[int] = None,
    ) -> np.ndarray:
        """
        Dibuja un rótulo compacto sobre una de las dos mitades de la vista
        de comparación, con el nombre del modelo, su tiempo de inferencia y
        el conteo de vehículos detectados en ese frame.

        Parámetros:
            outside_count (int | None): si la ROI está activa, cantidad de
                detecciones fuera de ella, para mostrar "N dentro / M fuera".

        Retorna:
            numpy.ndarray: copia del frame con el rótulo dibujado.
        """
        annotated = frame.copy()
        if outside_count is None:
            text = f'{model_id} | {inference_ms:.1f} ms | Veh: {vehicle_count}'
        else:
            text = f'{model_id} | {inference_ms:.1f} ms | Veh: {vehicle_count} din / {outside_count} fue'
        (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
        cv2.rectangle(annotated, (0, 0), (text_w + 20, text_h + baseline + 16), (0, 0, 0), -1)
        cv2.putText(annotated, text, (10, text_h + 8), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    def _draw_outside_boxes(self, frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Dibuja, sobre una copia del frame, las detecciones que quedaron
        fuera de la ROI: en gris tenue, grosor 1 y sin etiqueta. Ver qué se
        está descartando es justamente el valor de esta vista, así que no se
        ocultan del todo.

        Parámetros:
            frame (numpy.ndarray): frame ya anotado con las detecciones "dentro".
            detections (list[dict]): detecciones fuera de la ROI.

        Retorna:
            numpy.ndarray: copia del frame con las cajas grises agregadas.
        """
        annotated = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (120, 120, 120), 1)
        return annotated

    def _detect_with_roi(
        self, frame: np.ndarray, detector: VehicleDetector, roi_active: bool
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Corre la detección aplicando el modo de ROI configurado
        (`filter`/`crop`/`mask`) y separa el resultado en dentro/fuera.

        En modo `crop`, la inferencia corre solo sobre el recorte y los
        bboxes se restauran al sistema de coordenadas del frame completo con
        `restore_bboxes()`. En modo `mask`, la inferencia corre sobre el
        frame con el exterior de la ROI ennegrecido. En modo `filter` (o sin
        ROI), la inferencia corre sobre el frame completo sin modificar.

        En todos los casos, el resultado final se filtra con
        `roi_filter.filter()` para obtener la partición dentro/fuera exacta
        según el polígono (no solo el rectángulo envolvente de `crop`).

        Parámetros:
            frame (numpy.ndarray): frame original (resolución completa).
            detector (VehicleDetector): detector a usar.
            roi_active (bool): si la ROI está activa en este momento (permite
                apagarla en caliente con la tecla `o` sin reconstruir el
                visualizador).

        Retorna:
            tuple: (detecciones dentro, detecciones fuera, tiempo de
            inferencia en ms). Si no hay ROI o está desactivada, todas las
            detecciones van en "dentro" y "fuera" queda vacía.
        """
        roi = self.roi_filter if roi_active else None

        t0 = time.perf_counter()
        if roi is None:
            detections = detector.detect(frame)
        elif roi.mode == "crop":
            cropped, offset = roi.crop(frame)
            raw = detector.detect(cropped)
            detections = roi.restore_bboxes(raw, offset)
        elif roi.mode == "mask":
            detections = detector.detect(roi.mask(frame))
        else:
            detections = detector.detect(frame)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0

        if roi is None:
            return detections, [], inference_ms

        inside, outside = roi.filter(detections)
        return inside, outside, inference_ms

    def _annotate_side(
        self, frame: np.ndarray, detector: VehicleDetector, show_boxes: bool, roi_active: bool = True
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], List[Dict[str, Any]], float]:
        """
        Corre inferencia (con ROI aplicada si corresponde) con un detector
        dado sobre un frame y opcionalmente dibuja sus cajas. Usado por
        `run_compare` para procesar cada mitad.

        Parámetros:
            frame (numpy.ndarray): frame original (resolución completa).
            detector (VehicleDetector): detector a usar.
            show_boxes (bool): si se deben dibujar las cajas.
            roi_active (bool): si aplicar la ROI (cuando hay una configurada).

        Retorna:
            tuple: (frame anotado, detecciones dentro, detecciones fuera,
            tiempo de inferencia en ms).
        """
        inside, outside, inference_ms = self._detect_with_roi(frame, detector, roi_active)

        annotated = frame.copy()
        if show_boxes:
            annotated = self._draw_boxes(annotated, inside)
            if outside:
                annotated = self._draw_outside_boxes(annotated, outside)
        if roi_active and self.roi_filter is not None and self.roi_draw_overlay:
            annotated = self.roi_filter.draw(annotated)

        return annotated, inside, outside, inference_ms

    # ------------------------------------------------------------------
    # Seguimiento: color por ID, estelas y dibujado de tracks
    # ------------------------------------------------------------------

    def _get_id_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Retorna:
            tuple[int, int, int]: color BGR estable para un `track_id`
            (`_ID_PALETTE[track_id % len(_ID_PALETTE)]`).
        """
        return _ID_PALETTE[track_id % len(_ID_PALETTE)]

    def _update_trails(
        self, detections: List[Dict[str, Any]], frame_idx: int,
        trails: Dict[int, deque], last_seen: Dict[int, int],
    ) -> None:
        """
        Actualiza (in-place) el histórico de puntos de contacto por
        `track_id`. Se llama siempre, independientemente de si las estelas
        se están dibujando en este momento (`show_trails`), para no perder
        historial si el usuario las oculta y las vuelve a mostrar.

        Parámetros:
            detections (list[dict]): detecciones de `VehicleTracker.update()`
                (deben traer `"track_id"`).
            frame_idx (int): índice del frame actual.
            trails (dict[int, deque]): estado de estelas a actualizar (del
                tracker principal o de uno de los dos lados de comparación).
            last_seen (dict[int, int]): último frame en que se vio cada ID,
                actualizado en conjunto con `trails`.

        Retorna:
            None
        """
        for det in detections:
            track_id = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            point = (int(round((x1 + x2) / 2.0)), int(round(y2)))
            if track_id not in trails:
                trails[track_id] = deque(maxlen=self.trail_length)
            trails[track_id].append(point)
            last_seen[track_id] = frame_idx

    def _draw_trails(
        self, frame: np.ndarray, active_ids: set, frame_idx: int,
        trails: Dict[int, deque], last_seen: Dict[int, int],
    ) -> np.ndarray:
        """
        Dibuja las estelas activas y (opcionalmente) las recién perdidas.

        Elección de implementación para "grosor y opacidad decrecientes
        hacia el pasado": mezclar cada segmento por separado con
        `cv2.addWeighted` sería un blend por segmento por cada ID activo en
        cada frame — demasiado costoso. En vez de eso, toda la estela
        (con grosor creciente hacia el presente) se dibuja una vez sobre una
        única capa para todo el frame, esa capa se mezcla UNA sola vez con
        opacidad reducida, y encima se redibuja el segmento y el punto más
        recientes de cada ID a opacidad plena. El resultado visual es
        equivalente (cola tenue y angosta, cabeza nítida y gruesa) con un
        solo `addWeighted` por frame en vez de uno por segmento.

        Las estelas de IDs no vistos hace más de `trail_ttl_frames` se
        purgan (se eliminan de `trails`/`last_seen`) para que la memoria no
        crezca sin límite en videos largos.

        Parámetros:
            active_ids (set[int]): IDs con una detección en este frame.
            trails / last_seen: mismos diccionarios que actualiza `_update_trails`.

        Retorna:
            numpy.ndarray: copia del frame con las estelas dibujadas.
        """
        overlay = frame.copy()
        to_purge = []
        last_segments = []
        heads = []

        for track_id, points in trails.items():
            last = last_seen.get(track_id, frame_idx)
            age = frame_idx - last
            if age > self.trail_ttl_frames:
                to_purge.append(track_id)
                continue

            is_active = track_id in active_ids
            if not is_active and not self.show_lost_trails:
                continue

            color = self._get_id_color(track_id) if is_active else (130, 130, 130)
            pts = list(points)
            n = len(pts)

            if n < 2:
                if is_active and pts:
                    heads.append((pts[-1], color))
                continue

            for i in range(1, n):
                frac = i / (n - 1)
                thickness = max(1, int(round(1 + 2 * frac)))
                cv2.line(overlay, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

            if is_active:
                heads.append((pts[-1], color))
                last_segments.append((pts[-2], pts[-1], color))

        annotated = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        for p1, p2, color in last_segments:
            cv2.line(annotated, p1, p2, color, 3, cv2.LINE_AA)
        for point, color in heads:
            cv2.circle(annotated, point, 5, color, -1)

        for track_id in to_purge:
            trails.pop(track_id, None)
            last_seen.pop(track_id, None)

        return annotated

    def _resize_trail_buffers(self) -> None:
        """
        Reconstruye las colas de estela del tracker principal con el
        `trail_length` actual (tras `[`/`]`), preservando los puntos más
        recientes.

        Retorna:
            None
        """
        for track_id, points in list(self._trails.items()):
            self._trails[track_id] = deque(points, maxlen=self.trail_length)

    def _draw_tracks(
        self, frame: np.ndarray, detections: List[Dict[str, Any]],
        color_by_id: bool, show_ids: bool, tracker: VehicleTracker,
    ) -> np.ndarray:
        """
        Dibuja las cajas del modo seguimiento: color por ID (o por clase),
        etiqueta con formato `#14 car 0.86`, y un indicador discreto de
        antigüedad si el track lleva más de `min_track_frames` observaciones.

        Parámetros:
            color_by_id (bool): si es False, colorea por clase de vehículo
                (como en el modo de solo detección) en vez de por ID.
            show_ids (bool): si es False, dibuja la etiqueta sin el ID
                (equivalente al modo de solo detección).
            tracker (VehicleTracker): tracker de donde leer la antigüedad de
                cada track (permite reusar este método en `run_compare` con
                dos trackers distintos).

        Retorna:
            numpy.ndarray: copia del frame con las cajas y etiquetas dibujadas.
        """
        annotated = frame.copy()
        frame_h = annotated.shape[0]
        label_font_scale = self.id_font_scale if show_ids else self.font_scale

        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
            track_id = det["track_id"]

            color = (
                self._get_id_color(track_id) if color_by_id
                else _CLASS_COLORS.get(det["class_name"], _DEFAULT_COLOR)
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            if show_ids:
                track = tracker.tracks.get(track_id)
                age = track.length_frames() if track is not None else 1
                age_suffix = f' ({age}f)' if age > self.min_track_frames else ''
                label = f'#{track_id} {det["class_name"]} {det["confidence"]:.2f}{age_suffix}'
                thickness = 2  # "negrita visual": OpenCV no tiene bold real, se aproxima con más grosor
            else:
                label = f'{det["class_name"]} {det["confidence"]:.2f}'
                thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(label, _FONT, label_font_scale, thickness)
            pad = 4

            if y1 - text_h - baseline - pad < 0:
                label_top = y1
                label_bottom = min(frame_h, y1 + text_h + baseline + pad)
                text_y = y1 + text_h + 1
            else:
                label_top = y1 - text_h - baseline - pad
                label_bottom = y1
                text_y = y1 - baseline - 2

            cv2.rectangle(annotated, (x1, label_top), (x1 + text_w + pad * 2, label_bottom), color, -1)
            cv2.putText(
                annotated, label, (x1 + pad, text_y),
                _FONT, label_font_scale, (0, 0, 0), thickness, cv2.LINE_AA,
            )

        return annotated

    def _label_compare_tracking_half(
        self, frame: np.ndarray, tracker_name: str, inference_ms: float,
        active_count: int, total_ids: int,
    ) -> np.ndarray:
        """
        Rótulo compacto para una mitad de `run_compare` en modo comparación
        de trackers: nombre del tracker, tiempo de inferencia, tracks
        activos y total de IDs generados hasta el momento.

        Retorna:
            numpy.ndarray: copia del frame con el rótulo dibujado.
        """
        annotated = frame.copy()
        text = f'{tracker_name} | {inference_ms:.1f} ms | Activos: {active_count} | IDs: {total_ids}'
        (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, 0.6, 2)
        cv2.rectangle(annotated, (0, 0), (text_w + 20, text_h + baseline + 16), (0, 0, 0), -1)
        cv2.putText(annotated, text, (10, text_h + 8), _FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return annotated

    # ------------------------------------------------------------------
    # Modos de ejecución
    # ------------------------------------------------------------------

    def _process_live_frame(
        self, frame: np.ndarray, frame_idx: int, show_boxes: bool, show_hud: bool,
        roi_active: bool, paused: bool,
    ) -> Tuple[np.ndarray, float]:
        """
        Procesa un único frame para `run_live`: corre la detección (con ROI
        si corresponde), dibuja cajas/ROI/HUD según el estado actual de los
        toggles de teclado. Compartido entre el avance normal y el avance
        manual de un frame en pausa (tecla `n`), para no duplicar lógica.

        Parámetros:
            roi_active (bool): estado actual del toggle de ROI (tecla `o`).
            paused (bool): si el video está en pausa (se refleja en el HUD).

        Retorna:
            tuple: (frame anotado a resolución completa, tiempo de inferencia en ms).
        """
        inside, outside, inference_ms = self._detect_with_roi(frame, self.detector, roi_active)
        display_fps = self._tick_fps()

        annotated = self._draw_boxes(frame, inside) if show_boxes else frame.copy()
        if show_boxes and outside:
            annotated = self._draw_outside_boxes(annotated, outside)
        if roi_active and self.roi_filter is not None and self.roi_draw_overlay:
            annotated = self.roi_filter.draw(annotated)

        if show_hud:
            counts = self._count_by_class(inside)
            info: Dict[str, Any] = {
                "frame_idx": frame_idx,
                "total_frames": self.total_frames,
                "model_id": self.detector.model_id,
                "inference_ms": inference_ms,
                "display_fps": display_fps,
                "counts": counts,
                "total_vehicles": len(inside),
                "paused": paused,
                "conf_threshold": self.detector.confidence_threshold,
            }
            if roi_active and self.roi_filter is not None:
                info["roi_active"] = True
                info["roi_inside"] = len(inside)
                info["roi_outside"] = len(outside)
            annotated = self._draw_hud(annotated, info)

        return annotated, inference_ms

    def _process_tracking_frame(
        self, frame: np.ndarray, frame_idx: int, show_boxes: bool, show_hud: bool,
        show_trails: bool, show_ids: bool, color_by_id: bool, paused: bool,
    ) -> Tuple[np.ndarray, float]:
        """
        Procesa un único frame en modo seguimiento: corre `self.tracker.update()`,
        actualiza las estelas y dibuja cajas/estelas/HUD según los toggles.
        Análogo a `_process_live_frame` pero para el tracker principal.

        Retorna:
            tuple: (frame anotado a resolución completa, tiempo de inferencia en ms).
        """
        t0 = time.perf_counter()
        detections = self.tracker.update(frame, frame_idx)
        t1 = time.perf_counter()
        inference_ms = (t1 - t0) * 1000.0
        display_fps = self._tick_fps()

        active_ids = {det["track_id"] for det in detections}
        self._update_trails(detections, frame_idx, self._trails, self._trail_last_seen)

        annotated = frame.copy()
        if show_trails:
            annotated = self._draw_trails(annotated, active_ids, frame_idx, self._trails, self._trail_last_seen)
        if show_boxes:
            annotated = self._draw_tracks(annotated, detections, color_by_id, show_ids, self.tracker)

        if show_hud:
            counts = self._count_by_class(detections)
            info: Dict[str, Any] = {
                "frame_idx": frame_idx,
                "total_frames": self.total_frames,
                "model_id": self.detector.model_id,
                "inference_ms": inference_ms,
                "display_fps": display_fps,
                "counts": counts,
                "total_vehicles": len(detections),
                "paused": paused,
                "conf_threshold": self.detector.confidence_threshold,
                "tracker_name": self.tracker.tracker_name,
                "active_tracks": len(active_ids),
                "total_ids": len(self.tracker.tracks),
            }
            annotated = self._draw_hud(annotated, info)

        return annotated, inference_ms

    def run_live(self) -> None:
        """
        Reproduce el video en una ventana interactiva con las detecciones
        dibujadas en tiempo real, respondiendo a los controles de teclado
        (pausa, salir, captura, avanzar frame, mostrar/ocultar cajas, HUD y
        ROI, ajustar velocidad y reiniciar).

        Si el visualizador tiene un `tracker` configurado, delega en
        `_run_live_tracking()` (color por ID, estelas, HUD de seguimiento)
        en vez de correr el flujo de solo-detección.

        Retorna:
            None
        """
        if self.tracker is not None:
            self._run_live_tracking()
            return

        window_name = f"Deteccion vehicular - {self.detector.model_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        paused = False
        show_boxes = True
        show_hud = self.show_hud
        roi_active = self.roi_filter is not None

        last_full_annotated: Optional[np.ndarray] = None
        last_display: Optional[np.ndarray] = None
        frame_idx = self.start_frame - 1

        frames_processed = 0
        inference_times: List[float] = []
        snapshots_saved = 0

        try:
            while True:
                if not paused:
                    if self.max_frames is not None and frames_processed >= self.max_frames:
                        break
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    frames_processed += 1

                    annotated, inference_ms = self._process_live_frame(
                        frame, frame_idx, show_boxes, show_hud, roi_active, paused=False,
                    )
                    inference_times.append(inference_ms)

                    last_full_annotated = annotated
                    last_display = self._resize_for_display(annotated)
                else:
                    inference_ms = 0.0
                    if last_display is None:
                        break

                cv2.imshow(window_name, last_display)
                delay = self._compute_wait_delay(inference_ms, paused)
                key = cv2.waitKeyEx(delay)
                key_low = key & 0xFF if key != -1 else -1

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key_low in (ord('q'), 27):
                    break
                elif key_low == ord(' '):
                    paused = not paused
                elif key_low == ord('s') and last_full_annotated is not None:
                    self._save_snapshot(last_full_annotated, frame_idx)
                    snapshots_saved += 1
                    print(f"  Captura guardada (frame {frame_idx})")
                elif (key_low == ord('n') or key in _RIGHT_ARROW_CODES) and paused:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_idx += 1
                        frames_processed += 1

                        annotated, inference_ms = self._process_live_frame(
                            frame, frame_idx, show_boxes, show_hud, roi_active, paused=True,
                        )
                        inference_times.append(inference_ms)

                        last_full_annotated = annotated
                        last_display = self._resize_for_display(annotated)
                elif key_low == ord('h'):
                    show_hud = not show_hud
                elif key_low == ord('b'):
                    show_boxes = not show_boxes
                elif key_low == ord('o') and self.roi_filter is not None:
                    roi_active = not roi_active
                elif key_low == ord('+'):
                    self.playback_fps += 5
                elif key_low == ord('-'):
                    self.playback_fps = max(0, self.playback_fps - 5)
                elif key_low == ord('r'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = -1
                    paused = False
                    self._fps_deque.clear()
                    self._last_tick = None
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        mean_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
        pure_fps = 1000.0 / mean_inference if mean_inference > 0 else 0.0

        print("\nResumen de la vista previa:")
        print(f"  Frames procesados: {frames_processed}")
        print(f"  Inferencia promedio: {mean_inference:.1f} ms")
        print(f"  FPS de inferencia pura: {pure_fps:.1f}")
        print(f"  Capturas guardadas: {snapshots_saved}")
        logger.info(
            "run_live finalizado: %d frames, %.1f ms/frame, %.1f FPS pura, %d capturas",
            frames_processed, mean_inference, pure_fps, snapshots_saved,
        )

    def _run_live_tracking(self) -> None:
        """
        Implementa `run_live()` en modo seguimiento: ventana interactiva con
        color por ID, estelas y HUD extendido.

        Controles adicionales sobre el modo de solo detección: `t` (estelas),
        `i` (IDs), `c` (color por ID/clase), `[`/`]` (acortar/alargar estela).

        Retorna:
            None
        """
        window_name = f"Seguimiento - {self.detector.model_id} ({self.tracker.tracker_name})"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        paused = False
        show_boxes = True
        show_hud = self.show_hud
        show_trails = self.show_trails
        show_ids = self.show_ids
        color_by_id = self.color_by_id

        last_full_annotated: Optional[np.ndarray] = None
        last_display: Optional[np.ndarray] = None
        frame_idx = self.start_frame - 1

        frames_processed = 0
        inference_times: List[float] = []
        snapshots_saved = 0

        try:
            while True:
                if not paused:
                    if self.max_frames is not None and frames_processed >= self.max_frames:
                        break
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    frames_processed += 1

                    annotated, inference_ms = self._process_tracking_frame(
                        frame, frame_idx, show_boxes, show_hud, show_trails, show_ids,
                        color_by_id, paused=False,
                    )
                    inference_times.append(inference_ms)

                    last_full_annotated = annotated
                    last_display = self._resize_for_display(annotated)
                else:
                    inference_ms = 0.0
                    if last_display is None:
                        break

                cv2.imshow(window_name, last_display)
                delay = self._compute_wait_delay(inference_ms, paused)
                key = cv2.waitKeyEx(delay)
                key_low = key & 0xFF if key != -1 else -1

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key_low in (ord('q'), 27):
                    break
                elif key_low == ord(' '):
                    paused = not paused
                elif key_low == ord('s') and last_full_annotated is not None:
                    self._save_snapshot(last_full_annotated, frame_idx)
                    snapshots_saved += 1
                    print(f"  Captura guardada (frame {frame_idx})")
                elif (key_low == ord('n') or key in _RIGHT_ARROW_CODES) and paused:
                    ret, frame = self.cap.read()
                    if ret:
                        frame_idx += 1
                        frames_processed += 1

                        annotated, inference_ms = self._process_tracking_frame(
                            frame, frame_idx, show_boxes, show_hud, show_trails, show_ids,
                            color_by_id, paused=True,
                        )
                        inference_times.append(inference_ms)

                        last_full_annotated = annotated
                        last_display = self._resize_for_display(annotated)
                elif key_low == ord('h'):
                    show_hud = not show_hud
                elif key_low == ord('b'):
                    show_boxes = not show_boxes
                elif key_low == ord('t'):
                    show_trails = not show_trails
                elif key_low == ord('i'):
                    show_ids = not show_ids
                elif key_low == ord('c'):
                    color_by_id = not color_by_id
                elif key_low == ord('['):
                    self.trail_length = max(5, self.trail_length - 10)
                    self._resize_trail_buffers()
                elif key_low == ord(']'):
                    self.trail_length = self.trail_length + 10
                    self._resize_trail_buffers()
                elif key_low == ord('+'):
                    self.playback_fps += 5
                elif key_low == ord('-'):
                    self.playback_fps = max(0, self.playback_fps - 5)
                elif key_low == ord('r'):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = -1
                    paused = False
                    self._fps_deque.clear()
                    self._last_tick = None
                    self._trails.clear()
                    self._trail_last_seen.clear()
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        mean_inference = sum(inference_times) / len(inference_times) if inference_times else 0.0
        pure_fps = 1000.0 / mean_inference if mean_inference > 0 else 0.0

        print("\nResumen del seguimiento:")
        print(f"  Frames procesados: {frames_processed}")
        print(f"  Inferencia promedio: {mean_inference:.1f} ms")
        print(f"  FPS de inferencia pura: {pure_fps:.1f}")
        print(f"  IDs totales generados: {len(self.tracker.tracks)}")
        print(f"  Capturas guardadas: {snapshots_saved}")
        logger.info(
            "run_live (tracking) finalizado: %d frames, %.1f ms/frame, %d IDs totales, %d capturas",
            frames_processed, mean_inference, len(self.tracker.tracks), snapshots_saved,
        )

    def run_export(self, output_path: Optional[str] = None) -> None:
        """
        Genera un archivo de video con las detecciones dibujadas (cajas y
        HUD siempre activos), sin abrir ninguna ventana. Ideal para RPi4 por
        SSH o para generar material para el informe/defensa del TFG.

        Si el visualizador tiene un `tracker` configurado, delega en
        `_run_export_tracking()`.

        Parámetros:
            output_path (str | None): ruta del archivo de salida. Si es
                None, se genera automáticamente en `export_dir`.

        Retorna:
            None
        """
        if self.tracker is not None:
            self._run_export_tracking(output_path)
            return

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.export_dir, f"preview_{self.detector.model_id}_{timestamp}.mp4"
            )
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.export_codec)
        writer = cv2.VideoWriter(
            output_path, fourcc, self.source_fps, (self.display_width, self.display_height)
        )

        effective_total = self._effective_total()
        frame_idx = self.start_frame - 1
        frames_written = 0
        inference_times: List[float] = []
        roi_active = self.roi_filter is not None

        logger.info("Exportando vista previa a %s", output_path)
        try:
            progress = tqdm(
                total=effective_total if effective_total > 0 else None,
                desc=f"Exportando ({self.detector.model_id})", unit="frame",
            )
            while self.max_frames is None or frames_written < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_idx += 1

                inside, outside, inference_ms = self._detect_with_roi(frame, self.detector, roi_active)
                inference_times.append(inference_ms)
                display_fps = self._tick_fps()

                counts = self._count_by_class(inside)
                annotated = self._draw_boxes(frame, inside)
                if outside:
                    annotated = self._draw_outside_boxes(annotated, outside)
                if roi_active and self.roi_filter is not None and self.roi_draw_overlay:
                    annotated = self.roi_filter.draw(annotated)

                info: Dict[str, Any] = {
                    "frame_idx": frame_idx,
                    "total_frames": self.total_frames,
                    "model_id": self.detector.model_id,
                    "inference_ms": inference_ms,
                    "display_fps": display_fps,
                    "counts": counts,
                    "total_vehicles": len(inside),
                    "paused": False,
                    "conf_threshold": self.detector.confidence_threshold,
                }
                if roi_active and self.roi_filter is not None:
                    info["roi_active"] = True
                    info["roi_inside"] = len(inside)
                    info["roi_outside"] = len(outside)
                annotated = self._draw_hud(annotated, info)
                annotated = self._resize_for_display(annotated)

                if annotated.shape[1] != self.display_width or annotated.shape[0] != self.display_height:
                    annotated = cv2.resize(annotated, (self.display_width, self.display_height))

                writer.write(annotated)
                frames_written += 1
                progress.update(1)
            progress.close()
        finally:
            writer.release()

        self._verify_export(output_path, self.detector.model_id, frames_written)

    def _run_export_tracking(self, output_path: Optional[str] = None) -> None:
        """
        Implementa `run_export()` en modo seguimiento: siempre dibuja cajas,
        estelas y HUD (según los `show_*` configurados en `config.yaml`, sin
        teclas porque no hay ventana).

        Retorna:
            None
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.export_dir,
                f"track_{self.detector.model_id}_{self.tracker.tracker_name}_{timestamp}.mp4",
            )
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.export_codec)
        writer = cv2.VideoWriter(
            output_path, fourcc, self.source_fps, (self.display_width, self.display_height)
        )

        effective_total = self._effective_total()
        frame_idx = self.start_frame - 1
        frames_written = 0

        logger.info("Exportando seguimiento a %s", output_path)
        try:
            progress = tqdm(
                total=effective_total if effective_total > 0 else None,
                desc=f"Exportando seguimiento ({self.tracker.tracker_name})", unit="frame",
            )
            while self.max_frames is None or frames_written < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_idx += 1

                annotated, _inference_ms = self._process_tracking_frame(
                    frame, frame_idx, True, True, self.show_trails, self.show_ids,
                    self.color_by_id, paused=False,
                )
                annotated = self._resize_for_display(annotated)

                if annotated.shape[1] != self.display_width or annotated.shape[0] != self.display_height:
                    annotated = cv2.resize(annotated, (self.display_width, self.display_height))

                writer.write(annotated)
                frames_written += 1
                progress.update(1)
            progress.close()
        finally:
            writer.release()

        label = f"{self.detector.model_id}+{self.tracker.tracker_name}"
        self._verify_export(output_path, label, frames_written)

    def _verify_export(self, output_path: str, label: str, frames_written: int) -> None:
        """
        Verifica que el archivo de video exportado exista y tenga contenido,
        e imprime el resultado (ruta y tamaño, o un error explicativo).

        Parámetros:
            output_path (str): ruta del archivo que se intentó generar.
            label (str): etiqueta descriptiva para los mensajes en consola.
            frames_written (int): cantidad de frames efectivamente escritos.

        Retorna:
            None
        """
        exists = os.path.isfile(output_path)
        size_bytes = os.path.getsize(output_path) if exists else 0

        if not exists or size_bytes == 0:
            print(
                f"\n✗ La exportación de '{label}' falló: el archivo no se generó o quedó "
                f"vacío ({output_path}). Es probable que el códec '{self.export_codec}' no "
                f"esté disponible en tu instalación de OpenCV. Prueba con "
                f"--codec XVID y una ruta de salida terminada en .avi."
            )
            logger.error("Exportación fallida o vacía: %s (0 bytes o inexistente)", output_path)
            return

        size_mb = size_bytes / (1024 * 1024)
        print(f"\n✓ Video exportado ({frames_written} frames): {output_path} ({size_mb:.2f} MB)")
        logger.info("Video exportado: %s (%.2f MB, %d frames)", output_path, size_mb, frames_written)

    def run_compare(
        self, detector_b: Optional[VehicleDetector] = None, export: bool = False,
        output_path: Optional[str] = None, tracker_b: Optional[VehicleTracker] = None,
    ) -> None:
        """
        Muestra (o exporta) dos modelos lado a lado sobre el mismo video, o
        —si se pasa `tracker_b`— el mismo modelo con dos trackers distintos,
        para comparar visualmente su comportamiento.

        Parámetros:
            detector_b (VehicleDetector | None): segundo detector a comparar
                contra el detector principal (`self.detector`). Ignorado si
                se pasa `tracker_b`.
            export (bool): si es True, exporta el video comparativo en vez
                de abrir una ventana interactiva.
            output_path (str | None): ruta de salida cuando `export` es True.
                Si es None, se genera automáticamente en `export_dir`.
            tracker_b (VehicleTracker | None): segundo tracker a comparar
                contra `self.tracker` (mismo modelo, dos trackers). Cuando se
                pasa, el visualizador debe tener su propio `self.tracker`
                configurado; esta llamada compara ambos en vez de comparar
                `self.detector` contra `detector_b`.

        Retorna:
            None

        Excepciones:
            ValueError: si no se pasa ni `detector_b` ni `tracker_b`, o si se
                pasa `tracker_b` sin que el visualizador tenga `self.tracker`.
        """
        if tracker_b is not None:
            if self.tracker is None:
                raise ValueError(
                    "run_compare con tracker_b requiere que el visualizador tenga su propio "
                    "tracker (self.tracker) configurado — construilo con --track."
                )
            if export:
                self._run_compare_export_tracking(tracker_b, output_path)
            else:
                self._run_compare_live_tracking(tracker_b)
            return

        if detector_b is None:
            raise ValueError("run_compare requiere detector_b o tracker_b.")

        if export:
            self._run_compare_export(detector_b, output_path)
        else:
            self._run_compare_live(detector_b)

    def _compare_display_size(self) -> Tuple[int, int]:
        """
        Calcula el tamaño final (ancho, alto) del frame combinado de
        `run_compare`, reescalado para que su ancho no supere
        `max_display_width`.

        Retorna:
            tuple: (ancho, alto) del frame combinado final.
        """
        combined_w = self.width * 2
        combined_h = self.height
        if combined_w <= self.max_display_width:
            return combined_w, combined_h
        scale = self.max_display_width / combined_w
        return int(round(combined_w * scale)), int(round(combined_h * scale))

    def _build_compare_frame(
        self, frame: np.ndarray, detector_b: VehicleDetector, show_boxes: bool, show_labels: bool,
        roi_active: bool = True,
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Construye el frame combinado lado a lado para un instante del video.

        Parámetros:
            roi_active (bool): si aplicar la ROI (cuando hay una configurada)
                a ambos modelos.

        Retorna:
            tuple: (frame combinado ya reescalado, ms modelo A, ms modelo B,
            vehículos modelo A, vehículos modelo B).
        """
        annotated_a, inside_a, outside_a, ms_a = self._annotate_side(frame, self.detector, show_boxes, roi_active)
        annotated_b, inside_b, outside_b, ms_b = self._annotate_side(frame, detector_b, show_boxes, roi_active)

        if show_labels:
            roi_active_effective = roi_active and self.roi_filter is not None
            annotated_a = self._label_compare_half(
                annotated_a, self.detector.model_id, ms_a, len(inside_a),
                len(outside_a) if roi_active_effective else None,
            )
            annotated_b = self._label_compare_half(
                annotated_b, detector_b.model_id, ms_b, len(inside_b),
                len(outside_b) if roi_active_effective else None,
            )

        combined = np.hstack([annotated_a, annotated_b])
        cv2.line(combined, (self.width, 0), (self.width, self.height), (255, 255, 255), 2)

        target_w, target_h = self._compare_display_size()
        if combined.shape[1] != target_w or combined.shape[0] != target_h:
            combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return combined, ms_a, ms_b, len(inside_a), len(inside_b)

    def _run_compare_live(self, detector_b: VehicleDetector) -> None:
        """
        Implementa `run_compare` en modo ventana interactiva.

        Controles: espacio (pausa), q/ESC (salir), s (captura), h (mostrar/
        ocultar los rótulos de cada mitad).

        Retorna:
            None
        """
        window_name = f"Comparacion - {self.detector.model_id} vs {detector_b.model_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        paused = False
        show_boxes = True
        show_labels = True
        last_combined: Optional[np.ndarray] = None
        frame_idx = self.start_frame - 1
        frames_processed = 0
        snapshots_saved = 0

        try:
            while True:
                if not paused:
                    if self.max_frames is not None and frames_processed >= self.max_frames:
                        break
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    frames_processed += 1
                    last_combined, _, _, _, _ = self._build_compare_frame(
                        frame, detector_b, show_boxes, show_labels
                    )

                if last_combined is None:
                    break

                cv2.imshow(window_name, last_combined)
                delay = 30 if paused else 1
                key = cv2.waitKey(delay) & 0xFF

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s') and last_combined is not None:
                    self._save_snapshot(last_combined, frame_idx)
                    snapshots_saved += 1
                    print(f"  Captura guardada (frame {frame_idx})")
                elif key == ord('h'):
                    show_labels = not show_labels
                elif key == ord('b'):
                    show_boxes = not show_boxes
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        print(f"\nComparación finalizada. Capturas guardadas: {snapshots_saved}")
        logger.info("run_compare (live) finalizado: %d capturas guardadas", snapshots_saved)

    def _run_compare_export(self, detector_b: VehicleDetector, output_path: Optional[str]) -> None:
        """
        Implementa `run_compare` en modo exportación a archivo.

        Retorna:
            None
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.export_dir,
                f"compare_{self.detector.model_id}_vs_{detector_b.model_id}_{timestamp}.mp4",
            )
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        target_w, target_h = self._compare_display_size()
        fourcc = cv2.VideoWriter_fourcc(*self.export_codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.source_fps, (target_w, target_h))

        effective_total = self._effective_total()
        frames_written = 0

        logger.info("Exportando comparación a %s", output_path)
        try:
            progress = tqdm(
                total=effective_total if effective_total > 0 else None,
                desc=f"Exportando comparacion ({self.detector.model_id} vs {detector_b.model_id})",
                unit="frame",
            )
            while self.max_frames is None or frames_written < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                combined, _, _, _, _ = self._build_compare_frame(frame, detector_b, True, True)
                writer.write(combined)
                frames_written += 1
                progress.update(1)
            progress.close()
        finally:
            writer.release()

        label = f"{self.detector.model_id} vs {detector_b.model_id}"
        self._verify_export(output_path, label, frames_written)

    def _build_compare_frame_tracking(
        self, frame: np.ndarray, frame_idx: int, tracker_b: VehicleTracker,
        show_boxes: bool, show_labels: bool, show_trails: bool, show_ids: bool, color_by_id: bool,
        trails_a: Dict[int, deque], last_seen_a: Dict[int, int],
        trails_b: Dict[int, deque], last_seen_b: Dict[int, int],
    ) -> Tuple[np.ndarray, float, float, int, int]:
        """
        Construye el frame combinado lado a lado para la comparación de dos
        trackers (`self.tracker` vs `tracker_b`) sobre el mismo modelo y video.

        Cada lado mantiene su propio estado de estelas (`trails_a`/`trails_b`)
        porque los `track_id` de dos trackers independientes no son
        comparables entre sí (el ID 3 de ByteTrack no tiene relación con el
        ID 3 de BoTSORT).

        Retorna:
            tuple: (frame combinado ya reescalado, ms tracker A, ms tracker B,
            tracks activos A, tracks activos B).
        """
        t0 = time.perf_counter()
        dets_a = self.tracker.update(frame, frame_idx)
        ms_a = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        dets_b = tracker_b.update(frame, frame_idx)
        ms_b = (time.perf_counter() - t0) * 1000.0

        active_a = {d["track_id"] for d in dets_a}
        active_b = {d["track_id"] for d in dets_b}
        self._update_trails(dets_a, frame_idx, trails_a, last_seen_a)
        self._update_trails(dets_b, frame_idx, trails_b, last_seen_b)

        frame_a = frame.copy()
        frame_b = frame.copy()

        if show_trails:
            frame_a = self._draw_trails(frame_a, active_a, frame_idx, trails_a, last_seen_a)
            frame_b = self._draw_trails(frame_b, active_b, frame_idx, trails_b, last_seen_b)
        if show_boxes:
            frame_a = self._draw_tracks(frame_a, dets_a, color_by_id, show_ids, self.tracker)
            frame_b = self._draw_tracks(frame_b, dets_b, color_by_id, show_ids, tracker_b)

        if show_labels:
            frame_a = self._label_compare_tracking_half(
                frame_a, self.tracker.tracker_name, ms_a, len(active_a), len(self.tracker.tracks),
            )
            frame_b = self._label_compare_tracking_half(
                frame_b, tracker_b.tracker_name, ms_b, len(active_b), len(tracker_b.tracks),
            )

        combined = np.hstack([frame_a, frame_b])
        cv2.line(combined, (self.width, 0), (self.width, self.height), (255, 255, 255), 2)

        target_w, target_h = self._compare_display_size()
        if combined.shape[1] != target_w or combined.shape[0] != target_h:
            combined = cv2.resize(combined, (target_w, target_h), interpolation=cv2.INTER_AREA)

        return combined, ms_a, ms_b, len(active_a), len(active_b)

    def _run_compare_live_tracking(self, tracker_b: VehicleTracker) -> None:
        """
        Implementa `run_compare(tracker_b=...)` en modo ventana interactiva:
        el mismo modelo con dos trackers distintos, lado a lado, cada mitad
        con su propio conteo de IDs activos y totales.

        Controles: espacio (pausa), q/ESC (salir), s (captura), h (rótulos),
        b (cajas). Igual que `_run_compare_live`, no incluye los ajustes de
        estela/color de `run_live` para mantener el control simple.

        Retorna:
            None
        """
        window_name = f"Comparacion trackers - {self.tracker.tracker_name} vs {tracker_b.tracker_name}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        paused = False
        show_boxes = True
        show_labels = True
        last_combined: Optional[np.ndarray] = None
        frame_idx = self.start_frame - 1
        frames_processed = 0
        snapshots_saved = 0

        trails_a: Dict[int, deque] = {}
        last_seen_a: Dict[int, int] = {}
        trails_b: Dict[int, deque] = {}
        last_seen_b: Dict[int, int] = {}

        try:
            while True:
                if not paused:
                    if self.max_frames is not None and frames_processed >= self.max_frames:
                        break
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    frame_idx += 1
                    frames_processed += 1
                    last_combined, _, _, _, _ = self._build_compare_frame_tracking(
                        frame, frame_idx, tracker_b, show_boxes, show_labels,
                        self.show_trails, self.show_ids, self.color_by_id,
                        trails_a, last_seen_a, trails_b, last_seen_b,
                    )

                if last_combined is None:
                    break

                cv2.imshow(window_name, last_combined)
                delay = 30 if paused else 1
                key = cv2.waitKey(delay) & 0xFF

                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                if key in (ord('q'), 27):
                    break
                elif key == ord(' '):
                    paused = not paused
                elif key == ord('s') and last_combined is not None:
                    self._save_snapshot(last_combined, frame_idx)
                    snapshots_saved += 1
                    print(f"  Captura guardada (frame {frame_idx})")
                elif key == ord('h'):
                    show_labels = not show_labels
                elif key == ord('b'):
                    show_boxes = not show_boxes
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

        print(f"\nComparación de trackers finalizada. Capturas guardadas: {snapshots_saved}")
        print(f"  IDs totales — {self.tracker.tracker_name}: {len(self.tracker.tracks)} | "
              f"{tracker_b.tracker_name}: {len(tracker_b.tracks)}")
        logger.info(
            "run_compare tracking (live) finalizado: %d capturas, IDs %s=%d %s=%d",
            snapshots_saved, self.tracker.tracker_name, len(self.tracker.tracks),
            tracker_b.tracker_name, len(tracker_b.tracks),
        )

    def _run_compare_export_tracking(self, tracker_b: VehicleTracker, output_path: Optional[str]) -> None:
        """
        Implementa `run_compare(tracker_b=..., export=True)`: exporta el
        video comparativo de dos trackers a un archivo.

        Retorna:
            None
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.export_dir,
                f"compare_tracker_{self.tracker.tracker_name}_vs_{tracker_b.tracker_name}_{timestamp}.mp4",
            )
        else:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        target_w, target_h = self._compare_display_size()
        fourcc = cv2.VideoWriter_fourcc(*self.export_codec)
        writer = cv2.VideoWriter(output_path, fourcc, self.source_fps, (target_w, target_h))

        effective_total = self._effective_total()
        frame_idx = self.start_frame - 1
        frames_written = 0

        trails_a: Dict[int, deque] = {}
        last_seen_a: Dict[int, int] = {}
        trails_b: Dict[int, deque] = {}
        last_seen_b: Dict[int, int] = {}

        logger.info("Exportando comparación de trackers a %s", output_path)
        try:
            progress = tqdm(
                total=effective_total if effective_total > 0 else None,
                desc=f"Exportando comparacion ({self.tracker.tracker_name} vs {tracker_b.tracker_name})",
                unit="frame",
            )
            while self.max_frames is None or frames_written < self.max_frames:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_idx += 1
                combined, _, _, _, _ = self._build_compare_frame_tracking(
                    frame, frame_idx, tracker_b, True, True,
                    self.show_trails, self.show_ids, self.color_by_id,
                    trails_a, last_seen_a, trails_b, last_seen_b,
                )
                writer.write(combined)
                frames_written += 1
                progress.update(1)
            progress.close()
        finally:
            writer.release()

        label = f"{self.tracker.tracker_name} vs {tracker_b.tracker_name}"
        self._verify_export(output_path, label, frames_written)

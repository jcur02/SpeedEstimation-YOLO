"""
Región de interés (ROI): delimita la zona válida de la escena para reducir
ruido y falsos positivos en las detecciones (carril contrario, parqueos,
calles de fondo que entran en el encuadre de la cámara).

Aclaración conceptual importante: la ROI **no acelera la inferencia**. YOLO
redimensiona su entrada a un tamaño fijo (`imgsz`), así que el costo de
cómputo no depende del área de la escena. El beneficio de la ROI es
**reducir ruido y falsos positivos**, no ganar FPS. Cualquier ganancia de
rendimiento vendría de bajar `imgsz`, saltar frames o usar un modelo más
liviano — no de recortar la escena.

La única excepción es el modo `crop`, que sí puede **mejorar la precisión**
(no la velocidad): al recortar y reescalar la zona de interés, los vehículos
lejanos ocupan más píxeles de la entrada de la red y se detectan mejor.
"""
import hashlib
import json
import logging
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def save_roi(
    path: str,
    polygon_norm: List[List[float]],
    frame_size: Tuple[int, int],
    source_videos: List[str],
    reference_frame_md5: str,
    homography_points: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Guarda la ROI en disco como JSON, con coordenadas normalizadas 0.0-1.0.

    Si ya existe un archivo en `path`, se hace una copia de respaldo antes de
    sobrescribir, en `{nombre}.backup_{YYYYMMDD_HHMMSS}.json`.

    Parámetros:
        path (str): ruta donde guardar el archivo (ej. "config/roi.json").
        polygon_norm (list[list[float]]): vértices del polígono, normalizados
            en el rango [0, 1] respecto al ancho/alto del frame.
        frame_size (tuple[int, int]): `(ancho, alto)` en píxeles del frame
            usado para marcar la ROI, guardado como referencia.
        source_videos (list[str]): rutas de los videos usados para calibrar
            (marcar) esta ROI.
        reference_frame_md5 (str): hash MD5 del frame de referencia (ver
            `md5_of_frame`), para detectar más adelante si la cámara se movió.
        homography_points (dict | None): bloque `homography_points` a incluir.
            Si es None, se guarda un bloque vacío pendiente de calibración.

    Retorna:
        None
    """
    if os.path.isfile(path):
        backup_path = f"{os.path.splitext(path)[0]}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        shutil.copy2(path, backup_path)
        print(f"  Copia de respaldo de la ROI anterior: {backup_path}")
        logger.info("Backup de ROI anterior creado en %s", backup_path)

    data = {
        "version": 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_videos": list(source_videos),
        "frame_size": [int(frame_size[0]), int(frame_size[1])],
        "reference_frame_md5": reference_frame_md5,
        "roi_polygon": [[float(x), float(y)] for x, y in polygon_norm],
        "homography_points": homography_points or {
            "image_points": [],
            "world_points": None,
            "notes": "Pendiente: completar world_points en la fase de calibración.",
        },
    }

    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("ROI guardada en %s (%d vértices)", path, len(polygon_norm))


def load_roi(path: str) -> Dict[str, Any]:
    """
    Carga y valida un archivo de ROI generado por `save_roi` /
    `scripts/select_roi.py`.

    Parámetros:
        path (str): ruta al archivo JSON de la ROI.

    Retorna:
        dict: el contenido completo del archivo.

    Excepciones:
        FileNotFoundError: si el archivo no existe.
        ValueError: si el JSON está corrupto o el polígono es inválido
            (menos de 3 vértices, coordenadas fuera de [0, 1]).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No se encontró el archivo de ROI: '{path}'. Generalo primero con: "
            f"python scripts/select_roi.py"
        )

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"El archivo de ROI '{path}' está corrupto o no es JSON válido: {exc}") from exc

    polygon = data.get("roi_polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        found = len(polygon) if isinstance(polygon, list) else 0
        raise ValueError(
            f"El polígono en '{path}' es inválido: se requieren al menos 3 vértices, "
            f"se encontraron {found}."
        )

    for point in polygon:
        if not (isinstance(point, (list, tuple)) and len(point) == 2):
            raise ValueError(f"El polígono en '{path}' tiene un vértice mal formado: {point!r}")
        x, y = point
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise ValueError(
                f"Las coordenadas del polígono deben estar normalizadas en [0, 1]; "
                f"se encontró el punto ({x}, {y}) en '{path}'."
            )

    return data


def md5_of_frame(frame: np.ndarray) -> str:
    """
    Calcula un hash MD5 robusto de un frame, para detectar si la cámara se
    movió respecto a cuando se calibró la ROI.

    Reescala el frame a 160x90 en escala de grises antes de hashear, para que
    pequeñas diferencias de compresión/resolución entre capturas no generen
    un hash distinto.

    Parámetros:
        frame (numpy.ndarray): imagen BGR (formato OpenCV).

    Retorna:
        str: hash MD5 en hexadecimal.
    """
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
    return hashlib.md5(gray.tobytes()).hexdigest()


def reference_frame_path(roi_json_path: str) -> str:
    """
    Calcula la ruta del thumbnail de referencia asociado a un archivo de ROI.

    El JSON de la ROI solo guarda el *hash* del frame de referencia (ver
    `md5_of_frame`), que no es reversible: no alcanza para calcular una
    diferencia de píxeles tolerante más adelante (`select_roi.py --show`).
    Por eso, junto al JSON se guarda además este thumbnail en escala de
    grises (160x90, el mismo tamaño usado para el hash) como archivo PNG
    aparte, para poder comparar con margen de tolerancia si la cámara se
    movió, no solo con una igualdad binaria de hash.

    Parámetros:
        roi_json_path (str): ruta al archivo `roi.json`.

    Retorna:
        str: ruta del PNG de referencia (mismo nombre base, sufijo
        `_reference.png`).
    """
    return f"{os.path.splitext(roi_json_path)[0]}_reference.png"


def polygon_area_ratio(polygon_norm: List[List[float]]) -> float:
    """
    Calcula qué fracción del frame cubre un polígono, a partir de sus
    coordenadas normalizadas — sin necesitar la resolución real del video.

    Como las coordenadas ya están normalizadas en [0, 1], el área calculada
    con la fórmula del shoelace sobre ellas es directamente la fracción del
    área del frame (escalar x por el ancho e y por el alto multiplica el
    área por ancho*alto en ambos casos, así que la razón no cambia).

    Parámetros:
        polygon_norm (list[list[float]]): vértices normalizados del polígono.

    Retorna:
        float: fracción del área del frame cubierta, en [0, 1].
    """
    n = len(polygon_norm)
    area2 = 0.0
    for i in range(n):
        x1, y1 = polygon_norm[i]
        x2, y2 = polygon_norm[(i + 1) % n]
        area2 += x1 * y2 - x2 * y1
    return abs(area2) / 2.0


class ROIFilter:
    """
    Aplica una región de interés ya calibrada sobre las detecciones y/o los
    frames de un video, en la resolución concreta con la que se está
    procesando ese video.
    """

    def __init__(self, roi_data: Dict[str, Any], frame_size: Tuple[int, int], config: Dict[str, Any]) -> None:
        """
        Desnormaliza el polígono a la resolución de trabajo y precalcula las
        estructuras que dependen de ella (rectángulo envolvente, máscara).

        Parámetros:
            roi_data (dict): dict retornado por `load_roi()`.
            frame_size (tuple[int, int]): `(ancho, alto)` en píxeles del
                video que se va a procesar ahora (puede diferir de la
                resolución con la que se marcó la ROI originalmente).
            config (dict): configuración completa del proyecto, de donde se
                lee la subsección `roi` (modo, punto de tierra, tolerancia).

        Retorna:
            None
        """
        self.roi_data = roi_data
        self.frame_width, self.frame_height = int(frame_size[0]), int(frame_size[1])

        roi_cfg = config.get("roi", {})
        self.mode = roi_cfg.get("mode", "filter")
        self.ground_point_mode = roi_cfg.get("ground_point", "bottom_center")
        self.tolerance_px = roi_cfg.get("tolerance_px", 0)
        self.overlay_alpha = roi_cfg.get("overlay_alpha", 0.20)

        self.polygon_norm: List[List[float]] = roi_data["roi_polygon"]
        self.polygon_px = np.array(
            [
                [int(round(x * self.frame_width)), int(round(y * self.frame_height))]
                for x, y in self.polygon_norm
            ],
            dtype=np.int32,
        )

        self.bounding_rect: Tuple[int, int, int, int] = cv2.boundingRect(self.polygon_px)

        self._mask: Optional[np.ndarray] = None
        if self.mode == "mask":
            self._mask = self._build_mask()

    def _build_mask(self) -> np.ndarray:
        """
        Construye (una sola vez) la máscara binaria del polígono, usada por
        el modo `mask`.

        Retorna:
            numpy.ndarray: máscara `uint8` de tamaño `(alto, ancho)`, con 255
            dentro del polígono y 0 fuera.
        """
        mask = np.zeros((self.frame_height, self.frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [self.polygon_px], 255)
        return mask

    def ground_point(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calcula el punto del bbox usado para la prueba dentro/fuera de la ROI.

        Parámetros:
            bbox (list[float]): `[x1, y1, x2, y2]` de la detección.

        Retorna:
            tuple[float, float]: el punto a usar, según
            `config.roi.ground_point`:
              - `"bottom_center"` (default): `((x1+x2)/2, y2)`, aproxima
                dónde las llantas tocan el pavimento.
              - `"centroid"`: `((x1+x2)/2, (y1+y2)/2)`.
        """
        x1, y1, x2, y2 = bbox
        if self.ground_point_mode == "centroid":
            return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        # bottom_center es el default porque con el centroide, un camión alto
        # circulando en el carril contrario proyecta su centro sobre el
        # carril de interés y entra como falso positivo; con el punto
        # inferior (donde las llantas tocan el pavimento), no.
        return ((x1 + x2) / 2.0, y2)

    def contains(self, bbox: List[float]) -> bool:
        """
        Determina si una detección cae dentro de la ROI.

        Parámetros:
            bbox (list[float]): `[x1, y1, x2, y2]` de la detección.

        Retorna:
            bool: True si el punto de tierra está dentro del polígono o a
            una distancia `>= -tolerance_px` de su borde (distancia negativa
            = fuera). Con `tolerance_px = 0`, estrictamente dentro-o-en-el-borde.
        """
        point = self.ground_point(bbox)
        distance = cv2.pointPolygonTest(self.polygon_px, point, True)
        return distance >= -self.tolerance_px

    def filter(self, detections: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Separa una lista de detecciones en las que caen dentro y fuera de la ROI.

        Parámetros:
            detections (list[dict]): detecciones retornadas por `detect()`.

        Retorna:
            tuple[list[dict], list[dict]]: `(dentro, fuera)`. No modifica las
            detecciones originales.
        """
        inside, outside = [], []
        for det in detections:
            (inside if self.contains(det["bbox"]) else outside).append(det)
        return inside, outside

    def crop(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Recorta el frame al rectángulo envolvente del polígono de la ROI.

        Importante: quien use este modo debe sumar el `offset` retornado a
        las coordenadas de los bboxes detectados sobre el recorte, para
        devolverlas al sistema de coordenadas del frame completo — usa
        `restore_bboxes()` para eso.

        Parámetros:
            frame (numpy.ndarray): frame completo original.

        Retorna:
            tuple[numpy.ndarray, tuple[int, int]]: `(frame_recortado, offset)`,
            donde `offset` es `(x, y)` de la esquina superior izquierda del
            rectángulo envolvente dentro del frame original.
        """
        x, y, w, h = self.bounding_rect
        return frame[y:y + h, x:x + w].copy(), (x, y)

    def restore_bboxes(self, detections: List[Dict[str, Any]], offset: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Traslada los bboxes de detecciones hechas sobre un recorte (`crop()`)
        de vuelta al sistema de coordenadas del frame completo.

        Parámetros:
            detections (list[dict]): detecciones obtenidas sobre el frame
                recortado.
            offset (tuple[int, int]): el mismo `offset` retornado por `crop()`.

        Retorna:
            list[dict]: nuevas detecciones con `bbox` trasladado; no modifica
            las originales.
        """
        ox, oy = offset
        restored = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            new_det = dict(det)
            new_det["bbox"] = [x1 + ox, y1 + oy, x2 + ox, y2 + oy]
            restored.append(new_det)
        return restored

    def mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Ennegrece todo lo exterior al polígono de la ROI.

        Advertencia: este modo suele **empeorar** las detecciones cercanas al
        borde del polígono, porque produce una imagen fuera de la
        distribución de entrenamiento de la red (bordes duros negros que
        YOLO no vio en entrenamiento). Está incluido para poder compararlo
        empíricamente contra `filter`/`crop`, no como recomendación por defecto.

        Parámetros:
            frame (numpy.ndarray): frame completo original.

        Retorna:
            numpy.ndarray: copia del frame con el exterior del polígono en negro.
        """
        if self._mask is None:
            self._mask = self._build_mask()
        return cv2.bitwise_and(frame, frame, mask=self._mask)

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Dibuja el polígono de la ROI sobre una copia del frame: relleno
        translúcido más el contorno.

        Parámetros:
            frame (numpy.ndarray): frame sobre el que dibujar.

        Retorna:
            numpy.ndarray: copia del frame con la ROI dibujada encima.
        """
        annotated = frame.copy()
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.polygon_px], (0, 255, 255))
        annotated = cv2.addWeighted(overlay, self.overlay_alpha, annotated, 1 - self.overlay_alpha, 0)
        cv2.polylines(annotated, [self.polygon_px], isClosed=True, color=(0, 255, 255), thickness=2)
        return annotated

    def area_ratio(self) -> float:
        """
        Retorna qué fracción del frame cubre la ROI.

        Retorna:
            float: área del polígono / área total del frame, en [0, 1].
        """
        return polygon_area_ratio(self.polygon_norm)

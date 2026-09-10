"""
Mapa de calor de tráfico: acumula los puntos de contacto de todas las
detecciones a lo largo de uno o varios videos, para mostrar por dónde
circulan realmente los vehículos. El polígono de la ROI se traza sobre esa
evidencia en vez de sobre la intuición visual del usuario.
"""
import logging
import os
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class TrafficHeatmap:
    """
    Acumula intensidad en los puntos de contacto (bottom-center) de las
    detecciones de uno o varios videos del mismo sitio, para renderizar un
    mapa de calor de dónde pasan los vehículos.
    """

    def __init__(self, config: Dict[str, Any], detector: Any) -> None:
        """
        Prepara el acumulador (vacío hasta el primer frame procesado) y los
        contadores de progreso.

        Parámetros:
            config (dict): configuración completa del proyecto. Se usa la
                subsección `roi.heatmap`.
            detector (VehicleDetector): detector ya construido, usado para
                acumular. Su `confidence_threshold` se sobreescribe
                temporalmente por `roi.heatmap.confidence` durante
                `accumulate()`, y se restaura al terminar.

        Retorna:
            None
        """
        self.detector = detector
        self.heatmap_cfg = config.get("roi", {}).get("heatmap", {})

        self.accumulator: "np.ndarray | None" = None
        self.frames_sampled = 0
        self.detections_total = 0
        self.videos_processed = 0

    def _frame_indices(self, total_frames: int, source_fps: float) -> List[int]:
        """
        Calcula qué índices de frame muestrear, según `roi.heatmap.sample_mode`.

        Parámetros:
            total_frames (int): cantidad total de frames del video.
            source_fps (float): FPS del video (usado por el modo "range").

        Retorna:
            list[int]: índices de frame (0-based) a muestrear, en orden
            ascendente. Lista vacía si `total_frames <= 0`.
        """
        if total_frames <= 0:
            return []

        sample_mode = self.heatmap_cfg.get("sample_mode", "uniform")
        samples = max(1, self.heatmap_cfg.get("samples", 300))

        if sample_mode == "middle":
            # Ventana contigua centrada en la mitad del video.
            window = min(samples, total_frames)
            center = total_frames // 2
            start = max(0, center - window // 2)
            end = min(total_frames, start + window)
            return list(range(start, end))

        if sample_mode == "range":
            start_sec = self.heatmap_cfg.get("range_start_sec", 0.0)
            end_sec = self.heatmap_cfg.get("range_end_sec", 0.0)
            fps = source_fps or 30.0
            start_frame = max(0, int(round(start_sec * fps)))
            end_frame = total_frames if end_sec <= 0 else min(total_frames, int(round(end_sec * fps)))
            if end_frame <= start_frame:
                return []
            span = end_frame - start_frame
            count = min(samples, span)
            stride = max(1, span // count)
            return list(range(start_frame, end_frame, stride))[:count]

        # "uniform" (default y recomendado): reparte `samples` frames a lo
        # largo de TODO el video. Es el modo correcto cuando la actividad
        # puede estar en cualquier segundo (se esperó a que pasara un
        # vehículo): una ventana contigua (inicio/medio/final) podría caer
        # justo en un tramo vacío.
        stride = max(1, total_frames // samples)
        return list(range(0, total_frames, stride))[:samples]

    def accumulate(self, video_path: str) -> int:
        """
        Acumula las detecciones de un video en el mapa de calor.

        No usa seeking: lee el video secuencialmente con `cap.read()` y solo
        corre inferencia cuando el índice actual está en el conjunto de
        índices objetivo (`cv2.CAP_PROP_POS_FRAMES` es lento y poco fiable
        con ciertos codecs; leer secuencial y descartar es más rápido y
        robusto).

        Parámetros:
            video_path (str): ruta al video a procesar.

        Retorna:
            int: cantidad de detecciones acumuladas en este video.

        Excepciones:
            FileNotFoundError: si el video no existe.
            RuntimeError: si OpenCV no puede abrirlo.
            ValueError: si su resolución no coincide con la del acumulador
                ya iniciado por un video anterior (no se pueden mezclar
                videos de distinta resolución en el mismo mapa de calor).
        """
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No se encontró el video para el mapa de calor: '{video_path}'.")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video para el mapa de calor: '{video_path}'.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.accumulator is None:
            self.accumulator = np.zeros((height, width), dtype=np.float32)
        elif self.accumulator.shape != (height, width):
            cap.release()
            raise ValueError(
                f"El video '{video_path}' tiene resolución {width}x{height}, distinta a la del "
                f"acumulador ya iniciado ({self.accumulator.shape[1]}x{self.accumulator.shape[0]}). "
                f"No se pueden mezclar videos de distinta resolución en el mismo mapa de calor."
            )

        target_indices = set(self._frame_indices(total_frames, source_fps))
        original_conf = self.detector.confidence_threshold
        self.detector.confidence_threshold = self.heatmap_cfg.get("confidence", 0.25)

        detections_this_video = 0
        frame_idx = 0

        try:
            with tqdm(
                total=total_frames if total_frames > 0 else None,
                desc=f"Muestreando ({os.path.basename(video_path)})", unit="frame",
            ) as progress:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx in target_indices:
                        detections = self.detector.detect(frame)
                        if detections:
                            layer = np.zeros_like(self.accumulator)
                            for det in detections:
                                x1, y1, x2, y2 = det["bbox"]
                                cx = int(round((x1 + x2) / 2.0))
                                cy = int(round(y2))
                                radius = max(4, int(round((x2 - x1) / 8.0)))
                                cv2.circle(layer, (cx, cy), radius, 1.0, -1)
                            self.accumulator += layer
                        detections_this_video += len(detections)
                        self.frames_sampled += 1
                    frame_idx += 1
                    progress.update(1)
        finally:
            cap.release()
            self.detector.confidence_threshold = original_conf

        self.detections_total += detections_this_video
        self.videos_processed += 1
        logger.info(
            "Video %s: %d detecciones acumuladas sobre %d frames muestreados",
            video_path, detections_this_video, len(target_indices),
        )
        return detections_this_video

    def accumulate_many(self, video_paths: List[str]) -> int:
        """
        Acumula varios videos del mismo sitio en el mismo mapa de calor.

        Esta es la funcionalidad clave para videos cortos: si cada grabación
        captura solo dos o tres vehículos, acumular seis u ocho videos del
        mismo sitio da una nube de puntos suficiente para trazar el polígono
        con criterio.

        Parámetros:
            video_paths (list[str]): rutas de los videos a acumular.

        Retorna:
            int: total de detecciones acumuladas en todos los videos.
        """
        total = 0
        for path in video_paths:
            count = self.accumulate(path)
            print(f"  {os.path.basename(path)}: {count} detecciones acumuladas")
            total += count
        print(f"  Total: {total} detecciones en {len(video_paths)} video(s)")
        return total

    def render(self, background_frame: np.ndarray) -> np.ndarray:
        """
        Renderiza el acumulador como un mapa de calor mezclado sobre un frame
        de fondo.

        Parámetros:
            background_frame (numpy.ndarray): frame BGR de fondo, debe tener
                la misma resolución que el acumulador.

        Retorna:
            numpy.ndarray: frame con el mapa de calor superpuesto. Las zonas
            de intensidad cero quedan sin tinte (se ve el fondo limpio donde
            nunca pasó nada).

        Excepciones:
            RuntimeError: si todavía no se acumuló ningún video.
        """
        if self.accumulator is None:
            raise RuntimeError(
                "No hay datos acumulados: llamá a accumulate()/accumulate_many() antes de render()."
            )

        normalized = cv2.normalize(self.accumulator, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        kernel = self.heatmap_cfg.get("blur_kernel", 41)
        if kernel % 2 == 0:
            kernel += 1
        blurred = cv2.GaussianBlur(normalized, (kernel, kernel), 0)

        colormap_name = self.heatmap_cfg.get("colormap", "COLORMAP_JET")
        colormap = getattr(cv2, colormap_name, cv2.COLORMAP_JET)
        colored = cv2.applyColorMap(blurred, colormap)

        alpha = self.heatmap_cfg.get("alpha", 0.45)
        blended = cv2.addWeighted(colored, alpha, background_frame, 1 - alpha, 0)

        result = background_frame.copy()
        mask = blurred > 5
        result[mask] = blended[mask]
        return result

    def save(self, path: str, background_frame: np.ndarray) -> str:
        """
        Renderiza el mapa de calor y lo guarda como PNG.

        Parámetros:
            path (str): ruta de salida.
            background_frame (numpy.ndarray): frame de fondo para el render.

        Retorna:
            str: la misma ruta de salida.
        """
        rendered = self.render(background_frame)
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(path, rendered)
        print(f"  Mapa de calor guardado: {path}")
        logger.info("Mapa de calor guardado en %s", path)
        return path

    def is_empty(self) -> bool:
        """
        Retorna:
            bool: True si no se acumuló ninguna detección todavía.
        """
        return self.detections_total == 0

    def stats(self) -> Dict[str, Any]:
        """
        Retorna:
            dict: `frames_sampled`, `detections_total`, `videos_processed` y
            `detections_per_sampled_frame`.
        """
        return {
            "frames_sampled": self.frames_sampled,
            "detections_total": self.detections_total,
            "videos_processed": self.videos_processed,
            "detections_per_sampled_frame": (
                self.detections_total / self.frames_sampled if self.frames_sampled > 0 else 0.0
            ),
        }

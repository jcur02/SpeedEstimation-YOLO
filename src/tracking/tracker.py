"""
Envuelve un modelo YOLO en modo seguimiento (ByteTrack / BoT-SORT) y acumula
las trayectorias resultantes en objetos `Track`.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

from src.tracking.track import Track, TimeBase, smooth_ground_points

logger = logging.getLogger(__name__)

_VALID_TRACKERS = ("bytetrack", "botsort")


class VehicleTracker:
    """
    Corre un modelo YOLO en `mode="track"` (ByteTrack o BoT-SORT, ambos ya
    integrados en Ultralytics — no se reimplementa ningún algoritmo de
    seguimiento a mano) y acumula las trayectorias de cada vehículo en
    objetos `Track`, indexados por `track_id`.
    """

    def __init__(
        self, model_id: str, weights: str, tracker_name: str, config: Dict[str, Any],
        source_fps: float, roi_filter: Optional[Any] = None,
    ) -> None:
        """
        Carga un modelo YOLO nuevo y lo prepara para seguimiento.

        Parámetros:
            model_id (str): identificador corto del modelo (ej. "yolov8n").
            weights (str): nombre del archivo de pesos, buscado/descargado
                dentro de `paths.models`.
            tracker_name (str): `"bytetrack"` o `"botsort"`. Resuelve la ruta
                a `config/trackers/{tracker_name}.yaml`.
            config (dict): configuración completa cargada desde config.yaml.
            source_fps (float): FPS del video de origen, usados para
                construir la `TimeBase` de este tracker.
            roi_filter (ROIFilter | None): filtro de región de interés ya
                construido para la resolución del video. Si no es None,
                `update()` descarta (no crea ni alimenta ningún Track para)
                las detecciones cuyo punto de contacto caiga fuera del
                polígono. None desactiva el filtrado por completo.

        Retorna:
            None

        Excepciones:
            ValueError: si `tracker_name` no es `"bytetrack"` ni `"botsort"`,
                o si `source_fps <= 0` (propagada desde `TimeBase`).
            FileNotFoundError: si no existe
                `config/trackers/{tracker_name}.yaml`.
        """
        if tracker_name not in _VALID_TRACKERS:
            raise ValueError(
                f"Tracker '{tracker_name}' no reconocido. Válidos: {', '.join(_VALID_TRACKERS)}."
            )

        tracking_cfg = config.get("tracking", {})
        tracker_dir = tracking_cfg.get("tracker_config_dir", "config/trackers/")
        self.tracker_cfg_path = os.path.join(tracker_dir, f"{tracker_name}.yaml")
        if not os.path.isfile(self.tracker_cfg_path):
            raise FileNotFoundError(
                f"No se encontró la configuración del tracker '{tracker_name}' en "
                f"'{self.tracker_cfg_path}'. Verificá config.tracking.tracker_config_dir "
                f"o que el archivo config/trackers/{tracker_name}.yaml exista."
            )

        self.model_id = model_id
        self.weights = weights
        self.tracker_name = tracker_name
        self.config = config

        detection_cfg = config.get("detection", {})
        self.vehicle_classes = detection_cfg.get("vehicle_classes", {})
        self.vehicle_class_ids = list(self.vehicle_classes.keys())
        self.conf = detection_cfg.get("confidence_threshold", 0.40)
        self.iou = detection_cfg.get("iou_threshold", 0.45)
        # Ver VehicleDetector.unify_vehicle_classes: mismo flag, mismo significado.
        # class_id (abajo, en update()) se conserva sin tocar en los dos casos.
        self.unify_vehicle_classes = detection_cfg.get("unify_vehicle_classes", False)
        self.device = config.get("benchmark", {}).get("device", "cpu")

        models_dir = config.get("paths", {}).get("models", "models/")
        os.makedirs(models_dir, exist_ok=True)
        weights_path = os.path.join(models_dir, weights)

        # Decisión de diseño obligatoria (C): instancia NUEVA de YOLO por
        # cada combinación (modelo, tracker). Los trackers de Ultralytics
        # mantienen estado interno entre llamadas (persist=True): si se
        # reutilizara la misma instancia de YOLO para dos corridas distintas,
        # los IDs se arrastrarían de una corrida a la otra y el conteo de
        # vehículos únicos saldría inflado. Por eso este constructor SIEMPRE
        # crea un YOLO(...) propio — nunca lo recibe como parámetro ni lo
        # comparte con otro VehicleTracker.
        logger.info(
            "Cargando modelo %s (%s) con tracker %s en %s...",
            model_id, weights_path, tracker_name, self.device,
        )
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        try:
            self.num_params = sum(p.numel() for p in self.model.model.parameters())
        except Exception as exc:
            logger.warning("No se pudo contar parámetros del modelo %s: %s", model_id, exc)
            self.num_params = 0

        self.time_base = TimeBase(source_fps)
        self.source_fps = source_fps

        # ROI: se aplica en update(), justo después de extraer cada detección y antes
        # de crear/actualizar su Track (ver abajo) — descartando las detecciones cuyo
        # punto de contacto caiga fuera del polígono, igual que ya hace
        # `DetectionVisualizer._detect_with_roi` para el modo de solo-detección.
        self.roi_filter = roi_filter

        self.tracks: Dict[int, Track] = {}
        self.frames_processed = 0

        # Metadata para `save_tracks()`, seteada por el script que use esta
        # clase (ver scripts/run_tracking_benchmark.py) una vez que conoce el
        # video y su resolución.
        self.source_video: Optional[str] = None
        self.frame_size: Optional[Tuple[int, int]] = None

    def update(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Procesa un frame en modo seguimiento y retorna sus detecciones con ID.

        Parámetros:
            frame (numpy.ndarray): imagen BGR (formato OpenCV).
            frame_idx (int): índice del frame, usado para calcular `t_video`
                con la `TimeBase` de este tracker.

        Retorna:
            list[dict]: una entrada por track activo en este frame, con la
            misma forma que `VehicleDetector.detect()` más la clave
            `"track_id"`:
                {
                    "class_id": int, "class_name": str, "confidence": float,
                    "bbox": [x1, y1, x2, y2], "track_id": int,
                }
            Lista vacía si no hay ningún track activo en este frame — es el
            caso NORMAL (nadie en escena momentáneamente), no un error.
        """
        results = self.model.track(
            source=frame,
            persist=True,  # obligatorio al alimentar frame por frame: mantiene el estado del tracker
            tracker=self.tracker_cfg_path,
            conf=self.conf,
            iou=self.iou,
            classes=self.vehicle_class_ids,
            device=self.device,
            # Mismo motivo que en VehicleDetector.detect(): sin esto, dos cajas de clases
            # distintas sobre un mismo vehículo no se suprimen entre sí y el tracker les
            # abre un track_id separado a cada una (trayectorias fantasma).
            agnostic_nms=True,
            verbose=False,
        )

        self.frames_processed += 1

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            # Caso más común de crash en este tipo de código si no se
            # maneja explícitamente: boxes.id es None cuando no hay ningún
            # track activo en el frame (video vacío en ese instante, o
            # ningún vehículo superó new_track_thresh todavía).
            return []

        ids = boxes.id.int().cpu().tolist()
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.int().cpu().tolist()

        t_video = self.time_base.time_of(frame_idx)
        detections: List[Dict[str, Any]] = []

        for track_id, bbox, conf, class_id in zip(ids, xyxy, confs, clss):
            if class_id not in self.vehicle_classes:
                continue

            bbox_list = [float(v) for v in bbox]

            if self.roi_filter is not None and not self.roi_filter.contains(bbox_list):
                # Fuera de la zona calibrada: no se crea ni alimenta ningún Track con
                # esta detección, así nunca llega a formar parte de ninguna trayectoria.
                continue

            class_name = "vehiculo" if self.unify_vehicle_classes else self.vehicle_classes[class_id]

            if track_id not in self.tracks:
                self.tracks[track_id] = Track(track_id)
            self.tracks[track_id].add(frame_idx, t_video, bbox_list, float(conf), class_id, class_name)

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": float(conf),
                "bbox": bbox_list,
                "track_id": track_id,
            })

        return detections

    def synchronize(self) -> None:
        """
        Espera a que terminen las operaciones pendientes en el dispositivo.

        Mismo motivo que `VehicleDetector.synchronize()`: en CUDA la inferencia es
        asíncrona, y sin esta barrera una medición de tiempo alrededor de `update()`
        puede estar midiendo solo el encolado, no la ejecución real. En CPU es un no-op.

        Retorna:
            None
        """
        if self.device == "cuda":
            import torch
            torch.cuda.synchronize()

    def reset_state(self) -> None:
        """
        Limpia todo el estado acumulado por este tracker: los `Track` propios
        (`self.tracks`), el contador de frames procesados, y el estado interno
        del tracker de Ultralytics (BYTETrack/BoT-SORT) — posiciones, filtros
        de Kalman y el contador de IDs.

        Pensado para usarse después de un calentamiento con inferencia real
        (ver `scripts/run_tracking_benchmark.py`): `model.track(...,
        persist=True)` mantiene estado entre llamadas, así que sin este reset
        los vehículos "vistos" durante el calentamiento ensuciarían el conteo
        de IDs únicos y las trayectorias de la ventana que sí se mide.

        Si la versión de Ultralytics instalada no expone el tracker interno
        donde se esperaba (`model.predictor.trackers`), se advierte y se
        continúa solo con el reset del estado propio — mejor una limpieza
        parcial que una excepción.

        Retorna:
            None
        """
        self.tracks = {}
        self.frames_processed = 0

        try:
            predictor = getattr(self.model, "predictor", None)
            trackers = getattr(predictor, "trackers", None) if predictor is not None else None
            if trackers:
                for internal_tracker in trackers:
                    internal_tracker.reset()
            else:
                logger.warning(
                    "No se encontró el tracker interno de Ultralytics (model.predictor.trackers) "
                    "para reiniciar; el estado propio sí se reinició, pero los IDs internos "
                    "pueden seguir arrastrando el calentamiento."
                )
        except Exception as exc:
            logger.warning("No se pudo reiniciar el tracker interno de Ultralytics: %s", exc)

    def finalize(self) -> Dict[str, Any]:
        """
        Cierra el procesamiento: suaviza los puntos de contacto (si está
        activado en config) y marca los tracks estacionarios.

        Retorna:
            dict: resumen (`unique_ids`, `stationary_tracks`,
            `moving_tracks`, `frames_processed`).
        """
        tracking_cfg = self.config.get("tracking", {})
        smoothing_window = tracking_cfg.get("smoothing_window", 5)
        min_displacement_px = tracking_cfg.get("min_displacement_px", 40)

        for track in self.tracks.values():
            if smoothing_window and smoothing_window > 1:
                smoothed = smooth_ground_points(track.ground_points(), smoothing_window)
                track.set_smoothed_ground_points(smoothed)
            track.stationary = track.is_stationary(min_displacement_px)

        stationary_count = sum(1 for t in self.tracks.values() if t.stationary)

        return {
            "unique_ids": len(self.tracks),
            "stationary_tracks": stationary_count,
            "moving_tracks": len(self.tracks) - stationary_count,
            "frames_processed": self.frames_processed,
        }

    def save_tracks(self, path: str) -> str:
        """
        Escribe el JSON de trayectorias: el entregable principal de esta
        fase, que será la entrada directa de la etapa de estimación de
        velocidad.

        Importante: el campo `source_fps` queda destacado en el encabezado
        a propósito. La etapa de velocidad debe leer el tiempo directamente
        de `t_video` en cada observación (ya calculado con `TimeBase`),
        **nunca recalcularlo con un reloj** — ver el docstring de `TimeBase`.

        Parámetros:
            path (str): ruta de salida, típicamente
                `results/tracks/{model_id}_{tracker}_{nombre_video}.json`.

        Retorna:
            str: la misma ruta de salida.
        """
        tracking_cfg = self.config.get("tracking", {})

        payload = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_video": self.source_video or "",
            "frame_size": list(self.frame_size) if self.frame_size else None,
            "source_fps": self.source_fps,
            "model_id": self.model_id,
            "tracker": self.tracker_name,
            "confidence_threshold": self.conf,
            "smoothing_window": tracking_cfg.get("smoothing_window", 5),
            "total_frames_processed": self.frames_processed,
            "tracks": [track.to_dict() for track in self.tracks.values()],
        }

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Trayectorias guardadas en %s (%d tracks)", path, len(self.tracks))
        return path

    def get_model_info(self) -> Dict[str, Any]:
        """
        Retorna:
            dict: `{"model_id", "weights", "parameters_M", "device", "tracker"}`.
        """
        return {
            "model_id": self.model_id,
            "weights": self.weights,
            "parameters_M": round(self.num_params / 1e6, 2),
            "device": self.device,
            "tracker": self.tracker_name,
        }

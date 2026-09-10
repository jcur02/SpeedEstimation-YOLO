"""
Detección vehicular basada en modelos YOLO (Ultralytics).
"""
import logging
import os

import cv2
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# Colores BGR (formato OpenCV) por clase de vehículo, para dibujar las cajas.
_CLASS_COLORS = {
    "car": (0, 255, 0),
    "truck": (255, 100, 0),
    "bus": (0, 165, 255),
    "motorcycle": (0, 255, 255),
}
_DEFAULT_COLOR = (200, 200, 200)


class VehicleDetector:
    """
    Encapsula un modelo YOLO de Ultralytics especializado en detección
    de vehículos.

    Carga el modelo indicado, ejecuta inferencia sobre frames de video,
    filtra únicamente las clases de vehículo relevantes (auto, moto, bus,
    camión) según el config, y provee utilidades para dibujar los
    resultados sobre la imagen.
    """

    def __init__(self, model_id, weights, config):
        """
        Carga un modelo YOLO y lo prepara para inferencia.

        Parámetros:
            model_id (str): identificador corto del modelo (ej. "yolov8n").
            weights (str): nombre del archivo de pesos (ej. "yolov8n.pt").
                Se busca/descarga dentro de la carpeta `paths.models` del
                config; si no existe localmente, Ultralytics lo descarga
                automáticamente ahí.
            config (dict): configuración completa cargada desde config.yaml.

        Retorna:
            None
        """
        self.model_id = model_id
        self.weights = weights
        self.config = config
        self.device = config.get("benchmark", {}).get("device", "cpu")

        detection_cfg = config.get("detection", {})
        self.vehicle_classes = detection_cfg.get("vehicle_classes", {})
        self.confidence_threshold = detection_cfg.get("confidence_threshold", 0.40)
        self.iou_threshold = detection_cfg.get("iou_threshold", 0.45)

        models_dir = config.get("paths", {}).get("models", "models/")
        os.makedirs(models_dir, exist_ok=True)
        weights_path = os.path.join(models_dir, weights)

        logger.info("Cargando modelo %s (%s) en dispositivo %s...", model_id, weights_path, self.device)
        self.model = YOLO(weights_path)
        self.model.to(self.device)

        try:
            self.num_params = sum(p.numel() for p in self.model.model.parameters())
        except Exception as exc:
            logger.warning("No se pudo contar parámetros del modelo %s: %s", model_id, exc)
            self.num_params = 0

        logger.info(
            "Modelo %s cargado (%.2f M parámetros) en %s",
            model_id, self.num_params / 1e6, self.device,
        )

    def detect(self, frame):
        """
        Ejecuta detección de vehículos sobre un frame de video.

        Parámetros:
            frame (numpy.ndarray): imagen BGR (formato OpenCV).

        Retorna:
            list[dict]: lista de detecciones. Cada detección tiene la forma:
                {
                    "class_id": int,
                    "class_name": str,        # "car", "truck", "bus" o "motorcycle"
                    "confidence": float,
                    "bbox": [x1, y1, x2, y2], # coordenadas absolutas en píxeles
                }
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        if not results or results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in self.vehicle_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            detections.append({
                "class_id": class_id,
                "class_name": self.vehicle_classes[class_id],
                "confidence": float(box.conf[0]),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            })

        return detections

    def draw_detections(self, frame, detections):
        """
        Dibuja las detecciones de vehículos sobre una copia del frame.

        Parámetros:
            frame (numpy.ndarray): imagen BGR original.
            detections (list[dict]): detecciones retornadas por `detect()`.

        Retorna:
            numpy.ndarray: nueva imagen con las bounding boxes, etiquetas
            de clase/confianza y un overlay informativo dibujados encima.
            El frame original no se modifica.
        """
        annotated = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = (int(round(v)) for v in det["bbox"])
            color = _CLASS_COLORS.get(det["class_name"], _DEFAULT_COLOR)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f'{det["class_name"]} {det["confidence"]:.2f}'
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_top = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(annotated, (x1, label_top), (x1 + text_w + 4, y1), color, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - baseline - 2 if y1 - baseline - 2 > 0 else text_h),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
            )

        overlay_text = f"{self.model_id} | Vehiculos: {len(detections)}"
        (text_w, text_h), baseline = cv2.getTextSize(overlay_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (0, 0), (text_w + 20, text_h + 20), (0, 0, 0), -1)
        cv2.putText(
            annotated, overlay_text, (10, text_h + 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
        )

        return annotated

    def get_model_info(self):
        """
        Retorna información descriptiva del modelo cargado.

        Retorna:
            dict: {
                "model_id": str,
                "weights": str,
                "parameters_M": float,  # parámetros en millones
                "device": str,
            }
        """
        return {
            "model_id": self.model_id,
            "weights": self.weights,
            "parameters_M": round(self.num_params / 1e6, 2),
            "device": self.device,
        }

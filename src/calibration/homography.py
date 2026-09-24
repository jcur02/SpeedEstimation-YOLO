"""
Ajuste y evaluación de la homografía píxel→metro, y el objeto de solo
lectura que consumirá la fase de estimación de velocidad.

Esta fase **no calcula velocidades**. Solo produce y verifica la
transformación de píxeles a metros — una calibración silenciosamente mala
produce velocidades que parecen razonables pero están sistemáticamente
equivocadas, por eso la mitad de este módulo trata sobre verificación
(`_check_degenerate`, `leave_one_out`, `scale_map`) y no sobre el ajuste en
sí, que es una sola llamada a `cv2.findHomography`.
"""
import json
import logging
import os
import shutil
from datetime import datetime
from itertools import combinations
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class HomographyCalibration:
    """
    Ajusta una homografía píxel→metro a partir de pares de puntos de
    correspondencia, y cuantifica qué tan confiable es.

    Válida **solo para puntos sobre el plano de la calzada**: un vehículo en
    una rampa o sobre otro plano produce, con esta misma matriz, una
    conversión a metros sin sentido.
    """

    def __init__(
        self,
        pixel_points: List[List[float]],
        world_points: List[List[float]],
        frame_size: Tuple[int, int],
    ) -> None:
        """
        Parámetros:
            pixel_points (list[list[float]]): puntos Nx2 en píxeles de la
                imagen de la cámara.
            world_points (list[list[float]]): puntos Nx2 en metros del plano
                local (ver `src.calibration.geo.LocalENU`).
            frame_size (tuple[int, int]): `(ancho, alto)` en píxeles del
                frame de referencia.

        Excepciones:
            ValueError: si hay menos de 4 puntos, o si `pixel_points` y
                `world_points` no tienen la misma cantidad de puntos.

        Retorna:
            None
        """
        self.pixel_points = np.asarray(pixel_points, dtype=np.float64)
        self.world_points = np.asarray(world_points, dtype=np.float64)

        if len(self.pixel_points) < 4:
            raise ValueError(
                f"Se requieren al menos 4 puntos de correspondencia para calcular una homografía; "
                f"se recibieron {len(self.pixel_points)}."
            )
        if len(self.pixel_points) != len(self.world_points):
            raise ValueError(
                f"pixel_points ({len(self.pixel_points)}) y world_points ({len(self.world_points)}) "
                f"deben tener la misma cantidad de puntos."
            )

        self.frame_size = (int(frame_size[0]), int(frame_size[1]))
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.method_used: Optional[str] = None
        self.ransac_threshold_m: float = 0.5
        self.warnings: List[str] = []

    def _hull_area_ratio(self) -> float:
        """Fracción del área del frame cubierta por el casco convexo de los puntos en píxeles."""
        hull = cv2.convexHull(self.pixel_points.astype(np.float32))
        hull_area = cv2.contourArea(hull)
        frame_area = float(self.frame_size[0] * self.frame_size[1])
        return float(hull_area / frame_area) if frame_area > 0 else 0.0

    def _check_degenerate(self) -> List[str]:
        """
        Detecta configuraciones de puntos que producen homografías
        inestables, sin bloquear el ajuste: solo advierte.

        - **Colinealidad**: si tres puntos cualesquiera son casi colineales
          (área del triángulo pequeña respecto al tamaño del conjunto), la
          solución es degenerada — tres puntos colineales entre cuatro
          arruinan el ajuste.
        - **Cobertura**: si el casco convexo de los puntos cubre poco del
          frame, la homografía va a **extrapolar** fuera de esa zona, y la
          extrapolación en perspectiva se degrada rápido. Es el error más
          común: marcar los cuatro puntos muy juntos.
        - **Reparto**: si todos los puntos están en la mitad cercana o en la
          lejana de la escena, conviene repartirlos en el rango de
          profundidad donde después se medirán velocidades.

        Retorna:
            list[str]: mensajes de advertencia (vacía si no se detectó nada).
        """
        warnings: List[str] = []
        pts = self.pixel_points
        n = len(pts)

        bbox = pts.max(axis=0) - pts.min(axis=0)
        # Umbral relativo al tamaño del propio conjunto de puntos (su caja
        # envolvente), no al frame completo: eso ya lo cubre `hull_ratio` por
        # separado. 1% de esa área basta para atrapar triples genuinamente
        # degenerados (colineales o casi) sin marcar como sospechosa
        # cualquier terna moderadamente alineada en una configuración normal.
        point_set_area = float(bbox[0] * bbox[1])
        min_triangle_area = 0.01 * point_set_area if point_set_area > 0 else 0.0

        for i, j, k in combinations(range(n), 3):
            area2 = abs(
                (pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1])
                - (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1])
            )
            tri_area = area2 / 2.0
            if tri_area < min_triangle_area:
                warnings.append(
                    f"Los puntos {i + 1}, {j + 1} y {k + 1} son casi colineales "
                    f"(área del triángulo: {tri_area:.1f} px²); la solución puede ser inestable."
                )

        hull_ratio = self._hull_area_ratio()
        if hull_ratio < 0.05:
            warnings.append(
                f"Los puntos cubren solo el {hull_ratio * 100:.1f}% del frame: la homografía va a "
                f"extrapolar fuera de esa zona, y la extrapolación en perspectiva se degrada rápido."
            )

        median_y = self.frame_size[1] / 2.0
        near = int(np.sum(pts[:, 1] > median_y))
        far = int(np.sum(pts[:, 1] <= median_y))
        if near == 0 or far == 0:
            warnings.append(
                "Todos los puntos están en la misma mitad (cercana o lejana) de la escena; "
                "conviene repartirlos en el rango de profundidad donde se medirán velocidades."
            )

        return warnings

    def fit(self, method: str = "least_squares", ransac_threshold_m: float = 0.5) -> None:
        """
        Ajusta la homografía a partir de los puntos de correspondencia.

        Con exactamente 4 puntos, `cv2.findHomography` da la solución exacta
        con `method=0`. Con más de 4, `method=0` ajusta por mínimos
        cuadrados sobre todos los puntos. `method="ransac"` usa
        `cv2.RANSAC`, con `ransac_threshold_m` **en metros** — el umbral se
        aplica en el espacio de destino (mundo), no en píxeles; pasarle un
        valor pensado en píxeles es un error frecuente.

        Parámetros:
            method (str): `"least_squares"` o `"ransac"`.
            ransac_threshold_m (float): umbral de RANSAC, en metros. Ignorado
                si `method != "ransac"`.

        Excepciones:
            ValueError: si `method` no es reconocido, o si
                `cv2.findHomography` no pudo resolver (configuración
                degenerada de puntos).

        Retorna:
            None
        """
        if method not in ("least_squares", "ransac"):
            raise ValueError(f"Método de ajuste desconocido: '{method}'. Usá 'least_squares' o 'ransac'.")

        self.warnings = self._check_degenerate()
        for w in self.warnings:
            logger.warning(w)
            print(f"  ⚠ {w}")

        src = self.pixel_points.astype(np.float64)
        dst = self.world_points.astype(np.float64)

        if method == "ransac":
            h_matrix, _ = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold_m)
        else:
            h_matrix, _ = cv2.findHomography(src, dst, method=0)

        if h_matrix is None:
            raise ValueError(
                "cv2.findHomography no pudo calcular una homografía: la configuración de puntos es "
                "degenerada (colineales o duplicados). Revisá los puntos marcados."
            )

        self.H = h_matrix
        self.H_inv = np.linalg.inv(h_matrix)
        self.method_used = method
        self.ransac_threshold_m = ransac_threshold_m

    def _project(self, points: Any, h_matrix: np.ndarray) -> np.ndarray:
        """Proyecta puntos Nx2 con una matriz de homografía 3x3, vectorizado."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, h_matrix)
        return out.reshape(-1, 2)

    def project_pixel_to_world(self, points_px: Any) -> np.ndarray:
        """
        Proyecta puntos en píxeles al plano métrico, con la `H` ya ajustada.

        Parámetros:
            points_px (array-like): puntos Nx2 en píxeles.

        Retorna:
            numpy.ndarray: puntos Nx2 en metros.

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.
        """
        if self.H is None:
            raise ValueError("Llamá a fit() antes de proyectar puntos.")
        return self._project(points_px, self.H)

    def residuals(self) -> np.ndarray:
        """
        Error de reproyección de cada punto, en **metros**: proyecta cada
        punto en píxeles al plano métrico con `H` y mide la distancia
        euclidiana contra la coordenada del mundo declarada.

        Es el error que importa, porque está en las unidades del resultado
        final (a diferencia del error en píxeles, que no dice nada sobre el
        error en metros sin conocer la escala local).

        Retorna:
            numpy.ndarray: vector de residuos, en metros.

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.
        """
        if self.H is None:
            raise ValueError("Llamá a fit() antes de calcular los residuos.")
        projected_world = self._project(self.pixel_points, self.H)
        return np.linalg.norm(projected_world - self.world_points, axis=1)

    def residuals_px(self) -> np.ndarray:
        """
        Error de reproyección inverso, en píxeles: proyecta cada punto del
        mundo de vuelta a la imagen con `H_inv` y mide la distancia contra
        el píxel marcado. Dato secundario frente a `residuals()`.

        Retorna:
            numpy.ndarray: vector de residuos, en píxeles.

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.
        """
        if self.H_inv is None:
            raise ValueError("Llamá a fit() antes de calcular los residuos.")
        projected_px = self._project(self.world_points, self.H_inv)
        return np.linalg.norm(projected_px - self.pixel_points, axis=1)

    def quality(self, labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Resumen de calidad de la calibración.

        Parámetros:
            labels (list[str] | None): etiquetas legibles de cada punto (ej.
                `["A", "B", "C", "D"]`), en el mismo orden que se pasaron a
                `__init__`. Si es None, se usan índices 1-based.

        Retorna:
            dict con `n_points`, `reprojection_mean_m`,
            `reprojection_median_m`, `reprojection_max_m`,
            `reprojection_rms_m`, `reprojection_mean_px`,
            `worst_point_label` (el dato más accionable: casi siempre señala
            el decimal mal copiado o la esquina mal identificada),
            `worst_point_residual_m`, `hull_area_ratio` y `warnings`.
        """
        res_m = self.residuals()
        res_px = self.residuals_px()
        worst_idx = int(np.argmax(res_m))
        worst_label = labels[worst_idx] if labels else str(worst_idx + 1)

        return {
            "n_points": len(self.pixel_points),
            "reprojection_mean_m": float(np.mean(res_m)),
            "reprojection_median_m": float(np.median(res_m)),
            "reprojection_max_m": float(np.max(res_m)),
            "reprojection_rms_m": float(np.sqrt(np.mean(res_m ** 2))),
            "reprojection_mean_px": float(np.mean(res_px)),
            "worst_point_label": worst_label,
            "worst_point_residual_m": float(res_m[worst_idx]),
            "hull_area_ratio": self._hull_area_ratio(),
            "warnings": list(self.warnings),
        }

    def leave_one_out(self) -> Dict[str, Any]:
        """
        Validación cruzada dejando un punto fuera (requiere N >= 5): para
        cada punto, ajusta con los N-1 restantes y mide el error sobre el
        punto excluido.

        Esto importa porque el error de reproyección sobre los mismos puntos
        usados en el ajuste es optimista por construcción — con exactamente
        4 puntos es cero por definición, sin importar cuán mala sea la
        calibración, porque una homografía tiene 8 grados de libertad y 4
        puntos aportan exactamente 8 ecuaciones. El error dejando uno fuera
        mide la capacidad real de generalizar: **si el error de reproyección
        es bajo pero el de validación cruzada es alto, la calibración está
        sobreajustada y no es confiable.**

        Reutiliza el mismo método y umbral de RANSAC que se usó en `fit()`.

        Retorna:
            dict: si `n_points < 5`, `{"available": False, "message": ...}`
            explicando que conviene marcar 6 u 8 puntos. Si se pudo calcular,
            `{"available": True, "mean_m": ..., "max_m": ..., "errors_m": [...]}`.

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.
        """
        if self.H is None:
            raise ValueError("Llamá a fit() antes de la validación cruzada.")

        n = len(self.pixel_points)
        if n < 5:
            return {
                "available": False,
                "message": (
                    "La validación cruzada requiere al menos 5 puntos (se ajusta con N-1 y se mide "
                    "el error sobre el punto restante). Con exactamente 4 puntos el ajuste es exacto "
                    "y el residuo sobre los mismos puntos usados es cero por definición, aunque la "
                    "calibración sea pésima. Marcá 6 u 8 puntos para poder verificar de verdad."
                ),
            }

        errors: List[float] = []
        for i in range(n):
            idx = [j for j in range(n) if j != i]
            src = self.pixel_points[idx]
            dst = self.world_points[idx]
            if self.method_used == "ransac":
                h_partial, _ = cv2.findHomography(src, dst, cv2.RANSAC, self.ransac_threshold_m)
            else:
                h_partial, _ = cv2.findHomography(src, dst, method=0)
            if h_partial is None:
                continue
            projected = self._project(self.pixel_points[i:i + 1], h_partial)
            errors.append(float(np.linalg.norm(projected[0] - self.world_points[i])))

        if not errors:
            return {
                "available": False,
                "message": "No se pudo calcular la validación cruzada: todas las homografías parciales fallaron.",
            }

        errors_arr = np.array(errors)
        return {
            "available": True,
            "mean_m": float(np.mean(errors_arr)),
            "max_m": float(np.max(errors_arr)),
            "errors_m": errors_arr.tolist(),
        }

    def scale_map(self, grid_step: int = 40) -> Dict[str, Any]:
        """
        Metros por píxel en distintas zonas de la imagen: para cada punto de
        una rejilla, proyecta ese punto y otros desplazados un píxel a la
        derecha y uno hacia abajo, y mide las distancias métricas
        resultantes.

        La escala crece con la distancia a la cámara, y el mismo ruido de
        dos píxeles en la caja delimitadora vale muchas veces más en metros
        en el fondo de la escena que en el frente — es lo que determina en
        qué franja de la imagen tiene sentido medir velocidad.

        Parámetros:
            grid_step (int): separación de la rejilla, en píxeles.

        Retorna:
            dict con `grid_step_px`, `xs`, `ys`, `grid_m_per_px` (rejilla
            `len(ys) x len(xs)`), `min_m_per_px`, `max_m_per_px` y
            `at_centroid_m_per_px` (en el centroide del casco convexo de los
            puntos de calibración).

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.
        """
        if self.H is None:
            raise ValueError("Llamá a fit() antes de calcular el mapa de escala.")

        width, height = self.frame_size
        xs = np.arange(0, width, grid_step, dtype=np.float64)
        ys = np.arange(0, height, grid_step, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        grid_pts = np.stack([xx.ravel(), yy.ravel()], axis=1)

        p0 = self._project(grid_pts, self.H)
        px1 = self._project(grid_pts + np.array([1.0, 0.0]), self.H)
        py1 = self._project(grid_pts + np.array([0.0, 1.0]), self.H)

        mpp_x = np.linalg.norm(px1 - p0, axis=1)
        mpp_y = np.linalg.norm(py1 - p0, axis=1)
        mpp = (mpp_x + mpp_y) / 2.0
        grid_arr = mpp.reshape(len(ys), len(xs))

        hull = cv2.convexHull(self.pixel_points.astype(np.float32)).reshape(-1, 2)
        centroid_px = hull.mean(axis=0).astype(np.float64)
        c0 = self._project([centroid_px], self.H)[0]
        cx1 = self._project([centroid_px + [1.0, 0.0]], self.H)[0]
        cy1 = self._project([centroid_px + [0.0, 1.0]], self.H)[0]
        mpp_centroid = (float(np.linalg.norm(cx1 - c0)) + float(np.linalg.norm(cy1 - c0))) / 2.0

        return {
            "grid_step_px": grid_step,
            "xs": xs.tolist(),
            "ys": ys.tolist(),
            "grid_m_per_px": grid_arr.tolist(),
            "min_m_per_px": float(mpp.min()),
            "max_m_per_px": float(mpp.max()),
            "at_centroid_m_per_px": mpp_centroid,
        }

    def save(self, path: str, metadata: Dict[str, Any]) -> None:
        """
        Guarda la calibración completa como JSON en `path`.

        `metadata` debe traer ya armados todos los campos del esquema salvo
        `homography`/`homography_inv`, que se calculan acá: `version`,
        `created_at`, `source_video`, `frame_size`, `reference_frame_md5`,
        `geo_model`, `origin`, `points`, `quality`, `scale` y, si aplica,
        `aerial`.

        Si ya existe un archivo en `path`, se respalda antes de sobrescribir
        como `{nombre}.backup_{YYYYMMDD_HHMMSS}.json`.

        Parámetros:
            path (str): ruta de salida (ej. "config/calibration.json").
            metadata (dict): resto de los campos del esquema, ya armados por
                quien llama (típicamente `scripts/calibrate.py`).

        Excepciones:
            ValueError: si todavía no se llamó a `fit()`.

        Retorna:
            None
        """
        if self.H is None:
            raise ValueError("Llamá a fit() antes de guardar la calibración.")

        if os.path.isfile(path):
            backup_path = f"{os.path.splitext(path)[0]}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            shutil.copy2(path, backup_path)
            print(f"  Copia de respaldo de la calibración anterior: {backup_path}")
            logger.info("Backup de calibración anterior creado en %s", backup_path)

        data = dict(metadata)
        data["homography"] = self.H.tolist()
        data["homography_inv"] = self.H_inv.tolist()

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info("Calibración guardada en %s", path)


class CalibratedPlane:
    """
    Objeto liviano de solo lectura que expone una calibración ya guardada,
    para que la fase de velocidad convierta puntos de imagen a metros sin
    volver a ajustar nada. Mantenido mínimo y estable a propósito: es el
    contrato entre esta fase y la siguiente.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """
        Parámetros:
            data (dict): contenido ya cargado de `config/calibration.json`.

        Retorna:
            None
        """
        self._data = data
        self.H = np.array(data["homography"], dtype=np.float64)
        self.H_inv = np.array(data["homography_inv"], dtype=np.float64)
        self._frame_size = tuple(data["frame_size"])
        self._quality = data.get("quality", {})
        self._reference_md5 = data.get("reference_frame_md5")

    @classmethod
    def load(cls, path: str) -> "CalibratedPlane":
        """
        Carga y valida una calibración guardada por `HomographyCalibration.save`.

        Parámetros:
            path (str): ruta al JSON de calibración.

        Retorna:
            CalibratedPlane: la calibración lista para usar.

        Excepciones:
            FileNotFoundError: si el archivo no existe.
            ValueError: si el JSON está corrupto o le faltan campos
                requeridos (`homography`, `homography_inv`, `frame_size`).
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"No se encontró la calibración: '{path}'. Generala primero con: "
                f"python scripts/calibrate.py"
            )

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"El archivo de calibración '{path}' está corrupto o no es JSON válido: {exc}") from exc

        required = ("homography", "homography_inv", "frame_size")
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"El archivo de calibración '{path}' no tiene los campos requeridos: {missing}")

        return cls(data)

    def to_world(self, points_px: Any) -> np.ndarray:
        """
        Convierte puntos en píxeles a metros, vectorizado.

        Parámetros:
            points_px (array-like): puntos Nx2 en píxeles.

        Retorna:
            numpy.ndarray: puntos Nx2 en metros.
        """
        pts = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H).reshape(-1, 2)

    def to_pixel(self, points_m: Any) -> np.ndarray:
        """
        Inversa de `to_world`: convierte puntos en metros a píxeles.

        Parámetros:
            points_m (array-like): puntos Nx2 en metros.

        Retorna:
            numpy.ndarray: puntos Nx2 en píxeles.
        """
        pts = np.asarray(points_m, dtype=np.float64).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.H_inv).reshape(-1, 2)

    def project_pixel_to_world(self, points_px: Any) -> np.ndarray:
        """Alias de `to_world`, para poder reutilizar `src.calibration.report.render_birdseye`
        (que solo espera `.H`, `.world_points` y este método) directamente sobre una calibración
        ya guardada, como hace `scripts/calibrate.py --verify`."""
        return self.to_world(points_px)

    @property
    def world_points(self) -> np.ndarray:
        """Puntos de calibración en metros, reconstruidos desde `points[].world_m` del JSON."""
        pts = [p["world_m"] for p in self._data.get("points", []) if "world_m" in p]
        return np.array(pts, dtype=np.float64) if pts else np.zeros((0, 2))

    def scale_at(self, px: float, py: float) -> float:
        """
        Metros por píxel en un punto concreto de la imagen.

        Parámetros:
            px, py (float): coordenadas de píxel.

        Retorna:
            float: metros por píxel en ese punto.
        """
        p0 = self.to_world(np.array([[px, py]]))[0]
        p1x = self.to_world(np.array([[px + 1.0, py]]))[0]
        p1y = self.to_world(np.array([[px, py + 1.0]]))[0]
        return float((np.linalg.norm(p1x - p0) + np.linalg.norm(p1y - p0)) / 2.0)

    @property
    def frame_size(self) -> Tuple[int, int]:
        """`(ancho, alto)` en píxeles del frame con el que se calibró."""
        return self._frame_size

    @property
    def quality(self) -> Dict[str, Any]:
        """Dict de calidad guardado al calibrar (ver `HomographyCalibration.quality`)."""
        return self._quality

    def check_frame(self, frame: np.ndarray) -> bool:
        """
        Compara el MD5 del frame actual contra `reference_frame_md5`, para
        detectar si la cámara se movió desde que se calibró. No falla por
        esto — solo advierte, porque puede ser una diferencia inofensiva de
        compresión.

        Parámetros:
            frame (numpy.ndarray): frame actual, en BGR.

        Retorna:
            bool: True si coincide (o si no hay hash de referencia guardado
            para comparar), False si difiere.
        """
        from src.utils.roi import md5_of_frame  # reutiliza el mismo hash robusto que usa la ROI

        if not self._reference_md5:
            return True

        current_md5 = md5_of_frame(frame)
        if current_md5 != self._reference_md5:
            msg = (
                "El frame actual no coincide con el de referencia de la calibración: "
                "la cámara podría haberse movido y la calibración ya no valer."
            )
            logger.warning(msg)
            print(f"  ⚠ {msg}")
            return False
        return True

"""
Métricas de calidad de seguimiento, calculadas SIN verdad de referencia.

**Advertencia de honestidad metodológica.** Las métricas estándar de
seguimiento multi-objeto (MOTA, HOTA, IDF1, cambios de identidad reales)
requieren anotaciones manuales de verdad de referencia (ground truth), que
este trabajo no posee. Las métricas de este módulo son **indicadores
indirectos**, calculados únicamente a partir de la geometría de los tracks
(bordes tocados, desplazamiento, duración) sin ninguna anotación externa.
Son útiles para **comparar trackers entre sí sobre el mismo video**, que es
exactamente lo que se necesita en esta fase, pero **no son MOTA ni HOTA y no
deben reportarse como tales** en el informe. Por eso se nombran de forma que
no se puedan confundir con las métricas estándar (`identity_instability` en
vez de "ID switches", por ejemplo).

Nota sobre `interior_births`/`interior_deaths`: un track puede morir
legítimamente en el interior del frame si el vehículo se detiene y se queda
quieto, o si sale por una obstrucción de la escena que no es el borde del
frame. El indicador sirve para comparar trackers entre sí sobre el mismo
video, donde esos efectos afectan a ambos por igual, no como medida absoluta
de calidad.
"""
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from src.tracking.track import Track

logger = logging.getLogger(__name__)


class TrackingQuality:
    """
    Calcula indicadores indirectos de calidad de seguimiento a partir de un
    conjunto de tracks ya finalizados (`VehicleTracker.finalize()` ya corrido).
    """

    def __init__(
        self, tracks: Dict[int, Track], frame_size: Tuple[int, int], config: Dict[str, Any],
    ) -> None:
        """
        Parámetros:
            tracks (dict[int, Track]): tracks de un `VehicleTracker` ya
                finalizado (`self.tracks` tras llamar a `finalize()`).
            frame_size (tuple[int, int]): `(ancho, alto)` del video procesado.
            config (dict): configuración completa; se usa la subsección
                `tracking` para `min_track_frames`, `min_displacement_px` y
                `border_margin_px`.

        Retorna:
            None
        """
        self.tracks = tracks
        self.frame_size = frame_size

        tracking_cfg = config.get("tracking", {})
        self.min_track_frames = tracking_cfg.get("min_track_frames", 10)
        self.min_displacement_px = tracking_cfg.get("min_displacement_px", 40)
        self.border_margin_px = tracking_cfg.get("border_margin_px", 20)

    def compute(self) -> Dict[str, Any]:
        """
        Calcula todas las métricas sobre el conjunto de tracks actual.

        Retorna:
            dict: `unique_ids`, `active_tracks_mean`, `track_length_mean_frames`,
            `track_length_mean_seconds`, `track_length_median_frames`,
            `fragment_ratio`, `interior_births`, `interior_deaths`,
            `identity_instability`, `stationary_tracks`, `moving_tracks`,
            `straightness_mean`.
        """
        tracks = list(self.tracks.values())
        unique_ids = len(tracks)

        if unique_ids == 0:
            logger.warning("No hay tracks para calcular métricas de calidad.")
            return {
                "unique_ids": 0, "active_tracks_mean": 0.0,
                "track_length_mean_frames": 0.0, "track_length_mean_seconds": 0.0,
                "track_length_median_frames": 0.0, "fragment_ratio": 0.0,
                "interior_births": 0, "interior_deaths": 0, "identity_instability": 0.0,
                "stationary_tracks": 0, "moving_tracks": 0, "straightness_mean": 0.0,
            }

        lengths_frames = [t.length_frames() for t in tracks]
        lengths_seconds = [t.length_seconds() for t in tracks]

        fragments = sum(1 for length in lengths_frames if length < self.min_track_frames)

        interior_births = 0
        interior_deaths = 0
        for t in tracks:
            first_borders, last_borders = t.borders_touched(self.frame_size, self.border_margin_px)
            if not first_borders:
                interior_births += 1
            if not last_borders:
                interior_deaths += 1

        stationary = sum(1 for t in tracks if t.is_stationary(self.min_displacement_px))

        straightness_values = []
        for t in tracks:
            path_length = t.path_length_px()
            if path_length > 0:
                straightness_values.append(t.total_displacement_px() / path_length)

        # active_tracks_mean: promedio de tracks activos por frame, calculado
        # sobre el rango de frames efectivamente cubierto por al menos un
        # track (no sobre todo el video, que `TrackingQuality` no conoce).
        if tracks:
            frame_start = min(t.first_frame for t in tracks)
            frame_end = max(t.last_frame for t in tracks)
            total_span = frame_end - frame_start + 1
            active_per_frame = np.zeros(total_span, dtype=np.int32)
            for t in tracks:
                active_per_frame[t.first_frame - frame_start: t.last_frame - frame_start + 1] += 1
            active_tracks_mean = float(active_per_frame.mean())
        else:
            active_tracks_mean = 0.0

        return {
            "unique_ids": unique_ids,
            "active_tracks_mean": active_tracks_mean,
            "track_length_mean_frames": float(np.mean(lengths_frames)),
            "track_length_mean_seconds": float(np.mean(lengths_seconds)),
            "track_length_median_frames": float(np.median(lengths_frames)),
            "fragment_ratio": fragments / unique_ids,
            "interior_births": interior_births,
            "interior_deaths": interior_deaths,
            "identity_instability": (interior_births + interior_deaths) / unique_ids,
            "stationary_tracks": stationary,
            "moving_tracks": unique_ids - stationary,
            "straightness_mean": float(np.mean(straightness_values)) if straightness_values else 0.0,
        }

    @staticmethod
    def compare(results_by_tracker: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Arma una tabla comparativa a partir de los resultados de varios
        trackers (u combinaciones modelo+tracker) sobre el mismo video.

        Parámetros:
            results_by_tracker (dict[str, dict]): mapea una etiqueta
                (ej. `"yolov8n+bytetrack"`) al dict retornado por `compute()`
                de cada corrida.

        Retorna:
            list[dict]: una fila por combinación, cada una con su etiqueta
            (`"combination"`) y todas las métricas de `compute()`, ordenada
            por `identity_instability` ascendente (mejor primero).
        """
        rows = []
        for label, metrics in results_by_tracker.items():
            row = {"combination": label}
            row.update(metrics)
            rows.append(row)
        rows.sort(key=lambda r: r.get("identity_instability", float("inf")))
        return rows

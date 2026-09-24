"""
Estimación de velocidad vehicular a partir de trayectorias ya calibradas.

Dos decisiones técnicas obligatorias gobiernan este módulo:

A. **El tiempo sale siempre de `t_video`, nunca de un reloj.** Cada observación de
   una trayectoria ya trae su instante calculado en la fase de seguimiento como
   `frame_idx / source_fps` (ver `src.tracking.track.TimeBase`). Usalo tal cual.
   Nunca lo recalcules con `time.time()`, `time.perf_counter()` ni ningún reloj del
   sistema. En la Raspberry Pi el procesamiento correrá más lento que tiempo real;
   medir con el reloj de pared subestimaría todas las velocidades por el cociente
   entre ambas tasas. Es un error silencioso: los números salen plausibles,
   simplemente están mal.

B. **La distancia acumulada NO es el estimador principal.** Sumar las distancias
   entre puntos consecutivos y dividir entre el tiempo está sesgado hacia arriba, y
   el sesgo crece con el ruido: la longitud de arco es una suma de magnitudes
   siempre positivas, así que cada perturbación aleatoria del punto de contacto
   *agrega* recorrido, nunca lo quita. Un vehículo perfectamente quieto con ruido de
   medición acumula distancia y aparenta moverse. El estimador correcto (`estimate`,
   por ventana deslizante) ajusta posición contra tiempo: el ruido de media cero se
   cancela en el ajuste en vez de acumularse. `estimate_cumulative` existe solo como
   comparación, explícitamente marcada como sesgada.
"""
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.tracking.track import smooth_ground_points

logger = logging.getLogger(__name__)

# Conversión de m/s a km/h. Se hace SOLO al exponer resultados; todo el cálculo
# interno (metros, segundos) se queda en unidades SI para no arrastrar el factor
# de conversión por los ajustes numéricos.
_MS_TO_KMH = 3.6
_KMH_TO_MS = 1.0 / _MS_TO_KMH


class TrackSpeed:
    """
    Resultado del análisis de velocidad de una única trayectoria.

    Además de los atributos pedidos explícitamente, guarda dos adicionales,
    necesarios para que capas fuera de este módulo no tengan que rehacer la
    calibración/el filtrado:

    - `mean_distance_m`: distancia media del track al origen del plano métrico
      (la cámara, o su proyección en el suelo). Lo usa
      `SpeedValidation.by_group("distance")` para desglosar la validación por
      lejanía sin volver a proyectar nada.
    - `positions_m`: las posiciones métricas (ya suavizadas y filtradas) que
      produjeron el perfil de velocidad. Las usa
      `src.speed.report.plot_trajectories_metric` para dibujar las
      trayectorias sin repetir la conversión píxel→metro.

    Ninguno de los dos participa en la estimación en sí; son subproductos que
    ya existen en memoria durante `estimate()` y se exponen para evitar
    recalcularlos en la capa de reportes.
    """

    def __init__(
        self,
        track_id: int,
        class_name: Optional[str],
        n_observations: int,
        n_used: int,
        t_start: Optional[float],
        t_end: Optional[float],
        duration_s: float,
        distance_m: float,
        speed_mean_kmh: float,
        speed_median_kmh: float,
        speed_std_kmh: float,
        speed_min_kmh: float,
        speed_max_kmh: float,
        profile: List[Tuple[float, float]],
        r_squared: float,
        quality_flags: List[str],
        is_valid: bool,
        rejection_reason: Optional[str],
        mean_distance_m: Optional[float] = None,
        positions_m: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        self.track_id = track_id
        self.class_name = class_name
        self.n_observations = n_observations
        self.n_used = n_used
        self.t_start = t_start
        self.t_end = t_end
        self.duration_s = duration_s
        self.distance_m = distance_m
        self.speed_mean_kmh = speed_mean_kmh
        self.speed_median_kmh = speed_median_kmh
        self.speed_std_kmh = speed_std_kmh
        self.speed_min_kmh = speed_min_kmh
        self.speed_max_kmh = speed_max_kmh
        self.profile = profile
        self.r_squared = r_squared
        self.quality_flags = quality_flags
        self.is_valid = is_valid
        self.rejection_reason = rejection_reason
        self.mean_distance_m = mean_distance_m
        self.positions_m = positions_m or []

    def to_dict(self) -> Dict[str, Any]:
        """Retorna: dict con todos los campos, incluido el perfil completo (t, velocidad_kmh)."""
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "n_observations": self.n_observations,
            "n_used": self.n_used,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "duration_s": self.duration_s,
            "distance_m": self.distance_m,
            "mean_distance_m": self.mean_distance_m,
            "speed_mean_kmh": self.speed_mean_kmh,
            "speed_median_kmh": self.speed_median_kmh,
            "speed_std_kmh": self.speed_std_kmh,
            "speed_min_kmh": self.speed_min_kmh,
            "speed_max_kmh": self.speed_max_kmh,
            "r_squared": self.r_squared,
            "quality_flags": list(self.quality_flags),
            "is_valid": self.is_valid,
            "rejection_reason": self.rejection_reason,
            "profile": [[float(t), float(v)] for t, v in self.profile],
        }

    def to_row_dict(self) -> Dict[str, Any]:
        """Retorna: dict aplanado (sin perfil ni posiciones) para una fila de CSV."""
        row = self.to_dict()
        row.pop("profile")
        row["quality_flags"] = ";".join(self.quality_flags)
        return row


class SpeedEstimator:
    """
    Convierte trayectorias en píxeles a velocidades en km/h, vía tres
    estimadores de complejidad y robustez crecientes/decrecientes: ventana
    deslizante (principal), extremos (comparación simple) y acumulado
    (comparación explícitamente sesgada).
    """

    def __init__(self, config: Dict[str, Any], plane: Optional[Any] = None) -> None:
        """
        Parámetros:
            config (dict): configuración completa del proyecto (sección `speed`).
            plane: instancia de `CalibratedPlane` (o cualquier objeto con la misma
                interfaz — `to_world`, `scale_at`, `frame_size`, `world_points`),
                o `None` si todavía no hay calibración. Para pruebas sintéticas se
                inyecta una `CalibratedPlane` construida en memoria a partir de una
                homografía sintética (ver `src.speed.synthetic`), sin pasar por
                ningún archivo — así el banco de pruebas ejercita exactamente el
                mismo código que los datos reales.

        Retorna:
            None
        """
        self.config = config
        self.plane = plane

        speed_cfg = config.get("speed", {})
        self.window_s = speed_cfg.get("window_s", 0.7)
        self.min_window_points = speed_cfg.get("min_window_points", 5)
        self.smoothing_window_metric = speed_cfg.get("smoothing_window_metric", 5)
        self.skip_pixel_smoothing = speed_cfg.get("skip_pixel_smoothing", True)

        filters_cfg = speed_cfg.get("filters", {})
        self.min_observations = filters_cfg.get("min_observations", 15)
        self.min_distance_m = filters_cfg.get("min_distance_m", 5.0)
        self.min_speed_kmh = filters_cfg.get("min_speed_kmh", 3.0)
        self.max_plausible_kmh = filters_cfg.get("max_plausible_kmh", 150.0)
        self.max_scale_cm_per_px = filters_cfg.get("max_scale_cm_per_px", 15.0)
        self.hull_margin_m = filters_cfg.get("hull_margin_m", 2.0)
        self.min_r_squared = filters_cfg.get("min_r_squared", 0.85)

    def _to_metric(self, ground_points_px: List[List[float]]) -> np.ndarray:
        """
        Convierte Nx2 píxeles a Nx2 metros con `plane.to_world()`, vectorizado.

        Parámetros:
            ground_points_px (list[list[float]]): puntos de contacto en píxeles.

        Retorna:
            numpy.ndarray: puntos Nx2 en metros.

        Excepciones:
            ValueError: si no hay un `plane` configurado.
        """
        if self.plane is None:
            raise ValueError(
                "No hay una calibración (CalibratedPlane) configurada en este SpeedEstimator; "
                "no se pueden convertir puntos de píxeles a metros. Pasá `plane` al construirlo, "
                "o corré primero scripts/calibrate.py."
            )
        if not ground_points_px:
            return np.zeros((0, 2), dtype=np.float64)
        pts = np.asarray(ground_points_px, dtype=np.float64)
        return self.plane.to_world(pts)

    def _smooth_metric(self, points_m: np.ndarray, window: int) -> np.ndarray:
        """
        Media móvil sobre las coordenadas ya en metros, con ventana impar,
        preservando los extremos exactos (ver `smooth_ground_points`, reutilizada
        tal cual de la fase de seguimiento).

        El suavizado va acá, en metros, y no en píxeles, por el gradiente de
        escala: una ventana uniforme en píxeles equivale a una ventana que en
        metros vale mucho más en la zona lejana de la escena que en la cercana
        (el mismo desplazamiento de un píxel representa centímetros cerca de la
        cámara y puede representar decímetros lejos). Suavizar ya en el espacio
        métrico aplica el mismo criterio físico —una distancia fija en metros—
        en toda la escena, en vez de uno que varía con la posición.

        Si la fase de seguimiento ya suavizó en píxeles (`tracking.smoothing_window`
        > 1 al generar las trayectorias), configurá `speed.skip_pixel_smoothing` y
        volvé a generar las trayectorias con `tracking.smoothing_window: 0` para no
        suavizar dos veces con dos criterios distintos (uno en píxeles, otro en
        metros) encadenados.

        Parámetros:
            points_m (numpy.ndarray): puntos Nx2 en metros, en orden temporal.
            window (int): tamaño de ventana (se fuerza impar internamente).

        Retorna:
            numpy.ndarray: puntos suavizados, misma forma que la entrada.
        """
        if len(points_m) == 0:
            return points_m
        smoothed = smooth_ground_points(points_m.tolist(), window)
        return np.asarray(smoothed, dtype=np.float64)

    def _filter_observations(
        self, observations: List[Dict[str, Any]], points_m: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, List[str]]:
        """
        Descarta observaciones no aptas para estimar velocidad, antes de
        ajustar nada. Cada categoría de descarte deja una marca agregada (no
        una por observación, para no inflar `quality_flags`) en la lista
        retornada.

        Categorías, en el orden en que se aplican:
            - **Fuera de la zona calibrada**: el punto cae fuera del casco
              convexo de los puntos de calibración, más `hull_margin_m` de
              margen. Ahí la homografía extrapola y no es confiable.
            - **Escala excesiva**: `plane.scale_at()` por encima de
              `max_scale_cm_per_px`. Ahí el ruido de un píxel vale demasiados
              centímetros como para confiar en la posición.
            - **Salto imposible**: desplazamiento entre observaciones
              consecutivas *ya filtradas* que implique una velocidad por
              encima de `max_plausible_kmh`. Casi siempre es un cambio de
              identidad (el ID saltó de un vehículo a otro). El punto de
              llegada del salto se descarta y no se usa como ancla para el
              siguiente salto, para no arrastrar el rechazo en cascada sobre
              puntos buenos que vengan después.

        Parámetros:
            observations (list[dict]): observaciones originales del track.
            points_m (numpy.ndarray): sus puntos de contacto ya convertidos a
                metros (y suavizados si aplica), mismo orden y longitud.

        Retorna:
            tuple[list[dict], numpy.ndarray, list[str]]: observaciones y
            puntos que sobrevivieron el filtrado, más las marcas de calidad
            agregadas.
        """
        n = len(observations)
        keep_mask = np.ones(n, dtype=bool)
        flags: List[str] = []

        hull = None
        if self.plane is not None:
            world_points = getattr(self.plane, "world_points", None)
            if world_points is not None and len(world_points) >= 3:
                hull = cv2.convexHull(np.asarray(world_points, dtype=np.float32))

        n_outside_zone = 0
        n_excess_scale = 0
        for i in range(n):
            if hull is not None:
                pt = (float(points_m[i][0]), float(points_m[i][1]))
                dist = cv2.pointPolygonTest(hull, pt, True)
                if dist < -self.hull_margin_m:
                    keep_mask[i] = False
                    n_outside_zone += 1
                    continue
            if self.plane is not None:
                gp = observations[i]["ground_point"]
                scale_m_per_px = self.plane.scale_at(float(gp[0]), float(gp[1]))
                if scale_m_per_px * 100.0 > self.max_scale_cm_per_px:
                    keep_mask[i] = False
                    n_excess_scale += 1

        if n_outside_zone:
            flags.append(f"fuera_zona_calibrada:{n_outside_zone}")
        if n_excess_scale:
            flags.append(f"escala_excesiva:{n_excess_scale}")

        kept_indices = [i for i in range(n) if keep_mask[i]]
        n_jumps = 0
        last_good: Optional[int] = None
        for idx in kept_indices:
            if last_good is None:
                last_good = idx
                continue
            dt = observations[idx]["t_video"] - observations[last_good]["t_video"]
            if dt <= 0:
                continue
            dist_m = float(np.linalg.norm(points_m[idx] - points_m[last_good]))
            implied_kmh = (dist_m / dt) * _MS_TO_KMH
            if implied_kmh > self.max_plausible_kmh:
                keep_mask[idx] = False
                n_jumps += 1
            else:
                last_good = idx

        if n_jumps:
            flags.append(f"salto_imposible:{n_jumps}")
            flags.append("id_switch_suspected")

        kept_obs = [observations[i] for i in range(n) if keep_mask[i]]
        kept_pts = points_m[keep_mask]
        return kept_obs, kept_pts, flags

    def _prepare(
        self, track_dict: Dict[str, Any]
    ) -> Tuple[bool, Any]:
        """
        Pipeline común a los tres estimadores: convierte a métrico, suaviza,
        filtra, y valida que quede lo mínimo indispensable para ajustar algo.

        Retorna:
            tuple[bool, Any]:
              - `(True, (t, x, y, filtered_obs, n_observations, quality_flags))`
                si hay suficientes datos para estimar.
              - `(False, (n_observations, n_used, quality_flags, razon))` si el
                track debe rechazarse.
        """
        observations = track_dict["observations"]
        n_observations = len(observations)
        ground_points_px = [obs["ground_point"] for obs in observations]
        points_m_raw = self._to_metric(ground_points_px)

        if self.smoothing_window_metric and self.smoothing_window_metric > 1:
            points_m = self._smooth_metric(points_m_raw, self.smoothing_window_metric)
        else:
            points_m = points_m_raw

        filtered_obs, filtered_pts, quality_flags = self._filter_observations(observations, points_m)
        n_used = len(filtered_obs)

        if n_used < self.min_observations:
            return False, (
                n_observations, n_used, quality_flags,
                f"Solo {n_used} observaciones útiles tras filtrar (mínimo {self.min_observations}).",
            )

        t = np.array([obs["t_video"] for obs in filtered_obs], dtype=np.float64)
        x, y = filtered_pts[:, 0], filtered_pts[:, 1]

        distance_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        if distance_m < self.min_distance_m:
            return False, (
                n_observations, n_used, quality_flags,
                f"Desplazamiento total {distance_m:.1f} m menor al mínimo ({self.min_distance_m} m).",
            )

        duration_s = float(t[-1] - t[0])
        if duration_s <= 0:
            return False, (n_observations, n_used, quality_flags, "Duración del track es cero o negativa tras filtrar.")

        return True, (t, x, y, filtered_obs, n_observations, quality_flags)

    @staticmethod
    def _global_r_squared(t: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
        """
        R² del ajuste lineal conjunto de `x(t)` e `y(t)` sobre todo el track:
        combina los residuos de ambos ejes en una sola métrica de qué tan bien
        se explica la trayectoria completa con un movimiento a velocidad
        constante. Un valor bajo generalizado sugiere calibración mala o
        vehículos que de verdad aceleran/frenan/giran.

        Retorna:
            float: R² en [aprox. -inf, 1]; 1.0 si no hay varianza que explicar.
        """
        if len(t) < 2:
            return 0.0
        px = np.polyfit(t, x, 1)
        py = np.polyfit(t, y, 1)
        x_pred, y_pred = np.polyval(px, t), np.polyval(py, t)
        ss_res = float(np.sum((x - x_pred) ** 2) + np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((x - np.mean(x)) ** 2) + np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return 1.0
        return 1.0 - ss_res / ss_tot

    def _rejected(self, track_id: int, class_name: Optional[str], n_observations: int, n_used: int,
                  quality_flags: List[str], reason: str) -> TrackSpeed:
        """Construye el `TrackSpeed` de rechazo, común a los tres estimadores."""
        return TrackSpeed(
            track_id=track_id, class_name=class_name, n_observations=n_observations, n_used=n_used,
            t_start=None, t_end=None, duration_s=0.0, distance_m=0.0,
            speed_mean_kmh=0.0, speed_median_kmh=0.0, speed_std_kmh=0.0, speed_min_kmh=0.0, speed_max_kmh=0.0,
            profile=[], r_squared=0.0, quality_flags=quality_flags, is_valid=False, rejection_reason=reason,
        )

    def estimate(self, track_dict: Dict[str, Any], window_s: Optional[float] = None) -> TrackSpeed:
        """
        Estimador principal, por ventana deslizante: para cada observación,
        ajusta por mínimos cuadrados `x(t)` e `y(t)` (`np.polyfit` grado 1)
        sobre una ventana temporal centrada en ella; la velocidad es la
        magnitud del vector de pendientes. El ruido con media cero se cancela
        en el ajuste en vez de acumularse (ver decisión B en el docstring del
        módulo).

        La ventana debe cubrir al menos `min_window_points` puntos; si los FPS
        son bajos y no alcanza, se amplía automáticamente (se registra
        `"ventana_ampliada"` en `quality_flags`).

        Parámetros:
            track_dict (dict): un track con la misma forma que produce
                `Track.to_dict()` (ver `src.tracking.track`).
            window_s (float | None): ancho de la ventana, en segundos. Si es
                None, usa `speed.window_s` del config.

        Retorna:
            TrackSpeed
        """
        window_s = window_s if window_s is not None else self.window_s
        track_id = track_dict["track_id"]
        class_name = track_dict.get("class_name")

        ok, payload = self._prepare(track_dict)
        if not ok:
            n_observations, n_used, quality_flags, reason = payload
            return self._rejected(track_id, class_name, n_observations, n_used, quality_flags, reason)

        t, x, y, filtered_obs, n_observations, quality_flags = payload
        n_used = len(filtered_obs)
        quality_flags = list(quality_flags)

        profile: List[Tuple[float, float]] = []
        widened = False
        for i in range(n_used):
            ti = t[i]
            half_span = window_s / 2.0
            idx_window = np.where((t >= ti - half_span) & (t <= ti + half_span))[0]

            full_span = t[-1] - t[0]
            while len(idx_window) < self.min_window_points and half_span < full_span:
                half_span *= 1.5
                idx_window = np.where((t >= ti - half_span) & (t <= ti + half_span))[0]
                widened = True

            if len(idx_window) < 2 or len(set(t[idx_window].tolist())) < 2:
                continue

            tw, xw, yw = t[idx_window], x[idx_window], y[idx_window]
            slope_x = float(np.polyfit(tw, xw, 1)[0])
            slope_y = float(np.polyfit(tw, yw, 1)[0])
            speed_kmh = float(np.hypot(slope_x, slope_y)) * _MS_TO_KMH
            profile.append((float(ti), speed_kmh))

        if widened:
            quality_flags.append("ventana_ampliada")

        if not profile:
            return self._rejected(
                track_id, class_name, n_observations, n_used, quality_flags,
                "No se pudo ajustar ninguna ventana temporal (muy pocos puntos distintos en el tiempo).",
            )

        speeds = np.array([s for _, s in profile], dtype=np.float64)
        r_squared = self._global_r_squared(t, x, y)
        if r_squared < self.min_r_squared:
            quality_flags.append(f"r_squared_bajo:{r_squared:.2f}")

        distance_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        mean_distance_m = float(np.mean(np.hypot(x, y)))

        return TrackSpeed(
            track_id=track_id, class_name=class_name, n_observations=n_observations, n_used=n_used,
            t_start=float(t[0]), t_end=float(t[-1]), duration_s=float(t[-1] - t[0]), distance_m=distance_m,
            speed_mean_kmh=float(np.mean(speeds)), speed_median_kmh=float(np.median(speeds)),
            speed_std_kmh=float(np.std(speeds)), speed_min_kmh=float(np.min(speeds)), speed_max_kmh=float(np.max(speeds)),
            profile=profile, r_squared=r_squared, quality_flags=quality_flags, is_valid=True, rejection_reason=None,
            mean_distance_m=mean_distance_m, positions_m=list(zip(x.tolist(), y.tolist())),
        )

    def estimate_instantaneous(
        self, observations: List[Dict[str, Any]], window_s: Optional[float] = None,
    ) -> Optional[float]:
        """
        Velocidad actual de un track **todavía en curso**, para overlays en
        vivo (`DetectionVisualizer`) — no para el análisis final de un track
        ya terminado (usar `estimate()` para eso, que además filtra y agrega
        estadísticos completos).

        Ajusta una recta por mínimos cuadrados a `x(t)` e `y(t)` sobre una
        ventana hacia **atrás** que termina en la última observación
        disponible: en vivo no hay observaciones futuras, así que a
        diferencia de `estimate()` (ventana centrada) esta ventana es
        necesariamente causal. Por eso mismo el número mostrado en pantalla
        para un track recién aparecido puede tardar una fracción de segundo
        en estabilizarse — es información parcial, no un error.

        No aplica el filtrado de `_filter_observations` (zona calibrada,
        escala, saltos): es una lectura rápida y aproximada para inspección
        visual, no la cifra que se reporta en el informe.

        Parámetros:
            observations (list[dict]): observaciones del track hasta ahora,
                en orden temporal (`track.observations` de
                `src.tracking.track.Track`, que crece frame a frame).
            window_s (float | None): ancho de la ventana, en segundos. Si es
                None, usa `speed.window_s` del config.

        Retorna:
            float | None: velocidad en km/h, o None si todavía no hay
            suficientes observaciones (o suficiente variación temporal) para
            ajustar nada, o si el resultado supera `max_plausible_kmh`
            (probable ruido extremo o track recién creado).
        """
        if len(observations) < self.min_window_points:
            return None

        window_s = window_s if window_s is not None else self.window_s

        # Acota el trabajo a lo último de la trayectoria: un track viejo puede
        # tener miles de observaciones acumuladas, pero solo hace falta un
        # puñado de segundos recientes para esta lectura instantánea. Sin este
        # recorte, recalcular sobre el historial completo en cada frame haría
        # que el costo crezca con la edad del track.
        recent = observations[-90:]

        ground_points_px = [obs["ground_point"] for obs in recent]
        points_m = self._to_metric(ground_points_px)
        if self.smoothing_window_metric and self.smoothing_window_metric > 1:
            points_m = self._smooth_metric(points_m, self.smoothing_window_metric)

        t = np.array([obs["t_video"] for obs in recent], dtype=np.float64)
        t_now = t[-1]
        idx_window = np.where(t >= t_now - window_s)[0]

        if len(idx_window) < self.min_window_points:
            idx_window = np.arange(max(0, len(t) - self.min_window_points), len(t))
        if len(idx_window) < 2 or len(set(t[idx_window].tolist())) < 2:
            return None

        tw, xw, yw = t[idx_window], points_m[idx_window, 0], points_m[idx_window, 1]
        slope_x = float(np.polyfit(tw, xw, 1)[0])
        slope_y = float(np.polyfit(tw, yw, 1)[0])
        speed_kmh = float(np.hypot(slope_x, slope_y)) * _MS_TO_KMH

        if speed_kmh > self.max_plausible_kmh:
            return None
        return speed_kmh

    def estimate_endpoint(self, track_dict: Dict[str, Any]) -> TrackSpeed:
        """
        Desplazamiento entre el primer y el último punto de contacto,
        dividido entre el tiempo transcurrido. Insensible al ruido intermedio
        (no ve nada de lo que pasa en el medio del track), pero asume
        velocidad constante durante todo el recorrido. Útil como comparación
        simple contra la ventana deslizante.

        Parámetros:
            track_dict (dict): ver `estimate()`.

        Retorna:
            TrackSpeed
        """
        track_id = track_dict["track_id"]
        class_name = track_dict.get("class_name")

        ok, payload = self._prepare(track_dict)
        if not ok:
            n_observations, n_used, quality_flags, reason = payload
            return self._rejected(track_id, class_name, n_observations, n_used, quality_flags, reason)

        t, x, y, filtered_obs, n_observations, quality_flags = payload
        n_used = len(filtered_obs)

        duration_s = float(t[-1] - t[0])
        distance_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        speed_kmh = (distance_m / duration_s) * _MS_TO_KMH
        r_squared = self._global_r_squared(t, x, y)
        mean_distance_m = float(np.mean(np.hypot(x, y)))

        return TrackSpeed(
            track_id=track_id, class_name=class_name, n_observations=n_observations, n_used=n_used,
            t_start=float(t[0]), t_end=float(t[-1]), duration_s=duration_s, distance_m=distance_m,
            speed_mean_kmh=speed_kmh, speed_median_kmh=speed_kmh, speed_std_kmh=0.0,
            speed_min_kmh=speed_kmh, speed_max_kmh=speed_kmh,
            profile=[(float(t[0]), speed_kmh), (float(t[-1]), speed_kmh)],
            r_squared=r_squared, quality_flags=quality_flags, is_valid=True, rejection_reason=None,
            mean_distance_m=mean_distance_m, positions_m=list(zip(x.tolist(), y.tolist())),
        )

    def estimate_cumulative(self, track_dict: Dict[str, Any]) -> TrackSpeed:
        """
        Estimador de **comparación, explícitamente sesgado hacia arriba**:
        suma las distancias entre puntos consecutivos (longitud de arco) y
        divide entre el tiempo total transcurrido.

        La longitud de arco es una suma de magnitudes siempre positivas: cada
        perturbación aleatoria del punto de contacto *agrega* recorrido, nunca
        lo quita. Un vehículo perfectamente quieto con ruido de medición
        acumula distancia y aparenta moverse, y el sesgo crece con el ruido.
        Se incluye solo para cuantificar esa diferencia frente a la ventana
        deslizante — **nunca lo uses como estimador principal.**

        Parámetros:
            track_dict (dict): ver `estimate()`.

        Retorna:
            TrackSpeed
        """
        track_id = track_dict["track_id"]
        class_name = track_dict.get("class_name")

        ok, payload = self._prepare(track_dict)
        if not ok:
            n_observations, n_used, quality_flags, reason = payload
            return self._rejected(track_id, class_name, n_observations, n_used, quality_flags, reason)

        t, x, y, filtered_obs, n_observations, quality_flags = payload
        n_used = len(filtered_obs)
        quality_flags = list(quality_flags) + ["estimador_sesgado_hacia_arriba"]

        path_length_m = float(np.sum(np.hypot(np.diff(x), np.diff(y))))
        duration_s = float(t[-1] - t[0])
        distance_m = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
        speed_kmh = (path_length_m / duration_s) * _MS_TO_KMH
        r_squared = self._global_r_squared(t, x, y)
        mean_distance_m = float(np.mean(np.hypot(x, y)))

        return TrackSpeed(
            track_id=track_id, class_name=class_name, n_observations=n_observations, n_used=n_used,
            t_start=float(t[0]), t_end=float(t[-1]), duration_s=duration_s, distance_m=distance_m,
            speed_mean_kmh=speed_kmh, speed_median_kmh=speed_kmh, speed_std_kmh=0.0,
            speed_min_kmh=speed_kmh, speed_max_kmh=speed_kmh,
            profile=[(float(t[0]), speed_kmh), (float(t[-1]), speed_kmh)],
            r_squared=r_squared, quality_flags=quality_flags, is_valid=True, rejection_reason=None,
            mean_distance_m=mean_distance_m, positions_m=list(zip(x.tolist(), y.tolist())),
        )

    def estimate_all(
        self, tracks_json_path: str, estimator: str = "window", reference_frame: Optional[np.ndarray] = None,
    ) -> List[TrackSpeed]:
        """
        Procesa un archivo completo de trayectorias. Ignora (mostrándolos
        aparte) los tracks marcados como `is_stationary` en la fase de
        seguimiento: son vehículos parqueados, no en movimiento.

        Valida que el `frame_size` de las trayectorias coincida con el de la
        calibración: si difieren, la homografía fue ajustada para otra
        resolución, y aplicarla igual produciría un error de escala
        sistemático — exactamente el tipo de fallo silencioso que hay que
        evitar, así que esto aborta con un error claro en vez de continuar.

        Parámetros:
            tracks_json_path (str): ruta al JSON de trayectorias
                (`results/tracks/{modelo}_{tracker}_{video}.json`).
            estimator (str): `"window"`, `"endpoint"` o `"cumulative"`.
            reference_frame (numpy.ndarray | None): un frame de la cámara, si
                está disponible, para verificar con `plane.check_frame()` que
                no se movió desde la calibración (solo advierte, no aborta).

        Retorna:
            list[TrackSpeed]: uno por track en movimiento.

        Excepciones:
            FileNotFoundError: si `tracks_json_path` no existe.
            ValueError: si el `frame_size` no coincide con el de la calibración,
                o si `estimator` no es reconocido.
        """
        if not os.path.isfile(tracks_json_path):
            raise FileNotFoundError(f"No se encontró el archivo de trayectorias: '{tracks_json_path}'.")

        with open(tracks_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.plane is not None:
            tracks_frame_size = tuple(data.get("frame_size") or ())
            plane_frame_size = tuple(self.plane.frame_size)
            if tracks_frame_size and tracks_frame_size != plane_frame_size:
                raise ValueError(
                    f"El frame_size de las trayectorias ({tracks_frame_size[0]}x{tracks_frame_size[1]}) no "
                    f"coincide con el de la calibración ({plane_frame_size[0]}x{plane_frame_size[1]}). La "
                    f"homografía fue ajustada para otra resolución; usarla igual produciría un error de "
                    f"escala sistemático. Recalibrá con scripts/calibrate.py a la resolución correcta, o "
                    f"reprocesá el seguimiento a la resolución con la que se calibró."
                )
            if reference_frame is not None:
                self.plane.check_frame(reference_frame)

        tracking_smoothing_window = data.get("smoothing_window")
        if (
            self.skip_pixel_smoothing and tracking_smoothing_window and tracking_smoothing_window > 1
            and self.smoothing_window_metric and self.smoothing_window_metric > 1
        ):
            logger.info(
                "Las trayectorias ya se suavizaron en píxeles al generar el seguimiento (ventana=%d); este "
                "módulo aplica además su propio suavizado en metros (ventana=%d). Si preferís suavizar una "
                "sola vez, regenerá las trayectorias con tracking.smoothing_window: 0.",
                tracking_smoothing_window, self.smoothing_window_metric,
            )

        estimate_fn = {
            "window": self.estimate, "endpoint": self.estimate_endpoint, "cumulative": self.estimate_cumulative,
        }.get(estimator)
        if estimate_fn is None:
            raise ValueError(f"Estimador desconocido: '{estimator}'. Válidos: window, endpoint, cumulative.")

        moving_tracks = [t for t in data.get("tracks", []) if not t.get("is_stationary")]
        return [estimate_fn(track) for track in moving_tracks]

    def to_csv(self, results: List[TrackSpeed], path: str) -> None:
        """
        Escribe un CSV con una fila por track (sin el perfil completo), para
        análisis rápido en una hoja de cálculo o pandas.

        Retorna:
            None
        """
        import csv as csv_module

        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fieldnames = [
            "track_id", "class_name", "n_observations", "n_used", "t_start", "t_end", "duration_s",
            "distance_m", "mean_distance_m", "speed_mean_kmh", "speed_median_kmh", "speed_std_kmh",
            "speed_min_kmh", "speed_max_kmh", "r_squared", "is_valid", "rejection_reason", "quality_flags",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r.to_row_dict())

        logger.info("CSV de velocidades guardado en %s (%d tracks)", path, len(results))

    def to_json(self, results: List[TrackSpeed], path: str) -> None:
        """
        Escribe un JSON con los perfiles completos de velocidad de cada track.

        Retorna:
            None
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        payload = {
            "version": 1,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "n_tracks": len(results),
            "tracks": [r.to_dict() for r in results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("JSON de velocidades guardado en %s (%d tracks)", path, len(results))

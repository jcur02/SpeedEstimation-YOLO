"""
Compara estimaciones de velocidad contra mediciones de referencia obtenidas
manualmente (cinta métrica + cronómetro, GPS de un vehículo de prueba, o un
radar), para poder afirmar con respaldo numérico qué tan bien funciona el
sistema completo, no solo el estimador en aislamiento.

Guía de interpretación (sesgo contra dispersión) — es la lectura central de
este módulo, y va también en el README:

    El **sesgo** y la **dispersión** señalan causas distintas y se corrigen
    distinto.

    Sesgo grande con dispersión pequeña —todos los valores desviados en
    proporción similar— indica un **error de escala en la calibración**. Es
    sistemático y se corrige revisando las coordenadas del mundo, no
    tocando el estimador ni el seguimiento.

    Sesgo cercano a cero con dispersión grande indica **ruido de detección y
    seguimiento**. Se ataca con ventana de ajuste más larga, más suavizado, o
    restringiendo la zona de medición a donde la escala es favorable.

    Ambos grandes: revisar primero la calibración, porque el sesgo puede
    estar enmascarando el diagnóstico de la dispersión.
"""
import csv
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.speed.estimator import TrackSpeed

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = ["video", "track_id", "speed_ref_kmh", "method"]


def _ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


class SpeedValidation:
    """
    Empareja estimaciones (`TrackSpeed`) contra un CSV de referencia por
    `(video, track_id)`, y calcula métricas de acuerdo y desglose por grupo.

    Formato del CSV de referencia (`config/speed_reference.csv`):

    ```csv
    video,track_id,speed_ref_kmh,method,baseline_m,frame_in,frame_out,notes
    sitio_a_01.mp4,7,42.3,manual_baseline,22.4,318,375,"sedán gris"
    ```

    `method` permite distinguir procedencias (`manual_baseline`, `gps`,
    `radar`) y analizarlas por separado si se desglosa por esa columna.
    `baseline_m`, `frame_in`, `frame_out` y `notes` son opcionales.
    """

    def __init__(self) -> None:
        self.pairs: List[Tuple[Dict[str, Any], TrackSpeed]] = []

    @staticmethod
    def load_reference(path: str) -> List[Dict[str, Any]]:
        """
        Lee y valida el CSV de referencia, informando el número de línea de
        cualquier problema.

        Parámetros:
            path (str): ruta al CSV.

        Retorna:
            list[dict]: una entrada por fila válida.

        Excepciones:
            FileNotFoundError: si el archivo no existe.
            ValueError: si faltan columnas requeridas, o si alguna fila tiene
                `track_id`/`speed_ref_kmh` no numéricos.
        """
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No se encontró el CSV de referencia: '{path}'.")

        rows: List[Dict[str, Any]] = []
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or not all(c in reader.fieldnames for c in _REQUIRED_COLUMNS):
                raise ValueError(
                    f"El CSV de referencia '{path}' debe tener al menos las columnas: "
                    f"{', '.join(_REQUIRED_COLUMNS)}. Encontradas: {reader.fieldnames}"
                )
            for line_num, row in enumerate(reader, start=2):
                video = (row.get("video") or "").strip()
                if not video:
                    continue
                try:
                    track_id = int(row["track_id"])
                    speed_ref_kmh = float(row["speed_ref_kmh"])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(
                        f"Línea {line_num} del CSV de referencia '{path}': track_id o speed_ref_kmh "
                        f"inválidos ({exc})."
                    ) from exc

                def _opt_float(key: str) -> Optional[float]:
                    v = (row.get(key) or "").strip()
                    return float(v) if v else None

                def _opt_int(key: str) -> Optional[int]:
                    v = (row.get(key) or "").strip()
                    return int(v) if v else None

                rows.append({
                    "video": video,
                    "track_id": track_id,
                    "speed_ref_kmh": speed_ref_kmh,
                    "method": (row.get("method") or "").strip() or "unknown",
                    "baseline_m": _opt_float("baseline_m"),
                    "frame_in": _opt_int("frame_in"),
                    "frame_out": _opt_int("frame_out"),
                    "notes": (row.get("notes") or "").strip(),
                })

        if not rows:
            raise ValueError(f"El CSV de referencia '{path}' no tiene ninguna fila válida.")

        return rows

    def join(
        self, estimates: List[TrackSpeed], reference: List[Dict[str, Any]], video_name: str,
    ) -> List[Tuple[Dict[str, Any], TrackSpeed]]:
        """
        Empareja estimaciones y referencias por `(video, track_id)`, filtrando
        la referencia al video actual por nombre de archivo.

        Un emparejamiento bajo es en sí un hallazgo: significa que el
        seguimiento perdió vehículos que el ojo humano sí identificó al armar
        la referencia manual (o, más raro, que la referencia apunta a
        `track_id`s que no existen en esta corrida).

        Parámetros:
            estimates (list[TrackSpeed]): resultados de `SpeedEstimator.estimate_all`.
            reference (list[dict]): filas de `load_reference`.
            video_name (str): video actual (ruta o nombre de archivo), para
                filtrar las filas de referencia que le corresponden.

        Retorna:
            list[tuple[dict, TrackSpeed]]: pares emparejados. También queda
            guardado en `self.pairs`.
        """
        ref_for_video = [r for r in reference if os.path.basename(r["video"]) == os.path.basename(video_name)]
        estimates_by_id = {e.track_id: e for e in estimates}

        pairs: List[Tuple[Dict[str, Any], TrackSpeed]] = []
        unmatched_reference: List[Dict[str, Any]] = []
        for r in ref_for_video:
            est = estimates_by_id.get(r["track_id"])
            if est is not None and est.is_valid:
                pairs.append((r, est))
            else:
                unmatched_reference.append(r)

        matched_ids = {r["track_id"] for r, _ in pairs}
        unmatched_estimates = [e for e in estimates if e.is_valid and e.track_id not in matched_ids]

        print(f"Emparejamiento: {len(pairs)} par(es) formados de {len(ref_for_video)} referencia(s) para este video.")
        if unmatched_reference:
            ids = [r["track_id"] for r in unmatched_reference]
            print(f"  Referencias sin estimación correspondiente: {len(unmatched_reference)} (track_ids: {ids})")
            print(
                "  Un emparejamiento bajo es en sí un hallazgo: puede significar que el seguimiento perdió "
                "vehículos que el ojo humano sí identificó al armar la referencia manual, o que la referencia "
                "apunta a track_ids equivocados."
            )
        if unmatched_estimates:
            print(f"  Estimaciones válidas sin referencia: {len(unmatched_estimates)}")

        logger.info(
            "Emparejamiento de validación: %d pares, %d referencias sin match, %d estimaciones sin match",
            len(pairs), len(unmatched_reference), len(unmatched_estimates),
        )

        self.pairs = pairs
        return pairs

    def metrics(self, pairs: Optional[List[Tuple[Dict[str, Any], TrackSpeed]]] = None) -> Dict[str, Any]:
        """
        Métricas de acuerdo entre estimado y referencia.

        Parámetros:
            pairs (list | None): pares a usar; si es None, usa `self.pairs`
                (los del último `join()`).

        Retorna:
            dict: `n`, `mae_kmh`, `mape_pct`, `rmse_kmh`, `bias_kmh` (error
            medio CON signo: estimado - referencia), `std_error_kmh`,
            `pearson_r` y `limits_of_agreement` (`[sesgo - 1.96σ, sesgo + 1.96σ]`).

        Excepciones:
            ValueError: si no hay pares para calcular métricas.
        """
        pairs = pairs if pairs is not None else self.pairs
        if not pairs:
            raise ValueError("No hay pares emparejados; llamá a join() primero con al menos un par válido.")

        ref = np.array([r["speed_ref_kmh"] for r, _ in pairs], dtype=np.float64)
        est = np.array([e.speed_mean_kmh for _, e in pairs], dtype=np.float64)
        errors = est - ref
        n = len(ref)

        bias = float(np.mean(errors))
        std_error = float(np.std(errors))
        pearson_r = float(np.corrcoef(ref, est)[0, 1]) if (n >= 2 and np.std(ref) > 0 and np.std(est) > 0) else float("nan")

        return {
            "n": n,
            "mae_kmh": float(np.mean(np.abs(errors))),
            "mape_pct": float(np.mean(np.abs(errors / ref)) * 100.0),
            "rmse_kmh": float(np.sqrt(np.mean(errors ** 2))),
            "bias_kmh": bias,
            "std_error_kmh": std_error,
            "pearson_r": pearson_r,
            "limits_of_agreement": [bias - 1.96 * std_error, bias + 1.96 * std_error],
        }

    def by_group(self, key: str) -> Dict[str, Dict[str, Any]]:
        """
        Desglose de métricas por grupo, sobre los pares del último `join()`.

        Parámetros:
            key (str): `"class_name"`, `"speed_range"` o `"distance"`. El
                desglose por distancia debería mostrar degradación con la
                lejanía; si no aparece, conviene sospechar de los datos (o de
                que el rango de distancias cubierto es demasiado angosto).

        Retorna:
            dict: `{nombre_de_grupo: métricas}` (ver `metrics()`).

        Excepciones:
            ValueError: si no hay pares, o si `key` no es reconocida.
        """
        if not self.pairs:
            raise ValueError("No hay pares emparejados; llamá a join() primero.")

        if key == "class_name":
            def keyfunc(_ref: Dict[str, Any], est: TrackSpeed) -> str:
                return est.class_name or "desconocida"
        elif key == "speed_range":
            def keyfunc(ref: Dict[str, Any], _est: TrackSpeed) -> str:
                v = ref["speed_ref_kmh"]
                if v < 30:
                    return "< 30 km/h"
                if v < 50:
                    return "30-50 km/h"
                if v < 70:
                    return "50-70 km/h"
                return ">= 70 km/h"
        elif key == "distance":
            def keyfunc(_ref: Dict[str, Any], est: TrackSpeed) -> str:
                if est.mean_distance_m is None:
                    return "desconocida"
                if est.mean_distance_m < 20:
                    return "cercana (<20m)"
                if est.mean_distance_m < 50:
                    return "media (20-50m)"
                return "lejana (>=50m)"
        else:
            raise ValueError(f"Clave de agrupación desconocida: '{key}'. Válidas: class_name, speed_range, distance.")

        groups: Dict[str, List[Tuple[Dict[str, Any], TrackSpeed]]] = {}
        for ref, est in self.pairs:
            groups.setdefault(keyfunc(ref, est), []).append((ref, est))

        return {name: self.metrics(pairs=group_pairs) for name, group_pairs in groups.items()}


def plot_scatter_validation(pairs: List[Tuple[Dict[str, Any], TrackSpeed]], out_path: str) -> None:
    """
    Estimado contra referencia, con la recta identidad y el ajuste lineal.
    La pendiente del ajuste es un estimador directo del error de escala: 1.0
    es perfecto, y una pendiente sistemáticamente distinta de 1 apunta a la
    calibración, no al estimador.

    Retorna:
        None
    """
    ref = np.array([r["speed_ref_kmh"] for r, _ in pairs], dtype=np.float64)
    est = np.array([e.speed_mean_kmh for _, e in pairs], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(6, 6))
    upper = float(max(ref.max(), est.max()) * 1.1) if len(ref) else 1.0
    lims = [0.0, upper]
    ax.plot(lims, lims, "--", color="gray", label="Identidad (y = x)")

    if len(ref) >= 2 and np.std(ref) > 0:
        slope, intercept = np.polyfit(ref, est, 1)
        xs = np.array(lims)
        ax.plot(xs, slope * xs + intercept, color="#C44E52", label=f"Ajuste: y = {slope:.2f}x + {intercept:.1f}")

    ax.scatter(ref, est, color="#4C72B0", edgecolors="black", zorder=5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Velocidad de referencia (km/h)")
    ax.set_ylabel("Velocidad estimada (km/h)")
    ax.set_title("Validación: estimado vs. referencia")
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Gráfica de validación (dispersión) guardada en %s", out_path)


def plot_bland_altman(pairs: List[Tuple[Dict[str, Any], TrackSpeed]], out_path: str) -> None:
    """
    Diferencia (estimado - referencia) contra el promedio de ambas
    mediciones, con el sesgo y los límites de concordancia (±1.96σ). Revela
    si el error depende de la magnitud de la velocidad (un patrón en
    abanico indicaría eso).

    Retorna:
        None
    """
    ref = np.array([r["speed_ref_kmh"] for r, _ in pairs], dtype=np.float64)
    est = np.array([e.speed_mean_kmh for _, e in pairs], dtype=np.float64)
    mean_vals = (ref + est) / 2.0
    diff = est - ref
    bias = float(np.mean(diff))
    std = float(np.std(diff))
    loa_lower, loa_upper = bias - 1.96 * std, bias + 1.96 * std

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(mean_vals, diff, color="#4C72B0", edgecolors="black", zorder=5)
    ax.axhline(bias, color="#C44E52", linestyle="-", label=f"Sesgo: {bias:.2f} km/h")
    ax.axhline(loa_lower, color="gray", linestyle="--", label="Límites de concordancia (±1.96σ)")
    ax.axhline(loa_upper, color="gray", linestyle="--")
    ax.set_xlabel("Promedio (estimado, referencia) [km/h]")
    ax.set_ylabel("Diferencia: estimado - referencia [km/h]")
    ax.set_title("Bland-Altman")
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Gráfica de Bland-Altman guardada en %s", out_path)


def plot_error_vs_distance(pairs: List[Tuple[Dict[str, Any], TrackSpeed]], out_path: str) -> None:
    """
    Error absoluto (km/h) contra la distancia media del track a la cámara.
    Debería mostrar degradación con la lejanía (consistente con el gradiente
    de escala); si no aparece, conviene sospechar de los datos.

    No genera nada si ningún par tiene `mean_distance_m` (calibración sin
    puntos registrados).

    Retorna:
        None
    """
    dists, errs = [], []
    for r, e in pairs:
        if e.mean_distance_m is not None:
            dists.append(e.mean_distance_m)
            errs.append(e.speed_mean_kmh - r["speed_ref_kmh"])

    if not dists:
        logger.warning("plot_error_vs_distance: ningún par tiene mean_distance_m; no se generó la gráfica.")
        return

    dists_arr, errs_arr = np.array(dists), np.abs(np.array(errs))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dists_arr, errs_arr, color="#4C72B0", edgecolors="black", zorder=5)
    if len(dists_arr) >= 2 and np.std(dists_arr) > 0:
        slope, intercept = np.polyfit(dists_arr, errs_arr, 1)
        xs = np.linspace(dists_arr.min(), dists_arr.max(), 50)
        ax.plot(xs, slope * xs + intercept, color="#C44E52", label=f"Tendencia: {slope:.3f} km/h por metro")
        ax.legend()
    ax.set_xlabel("Distancia media del track a la cámara (m)")
    ax.set_ylabel("Error absoluto (km/h)")
    ax.set_title("Error vs. distancia a la cámara")
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Gráfica de error vs. distancia guardada en %s", out_path)


def interpret_bias_dispersion(metrics: Dict[str, Any], bias_threshold_kmh: float = 3.0, std_threshold_kmh: float = 3.0) -> str:
    """
    Aplica la guía de interpretación del docstring del módulo a un resultado
    concreto de `SpeedValidation.metrics()`, y devuelve el mensaje
    correspondiente en español, listo para consola o para el reporte.

    Parámetros:
        metrics (dict): salida de `metrics()`.
        bias_threshold_kmh (float): a partir de aquí, el sesgo se considera "grande".
        std_threshold_kmh (float): a partir de aquí, la dispersión se considera "grande".

    Retorna:
        str: mensaje de interpretación.
    """
    bias, std = abs(metrics["bias_kmh"]), metrics["std_error_kmh"]
    bias_big, std_big = bias >= bias_threshold_kmh, std >= std_threshold_kmh

    if bias_big and not std_big:
        return (
            f"Sesgo grande ({metrics['bias_kmh']:+.2f} km/h) con dispersión pequeña ({std:.2f} km/h): "
            f"apunta a un error de escala en la calibración. Es sistemático — revisá las coordenadas del "
            f"mundo (decimales, puntos mal emparejados), no el estimador ni el seguimiento."
        )
    if std_big and not bias_big:
        return (
            f"Sesgo cercano a cero ({metrics['bias_kmh']:+.2f} km/h) con dispersión grande ({std:.2f} km/h): "
            f"apunta a ruido de detección y seguimiento. Probá una ventana de ajuste más larga, más "
            f"suavizado, o restringí la medición a la zona donde la escala es favorable."
        )
    if bias_big and std_big:
        return (
            f"Sesgo ({metrics['bias_kmh']:+.2f} km/h) y dispersión ({std:.2f} km/h) grandes: revisá primero "
            f"la calibración — el sesgo puede estar enmascarando el diagnóstico real de la dispersión."
        )
    return (
        f"Sesgo ({metrics['bias_kmh']:+.2f} km/h) y dispersión ({std:.2f} km/h) dentro de rangos razonables: "
        f"no hay una causa dominante evidente."
    )

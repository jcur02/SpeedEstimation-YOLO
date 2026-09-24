"""
Gráficas de resultados de velocidad sobre datos reales (a diferencia de
`src.speed.synthetic`, que grafica el comportamiento del estimador sobre
datos sintéticos, y de `src.speed.validation`, que grafica el acuerdo contra
mediciones de referencia).
"""
import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.calibration.homography import CalibratedPlane
    from src.speed.estimator import TrackSpeed

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def plot_speed_profile(track_speed: "TrackSpeed", out_path: str) -> None:
    """
    Velocidad contra tiempo de un único track, con la media y una banda de
    ±1 desviación estándar. Para un vehículo a velocidad sostenida, el
    perfil debe verse aproximadamente plano; picos aislados suelen indicar
    un cambio de identidad o suavizado insuficiente.

    No genera nada si el track no tiene perfil (inválido/rechazado).

    Retorna:
        None
    """
    if not track_speed.profile:
        logger.warning("plot_speed_profile: track %s no tiene perfil (inválido); no se generó la gráfica.", track_speed.track_id)
        return

    t = np.array([p[0] for p in track_speed.profile])
    v = np.array([p[1] for p in track_speed.profile])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, v, color="#4C72B0", marker="o", markersize=3, linewidth=1, label="Velocidad instantánea")
    ax.axhline(track_speed.speed_mean_kmh, color="#C44E52", linestyle="--", label=f"Media: {track_speed.speed_mean_kmh:.1f} km/h")
    ax.fill_between(
        t, track_speed.speed_mean_kmh - track_speed.speed_std_kmh, track_speed.speed_mean_kmh + track_speed.speed_std_kmh,
        color="#C44E52", alpha=0.15, label="±1 desviación estándar",
    )
    ax.set_xlabel("t_video (s)")
    ax.set_ylabel("Velocidad (km/h)")
    ax.set_title(f"Perfil de velocidad — track {track_speed.track_id} ({track_speed.class_name or '?'})")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Perfil de velocidad guardado en %s", out_path)


def plot_speed_histogram(results: List["TrackSpeed"], out_path: str) -> None:
    """
    Distribución de velocidades medias, separada por clase de vehículo.

    Retorna:
        None
    """
    valid = [r for r in results if r.is_valid]
    if not valid:
        logger.warning("plot_speed_histogram: no hay tracks válidos; no se generó la gráfica.")
        return

    all_speeds = [r.speed_mean_kmh for r in valid]
    classes = sorted({r.class_name or "desconocida" for r in valid})
    bins = np.linspace(0, max(all_speeds) * 1.05 if max(all_speeds) > 0 else 1.0, 25)

    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in classes:
        speeds = [r.speed_mean_kmh for r in valid if (r.class_name or "desconocida") == cls]
        ax.hist(speeds, bins=bins, alpha=0.6, label=f"{cls} (n={len(speeds)})")
    ax.set_xlabel("Velocidad media (km/h)")
    ax.set_ylabel("Cantidad de vehículos")
    ax.set_title("Distribución de velocidades por clase")
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Histograma de velocidades guardado en %s", out_path)


def plot_trajectories_metric(results: List["TrackSpeed"], calibration: "CalibratedPlane", out_path: str) -> None:
    """
    Todas las trayectorias válidas en el plano métrico, coloreadas por
    velocidad, con la zona calibrada (casco convexo de los puntos de
    calibración) marcada de fondo. Es la vista de conjunto más informativa:
    se ven de golpe los carriles, los sentidos de circulación y las
    trayectorias anómalas.

    Retorna:
        None
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    world_points = getattr(calibration, "world_points", None)
    if world_points is not None and len(world_points) >= 3:
        hull = cv2.convexHull(np.asarray(world_points, dtype=np.float32)).reshape(-1, 2)
        hull_closed = np.vstack([hull, hull[0]])
        ax.plot(hull_closed[:, 0], hull_closed[:, 1], "--", color="gray", linewidth=1, label="Zona calibrada")

    valid = [r for r in results if r.is_valid and r.positions_m]
    if not valid:
        ax.set_title("Trayectorias en el plano métrico (sin tracks válidos)")
        ax.set_xlabel("Este (m)")
        ax.set_ylabel("Norte (m)")
        fig.tight_layout()
        _ensure_dir(out_path)
        fig.savefig(out_path, dpi=100)
        plt.close(fig)
        logger.warning("plot_trajectories_metric: no hay tracks válidos con posiciones; se guardó una gráfica vacía.")
        return

    all_speeds = [r.speed_mean_kmh for r in valid]
    vmin, vmax = min(all_speeds), max(all_speeds)
    cmap = plt.get_cmap("turbo")
    norm = plt.Normalize(vmin=vmin, vmax=vmax if vmax > vmin else vmin + 1.0)

    for r in valid:
        pts = np.array(r.positions_m)
        color = cmap(norm(r.speed_mean_kmh))
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=1.5, alpha=0.85)
        ax.scatter(pts[0, 0], pts[0, 1], color=color, marker="o", s=15, zorder=5)
        ax.scatter(pts[-1, 0], pts[-1, 1], color=color, marker="s", s=15, zorder=5)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("Velocidad media (km/h)")

    ax.set_xlabel("Este (m)")
    ax.set_ylabel("Norte (m)")
    ax.set_title("Trayectorias en el plano métrico (○ inicio, □ fin)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Trayectorias en el plano métrico guardadas en %s (%d tracks)", out_path, len(valid))


def plot_estimator_comparison(results: Dict[str, List["TrackSpeed"]], out_path: str) -> None:
    """
    Los tres estimadores sobre los mismos tracks, para mostrar de un vistazo
    la diferencia entre el sesgado (acumulado) y los otros dos.

    Parámetros:
        results (dict): `{"window": [...], "endpoint": [...], "cumulative": [...]}`,
            resultado de estimar el mismo archivo de trayectorias con cada
            uno de los tres métodos (mismos `track_id`).
        out_path (str): ruta del PNG de salida.

    Retorna:
        None
    """
    colors = {"window": "#4C72B0", "endpoint": "#55A868", "cumulative": "#C44E52"}
    labels = {"window": "Ventana deslizante", "endpoint": "Extremos", "cumulative": "Acumulado (sesgado)"}
    estimator_names = [name for name in ("window", "endpoint", "cumulative") if name in results]

    common_ids: Any = None
    for name in estimator_names:
        ids = {r.track_id for r in results[name] if r.is_valid}
        common_ids = ids if common_ids is None else (common_ids & ids)
    common_ids = sorted(common_ids or [])

    if not common_ids:
        logger.warning("plot_estimator_comparison: no hay tracks válidos en común entre los estimadores; no se generó la gráfica.")
        return

    by_name_by_id = {name: {r.track_id: r for r in results[name]} for name in estimator_names}

    fig, ax = plt.subplots(figsize=(max(6, len(common_ids) * 0.6), 5))
    width = 0.8 / len(estimator_names)
    x = np.arange(len(common_ids))
    for i, name in enumerate(estimator_names):
        speeds = [by_name_by_id[name][tid].speed_mean_kmh for tid in common_ids]
        offset = (i - (len(estimator_names) - 1) / 2.0) * width
        ax.bar(x + offset, speeds, width=width, label=labels[name], color=colors[name])

    ax.set_xticks(x)
    ax.set_xticklabels([str(tid) for tid in common_ids], rotation=90 if len(common_ids) > 15 else 0)
    ax.set_xlabel("Track ID")
    ax.set_ylabel("Velocidad estimada (km/h)")
    ax.set_title("Comparación de estimadores sobre los mismos tracks")
    ax.legend()
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Comparación de estimadores guardada en %s (%d tracks en común)", out_path, len(common_ids))

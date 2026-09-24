"""
Reportes visuales de la calibración: mapa de escala, vista cenital
rectificada y gráfica de residuos.

Estas tres imágenes son, en conjunto, la verificación de que la homografía
es razonable — mucho más convincente a primera vista que cualquier número
aislado. En particular la vista cenital (`render_birdseye`): si los bordes
de la calle salen rectos y paralelos, la calibración es correcta; si
convergen, se curvan o el carril se ensancha con la distancia, está mal, y
eso se ve en dos segundos sin leer un solo número.
"""
import logging
import os
from typing import TYPE_CHECKING, List, Optional

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from src.calibration.homography import HomographyCalibration

logger = logging.getLogger(__name__)


def _ensure_dir(path: str) -> None:
    """Crea el directorio contenedor de `path` si hace falta."""
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def render_scale_map(
    frame: np.ndarray,
    calibration: "HomographyCalibration",
    out_path: str,
    grid_step: int = 40,
) -> None:
    """
    Superpone el mapa de escala (metros por píxel, mostrado en cm/píxel)
    sobre el frame como mapa de calor, con curvas de nivel etiquetadas y una
    barra de color. Marca los puntos de calibración y el casco convexo que
    forman, para que se vea de un vistazo qué zona está respaldada por datos
    y cuál es extrapolación.

    Parámetros:
        frame (numpy.ndarray): frame de fondo (BGR).
        calibration (HomographyCalibration): calibración ya ajustada
            (`fit()` ya llamado).
        out_path (str): ruta del PNG de salida.
        grid_step (int): separación de la rejilla de muestreo, en píxeles.

    Retorna:
        None
    """
    scale_data = calibration.scale_map(grid_step=grid_step)
    xs = np.array(scale_data["xs"])
    ys = np.array(scale_data["ys"])
    grid_cm_per_px = np.array(scale_data["grid_m_per_px"]) * 100.0

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(frame_rgb, extent=(0, width, height, 0))

    xx, yy = np.meshgrid(xs, ys)
    contourf = ax.contourf(xx, yy, grid_cm_per_px, levels=12, cmap="turbo", alpha=0.55)
    contour_lines = ax.contour(xx, yy, grid_cm_per_px, levels=8, colors="white", linewidths=0.6)
    ax.clabel(contour_lines, inline=True, fontsize=7, fmt="%.0f cm/px")
    cbar = fig.colorbar(contourf, ax=ax, shrink=0.8)
    cbar.set_label("cm/píxel")

    pixel_points = calibration.pixel_points
    hull = cv2.convexHull(pixel_points.astype(np.float32)).reshape(-1, 2)
    hull_closed = np.vstack([hull, hull[0]])
    ax.plot(
        hull_closed[:, 0], hull_closed[:, 1],
        color="yellow", linewidth=1.5, linestyle="--", label="Casco convexo (zona con datos)",
    )
    ax.scatter(
        pixel_points[:, 0], pixel_points[:, 1],
        color="red", edgecolors="black", zorder=5, label="Puntos de calibración",
    )

    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.legend(loc="upper right", fontsize=8)
    ax.axis("off")
    fig.suptitle("Mapa de escala (cm/píxel) — fuera del casco convexo es extrapolación", fontsize=10)

    _ensure_dir(out_path)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Mapa de escala guardado en %s", out_path)


def render_birdseye(
    frame: np.ndarray,
    calibration: "HomographyCalibration",
    out_path: str,
    px_per_meter: int = 20,
    margin_m: float = 5.0,
) -> None:
    """
    Rectifica el frame completo al plano métrico con `cv2.warpPerspective`:
    la verificación visual más convincente de todas.

    Cómo se lee: si la calibración es correcta, los bordes de la calle salen
    **rectos y paralelos**, y el ancho del carril se mantiene constante de
    cerca a lejos. Si los bordes convergen, se curvan o el carril se
    ensancha con la distancia, la calibración está mal.

    Los límites en metros se calculan proyectando las cuatro esquinas del
    frame, pero **acotados** al entorno de los puntos de calibración más el
    margen: sin acotar, la proyección del horizonte tiende a infinito y la
    imagen resultante es inmanejable.

    Parámetros:
        frame (numpy.ndarray): frame de fondo (BGR).
        calibration (HomographyCalibration): calibración ya ajustada.
        out_path (str): ruta del PNG de salida.
        px_per_meter (int): resolución del canvas rectificado.
        margin_m (float): margen alrededor de los puntos de calibración, en
            metros.

    Retorna:
        None
    """
    height, width = frame.shape[:2]
    corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    corners_world = calibration.project_pixel_to_world(corners_px)

    cal_pts = calibration.world_points
    cal_min = cal_pts.min(axis=0) - margin_m
    cal_max = cal_pts.max(axis=0) + margin_m

    finite_mask = np.all(np.isfinite(corners_world), axis=1)
    if finite_mask.any():
        corner_min = corners_world[finite_mask].min(axis=0)
        corner_max = corners_world[finite_mask].max(axis=0)
    else:
        corner_min, corner_max = cal_min, cal_max

    world_min = np.maximum(corner_min, cal_min)
    world_max = np.minimum(corner_max, cal_max)
    if np.any(world_max <= world_min):
        # El horizonte se proyectó fuera de rango, o las esquinas del frame no
        # llegan a cubrir la zona calibrada: usar directamente el entorno de
        # los puntos de calibración en vez de un rango vacío/invertido.
        world_min, world_max = cal_min, cal_max

    canvas_w = max(int(np.ceil((world_max[0] - world_min[0]) * px_per_meter)), 10)
    canvas_h = max(int(np.ceil((world_max[1] - world_min[1]) * px_per_meter)), 10)

    to_canvas = np.array([
        [px_per_meter, 0.0, -world_min[0] * px_per_meter],
        [0.0, -px_per_meter, world_max[1] * px_per_meter],
        [0.0, 0.0, 1.0],
    ])
    h_full = to_canvas @ calibration.H

    birdseye = cv2.warpPerspective(frame, h_full, (canvas_w, canvas_h))

    def world_to_canvas(x_m: float, y_m: float) -> tuple:
        pt = to_canvas @ np.array([x_m, y_m, 1.0])
        return int(round(pt[0] / pt[2])), int(round(pt[1] / pt[2]))

    grid_x0, grid_x1 = int(np.floor(world_min[0])), int(np.ceil(world_max[0]))
    grid_y0, grid_y1 = int(np.floor(world_min[1])), int(np.ceil(world_max[1]))

    for gx in range(grid_x0, grid_x1 + 1):
        p1 = world_to_canvas(gx, world_min[1])
        p2 = world_to_canvas(gx, world_max[1])
        major = gx % 5 == 0
        cv2.line(birdseye, p1, p2, (255, 255, 255) if major else (110, 110, 110), 1)
        if major:
            cv2.putText(birdseye, f"{gx}m", (p1[0] + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    for gy in range(grid_y0, grid_y1 + 1):
        p1 = world_to_canvas(world_min[0], gy)
        p2 = world_to_canvas(world_max[0], gy)
        major = gy % 5 == 0
        cv2.line(birdseye, p1, p2, (255, 255, 255) if major else (110, 110, 110), 1)
        if major:
            cv2.putText(birdseye, f"{gy}m", (2, p1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    for pt_m in cal_pts:
        cx, cy = world_to_canvas(float(pt_m[0]), float(pt_m[1]))
        cv2.circle(birdseye, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(birdseye, (cx, cy), 5, (255, 255, 255), 1)

    _ensure_dir(out_path)
    cv2.imwrite(out_path, birdseye)
    logger.info("Vista cenital guardada en %s", out_path)


def render_residuals(
    calibration: "HomographyCalibration",
    out_path: str,
    labels: Optional[List[str]] = None,
) -> None:
    """
    Gráfica de barras del residuo de reproyección por punto, en metros, con
    una línea horizontal en el promedio y el peor punto destacado en color
    distinto.

    Parámetros:
        calibration (HomographyCalibration): calibración ya ajustada.
        out_path (str): ruta del PNG de salida.
        labels (list[str] | None): etiquetas de cada punto, en el mismo
            orden usado al construir la calibración. Si es None, se usan
            índices 1-based.

    Retorna:
        None
    """
    residuals_m = calibration.residuals()
    n = len(residuals_m)
    point_labels = labels if labels else [str(i + 1) for i in range(n)]
    worst_idx = int(np.argmax(residuals_m))
    colors = ["#4C72B0"] * n
    colors[worst_idx] = "#C44E52"

    fig, ax = plt.subplots(figsize=(max(4, n * 0.8), 4))
    ax.bar(point_labels, residuals_m, color=colors)
    mean_val = float(np.mean(residuals_m))
    ax.axhline(mean_val, color="gray", linestyle="--", linewidth=1, label=f"Promedio: {mean_val:.3f} m")
    ax.set_ylabel("Error de reproyección (m)")
    ax.set_xlabel("Punto")
    ax.set_title("Residuo de reproyección por punto")
    ax.legend()
    fig.tight_layout()

    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    logger.info("Gráfica de residuos guardada en %s", out_path)

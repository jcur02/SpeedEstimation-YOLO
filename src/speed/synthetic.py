"""
Banco de pruebas sintético para el estimador de velocidad.

La calibración real del sitio puede no existir todavía (ver el aviso en el
docstring del módulo del paquete). Este banco no depende de ella en absoluto:
construye su propia homografía a partir de parámetros explícitos de cámara
(altura, inclinación, distancia focal) y genera trayectorias con velocidad
**exactamente conocida**, para poder medir el error del estimador en vez de
suponerlo.

Esa separación permite responder por separado dos preguntas que con datos
reales quedan confundidas: cuánto error introduce el **estimador** en sí, y
cuánto introducen la **calibración** y el **seguimiento**. Por diseño, las
trayectorias sintéticas tienen exactamente la misma forma que las reales (ver
`simulate_track`), así que `SpeedEstimator.estimate()` las procesa con el
mismo código, sin ninguna ruta especial para "modo prueba".
"""
import copy
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.calibration.homography import CalibratedPlane
from src.speed.estimator import SpeedEstimator

logger = logging.getLogger(__name__)

_KMH_TO_MS = 1.0 / 3.6

# Parámetros de cámara para las dos configuraciones representativas del
# sitio real. No son ajustables por config a propósito: son fijos del banco,
# elegidos para producir un gradiente de escala realista (una cámara de unos
# 6 m de altura mirando la calzada con una inclinación moderada).
_CAMERA_HEIGHT_M = 6.0
_CAMERA_TILT_DEG = 20.0
_CAMERA_FOCAL_PX = 800.0
_CAMERA_FRAME_SIZE = (1920, 1080)

# Distancia a la cámara (metros) de cada tramo representativo. El vehículo
# siempre viaja físicamente en +Y (dirección de la vía); cuál eje del mundo
# es "profundidad" respecto a la cámara depende de la geometría (ver
# `_start_position`), así que estos valores se mapean al eje que corresponda.
_ZONES: Dict[str, float] = {"cercano": 15.0, "medio": 40.0, "lejano": 80.0}

# Desplazamiento lateral fijo del vehículo (un carril típico), en el eje que
# en cada geometría NO actúa como profundidad.
_LATERAL_OFFSET_M = 5.0

# yaw_deg de `make_synthetic_homography` para cada geometría representativa.
_GEOMETRIES: Dict[str, float] = {
    "longitudinal": 0.0,
    "transversal": 90.0,
}


def _start_position(geometry_name: str, zone_depth_m: float) -> Tuple[float, float]:
    """
    Ubicación inicial `(este, norte)` para un tramo, según qué eje del mundo
    es la profundidad respecto a la cámara en esta geometría.

    En **longitudinal** (la cámara mira a lo largo de +Y) la profundidad es
    Y: el vehículo arranca a `zone_depth_m` de distancia y se aleja viajando
    en +Y, así que su profundidad varía mucho durante el tramo (es
    precisamente el efecto que esa geometría debe exhibir). En
    **transversal** (la cámara mira a lo largo de +X) la profundidad es X:
    el vehículo mantiene `zone_depth_m` de profundidad **constante** todo el
    tramo, porque viaja en +Y, que ahí es el eje lateral en la imagen.
    """
    if geometry_name == "longitudinal":
        return (_LATERAL_OFFSET_M, zone_depth_m)
    return (zone_depth_m, _LATERAL_OFFSET_M)

_ESTIMATOR_NAMES = ("window", "endpoint", "cumulative")
_ESTIMATOR_LABELS = {"window": "Ventana deslizante", "endpoint": "Extremos", "cumulative": "Acumulado (sesgado)"}
_ESTIMATOR_COLORS = {"window": "#4C72B0", "endpoint": "#55A868", "cumulative": "#C44E52"}


def make_synthetic_homography(
    camera_height_m: float, tilt_deg: float, focal_px: float, frame_size: Tuple[int, int],
    yaw_deg: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construye una homografía píxel↔metro plausible a partir de parámetros de
    cámara explícitos (modelo pinhole, sin distorsión de lente), proyectando
    el plano del suelo.

    Sistema de coordenadas del mundo: `X` lateral, `Y` a lo largo de la vía
    (dirección de circulación de los vehículos), `Z` hacia arriba; el origen
    es el punto del suelo directamente debajo de la cámara.

    Derivación (convención de visión por computadora: eje x de cámara hacia
    la derecha, eje y hacia abajo en la imagen, eje z el óptico/hacia adelante):

    1. Con `tilt_deg = 0` (mirando al horizonte) y `yaw_deg = 0`, la cámara
       mira a lo largo de `+Y` horizontalmente: `cam_z = (0,1,0)`, y "abajo en
       la imagen" es hacia el suelo, `cam_y = (0,0,-1)`.
    2. `tilt_deg` rota `cam_y`/`cam_z` alrededor del eje `cam_x` (que la rotación
       deja fijo), inclinando la mirada hacia abajo: con `tilt_deg = 90` la
       cámara mira derecho hacia abajo (`cam_z = (0,0,-1)`, nadir).
    3. `yaw_deg` rota luego los tres ejes alrededor de `Z` (mundo), fijando
       hacia qué dirección horizontal apunta la cámara respecto a la vía:
       `yaw_deg = 0` → **vista longitudinal** (la cámara mira a lo largo de
       la vía: la dirección de circulación coincide con el eje de
       profundidad de la cámara, así que la escala es muy desigual —
       vehículos lejos ocupan muchos menos píxeles por metro que cerca).
       `yaw_deg = 90` → **vista transversal** (la cámara mira perpendicular
       a la vía: la dirección de circulación queda de lado a lado en la
       imagen a profundidad aproximadamente constante, con escala pareja).
    4. Los tres ejes de cámara resultantes, expresados en el mundo, son las
       FILAS de la matriz de rotación mundo→cámara `R` (un punto del mundo se
       expresa en coordenadas de cámara proyectándolo sobre cada eje).
    5. Para puntos del plano del suelo (`Z=0`), la tercera columna de `R` (la
       que multiplicaría a `Z`) no interviene: la proyección completa
       `K [R|t]` se reduce a una homografía 3×3 que toma `(X, Y, 1)` en vez de
       `(X, Y, Z, 1)`.

    Parámetros:
        camera_height_m (float): altura de la cámara sobre el suelo, en metros.
        tilt_deg (float): inclinación medida desde la horizontal (0° = mirando
            al horizonte, 90° = mirando derecho hacia abajo).
        focal_px (float): distancia focal en píxeles (`fx = fy`, sin distorsión).
        frame_size (tuple[int, int]): `(ancho, alto)` en píxeles.
        yaw_deg (float): orientación horizontal respecto a la vía (ver arriba).

    Retorna:
        tuple[numpy.ndarray, numpy.ndarray]: `(H, H_inv)`, ambas 3×3. `H`
        mapea píxeles a metros (misma convención que `CalibratedPlane.H`);
        `H_inv` mapea metros a píxeles.
    """
    theta = np.radians(tilt_deg)
    phi = np.radians(yaw_deg)

    cam_x = np.array([np.cos(phi), -np.sin(phi), 0.0])
    cam_y = np.array([-np.sin(theta) * np.sin(phi), -np.sin(theta) * np.cos(phi), -np.cos(theta)])
    cam_z = np.array([np.cos(theta) * np.sin(phi), np.cos(theta) * np.cos(phi), -np.sin(theta)])

    r_mat = np.stack([cam_x, cam_y, cam_z], axis=0)  # mundo -> cámara

    camera_center_world = np.array([0.0, 0.0, camera_height_m])
    t_vec = -r_mat @ camera_center_world

    # Homografía mundo(X,Y,1) -> cámara, ignorando la columna de Z (plano del suelo).
    m_mat = np.column_stack([r_mat[:, 0], r_mat[:, 1], t_vec])

    fx = fy = float(focal_px)
    cx, cy = frame_size[0] / 2.0, frame_size[1] / 2.0
    k_mat = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    h_inv = k_mat @ m_mat  # metros -> píxeles
    h_mat = np.linalg.inv(h_inv)  # píxeles -> metros
    return h_mat, h_inv


def simulate_track(
    speed_kmh: float, fps: float, duration_s: float, h_inv: np.ndarray,
    start_m: Tuple[float, float], direction: Tuple[float, float], noise_px: float,
    quantize: bool = True, accel_kmh_s: float = 0.0, track_id: int = 1,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Genera una trayectoria sintética con velocidad exactamente conocida.

    1. Calcula posiciones métricas de un vehículo en línea recta a
       `speed_kmh` (más `accel_kmh_s` opcional).
    2. Las proyecta a píxeles con `h_inv`.
    3. Agrega ruido gaussiano de desviación `noise_px` por eje, simulando el
       temblor de la caja delimitadora entre frames.
    4. Si `quantize`, redondea a píxeles enteros, como ocurre en la realidad
       (las cajas de YOLO son enteras).
    5. Devuelve un dict con la misma forma que produce `Track.to_dict()`, para
       que `SpeedEstimator` lo procese sin ninguna ruta de código especial.

    Parámetros:
        speed_kmh (float): velocidad constante (o inicial, si `accel_kmh_s != 0`).
        fps (float): frames por segundo simulados.
        duration_s (float): duración del track, en segundos.
        h_inv (numpy.ndarray): homografía metros→píxeles (ver `make_synthetic_homography`).
        start_m (tuple[float, float]): posición inicial `(este, norte)`, en metros.
        direction (tuple[float, float]): dirección de circulación (no hace
            falta que esté normalizada).
        noise_px (float): desviación estándar del ruido gaussiano, en píxeles.
        quantize (bool): si True, redondea las posiciones de píxel a enteros.
        accel_kmh_s (float): aceleración constante, en (km/h) por segundo.
            Permite evaluar el estimador fuera del supuesto de velocidad
            constante.
        track_id (int): id a asignar al track sintético.
        rng (numpy.random.Generator | None): generador reproducible. Si es
            None, se crea uno sin semilla fija (no reproducible) — para
            resultados reproducibles, siempre pasar uno construido con
            `np.random.default_rng(seed)`.

    Retorna:
        dict: track sintético, con dos claves extra respecto a un track real
        (`true_speed_kmh`, `true_accel_kmh_s`) para poder comparar contra la
        velocidad exacta con la que se generó.
    """
    if rng is None:
        rng = np.random.default_rng()

    direction_arr = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(direction_arr)
    if norm == 0:
        raise ValueError("`direction` no puede ser el vector nulo.")
    direction_unit = direction_arr / norm

    n_frames = max(int(round(duration_s * fps)), 2)
    speed_ms = speed_kmh * _KMH_TO_MS
    accel_ms2 = accel_kmh_s * _KMH_TO_MS  # (km/h)/s -> m/s^2: misma conversión km/h -> m/s
    start = np.asarray(start_m, dtype=np.float64)

    observations: List[Dict[str, Any]] = []
    for i in range(n_frames):
        t = i / fps
        traveled_m = speed_ms * t + 0.5 * accel_ms2 * t * t
        pos_m = start + direction_unit * traveled_m

        pt_h = h_inv @ np.array([pos_m[0], pos_m[1], 1.0])
        px, py = pt_h[0] / pt_h[2], pt_h[1] / pt_h[2]

        if noise_px > 0:
            px += rng.normal(0.0, noise_px)
            py += rng.normal(0.0, noise_px)
        if quantize:
            px, py = round(px), round(py)

        box_w, box_h = 40.0, 30.0
        bbox = [px - box_w / 2.0, py - box_h, px + box_w / 2.0, py]

        observations.append({
            "frame_idx": i,
            "t_video": t,
            "bbox": [float(v) for v in bbox],
            "confidence": 0.9,
            "class_id": 2,
            "ground_point": [float(px), float(py)],
        })

    first_pt, last_pt = observations[0]["ground_point"], observations[-1]["ground_point"]
    total_displacement_px = float(np.hypot(last_pt[0] - first_pt[0], last_pt[1] - first_pt[1]))

    return {
        "track_id": track_id,
        "class_name": "car",
        "first_frame": 0,
        "last_frame": n_frames - 1,
        "first_seen_t": observations[0]["t_video"],
        "last_seen_t": observations[-1]["t_video"],
        "length_frames": n_frames,
        "total_displacement_px": total_displacement_px,
        "is_stationary": False,
        "observations": observations,
        "true_speed_kmh": speed_kmh,
        "true_accel_kmh_s": accel_kmh_s,
    }


def _permissive_speed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copia `config` relajando los filtros de calidad de `speed.filters` a
    propósito: este banco mide la respuesta del **estimador** al ruido puro
    y a la geometría, no la interacción con el filtro de calidad (los
    filtros —zona calibrada, escala, saltos imposibles— se ejercitan con
    datos reales en `SpeedEstimator.estimate_all`, con los umbrales
    configurados normalmente). Sin este relajo, las combinaciones de mucho
    ruido a mucha distancia rechazarían observaciones por "escala excesiva"
    o "salto imposible" y esconderían justo el efecto que el banco quiere
    medir.
    """
    permissive = copy.deepcopy(config)
    filters = permissive.setdefault("speed", {}).setdefault("filters", {})
    filters["max_scale_cm_per_px"] = 1.0e9
    filters["max_plausible_kmh"] = 1.0e9
    filters["min_observations"] = 10
    filters["min_distance_m"] = 0.0
    return permissive


def _build_plane(yaw_deg: float) -> Tuple[CalibratedPlane, np.ndarray]:
    """Construye una `CalibratedPlane` en memoria (sin archivo) para una geometría dada."""
    h_mat, h_inv = make_synthetic_homography(
        _CAMERA_HEIGHT_M, _CAMERA_TILT_DEG, _CAMERA_FOCAL_PX, _CAMERA_FRAME_SIZE, yaw_deg=yaw_deg,
    )
    plane = CalibratedPlane({
        "homography": h_mat.tolist(),
        "homography_inv": h_inv.tolist(),
        "frame_size": list(_CAMERA_FRAME_SIZE),
    })
    return plane, h_inv


def _ensure_dir(path: str) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)


def _plot_error_vs_noise(df: pd.DataFrame, out_path: str) -> None:
    """
    Error absoluto (km/h) contra ruido (px), una curva por estimador.

    Agrega con la **mediana**, no el promedio: cerca del punto de fuga de la
    geometría longitudinal a distancia lejana, la escala metros/píxel crece
    sin cota y unas pocas repeticiones con error enorme (legítimo, no un
    bug) dominarían un promedio y taparían la tendencia típica.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = df.groupby(["estimator", "noise_px"])["error_kmh"].apply(lambda s: np.median(np.abs(s.dropna())))
    for name in _ESTIMATOR_NAMES:
        sub = grouped.loc[name].sort_index()
        ax.plot(sub.index, sub.values, marker="o", label=_ESTIMATOR_LABELS[name], color=_ESTIMATOR_COLORS[name])
    ax.set_xlabel("Ruido (desviación estándar, píxeles)")
    ax.set_ylabel("Error absoluto mediano (km/h)")
    ax.set_title("Error vs. ruido de detección")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_error_vs_distance(df: pd.DataFrame, out_path: str) -> None:
    """Error absoluto mediano (km/h) contra tramo (cercano/medio/lejano), un panel por geometría."""
    zone_order = list(_ZONES.keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, geometry in zip(axes, _GEOMETRIES.keys()):
        sub_df = df[df["geometry"] == geometry]
        grouped = sub_df.groupby(["estimator", "zone"])["error_kmh"].apply(lambda s: np.median(np.abs(s.dropna())))
        for name in _ESTIMATOR_NAMES:
            values = [grouped.get((name, z), np.nan) for z in zone_order]
            ax.plot(zone_order, values, marker="o", label=_ESTIMATOR_LABELS[name], color=_ESTIMATOR_COLORS[name])
        ax.set_title(geometry)
        ax.set_xlabel("Tramo")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Error absoluto mediano (km/h)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Error vs. distancia a la cámara, por geometría")
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def _plot_estimator_bias(df: pd.DataFrame, out_path: str) -> None:
    """La gráfica clave: sesgo (error CON signo, mediana) de cada estimador contra el ruido."""
    fig, ax = plt.subplots(figsize=(7, 5))
    grouped = df.groupby(["estimator", "noise_px"])["error_kmh"].apply(lambda s: np.median(s.dropna()))
    for name in _ESTIMATOR_NAMES:
        sub = grouped.loc[name].sort_index()
        ax.plot(sub.index, sub.values, marker="o", label=_ESTIMATOR_LABELS[name], color=_ESTIMATOR_COLORS[name])
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("Ruido (desviación estándar, píxeles)")
    ax.set_ylabel("Sesgo: error mediano con signo (km/h)")
    ax.set_title("Sesgo de cada estimador contra el ruido")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _ensure_dir(out_path)
    fig.savefig(out_path, dpi=100)
    plt.close(fig)


def run_suite(config: Dict[str, Any], out_dir: str) -> Dict[str, Any]:
    """
    Barrido sistemático del estimador sobre trayectorias sintéticas: para
    cada combinación de velocidad, ruido, distancia, geometría y estimador,
    corre varias repeticiones con semillas distintas (reproducibles) y agrega
    media y desviación del error.

    Genera `suite_{timestamp}.csv` con todos los resultados individuales, más
    tres gráficas (`error_vs_noise.png`, `error_vs_distance.png`,
    `estimator_bias.png`) en `out_dir`.

    También confirma los comportamientos esperados si la implementación es
    correcta (ver `checks` en el valor de retorno):
        - Con ruido cero, la ventana deslizante recupera la velocidad exacta
          (salvo el error numérico de la cuantización a píxel entero).
        - El error de la ventana deslizante crece con el ruido pero sin
          sesgo: el error medio con signo se mantiene cerca de cero.
        - El acumulado muestra sesgo positivo, creciente con el ruido.
        - El error es notablemente mayor en la geometría longitudinal que en
          la transversal, para el mismo ruido.

    Parámetros:
        config (dict): configuración completa (sección `speed.synthetic`).
        out_dir (str): directorio de salida para el CSV y las gráficas.

    Retorna:
        dict: `{"csv_path": str, "plot_paths": dict, "checks": dict, "summary": pandas.DataFrame}`.
        Cada entrada de `checks` es `{"passed": bool, "detail": str}`.
    """
    synth_cfg = config.get("speed", {}).get("synthetic", {})
    speeds = synth_cfg.get("speeds_kmh", [20, 30, 40, 50, 60, 80])
    noises = synth_cfg.get("noise_px", [0, 1, 2, 4, 8])
    repetitions = synth_cfg.get("repetitions", 30)
    base_seed = synth_cfg.get("seed", 42)
    fps = synth_cfg.get("fps", 30.0)
    duration_s = synth_cfg.get("duration_s", 3.0)

    os.makedirs(out_dir, exist_ok=True)
    permissive_config = _permissive_speed_config(config)

    # Una SeedSequence raíz y semillas hijas por cada combinación: reproducible
    # de punta a punta (misma config -> mismos resultados) y sin solapar el
    # flujo de números aleatorios entre combinaciones distintas.
    seed_seq = np.random.SeedSequence(base_seed)

    rows: List[Dict[str, Any]] = []
    for geometry_name, yaw_deg in _GEOMETRIES.items():
        plane, h_inv = _build_plane(yaw_deg)
        estimator = SpeedEstimator(permissive_config, plane=plane)

        for zone_name, zone_depth_m in _ZONES.items():
            start_m = _start_position(geometry_name, zone_depth_m)
            for speed_kmh in speeds:
                for noise_px in noises:
                    child_seed = seed_seq.spawn(1)[0]
                    rng = np.random.default_rng(child_seed)

                    for rep in range(repetitions):
                        track = simulate_track(
                            speed_kmh=speed_kmh, fps=fps, duration_s=duration_s, h_inv=h_inv,
                            start_m=start_m, direction=(0.0, 1.0), noise_px=noise_px,
                            quantize=True, track_id=rep, rng=rng,
                        )
                        for est_name in _ESTIMATOR_NAMES:
                            fn = {
                                "window": estimator.estimate, "endpoint": estimator.estimate_endpoint,
                                "cumulative": estimator.estimate_cumulative,
                            }[est_name]
                            result = fn(track)
                            estimated = result.speed_mean_kmh if result.is_valid else np.nan
                            error_kmh = (estimated - speed_kmh) if result.is_valid else np.nan
                            rows.append({
                                "geometry": geometry_name, "zone": zone_name, "speed_kmh": speed_kmh,
                                "noise_px": noise_px, "estimator": est_name, "repetition": rep,
                                "estimated_speed_kmh": estimated, "error_kmh": error_kmh,
                                "is_valid": result.is_valid,
                            })

    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"suite_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    logger.info("Resultados del banco sintético guardados en %s (%d filas)", csv_path, len(df))

    plot_paths = {
        "error_vs_noise": os.path.join(out_dir, "error_vs_noise.png"),
        "error_vs_distance": os.path.join(out_dir, "error_vs_distance.png"),
        "estimator_bias": os.path.join(out_dir, "estimator_bias.png"),
    }
    _plot_error_vs_noise(df, plot_paths["error_vs_noise"])
    _plot_error_vs_distance(df, plot_paths["error_vs_distance"])
    _plot_estimator_bias(df, plot_paths["estimator_bias"])

    checks = _run_acceptance_checks(df)

    return {"csv_path": csv_path, "plot_paths": plot_paths, "checks": checks, "summary": df}


def _run_acceptance_checks(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Calcula los criterios de aceptación descritos en `run_suite`, a partir de
    sus resultados.

    Usa la **mediana**, no el promedio, para agregar error/sesgo. Cerca del
    punto de fuga de una cámara con inclinación baja (la geometría
    longitudinal a distancia lejana), la escala metros/píxel crece sin cota,
    así que el mismo ruido en píxeles puede traducirse ocasionalmente en un
    error en metros enorme: son observaciones legítimas (no un error del
    banco), pero de cola muy pesada, y un promedio se deja dominar por esas
    pocas peores repeticiones en vez de reflejar el comportamiento típico del
    estimador. La mediana es la forma estándar de resumir una distribución
    así sin que esa cola tape la señal que estos chequeos buscan confirmar.
    """
    checks: Dict[str, Dict[str, Any]] = {}

    window_zero_noise = df[(df["estimator"] == "window") & (df["noise_px"] == 0) & df["is_valid"]]
    mae_zero = float(np.median(np.abs(window_zero_noise["error_kmh"]))) if len(window_zero_noise) else float("nan")
    checks["ruido_cero_exacto"] = {
        "passed": bool(mae_zero < 1.0),
        "detail": f"Ventana deslizante con ruido cero: error absoluto mediano = {mae_zero:.3f} km/h (umbral: < 1.0 km/h, por la cuantización a píxel entero).",
    }

    # Excluye el tramo "lejano" de este chequeo puntual: ahí la escala es tan
    # gruesa (hasta ~2 m/px en longitudinal) que 8px de ruido representan
    # metros de salto, y NINGÚN estimador podría lucir sin sesgo — eso es
    # justamente la degradación con la distancia que confirma
    # `longitudinal_peor_que_transversal` y muestra `error_vs_distance.png`,
    # una pregunta distinta de si el ajuste en sí introduce sesgo cuando la
    # escala es razonable.
    measurable = df[df["zone"] != "lejano"]
    window_bias = measurable[(measurable["estimator"] == "window") & (measurable["noise_px"] > 0) & measurable["is_valid"]].groupby("noise_px")["error_kmh"].median()
    max_abs_bias = float(np.max(np.abs(window_bias))) if len(window_bias) else float("nan")
    checks["ventana_sin_sesgo"] = {
        "passed": bool(max_abs_bias < 2.0),
        "detail": (
            f"Ventana deslizante, |sesgo mediano| máximo entre niveles de ruido > 0 (tramos cercano+medio): "
            f"{max_abs_bias:.3f} km/h (umbral: < 2.0 km/h)."
        ),
    }

    cum_bias = df[(df["estimator"] == "cumulative") & df["is_valid"]].groupby("noise_px")["error_kmh"].median().sort_index()
    cum_bias_at_max_noise = float(cum_bias.iloc[-1]) if len(cum_bias) else float("nan")
    cum_bias_increasing = bool(len(cum_bias) >= 2 and cum_bias.iloc[-1] > cum_bias.iloc[0])
    checks["acumulado_sesgo_positivo_creciente"] = {
        "passed": bool(cum_bias_at_max_noise > 1.0 and cum_bias_increasing),
        "detail": (
            f"Acumulado: sesgo mediano al mayor ruido = {cum_bias_at_max_noise:.3f} km/h (umbral: > 1.0), "
            f"creciente respecto al menor ruido: {cum_bias_increasing}."
        ),
    }

    max_noise = df["noise_px"].max()
    window_high_noise = df[(df["estimator"] == "window") & (df["noise_px"] == max_noise) & df["is_valid"]]
    mae_longitudinal = float(np.median(np.abs(window_high_noise[window_high_noise["geometry"] == "longitudinal"]["error_kmh"])))
    mae_transversal = float(np.median(np.abs(window_high_noise[window_high_noise["geometry"] == "transversal"]["error_kmh"])))
    checks["longitudinal_peor_que_transversal"] = {
        "passed": bool(mae_longitudinal > mae_transversal),
        "detail": (
            f"Con ruido={max_noise}px: error absoluto mediano longitudinal = {mae_longitudinal:.3f} km/h vs. "
            f"transversal = {mae_transversal:.3f} km/h."
        ),
    }

    return checks


def estimate_error_budget(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Presupuesto de error: cuántos km/h aporta cada fuente —cuantización de
    píxeles, ruido de detección, geometría de la instalación— bajo
    condiciones representativas (velocidad y tramo medios).

    Corre sus propias simulaciones dirigidas (no reutiliza el CSV de
    `run_suite`, que no varía `quantize`): así puede aislar el aporte de la
    cuantización comparando `quantize=True` contra `quantize=False` con ruido
    cero, algo que el barrido general no necesita para sus propios fines.

    Es lo que permite afirmar en el informe, con respaldo numérico, que
    cierta fracción del error observado es irreducible dada la instalación
    (geometría, resolución de píxel), y cuál es en cambio atribuible al
    algoritmo o al ruido de detección.

    Parámetros:
        config (dict): configuración completa (sección `speed.synthetic`).

    Retorna:
        dict: presupuesto con `condiciones`, `cuantizacion_kmh`,
        `ruido_deteccion_kmh`, `geometria_kmh`, `total_transversal_kmh` y
        `total_longitudinal_kmh` (todos en km/h, salvo `condiciones`).
    """
    synth_cfg = config.get("speed", {}).get("synthetic", {})
    fps = synth_cfg.get("fps", 30.0)
    duration_s = synth_cfg.get("duration_s", 3.0)
    repetitions = max(synth_cfg.get("repetitions", 30), 10)
    base_seed = synth_cfg.get("seed", 42)

    representative_speed = 50.0
    representative_noise = 2.0
    mid_zone_depth = _ZONES["medio"]
    permissive_config = _permissive_speed_config(config)

    def _mean_abs_error(geometry_name: str, noise_px: float, quantize: bool, seed_offset: int) -> float:
        plane, h_inv = _build_plane(_GEOMETRIES[geometry_name])
        estimator = SpeedEstimator(permissive_config, plane=plane)
        rng = np.random.default_rng(base_seed + seed_offset)
        start_m = _start_position(geometry_name, mid_zone_depth)
        errors = []
        for rep in range(repetitions):
            track = simulate_track(
                speed_kmh=representative_speed, fps=fps, duration_s=duration_s, h_inv=h_inv,
                start_m=start_m, direction=(0.0, 1.0), noise_px=noise_px, quantize=quantize,
                track_id=rep, rng=rng,
            )
            result = estimator.estimate(track)
            if result.is_valid:
                errors.append(result.speed_mean_kmh - representative_speed)
        return float(np.mean(np.abs(errors))) if errors else float("nan")

    baseline = _mean_abs_error("transversal", 0.0, quantize=False, seed_offset=1000)
    quantization_only = _mean_abs_error("transversal", 0.0, quantize=True, seed_offset=2000)
    detection_noise = _mean_abs_error("transversal", representative_noise, quantize=True, seed_offset=3000)
    geometry_transversal = _mean_abs_error("transversal", representative_noise, quantize=True, seed_offset=4000)
    geometry_longitudinal = _mean_abs_error("longitudinal", representative_noise, quantize=True, seed_offset=4000)

    return {
        "condiciones": {
            "speed_kmh": representative_speed, "noise_px": representative_noise, "zone_depth_m": mid_zone_depth,
        },
        "baseline_kmh": baseline,
        "cuantizacion_kmh": max(quantization_only - baseline, 0.0),
        "ruido_deteccion_kmh": max(detection_noise - quantization_only, 0.0),
        "geometria_kmh": max(geometry_longitudinal - geometry_transversal, 0.0),
        "total_transversal_kmh": detection_noise,
        "total_longitudinal_kmh": geometry_longitudinal,
    }

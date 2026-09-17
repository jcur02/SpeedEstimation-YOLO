"""
Trayectorias de vehículos y base de tiempo para la etapa de seguimiento.

Sin identidad persistente no existe la velocidad: la velocidad es el
desplazamiento del **mismo** vehículo entre dos instantes. Este módulo define
la unidad que representa esa identidad (`Track`) y la clase que calcula esos
instantes de forma correcta (`TimeBase`).
"""
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


class TimeBase:
    """
    Calcula el tiempo asociado a un frame de video a partir de su índice y
    los FPS del video — nunca del reloj de pared.

    Este es el punto más peligroso de toda la etapa de velocidad. El tiempo
    de un frame debe calcularse siempre como `t_video = frame_idx / source_fps`,
    **nunca** con `time.time()`, `time.perf_counter()` ni ningún reloj del
    sistema.

    Razón: en la Raspberry Pi 4 el procesamiento correrá más lento que tiempo
    real — por ejemplo, 8 FPS procesando un video grabado a 30 FPS. Si el
    tiempo se midiera con el reloj de pared, todas las velocidades saldrían
    subestimadas por un factor cercano a 4 (30/8). Es un error **silencioso**:
    los números resultantes parecen plausibles (una velocidad "razonable"),
    simplemente están mal, y nada en la salida delata el error. Por eso esta
    clase existe como punto único y explícito de cálculo del tiempo, en vez
    de dejarlo como una división suelta en medio de otro código donde sería
    fácil, más adelante, reemplazarla por algo que "parece" equivalente pero
    no lo es.
    """

    def __init__(self, source_fps: float) -> None:
        """
        Parámetros:
            source_fps (float): FPS reportados por el video de origen (no del
                hardware de procesamiento).

        Retorna:
            None

        Excepciones:
            ValueError: si `source_fps <= 0`. El video no reporta FPS
                válidos (metadata corrupta o contenedor sin esa información);
                en ese caso hay que forzarlos manualmente con `--fps`.
        """
        if source_fps <= 0:
            raise ValueError(
                f"El video reporta FPS inválidos ({source_fps}). No se puede calcular una base de "
                f"tiempo confiable a partir de él. Pasá los FPS reales manualmente con --fps."
            )
        self.source_fps = source_fps

    def time_of(self, frame_idx: int) -> float:
        """
        Parámetros:
            frame_idx (int): índice de frame (0-based).

        Retorna:
            float: instante en segundos de ese frame, según el video de origen.
        """
        return frame_idx / self.source_fps

    def delta(self, frame_a: int, frame_b: int) -> float:
        """
        Parámetros:
            frame_a (int): índice de frame inicial.
            frame_b (int): índice de frame final.

        Retorna:
            float: diferencia de tiempo en segundos entre ambos frames
            (`time_of(frame_b) - time_of(frame_a)`).
        """
        return self.time_of(frame_b) - self.time_of(frame_a)


class Track:
    """
    Representa la trayectoria de un único vehículo a lo largo del tiempo:
    todas sus observaciones (bbox + confianza + tiempo) bajo un mismo
    `track_id` persistente.
    """

    def __init__(self, track_id: int) -> None:
        """
        Crea un track vacío. Las observaciones se agregan con `add()`.

        Parámetros:
            track_id (int): identificador persistente asignado por el tracker.

        Retorna:
            None
        """
        self.track_id = track_id
        self.class_id: Optional[int] = None
        self.class_name: Optional[str] = None
        self._class_votes: Counter = Counter()

        self.observations: List[Dict[str, Any]] = []
        self.first_frame: Optional[int] = None
        self.last_frame: Optional[int] = None
        self.first_seen_t: Optional[float] = None
        self.last_seen_t: Optional[float] = None

        # Cacheado por `VehicleTracker.finalize()`, usado por `to_dict()`.
        # No se llama `is_stationary` para no chocar con el método homónimo.
        self.stationary: bool = False

    def add(
        self, frame_idx: int, t_video: float, bbox: List[float], confidence: float,
        class_id: int, class_name: str,
    ) -> None:
        """
        Agrega una observación al track y actualiza el voto de clase.

        La clase final del track (`self.class_name`) es la de **mayoría de
        votos** entre todas las observaciones, no la del primer frame: un
        vehículo puede clasificarse mal en un frame suelto (ej. un carro
        confundido con un bus en un ángulo raro) y el voto lo corrige.

        Parámetros:
            frame_idx (int): índice del frame de esta observación.
            t_video (float): instante en segundos, calculado con `TimeBase`.
            bbox (list[float]): `[x1, y1, x2, y2]` en píxeles.
            confidence (float): confianza de la detección en este frame.
            class_id (int): id de clase COCO de esta observación.
            class_name (str): nombre de clase de esta observación.

        Retorna:
            None
        """
        x1, y1, x2, y2 = bbox
        ground_point = [(x1 + x2) / 2.0, y2]

        self.observations.append({
            "frame_idx": frame_idx,
            "t_video": t_video,
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(confidence),
            "class_id": int(class_id),
            "ground_point": ground_point,
        })

        self._class_votes[(class_id, class_name)] += 1
        (self.class_id, self.class_name), _ = self._class_votes.most_common(1)[0]

        if self.first_frame is None:
            self.first_frame = frame_idx
            self.first_seen_t = t_video
        self.last_frame = frame_idx
        self.last_seen_t = t_video

    def length_frames(self) -> int:
        """
        Retorna:
            int: cantidad de frames que abarca el track (last_frame -
            first_frame + 1), o 0 si no tiene observaciones.
        """
        if not self.observations:
            return 0
        return self.last_frame - self.first_frame + 1

    def length_seconds(self) -> float:
        """
        Retorna:
            float: duración del track en segundos de video
            (`last_seen_t - first_seen_t`), o 0.0 si no tiene observaciones.
        """
        if not self.observations:
            return 0.0
        return self.last_seen_t - self.first_seen_t

    def ground_points(self) -> List[List[float]]:
        """
        Retorna:
            list[list[float]]: los puntos de contacto (bottom-center del
            bbox) de todas las observaciones, en orden temporal.
        """
        return [obs["ground_point"] for obs in self.observations]

    def total_displacement_px(self) -> float:
        """
        Retorna:
            float: distancia euclidiana entre el primer y el último punto de
            contacto, en píxeles. 0.0 si el track tiene menos de 2 observaciones.
        """
        points = self.ground_points()
        if len(points) < 2:
            return 0.0
        (x1, y1), (x2, y2) = points[0], points[-1]
        return float(np.hypot(x2 - x1, y2 - y1))

    def path_length_px(self) -> float:
        """
        Retorna:
            float: suma de las distancias entre puntos de contacto
            consecutivos, en píxeles. La razón `total_displacement_px() /
            path_length_px()` cercana a 1 indica trayectoria recta; muy por
            debajo de 1 indica ruido o un vehículo estacionado temblando
            (mucho recorrido acumulado sin avanzar realmente).
        """
        points = self.ground_points()
        total = 0.0
        for i in range(1, len(points)):
            (x1, y1), (x2, y2) = points[i - 1], points[i]
            total += float(np.hypot(x2 - x1, y2 - y1))
        return total

    def is_stationary(self, threshold_px: float) -> bool:
        """
        Parámetros:
            threshold_px (float): desplazamiento mínimo, en píxeles, para
                considerar que el vehículo se movió.

        Retorna:
            bool: True si el desplazamiento total es menor al umbral. Sirve
            para identificar vehículos parqueados, que reciben un ID estable
            y viven todo el video inflando el conteo de vehículos únicos.
        """
        return self.total_displacement_px() < threshold_px

    def borders_touched(self, frame_size: Tuple[int, int], margin_px: int) -> Tuple[Set[str], Set[str]]:
        """
        Determina qué bordes del frame toca el primer y el último bbox del
        track, dentro de un margen. Se usa para la métrica de fragmentación:
        un vehículo debería entrar y salir de la escena por un borde.

        Parámetros:
            frame_size (tuple[int, int]): `(ancho, alto)` del frame.
            margin_px (int): margen, en píxeles, para considerar que un bbox
                "toca" un borde.

        Retorna:
            tuple[set[str], set[str]]: `(bordes_primer_bbox, bordes_ultimo_bbox)`,
            cada uno un subconjunto de `{"left", "right", "top", "bottom"}`
            (vacío si no toca ningún borde).
        """
        if not self.observations:
            return set(), set()

        width, height = frame_size

        def _borders(bbox: List[float]) -> Set[str]:
            x1, y1, x2, y2 = bbox
            borders = set()
            if x1 <= margin_px:
                borders.add("left")
            if x2 >= width - margin_px:
                borders.add("right")
            if y1 <= margin_px:
                borders.add("top")
            if y2 >= height - margin_px:
                borders.add("bottom")
            return borders

        first_borders = _borders(self.observations[0]["bbox"])
        last_borders = _borders(self.observations[-1]["bbox"])
        return first_borders, last_borders

    def set_smoothed_ground_points(self, points: List[List[float]]) -> None:
        """
        Reemplaza los puntos de contacto de las observaciones por una
        versión suavizada (ver `smooth_ground_points`), preservando el resto
        de cada observación intacto.

        Parámetros:
            points (list[list[float]]): un punto por observación, mismo
                orden y longitud que `self.observations`.

        Retorna:
            None
        """
        for obs, point in zip(self.observations, points):
            obs["ground_point"] = list(point)

    def to_dict(self) -> Dict[str, Any]:
        """
        Retorna:
            dict: serialización completa del track para el JSON de
            trayectorias (ver `VehicleTracker.save_tracks`).
        """
        return {
            "track_id": self.track_id,
            "class_name": self.class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "first_seen_t": self.first_seen_t,
            "last_seen_t": self.last_seen_t,
            "length_frames": self.length_frames(),
            "total_displacement_px": self.total_displacement_px(),
            "is_stationary": self.stationary,
            "observations": self.observations,
        }


def smooth_ground_points(points: List[List[float]], window: int) -> List[List[float]]:
    """
    Aplica una media móvil a una lista de puntos de contacto.

    El bbox oscila ligeramente entre frames aunque el vehículo se mueva de
    forma uniforme; esa oscilación se amplifica al derivar para obtener
    velocidad más adelante. Suavizar acá evita ese ruido antes de que llegue
    a la etapa de velocidad.

    El primer y el último punto se preservan **sin alterar**: la ventana se
    trunca cerca de los extremos en vez de promediar con puntos que no
    existen (o de descartar los extremos, que son justo los que delimitan
    `first_seen_t`/`last_seen_t`).

    Parámetros:
        points (list[list[float]]): puntos `[x, y]` en orden temporal.
        window (int): tamaño de ventana, forzado a impar (se le suma 1 si
            viene par). `window <= 1` o menos de 3 puntos desactiva el
            suavizado (retorna una copia sin modificar).

    Retorna:
        list[list[float]]: puntos suavizados, misma longitud que la entrada.
    """
    if window <= 1 or len(points) < 3:
        return [list(p) for p in points]

    if window % 2 == 0:
        window += 1

    half = window // 2
    n = len(points)
    smoothed = []
    for i in range(n):
        if i == 0 or i == n - 1:
            smoothed.append(list(points[i]))
            continue
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        window_points = points[lo:hi]
        avg_x = sum(p[0] for p in window_points) / len(window_points)
        avg_y = sum(p[1] for p in window_points) / len(window_points)
        smoothed.append([avg_x, avg_y])
    return smoothed

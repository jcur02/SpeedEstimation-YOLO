"""
Conversión de coordenadas geográficas (latitud/longitud) a un plano métrico
local este/norte, y utilidades de precisión decimal.

Sobre distancias de decenas de metros —la escala de un cruce o un tramo de
calle— la aproximación de plano tangente centrado en un punto de origen es
excelente: el error introducido por ignorar la curvatura terrestre es
milimétrico. Esto evita resolver una proyección cartográfica completa (UTM,
etc.) solo para calibrar una cámara de tránsito.
"""
import math
from typing import Tuple


class LocalENU:
    """
    Plano métrico local (este/norte, en metros) tangente a la Tierra en un
    punto de origen `(lat0, lon0)`.

    Usa los coeficientes de la elipsoide WGS84, dependientes de la latitud
    del origen, en vez de un radio esférico constante: a la latitud de Costa
    Rica (~10°N) la diferencia frente a un radio esférico fijo ya vale varios
    metros por cada 100 km, y aunque acá no se opera a esa escala, no cuesta
    nada usar la fórmula correcta.
    """

    def __init__(self, lat0: float, lon0: float) -> None:
        """
        Precalcula los metros por grado de latitud/longitud en el origen.

        Parámetros:
            lat0 (float): latitud del origen, en grados decimales.
            lon0 (float): longitud del origen, en grados decimales.

        Retorna:
            None
        """
        self.lat0 = lat0
        self.lon0 = lon0

        phi = math.radians(lat0)
        self.m_per_deg_lat = (
            111132.92
            - 559.82 * math.cos(2 * phi)
            + 1.175 * math.cos(4 * phi)
            - 0.0023 * math.cos(6 * phi)
        )
        self.m_per_deg_lon = (
            111412.84 * math.cos(phi)
            - 93.5 * math.cos(3 * phi)
            + 0.118 * math.cos(5 * phi)
        )

    def to_meters(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convierte un punto lat/lon a metros este/norte respecto al origen.

        Parámetros:
            lat (float): latitud del punto, en grados decimales.
            lon (float): longitud del punto, en grados decimales.

        Retorna:
            tuple[float, float]: `(este, norte)` en metros. El origen mapea
            a `(0.0, 0.0)`.
        """
        este = (lon - self.lon0) * self.m_per_deg_lon
        norte = (lat - self.lat0) * self.m_per_deg_lat
        return este, norte

    def to_latlon(self, este: float, norte: float) -> Tuple[float, float]:
        """
        Inversa de `to_meters`: convierte un punto en metros este/norte de
        vuelta a latitud/longitud.

        Parámetros:
            este (float): coordenada este, en metros, respecto al origen.
            norte (float): coordenada norte, en metros, respecto al origen.

        Retorna:
            tuple[float, float]: `(lat, lon)` en grados decimales.
        """
        lat = self.lat0 + norte / self.m_per_deg_lat
        lon = self.lon0 + este / self.m_per_deg_lon
        return lat, lon

    @staticmethod
    def _count_decimals(coord_str: str) -> int:
        """
        Cuenta los decimales de una coordenada tal como fue tecleada.

        Parámetros:
            coord_str (str): la coordenada como texto (ej. "9.8574123").

        Retorna:
            int: cantidad de dígitos después del punto decimal (0 si no hay
            punto decimal).
        """
        text = coord_str.strip()
        if "." not in text:
            return 0
        frac = text.split(".", 1)[1]
        frac_digits = "".join(ch for ch in frac if ch.isdigit())
        return len(frac_digits)

    @staticmethod
    def validate_precision(lat_str: str, lon_str: str) -> Tuple[bool, str]:
        """
        Valida la precisión decimal de una coordenada tecleada o pegada por
        el usuario, **antes** de convertirla a float.

        Truncar decimales es un error invisible: el número sigue pareciendo
        una coordenada válida, pero el punto calibrado queda desplazado
        varios metros del real, y esa homografía va a producir velocidades
        sistemáticamente equivocadas sin que nada se vea roto en el camino.

        Parámetros:
            lat_str (str): latitud, como la escribió o pegó el usuario.
            lon_str (str): longitud, como la escribió o pegó el usuario.

        Retorna:
            tuple[bool, str]: `(ok, mensaje)`.
              - `ok = False` si alguna de las dos coordenadas tiene menos de
                5 decimales, o si el texto no es un número válido: el error
                posicional supera el metro y no debería continuarse con ese
                punto.
              - `ok = True` en los demás casos, con un mensaje que además
                advierte fuerte si hay exactamente 5 decimales (~1 m de
                error), o confirma que la precisión es correcta con 6 o más.
        """
        try:
            float(lat_str)
            float(lon_str)
        except (TypeError, ValueError):
            return False, (
                f"Coordenada inválida: '{lat_str}', '{lon_str}' no son números. "
                f"Ingresá latitud y longitud en grados decimales (ej. 9.8574123)."
            )

        decimals = min(LocalENU._count_decimals(lat_str), LocalENU._count_decimals(lon_str))
        note = (
            "En la séptima cifra decimal cada unidad vale aproximadamente un centímetro; "
            "Google Maps entrega esa precisión al hacer clic derecho sobre un punto y elegir "
            "las coordenadas."
        )

        if decimals < 5:
            return False, (
                f"Precisión insuficiente: {decimals} decimal(es). Con menos de 5 decimales el "
                f"error posicional supera el metro y arruina la calibración. {note}"
            )
        if decimals == 5:
            return True, (
                f"Advertencia: 5 decimales implican ~1 m de error posicional. Se puede continuar, "
                f"pero se recomienda más precisión si es posible. {note}"
            )
        return True, f"Precisión de {decimals} decimales: correcta. {note}"

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Distancia sobre la superficie terrestre entre dos puntos lat/lon,
        por la fórmula de Haversine (radio esférico medio de la Tierra).

        Se usa **solo para reportar distancias en la consola** (el chequeo
        de cordura de `scripts/calibrate.py`), nunca para la homografía: esa
        necesita coordenadas planas en dos dimensiones (`LocalENU.to_meters`),
        no distancias entre pares de puntos.

        Parámetros:
            lat1, lon1 (float): coordenadas del primer punto, en grados.
            lat2, lon2 (float): coordenadas del segundo punto, en grados.

        Retorna:
            float: distancia en metros.
        """
        r = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)
        a = (
            math.sin(d_phi / 2) ** 2
            + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

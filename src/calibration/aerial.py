"""
Obtención y georreferenciación de una imagen aérea satelital (opcional).

Este módulo es un front-end opcional cuyo único trabajo es dejar marcar
puntos de correspondencia con clics en vez de tipear latitud/longitud a
mano. Nada en el resto de la fase de calibración depende de él: sin llave de
MapTiler configurada, `scripts/calibrate.py` cae automáticamente al modo
manual, y una vez descargada, la imagen se guarda en disco junto a un
archivo lateral `.meta.json` — a partir de ahí el selector funciona sin red,
lo que importa porque el destino final es una Raspberry Pi sin conexión
permanente.
"""
import json
import logging
import math
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

_TILE_SIZE = 256


class AerialImage:
    """
    Imagen aérea georreferenciada: sabe a qué latitud/longitud corresponde
    cada uno de sus píxeles, vía la proyección Web Mercator (EPSG:3857) que
    usan los mapas de teselas (incluido MapTiler).
    """

    def __init__(
        self,
        image_path: str,
        center_lat: float,
        center_lon: float,
        zoom: int,
        width: int,
        height: int,
        scale_factor: int,
    ) -> None:
        """
        Parámetros:
            image_path (str): ruta del archivo de imagen en disco.
            center_lat, center_lon (float): coordenadas del centro de la
                imagen, en grados decimales.
            zoom (int): nivel de zoom de teselas con el que se descargó.
            width, height (int): dimensiones **reales en píxeles** de la
                imagen guardada (ya multiplicadas por `scale_factor` si se
                pidió a mayor resolución, ej. `@2x`).
            scale_factor (int): factor de escala de la descarga (2 para
                `@2x`), tal como aparece en las fórmulas de Web Mercator.

        Excepciones:
            AssertionError: si la autoverificación de ida y vuelta de la
                proyección falla (indicaría un error de implementación, no
                un error del usuario).

        Retorna:
            None
        """
        self.image_path = image_path
        self.center_lat = center_lat
        self.center_lon = center_lon
        self.zoom = zoom
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self._verify_projection()

    @classmethod
    def fetch(
        cls,
        lat: float,
        lon: float,
        zoom: int,
        width: int,
        height: int,
        api_key: Optional[str],
        out_path: str,
    ) -> Optional["AerialImage"]:
        """
        Descarga una imagen satelital estática de MapTiler centrada en
        `(lat, lon)`, y la guarda en disco junto a su archivo lateral
        `{out_path}.meta.json`.

        La ausencia de llave, un fallo de red o un error HTTP **no son
        errores fatales**: se informan por consola y se retorna `None`, para
        que el que llama pueda seguir en modo manual sin interrumpir el
        programa.

        Parámetros:
            lat, lon (float): centro de la imagen a descargar.
            zoom (int): nivel de zoom de teselas.
            width, height (int): tamaño solicitado, en píxeles, antes del
                factor `@2x` (el archivo resultante mide el doble).
            api_key (str | None): llave de MapTiler. Si es None o vacía, no
                se intenta la descarga.
            out_path (str): ruta donde guardar la imagen descargada.

        Retorna:
            AerialImage | None: la imagen descargada y georreferenciada, o
            None si no se pudo obtener (sin llave, red caída, error HTTP).
        """
        if not api_key:
            print(
                "  No hay llave de MapTiler configurada (variable de entorno MAPTILER_KEY); "
                "se continúa en modo manual, sin imagen aérea."
            )
            return None

        url = (
            f"https://api.maptiler.com/maps/satellite/static/"
            f"{lon},{lat},{zoom}/{width}x{height}@2x.png?key={api_key}"
        )
        try:
            with urllib.request.urlopen(url, timeout=20) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:
            print(f"✗ Error HTTP {exc.code} al descargar la imagen aérea: {exc.reason}. Se sigue en modo manual.")
            return None
        except urllib.error.URLError as exc:
            print(f"✗ Error de red al descargar la imagen aérea: {exc.reason}. Se sigue en modo manual.")
            return None
        except Exception as exc:  # descarga externa: cualquier fallo no debe tumbar el programa
            print(f"✗ Error inesperado al descargar la imagen aérea: {exc}. Se sigue en modo manual.")
            return None

        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(data)

        real_width, real_height = cls._read_image_size(out_path)
        scale_factor = 2  # la URL siempre pide "@2x"

        meta = {
            "center_lat": lat,
            "center_lon": lon,
            "zoom": zoom,
            "width": real_width,
            "height": real_height,
            "scale_factor": scale_factor,
        }
        with open(f"{out_path}.meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Imagen aérea descargada en %s (%dx%d, zoom %d)", out_path, real_width, real_height, zoom)
        print(f"  Imagen aérea guardada en {out_path} ({real_width}x{real_height}, zoom {zoom}).")
        return cls(out_path, lat, lon, zoom, real_width, real_height, scale_factor)

    @staticmethod
    def _read_image_size(image_path: str) -> Tuple[int, int]:
        """
        Lee el tamaño real en píxeles de una imagen ya guardada en disco.

        Parámetros:
            image_path (str): ruta de la imagen.

        Retorna:
            tuple[int, int]: `(ancho, alto)` en píxeles.

        Excepciones:
            ValueError: si el archivo no se pudo leer como imagen.
        """
        import cv2  # import local: solo esta función lo necesita

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"No se pudo leer la imagen aérea recién descargada: '{image_path}'.")
        height, width = img.shape[:2]
        return width, height

    @classmethod
    def load(cls, image_path: str) -> "AerialImage":
        """
        Reconstruye una `AerialImage` a partir de un archivo ya descargado y
        su archivo lateral `.meta.json`.

        Parámetros:
            image_path (str): ruta de la imagen aérea.

        Retorna:
            AerialImage: la imagen reconstruida.

        Excepciones:
            FileNotFoundError: si la imagen o su `.meta.json` no existen. Sin
                el lateral, la imagen no está georreferenciada y no sirve
                para marcar puntos.
            ValueError: si el `.meta.json` está corrupto o incompleto.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"No se encontró la imagen aérea: '{image_path}'.")

        meta_path = f"{image_path}.meta.json"
        if not os.path.isfile(meta_path):
            raise FileNotFoundError(
                f"La imagen '{image_path}' no tiene su archivo lateral '{meta_path}': "
                f"no está georreferenciada y no se puede usar para marcar puntos. "
                f"Volvé a descargarla con --fetch-aerial."
            )

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"El archivo lateral '{meta_path}' está corrupto o no es JSON válido: {exc}") from exc

        required = ("center_lat", "center_lon", "zoom", "width", "height", "scale_factor")
        missing = [k for k in required if k not in meta]
        if missing:
            raise ValueError(f"El archivo lateral '{meta_path}' no tiene los campos requeridos: {missing}")

        return cls(
            image_path, meta["center_lat"], meta["center_lon"], meta["zoom"],
            meta["width"], meta["height"], meta["scale_factor"],
        )

    def meters_per_pixel(self) -> float:
        """
        Calcula la resolución de la imagen aérea en metros por píxel, en su
        centro (la resolución de Web Mercator varía con la latitud).

        Si supera 0.20 m/px, advierte que a esa resolución no se distingue
        el borde de una marca vial y conviene subir el zoom antes de marcar
        puntos.

        Retorna:
            float: metros por píxel.
        """
        mpp = 156543.03392 * math.cos(math.radians(self.center_lat)) / (2 ** self.zoom) / self.scale_factor
        if mpp > 0.20:
            msg = (
                f"La resolución de la imagen aérea es de {mpp:.3f} m/px: a esa resolución no se "
                f"distingue el borde de una marca vial. Convendría subir --zoom antes de marcar puntos."
            )
            logger.warning(msg)
            print(f"  ⚠ {msg}")
        return mpp

    def _world_px_at_zoom(self) -> float:
        """Tamaño en píxeles del mapa del mundo completo, a este zoom y escala."""
        return _TILE_SIZE * (2 ** self.zoom) * self.scale_factor

    def _latlon_to_world_px(self, lat: float, lon: float) -> Tuple[float, float]:
        """Proyecta lat/lon a coordenadas de píxel del mapa del mundo completo (Web Mercator)."""
        # Proyección Web Mercator: x en [0,1], y en [0,1] sobre el mundo completo
        x = (lon + 180.0) / 360.0
        y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0
        # Escala en píxeles al zoom dado (tamaño de tesela 256)
        world_px = self._world_px_at_zoom()
        return x * world_px, y * world_px

    def latlon_to_pixel(self, lat: float, lon: float) -> Tuple[float, float]:
        """
        Convierte una coordenada lat/lon al píxel correspondiente de esta
        imagen concreta (desplazamiento respecto a su centro).

        Parámetros:
            lat, lon (float): coordenadas a proyectar.

        Retorna:
            tuple[float, float]: `(px, py)` en la imagen, con origen en la
            esquina superior izquierda (convención OpenCV).
        """
        wx, wy = self._latlon_to_world_px(lat, lon)
        cx, cy = self._latlon_to_world_px(self.center_lat, self.center_lon)
        px = self.width / 2.0 + (wx - cx)
        py = self.height / 2.0 + (wy - cy)
        return px, py

    def pixel_to_latlon(self, px: float, py: float) -> Tuple[float, float]:
        """
        Inversa de `latlon_to_pixel`: convierte un píxel de esta imagen a
        latitud/longitud.

        Parámetros:
            px, py (float): coordenadas de píxel en la imagen.

        Retorna:
            tuple[float, float]: `(lat, lon)` en grados decimales.
        """
        cx, cy = self._latlon_to_world_px(self.center_lat, self.center_lon)
        wx = cx + (px - self.width / 2.0)
        wy = cy + (py - self.height / 2.0)
        world_px = self._world_px_at_zoom()
        x = wx / world_px
        y = wy / world_px
        lon = x * 360.0 - 180.0
        lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * y))))
        return lat, lon

    def _verify_projection(self) -> None:
        """
        Autoverificación interna: proyectar el centro de la imagen a píxel y
        de vuelta a lat/lon debe devolver el mismo punto dentro de una
        tolerancia de un píxel. Se corre una sola vez, al construir el
        objeto, para detectar cualquier error de implementación temprano.

        Excepciones:
            AssertionError: si la ida y vuelta no coincide dentro de
                tolerancia.

        Retorna:
            None
        """
        px, py = self.latlon_to_pixel(self.center_lat, self.center_lon)
        expected_px, expected_py = self.width / 2.0, self.height / 2.0
        if abs(px - expected_px) > 1.0 or abs(py - expected_py) > 1.0:
            raise AssertionError(
                f"Autoverificación de la proyección Web Mercator falló: el centro proyectó a "
                f"({px:.2f}, {py:.2f}), se esperaba ({expected_px:.2f}, {expected_py:.2f})."
            )

        lat2, lon2 = self.pixel_to_latlon(expected_px, expected_py)
        if abs(lat2 - self.center_lat) > 1e-6 or abs(lon2 - self.center_lon) > 1e-6:
            raise AssertionError(
                f"Autoverificación de la proyección Web Mercator falló en la inversa: "
                f"({lat2}, {lon2}) != ({self.center_lat}, {self.center_lon})."
            )

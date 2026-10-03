"""
Selector gráfico (Tkinter) de puntos de correspondencia píxel <-> lat/lon
para la calibración por homografía.

Reemplaza el marcado interactivo basado en OpenCV (que pedía latitud y
longitud por separado en la consola): agrega un único campo de texto donde
pegar las coordenadas tal como las entrega Google Maps, edición de puntos
ya marcados, y un panel de calidad que se recalcula en vivo mientras se
marca. No cambia nada del cálculo: reutiliza `LocalENU` y
`HomographyCalibration` exactamente como están.
"""
import csv
import logging
import math
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageTk

from src.calibration.aerial import AerialImage
from src.calibration.geo import LocalENU
from src.calibration.homography import HomographyCalibration

logger = logging.getLogger(__name__)

# Acepta "9.8574123, -83.9112345", con o sin espacios, con coma o solo
# espacio como separador, y con el símbolo de grado opcional en cualquiera
# de los dos valores (lo que entrega Google Maps al copiar coordenadas).
_COORD_RE = re.compile(
    r"^\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*[,\s]+\s*([+-]?\d+(?:\.\d+)?)\s*°?\s*$"
)


def parse_latlon(texto: str) -> Optional[Tuple[float, float, str, str]]:
    """
    Analiza un texto de coordenadas pegado o tecleado por el usuario.

    Acepta formatos como "9.8574123, -83.9112345", "9.8574123,-83.9112345",
    "9.8574123 -83.9112345" o "9.8574123°, -83.9112345°", con espacios
    sobrantes en cualquier posición.

    Parámetros:
        texto (str): el contenido del campo de coordenadas.

    Retorna:
        tuple[float, float, str, str] | None: `(lat, lon, lat_str, lon_str)`
        si el texto produce exactamente dos números, o `None` si no. Las dos
        últimas posiciones son las subcadenas numéricas originales (sin el
        símbolo de grado ni espacios, pero **antes** de convertir a float):
        son las que hay que pasarle a `LocalENU.validate_precision()`, nunca
        `str(lat)`/`str(lon)` reconstruidas desde el float, porque la
        conversión a float no garantiza reproducir la cantidad exacta de
        decimales que el usuario pegó.
    """
    if not texto:
        return None
    match = _COORD_RE.fullmatch(texto.strip())
    if match is None:
        return None
    lat_str, lon_str = match.group(1), match.group(2)
    try:
        lat, lon = float(lat_str), float(lon_str)
    except ValueError:
        return None
    return lat, lon, lat_str, lon_str


class _ZoomPanView:
    """
    Estado de zoom/desplazamiento de un lienzo (cámara o aérea) y las
    conversiones entre coordenadas del lienzo y de la imagen fuente.

    Se instancia una por lienzo: la cámara y la vista aérea tienen cada una
    su propio zoom y desplazamiento independientes.
    """

    MIN_SCALE = 0.25
    MAX_SCALE = 8.0

    def __init__(self) -> None:
        self.scale: float = 1.0
        self.offset_x: float = 0.0
        self.offset_y: float = 0.0

    def canvas_to_image(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convierte un punto en coordenadas del lienzo a la imagen fuente a resolución completa."""
        return (self.offset_x + cx / self.scale, self.offset_y + cy / self.scale)

    def image_to_canvas(self, ix: float, iy: float) -> Tuple[float, float]:
        """Convierte un punto de la imagen fuente a coordenadas del lienzo, con el zoom/offset actual."""
        return ((ix - self.offset_x) * self.scale, (iy - self.offset_y) * self.scale)

    def _verify_roundtrip(self, cx: float, cy: float) -> None:
        """
        Comprueba que canvas->imagen->canvas devuelve el mismo punto dentro
        de un píxel. Si esto falla, la calibración quedaría desplazada de
        forma silenciosa: todo corre, nada falla, y la homografía es
        incorrecta.
        """
        ix, iy = self.canvas_to_image(cx, cy)
        bx, by = self.image_to_canvas(ix, iy)
        assert abs(bx - cx) < 1.0 and abs(by - cy) < 1.0, (
            "Conversión canvas<->imagen inconsistente tras un cambio de zoom: "
            f"canvas({cx:.2f},{cy:.2f}) -> imagen({ix:.2f},{iy:.2f}) -> canvas({bx:.2f},{by:.2f})"
        )

    def zoom_at(self, cx: float, cy: float, factor: float) -> None:
        """Aplica un factor de zoom (>1 acerca, <1 aleja) centrado en un punto del lienzo."""
        ix, iy = self.canvas_to_image(cx, cy)
        new_scale = min(max(self.scale * factor, self.MIN_SCALE), self.MAX_SCALE)
        if new_scale == self.scale:
            return
        self.scale = new_scale
        self.offset_x = ix - cx / self.scale
        self.offset_y = iy - cy / self.scale
        self._verify_roundtrip(cx, cy)

    def pan(self, dcx: float, dcy: float) -> None:
        """Desplaza la vista por un delta en coordenadas del lienzo (arrastre del mouse)."""
        self.offset_x -= dcx / self.scale
        self.offset_y -= dcy / self.scale

    def fit_to_window(self, img_w: int, img_h: int, canvas_w: int, canvas_h: int) -> None:
        """Ajusta la escala y el desplazamiento para mostrar la imagen completa dentro del lienzo."""
        if img_w <= 0 or img_h <= 0 or canvas_w <= 0 or canvas_h <= 0:
            return
        scale = min(canvas_w / img_w, canvas_h / img_h)
        self.scale = min(max(scale, self.MIN_SCALE), self.MAX_SCALE)
        self.offset_x = (img_w - canvas_w / self.scale) / 2.0
        self.offset_y = (img_h - canvas_h / self.scale) / 2.0
        self._verify_roundtrip(canvas_w / 2.0, canvas_h / 2.0)

    def render(self, source_bgr: np.ndarray, canvas_w: int, canvas_h: int) -> np.ndarray:
        """
        Compone la región visible de `source_bgr` (según el zoom/offset
        actual) sobre un lienzo de `canvas_w` x `canvas_h`, en gris oscuro
        donde la imagen no llega a cubrir.
        """
        out = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)
        h, w = source_bgr.shape[:2]

        x0, y0 = self.offset_x, self.offset_y
        x1, y1 = x0 + canvas_w / self.scale, y0 + canvas_h / self.scale
        sx0, sy0 = max(0.0, x0), max(0.0, y0)
        sx1, sy1 = min(float(w), x1), min(float(h), y1)
        if sx1 <= sx0 or sy1 <= sy0:
            return out

        crop = source_bgr[int(round(sy0)):int(round(sy1)), int(round(sx0)):int(round(sx1))]
        if crop.size == 0:
            return out

        dst_x0 = int(round((sx0 - x0) * self.scale))
        dst_y0 = int(round((sy0 - y0) * self.scale))
        dst_w = max(1, int(round(crop.shape[1] * self.scale)))
        dst_h = max(1, int(round(crop.shape[0] * self.scale)))
        resized = cv2.resize(crop, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

        dst_x1 = min(canvas_w, dst_x0 + dst_w)
        dst_y1 = min(canvas_h, dst_y0 + dst_h)
        if dst_x1 <= dst_x0 or dst_y1 <= dst_y0:
            return out
        out[dst_y0:dst_y1, dst_x0:dst_x1] = resized[0:dst_y1 - dst_y0, 0:dst_x1 - dst_x0]
        return out


def _draw_magnifier(
    canvas_img: np.ndarray, source_xy: Tuple[float, float], source_bgr: np.ndarray,
    size: int = 140, zoom: int = 3,
) -> None:
    """
    Dibuja, en la esquina superior derecha de `canvas_img`, una lupa con la
    zona bajo el cursor (en coordenadas de `source_bgr`) ampliada. Es lo que
    permite marcar con precisión de píxel.
    """
    sx, sy = source_xy
    half = max(size // (2 * zoom), 4)
    x0, y0 = int(sx - half), int(sy - half)
    x1, y1 = x0 + 2 * half, y0 + 2 * half
    h, w = source_bgr.shape[:2]
    x0c, y0c, x1c, y1c = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
    if x1c <= x0c or y1c <= y0c:
        return

    crop = source_bgr[y0c:y1c, x0c:x1c]
    mag = cv2.resize(crop, (size, size), interpolation=cv2.INTER_NEAREST)
    cv2.line(mag, (size // 2 - 8, size // 2), (size // 2 + 8, size // 2), (0, 255, 0), 1)
    cv2.line(mag, (size // 2, size // 2 - 8), (size // 2, size // 2 + 8), (0, 255, 0), 1)
    cv2.rectangle(mag, (0, 0), (size - 1, size - 1), (255, 255, 255), 1)

    ch, cw = canvas_img.shape[:2]
    px, py = cw - size - 10, 10
    if px < 0 or py < 0 or px + size > cw or py + size > ch:
        return
    canvas_img[py:py + size, px:px + size] = mag


def _draw_dashed_line(
    img: np.ndarray, p1: Tuple[float, float], p2: Tuple[float, float],
    color: Tuple[int, int, int], dash: int = 8, gap: int = 6, thickness: int = 1,
) -> None:
    """Dibuja un segmento punteado entre `p1` y `p2` (cv2 no tiene un primitivo nativo para esto)."""
    x1, y1 = p1
    x2, y2 = p2
    length = math.hypot(x2 - x1, y2 - y1)
    if length < 1e-6:
        return
    dx, dy = (x2 - x1) / length, (y2 - y1) / length
    step = dash + gap
    n_segments = int(length // step) + 1
    for i in range(n_segments):
        start = i * step
        if start >= length:
            break
        end = min(start + dash, length)
        sx, sy = x1 + dx * start, y1 + dy * start
        ex, ey = x1 + dx * end, y1 + dy * end
        cv2.line(img, (int(sx), int(sy)), (int(ex), int(ey)), color, thickness, cv2.LINE_AA)


class CalibrationPicker:
    """
    Ventana única de marcado de puntos de correspondencia para la
    calibración por homografía.

    Un clic sobre el frame de la cámara crea un punto pendiente; sus
    coordenadas se pegan o teclean en un único campo de texto (formato de
    Google Maps) y se confirman con Enter. Si hay una imagen aérea
    disponible, una pestaña adicional permite obtenerlas con un clic en vez
    de escribirlas. Un panel lateral muestra la lista de puntos y recalcula
    en vivo la calidad de la calibración (reproyección, validación cruzada,
    cobertura) a medida que se marca.
    """

    _LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _SIDEBAR_WIDTH = 320
    _MAGNIFIER_SIZE = 140
    _MAGNIFIER_ZOOM = 3
    _ZOOM_STEP = 1.2

    _OK_COLOR = "#1a7f37"
    _WARN_COLOR = "#b36b00"
    _REJECT_COLOR = "#c0392b"
    _NEUTRAL_COLOR = "#333333"

    def __init__(
        self,
        frame: np.ndarray,
        config: dict,
        existing_points: Optional[List[Dict[str, Any]]] = None,
        aerial: Optional[AerialImage] = None,
    ) -> None:
        """
        Parámetros:
            frame (numpy.ndarray): imagen BGR de OpenCV, el frame de
                referencia de la cámara sobre el que se marcan los puntos.
            config (dict): configuración completa del proyecto (se lee
                `calibration.points_csv`, `calibration.thresholds`,
                `calibration.fit_method` y `calibration.ransac_threshold_m`).
            existing_points (list[dict] | None): puntos ya marcados, para
                continuar o editar una sesión previa (típicamente cargados
                de un CSV por `scripts/calibrate.py`).
            aerial (AerialImage | None): imagen aérea georreferenciada. Si
                viene, habilita la pestaña aérea; si es `None`, el selector
                muestra únicamente el lienzo de la cámara.

        Retorna:
            None
        """
        self.frame = frame
        self.frame_h, self.frame_w = frame.shape[:2]
        self.config = config

        self.aerial = aerial
        self.aerial_img: Optional[np.ndarray] = None
        if aerial is not None:
            img = cv2.imread(aerial.image_path)
            if img is None:
                logger.warning("No se pudo leer la imagen aérea '%s'; se omite la pestaña aérea.", aerial.image_path)
            else:
                self.aerial_img = img

        self.points: List[Dict[str, Any]] = [dict(p) for p in (existing_points or [])]
        self.pending: Optional[Dict[str, Any]] = None
        self.editing_index: Optional[int] = None
        self.worst_label: Optional[str] = None
        self.hull_points: Optional[List[Tuple[float, float]]] = None
        self._aerial_click_marker: Optional[Tuple[float, float]] = None
        self._cam_mouse: Optional[Tuple[int, int]] = None
        self._aer_mouse: Optional[Tuple[int, int]] = None
        self._pan_last: Optional[Tuple[int, int]] = None
        self._cam_fitted: bool = False
        self._aer_fitted: bool = False
        self._dirty: bool = False
        self.result: Optional[List[Dict[str, Any]]] = None

        calib_cfg = config.get("calibration", {})
        self.points_csv_path: str = calib_cfg.get("points_csv", "config/points.csv")
        self.draft_path: str = self._draft_path_for(self.points_csv_path)
        self.thresholds: Dict[str, Any] = calib_cfg.get("thresholds", {})
        self.fit_method: str = calib_cfg.get("fit_method", "least_squares")
        self.ransac_threshold_m: float = calib_cfg.get("ransac_threshold_m", 0.5)

        # Tipos poblados en run()/_build_ui(); declarados acá para que el
        # resto de los métodos los puedan referenciar con seguridad de tipos.
        self.root: tk.Tk
        self.cam_canvas: tk.Canvas
        self.aer_canvas: Optional[tk.Canvas] = None
        self.cam_view = _ZoomPanView()
        self.aer_view = _ZoomPanView()

    @staticmethod
    def _draft_path_for(points_csv_path: str) -> str:
        """Deriva la ruta del borrador a partir de la ruta del CSV principal (ej. points.csv -> points.draft.csv)."""
        base, _ext = os.path.splitext(points_csv_path)
        return f"{base}.draft.csv"

    # ------------------------------------------------------------------
    # Punto de entrada
    # ------------------------------------------------------------------

    def run(self) -> Optional[List[Dict[str, Any]]]:
        """
        Abre la ventana y bloquea hasta que el usuario guarda o cancela.

        Retorna:
            list[dict] | None: la lista de puntos guardados, con la forma
            `{"label", "pixel": [px, py], "latlon": [lat, lon], "note"}`, o
            `None` si el usuario cerró la ventana sin guardar.
        """
        self._build_ui()
        # El ajuste inicial de zoom NO se hace acá: recién construida, la
        # ventana todavía no fue dibujada por el gestor de ventanas y
        # winfo_width()/height() del lienzo devuelven un tamaño mínimo
        # ficticio (no el real), lo que produciría un zoom degenerado. En
        # cambio, se engancha al primer evento <Configure> real de cada
        # lienzo (ver `_on_canvas_configure`), que es cuando Tkinter ya
        # conoce su tamaño verdadero.
        self.root.update_idletasks()

        self._maybe_recover_draft()
        self._refresh_tree()
        self._recalculate_quality()
        self._render_canvas("cam")
        if self.aer_canvas is not None:
            self._render_canvas("aer")

        self.root.mainloop()
        return self.result

    # ------------------------------------------------------------------
    # Conversión de coordenadas del lienzo de la cámara (nombres explícitos
    # pedidos por el diseño: nunca mezclar coordenadas de lienzo y de frame)
    # ------------------------------------------------------------------

    def canvas_to_frame(self, cx: float, cy: float) -> Tuple[float, float]:
        """Convierte un punto del lienzo de la cámara a píxeles del frame original a resolución completa."""
        return self.cam_view.canvas_to_image(cx, cy)

    def frame_to_canvas(self, fx: float, fy: float) -> Tuple[float, float]:
        """Convierte un píxel del frame original a coordenadas del lienzo de la cámara, con el zoom actual."""
        return self.cam_view.image_to_canvas(fx, fy)

    # ------------------------------------------------------------------
    # Construcción de la interfaz
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self.root = tk.Tk()
        self.root.title("Calibración — selector de puntos")
        self.root.geometry("1280x800")
        self.root.minsize(900, 600)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.bind("<Escape>", self._on_close)
        self.root.bind("<Control-z>", self._on_undo)
        self.root.bind("<Control-Z>", self._on_undo)
        self.root.bind("<Control-s>", lambda e: self._save_and_exit())
        self.root.bind("<Control-S>", lambda e: self._save_and_exit())
        self.root.bind("<Delete>", lambda e: self._on_delete_selected())

        self.status_var = tk.StringVar(value="Hacé clic sobre un punto del pavimento y pegá sus coordenadas.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken", bd=1)
        status_bar.pack(side="bottom", fill="x")

        main = tk.Frame(self.root)
        main.pack(fill="both", expand=True)

        image_container = tk.Frame(main)
        image_container.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(main, width=self._SIDEBAR_WIDTH)
        sidebar.pack(side="right", fill="y")
        sidebar.pack_propagate(False)

        self._build_image_area(image_container)
        self._build_sidebar(sidebar)

    def _build_image_area(self, parent: tk.Frame) -> None:
        if self.aerial_img is not None:
            notebook = ttk.Notebook(parent)
            notebook.pack(fill="both", expand=True)
            cam_tab = tk.Frame(notebook)
            aer_tab = tk.Frame(notebook)
            notebook.add(cam_tab, text="Cámara")
            notebook.add(aer_tab, text="Aérea")

            self.cam_canvas = tk.Canvas(cam_tab, bg="#222222", highlightthickness=0)
            self.cam_canvas.pack(fill="both", expand=True)
            self.aer_canvas = tk.Canvas(aer_tab, bg="#222222", highlightthickness=0)
            self.aer_canvas.pack(fill="both", expand=True)
            self._bind_canvas_events(self.aer_canvas, "aer")
        else:
            self.cam_canvas = tk.Canvas(parent, bg="#222222", highlightthickness=0)
            self.cam_canvas.pack(fill="both", expand=True)
            self.aer_canvas = None

        self._bind_canvas_events(self.cam_canvas, "cam")

    def _bind_canvas_events(self, canvas: tk.Canvas, which: str) -> None:
        canvas.bind("<Button-1>", lambda e, w=which: self._on_canvas_click(w, e))
        canvas.bind("<MouseWheel>", lambda e, w=which: self._on_wheel(w, e))
        canvas.bind("<Button-4>", lambda e, w=which: self._on_wheel_linux(w, 1))
        canvas.bind("<Button-5>", lambda e, w=which: self._on_wheel_linux(w, -1))
        canvas.bind("<ButtonPress-3>", lambda e: self._start_pan(e))
        canvas.bind("<B3-Motion>", lambda e, w=which: self._do_pan(w, e))
        canvas.bind("<ButtonPress-2>", lambda e: self._start_pan(e))
        canvas.bind("<B2-Motion>", lambda e, w=which: self._do_pan(w, e))
        canvas.bind("<Motion>", lambda e, w=which: self._on_motion(w, e))
        canvas.bind("<Enter>", lambda e, c=canvas: c.focus_set())
        canvas.bind("<Key-f>", lambda e, w=which: self._fit(w))
        canvas.bind("<Key-F>", lambda e, w=which: self._fit(w))
        canvas.bind("<Configure>", lambda e, w=which: self._on_canvas_configure(w))

    def _build_sidebar(self, parent: tk.Frame) -> None:
        pad = {"padx": 8, "pady": 2}

        tk.Label(parent, text="Punto pendiente:").pack(anchor="w", **pad)
        self.lbl_pending_val = tk.Label(parent, text="(ninguno)", font=("TkDefaultFont", 10, "bold"))
        self.lbl_pending_val.pack(anchor="w", **pad)

        tk.Label(parent, text="Píxel:").pack(anchor="w", **pad)
        self.lbl_pixel_val = tk.Label(parent, text="—")
        self.lbl_pixel_val.pack(anchor="w", **pad)

        tk.Label(parent, text="Coordenadas (lat, lon):").pack(anchor="w", **pad)
        coord_frame = tk.Frame(parent)
        coord_frame.pack(fill="x", **pad)
        self.coord_var = tk.StringVar()
        self.coord_entry = tk.Entry(coord_frame, textvariable=self.coord_var)
        self.coord_entry.pack(side="left", fill="x", expand=True)
        self.coord_entry.bind("<Return>", self._on_confirm)
        tk.Button(coord_frame, text="Pegar", command=self._on_paste).pack(side="left", padx=(4, 0))

        self.lbl_validation = tk.Label(
            parent, text="", wraplength=self._SIDEBAR_WIDTH - 16, justify="left", fg=self._NEUTRAL_COLOR,
        )
        self.lbl_validation.pack(anchor="w", fill="x", **pad)

        tk.Label(parent, text="Nota (opcional):").pack(anchor="w", **pad)
        self.note_var = tk.StringVar()
        self.note_entry = tk.Entry(parent, textvariable=self.note_var)
        self.note_entry.pack(fill="x", **pad)
        self.note_entry.bind("<Return>", self._on_confirm)

        self.confirm_button = tk.Button(parent, text="Agregar punto", command=self._on_confirm)
        self.confirm_button.pack(fill="x", **pad)

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        tk.Label(parent, text="Puntos", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", **pad)

        tree_frame = tk.Frame(parent)
        tree_frame.pack(fill="both", expand=True, **pad)
        columns = ("label", "pixel", "latlon", "residual")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=8, selectmode="browse")
        for col, text, width in (
            ("label", "Etq.", 34), ("pixel", "Píxel", 90), ("latlon", "Lat, Lon", 116), ("residual", "Resid.", 56),
        ):
            self.tree.heading(col, text=text)
            self.tree.column(col, width=width, anchor="w")
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll.pack(side="right", fill="y")
        self.tree.tag_configure("worst", background="#ffd6d6")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        btn_frame = tk.Frame(parent)
        btn_frame.pack(fill="x", **pad)
        tk.Button(btn_frame, text="Editar", command=self._on_edit).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Borrar", command=self._on_delete_selected).pack(side="left", expand=True, fill="x")
        tk.Button(btn_frame, text="Todo", command=self._on_delete_all).pack(side="left", expand=True, fill="x")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        tk.Label(parent, text="Calidad", font=("TkDefaultFont", 10, "bold")).pack(anchor="w", **pad)

        self.lbl_points_val = self._quality_row(parent, "Puntos:")
        self.lbl_reproj_val = self._quality_row(parent, "Reproyección:")
        self.lbl_loo_val = self._quality_row(parent, "Val. cruzada:")
        self.lbl_coverage_val = self._quality_row(parent, "Cobertura:")

        ttk.Separator(parent, orient="horizontal").pack(fill="x", pady=6)
        tk.Button(
            parent, text="Guardar y salir", command=self._save_and_exit, font=("TkDefaultFont", 10, "bold"),
        ).pack(fill="x", **pad)

    def _quality_row(self, parent: tk.Frame, label_text: str) -> tk.Label:
        row = tk.Frame(parent)
        row.pack(fill="x", padx=8, pady=1)
        tk.Label(row, text=label_text, width=14, anchor="w").pack(side="left")
        val = tk.Label(row, text="—", anchor="w", fg=self._NEUTRAL_COLOR)
        val.pack(side="left", fill="x", expand=True)
        return val

    # ------------------------------------------------------------------
    # Eventos del lienzo: clic, zoom, desplazamiento, ajuste a ventana
    # ------------------------------------------------------------------

    def _on_canvas_click(self, which: str, event: "tk.Event") -> None:
        if which == "cam":
            self._on_cam_click(event)
        else:
            self._on_aer_click(event)

    def _on_cam_click(self, event: "tk.Event") -> None:
        fx, fy = self.canvas_to_frame(event.x, event.y)
        fx = min(max(fx, 0.0), self.frame_w - 1)
        fy = min(max(fy, 0.0), self.frame_h - 1)
        # Un clic nuevo reubica el punto pendiente en vez de crear otro,
        # tanto si viene de un clic anterior sin confirmar como si viene de
        # "Editar" (editing_index ya está fijado en ese caso).
        self.pending = {"pixel": [fx, fy]}
        self._update_pending_labels()
        self._render_canvas("cam")
        self._set_status("Punto marcado. Pegá o escribí sus coordenadas y presioná Enter.")
        self.coord_entry.focus_set()

    def _on_aer_click(self, event: "tk.Event") -> None:
        if self.aerial is None or self.aerial_img is None:
            return
        if self.pending is None:
            self._set_status("Primero hacé clic sobre la imagen de la cámara para marcar el punto pendiente.")
            return

        ax, ay = self.aer_view.canvas_to_image(event.x, event.y)
        ah, aw = self.aerial_img.shape[:2]
        ax = min(max(ax, 0.0), aw - 1)
        ay = min(max(ay, 0.0), ah - 1)

        lat, lon = self.aerial.pixel_to_latlon(ax, ay)
        self.coord_var.set(f"{lat:.7f}, {lon:.7f}")
        self._aerial_click_marker = (ax, ay)
        self._render_canvas("aer")
        self._set_status("Coordenadas tomadas de la imagen aérea. Revisá y confirmá con Enter.")
        self.coord_entry.focus_set()
        self.coord_entry.icursor("end")

    def _on_wheel(self, which: str, event: "tk.Event") -> None:
        factor = self._ZOOM_STEP if event.delta > 0 else (1.0 / self._ZOOM_STEP)
        self._zoom(which, event.x, event.y, factor)

    def _on_wheel_linux(self, which: str, direction: int) -> None:
        # En Linux la rueda llega como <Button-4>/<Button-5>, sin información
        # de posición útil más allá de la última conocida por <Motion>.
        mouse = self._cam_mouse if which == "cam" else self._aer_mouse
        if mouse is None:
            return
        factor = self._ZOOM_STEP if direction > 0 else (1.0 / self._ZOOM_STEP)
        self._zoom(which, mouse[0], mouse[1], factor)

    def _zoom(self, which: str, cx: float, cy: float, factor: float) -> None:
        view = self.cam_view if which == "cam" else self.aer_view
        view.zoom_at(cx, cy, factor)
        self._render_canvas(which)

    def _start_pan(self, event: "tk.Event") -> None:
        self._pan_last = (event.x, event.y)

    def _do_pan(self, which: str, event: "tk.Event") -> None:
        if self._pan_last is None:
            self._pan_last = (event.x, event.y)
            return
        dx, dy = event.x - self._pan_last[0], event.y - self._pan_last[1]
        view = self.cam_view if which == "cam" else self.aer_view
        view.pan(dx, dy)
        self._pan_last = (event.x, event.y)
        self._render_canvas(which)

    def _on_motion(self, which: str, event: "tk.Event") -> None:
        if which == "cam":
            self._cam_mouse = (event.x, event.y)
        else:
            self._aer_mouse = (event.x, event.y)
        self._render_canvas(which)

    def _on_canvas_configure(self, which: str) -> None:
        """
        Maneja <Configure> del lienzo. La primera vez que se dispara, el
        tamaño ya es el real (a diferencia de justo después de construir la
        ventana), así que ahí se hace el ajuste inicial a la ventana; las
        veces siguientes (redimensionar la ventana) solo se vuelve a
        renderizar con el zoom/desplazamiento que ya tenía el usuario.
        """
        fitted_attr = "_cam_fitted" if which == "cam" else "_aer_fitted"
        if not getattr(self, fitted_attr):
            setattr(self, fitted_attr, True)
            self._fit(which)
        else:
            self._render_canvas(which)

    def _fit(self, which: str) -> None:
        if which == "cam":
            self.cam_view.fit_to_window(
                self.frame_w, self.frame_h, self.cam_canvas.winfo_width(), self.cam_canvas.winfo_height(),
            )
        else:
            if self.aerial_img is None:
                return
            ah, aw = self.aerial_img.shape[:2]
            self.aer_view.fit_to_window(aw, ah, self.aer_canvas.winfo_width(), self.aer_canvas.winfo_height())
        self._render_canvas(which)

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def _render_canvas(self, which: str) -> None:
        canvas = self.cam_canvas if which == "cam" else self.aer_canvas
        source = self.frame if which == "cam" else self.aerial_img
        view = self.cam_view if which == "cam" else self.aer_view
        if canvas is None or source is None:
            return

        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        canvas_img = view.render(source, cw, ch)

        if which == "cam":
            self._draw_hull_on(canvas_img, view)
            self._draw_points_on(canvas_img, view)
            self._draw_pending_on(canvas_img, view)
        else:
            self._draw_aer_marker_on(canvas_img, view)

        mouse = self._cam_mouse if which == "cam" else self._aer_mouse
        if mouse is not None:
            source_xy = view.canvas_to_image(*mouse)
            _draw_magnifier(canvas_img, source_xy, source, size=self._MAGNIFIER_SIZE, zoom=self._MAGNIFIER_ZOOM)

        rgb = cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo)
        # Tkinter NO conserva referencias a las imágenes que muestra: si no
        # guardamos `photo` como atributo de la instancia, el recolector de
        # basura la descarta en cuanto termina esta función y el lienzo
        # queda en blanco. Es el fallo más común de Tkinter con imágenes.
        if which == "cam":
            self._cam_photo = photo
        else:
            self._aer_photo = photo

    def _draw_points_on(self, canvas_img: np.ndarray, view: _ZoomPanView) -> None:
        selected = self._selected_label()
        for p in self.points:
            cx, cy = view.image_to_canvas(*p["pixel"])
            if p["label"] == self.worst_label:
                color = (0, 0, 255)
            elif p["label"] == selected:
                color = (255, 220, 0)
            else:
                color = (0, 140, 255)
            cv2.circle(canvas_img, (int(cx), int(cy)), 6, color, -1)
            cv2.putText(
                canvas_img, p["label"], (int(cx) + 8, int(cy) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
            )

    def _draw_pending_on(self, canvas_img: np.ndarray, view: _ZoomPanView) -> None:
        if self.pending is None:
            return
        cx, cy = view.image_to_canvas(*self.pending["pixel"])
        if self.editing_index is not None:
            label = self.points[self.editing_index]["label"]
        else:
            label = self._LABELS[len(self.points)]
        cv2.circle(canvas_img, (int(cx), int(cy)), 8, (0, 255, 255), 2)
        cv2.putText(
            canvas_img, label, (int(cx) + 10, int(cy) - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA,
        )

    def _draw_aer_marker_on(self, canvas_img: np.ndarray, view: _ZoomPanView) -> None:
        if self._aerial_click_marker is None:
            return
        cx, cy = view.image_to_canvas(*self._aerial_click_marker)
        cv2.circle(canvas_img, (int(cx), int(cy)), 8, (0, 255, 255), 2)

    def _draw_hull_on(self, canvas_img: np.ndarray, view: _ZoomPanView) -> None:
        if not self.hull_points:
            return
        pts = [view.image_to_canvas(x, y) for (x, y) in self.hull_points]
        n = len(pts)
        for i in range(n):
            _draw_dashed_line(canvas_img, pts[i], pts[(i + 1) % n], color=(255, 255, 0))

    def _selected_label(self) -> Optional[str]:
        sel = self.tree.selection()
        return sel[0] if sel else None

    # ------------------------------------------------------------------
    # Confirmación del punto pendiente (campo de coordenadas)
    # ------------------------------------------------------------------

    def _on_confirm(self, event: Optional["tk.Event"] = None) -> None:
        if self.pending is None:
            self._set_status("Primero hacé clic sobre la imagen para marcar un punto.")
            return

        parsed = parse_latlon(self.coord_var.get())
        if parsed is None:
            self._set_validation(
                "Formato no reconocido. Ejemplo válido: 9.8574123, -83.9112345", "error",
            )
            return
        lat, lon, lat_str, lon_str = parsed

        # Heurística de coordenadas invertidas: en Costa Rica la longitud
        # ronda -84 y la latitud ronda 9, así que si el primer valor no
        # puede ser una latitud válida pero sí una longitud, y el segundo
        # valor sí podría ser una latitud válida, es casi seguro que el
        # usuario pegó lon,lat en vez de lat,lon.
        if abs(lat) > 90.0 and abs(lon) <= 90.0:
            swap = messagebox.askyesno(
                "¿Coordenadas invertidas?",
                f"La latitud ({lat}) está fuera de rango [-90, 90], pero el segundo valor ({lon}) sí "
                "podría ser una latitud válida. ¿Intercambiar latitud y longitud?",
                parent=self.root,
            )
            if swap:
                lat, lon = lon, lat
                lat_str, lon_str = lon_str, lat_str
            else:
                self._set_validation(f"Latitud fuera de rango [-90, 90]: {lat}", "error")
                return

        if not (-90.0 <= lat <= 90.0):
            self._set_validation(f"Latitud fuera de rango [-90, 90]: {lat}", "error")
            return
        if not (-180.0 <= lon <= 180.0):
            self._set_validation(f"Longitud fuera de rango [-180, 180]: {lon}", "error")
            return

        ok, msg = LocalENU.validate_precision(lat_str, lon_str)
        self._set_validation(msg, "ok" if (ok and "Advertencia" not in msg) else ("warn" if ok else "error"))
        if not ok:
            return

        note = self.note_var.get().strip()
        if self.editing_index is not None:
            idx = self.editing_index
            self.points[idx]["pixel"] = list(self.pending["pixel"])
            self.points[idx]["latlon"] = [lat, lon]
            self.points[idx]["note"] = note
            self.editing_index = None
            self.confirm_button.config(text="Agregar punto")
        else:
            label = self._LABELS[len(self.points)]
            self.points.append({
                "label": label, "pixel": list(self.pending["pixel"]),
                "latlon": [lat, lon], "note": note, "residual_m": None,
            })

        self.pending = None
        self._aerial_click_marker = None
        self.coord_var.set("")
        self.note_var.set("")
        self._dirty = True
        self._write_csv(self.draft_path)
        self._update_pending_labels()
        self._refresh_tree()
        self._recalculate_quality()
        self._render_canvas("cam")
        if self.aer_canvas is not None:
            self._render_canvas("aer")
        self._set_status("Punto agregado. Hacé clic sobre la imagen para marcar el siguiente.")
        self.cam_canvas.focus_set()

    def _on_paste(self) -> None:
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            self._set_status("El portapapeles está vacío o no es accesible.")
            return
        self.coord_var.set(text.strip())
        self.coord_entry.focus_set()
        self.coord_entry.icursor("end")

    # ------------------------------------------------------------------
    # Lista de puntos: selección, edición, borrado, deshacer
    # ------------------------------------------------------------------

    def _on_tree_select(self, event: Optional["tk.Event"] = None) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        self._center_on(*self.points[idx]["pixel"])
        self._render_canvas("cam")

    def _on_edit(self) -> None:
        sel = self.tree.selection()
        if not sel:
            self._set_status("Seleccioná un punto de la lista para editarlo.")
            return
        idx = self.tree.index(sel[0])
        p = self.points[idx]
        self.editing_index = idx
        self.pending = {"pixel": list(p["pixel"])}
        self.coord_var.set(f"{p['latlon'][0]}, {p['latlon'][1]}")
        self.note_var.set(p.get("note", ""))
        self.confirm_button.config(text="Guardar cambios")
        self._update_pending_labels()
        self._center_on(*p["pixel"])
        self._render_canvas("cam")
        self._set_status(f"Editando el punto {p['label']}: hacé clic para reubicarlo, o editá sus coordenadas.")
        self.coord_entry.focus_set()

    def _on_delete_selected(self) -> None:
        sel = self.tree.selection()
        if not sel:
            return
        idx = self.tree.index(sel[0])
        removed = self.points.pop(idx)
        for i, p in enumerate(self.points):
            p["label"] = self._LABELS[i]
        if self.editing_index == idx:
            self._cancel_pending()
        elif self.editing_index is not None and self.editing_index > idx:
            self.editing_index -= 1
        self._dirty = True
        self._write_csv(self.draft_path)
        self._refresh_tree()
        self._recalculate_quality()
        self._render_canvas("cam")
        self._set_status(f"Borrado: punto {removed['label']}.")

    def _on_delete_all(self) -> None:
        if not self.points:
            return
        if not messagebox.askyesno("Borrar todo", "¿Eliminar todos los puntos marcados?", parent=self.root):
            return
        self.points.clear()
        self._cancel_pending()
        self._dirty = True
        self._write_csv(self.draft_path)
        self._refresh_tree()
        self._recalculate_quality()
        self._render_canvas("cam")
        self._set_status("Se borraron todos los puntos.")

    def _on_undo(self, event: Optional["tk.Event"] = None) -> None:
        if not self.points:
            return
        removed = self.points.pop()
        if self.editing_index is not None and self.editing_index >= len(self.points):
            self._cancel_pending()
        self._dirty = True
        self._write_csv(self.draft_path)
        self._refresh_tree()
        self._recalculate_quality()
        self._render_canvas("cam")
        self._set_status(f"Deshecho: punto {removed['label']}.")

    def _cancel_pending(self) -> None:
        self.editing_index = None
        self.pending = None
        self.confirm_button.config(text="Agregar punto")
        self._update_pending_labels()

    def _center_on(self, x: float, y: float) -> None:
        cw, ch = self.cam_canvas.winfo_width(), self.cam_canvas.winfo_height()
        self.cam_view.offset_x = x - (cw / 2.0) / self.cam_view.scale
        self.cam_view.offset_y = y - (ch / 2.0) / self.cam_view.scale

    def _refresh_tree(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for p in self.points:
            residual = p.get("residual_m")
            residual_txt = f"{residual:.2f} m" if residual is not None else "—"
            tags = ("worst",) if p["label"] == self.worst_label else ()
            self.tree.insert(
                "", "end", iid=p["label"], tags=tags,
                values=(
                    p["label"],
                    f"({p['pixel'][0]:.0f},{p['pixel'][1]:.0f})",
                    f"{p['latlon'][0]:.5f}, {p['latlon'][1]:.5f}",
                    residual_txt,
                ),
            )

    def _update_pending_labels(self) -> None:
        if self.pending is None:
            self.lbl_pending_val.config(text="(ninguno)")
            self.lbl_pixel_val.config(text="—")
            return
        if self.editing_index is not None:
            label = self.points[self.editing_index]["label"]
        else:
            label = self._LABELS[len(self.points)]
        x, y = self.pending["pixel"]
        self.lbl_pending_val.config(text=label)
        self.lbl_pixel_val.config(text=f"({x:.0f}, {y:.0f})")

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _set_validation(self, msg: str, level: str) -> None:
        color = {"ok": self._OK_COLOR, "warn": self._WARN_COLOR, "error": self._REJECT_COLOR}.get(
            level, self._NEUTRAL_COLOR,
        )
        self.lbl_validation.config(text=msg, fg=color)

    # ------------------------------------------------------------------
    # Panel de calidad en vivo
    # ------------------------------------------------------------------

    @staticmethod
    def _status_color(value: float, ok_th: float, warn_th: float) -> str:
        if value <= ok_th:
            return CalibrationPicker._OK_COLOR
        if value <= warn_th:
            return CalibrationPicker._WARN_COLOR
        return CalibrationPicker._REJECT_COLOR

    def _recalculate_quality(self) -> None:
        n = len(self.points)
        self.worst_label = None

        if n >= 4:
            hull = cv2.convexHull(np.array([p["pixel"] for p in self.points], dtype=np.float32))
            self.hull_points = [(float(pt[0][0]), float(pt[0][1])) for pt in hull]
        else:
            self.hull_points = None

        if n < 4:
            for p in self.points:
                p["residual_m"] = None
            self.lbl_points_val.config(text=str(n))
            self.lbl_reproj_val.config(text="—", fg=self._NEUTRAL_COLOR)
            self.lbl_loo_val.config(text="—", fg=self._NEUTRAL_COLOR)
            self.lbl_coverage_val.config(text="—", fg=self._NEUTRAL_COLOR)
            return

        labels = [p["label"] for p in self.points]
        origin_lat, origin_lon = self.points[0]["latlon"]
        enu = LocalENU(origin_lat, origin_lon)
        world_points = [list(enu.to_meters(*p["latlon"])) for p in self.points]
        pixel_points = [p["pixel"] for p in self.points]

        try:
            calib = HomographyCalibration(pixel_points, world_points, (self.frame_w, self.frame_h))
            calib.fit(method=self.fit_method, ransac_threshold_m=self.ransac_threshold_m)
        except ValueError:
            for p in self.points:
                p["residual_m"] = None
            self.lbl_points_val.config(text=str(n))
            self.lbl_reproj_val.config(text="configuración degenerada", fg=self._REJECT_COLOR)
            self.lbl_loo_val.config(text="—", fg=self._NEUTRAL_COLOR)
            self.lbl_coverage_val.config(text="—", fg=self._NEUTRAL_COLOR)
            self._refresh_tree()
            return

        for p, r in zip(self.points, calib.residuals()):
            p["residual_m"] = float(r)

        quality = calib.quality(labels=labels)
        loo = calib.leave_one_out()
        self.worst_label = quality.get("worst_point_label")

        self.lbl_points_val.config(text=str(n))

        reproj = quality["reprojection_mean_m"]
        reproj_color = self._status_color(
            reproj, self.thresholds.get("reprojection_ok_m", 0.15), self.thresholds.get("reprojection_warn_m", 0.30),
        )
        note = "  (con 4 puntos, cero por construcción)" if n == 4 else ""
        self.lbl_reproj_val.config(text=f"{reproj:.2f} m{note}", fg=reproj_color)

        if loo.get("available"):
            loo_color = self._status_color(
                loo["mean_m"], self.thresholds.get("loo_ok_m", 0.30), self.thresholds.get("loo_warn_m", 0.60),
            )
            self.lbl_loo_val.config(text=f"{loo['mean_m']:.2f} m", fg=loo_color)
        else:
            missing = max(0, 5 - n)
            self.lbl_loo_val.config(text=f"faltan {missing} punto(s)", fg=self._NEUTRAL_COLOR)

        coverage = quality["hull_area_ratio"]
        cov_color = self._OK_COLOR if coverage >= self.thresholds.get("min_hull_ratio", 0.05) else self._REJECT_COLOR
        self.lbl_coverage_val.config(text=f"{coverage * 100:.1f}%", fg=cov_color)

        self._refresh_tree()

    # ------------------------------------------------------------------
    # Guardado, sesiones y borrador
    # ------------------------------------------------------------------

    def _write_csv(self, path: str) -> None:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "px", "py", "lat", "lon", "note"])
            for p in self.points:
                writer.writerow([
                    p["label"], p["pixel"][0], p["pixel"][1], p["latlon"][0], p["latlon"][1], p.get("note", ""),
                ])

    @staticmethod
    def _load_csv(path: str) -> List[Dict[str, Any]]:
        """
        Carga puntos desde un CSV con columnas `label,px,py,lat,lon,note`.

        Más permisivo que `scripts/calibrate.py:load_points_csv` (no valida
        precisión decimal ni corta el programa ante un error): se usa solo
        para recuperar el borrador que el propio selector escribió, no para
        cargar un CSV externo del usuario.
        """
        points: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            rows = [row for row in f if row.strip() and not row.strip().startswith("#")]
        reader = csv.DictReader(rows)
        for row in reader:
            label = (row.get("label") or "").strip()
            if not label:
                continue
            points.append({
                "label": label,
                "pixel": [float(row["px"]), float(row["py"])],
                "latlon": [float(row["lat"]), float(row["lon"])],
                "note": (row.get("note") or "").strip(),
            })
        return points

    def _maybe_recover_draft(self) -> None:
        if not os.path.isfile(self.draft_path):
            return
        draft_mtime = os.path.getmtime(self.draft_path)
        main_mtime = os.path.getmtime(self.points_csv_path) if os.path.isfile(self.points_csv_path) else -1.0
        if draft_mtime <= main_mtime:
            return

        recover = messagebox.askyesno(
            "Borrador encontrado",
            f"Se encontró un borrador más reciente que '{self.points_csv_path}' en:\n{self.draft_path}\n\n"
            "¿Recuperarlo? Esto reemplaza los puntos cargados actualmente.",
            parent=self.root,
        )
        if not recover:
            return
        try:
            self.points = self._load_csv(self.draft_path)
        except (OSError, ValueError, KeyError) as exc:
            messagebox.showerror("Error al recuperar", f"No se pudo leer el borrador: {exc}", parent=self.root)
            return
        self._dirty = True

    def _save_and_exit(self) -> None:
        if len(self.points) < 4:
            messagebox.showwarning(
                "Puntos insuficientes", "Se requieren al menos 4 puntos para guardar.", parent=self.root,
            )
            return
        self._write_csv(self.points_csv_path)
        if os.path.isfile(self.draft_path):
            try:
                os.remove(self.draft_path)
            except OSError:
                pass
        self.result = [dict(p) for p in self.points]
        self._dirty = False
        self.root.destroy()

    def _on_close(self, event: Optional["tk.Event"] = None) -> None:
        if self._dirty and self.points:
            if not messagebox.askyesno(
                "¿Salir sin guardar?", "Hay puntos sin guardar. ¿Salir de todas formas?", parent=self.root,
            ):
                return
        self.result = None
        self.root.destroy()

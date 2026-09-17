# SpeedEstimation-YOLO

Sistema de estimación de velocidad vehicular para cámaras de videovigilancia sobre
hardware embebido de bajo costo (Raspberry Pi 4 / NVIDIA Jetson Nano). Esta primera
fase implementa un **benchmark de detección vehicular** con distintas variantes YOLO
para determinar cuál ofrece el mejor balance entre precisión, velocidad (FPS) y
consumo de recursos, como base para el Trabajo Final de Graduación en Ingeniería en
Computadores del TEC (Costa Rica).

## Estructura del proyecto

```
SpeedEstimation-YOLO/
├── config/
│   └── config.yaml
├── data/
│   └── videos/               ← coloca aquí tus videos .mp4
├── models/                   ← carpeta vacía, los pesos se descargan automáticamente
├── results/
│   ├── benchmarks/           ← CSV y JSON con resultados
│   ├── plots/                ← gráficas comparativas PNG
│   └── frames/                ← frames anotados de muestra (por modelo)
├── src/
│   ├── detection/
│   │   ├── detector.py
│   │   └── benchmark.py
│   └── utils/
│       └── metrics.py
├── scripts/
│   ├── check_gpu.py
│   └── run_benchmark.py
├── notebooks/
│   └── 01_analisis_benchmark.ipynb
├── requirements.txt
└── README.md
```

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv venv
source venv/bin/activate        # Linux/Mac
# o: venv\Scripts\activate      # Windows

# 2. PASO CRÍTICO: instalar PyTorch (por separado del resto de dependencias)
#    Verifica primero tu versión de CUDA con: nvidia-smi

#    Con GPU NVIDIA, CUDA 12.4 (recomendado):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
#    CUDA 12.1: --index-url https://download.pytorch.org/whl/cu121
#    CUDA 11.8: --index-url https://download.pytorch.org/whl/cu118

#    Sin GPU (solo CPU):
pip install torch torchvision torchaudio

# 3. Instalar el resto de dependencias
pip install -r requirements.txt

# 4. Verificar la instalación
python scripts/check_gpu.py
```

## Uso rápido

```bash
# Coloca un video de prueba en data/videos/, luego:
python scripts/run_benchmark.py --device cpu
```

Si no se indica `--video`, el script usa automáticamente el primer `.mp4` encontrado
en `data/videos/`.

## Argumentos de `run_benchmark.py`

| Argumento      | Tipo | Descripción                                                                 | Default              |
|----------------|------|-------------------------------------------------------------------------------|-----------------------|
| `--video`      | str  | Ruta al video de prueba. Si se omite, se busca el primer `.mp4` en `data/videos/`. | `None` (auto)         |
| `--config`     | str  | Ruta al archivo de configuración.                                             | `config/config.yaml`  |
| `--models`     | str  | Lista de `model_id` separados por coma a evaluar (ej. `yolov8n,yolo11n`).      | `None` (todos)         |
| `--max-frames` | int  | Sobreescribe el número máximo de frames a procesar por video.                | valor de `config.yaml` |
| `--device`     | str  | Dispositivo de inferencia: `cpu` o `cuda`.                                    | valor de `config.yaml` |

## Formato de los resultados

Cada corrida genera, con un timestamp común `YYYYMMDD_HHMMSS`:

- **`results/benchmarks/frame_data_{timestamp}.csv`** — una fila por frame por
  modelo, con `inference_ms`, `fps`, conteos por clase de vehículo, confianza
  promedio, y uso de CPU/RAM en ese instante.
- **`results/benchmarks/summary_{timestamp}.json`** — resumen estadístico por
  modelo (FPS medio/std/p5/p95, RAM media/pico, CPU media, etc.), junto con
  `video_path`, `system_info` y la configuración completa usada, para trazabilidad.
- **`results/plots/benchmark_{timestamp}.png`** — 4 subplots comparativos (FPS,
  tiempo de inferencia, RAM, vehículos/frame) por modelo.
- **`results/plots/fps_vs_ram_{timestamp}.png`** — scatter de balance FPS vs. RAM.
- **`results/frames/{model_id}/`** — frames anotados de muestra por modelo.
- **`results/benchmark.log`** — log completo de la corrida (nivel INFO).

El notebook `notebooks/01_analisis_benchmark.ipynb` carga automáticamente los
resultados más recientes y genera además `results/benchmarks/tabla_informe.csv`,
la tabla lista para el informe académico.

## Variantes YOLO evaluadas

| model_id  | Descripción                                              |
|-----------|-----------------------------------------------------------|
| `yolov8n` | YOLOv8 Nano — más liviano, mayor velocidad                |
| `yolov8s` | YOLOv8 Small — balance entre velocidad y precisión         |
| `yolov8m` | YOLOv8 Medium — mayor precisión, más pesado                |
| `yolo11n` | YOLO11 Nano — versión más reciente, muy liviana            |
| `yolo11s` | YOLO11 Small — versión más reciente, liviana                |

Los pesos (`.pt`) se descargan automáticamente en `models/` la primera vez que se
usa cada modelo.

## Vista previa de detección en video

Herramienta de **validación cualitativa**, independiente del benchmark: permite ver el
video corriendo con las bounding boxes dibujadas en tiempo real (o exportarlo anotado a
un archivo), para revisar a simple vista que un modelo detecta bien — que no pierde
vehículos entre frames, que las cajas no parpadean, que no confunde clases y que el
umbral de confianza elegido tiene sentido. Los FPS que muestra en pantalla son **solo
informativos**: no son los del benchmark, porque incluyen el costo de dibujar las cajas,
el HUD y renderizar la ventana (o codificar el video de salida).

No escribe nunca en `results/benchmarks/`, `results/plots/` ni `results/frames/`: toda
su salida va a `results/preview/`.

### Argumentos de `preview_detection.py`

| Argumento      | Tipo  | Default                              | Descripción                                                        |
|----------------|-------|---------------------------------------|---------------------------------------------------------------------|
| `--video`      | str   | `None` (auto)                         | Ruta al video. Si se omite, toma el primer `.mp4` de `data/videos/`. |
| `--model`      | str   | `preview.default_model` del config    | `model_id` a usar (debe existir en `detection.models`).             |
| `--compare`    | str   | `None`                                 | Segundo `model_id` para vista lado a lado.                          |
| `--export`     | flag  | `False`                                | Exporta a archivo en vez de abrir ventana.                          |
| `--output`     | str   | `None` (auto)                          | Ruta del archivo de salida en modo export.                          |
| `--codec`      | str   | `preview.export_codec` del config      | Codec del `VideoWriter`.                                            |
| `--conf`       | float | `detection.confidence_threshold` del config | Umbral de confianza, para probar valores en caliente.          |
| `--device`     | str   | `benchmark.device` del config          | `cpu` o `cuda`.                                                      |
| `--start-sec`  | float | `0`                                     | Segundo del video donde empezar.                                    |
| `--max-frames` | int   | `None` (video completo)                | Limita cuántos frames procesar.                                     |
| `--fps`        | int   | `preview.playback_fps` del config      | Limita la tasa de reproducción.                                     |
| `--no-hud`     | flag  | `False`                                 | Arranca con el HUD oculto.                                           |

### Controles de teclado (modo ventana)

| Tecla       | Acción                                                              |
|-------------|-----------------------------------------------------------------------|
| `espacio`   | Pausa / reanuda                                                       |
| `q` / `ESC` | Salir                                                                  |
| `s`         | Guarda el frame anotado actual en `results/preview/snapshots/`        |
| `n` / `→`   | Estando en pausa, avanza exactamente un frame                        |
| `h`         | Muestra/oculta el HUD (o los rótulos, en modo comparación)            |
| `b`         | Muestra/oculta las bounding boxes                                    |
| `+` / `-`   | Sube o baja `playback_fps` en pasos de 5 (mínimo 0 = sin límite)       |
| `r`         | Reinicia el video desde el frame 0                                    |

### Ejemplos de uso

```bash
# Ver el video con el modelo por defecto
python scripts/preview_detection.py

# Un modelo específico, empezando en el segundo 30
python scripts/preview_detection.py --model yolo11n --start-sec 30

# Probar un umbral de confianza más bajo para ver si aparecen vehículos lejanos
python scripts/preview_detection.py --conf 0.25

# Comparar dos modelos lado a lado
python scripts/preview_detection.py --model yolov8n --compare yolov8m

# Exportar un video anotado (para el informe o la defensa)
python scripts/preview_detection.py --model yolov8n --export --max-frames 600

# En Raspberry Pi por SSH (sin display): cae solo a modo exportación
python scripts/preview_detection.py --device cpu --export
```

**Codecs:** si el `.mp4` sale corrupto o de 0 bytes, usa `--codec XVID` con un nombre de
salida terminado en `.avi` (`--output resultado.avi`).

**Raspberry Pi 4:** `cv2.imshow` requiere una instalación de `opencv-python` con soporte
de GUI; en instalaciones headless (por SSH, sin X) conviene usar siempre `--export`. El
script detecta automáticamente la ausencia de display y cae a modo exportación por su
cuenta, pero pasar `--export` explícitamente evita la advertencia y dejará el archivo
donde se indique con `--output`.

## Región de interés (ROI)

Las cámaras suelen encuadrar más que la vía de interés: carril contrario, vehículos
parqueados, un parqueo lateral, una calle de fondo. Todo eso contamina el conteo de
vehículos y, más adelante, contaminaría el seguimiento y produciría velocidades sin
sentido. La ROI delimita esa zona válida con un polígono trazado una sola vez sobre la
escena, que se guarda en disco y se reutiliza en todas las corridas de ese sitio.

**Aclaración conceptual importante: la ROI no acelera la inferencia.** YOLO redimensiona
su entrada a un tamaño fijo (`imgsz`), así que el costo de cómputo no depende del área de
la escena. El beneficio de la ROI es **reducir ruido y falsos positivos**, no ganar FPS.
Cualquier ganancia de rendimiento vendría de bajar `imgsz`, saltar frames o usar un modelo
más liviano — no de recortar la escena.

La única excepción es el modo `crop`, que sí puede **mejorar la precisión** (no la
velocidad): al recortar y reescalar la zona de interés, los vehículos lejanos ocupan más
píxeles de la entrada de la red y se detectan mejor. Vale la pena medir ese efecto.

### Los tres modos

| Modo | Qué hace | Cuándo usarlo |
|---|---|---|
| `filter` (default) | Corre la inferencia sobre el frame completo y descarta después las detecciones fuera del polígono. | Uso general. No afecta la calidad de detección, solo el conteo. |
| `crop` | Recorta el frame al rectángulo envolvente de la ROI antes de correr la inferencia. | Cuando los vehículos lejanos dentro de la ROI son pequeños y se pierden — el recorte los agranda relativamente. |
| `mask` | Ennegrece todo lo exterior al polígono antes de la inferencia. | Solo para comparar empíricamente contra los otros dos; suele **empeorar** las detecciones cerca del borde porque genera una imagen fuera de la distribución de entrenamiento de la red. |

### Supuesto de diseño: cámara fija

La ROI asume que la cámara no se mueve entre la calibración y el uso. Si se reorienta o
se reposiciona, hay que volver a marcarla. El campo `reference_frame_md5` del JSON (más
un thumbnail de referencia guardado aparte) sirve para detectar esto: `select_roi.py
--show` compara el frame actual contra el de referencia y avisa si difieren demasiado.

### Tabla de argumentos de `select_roi.py`

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--video` | str (múltiple) | auto | Uno o varios videos (rutas o comodines). Default: todos los `.mp4` de `data/videos/`. |
| `--output` | str | `roi.file` del config | Dónde guardar el JSON. |
| `--no-heatmap` | flag | False | Salta el mapa de calor y marca sobre un frame limpio. |
| `--frame-sec` | float | None | Segundo del que se toma el frame de fondo. Default: el frame medio del primer video. |
| `--sample-mode` | str | `uniform` | `uniform`, `middle` o `range`. |
| `--samples` | int | `roi.heatmap.samples` | Cuántos frames muestrear para el mapa de calor. |
| `--model` | str | `roi.heatmap.model` | Modelo usado para acumular el mapa de calor. |
| `--device` | str | del config | `cpu` o `cuda`. |
| `--with-homography` | flag | False | Pide además 4 puntos sobre el asfalto, para la calibración futura. |
| `--show` | flag | False | Solo muestra la ROI ya guardada, sin editarla. |

### Controles del marcado interactivo

| Entrada | Acción |
|---|---|
| Clic izquierdo | Agrega un vértice |
| Clic derecho / `z` | Elimina el último vértice |
| `c` | Borra todos los vértices y reinicia |
| `t` | Alterna el mapa de calor encendido/apagado |
| `Enter` | Cierra el polígono y confirma (requiere ≥ 3 vértices, sin autointersección) |
| `q` / `ESC` | Sale sin guardar, pidiendo confirmación en consola |

### Formato de `roi.json`

Las coordenadas del polígono se guardan **normalizadas en [0, 1]**, no en píxeles, para
que la misma ROI sirva si más adelante se procesa el video a otra resolución (por ejemplo
720p en la Raspberry Pi en vez de 1080p en la máquina de desarrollo):

```json
{
  "version": 1,
  "created_at": "2026-09-10T14:32:07",
  "source_videos": ["data/videos/sitio_a_01.mp4"],
  "frame_size": [1920, 1080],
  "reference_frame_md5": "a3f1...",
  "roi_polygon": [[0.133, 0.352], [0.164, 0.352], [0.208, 0.930], [0.078, 0.930]],
  "homography_points": {
    "image_points": [],
    "world_points": null,
    "notes": "Pendiente: completar world_points en la fase de calibración."
  }
}
```

### Ejemplos de uso

```bash
# Marcar la ROI con mapa de calor sobre todos los videos del sitio
python scripts/select_roi.py --video "data/videos/sitio_a_*.mp4"

# Videos cortos: subir el muestreo y acumular varios
python scripts/select_roi.py --video data/videos/*.mp4 --samples 600

# Marcar también los 4 puntos para la homografía futura
python scripts/select_roi.py --with-homography

# Revisar la ROI guardada y verificar que la cámara no se movió
python scripts/select_roi.py --show

# Usarla
python scripts/preview_detection.py --roi
python scripts/run_benchmark.py --roi
```

Igual que `select_roi.py`, es una herramienta inherentemente interactiva: si `DISPLAY`
no está disponible (sesión SSH sin X, como en la Raspberry Pi), el script lo detecta e
imprime un error explicando que la ROI debe marcarse en una máquina con interfaz gráfica.
El `roi.json` resultante puede copiarse luego a la Raspberry Pi sin volver a marcarlo.

## Seguimiento multi-objeto

Sin identidad persistente no existe la velocidad: la velocidad es el desplazamiento del
**mismo** vehículo entre dos instantes. Hasta acá, el sistema detecta de forma
independiente en cada frame y no tiene forma de saber que el vehículo del frame 100 es
el mismo del frame 130. Esta fase agrega seguimiento multi-objeto (ByteTrack y BoTSORT,
ambos ya integrados en Ultralytics — no se reimplementa ningún algoritmo de seguimiento
a mano), produce trayectorias con identificadores estables, y compara empíricamente
ambos trackers sobre los 5 modelos. El producto principal es un **archivo de
trayectorias** (`results/tracks/*.json`) que será la entrada directa de la etapa de
estimación de velocidad.

### La base de tiempo: el error más peligroso de esta etapa

El tiempo asociado a un frame se calcula siempre como:

```python
t_video = frame_idx / source_fps
```

**Nunca** con `time.time()`, `time.perf_counter()` ni ningún reloj del sistema.

Razón: en la Raspberry Pi 4 el procesamiento correrá más lento que tiempo real — por
ejemplo, 8 FPS procesando un video grabado a 30 FPS. Si el tiempo se midiera con el
reloj de pared, todas las velocidades saldrían subestimadas por un factor cercano a 4.
Es un error **silencioso**: los números resultantes parecen plausibles, simplemente
están mal. Por eso existe la clase `TimeBase` (`src/tracking/track.py`) como punto único
y explícito de cálculo del tiempo. La etapa de velocidad debe leer el campo `t_video` de
cada observación del JSON de trayectorias — nunca recalcularlo con un reloj propio.

### Advertencia metodológica sobre las métricas de calidad

Las métricas estándar de seguimiento multi-objeto (MOTA, HOTA, IDF1, cambios de
identidad reales) requieren **anotaciones manuales de verdad de referencia**, que este
trabajo no posee. Las métricas de `src/tracking/quality.py` son **indicadores
indirectos**, calculados sin verdad de referencia. Son útiles para comparar trackers
entre sí sobre el mismo video, que es exactamente lo que se necesita acá, pero **no son
MOTA ni HOTA y no deben reportarse como tales** en el informe.

### ByteTrack vs. BoTSORT

| | ByteTrack | BoTSORT |
|---|---|---|
| Asociación | IoU + movimiento, dos pasadas por umbral de confianza | Igual que ByteTrack, más compensación de movimiento de cámara (GMC) |
| Re-identificación por apariencia | No | Opcional (`with_reid`, apagado por defecto) |
| Costo | Más liviano | Más pesado — especialmente con `with_reid` activado |
| Cámara fija (este proyecto) | Candidato natural para la RPi4 | El GMC no aporta mucho con cámara fija y cuesta tiempo |
| Cuándo preferirlo | Por defecto, y siempre que el presupuesto de FPS sea ajustado | Escenas con cruces de trayectorias frecuentes o oclusiones muy largas, si sobra presupuesto de cómputo |

La comparación empírica de `scripts/run_tracking_benchmark.py` existe para no tener que
adivinar cuál conviene en el hardware real.

### Argumentos nuevos de `preview_detection.py`

| Argumento | Default | Descripción |
|---|---|---|
| `--track` | `False` | Activa el modo seguimiento. |
| `--tracker` | `tracking.default_tracker` del config | `bytetrack` o `botsort`. |
| `--compare-tracker` | `None` | Segundo tracker (mismo modelo) para vista lado a lado. Requiere `--track`. |
| `--trail` | `tracking.visualization.trail_length` del config | Longitud de la estela en frames. |
| `--no-trail` | `False` | Arranca con las estelas ocultas. |

### Argumentos de `run_tracking_benchmark.py`

| Argumento | Tipo | Default | Descripción |
|---|---|---|---|
| `--video` | str | auto | Video a procesar. |
| `--models` | str | todos | Lista de `model_id` separados por coma. |
| `--trackers` | str | `bytetrack,botsort` | Trackers a comparar. |
| `--conf` | float | del config | Umbral de confianza. |
| `--device` | str | del config | `cpu` o `cuda`. |
| `--max-frames` | int | `None` | Limita frames por corrida, para pruebas rápidas. |
| `--fps` | float | `None` | Fuerza los FPS si el video los reporta mal. |
| `--no-save-tracks` | flag | `False` | No escribe los JSON de trayectorias. |

### Controles de teclado (modo seguimiento)

Además de los controles del modo de solo detección (espacio, `q`/ESC, `s`, `n`, `h`,
`b`, `+`/`-`, `r`):

| Tecla | Acción |
|---|---|
| `t` | Muestra/oculta las estelas |
| `i` | Muestra/oculta los IDs |
| `c` | Alterna color por ID / por clase |
| `[` / `]` | Acorta o alarga la estela en pasos de 10 frames |

### Esquema del JSON de trayectorias

Entrada directa de la etapa de estimación de velocidad. Cada observación trae su
`t_video` ya calculado con `TimeBase` — **la etapa de velocidad debe leerlo de acá,
nunca recalcularlo**:

```json
{
  "version": 1,
  "created_at": "2026-09-16T10:22:41",
  "source_video": "data/videos/sitio_a_01.mp4",
  "frame_size": [1920, 1080],
  "source_fps": 29.97,
  "model_id": "yolov8n",
  "tracker": "bytetrack",
  "confidence_threshold": 0.4,
  "smoothing_window": 5,
  "total_frames_processed": 900,
  "tracks": [
    {
      "track_id": 3,
      "class_name": "car",
      "first_frame": 112, "last_frame": 268,
      "first_seen_t": 3.737, "last_seen_t": 8.942,
      "length_frames": 157,
      "total_displacement_px": 842.3,
      "is_stationary": false,
      "observations": [
        {"frame_idx": 112, "t_video": 3.737, "bbox": [820, 410, 910, 470],
         "confidence": 0.86, "ground_point": [865.0, 470.0]}
      ]
    }
  ]
}
```

### Guía de lectura visual: ¿el tracker está funcionando?

- **Colores estables**: si un vehículo conserva su color a lo largo de toda la escena,
  el seguimiento es correcto. Si cambia de color a media cuadra, acaba de ocurrir un
  cambio de identidad — a simple vista, sin necesitar ninguna métrica.
- **Estelas continuas y suaves**: una estela quebrada o que salta bruscamente de lugar
  indica una asociación errónea entre frames.
- **IDs que no saltan entre vehículos cercanos**: en cruces o adelantamientos, el ID
  debe seguir al mismo vehículo, no "pegarse" al más cercano del frame siguiente.
- **Comportamiento en oclusiones** (un vehículo pasa detrás de un poste, otro vehículo,
  o sale del encuadre y vuelve a entrar): si recupera su ID original, el `track_buffer`
  del tracker está bien calibrado para esa oclusión. Si arranca un ID nuevo, subí
  `track_buffer` en `config/trackers/{bytetrack,botsort}.yaml` — es **el parámetro más
  relevante para oclusiones**. Si en cambio dos vehículos distintos terminan compartiendo
  un ID, `track_buffer` está probablemente demasiado alto para la densidad de tráfico de
  la escena; bajalo.
- **Falsos arranques de ID** (un ID nuevo aparece sin que haya un vehículo nuevo en
  escena): subí `new_track_thresh`.

### Cómo leer la comparación y la gráfica de compromiso

`scripts/run_tracking_benchmark.py` genera `results/plots/tracking_tradeoff.png`: FPS
contra `identity_instability` (el indicador indirecto de cambios de identidad), con cada
combinación modelo+tracker etiquetada. Es la gráfica que justifica la elección final y
la que va al informe: la combinación ideal está arriba (más FPS) y a la izquierda (menor
inestabilidad); todo lo demás es un compromiso entre ambos ejes. `tracking_quality.png`
desglosa `identity_instability` y `fragment_ratio` por combinación, y
`tracking_fps_comparison.png` muestra el FPS puro de cada modelo agrupado por tracker.
La tabla en consola también reporta el overhead de tracking sobre la detección pura del
mismo modelo (si existe un benchmark de detección previo en `results/benchmarks/`), para
cuantificar cuánto cuesta agregar seguimiento en la Raspberry Pi.

### Ejemplos de uso

```bash
# Ver el seguimiento con estelas
python scripts/preview_detection.py --track --tracker bytetrack

# Comparar los dos trackers lado a lado
python scripts/preview_detection.py --track --tracker bytetrack --compare-tracker botsort

# Matriz completa 5 modelos x 2 trackers
python scripts/run_tracking_benchmark.py

# Prueba rápida
python scripts/run_tracking_benchmark.py --models yolov8n --max-frames 300

# Exportar video anotado con IDs y estelas para el informe
python scripts/preview_detection.py --track --export
```

## Notas para hardware embebido

- El benchmark corre por defecto con `--device cpu` para aproximar las condiciones
  de una Raspberry Pi 4 (sin GPU). Usa `--device cuda` únicamente para desarrollo
  rápido en una PC con GPU NVIDIA.
- Las gráficas incluyen líneas de referencia en **15 FPS** (mínimo razonable para
  seguimiento de vehículos en tiempo real) y **512 MB de RAM** (límite típico
  disponible en una RPi4 de 1 GB tras el sistema operativo).
- Para desplegar en una Raspberry Pi 4:
  1. Instala Python 3.9+ y `pip install -r requirements.txt` (usa la versión CPU
     de PyTorch; no existen builds CUDA para RPi4).
  2. Considera usar modelos exportados a formato **NCNN** o **ONNX** vía
     `model.export(format="ncnn")` de Ultralytics para mayor velocidad en ARM.
  3. Los resultados de este benchmark en PC (CPU) sirven como referencia relativa
     entre modelos, no como predicción exacta del FPS en la RPi4 — se recomienda
     repetir el benchmark directamente en el hardware objetivo antes de la
     integración final.

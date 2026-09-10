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

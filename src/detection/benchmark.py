"""
Benchmarking comparativo de variantes YOLO para detección vehicular.
"""
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

from src.utils.metrics import SystemMetrics
from src.utils.roi import ROIFilter, load_roi

logger = logging.getLogger(__name__)

# Fracción recortada de cada cola para la media recortada de Defecto 4 (5% = recorte
# moderado, suficiente para neutralizar uno o dos atípicos sin descartar datos válidos).
_TRIM_FRAC = 0.05
# Raíz del proyecto (src/detection/benchmark.py -> src/detection -> src -> raíz),
# necesaria para invocar el worker aislado como `python -m src.detection.worker`.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _trimmed_mean(values, trim_frac=_TRIM_FRAC):
    """
    Media recortada: descarta la fracción `trim_frac` de valores más bajos y
    más altos antes de promediar, para que uno o dos atípicos no desplacen
    el resultado. Implementada con numpy puro para no agregar scipy como
    dependencia.

    Parámetros:
        values (array-like): valores a promediar.
        trim_frac (float): fracción a recortar de cada cola (0.05 = 5%).

    Retorna:
        float: media recortada. Si el recorte dejaría 0 o menos valores,
        promedia sin recortar.
    """
    arr = np.sort(np.asarray(values, dtype=np.float64))
    n = len(arr)
    k = int(np.floor(n * trim_frac))
    trimmed = arr[k:n - k] if n - 2 * k > 0 else arr
    return float(np.mean(trimmed))


def _counts_by_name(detections, vehicle_classes):
    """
    Cuenta detecciones por nombre de clase real (car/truck/bus/motorcycle),
    a partir de `class_id` — nunca de `class_name`. Con
    `detection.unify_vehicle_classes` activo, `class_name` vale "vehiculo"
    para todas las detecciones por igual, así que agrupar por ese campo
    dejaría el conteo por clase real vacío (o, agrupando contra un dict con
    las 4 claves fijas, directamente un `KeyError`). `class_id` en cambio
    siempre preserva la clase real que entregó el modelo.

    Parámetros:
        detections (list[dict]): detecciones con clave "class_id".
        vehicle_classes (dict[int, str]): mapeo class_id -> nombre real, tal
            como lo expone `VehicleDetector.vehicle_classes`.

    Retorna:
        dict[str, int]: conteo por nombre real, con todas las clases del
        mapeo siempre presentes (aunque valgan 0).
    """
    counts = {name: 0 for name in vehicle_classes.values()}
    for det in detections:
        name = vehicle_classes.get(det["class_id"])
        if name is not None:
            counts[name] += 1
    return counts


class YOLOBenchmark:
    """
    Ejecuta un benchmark comparativo de distintas variantes YOLO sobre un
    video de prueba, midiendo FPS, tiempo de inferencia, uso de CPU/RAM y
    cantidad de vehículos detectados por cada modelo configurado.
    """

    def __init__(self, config, video_path, roi_filter=None):
        """
        Prepara el benchmark: valida el video, arma la lista de modelos a
        evaluar y crea los directorios de resultados necesarios.

        Parámetros:
            config (dict): configuración completa cargada desde config.yaml
                (ya con los overrides de línea de comandos aplicados).
            video_path (str): ruta al archivo de video de prueba.
            roi_filter (ROIFilter | None): filtro de región de interés ya
                construido. Si es None pero `config.roi.enabled` es True, se
                construye uno automáticamente a partir de `config.roi.file`
                para la resolución de este video. La ROI está desactivada
                por defecto: solo se aplica si se pasa explícitamente o si
                el config la habilita.

        Retorna:
            None
        """
        self.config = config
        self.video_path = video_path

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No se encontró el video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        opened = cap.isOpened()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if not opened:
            raise RuntimeError(f"No se pudo abrir el video con OpenCV: {video_path}")

        self.models = config.get("detection", {}).get("models", [])
        if not self.models:
            raise ValueError("No hay modelos configurados para evaluar.")

        paths = config.get("paths", {})
        self.benchmarks_dir = paths.get("benchmarks", "results/benchmarks/")
        self.plots_dir = paths.get("plots", "results/plots/")
        self.frames_dir = paths.get("frames", "results/frames/")
        for directory in (self.benchmarks_dir, self.plots_dir, self.frames_dir):
            os.makedirs(directory, exist_ok=True)

        self.benchmark_cfg = config.get("benchmark", {})
        self.metrics = SystemMetrics()

        roi_cfg = config.get("roi", {})
        if roi_filter is not None:
            self.roi_filter = roi_filter
        elif roi_cfg.get("enabled"):
            roi_data = load_roi(roi_cfg.get("file", "config/roi.json"))
            self.roi_filter = ROIFilter(roi_data, (width, height), config)
            logger.info("ROI activa para este benchmark (%.1f%% del frame)", self.roi_filter.area_ratio() * 100)
        else:
            self.roi_filter = None

    def _process_video(self, detector):
        """
        Procesa el video de prueba con un detector ya cargado, midiendo el
        desempeño de inferencia frame a frame.

        Parámetros:
            detector (VehicleDetector): detector ya inicializado.

        Retorna:
            list[dict]: una entrada por frame procesado, con las claves
            frame_idx, inference_ms, fps, vehicle_count, car_count,
            truck_count, bus_count, motorcycle_count, avg_confidence,
            cpu_percent, ram_mb, gpu_mem_allocated_mb y gpu_mem_peak_mb
            (estas dos últimas None si no hay CUDA), más las columnas
            adicionales vehicle_count_roi, car_count_roi, truck_count_roi,
            bus_count_roi y motorcycle_count_roi (iguales a sus contrapartes
            sin sufijo si no hay ROI activa).
        """
        warmup_frames = self.benchmark_cfg.get("warmup_frames", 15)
        max_frames = self.benchmark_cfg.get("max_frames", 300)
        save_annotated = self.benchmark_cfg.get("save_annotated_frames", True)
        annotated_interval = max(1, self.benchmark_cfg.get("annotated_frame_interval", 50))

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {self.video_path}")

        model_frames_dir = os.path.join(self.frames_dir, detector.model_id)
        if save_annotated:
            os.makedirs(model_frames_dir, exist_ok=True)

        # El calentamiento tiene que ejecutar inferencia de verdad, no solo avanzar el
        # video: la primera llamada real al modelo paga la inicialización del contexto
        # CUDA, el autotuning de cuDNN y la transferencia de los pesos a la GPU — en este
        # equipo, ese costo se midió en hasta 22 segundos. Si el calentamiento no corre el
        # modelo, ese costo cae dentro del frame 0 ya medido y arruina las estadísticas de
        # todo el modelo (media, std y hasta el orden esperado entre modelos).
        print(f"  ⏳ Calentamiento ({warmup_frames} frames)...")
        warmed = 0
        for _ in range(warmup_frames):
            ret, frame = cap.read()
            if not ret:
                break
            detector.detect(frame)
            warmed += 1

        if warmed == 0:
            logger.warning(
                "No se pudo calentar el modelo: el video no tiene frames suficientes. "
                "Las primeras mediciones incluirán el costo de inicialización."
            )

        detector.synchronize()

        self.metrics.reset_gpu_peak()
        self.metrics.start()
        frame_data = []

        print(f"  ▶ Procesando {max_frames} frames...")
        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                logger.info("Video agotado en el frame %d (antes de llegar a max_frames).", frame_idx)
                break

            detector.synchronize()  # no arrastrar trabajo pendiente del frame previo
            t_start = time.perf_counter()
            detections = detector.detect(frame)
            detector.synchronize()  # esperar a que la inferencia de este frame termine de verdad
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000.0
            fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            sys_metrics = self.metrics.measure()

            counts = _counts_by_name(detections, detector.vehicle_classes)
            confidences = [det["confidence"] for det in detections]

            row = {
                "frame_idx": frame_idx,
                "inference_ms": inference_ms,
                "fps": fps,
                "vehicle_count": len(detections),
                "car_count": counts.get("car", 0),
                "truck_count": counts.get("truck", 0),
                "bus_count": counts.get("bus", 0),
                "motorcycle_count": counts.get("motorcycle", 0),
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "cpu_percent": sys_metrics["cpu_percent"],
                "ram_mb": sys_metrics["ram_mb"],
                "gpu_mem_allocated_mb": sys_metrics.get("gpu_mem_allocated_mb"),
                "gpu_mem_peak_mb": sys_metrics.get("gpu_mem_peak_mb"),
            }

            # Columnas adicionales de ROI, estrictamente al final: si no hay
            # ROI activa, `inside` es la lista completa de detecciones y
            # estas columnas quedan idénticas a sus contrapartes sin sufijo.
            if self.roi_filter is not None:
                inside, _outside = self.roi_filter.filter(detections)
            else:
                inside = detections

            roi_counts = _counts_by_name(inside, detector.vehicle_classes)

            row["vehicle_count_roi"] = len(inside)
            row["car_count_roi"] = roi_counts.get("car", 0)
            row["truck_count_roi"] = roi_counts.get("truck", 0)
            row["bus_count_roi"] = roi_counts.get("bus", 0)
            row["motorcycle_count_roi"] = roi_counts.get("motorcycle", 0)

            frame_data.append(row)

            if save_annotated and frame_idx % annotated_interval == 0:
                annotated = detector.draw_detections(frame, detections)
                frame_path = os.path.join(model_frames_dir, f"frame_{frame_idx:05d}.jpg")
                cv2.imwrite(frame_path, annotated)

        cap.release()
        return frame_data

    def _compute_summary(self, model_info, frame_data):
        """
        Calcula estadísticas resumen a partir de los datos de todos los
        frames procesados para un modelo.

        Parámetros:
            model_info (dict): info del modelo, ver `VehicleDetector.get_model_info`.
            frame_data (list[dict]): datos por frame generados por `_process_video`.

        Retorna:
            dict: resumen de métricas del modelo (fps_mean, fps_std, fps_p5,
            fps_p95, inference_ms_mean/std, vehicles_per_frame_mean/std,
            avg_confidence_mean, cpu_percent_mean, ram_mb_mean, ram_mb_peak,
            entre otros), los campos de ROI (roi_enabled, roi_area_ratio,
            vehicles_per_frame_roi_mean, false_positive_ratio), y al final:
            inference_ms_median/p95/trimmed_mean, fps_median, outlier_frames
            (estadísticas robustas a atípicos, Defecto 4), gpu_mem_peak_mb
            (None si no hay CUDA, Defecto 3) y vehicles_per_frame_raw_mean
            (conteo sin filtrar por ROI, para conservar la comparación;
            Defecto 5).

            Con ROI activa, `vehicles_per_frame_mean`/`_std` reflejan el
            carril de interés (se calculan sobre `vehicle_count_roi`); sin
            ROI, reflejan la escena completa (`vehicle_count`). En ambos
            casos `vehicles_per_frame_raw_mean` guarda siempre el valor sin
            filtrar, para poder comparar.
        """
        df = pd.DataFrame(frame_data)

        roi_enabled = self.roi_filter is not None
        count_col = "vehicle_count_roi" if roi_enabled else "vehicle_count"

        summary = {
            "model_id": model_info["model_id"],
            "weights": model_info["weights"],
            "parameters_M": model_info["parameters_M"],
            "device": model_info["device"],
            "frames_processed": len(df),
            "fps_mean": float(df["fps"].mean()),
            "fps_std": float(df["fps"].std(ddof=0)),
            "fps_p5": float(np.percentile(df["fps"], 5)),
            "fps_p95": float(np.percentile(df["fps"], 95)),
            "inference_ms_mean": float(df["inference_ms"].mean()),
            "inference_ms_std": float(df["inference_ms"].std(ddof=0)),
            "vehicles_per_frame_mean": float(df[count_col].mean()),
            "vehicles_per_frame_std": float(df[count_col].std(ddof=0)),
            "avg_confidence_mean": float(df["avg_confidence"].mean()),
            "cpu_percent_mean": float(df["cpu_percent"].mean()),
            "ram_mb_mean": float(df["ram_mb"].mean()),
            "ram_mb_peak": float(df["ram_mb"].max()),
        }

        total_all = df["vehicle_count"].sum()
        total_inside = df["vehicle_count_roi"].sum()

        summary["roi_enabled"] = roi_enabled
        summary["roi_area_ratio"] = self.roi_filter.area_ratio() if roi_enabled else None
        summary["vehicles_per_frame_roi_mean"] = float(df["vehicle_count_roi"].mean())
        summary["false_positive_ratio"] = (
            float((total_all - total_inside) / total_all) if roi_enabled and total_all > 0 else None
        )

        # Defecto 4: estadísticas robustas a atípicos, adicionales a mean/std.
        inference_ms = df["inference_ms"].to_numpy(dtype=np.float64)
        inference_ms_median = float(np.median(inference_ms))
        summary["inference_ms_median"] = inference_ms_median
        summary["inference_ms_p95"] = float(np.percentile(inference_ms, 95))
        summary["inference_ms_trimmed_mean"] = _trimmed_mean(inference_ms)
        summary["fps_median"] = float(df["fps"].median())
        summary["outlier_frames"] = int(np.sum(inference_ms > 10 * inference_ms_median))

        # Defecto 3: memoria pico de GPU (None si la corrida fue en CPU).
        if "gpu_mem_peak_mb" in df.columns and df["gpu_mem_peak_mb"].notna().any():
            summary["gpu_mem_peak_mb"] = float(df["gpu_mem_peak_mb"].max())
        else:
            summary["gpu_mem_peak_mb"] = None

        # Defecto 5: conteo crudo, para comparar contra el filtrado por ROI arriba.
        summary["vehicles_per_frame_raw_mean"] = float(df["vehicle_count"].mean())

        return summary

    def run(self):
        """
        Ejecuta el benchmark completo: evalúa cada modelo configurado sobre
        el video de prueba, guarda los resultados y genera las gráficas
        comparativas.

        Cada modelo se evalúa en un subproceso propio (`src.detection.worker`)
        en lugar de en el proceso actual: PyTorch no siempre libera la memoria
        de un modelo antes de cargar el siguiente, así que medir varios modelos
        de forma secuencial en un mismo proceso infla la RAM de los que se
        evalúan más tarde. Un proceso nuevo por modelo da una medición de RAM
        limpia, independiente del orden de evaluación.

        Si un modelo falla al cargar o al procesarse, el error se registra y
        se continúa con el siguiente modelo (no aborta todo el benchmark).

        Retorna:
            list[dict]: lista de resúmenes, uno por modelo evaluado exitosamente.
        """
        all_frame_data = []
        all_summaries = []
        total = len(self.models)
        config_path = self._write_temp_config()

        try:
            for idx, model_cfg in enumerate(self.models, start=1):
                model_id = model_cfg["id"]
                weights = model_cfg["weights"]
                print(f"\n[{idx}/{total}] Evaluando modelo: {model_id}...")
                logger.info("[%d/%d] Evaluando modelo: %s", idx, total, model_id)

                fd, output_path = tempfile.mkstemp(suffix=".json", prefix=f"yolo_{model_id}_")
                os.close(fd)

                try:
                    proc = subprocess.run(
                        [
                            sys.executable, "-m", "src.detection.worker",
                            "--config", config_path,
                            "--video", self.video_path,
                            "--model-id", model_id,
                            "--weights", weights,
                            "--output", output_path,
                        ],
                        cwd=_PROJECT_ROOT,
                    )

                    try:
                        with open(output_path, "r", encoding="utf-8") as f:
                            result = json.load(f)
                    except (OSError, json.JSONDecodeError):
                        result = None

                    if result is None or "error" in result:
                        reason = result["error"] if result else f"el worker terminó con código {proc.returncode}"
                        raise RuntimeError(reason)

                    frame_data = result["frame_data"]
                    summary = result["summary"]

                    if not frame_data:
                        logger.warning("El modelo %s no produjo frames procesados; se omite.", model_id)
                        print(f"  ✗ {model_id}: no se procesó ningún frame (video vacío o muy corto)")
                        continue

                    all_frame_data.extend({"model_id": model_id, **row} for row in frame_data)
                    all_summaries.append(summary)

                    print(
                        f"  ✓ {model_id} completado: "
                        f"{summary['fps_mean']:.1f} FPS | "
                        f"{summary['inference_ms_mean']:.1f} ms | "
                        f"{summary['ram_mb_mean']:.0f} MB RAM"
                    )
                except Exception as exc:
                    logger.error("Error evaluando el modelo %s: %s", model_id, exc, exc_info=True)
                    print(f"  ✗ Error evaluando {model_id}: {exc}")
                    continue
                finally:
                    if os.path.exists(output_path):
                        os.remove(output_path)
        finally:
            if os.path.exists(config_path):
                os.remove(config_path)

        if not all_summaries:
            logger.error("Ningún modelo pudo evaluarse correctamente.")
            print("\n✗ Ningún modelo pudo evaluarse correctamente. No se generaron resultados.")
            return []

        timestamp = self._save_results(all_frame_data, all_summaries)
        self._generate_plots(all_summaries, timestamp)

        return all_summaries

    def _write_temp_config(self):
        """
        Serializa la configuración actual a un archivo YAML temporal, para
        poder pasársela tal cual a los procesos worker (uno por modelo).

        Retorna:
            str: ruta al archivo YAML temporal creado.
        """
        fd, path = tempfile.mkstemp(suffix=".yaml", prefix="yolo_bench_config_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.config, f)
        return path

    def _save_results(self, all_frame_data, all_summaries):
        """
        Guarda los datos crudos por frame (CSV) y el resumen por modelo
        (JSON) en `results/benchmarks/`.

        Parámetros:
            all_frame_data (list[dict]): datos de todos los frames de todos
                los modelos evaluados.
            all_summaries (list[dict]): resúmenes de todos los modelos.

        Retorna:
            str: el timestamp (`YYYYMMDD_HHMMSS`) usado para nombrar los
            archivos generados, para que `_generate_plots` use el mismo.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        csv_path = os.path.join(self.benchmarks_dir, f"frame_data_{timestamp}.csv")
        columns = [
            "model_id", "frame_idx", "inference_ms", "fps", "vehicle_count",
            "car_count", "truck_count", "bus_count", "motorcycle_count",
            "avg_confidence", "cpu_percent", "ram_mb",
            "vehicle_count_roi", "car_count_roi", "truck_count_roi",
            "bus_count_roi", "motorcycle_count_roi",
            "gpu_mem_allocated_mb", "gpu_mem_peak_mb",
        ]
        pd.DataFrame(all_frame_data, columns=columns).to_csv(csv_path, index=False)

        json_path = os.path.join(self.benchmarks_dir, f"summary_{timestamp}.json")
        summary_payload = {
            "video_path": self.video_path,
            "timestamp": timestamp,
            "system_info": self.metrics.get_system_info(),
            "config": self.config,
            "models": all_summaries,
            "roi": {
                "enabled": self.roi_filter is not None,
                "mode": self.roi_filter.mode if self.roi_filter else None,
                "polygon_norm": self.roi_filter.polygon_norm if self.roi_filter else None,
                "area_ratio": self.roi_filter.area_ratio() if self.roi_filter else None,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        print("\nArchivos generados:")
        print(f"  \U0001F4CA {csv_path}")
        print(f"  \U0001F4CB {json_path}")
        logger.info("Resultados guardados en %s y %s", csv_path, json_path)

        return timestamp

    def _generate_plots(self, all_summaries, timestamp):
        """
        Genera las gráficas comparativas del benchmark y las guarda en
        `results/plots/`.

        Produce dos archivos: `benchmark_{timestamp}.png` (4 subplots con
        FPS, tiempo de inferencia, RAM y vehículos/frame por modelo) y
        `fps_vs_ram_{timestamp}.png` (scatter FPS vs. RAM).

        Parámetros:
            all_summaries (list[dict]): resúmenes de todos los modelos.
            timestamp (str): timestamp usado para nombrar los archivos.

        Retorna:
            None
        """
        sns.set_theme(style="whitegrid")
        df = pd.DataFrame(all_summaries).sort_values("fps_mean", ascending=False)
        device = df["device"].iloc[0] if not df.empty else "desconocido"

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f"Comparación de variantes YOLO — Hardware: {device} — {timestamp}", fontsize=14)

        # Subplot 1: FPS promedio por modelo (ya ordenado de mayor a menor)
        ax = axes[0, 0]
        ax.bar(df["model_id"], df["fps_mean"], yerr=df["fps_std"], capsize=4, color="steelblue")
        ax.axhline(15, color="red", linestyle="--", label="Mínimo embebido (15 FPS)")
        ax.set_title("FPS promedio por modelo")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("FPS")
        ax.legend()

        # Subplot 2: Tiempo de inferencia (ms), de azul (rápido) a rojo (lento)
        ax = axes[0, 1]
        df_inf = df.sort_values("inference_ms_mean")
        norm = plt.Normalize(df_inf["inference_ms_mean"].min(), df_inf["inference_ms_mean"].max())
        colors = plt.cm.coolwarm(norm(df_inf["inference_ms_mean"]))
        ax.bar(df_inf["model_id"], df_inf["inference_ms_mean"], yerr=df_inf["inference_ms_std"],
               capsize=4, color=colors)
        ax.set_title("Tiempo de inferencia promedio")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Inferencia (ms)")

        # Subplot 3: memoria — pico de GPU si la corrida fue en CUDA y hay datos; si no,
        # RSS del proceso. La línea de 512 MB (límite típico RPi4) se saca en los dos
        # casos: en CUDA compara magnitudes que no existen en la RPi (no tiene ese
        # runtime cargado), y en CPU el RSS incluye intérprete y bibliotecas, no solo el
        # modelo, así que tampoco es comparable directamente contra ese límite.
        ax = axes[1, 0]
        has_gpu_mem = "gpu_mem_peak_mb" in df.columns and df["gpu_mem_peak_mb"].notna().any()
        if device == "cuda" and has_gpu_mem:
            ax.bar(df["model_id"], df["gpu_mem_peak_mb"], color="seagreen")
            ax.set_title("Memoria pico de GPU por modelo")
            ax.set_ylabel("Memoria GPU (MB)")
        else:
            ax.bar(df["model_id"], df["ram_mb_mean"], color="seagreen")
            ax.set_title("RSS del proceso por modelo (incluye intérprete y bibliotecas)")
            ax.set_ylabel("RAM (MB)")
        ax.set_xlabel("Modelo")

        # Subplot 4: vehículos por frame — si hay ROI activa, crudo vs. filtrado por el
        # polígono, para que el contraste muestre cuánto ruido mete el parqueo de fondo.
        ax = axes[1, 1]
        roi_active = bool(df["roi_enabled"].iloc[0]) if "roi_enabled" in df.columns and not df.empty else False
        if roi_active and "vehicles_per_frame_raw_mean" in df.columns:
            x = np.arange(len(df))
            bar_w = 0.35
            ax.bar(x - bar_w / 2, df["vehicles_per_frame_raw_mean"], bar_w,
                   label="Escena completa", color="darkorange")
            ax.bar(x + bar_w / 2, df["vehicles_per_frame_mean"], bar_w,
                   label="Zona de interés (ROI)", color="steelblue")
            ax.set_xticks(x)
            ax.set_xticklabels(df["model_id"])
            ax.set_title("Vehículos por frame (escena completa vs. zona de interés)")
            ax.legend()
        else:
            ax.bar(df["model_id"], df["vehicles_per_frame_mean"], yerr=df["vehicles_per_frame_std"],
                   capsize=4, color="darkorange")
            ax.set_title("Vehículos detectados por frame")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Vehículos / frame")

        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=20)

        fig.text(
            0.5, 0.01,
            "El consumo de memoria del equipo de desarrollo no es extrapolable a la plataforma embebida.",
            ha="center", fontsize=8, style="italic",
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        benchmark_plot_path = os.path.join(self.plots_dir, f"benchmark_{timestamp}.png")
        fig.savefig(benchmark_plot_path, dpi=150)
        plt.close(fig)

        # Segunda figura: balance FPS vs. RAM
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        palette = sns.color_palette("husl", len(df))
        for (_, row), color in zip(df.iterrows(), palette):
            ax2.scatter(row["ram_mb_mean"], row["fps_mean"], s=100, color=color)
            ax2.annotate(
                row["model_id"], (row["ram_mb_mean"], row["fps_mean"]),
                textcoords="offset points", xytext=(6, 6), fontsize=9,
            )
        ax2.set_title("Balance FPS vs. Consumo de RAM por modelo")
        ax2.set_xlabel("RAM (MB)")
        ax2.set_ylabel("FPS")
        fig2.tight_layout()

        fps_ram_path = os.path.join(self.plots_dir, f"fps_vs_ram_{timestamp}.png")
        fig2.savefig(fps_ram_path, dpi=150)
        plt.close(fig2)

        print(f"  \U0001F4C8 {benchmark_plot_path}")
        print(f"  \U0001F4C8 {fps_ram_path}")
        logger.info("Gráficas guardadas en %s y %s", benchmark_plot_path, fps_ram_path)

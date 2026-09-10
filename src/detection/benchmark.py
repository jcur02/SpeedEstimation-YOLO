"""
Benchmarking comparativo de variantes YOLO para detección vehicular.
"""
import json
import logging
import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.detection.detector import VehicleDetector
from src.utils.metrics import SystemMetrics

logger = logging.getLogger(__name__)

_VEHICLE_CLASS_NAMES = ["car", "truck", "bus", "motorcycle"]


class YOLOBenchmark:
    """
    Ejecuta un benchmark comparativo de distintas variantes YOLO sobre un
    video de prueba, midiendo FPS, tiempo de inferencia, uso de CPU/RAM y
    cantidad de vehículos detectados por cada modelo configurado.
    """

    def __init__(self, config, video_path):
        """
        Prepara el benchmark: valida el video, arma la lista de modelos a
        evaluar y crea los directorios de resultados necesarios.

        Parámetros:
            config (dict): configuración completa cargada desde config.yaml
                (ya con los overrides de línea de comandos aplicados).
            video_path (str): ruta al archivo de video de prueba.

        Retorna:
            None
        """
        self.config = config
        self.video_path = video_path

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"No se encontró el video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        opened = cap.isOpened()
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
            cpu_percent y ram_mb.
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

        print(f"  ⏳ Calentamiento ({warmup_frames} frames)...")
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                break

        self.metrics.start()
        frame_data = []

        print(f"  ▶ Procesando {max_frames} frames...")
        for frame_idx in range(max_frames):
            ret, frame = cap.read()
            if not ret:
                logger.info("Video agotado en el frame %d (antes de llegar a max_frames).", frame_idx)
                break

            t_start = time.perf_counter()
            detections = detector.detect(frame)
            t_end = time.perf_counter()

            inference_ms = (t_end - t_start) * 1000.0
            fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
            sys_metrics = self.metrics.measure()

            counts = {name: 0 for name in _VEHICLE_CLASS_NAMES}
            confidences = []
            for det in detections:
                counts[det["class_name"]] += 1
                confidences.append(det["confidence"])

            frame_data.append({
                "frame_idx": frame_idx,
                "inference_ms": inference_ms,
                "fps": fps,
                "vehicle_count": len(detections),
                "car_count": counts["car"],
                "truck_count": counts["truck"],
                "bus_count": counts["bus"],
                "motorcycle_count": counts["motorcycle"],
                "avg_confidence": float(np.mean(confidences)) if confidences else 0.0,
                "cpu_percent": sys_metrics["cpu_percent"],
                "ram_mb": sys_metrics["ram_mb"],
            })

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
            entre otros).
        """
        df = pd.DataFrame(frame_data)

        return {
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
            "vehicles_per_frame_mean": float(df["vehicle_count"].mean()),
            "vehicles_per_frame_std": float(df["vehicle_count"].std(ddof=0)),
            "avg_confidence_mean": float(df["avg_confidence"].mean()),
            "cpu_percent_mean": float(df["cpu_percent"].mean()),
            "ram_mb_mean": float(df["ram_mb"].mean()),
            "ram_mb_peak": float(df["ram_mb"].max()),
        }

    def run(self):
        """
        Ejecuta el benchmark completo: evalúa cada modelo configurado sobre
        el video de prueba, guarda los resultados y genera las gráficas
        comparativas.

        Si un modelo falla al cargar o al procesarse, el error se registra
        y se continúa con el siguiente modelo (no aborta todo el benchmark).

        Retorna:
            list[dict]: lista de resúmenes, uno por modelo evaluado exitosamente.
        """
        all_frame_data = []
        all_summaries = []
        total = len(self.models)

        for idx, model_cfg in enumerate(self.models, start=1):
            model_id = model_cfg["id"]
            weights = model_cfg["weights"]
            print(f"\n[{idx}/{total}] Evaluando modelo: {model_id}...")
            logger.info("[%d/%d] Evaluando modelo: %s", idx, total, model_id)

            try:
                detector = VehicleDetector(model_id, weights, self.config)
                model_info = detector.get_model_info()
                print(f"  ✓ Modelo cargado ({model_info['parameters_M']:.1f} M parámetros)")

                frame_data = self._process_video(detector)
                if not frame_data:
                    logger.warning("El modelo %s no produjo frames procesados; se omite.", model_id)
                    print(f"  ✗ {model_id}: no se procesó ningún frame (video vacío o muy corto)")
                    continue

                summary = self._compute_summary(model_info, frame_data)
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

        if not all_summaries:
            logger.error("Ningún modelo pudo evaluarse correctamente.")
            print("\n✗ Ningún modelo pudo evaluarse correctamente. No se generaron resultados.")
            return []

        timestamp = self._save_results(all_frame_data, all_summaries)
        self._generate_plots(all_summaries, timestamp)

        return all_summaries

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
        ]
        pd.DataFrame(all_frame_data, columns=columns).to_csv(csv_path, index=False)

        json_path = os.path.join(self.benchmarks_dir, f"summary_{timestamp}.json")
        summary_payload = {
            "video_path": self.video_path,
            "timestamp": timestamp,
            "system_info": self.metrics.get_system_info(),
            "config": self.config,
            "models": all_summaries,
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

        # Subplot 3: RAM usada (MB)
        ax = axes[1, 0]
        ax.bar(df["model_id"], df["ram_mb_mean"], color="seagreen")
        ax.axhline(512, color="darkred", linestyle="--", label="Límite típico RPi4 (512 MB)")
        ax.set_title("RAM usada por modelo")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("RAM (MB)")
        ax.legend()

        # Subplot 4: Vehículos detectados por frame
        ax = axes[1, 1]
        ax.bar(df["model_id"], df["vehicles_per_frame_mean"], yerr=df["vehicles_per_frame_std"],
               capsize=4, color="darkorange")
        ax.set_title("Vehículos detectados por frame")
        ax.set_xlabel("Modelo")
        ax.set_ylabel("Vehículos / frame")

        for ax in axes.flat:
            ax.tick_params(axis="x", rotation=20)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
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

#!/usr/bin/env python3
"""
Verifica que la corrección del banco de detección y del banco de seguimiento
(calentamiento real, sincronización CUDA, estadísticas robustas a atípicos,
baseline de overhead del mismo dispositivo) se comporten como se espera,
sobre un `summary_*.json` (detección) o `tracking_summary_*.json`
(seguimiento) ya generado.

El tipo de resumen se detecta por sus claves (`"models"` = detección,
`"results"` = seguimiento), no por el nombre del archivo.

Uso:
    python scripts/verify_benchmark_fix.py
    python scripts/verify_benchmark_fix.py --summary results/benchmarks/summary_20261012_123456.json
    python scripts/verify_benchmark_fix.py --summary results/benchmarks/tracking_summary_20261012_123456.json
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FAMILIES = [["yolov8n", "yolov8s", "yolov8m"], ["yolo11n", "yolo11s"]]


def parse_args() -> argparse.Namespace:
    """Define y parsea los argumentos de línea de comandos del script."""
    parser = argparse.ArgumentParser(
        description="Verifica la corrección del benchmark de detección y/o de seguimiento."
    )
    parser.add_argument(
        "--summary", type=str, default=None,
        help="Ruta a un summary_*.json o tracking_summary_*.json concreto.",
    )
    parser.add_argument(
        "--benchmarks-dir", type=str, default="results/benchmarks/",
        help="Directorio donde buscar el summary más reciente (de cualquiera de los dos "
             "tipos) si no se pasa --summary.",
    )
    return parser.parse_args()


def find_latest_summary(benchmarks_dir: str):
    """
    Busca el resumen más reciente (por fecha de modificación) entre
    `summary_*.json` (detección) y `tracking_summary_*.json` (seguimiento).

    Retorna:
        str | None: ruta al más reciente de ambos tipos, o None si no hay
        ninguno.
    """
    candidates = (
        glob.glob(os.path.join(benchmarks_dir, "summary_*.json"))
        + glob.glob(os.path.join(benchmarks_dir, "tracking_summary_*.json"))
    )
    return max(candidates, key=os.path.getmtime) if candidates else None


def _row_label(r: dict) -> str:
    """
    Etiqueta legible de una fila para los mensajes de detalle: "model_id"
    para un resumen de detección, "model_id+tracker" para uno de
    seguimiento (donde el mismo model_id puede repetirse con trackers
    distintos, y dejaría mensajes ambiguos si se etiquetara solo por modelo).

    Retorna:
        str
    """
    model_id = r.get("model_id", "?")
    tracker = r.get("tracker")
    return f"{model_id}+{tracker}" if tracker else model_id


def check_fps_time_coherence(rows):
    """
    Comprobación: `abs(1000 / inference_ms_median - fps_median) < 1` para
    cada fila. Si falla, las dos métricas siguen midiendo cosas distintas.
    Aplica igual a resúmenes de detección y de seguimiento.

    Retorna:
        tuple[bool, str]: (pasó, detalle).
    """
    problems = []
    for r in rows:
        median, fps_median = r.get("inference_ms_median"), r.get("fps_median")
        if median is None or fps_median is None:
            problems.append(f"{_row_label(r)}: faltan inference_ms_median/fps_median (¿se corrió con el código corregido?)")
            continue
        diff = abs(1000.0 / median - fps_median) if median > 0 else float("inf")
        if diff >= 1.0:
            problems.append(f"{_row_label(r)}: diferencia de {diff:.2f} FPS (esperado < 1)")
    passed = not problems
    return passed, ("todas las coherencias dentro de 1 FPS" if passed else "; ".join(problems))


def check_no_startup_outliers(rows):
    """
    Comprobación: `outlier_frames == 0` en todas las filas. Aplica igual a
    resúmenes de detección y de seguimiento.

    Retorna:
        tuple[bool, str]: (pasó, detalle).
    """
    problems = [
        f"{_row_label(r)}: {r.get('outlier_frames')} frame(s) atípico(s)"
        for r in rows if r.get("outlier_frames", 0) != 0
    ]
    passed = not problems
    return passed, ("sin atípicos de arranque en ninguna fila" if passed else "; ".join(problems))


def check_monotonic_order(rows):
    """
    Comprobación (solo detección, no bloqueante): dentro de cada familia, el
    modelo con más parámetros debería reportar `inference_ms_median` mayor o
    igual que el más chico (yolov8n ≤ yolov8s ≤ yolov8m; yolo11n ≤ yolo11s).
    Puede violarse por optimizaciones específicas de arquitectura, pero
    merece revisión manual.

    Retorna:
        tuple[bool, str]: (se respetó el orden, detalle).
    """
    by_id = {r["model_id"]: r for r in rows}
    violations = []
    for family in _FAMILIES:
        present = [mid for mid in family if mid in by_id]
        for smaller, bigger in zip(present, present[1:]):
            ms_smaller = by_id[smaller].get("inference_ms_median")
            ms_bigger = by_id[bigger].get("inference_ms_median")
            if ms_smaller is None or ms_bigger is None:
                continue
            if ms_smaller > ms_bigger:
                violations.append(f"{smaller} ({ms_smaller:.1f}ms) > {bigger} ({ms_bigger:.1f}ms)")
    passed = not violations
    return passed, ("orden monótono respetado en todas las familias" if passed else "; ".join(violations))


def check_bounded_std(rows):
    """
    Comprobación: `inference_ms_std` no debe superar 3× la mediana en
    ninguna fila. Bloqueante para detección, advertencia para seguimiento
    (ver tabla de comprobaciones en `main()`).

    Retorna:
        tuple[bool, str]: (pasó, detalle).
    """
    problems = []
    for r in rows:
        std, median = r.get("inference_ms_std"), r.get("inference_ms_median")
        if std is None or median is None or median <= 0:
            continue
        if std > 3 * median:
            problems.append(f"{_row_label(r)}: std={std:.1f}ms supera 3× la mediana ({3 * median:.1f}ms)")
    passed = not problems
    return passed, ("desviación acotada en todas las filas" if passed else "; ".join(problems))


def check_tracking_overhead_non_negative(rows):
    """
    Comprobación (solo seguimiento): si `tracking_overhead_ms` no es None,
    debe ser mayor que cero. Agregar un seguidor nunca puede ser más rápido
    que solo detectar; un overhead negativo siempre delata un baseline de
    detección mal elegido (ver Defecto 2 de PROMPT_fix_tracking.md).

    Retorna:
        tuple[bool, str]: (pasó, detalle).
    """
    problems = []
    for r in rows:
        overhead = r.get("tracking_overhead_ms")
        if overhead is None:
            continue
        if overhead <= 0:
            problems.append(f"{_row_label(r)}: overhead={overhead:.2f}ms (debe ser > 0)")
    passed = not problems
    return passed, ("overhead positivo en todas las filas con baseline" if passed else "; ".join(problems))


def check_baseline_device_coherence(rows):
    """
    Comprobación (solo seguimiento): donde haya overhead calculado,
    `detection_baseline_device` debe coincidir con el `device` de esa misma
    fila — comparar tiempos entre dispositivos distintos invalida el
    overhead (ver Defecto 2).

    Retorna:
        tuple[bool, str]: (pasó, detalle).
    """
    problems = []
    for r in rows:
        if r.get("tracking_overhead_ms") is None:
            continue
        baseline_device, run_device = r.get("detection_baseline_device"), r.get("device")
        if baseline_device is not None and run_device is not None and baseline_device != run_device:
            problems.append(f"{_row_label(r)}: baseline device='{baseline_device}' vs. corrida device='{run_device}'")
    passed = not problems
    return passed, ("baseline del mismo dispositivo en todos los casos" if passed else "; ".join(problems))


_DETECTION_CHECKS = [
    ("1. Coherencia FPS-tiempo", check_fps_time_coherence, True),
    ("2. Ausencia de atípicos de arranque", check_no_startup_outliers, True),
    ("3. Orden monótono dentro de cada familia", check_monotonic_order, False),
    ("4. Desviación acotada", check_bounded_std, True),
]

_TRACKING_CHECKS = [
    ("1. Coherencia FPS-tiempo", check_fps_time_coherence, True),
    ("2. Ausencia de atípicos de arranque", check_no_startup_outliers, True),
    ("3. Overhead de seguimiento no negativo", check_tracking_overhead_non_negative, True),
    ("4. Coherencia del baseline", check_baseline_device_coherence, True),
    ("5. Desviación acotada", check_bounded_std, False),
]


def main() -> None:
    """Punto de entrada principal."""
    args = parse_args()
    summary_path = args.summary or find_latest_summary(args.benchmarks_dir)
    if not summary_path or not os.path.isfile(summary_path):
        print(
            f"✗ No se encontró ningún summary_*.json ni tracking_summary_*.json en "
            f"{args.benchmarks_dir}. Corré primero scripts/run_benchmark.py o "
            f"scripts/run_tracking_benchmark.py, o pasá --summary."
        )
        sys.exit(1)

    with open(summary_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "results" in data:
        kind, rows, checks = "seguimiento", data["results"], _TRACKING_CHECKS
    elif "models" in data:
        kind, rows, checks = "detección", data["models"], _DETECTION_CHECKS
    else:
        print(f"✗ {summary_path} no tiene ni 'models' ni 'results': no se reconoce el tipo de resumen.")
        sys.exit(1)

    if not rows:
        print(f"✗ {summary_path} no contiene filas para verificar.")
        sys.exit(1)

    print(f"Verificando ({kind}): {summary_path}\n")

    all_passed = True
    for name, check_fn, blocking in checks:
        passed, detail = check_fn(rows)
        if blocking:
            mark = "✓" if passed else "✗"
            all_passed = all_passed and passed
        else:
            mark = "✓" if passed else "⚠"
        print(f"  [{mark}] {name}: {detail}")

    if all_passed:
        print("\n✓ El benchmark corregido pasa todas las comprobaciones bloqueantes.")
    else:
        print("\n✗ Alguna comprobación bloqueante falló — revisá el detalle arriba antes de confiar en estos números.")
        sys.exit(1)


if __name__ == "__main__":
    main()

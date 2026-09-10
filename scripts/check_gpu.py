"""
Script de verificación: comprueba si GPU está disponible para el proyecto.
Ejecutar antes del benchmark: python scripts/check_gpu.py
"""
import subprocess
import sys

def check_cuda():
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"PyTorch version  : {torch.__version__}")
        print(f"CUDA disponible  : {cuda_ok}")
        if cuda_ok:
            print(f"GPU detectada    : {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"VRAM total       : {props.total_memory / 1024**3:.1f} GB")
            print(f"CUDA version     : {torch.version.cuda}")
            print(f"\n✓ Listo para usar --device cuda")
        else:
            print("\n⚠ No se detectó GPU CUDA. Usa --device cpu")
    except ImportError:
        print("✗ PyTorch no instalado. Sigue las instrucciones de instalación.")

def check_ultralytics():
    try:
        from ultralytics import YOLO
        print("\nUltralytics YOLO : instalado ✓")
    except ImportError:
        print("\n✗ Ultralytics no instalado. Ejecuta: pip install -r requirements.txt")

if __name__ == "__main__":
    check_cuda()
    check_ultralytics()

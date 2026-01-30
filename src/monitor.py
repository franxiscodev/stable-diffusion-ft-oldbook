"""
Monitor de entrenamiento para Stable Diffusion fine-tuning.

Lee el log de entrenamiento y muestra el estado actual.
Ejecutar en otra terminal mientras corre finetune.py:

  python -m src.monitor
  python -m src.monitor --follow   # actualiza cada 5s
"""

import argparse
import re
import time
from pathlib import Path

from src.config import PROJECT_ROOT

LOG_FILE = PROJECT_ROOT / "outputs" / "training_log.txt"
CHECKPOINT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"


def parse_last_batch_line(lines):
    """Extrae info del ultimo log de batch."""
    pattern = re.compile(
        r"\[Batch (\d+)/(\d+)\] \[Global (\d+)%\] "
        r"Loss: ([\d.]+) \| ([\d.]+)s/batch \| ETA: (.+)"
    )
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            return {
                "batch": int(m.group(1)),
                "total_batches": int(m.group(2)),
                "pct": int(m.group(3)),
                "loss": float(m.group(4)),
                "batch_time": float(m.group(5)),
                "eta": m.group(6),
            }
    return None


def parse_last_epoch_line(lines):
    """Extrae info del ultimo log de epoch completado."""
    pattern = re.compile(
        r"Epoch (\d+)/(\d+) completado - Loss promedio: ([\d.]+) - Tiempo: (.+)"
    )
    for line in reversed(lines):
        m = pattern.search(line)
        if m:
            return {
                "epoch": int(m.group(1)),
                "total_epochs": int(m.group(2)),
                "avg_loss": float(m.group(3)),
                "epoch_time": m.group(4),
            }
    return None


def find_checkpoints():
    """Lista los checkpoints existentes."""
    if not CHECKPOINT_DIR.exists():
        return []
    return sorted(
        [d.name for d in CHECKPOINT_DIR.iterdir() if d.is_dir()],
    )


def show_status():
    """Muestra el estado actual del entrenamiento."""
    if not LOG_FILE.exists():
        print(f"Log no encontrado: {LOG_FILE}")
        print("El entrenamiento no ha empezado o no se ha creado el log.")
        return False

    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    if not lines:
        print("Log vacio. Esperando datos...")
        return False

    print("=" * 60)
    print("  MONITOR DE ENTRENAMIENTO - Stable Diffusion Fine-tuning")
    print("=" * 60)

    # Config line
    for line in lines[:5]:
        if "Config:" in line:
            print(f"\n  {line.split('] ')[-1]}")
            break

    # Ultimo batch
    batch_info = parse_last_batch_line(lines)
    if batch_info:
        print(f"\n  Progreso:  {batch_info['pct']}% "
              f"({batch_info['batch']}/{batch_info['total_batches']} batches)")
        print(f"  Loss:      {batch_info['loss']:.4f}")
        print(f"  Velocidad: {batch_info['batch_time']:.1f}s/batch")
        print(f"  ETA:       {batch_info['eta']}")

        # Barra de progreso simple
        filled = batch_info['pct'] // 2
        bar = "#" * filled + "-" * (50 - filled)
        print(f"\n  [{bar}] {batch_info['pct']}%")

    # Ultimo epoch
    epoch_info = parse_last_epoch_line(lines)
    if epoch_info:
        print(f"\n  Ultimo epoch completado: {epoch_info['epoch']}/{epoch_info['total_epochs']}")
        print(f"  Loss promedio: {epoch_info['avg_loss']:.6f}")
        print(f"  Tiempo epoch: {epoch_info['epoch_time']}")

    # Checkpoints
    checkpoints = find_checkpoints()
    if checkpoints:
        print(f"\n  Checkpoints guardados ({len(checkpoints)}):")
        for ckpt in checkpoints:
            print(f"    - {ckpt}")

    # Ultima linea del log (timestamp)
    if lines:
        last = lines[-1]
        ts_match = re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]", last)
        if ts_match:
            print(f"\n  Ultima actualizacion: {ts_match.group(1)}")

    # Completado?
    completed = any("Entrenamiento completado" in l for l in lines)
    if completed:
        print("\n  >>> ENTRENAMIENTO COMPLETADO <<<")

    print("=" * 60)
    return completed


def main():
    parser = argparse.ArgumentParser(description="Monitor de entrenamiento SD")
    parser.add_argument(
        "--follow", "-f",
        action="store_true",
        help="Modo seguimiento: actualiza cada 5 segundos",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Intervalo de actualizacion en segundos (default: 5)",
    )
    args = parser.parse_args()

    if args.follow:
        print("Modo seguimiento activado. Ctrl+C para salir.\n")
        try:
            while True:
                # Limpiar pantalla (compatible Windows/Unix)
                print("\033[2J\033[H", end="")
                completed = show_status()
                if completed:
                    break
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nMonitor detenido.")
    else:
        show_status()


if __name__ == "__main__":
    main()

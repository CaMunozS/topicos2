#!/usr/bin/env python
"""Monitoreo: genera el dashboard HTML de una corrida (puede correr en paralelo al entrenamiento).

  python monitor.py                       # última corrida
  python monitor.py --run runs/<corrida>  # corrida específica
  python monitor.py --watch 30            # regenera cada 30 s (la página se autorrefresca)
  tensorboard --logdir runs               # alternativa: TensorBoard
"""
import os
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Se requiere Python 3.10 o superior (tienes %d.%d). En Mac: brew install python@3.12" % sys.version_info[:2])
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac: operaciones sin soporte MPS caen a CPU en vez de fallar
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import time
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slm.dashboard import build_dashboard  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--watch", type=int, default=0, help="segundos entre actualizaciones (0 = una vez)")
    ap.add_argument("--open", action="store_true", help="abrir en el navegador")
    a = ap.parse_args()
    if a.run:
        run = Path(a.run)
    else:
        runs = [p for p in (ROOT / "runs").glob("*") if (p / "state.json").exists()]
        if not runs:
            raise SystemExit("Aún no hay corridas en runs/.")
        run = max(runs, key=lambda p: p.stat().st_mtime)
    out = build_dashboard(run, refresh_s=a.watch or None)
    print(f"Dashboard: {out}")
    if a.open:
        webbrowser.open(out.resolve().as_uri())
    while a.watch:
        time.sleep(a.watch)
        build_dashboard(run, refresh_s=a.watch)
        print(time.strftime("%H:%M:%S"), "actualizado")


if __name__ == "__main__":
    main()

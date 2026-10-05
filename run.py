#!/usr/bin/env python
"""Punto de entrada: entrena el SLM de forma autónoma usando la API de OpenAI como profesor.

  python run.py                         # corrida real (requiere OPENAI_API_KEY en .env)
  python run.py --simulate              # prueba completa sin clave, sin GPU y sin descargas
  python run.py --resume runs/<corrida> # reanuda una corrida interrumpida
  python run.py --resume-last           # reanuda la última corrida
  python run.py --rounds 3 --budget 5   # límites rápidos por línea de comandos

Para detener de forma ordenada: crea un archivo llamado STOP dentro de la carpeta de la corrida.
"""
import os
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Se requiere Python 3.10 o superior (tienes %d.%d). En Mac: brew install python@3.12" % sys.version_info[:2])
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # Mac: operaciones sin soporte MPS caen a CPU en vez de fallar
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from slm.config import ROOT, load_config, require_api_key  # noqa: E402
from slm.orchestrator import Orchestrator  # noqa: E402
from slm.utils import read_json  # noqa: E402


def last_run(output_dir: Path) -> Path | None:
    runs = [p for p in output_dir.glob("*") if (p / "state.json").exists()]
    return max(runs, key=lambda p: p.stat().st_mtime) if runs else None


def main() -> None:
    ap = argparse.ArgumentParser(description="SLM autónomo: Llama local que aprende de la API de OpenAI")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--simulate", action="store_true", help="sin OpenAI, sin GPU, sin descargas")
    ap.add_argument("--resume", type=str, default=None, help="carpeta de la corrida a reanudar")
    ap.add_argument("--resume-last", action="store_true")
    ap.add_argument("--rounds", type=int, default=None, help="máximo de rondas")
    ap.add_argument("--budget", type=float, default=None, help="tope de gasto en USD")
    ap.add_argument("--model", type=str, default=None, help="id del modelo estudiante en Hugging Face")
    args = ap.parse_args()

    run_dir = None
    simulate = args.simulate
    if args.resume or args.resume_last:
        cfg0 = load_config(args.config)
        run_dir = Path(args.resume) if args.resume else last_run(ROOT / cfg0.run.output_dir)
        if not run_dir or not (run_dir / "state.json").exists():
            raise SystemExit("No encontré una corrida para reanudar.")
        simulate = bool(read_json(run_dir / "state.json", {}).get("simulate", simulate))
        print(f"Reanudando {run_dir} …")

    cfg = load_config(args.config, simulate=simulate)
    if not Path(cfg.run.output_dir).is_absolute():
        cfg.run.output_dir = str(ROOT / cfg.run.output_dir)
    if args.rounds:
        cfg.autonomy.max_rounds = args.rounds
    if args.budget is not None:
        cfg.autonomy.budget_usd = args.budget
    if args.model:
        cfg.student.model_id = args.model
    require_api_key(cfg)

    state = Orchestrator(cfg, run_dir).run()
    sys.exit(0 if state.get("status") in ("finished", "stopped") else 1)


if __name__ == "__main__":
    main()

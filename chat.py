#!/usr/bin/env python
"""Conversa con tu SLM entrenado (usa el mejor checkpoint de la corrida).

  python chat.py                          # última corrida, mejor checkpoint
  python chat.py --run runs/<corrida> --round 3
  python chat.py --compare                # muestra base vs. entrenado lado a lado
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

from slm.config import ROOT as _R, Cfg, _load_dotenv  # noqa: E402
from slm.student import Student  # noqa: E402
from slm.utils import get_logger, read_json  # noqa: E402

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--round", default="best")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=None)
    a = ap.parse_args()
    if a.run:
        run = Path(a.run)
    else:
        runs = [p for p in (ROOT / "runs").glob("*") if (p / "state.json").exists()]
        if not runs:
            raise SystemExit("No hay corridas en runs/. Ejecuta primero: python run.py")
        run = max(runs, key=lambda p: p.stat().st_mtime)
    _load_dotenv(_R / ".env")
    cfg = Cfg.wrap(read_json(run / "config_snapshot.json"))
    import os

    cfg.student["hf_token"] = os.getenv("HF_TOKEN")
    state = read_json(run / "state.json")
    rnd = state["best_round"] if a.round == "best" else int(a.round)
    ckpt = run / "checkpoints" / f"round_{rnd:03d}"
    logger = get_logger("chat")
    st = Student(cfg, run, logger)
    st.load()
    st.load_adapter_weights(ckpt)
    print(f"\nSLM listo · {cfg.student.model_id} + LoRA de la ronda {rnd} ({ckpt}). Escribe 'salir' para terminar.\n")
    while True:
        try:
            q = input("Tú › ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("salir", "exit", "quit"):
            break
        if not q:
            continue
        ans = st.generate([q], max_new_tokens=a.max_new_tokens, batch_size=1)[0]
        print(f"\nSLM (ronda {rnd}) › {ans}\n")
        if a.compare:
            base = st.generate([q], max_new_tokens=a.max_new_tokens, batch_size=1, use_adapter=False)[0]
            print(f"Base sin entrenar › {base}\n")


if __name__ == "__main__":
    main()

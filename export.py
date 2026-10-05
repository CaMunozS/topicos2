#!/usr/bin/env python
"""Exporta el mejor checkpoint como modelo completo (base + LoRA fusionados) listo para usar.

  python export.py                          # última corrida → exports/<corrida>-rX
  python export.py --run runs/<corrida> --round 4

El resultado es un modelo Hugging Face estándar. Para Ollama/LM Studio puedes convertirlo a
GGUF con llama.cpp (convert_hf_to_gguf.py).
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

import torch  # noqa: E402

from slm.config import ROOT as _R, _load_dotenv  # noqa: E402
from slm.utils import read_json  # noqa: E402

_load_dotenv(_R / ".env")

ROOT = Path(__file__).resolve().parent


def main() -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None)
    ap.add_argument("--round", default="best")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.run:
        run = Path(a.run)
    else:
        runs = [p for p in (ROOT / "runs").glob("*") if (p / "state.json").exists()]
        run = max(runs, key=lambda p: p.stat().st_mtime)
    cfg = read_json(run / "config_snapshot.json")
    state = read_json(run / "state.json")
    rnd = state["best_round"] if a.round == "best" else int(a.round)
    ckpt = run / "checkpoints" / f"round_{rnd:03d}"
    model_id = cfg["student"]["model_id"]
    src = str(run / "base_model") if model_id == "tiny-random-llama" else model_id
    out = Path(a.out) if a.out else ROOT / "exports" / f"{run.name}-r{rnd}"
    print(f"Fusionando {model_id} + LoRA ronda {rnd} → {out}")
    tok = AutoTokenizer.from_pretrained(src, token=os.getenv("HF_TOKEN"))
    base = AutoModelForCausalLM.from_pretrained(src, dtype=torch.bfloat16 if model_id != "tiny-random-llama" else torch.float32,
                                                token=os.getenv("HF_TOKEN"))
    merged = PeftModel.from_pretrained(base, str(ckpt)).merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    print("Listo.")


if __name__ == "__main__":
    main()

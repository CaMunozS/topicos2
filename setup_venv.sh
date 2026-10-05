#!/usr/bin/env bash
# Crea el entorno virtual (.venv) e instala todo. macOS (Apple Silicon) y Linux.
set -euo pipefail
cd "$(dirname "$0")"

# 1) Buscar un Python >= 3.10 (en macOS, "python3" del sistema suele ser 3.9)
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3.13 python3.11 python3.10 python3 /opt/homebrew/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)' 2>/dev/null; then
      PY="$c"; break
    fi
  done
fi
if [ -z "$PY" ]; then
  echo "No encontré Python 3.10–3.13."
  echo "  macOS:  brew install python@3.12   (y vuelve a ejecutar ./setup_venv.sh)"
  echo "  Linux:  sudo apt install python3.12 python3.12-venv"
  exit 1
fi
echo ">> Usando $("$PY" --version) ($PY)"

# 2) Entorno virtual limpio
if [ -d .venv ] && ! .venv/bin/python -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
  echo ">> .venv existente con Python antiguo: se recrea"; rm -rf .venv
fi
"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip wheel

# 3) PyTorch según hardware
if command -v nvidia-smi >/dev/null 2>&1; then
  echo ">> GPU NVIDIA detectada: PyTorch con CUDA"
  pip install "torch>=2.5"
elif [[ "$(uname)" == "Darwin" ]]; then
  echo ">> macOS: PyTorch con aceleración MPS (Apple Silicon)"
  pip install "torch>=2.5"
else
  echo ">> Sin GPU: PyTorch para CPU (sirve para --simulate; el modelo de 4B será muy lento)"
  pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cpu || pip install "torch>=2.5"
fi

# 4) Resto de dependencias (versiones probadas juntas)
pip install -r requirements.txt
[ -f .env ] || cp .env.example .env

python - <<'PY'
import torch, transformers, peft
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nListo. torch {torch.__version__} · transformers {transformers.__version__} · peft {peft.__version__} · dispositivo: {dev}")
PY
echo
echo "Siguiente paso:"
echo "  1) Pega tu clave en .env  (OPENAI_API_KEY=sk-...)"
echo "  2) source .venv/bin/activate"
echo "  3) python run.py --simulate   # prueba sin costo (1-3 min)"
echo "  4) python run.py              # entrenamiento real autónomo"

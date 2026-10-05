# Crea el entorno virtual (.venv) e instala todo. Windows (PowerShell).
# Si PowerShell bloquea el script:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
Set-Location $PSScriptRoot

# 1) Buscar Python 3.10–3.13 con el lanzador "py"
$PYV = $null
foreach ($v in @("-3.12", "-3.13", "-3.11", "-3.10")) {
  & py $v -c "import sys" *> $null
  if ($LASTEXITCODE -eq 0) { $PYV = $v; break }
}
if (-not $PYV) { Write-Host "No encontré Python 3.10–3.13. Instálalo desde python.org (marca 'Add to PATH')."; exit 1 }
Write-Host ">> Usando Python $PYV"

& py $PYV -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip wheel

# 2) PyTorch según hardware
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
  Write-Host ">> GPU NVIDIA detectada: PyTorch con CUDA"
  pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu128
  if ($LASTEXITCODE -ne 0) { pip install "torch>=2.5" --index-url https://download.pytorch.org/whl/cu126 }
} else {
  Write-Host ">> Sin GPU NVIDIA: PyTorch para CPU (sirve para --simulate)"
  pip install "torch>=2.5"
}

# 3) Resto de dependencias
pip install -r requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
python -c "import torch, transformers, peft; print('Listo. torch', torch.__version__, '| transformers', transformers.__version__, '| peft', peft.__version__, '| CUDA:', torch.cuda.is_available())"
Write-Host "`nSiguiente paso:`n  1) Pega tu clave en .env`n  2) .\.venv\Scripts\Activate.ps1`n  3) python run.py --simulate`n  4) python run.py"

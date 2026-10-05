# SLM autónomo · un Llama de 4B que aprende de la API de OpenAI

Entrena **de forma automática y autónoma** un modelo pequeño (SLM) que corre en tu máquina,
usando la API de OpenAI como **profesor** (genera el currículo y las respuestas) y como
**juez** (evalúa cada versión). Deja **checkpoints**, **registros de uso y de entrenamiento**
y un **dashboard de monitoreo**.

```
 OpenAI (profesor + juez)                       Tu máquina (venv)
 ┌──────────────────────┐   tareas + respuestas  ┌───────────────────────────────┐
 │ currículo adaptativo │ ─────────────────────▶ │ Llama-3.1-Nemotron-Nano-4B    │
 │ respuestas expertas  │                        │ + LoRA (30,4 M parámetros)     │
 │ juez con rúbrica     │ ◀───────────────────── │ checkpoints · logs · dashboard │
 └──────────────────────┘   respuestas a evaluar └───────────────────────────────┘
```

## 1 · Instalar (una vez)

```bash
# macOS / Linux
./setup_venv.sh
source .venv/bin/activate

# Windows (PowerShell)
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
```

## 2 · Colocar la clave (el único paso obligatorio)

Abre `.env` y completa:

```
OPENAI_API_KEY=sk-...
```

## 3 · Probar sin costo y luego entrenar

```bash
python run.py --simulate     # todo el pipeline sin OpenAI, sin GPU, sin descargas (~1 min)
python run.py                # entrenamiento real autónomo
python monitor.py --watch 30 --open   # dashboard en vivo (en otra terminal)
python chat.py --compare     # conversa con el mejor checkpoint vs. el modelo base
python export.py             # fusiona base + LoRA en un modelo completo
```

Reanudar tras un corte: `python run.py --resume-last` · Detener ordenadamente: crea un archivo
`STOP` dentro de la carpeta de la corrida.

## El ciclo autónomo

| Ronda | Qué ocurre |
|---|---|
| 0 | Taxonomía de temas → **set de evaluación fijo** (nunca se entrena con él) → nota base del estudiante |
| r ≥ 1 | 1) **Currículo adaptativo**: más tareas en los temas con peor nota + las debilidades que detectó el juez |
| | 2) **Profesor** responde cada tarea (salida estructurada JSON) · deduplicación · filtro de calidad |
| | 3) **LoRA**: entrena desde el checkpoint vigente con datos nuevos + repaso (anti-olvido) |
| | 4) **Juez** califica 4 criterios (1–10) contra la respuesta de referencia |
| | 5) **Decide**: PROMOVER (nuevo mejor) · ACEPTAR (dentro de tolerancia) · REVERTIR (vuelve al mejor) |
| fin | Nota objetivo · paciencia · presupuesto USD · máx. rondas · máx. horas · archivo STOP |

## Qué queda registrado (`runs/<corrida>/`)

| Archivo | Contenido |
|---|---|
| `checkpoints/round_XXX/` | adaptador LoRA + `meta.json` (nota, decisión, métricas de entrenamiento) |
| `checkpoints/best/` | copia del mejor checkpoint |
| `data/round_XXX.jsonl`, `data/eval_set.jsonl` | datos generados por el profesor por ronda + set de evaluación |
| `logs/openai_usage.jsonl` | **cada llamada**: propósito, modelo, tokens (entrada, caché, salida, razonamiento), costo USD, latencia, estado |
| `logs/train_metrics.jsonl` | **cada paso**: pérdida, tasa de aprendizaje, norma del gradiente, tokens/s, memoria |
| `logs/eval_metrics.jsonl`, `eval_details/` | nota media, aprobación, por tema, por criterio, por dificultad · detalle por pregunta |
| `logs/rounds.jsonl`, `logs/events.jsonl`, `logs/run.log` | resumen por ronda, bitácora de eventos, log de consola |
| `usage_summary.json`, `state.json`, `system.json` | uso agregado, estado reanudable, hardware |
| `dashboard.html`, `tensorboard/` | monitoreo (`tensorboard --logdir runs`) |

La clave **nunca** se escribe en disco (`config_snapshot.json` la excluye).

## Hardware (estimaciones para el modelo de 4,5 B)

| Modo | Memoria aprox. | Dónde |
|---|---|---|
| QLoRA 4 bits (automático en GPU NVIDIA < 20 GB) | ≈ 7–8 GB | RTX 3060 12 GB, 4070, T4 16 GB |
| LoRA bf16 | ≈ 12–14 GB | RTX 4090/3090 24 GB, A10, L4 |
| Apple Silicon (MPS, bf16) | ≥ 24 GB de memoria unificada | M2/M3/M4 Pro o Max |
| CPU | no recomendado para 4B | solo `--simulate` |

¿Equipo más modesto? Cambia `student.model_id` a `meta-llama/Llama-3.2-1B-Instruct` (requiere `HF_TOKEN`).

## Costo esperado de la API

Con la configuración por defecto (200 tareas por ronda, 40 preguntas de evaluación,
profesor y juez `gpt-6-sol`, currículo `gpt-6-luna`) una ronda cuesta del orden de
**US$ 1,5** y 8 rondas ≈ **US$ 13**. El tope duro es `autonomy.budget_usd` (US$ 20 por defecto).
Verifica precios vigentes en https://openai.com/api/pricing y ajústalos en `config.yaml`.

## Licencias y uso responsable

- Modelo estudiante: NVIDIA Open Model License + Llama 3.1 Community License ("Built with Llama").
- Términos de OpenAI: restringen usar sus salidas para desarrollar modelos que compitan con OpenAI.
  Úsalo para especializar asistentes internos/educativos y revisa los términos vigentes con tu equipo legal.
- No envíes datos personales o confidenciales al profesor sin las autorizaciones correspondientes.

## Estructura

```
run.py  monitor.py  chat.py  export.py  config.yaml  .env  requirements.txt  setup_venv.sh/.ps1
src/slm/
  orchestrator.py  bucle autónomo, estado, promoción/reversión, criterios de parada
  curriculum.py    taxonomía, pesos por debilidad, generación y deduplicación de tareas
  teacher.py       cliente OpenAI: JSON Schema, reintentos, capacidades, costo, presupuesto
  student.py       carga HF, QLoRA/LoRA, formato de chat, generación, checkpoints
  trainer.py       bucle SFT con enmascarado de la instrucción y métricas por paso
  evaluator.py     juez LLM con rúbrica y referencia
  dashboard.py     dashboard HTML autocontenido
  simulate.py      profesor/juez simulados para pruebas
tests/test_smoke.py
```

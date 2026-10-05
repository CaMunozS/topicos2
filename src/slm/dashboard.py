"""DASHBOARD DE MONITOREO: genera runs/<corrida>/dashboard.html (autocontenido, funciona sin internet).

Lee solo los registros (logs/*.jsonl, state.json, usage_summary.json), así que puede
ejecutarse en paralelo mientras el entrenamiento corre:  python monitor.py --watch 30
"""
from __future__ import annotations

import base64
import html
import io
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .utils import fmt_secs, fmt_usd, read_json, read_jsonl  # noqa: E402

# Paleta validada (CVD/contraste) para fondo claro
C1, C2, C3 = "#0099BF", "#4A4FB5", "#D9653B"
INK, INK2, GRID = "#1d1d1f", "#5b5b60", "#e6e6e9"
PURPOSE_COLOR = {"teacher": C1, "judge": C2, "curriculum": C3}
PURPOSE_LABEL = {"teacher": "Profesor (respuestas)", "judge": "Juez (evaluación)", "curriculum": "Currículo (tareas)"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#EAF7FB", "#7FD0E6", "#0099BF", "#00475C"])

from matplotlib import font_manager as _fm  # noqa: E402

_AVAILABLE = {f.name for f in _fm.fontManager.ttflist}
_FONTS = [f for f in ("Carlito", "Calibri", "Segoe UI", "Helvetica Neue", "Arial") if f in _AVAILABLE] + ["DejaVu Sans"]

plt.rcParams.update({
    "font.family": _FONTS, "font.size": 10.5, "axes.edgecolor": GRID,
    "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.titlecolor": INK, "axes.titlelocation": "left", "legend.frameon": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _ema(xs: list[float], alpha: float = 0.15) -> list[float]:
    out, m = [], None
    for x in xs:
        m = x if m is None else alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def load_run(run_dir: Path) -> dict:
    logs = run_dir / "logs"
    return {
        "state": read_json(run_dir / "state.json", {}) or {},
        "system": read_json(run_dir / "system.json", {}) or {},
        "usage": read_json(run_dir / "usage_summary.json", {}) or {},
        "train": read_jsonl(logs / "train_metrics.jsonl"),
        "eval": read_jsonl(logs / "eval_metrics.jsonl"),
        "rounds": read_jsonl(logs / "rounds.jsonl"),
        "calls": read_jsonl(logs / "openai_usage.jsonl"),
        "events": read_jsonl(logs / "events.jsonl"),
        "config": read_json(run_dir / "config_snapshot.json", {}) or {},
    }


# ----------------------------------------------------------------- gráficos
def chart_loss(train: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if not train:
        ax.text(0.5, 0.5, "Sin pasos de entrenamiento aún", ha="center", va="center", color=INK2, transform=ax.transAxes)
        return fig
    x = [r["global_step"] for r in train]
    y = [r["loss"] for r in train]
    ax.plot(x, y, color=C1, alpha=0.3, lw=1.2, label="por paso")
    ax.plot(x, _ema(y), color=C1, lw=2.2, label="media móvil")
    starts: dict[int, int] = {}
    for r in train:
        starts.setdefault(r["round"], r["global_step"])
    for rd, s in starts.items():
        ax.axvline(s, color=GRID, lw=1.2, zorder=0)
        ax.text(s, ax.get_ylim()[1], f" R{rd}", va="top", fontsize=9, color=INK2)
    ax.set_title("Pérdida de entrenamiento (solo tokens de la respuesta)")
    ax.set_xlabel("paso de optimización global")
    ax.set_ylabel("pérdida (entropía cruzada)")
    ax.legend(loc="upper right")
    return fig


def chart_score(evals: list[dict], rounds: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if not evals:
        ax.text(0.5, 0.5, "Sin evaluaciones aún", ha="center", va="center", color=INK2, transform=ax.transAxes)
        return fig
    dec = {r["round"]: r["decision"] for r in rounds}
    xs = [e["round"] for e in evals]
    ys = [e["mean_score"] for e in evals]
    best, bests = -1, []
    for x, y in zip(xs, ys):
        if dec.get(x) in ("base", "promovido"):
            best = y
        bests.append(best)
    ax.step(xs, bests, where="post", color=INK2, lw=1.4, ls="--", label="mejor checkpoint")
    ax.plot(xs, ys, color=C1, lw=2, zorder=2)
    for x, y in zip(xs, ys):
        d_ = dec.get(x)
        if d_ == "revertido":
            ax.scatter([x], [y], s=70, zorder=3, marker="X", color=C3, edgecolor="white", linewidth=1)
        else:
            ax.scatter([x], [y], s=70, zorder=3, color=C1 if d_ in ("base", "promovido") else "white",
                       edgecolor=C1, linewidth=2)
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 10.5)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"R{x}" for x in xs])
    ax.set_title("Nota del juez por ronda (● promovido · ○ aceptado · ✕ revertido)")
    ax.set_ylabel("nota media (1–10)")
    ax.legend(loc="lower right")
    return fig


def chart_pass(evals: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if evals:
        xs = [f"R{e['round']}" for e in evals]
        ys = [100 * e["pass_rate"] for e in evals]
        bars = ax.bar(xs, ys, color=C1, width=0.6)
        for b, v in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9, color=INK)
        ax.set_ylim(0, 108)
    ax.set_title("Tasa de aprobación (nota ≥ umbral)")
    ax.set_ylabel("% de preguntas")
    ax.grid(axis="x", visible=False)
    return fig


def chart_topics(evals: list[dict]):
    topics = sorted({t for e in evals for t in e.get("by_topic", {})})
    fig, ax = plt.subplots(figsize=(6.0, max(3.6, 0.42 * len(topics) + 1.2)))
    if not topics:
        return fig
    rounds = [e["round"] for e in evals]
    M = [[e["by_topic"].get(t, float("nan")) for e in evals] for t in topics]
    im = ax.imshow(M, cmap=SEQ, vmin=1, vmax=10, aspect="auto")
    for i, row in enumerate(M):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if v > 6.2 else INK)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_yticks(range(len(topics)))
    ax.set_yticklabels(topics)
    ax.grid(False)
    ax.set_title("Nota por tema y ronda (el currículo refuerza los temas más débiles)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    return fig


def chart_cost(calls: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    cum: dict[str, list[float]] = defaultdict(list)
    idx: dict[str, list[int]] = defaultdict(list)
    tot = defaultdict(float)
    for i, c in enumerate(calls, 1):
        p = c.get("purpose", "otro")
        tot[p] += c.get("cost_usd", 0) or 0
        cum[p].append(tot[p])
        idx[p].append(i)
    for p in ("teacher", "judge", "curriculum"):
        if p in cum:
            ax.plot(idx[p], cum[p], color=PURPOSE_COLOR[p], lw=2, label=PURPOSE_LABEL[p])
            ax.annotate(f"US$ {cum[p][-1]:.2f}", (idx[p][-1], cum[p][-1]), textcoords="offset points",
                        xytext=(4, 0), fontsize=9, color=INK, va="center")
    ax.set_title("Costo acumulado de la API por propósito")
    ax.set_xlabel("n.º de llamada")
    ax.set_ylabel("USD acumulados")
    ax.legend(loc="upper left")
    return fig


def chart_tokens_round(calls: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    agg: dict[tuple[int, str], float] = defaultdict(float)
    for c in calls:
        agg[(c.get("round", 0), c.get("purpose"))] += ((c.get("input_tokens") or 0) + (c.get("output_tokens") or 0)) / 1000
    rounds = sorted({r for r, _ in agg})
    w = 0.26
    for k, p in enumerate(("teacher", "judge", "curriculum")):
        vals = [agg.get((r, p), 0) for r in rounds]
        ax.bar([i + (k - 1) * w for i in range(len(rounds))], vals, width=w - 0.03, color=PURPOSE_COLOR[p],
               label=PURPOSE_LABEL[p].split(" ")[0])
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_title("Miles de tokens por ronda")
    ax.legend(fontsize=8.5)
    ax.grid(axis="x", visible=False)
    return fig


def chart_lr(train: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if train:
        ax.plot([r["global_step"] for r in train], [r["lr"] for r in train], color=C2, lw=1.8)
    ax.set_title("Tasa de aprendizaje (coseno con calentamiento)")
    ax.set_xlabel("paso global")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    return fig


def chart_throughput(train: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    if train:
        ax.plot([r["global_step"] for r in train], [r["tokens_per_s"] for r in train], color=C1, lw=1.4)
    ax.set_title("Rendimiento: tokens entrenados por segundo")
    ax.set_xlabel("paso global")
    return fig


def chart_latency(calls: list[dict]):
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    data = [[c["latency_s"] for c in calls if c.get("purpose") == p and c.get("status") == "ok"] for p in
            ("teacher", "judge", "curriculum")]
    labels = ["Profesor", "Juez", "Currículo"]
    keep = [(d, l, PURPOSE_COLOR[p]) for d, l, p in zip(data, labels, ("teacher", "judge", "curriculum")) if d]
    if keep:
        bp = ax.boxplot([k[0] for k in keep], tick_labels=[k[1] for k in keep], patch_artist=True, widths=0.5,
                        medianprops={"color": INK})
        for patch, (_, _, col) in zip(bp["boxes"], keep):
            patch.set_facecolor(col)
            patch.set_alpha(0.55)
    ax.set_title("Latencia por llamada (s)")
    ax.grid(axis="x", visible=False)
    return fig


# --------------------------------------------------------------------- HTML
CSS = """
:root{--ink:#1d1d1f;--ink2:#5b5b60;--line:#e6e6e9;--c1:#0099BF;--bg:#f6f7f9}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 Carlito,Calibri,'Segoe UI',system-ui,sans-serif}
header{background:#0b0b0c;color:#fff;padding:22px 28px;border-left:10px solid var(--c1)}
header h1{margin:0;font-size:26px}header p{margin:4px 0 0;color:#b9bcc2}
main{padding:22px 28px;max-width:1500px;margin:auto}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:18px}
.kpi{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.kpi b{display:block;font-size:24px}.kpi span{color:var(--ink2);font-size:13px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(460px,1fr));gap:14px}
.card{background:#fff;border:1px solid var(--line);border-radius:10px;padding:12px}
.card img{width:100%;height:auto}
h2{font-size:18px;margin:26px 0 10px;border-left:5px solid var(--c1);padding-left:8px}
table{border-collapse:collapse;width:100%;background:#fff;font-size:13.5px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{background:#f0f2f5}.tag{padding:1px 8px;border-radius:9px;font-size:12px;border:1px solid}
.ok{color:#00708c;border-color:#00708c}.no{color:#8a8a90;border-color:#b5b5ba}
.mono{font-family:ui-monospace,Consolas,monospace;font-size:12.5px}
.sim{background:#fff4e5;border:1px solid #f0c27a;padding:8px 12px;border-radius:8px;margin-bottom:14px}
"""


def build_dashboard(run_dir: Path, refresh_s: int | None = None) -> Path:
    run_dir = Path(run_dir)
    d = load_run(run_dir)
    st, usage, evals, rounds, calls, train = d["state"], d["usage"], d["eval"], d["rounds"], d["calls"], d["train"]
    tot = usage.get("total", {})
    trn = usage.get("training", {})
    base = evals[0]["mean_score"] if evals else None
    best = st.get("best_score")
    n_samples = sum(r.get("new_samples") or 0 for r in rounds)
    kpis = [
        ("Mejor nota (1–10)", f"{best:.2f}" if best is not None else "–"),
        ("Mejora vs. base", f"{best - base:+.2f}" if best is not None and base is not None else "–"),
        ("Mejor checkpoint", f"ronda {st.get('best_round')}" if st.get("best_round") is not None else "–"),
        ("Rondas evaluadas", str(max(len(evals) - 1, 0))),
        ("Ejemplos del profesor", f"{n_samples:,}"),
        ("Gasto API", fmt_usd(tot.get("cost_usd", 0))),
        ("Llamadas API", f"{int(tot.get('calls', 0)):,}"),
        ("Tokens API (entrada / salida)", f"{int(tot.get('input_tokens', 0))/1e3:,.0f}k / {int(tot.get('output_tokens', 0))/1e3:,.0f}k"),
        ("Tiempo de entrenamiento", fmt_secs(trn.get("train_seconds", 0))),
        ("Memoria pico", f"{trn.get('peak_mem_gb', 0):.1f} GB"),
        ("Estado", f"{st.get('status', '–')}"),
    ]
    charts = [
        _png(chart_score(evals, rounds)), _png(chart_pass(evals)),
        _png(chart_loss(train)), _png(chart_tokens_round(calls)),
        _png(chart_topics(evals)), _png(chart_lr(train)),
        _png(chart_cost(calls)), _png(chart_latency(calls)), _png(chart_throughput(train)),
    ]
    e = html.escape
    rows_html = "".join(
        f"<tr><td>R{r['round']}</td><td><span class='tag {'ok' if r['decision'] in ('promovido','base','aceptado') else 'no'}'>{e(r['decision'])}</span></td>"
        f"<td>{r['mean_score']:.2f}</td><td>{(r.get('delta') or 0):+.2f}</td><td>{100*r['pass_rate']:.0f}%</td>"
        f"<td>{r.get('new_samples') or 0}</td><td>{r.get('train_samples') or 0}</td><td>{r.get('loss_last') or '–'}</td>"
        f"<td>{fmt_usd(r.get('round_cost_usd') or 0)}</td><td>{fmt_secs(r.get('round_seconds') or 0)}</td>"
        f"<td class='mono'>checkpoints/round_{r['round']:03d}</td></tr>" for r in rounds)
    events_html = "".join(f"<tr><td class='mono'>{e(str(ev.get('ts',''))[11:19])}</td><td>{e(ev.get('kind',''))}</td>"
                          f"<td>{e(ev.get('msg',''))}</td></tr>" for ev in d["events"][-18:][::-1])
    worst_html = ""
    last_r = evals[-1]["round"] if evals else None
    if last_r is not None:
        det = read_jsonl(run_dir / "eval_details" / f"round_{last_r:03d}.jsonl")
        for x in sorted(det, key=lambda x: x["nota_global"])[:6]:
            worst_html += (f"<tr><td>{x['nota_global']}</td><td>{e(x['topic'])}</td><td>{e(x['instruction'][:160])}</td>"
                           f"<td>{e((x['student_answer'] or '')[:220])}</td><td>{e(x['retroalimentacion'])}</td></tr>")
    usage_rows = "".join(
        f"<tr><td>{e(k.split(':',1)[1])}</td><td>{int(v['calls'])}</td><td>{int(v['input_tokens']):,}</td>"
        f"<td>{int(v['cached_tokens']):,}</td><td>{int(v['output_tokens']):,}</td><td>{int(v['reasoning_tokens']):,}</td>"
        f"<td>{fmt_usd(v['cost_usd'])}</td><td>{int(v['errors'])}</td></tr>"
        for k, v in usage.items() if k.startswith(("purpose:", "model:")))
    sysinfo = " · ".join(f"{k}: {v}" for k, v in d["system"].items())
    sim = ("<div class='sim'><b>Corrida en MODO SIMULACIÓN.</b> Profesor y juez simulados localmente (sin OpenAI); "
           "los costos son estimaciones con la tabla de precios. Sirve para validar el pipeline, no la calidad.</div>"
           if st.get("simulate") else "")
    meta = f"<meta http-equiv='refresh' content='{refresh_s}'>" if refresh_s else ""
    page = f"""<!doctype html><html lang="es"><head><meta charset="utf-8">{meta}
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Monitor SLM · {e(run_dir.name)}</title>
<style>{CSS}</style></head><body>
<header><h1>Monitor del SLM autónomo · {e(run_dir.name)}</h1>
<p>Estudiante: {e(str(d['config'].get('student', {}).get('model_id', '–')))} · {e(sysinfo)}</p></header><main>{sim}
<div class="kpis">{''.join(f"<div class='kpi'><b>{e(v)}</b><span>{e(k)}</span></div>" for k, v in kpis)}</div>
<h2>Calidad</h2><div class="grid"><div class="card"><img src="data:image/png;base64,{charts[0]}"></div>
<div class="card"><img src="data:image/png;base64,{charts[1]}"></div>
<div class="card"><img src="data:image/png;base64,{charts[4]}"></div></div>
<h2>Entrenamiento</h2><div class="grid"><div class="card"><img src="data:image/png;base64,{charts[2]}"></div>
<div class="card"><img src="data:image/png;base64,{charts[5]}"></div><div class="card"><img src="data:image/png;base64,{charts[8]}"></div></div>
<h2>Uso y costo de la API de OpenAI</h2><div class="grid"><div class="card"><img src="data:image/png;base64,{charts[6]}"></div>
<div class="card"><img src="data:image/png;base64,{charts[3]}"></div><div class="card"><img src="data:image/png;base64,{charts[7]}"></div></div>
<h2>Rondas y checkpoints</h2><table><tr><th>Ronda</th><th>Decisión</th><th>Nota</th><th>Δ vs mejor</th><th>Aprob.</th>
<th>Nuevos</th><th>Entrenados</th><th>Pérdida final</th><th>Costo ronda</th><th>Duración</th><th>Checkpoint</th></tr>{rows_html}</table>
<h2>Detalle de uso por propósito y modelo</h2><table><tr><th>Clave</th><th>Llamadas</th><th>Tokens entrada</th><th>En caché</th>
<th>Tokens salida</th><th>Razonamiento</th><th>Costo</th><th>Errores/reintentos</th></tr>{usage_rows}</table>
<h2>Peores respuestas de la última evaluación (insumo del próximo currículo)</h2><table><tr><th>Nota</th><th>Tema</th>
<th>Instrucción</th><th>Respuesta del estudiante</th><th>Retroalimentación del juez</th></tr>{worst_html}</table>
<h2>Bitácora de eventos</h2><table><tr><th>Hora</th><th>Tipo</th><th>Evento</th></tr>{events_html}</table>
</main></body></html>"""
    out = run_dir / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out

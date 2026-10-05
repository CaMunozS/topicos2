"""EVALUACIÓN con juez LLM (LLM-as-a-judge guiado por referencia).

- Set de evaluación FIJO y separado (nunca se entrena con él) → las notas son comparables entre rondas.
- El juez recibe: instrucción, respuesta de referencia del profesor y respuesta del estudiante.
- Rúbrica de 4 criterios (1–10) + nota global + retroalimentación breve.
- La retroalimentación de las peores respuestas alimenta el currículo de la ronda siguiente.
"""
from __future__ import annotations

import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from .teacher import BudgetExceeded

CRITERIA = ["correccion", "completitud", "claridad", "idioma_formato"]

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        **{c: {"type": "integer"} for c in CRITERIA},
        "nota_global": {"type": "integer"},
        "retroalimentacion": {"type": "string"},
    },
    "required": CRITERIA + ["nota_global", "retroalimentacion"],
    "additionalProperties": False,
}

JUDGE_SYSTEM = (
    "Eres un evaluador riguroso e imparcial de asistentes de IA. Comparas la respuesta de un ESTUDIANTE "
    "con una respuesta de REFERENCIA de alta calidad. Calificas de 1 a 10 cada criterio:\n"
    "- correccion: exactitud factual y técnica; penaliza fuerte cualquier error o invención.\n"
    "- completitud: cubre los puntos esenciales de la referencia.\n"
    "- claridad: estructura, concisión y utilidad práctica.\n"
    "- idioma_formato: responde en el idioma pedido, con formato adecuado y sin texto basura.\n"
    "nota_global: juicio integral de 1 a 10 (7 = aceptable para un usuario real). "
    "Evita premiar la longitud por sí misma. Si la respuesta está vacía, incoherente o en otro idioma, nota_global ≤ 2. "
    "retroalimentacion: una oración concreta sobre la principal debilidad."
)


def judge_prompt(item: dict, student_answer: str, max_chars: int) -> str:
    ans = (student_answer or "").strip()[:max_chars] or "(respuesta vacía)"
    return (f"INSTRUCCIÓN:\n{item['instruction']}\n\n"
            f"RESPUESTA DE REFERENCIA:\n{item['response']}\n\n"
            f"RESPUESTA DEL ESTUDIANTE:\n{ans}")


def _clamp(x) -> int:
    try:
        return max(1, min(10, int(x)))
    except Exception:
        return 1


def evaluate(student, llm, cfg, eval_set: list[dict], round_idx: int, logger) -> tuple[dict, list[dict]]:
    """Genera con el estudiante y califica con el juez. Devuelve (resumen, detalle por ítem)."""
    ecfg = cfg.evaluation
    logger.info(f"Ronda {round_idx}: el estudiante responde {len(eval_set)} preguntas del set de evaluación…")
    answers = student.generate([x["instruction"] for x in eval_set])
    details: list[dict] = [None] * len(eval_set)  # type: ignore[list-item]

    def _job(i: int) -> dict:
        item = eval_set[i]
        data = llm.json_call("judge", cfg.openai.models.judge, system=JUDGE_SYSTEM,
                             user=judge_prompt(item, answers[i], int(ecfg.max_answer_chars_for_judge)),
                             schema_name="evaluacion", schema=JUDGE_SCHEMA,
                             max_output_tokens=int(ecfg.judge_max_output_tokens))
        row = {"round": round_idx, "id": item["id"], "topic": item["topic"], "difficulty": item["difficulty"],
               "instruction": item["instruction"], "student_answer": answers[i],
               "answer_chars": len(answers[i] or "")}
        for c in CRITERIA + ["nota_global"]:
            row[c] = _clamp(data.get(c))
        row["retroalimentacion"] = (data.get("retroalimentacion") or "").strip()
        return row

    logger.info(f"Ronda {round_idx}: el juez ({cfg.openai.models.judge}) califica las respuestas…")
    with ThreadPoolExecutor(max_workers=int(cfg.openai.max_workers)) as ex:
        futs = {ex.submit(_job, i): i for i in range(len(eval_set))}
        for f in as_completed(futs):
            i = futs[f]
            try:
                details[i] = f.result()
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Ítem {i} sin calificación del juez ({e}); se omite del promedio.")
    rows = [d for d in details if d]
    if not rows:
        raise RuntimeError("El juez no pudo calificar ningún ítem.")
    return summarize(rows, round_idx, float(ecfg.pass_threshold)), rows


def summarize(rows: list[dict], round_idx: int, pass_threshold: float) -> dict:
    scores = [r["nota_global"] for r in rows]
    by_topic: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_topic[r["topic"]].append(r["nota_global"])
    worst = sorted(rows, key=lambda r: r["nota_global"])[: max(3, len(rows) // 5)]
    weaknesses: dict[str, list[str]] = defaultdict(list)
    for r in worst:
        if r["retroalimentacion"]:
            weaknesses[r["topic"]].append(r["retroalimentacion"])
    return {
        "round": round_idx,
        "n": len(rows),
        "mean_score": round(statistics.mean(scores), 4),
        "median_score": statistics.median(scores),
        "std_score": round(statistics.pstdev(scores), 4),
        "pass_rate": round(sum(s >= pass_threshold for s in scores) / len(scores), 4),
        "criteria": {c: round(statistics.mean(r[c] for r in rows), 3) for c in CRITERIA},
        "by_topic": {t: round(statistics.mean(v), 3) for t, v in sorted(by_topic.items())},
        "by_difficulty": {d: round(statistics.mean(r["nota_global"] for r in rows if r["difficulty"] == d), 3)
                          for d in sorted({r["difficulty"] for r in rows})},
        "mean_answer_chars": round(statistics.mean(r["answer_chars"] for r in rows), 1),
        "weaknesses": dict(weaknesses),
    }

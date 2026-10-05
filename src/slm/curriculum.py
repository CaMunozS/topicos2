"""CURRÍCULO ADAPTATIVO: el profesor decide qué enseñar en cada ronda.

1. Taxonomía de temas (desde config.yaml o propuesta por el profesor).
2. Pesos por debilidad: los temas donde el estudiante sacó peor nota reciben más tareas.
3. Generación de tareas nuevas (instrucciones) con salida estructurada + deduplicación.
4. El profesor responde cada tarea → pares (instrucción, respuesta) para entrenar.
"""
from __future__ import annotations

import math
import random
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from .teacher import BudgetExceeded
from .utils import NearDuplicateIndex

DIFFICULTIES = ["basica", "intermedia", "avanzada"]

TAXONOMY_SCHEMA = {
    "type": "object",
    "properties": {"topics": {"type": "array", "items": {"type": "string"}}},
    "required": ["topics"],
    "additionalProperties": False,
}

TASKS_SCHEMA = {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "difficulty": {"type": "string", "enum": DIFFICULTIES},
                    "instruction": {"type": "string"},
                },
                "required": ["difficulty", "instruction"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tasks"],
    "additionalProperties": False,
}

ANSWER_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


def propose_topics(llm, cfg) -> list[str]:
    if cfg.domain.get("topics"):
        return list(cfg.domain.topics)
    data = llm.json_call(
        "curriculum", cfg.openai.models.curriculum,
        system="Eres un diseñador instruccional experto. Devuelves taxonomías de temas concisas.",
        user=(f"Dominio del asistente: {cfg.domain.description}\n"
              f"Propón entre 6 y 10 temas que cubran el dominio, sin solaparse. "
              f"Cada tema en 2 a 6 palabras, en {cfg.domain.language}."),
        schema_name="taxonomia", schema=TAXONOMY_SCHEMA, max_output_tokens=800,
    )
    return [t.strip() for t in data["topics"] if t.strip()][:10]


def topic_weights(topics: list[str], topic_scores: dict[str, float] | None, gamma: float, floor: float) -> dict[str, float]:
    """peso_t ∝ max(floor, ((10 − nota_t)/10)^gamma). Sin evaluación previa → uniforme."""
    if not topic_scores:
        return {t: 1 / len(topics) for t in topics}
    raw = {}
    for t in topics:
        score = topic_scores.get(t, 5.0)
        weakness = max(0.0, (10.0 - score) / 10.0)
        raw[t] = max(floor, weakness ** gamma)
    z = sum(raw.values())
    return {t: v / z for t, v in raw.items()}


def allocate(n: int, weights: dict[str, float]) -> dict[str, int]:
    """Reparto entero por restos mayores (suma exacta = n)."""
    exact = {t: n * w for t, w in weights.items()}
    alloc = {t: math.floor(v) for t, v in exact.items()}
    rest = n - sum(alloc.values())
    for t in sorted(exact, key=lambda t: exact[t] - alloc[t], reverse=True)[:rest]:
        alloc[t] += 1
    return alloc


def _task_prompt(cfg, topic: str, k: int, examples: list[str], weaknesses: list[str], purpose: str) -> str:
    mix = cfg.curriculum.difficulty_mix
    lines = [
        f"Dominio del asistente: {cfg.domain.description}",
        f"Idioma de las tareas: {cfg.domain.language}.",
        f"Tema: {topic}",
        f"Genera exactamente {k} tareas NUEVAS y variadas que un usuario real le pediría a este asistente sobre el tema.",
        "Varía el formato: explicaciones, comparaciones, casos aplicados, pasos a seguir, diagnósticos de errores, ejemplos concretos.",
        f"Mezcla de dificultad aproximada: básica {mix.basica:.0%}, intermedia {mix.intermedia:.0%}, avanzada {mix.avanzada:.0%}.",
        "Cada instrucción debe ser autocontenida (una a tres oraciones) y responderse en menos de 250 palabras.",
    ]
    if purpose == "eval":
        lines.append("Estas tareas formarán un EXAMEN: que sean representativas y discriminen calidad.")
    if weaknesses:
        lines.append("El estudiante mostró estas debilidades en la última evaluación; incluye tareas que las ejerciten:")
        lines += [f"- {w}" for w in weaknesses[:8]]
    if examples:
        lines.append("Evita repetir o parafrasear estas tareas ya existentes:")
        lines += [f"- {e[:160]}" for e in examples[-15:]]
    return "\n".join(lines)


def generate_tasks(llm, cfg, topics_alloc: dict[str, int], dedup: NearDuplicateIndex,
                   existing_by_topic: dict[str, list[str]], weaknesses: dict[str, list[str]],
                   purpose: str, logger) -> list[dict]:
    """Pide tareas al profesor por tema, en lotes paralelos, y descarta casi-duplicados."""
    per_call = int(cfg.curriculum.tasks_per_call)
    jobs = []
    for topic, k in topics_alloc.items():
        while k > 0:
            n = min(per_call, k)
            jobs.append((topic, n))
            k -= n
    tasks: list[dict] = []

    def _job(topic: str, n: int) -> list[dict]:
        data = llm.json_call(
            "curriculum", cfg.openai.models.curriculum,
            system="Eres un diseñador de currículos para entrenar asistentes de IA. Escribes tareas claras, realistas y diversas.",
            user=_task_prompt(cfg, topic, n, existing_by_topic.get(topic, []), weaknesses.get(topic, []), purpose),
            schema_name="tareas", schema=TASKS_SCHEMA,
            max_output_tokens=int(cfg.curriculum.max_output_tokens),
        )
        return [{"topic": topic, **t} for t in data.get("tasks", [])][:n]

    with ThreadPoolExecutor(max_workers=int(cfg.openai.max_workers)) as ex:
        futs = [ex.submit(_job, t, n) for t, n in jobs]
        for f in as_completed(futs):
            try:
                batch = f.result()
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Lote de tareas descartado: {e}")
                continue
            for t in batch:
                instr = (t.get("instruction") or "").strip()
                if len(instr) < 12 or dedup.is_duplicate(instr):
                    continue
                dedup.add(instr)
                diff = t.get("difficulty") if t.get("difficulty") in DIFFICULTIES else "intermedia"
                tasks.append({"id": uuid.uuid4().hex[:12], "topic": t["topic"], "difficulty": diff, "instruction": instr})
    return tasks


def teacher_system_prompt(cfg) -> str:
    return (
        f"Eres el PROFESOR experto de un asistente de IA. Dominio: {cfg.domain.description}\n"
        f"Responde en {cfg.domain.language}. Estilo: {cfg.domain.style}\n"
        "Da la mejor respuesta posible: correcta, completa, concreta y bien estructurada, "
        f"en un máximo de {cfg.curriculum.max_answer_words} palabras. Sin preámbulos ni despedidas."
    )


def answer_tasks(llm, cfg, tasks: list[dict], round_idx: int, logger) -> list[dict]:
    """El profesor responde cada tarea en paralelo → registros de entrenamiento/evaluación."""
    system = teacher_system_prompt(cfg)
    min_c, max_c = int(cfg.curriculum.min_answer_chars), int(cfg.curriculum.max_answer_chars)
    out: list[dict] = []

    def _job(t: dict) -> dict | None:
        data = llm.json_call(
            "teacher", cfg.openai.models.teacher, system=system, user=t["instruction"],
            schema_name="respuesta", schema=ANSWER_SCHEMA,
            max_output_tokens=int(cfg.curriculum.answer_max_output_tokens),
        )
        ans = (data.get("answer") or "").strip()
        if not (min_c <= len(ans) <= max_c):
            return None
        return {**t, "response": ans, "round": round_idx, "teacher_model": cfg.openai.models.teacher}

    with ThreadPoolExecutor(max_workers=int(cfg.openai.max_workers)) as ex:
        futs = [ex.submit(_job, t) for t in tasks]
        for f in as_completed(futs):
            try:
                r = f.result()
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Respuesta del profesor descartada: {e}")
                continue
            if r:
                out.append(r)
    dropped = len(tasks) - len(out)
    if dropped:
        logger.info(f"Filtro de calidad: {dropped} respuestas descartadas (largo fuera de rango o error).")
    out.sort(key=lambda r: (r["topic"], r["id"]))
    return out


def build_training_mix(new: list[dict], pool_old: list[dict], replay_ratio: float, seed: int) -> list[dict]:
    """Datos nuevos + repaso (replay) de rondas anteriores para evitar olvido catastrófico."""
    rng = random.Random(seed)
    k = min(len(pool_old), int(round(replay_ratio * len(new))))
    mix = list(new) + rng.sample(pool_old, k) if k else list(new)
    rng.shuffle(mix)
    return mix

"""MODO SIMULACIÓN: un 'profesor' local, determinista y sin red.

Permite verificar el pipeline completo (currículo → respuestas → LoRA → juez → checkpoints →
monitoreo) sin clave de OpenAI y sin GPU. El juez simulado califica por solapamiento de
contenido con la referencia (F1 de palabras), así que las notas SÍ reflejan lo aprendido,
pero NO equivalen a la calidad que mediría un juez real.
"""
from __future__ import annotations

import random
import re
from collections import Counter

from .utils import normalize_text

KB: dict[str, dict[str, tuple[str, str, str]]] = {
    "Fundamentos de LLM": {
        "transformer": ("Un transformer es una arquitectura de red neuronal basada en atención que procesa secuencias en paralelo.",
                        "Reemplazó a las redes recurrentes porque escala mejor con datos y cómputo.",
                        "sirve como base de los asistentes que leen documentos y redactan respuestas"),
        "atención": ("La atención es el mecanismo que pondera qué tokens del contexto son relevantes para cada token generado.",
                     "Su costo crece de forma cuadrática con el largo de la secuencia.",
                     "permite relacionar una consulta con la parte exacta de un reglamento"),
        "tokenización": ("La tokenización divide el texto en unidades llamadas tokens antes de entrar al modelo.",
                         "El costo y el límite de contexto se miden en tokens, no en palabras.",
                         "ayuda a estimar el costo mensual de procesar miles de solicitudes"),
        "ventana de contexto": ("La ventana de contexto es la cantidad máxima de tokens que el modelo considera a la vez.",
                                "Si la información no cabe, el modelo no puede usarla.",
                                "obliga a resumir o recuperar solo los documentos pertinentes"),
        "temperatura": ("La temperatura controla la aleatoriedad del muestreo al generar texto.",
                        "Valores bajos dan respuestas estables y valores altos dan respuestas creativas.",
                        "conviene usar temperatura baja en respuestas normativas"),
    },
    "Ingeniería de prompts": {
        "few-shot": ("Few-shot consiste en incluir algunos ejemplos resueltos dentro del prompt.",
                     "El modelo imita el patrón de los ejemplos sin reentrenarse.",
                     "mejora la clasificación de reclamos con pocos casos de muestra"),
        "cadena de pensamiento": ("La cadena de pensamiento pide al modelo razonar por pasos antes de responder.",
                                  "Mejora problemas de varios pasos a cambio de más tokens.",
                                  "es útil para cálculos de cuotas o plazos"),
        "rol de sistema": ("El rol de sistema fija la identidad, el tono y las reglas del asistente.",
                           "Tiene prioridad sobre las instrucciones del usuario.",
                           "asegura que el asistente respete las políticas internas"),
        "formato de salida": ("Definir el formato de salida indica la estructura exacta que debe tener la respuesta.",
                              "Un esquema JSON facilita integrar la respuesta con otros sistemas.",
                              "permite cargar resultados directo en un tablero"),
        "instrucciones explícitas": ("Las instrucciones explícitas describen la tarea, el público y los criterios de éxito.",
                                     "Reducen la ambigüedad y las respuestas genéricas.",
                                     "evitan respuestas vagas a clientes"),
    },
    "RAG": {
        "embeddings": ("Un embedding es un vector numérico que representa el significado de un texto.",
                       "Textos parecidos quedan cerca en el espacio vectorial.",
                       "permite encontrar documentos similares a la pregunta del cliente"),
        "base vectorial": ("Una base vectorial almacena embeddings y busca los más cercanos a una consulta.",
                           "Es el motor de recuperación de un sistema RAG.",
                           "indexa manuales y políticas para responder con evidencia"),
        "chunking": ("El chunking divide los documentos en fragmentos antes de indexarlos.",
                     "Fragmentos muy grandes diluyen la relevancia y muy pequeños pierden contexto.",
                     "se ajusta según la estructura de cada documento"),
        "reranking": ("El reranking reordena los fragmentos recuperados con un modelo más preciso.",
                      "Aumenta la precisión de los primeros resultados.",
                      "deja arriba la cláusula que realmente responde la duda"),
        "alucinación": ("Una alucinación es una respuesta fluida pero falsa o sin respaldo.",
                        "RAG la reduce al obligar al modelo a citar fuentes.",
                        "se controla exigiendo citas y validando contra documentos"),
    },
    "Agentes de IA": {
        "herramientas": ("Las herramientas son funciones externas que el agente puede invocar, como APIs o bases de datos.",
                         "Extienden al modelo más allá del texto.",
                         "permiten consultar saldos o crear tickets"),
        "planificación": ("La planificación descompone un objetivo en pasos ordenados antes de ejecutarlos.",
                          "Reduce errores en tareas largas.",
                          "organiza un flujo de varias aprobaciones"),
        "memoria": ("La memoria guarda información de interacciones previas para usarla después.",
                    "Puede ser de corto plazo en el contexto o de largo plazo en una base externa.",
                    "recuerda las preferencias de cada cliente"),
        "bucle ReAct": ("ReAct alterna razonamiento y acción: el agente piensa, usa una herramienta y observa el resultado.",
                        "Repite el ciclo hasta resolver la tarea.",
                        "resuelve consultas que requieren varios datos"),
        "guardrails": ("Los guardrails son controles que limitan lo que el agente puede decir o hacer.",
                       "Incluyen validaciones, listas de acciones permitidas y aprobación humana.",
                       "evitan operaciones riesgosas sin supervisión"),
    },
    "Ajuste fino": {
        "LoRA": ("LoRA entrena matrices de bajo rango añadidas al modelo y deja congelados los pesos originales.",
                 "Entrena menos del uno por ciento de los parámetros.",
                 "permite especializar un modelo con una sola GPU"),
        "QLoRA": ("QLoRA combina LoRA con un modelo base cuantizado a cuatro bits.",
                  "Reduce la memoria necesaria a cerca de un tercio.",
                  "hace posible ajustar un modelo de cuatro mil millones de parámetros en una GPU de consumo"),
        "destilación": ("La destilación transfiere el conocimiento de un modelo grande profesor a uno pequeño estudiante.",
                        "El estudiante aprende de las respuestas del profesor.",
                        "produce un modelo local más barato y rápido"),
        "sobreajuste": ("El sobreajuste ocurre cuando el modelo memoriza el entrenamiento y generaliza mal.",
                        "Se detecta con un set de evaluación separado.",
                        "se evita con más datos variados y menos épocas"),
        "olvido catastrófico": ("El olvido catastrófico es la pérdida de habilidades previas al entrenar con datos nuevos.",
                                "Se mitiga repasando datos antiguos en cada ronda.",
                                "protege lo aprendido en rondas anteriores"),
    },
    "Evaluación y LLMOps": {
        "LLM como juez": ("Un LLM como juez califica respuestas con una rúbrica y una referencia.",
                          "Escala la evaluación pero puede tener sesgos de largo y de posición.",
                          "mide la calidad de cada versión del modelo"),
        "set de evaluación": ("El set de evaluación es un conjunto fijo de preguntas que nunca se usa para entrenar.",
                              "Hace comparables las notas entre versiones.",
                              "verifica si cada ronda mejora o empeora"),
        "monitoreo": ("El monitoreo registra métricas de calidad, costo y latencia durante la operación.",
                      "Permite detectar degradaciones a tiempo.",
                      "alerta cuando la calidad cae bajo el umbral"),
        "costo por token": ("El costo por token es el precio que cobra la API por cada token de entrada y de salida.",
                            "La salida suele costar varias veces más que la entrada.",
                            "define el presupuesto de cada ronda de entrenamiento"),
        "checkpoint": ("Un checkpoint es una copia guardada de los pesos del modelo en un momento del entrenamiento.",
                       "Permite reanudar o volver a una versión mejor.",
                       "permite revertir si una ronda empeora la calidad"),
    },
}

CONTEXTS = ["un banco", "una AFP", "un retail", "una clínica", "una universidad", "una empresa de logística"]
TEMPLATES = [
    "Explica qué es {c} y para qué sirve en {ctx}.",
    "¿Cómo aplicarías {c} en {ctx}?",
    "Describe {c} con un ejemplo para {ctx}.",
    "¿Qué ventajas tiene {c} para {ctx}?",
    "Resume el concepto de {c} pensando en {ctx}.",
    "Un equipo de {ctx} pregunta por {c}: ¿qué le respondes?",
]
_STOP = set("el la los las un una unos unas de del en y o que a al para por con se es su sus lo como mas más "
            "este esta pero sin sobre entre cada ya muy no".split())


def topics() -> list[str]:
    return list(KB)


def corpus() -> list[str]:
    out = []
    for concepts in KB.values():
        for c, (d, k, a) in concepts.items():
            out += [d, k, a, c]
    out += CONTEXTS + TEMPLATES + ["Respuesta del asistente.", "detailed thinking off"]
    return out * 3


def _answer(concept: str, ctx: str) -> str:
    for concepts in KB.values():
        if concept in concepts:
            d, k, a = concepts[concept]
            return f"{d} {k} En {ctx}, {a}."
    return "No tengo información sobre ese concepto."


def _words(s: str) -> list[str]:
    return [w for w in normalize_text(s).split() if w not in _STOP and len(w) > 2]


class MockTeacher:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def __call__(self, schema_name: str, system: str, user: str) -> dict:
        if schema_name == "taxonomia":
            return {"topics": topics()}
        if schema_name == "tareas":
            topic = re.search(r"Tema: (.+)", user).group(1).strip()
            k = int(re.search(r"exactamente (\d+)", user).group(1))
            concepts = list(KB.get(topic, KB[topics()[0]]))
            tasks = []
            for _ in range(k):
                c, ctx, tpl = self.rng.choice(concepts), self.rng.choice(CONTEXTS), self.rng.choice(TEMPLATES)
                tasks.append({"difficulty": self.rng.choice(["basica", "intermedia", "avanzada"]),
                              "instruction": tpl.format(c=c, ctx=ctx)})
            return {"tasks": tasks}
        if schema_name == "respuesta":
            concept = next((c for t in KB.values() for c in sorted(t, key=len, reverse=True) if c in user), None)
            ctx = next((x for x in CONTEXTS if x in user), CONTEXTS[0])
            return {"answer": _answer(concept, ctx) if concept else "Sin información."}
        if schema_name == "evaluacion":
            ref = user.split("RESPUESTA DE REFERENCIA:")[1].split("RESPUESTA DEL ESTUDIANTE:")[0]
            stu = user.split("RESPUESTA DEL ESTUDIANTE:")[1]
            rw, sw = Counter(_words(ref)), Counter(_words(stu))
            common = sum((rw & sw).values())
            prec = common / max(sum(sw.values()), 1)
            rec = common / max(sum(rw.values()), 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-9)
            score = lambda x: max(1, min(10, round(1 + 9 * x)))  # noqa: E731
            missing = [w for w, _ in (rw - sw).most_common(4)]
            fb = ("Faltan ideas clave: " + ", ".join(missing) + ".") if missing else "Cubre los puntos de la referencia."
            return {"correccion": score(prec), "completitud": score(rec), "claridad": score(f1 ** 0.8),
                    "idioma_formato": score(min(1.0, prec + 0.3)), "nota_global": score(f1),
                    "retroalimentacion": fb}
        raise ValueError(f"Esquema desconocido en simulación: {schema_name}")

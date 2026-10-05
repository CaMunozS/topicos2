"""Cliente del PROFESOR (API de OpenAI) con:

- Salidas estructuradas (JSON Schema estricto) para currículo, respuestas y juez.
- Reintentos con backoff exponencial (429, 5xx, cortes de red) + reintento si el JSON viene roto.
- Detección automática de capacidades por modelo (si un modelo no acepta `reasoning`
  o `json_schema`, se desactiva solo para ese modelo y se sigue funcionando).
- Registro de USO por llamada: tokens de entrada/salida/caché/razonamiento, costo USD,
  latencia, propósito (curriculum | teacher | judge) y ronda.
- Guardián de presupuesto: si el gasto acumulado supera `autonomy.budget_usd`, se detiene.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from .utils import JsonlWriter


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0


def price_for(model: str, pricing: dict) -> tuple[dict, bool]:
    """Devuelve (precios, conocido). Busca coincidencia exacta y luego por prefijo."""
    if model in pricing:
        return pricing[model], True
    for name in sorted(pricing, key=len, reverse=True):
        if name != "default" and model.startswith(name):
            return pricing[name], True
    return pricing.get("default", {"input": 2.0, "cached_input": 0.2, "output": 10.0}), False


def cost_usd(u: Usage, prices: dict) -> float:
    uncached = max(u.input_tokens - u.cached_tokens, 0)
    return (
        uncached * prices.get("input", 0)
        + u.cached_tokens * prices.get("cached_input", prices.get("input", 0))
        + u.output_tokens * prices.get("output", 0)
    ) / 1_000_000


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise


class LLMClient:
    """Envoltorio de la API de OpenAI. Seguro para usar desde varios hilos."""

    def __init__(self, cfg, usage_log: JsonlWriter, logger, spent_so_far: float = 0.0):
        from openai import OpenAI

        oa = cfg.openai
        self.cfg = cfg
        self.logger = logger
        self.usage_log = usage_log
        self.pricing = oa.pricing
        self.budget = float(cfg.autonomy.budget_usd)
        self.api_style = oa.get("api_style", "responses")
        self.effort = oa.get("reasoning_effort") or ""
        self.spent = float(spent_so_far)
        self.round = 0
        self._lock = threading.Lock()
        self._caps: dict[str, dict[str, bool]] = {}
        self._warned_price: set[str] = set()
        kwargs: dict[str, Any] = {
            "api_key": oa.api_key,
            "max_retries": int(oa.get("max_retries", 5)),
            "timeout": float(oa.get("timeout_s", 120)),
        }
        if oa.get("base_url"):
            kwargs["base_url"] = oa.base_url
        if oa.get("organization"):
            kwargs["organization"] = oa.organization
        self.client = OpenAI(**kwargs)

    # ------------------------------------------------------------------ util
    def caps(self, model: str) -> dict[str, bool]:
        with self._lock:
            return self._caps.setdefault(model, {"reasoning": bool(self.effort), "schema": True})

    def _account(self, purpose: str, model: str, u: Usage, latency: float, status: str,
                 error: str = "", simulated: bool = False) -> float:
        prices, known = price_for(model, self.pricing)
        if not known and model not in self._warned_price:
            self._warned_price.add(model)
            self.logger.warning(f"Modelo '{model}' sin precio en config.yaml: se usa el precio 'default' para estimar.")
        c = cost_usd(u, prices)
        with self._lock:
            self.spent += c
            spent = self.spent
        self.usage_log.write({
            "round": self.round, "purpose": purpose, "model": model, "api_style": self.api_style,
            "input_tokens": u.input_tokens, "cached_tokens": u.cached_tokens,
            "output_tokens": u.output_tokens, "reasoning_tokens": u.reasoning_tokens,
            "cost_usd": round(c, 6), "cum_cost_usd": round(spent, 6), "latency_s": round(latency, 3),
            "status": status, "error": error[:300], "simulated": simulated,
        })
        return c

    def check_budget(self) -> None:
        if self.spent >= self.budget:
            raise BudgetExceeded(f"Presupuesto agotado: gastado US$ {self.spent:.2f} de US$ {self.budget:.2f}")

    # ------------------------------------------------------------- llamadas
    def json_call(self, purpose: str, model: str, system: str, user: str, schema_name: str,
                  schema: dict, max_output_tokens: int = 1500, temperature: float | None = None) -> dict:
        """Llama al modelo y devuelve un dict que cumple `schema`."""
        self.check_budget()
        last_err: Exception | None = None
        max_tokens = max_output_tokens
        for attempt in range(4):
            caps = self.caps(model)
            t0 = time.perf_counter()
            try:
                if self.api_style == "chat":
                    text, u, truncated = self._chat(model, system, user, schema_name, schema, max_tokens, caps, temperature)
                else:
                    text, u, truncated = self._responses(model, system, user, schema_name, schema, max_tokens, caps, temperature)
                latency = time.perf_counter() - t0
                if truncated:
                    self._account(purpose, model, u, latency, "truncated")
                    max_tokens = int(max_tokens * 2)
                    continue
                data = _extract_json(text)
                self._account(purpose, model, u, latency, "ok")
                return data
            except json.JSONDecodeError as e:
                last_err = e
                self._account(purpose, model, Usage(), time.perf_counter() - t0, "bad_json", str(e))
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e).lower()
                status = getattr(e, "status_code", None)
                self._account(purpose, model, Usage(), time.perf_counter() - t0, "error", str(e))
                if status == 400 and "reasoning" in msg and caps["reasoning"]:
                    self.logger.warning(f"{model}: no acepta 'reasoning'; se desactiva para este modelo.")
                    caps["reasoning"] = False
                    continue
                if status == 400 and any(k in msg for k in ("json_schema", "response_format", "text.format", "structured")) and caps["schema"]:
                    self.logger.warning(f"{model}: sin salidas estructuradas; se usa modo JSON por instrucción.")
                    caps["schema"] = False
                    continue
                if status == 400 and "temperature" in msg and temperature is not None:
                    temperature = None
                    continue
                if status in (401, 403, 404):
                    raise  # clave inválida, sin acceso o modelo inexistente: no tiene sentido reintentar
                time.sleep(min(2 ** attempt + random.random(), 30))
        raise RuntimeError(f"Falló la llamada '{purpose}' a {model} tras varios intentos: {last_err}")

    def _responses(self, model, system, user, schema_name, schema, max_tokens, caps, temperature):
        kwargs: dict[str, Any] = {"model": model, "instructions": system, "input": user,
                                  "max_output_tokens": max_tokens}
        if caps["schema"]:
            kwargs["text"] = {"format": {"type": "json_schema", "name": schema_name, "schema": schema, "strict": True}}
        else:
            kwargs["instructions"] = system + "\n\nResponde SOLO con un objeto JSON válido que cumpla este esquema:\n" + json.dumps(schema, ensure_ascii=False)
        if caps["reasoning"] and self.effort:
            kwargs["reasoning"] = {"effort": self.effort}
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = self.client.responses.create(**kwargs)
        u = Usage()
        if getattr(r, "usage", None):
            u.input_tokens = r.usage.input_tokens or 0
            u.output_tokens = r.usage.output_tokens or 0
            itd = getattr(r.usage, "input_tokens_details", None)
            otd = getattr(r.usage, "output_tokens_details", None)
            u.cached_tokens = (getattr(itd, "cached_tokens", 0) or 0) if itd else 0
            u.reasoning_tokens = (getattr(otd, "reasoning_tokens", 0) or 0) if otd else 0
        truncated = getattr(r, "status", "") == "incomplete" and not (r.output_text or "").strip().endswith("}")
        return r.output_text, u, truncated

    def _chat(self, model, system, user, schema_name, schema, max_tokens, caps, temperature):
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kwargs: dict[str, Any] = {"model": model, "messages": messages, "max_completion_tokens": max_tokens}
        if caps["schema"]:
            kwargs["response_format"] = {"type": "json_schema", "json_schema": {"name": schema_name, "schema": schema, "strict": True}}
        else:
            kwargs["response_format"] = {"type": "json_object"}
            messages[0]["content"] += "\n\nResponde SOLO con JSON válido que cumpla:\n" + json.dumps(schema, ensure_ascii=False)
        if caps["reasoning"] and self.effort:
            kwargs["reasoning_effort"] = self.effort
        if temperature is not None:
            kwargs["temperature"] = temperature
        r = self.client.chat.completions.create(**kwargs)
        u = Usage()
        if r.usage:
            u.input_tokens = r.usage.prompt_tokens or 0
            u.output_tokens = r.usage.completion_tokens or 0
            ptd = getattr(r.usage, "prompt_tokens_details", None)
            ctd = getattr(r.usage, "completion_tokens_details", None)
            u.cached_tokens = (getattr(ptd, "cached_tokens", 0) or 0) if ptd else 0
            u.reasoning_tokens = (getattr(ctd, "reasoning_tokens", 0) or 0) if ctd else 0
        choice = r.choices[0]
        return choice.message.content or "", u, choice.finish_reason == "length"


class SimulatedLLMClient(LLMClient):
    """Mismo contrato que LLMClient, pero sin red: el 'profesor' es un simulador local.

    Sirve para probar todo el pipeline (datos → entrenamiento → evaluación → checkpoints →
    dashboard) sin clave ni costo. Los tokens se estiman (~4 caracteres por token) y el costo
    se calcula con la misma tabla de precios para que el monitoreo muestre cifras realistas.
    """

    def __init__(self, cfg, usage_log: JsonlWriter, logger, handler: Callable[[str, str, str], dict],
                 spent_so_far: float = 0.0):
        self.cfg = cfg
        self.logger = logger
        self.usage_log = usage_log
        self.pricing = cfg.openai.pricing
        self.budget = float(cfg.autonomy.budget_usd)
        self.api_style = "simulado"
        self.effort = ""
        self.spent = float(spent_so_far)
        self.round = 0
        self._lock = threading.Lock()
        self._caps = {}
        self._warned_price = set()
        self.handler = handler
        self._rng = random.Random(cfg.run.seed)

    def json_call(self, purpose, model, system, user, schema_name, schema, max_output_tokens=1500, temperature=None):
        self.check_budget()
        t0 = time.perf_counter()
        data = self.handler(schema_name, system, user)
        out = json.dumps(data, ensure_ascii=False)
        u = Usage(input_tokens=(len(system) + len(user)) // 4, output_tokens=len(out) // 4)
        latency = time.perf_counter() - t0 + self._rng.uniform(0.6, 2.4)  # latencia típica simulada
        self._account(purpose, model, u, latency, "ok", simulated=True)
        return data

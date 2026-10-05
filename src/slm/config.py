"""Carga de configuración: config.yaml + variables de entorno (.env).

Regla del proyecto: lo único que el usuario DEBE tocar es OPENAI_API_KEY en .env.
Todo lo demás trae valores por defecto razonables y se puede ajustar si se quiere.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


class Cfg(dict):
    """Diccionario con acceso por atributo (cfg.student.model_id)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as e:
            raise AttributeError(key) from e

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    @staticmethod
    def wrap(obj: Any) -> Any:
        if isinstance(obj, dict):
            return Cfg({k: Cfg.wrap(v) for k, v in obj.items()})
        if isinstance(obj, list):
            return [Cfg.wrap(v) for v in obj]
        return obj


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_dotenv(env_path: Path) -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:  # lector mínimo si python-dotenv no está instalado
        if not env_path.exists():
            return
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _env(name: str) -> str | None:
    v = os.getenv(name)
    if v is None:
        return None
    v = v.strip()
    return v or None


def load_config(config_path: str | Path = ROOT / "config.yaml", simulate: bool = False) -> Cfg:
    config_path = Path(config_path)
    _load_dotenv(ROOT / ".env")
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if simulate:
        raw = deep_merge(raw, raw.get("simulation", {}).get("overrides", {}))

    # --- Variables de entorno (.env) tienen prioridad sobre config.yaml ---
    oa = raw.setdefault("openai", {})
    oa["api_key"] = _env("OPENAI_API_KEY")
    oa["base_url"] = _env("OPENAI_BASE_URL")
    oa["organization"] = _env("OPENAI_ORG_ID")
    models = oa.setdefault("models", {})
    for role in ("curriculum", "teacher", "judge"):
        v = _env(f"OPENAI_MODEL_{role.upper()}")
        if v:
            models[role] = v
    effort = _env("OPENAI_REASONING_EFFORT")
    if effort is not None:
        oa["reasoning_effort"] = "" if effort.lower() in ("off", "ninguno") else effort
    if _env("OPENAI_API_STYLE"):
        oa["api_style"] = _env("OPENAI_API_STYLE")

    if _env("STUDENT_MODEL_ID") and not simulate:
        raw["student"]["model_id"] = _env("STUDENT_MODEL_ID")
    raw["student"]["hf_token"] = _env("HF_TOKEN")
    if _env("BUDGET_USD"):
        raw["autonomy"]["budget_usd"] = float(_env("BUDGET_USD"))

    raw["simulate"] = bool(simulate)
    return Cfg.wrap(raw)


def require_api_key(cfg: Cfg) -> None:
    if cfg.simulate:
        return
    key = cfg.openai.get("api_key")
    if not key or key.startswith("sk-tu-clave") or key == "COLOCA_AQUI_TU_CLAVE":
        raise SystemExit(
            "\n[ERROR] Falta la clave de OpenAI.\n"
            "  1) Abre el archivo .env en la raíz del proyecto.\n"
            "  2) Escribe tu clave:  OPENAI_API_KEY=sk-...\n"
            "  3) Vuelve a ejecutar:  python run.py\n"
            "  (Para probar sin clave ni GPU:  python run.py --simulate)\n"
        )

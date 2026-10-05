"""Utilidades comunes: registros JSONL, estado atómico, semillas, deduplicación y consola."""
from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Consola
# ---------------------------------------------------------------------------

def get_logger(name: str = "slm", log_file: Path | None = None) -> logging.Logger:
    """Logger con salida enriquecida (rich) si está disponible + archivo de texto."""
    logger = logging.getLogger(name)
    if logger.handlers:
        if log_file is not None and not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
            _add_file_handler(logger, log_file)
        return logger
    logger.setLevel(logging.INFO)
    try:
        from rich.logging import RichHandler

        handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False, markup=False)
        handler.setFormatter(logging.Formatter("%(message)s", datefmt="%H:%M:%S"))
    except Exception:  # pragma: no cover - rich es opcional
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S"))
    logger.addHandler(handler)
    logger.propagate = False
    if log_file is not None:
        _add_file_handler(logger, log_file)
    return logger


def _add_file_handler(logger: logging.Logger, log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)


# ---------------------------------------------------------------------------
# Tiempo y archivos
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


class JsonlWriter:
    """Escritor JSONL seguro entre hilos: una línea por evento, con flush inmediato."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        row = {"ts": now_iso(), **row}
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # línea truncada por un corte abrupto: se ignora
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    """Escritura atómica (archivo temporal + rename) para no corromper el estado."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Reproducibilidad
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Texto y deduplicación
# ---------------------------------------------------------------------------

def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9ñ ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def char_ngrams(s: str, n: int = 3) -> set[str]:
    s = normalize_text(s)
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NearDuplicateIndex:
    """Índice simple de casi-duplicados por trigramas de caracteres (sin dependencias)."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self._items: list[set[str]] = []
        self._exact: set[str] = set()

    def add(self, text: str) -> None:
        self._exact.add(normalize_text(text))
        self._items.append(char_ngrams(text))

    def is_duplicate(self, text: str) -> bool:
        norm = normalize_text(text)
        if norm in self._exact:
            return True
        grams = char_ngrams(text)
        return any(jaccard(grams, g) >= self.threshold for g in self._items)

    def __len__(self) -> int:
        return len(self._items)


def strip_think(text: str) -> str:
    """Quita bloques <think>...</think> (modelos con razonamiento visible)."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.replace("<think>", "").replace("</think>", "")
    return text.strip()


def fmt_usd(x: float) -> str:
    return f"US$ {x:,.4f}" if x < 1 else f"US$ {x:,.2f}"


def fmt_secs(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    if m:
        return f"{m} min {sec:02d} s"
    return f"{sec} s"


class Timer:
    def __init__(self):
        self.t0 = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

"""Prueba de humo: corre el pipeline completo en modo simulación (sin red, sin GPU)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.config import load_config  # noqa: E402
from slm.orchestrator import Orchestrator  # noqa: E402


def test_pipeline_simulado(tmp_path):
    cfg = load_config(ROOT / "config.yaml", simulate=True)
    cfg.autonomy.max_rounds = 2
    cfg.curriculum.tasks_per_round = 24
    cfg.evaluation.eval_size = 8
    cfg.training.epochs_per_round = 1
    cfg.monitoring.tensorboard = False
    run = tmp_path / "sim"
    state = Orchestrator(cfg, run).run()
    assert state["status"] == "finished"
    for f in ["state.json", "dashboard.html", "usage_summary.json", "config_snapshot.json",
              "logs/openai_usage.jsonl", "logs/train_metrics.jsonl", "logs/eval_metrics.jsonl",
              "data/eval_set.jsonl", "checkpoints/round_000/adapter_model.safetensors",
              "checkpoints/round_002/adapter_model.safetensors", "checkpoints/best/adapter_model.safetensors"]:
        assert (run / f).exists(), f
    snap = json.loads((run / "config_snapshot.json").read_text())
    assert "api_key" not in snap["openai"]
    # reanudar una corrida terminada no repite trabajo
    cfg.autonomy.max_rounds = 3
    state = Orchestrator(cfg, run).run()
    assert (run / "checkpoints/round_003/meta.json").exists()

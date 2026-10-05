"""ORQUESTADOR AUTÓNOMO: el bucle que entrena al SLM sin intervención humana.

Ronda 0  : taxonomía → set de evaluación fijo → nota base del estudiante (sin entrenar).
Ronda r≥1: currículo adaptativo → respuestas del profesor → LoRA (nuevos + repaso) →
           evaluación con juez → PROMOVER (nuevo mejor) · ACEPTAR (dentro de tolerancia,
           se sigue desde aquí) · REVERTIR (empeoró: vuelve al mejor) → ¿parar?

Criterios de parada: nota objetivo, paciencia (rondas sin mejora), presupuesto en USD,
máximo de rondas, máximo de horas, o archivo STOP en la carpeta de la corrida.
Todo es reanudable: `python run.py --resume runs/<corrida>`.
"""
from __future__ import annotations

import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from . import curriculum as cur
from .evaluator import evaluate
from .student import Student, system_info
from .teacher import BudgetExceeded, LLMClient, SimulatedLLMClient
from .trainer import train_round
from .utils import (JsonlWriter, NearDuplicateIndex, fmt_secs, fmt_usd, get_logger, read_json, read_jsonl,
                    set_seed, write_json, write_jsonl)


class Orchestrator:
    def __init__(self, cfg, run_dir: Path | None = None):
        self.cfg = cfg
        if run_dir is None:
            name = cfg.run.name if cfg.run.name not in (None, "", "auto") else datetime.now().strftime("%Y%m%d-%H%M%S")
            if cfg.simulate:
                name = f"sim-{name}"
            run_dir = Path(cfg.run.output_dir) / name
        self.run_dir = Path(run_dir)
        self.dirs = {k: self.run_dir / k for k in ("data", "checkpoints", "logs", "eval_details")}
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("slm", self.dirs["logs"] / "run.log")
        self.events = JsonlWriter(self.dirs["logs"] / "events.jsonl")
        self.usage_log = JsonlWriter(self.dirs["logs"] / "openai_usage.jsonl")
        self.train_log = JsonlWriter(self.dirs["logs"] / "train_metrics.jsonl")
        self.eval_log = JsonlWriter(self.dirs["logs"] / "eval_metrics.jsonl")
        self.round_log = JsonlWriter(self.dirs["logs"] / "rounds.jsonl")
        self.state_path = self.run_dir / "state.json"
        self.stop_file = self.run_dir / "STOP"
        self.state = read_json(self.state_path) or {
            "status": "created", "created": datetime.now().isoformat(timespec="seconds"),
            "simulate": cfg.simulate, "round": 0, "phase": {}, "best_round": None, "best_score": None,
            "rounds_without_improvement": 0, "global_step": 0, "train_seconds": 0.0, "topics": [],
            "elapsed_s": 0.0,
        }
        self.tb = None

    # ------------------------------------------------------------ helpers
    def save_state(self, **kw) -> None:
        self.state.update(kw)
        write_json(self.state_path, self.state)

    def event(self, kind: str, msg: str, **data) -> None:
        self.events.write({"round": self.state["round"], "kind": kind, "msg": msg, **data})
        self.logger.info(msg)

    def _spent(self) -> float:
        rows = read_jsonl(self.dirs["logs"] / "openai_usage.jsonl")
        return float(sum(r.get("cost_usd", 0) for r in rows))

    def _ckpt(self, r: int) -> Path:
        return self.dirs["checkpoints"] / f"round_{r:03d}"

    def _stop_reason(self, t0: float) -> str | None:
        a = self.cfg.autonomy
        if self.stop_file.exists():
            return "archivo STOP"
        if self.llm.spent >= float(a.budget_usd):
            return f"presupuesto agotado ({fmt_usd(self.llm.spent)})"
        hours = (self.state["elapsed_s"] + time.time() - t0) / 3600
        if hours >= float(a.max_hours):
            return f"tiempo máximo ({a.max_hours} h)"
        if self.state["best_score"] is not None and self.state["best_score"] >= float(a.target_score):
            return f"nota objetivo alcanzada ({self.state['best_score']:.2f} ≥ {a.target_score})"
        if self.state["rounds_without_improvement"] >= int(a.patience):
            return f"paciencia agotada ({a.patience} rondas sin mejora)"
        return None

    def _setup_tensorboard(self) -> None:
        if not self.cfg.monitoring.get("tensorboard", True):
            return
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(str(self.run_dir / "tensorboard"))
        except Exception:
            self.tb = None

    def _dashboard(self) -> None:
        try:
            from .dashboard import build_dashboard

            build_dashboard(self.run_dir)
        except Exception as e:  # noqa: BLE001 - el monitoreo nunca debe botar el entrenamiento
            self.logger.warning(f"No se pudo actualizar el dashboard: {e}")

    def _usage_summary(self) -> dict:
        rows = read_jsonl(self.dirs["logs"] / "openai_usage.jsonl")
        agg: dict = defaultdict(lambda: defaultdict(float))
        for r in rows:
            for key in (f"purpose:{r['purpose']}", f"model:{r['model']}", "total"):
                a = agg[key]
                a["calls"] += 1
                a["errors"] += r.get("status") not in ("ok",)
                for f in ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens", "cost_usd", "latency_s"):
                    a[f] += r.get(f, 0) or 0
        train = read_jsonl(self.dirs["logs"] / "train_metrics.jsonl")
        summary = {k: {f: round(v, 6) for f, v in d.items()} for k, d in agg.items()}
        summary["training"] = {"optimizer_steps": len(train), "train_tokens": int(sum(r["tokens"] for r in train)),
                               "train_seconds": round(self.state["train_seconds"], 1),
                               "peak_mem_gb": max((r.get("mem_gb", 0) for r in train), default=0)}
        write_json(self.run_dir / "usage_summary.json", summary)
        return summary

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        cfg = self.cfg
        set_seed(int(cfg.run.seed))
        t0 = time.time()
        self.save_state(status="running")
        snap = {k: v for k, v in cfg.items()}
        snap["openai"] = {k: v for k, v in cfg.openai.items() if k not in ("api_key", "organization")}
        write_json(self.run_dir / "config_snapshot.json", snap)  # sin la clave: nunca se escribe a disco
        info = system_info()
        write_json(self.run_dir / "system.json", info)
        self.event("start", f"Corrida {self.run_dir.name} · dispositivo {info['device']} · "
                            f"{'SIMULACIÓN' if cfg.simulate else 'OpenAI real'}", system=info)
        spent = self._spent()
        if cfg.simulate:
            from .simulate import MockTeacher

            self.llm = SimulatedLLMClient(cfg, self.usage_log, self.logger, MockTeacher(int(cfg.run.seed)), spent)
        else:
            self.llm = LLMClient(cfg, self.usage_log, self.logger, spent)
        self._setup_tensorboard()

        try:
            eval_set = self._prepare_eval_set()
            self.student = Student(cfg, self.run_dir, self.logger)
            tiny_corpus = None
            if cfg.student.model_id == "tiny-random-llama":
                from .simulate import corpus

                tiny_corpus = corpus() + [x["instruction"] + " " + x["response"] for x in eval_set]
            self.student.load(tiny_corpus)
            trainable, total = Student.param_counts(self.student.model)
            self.save_state(trainable_params=trainable, total_params=total)
            self._baseline(eval_set)
            self._loop(eval_set, t0)
        except BudgetExceeded as e:
            self.event("stop", f"Detenido: {e}")
            self.save_state(status="stopped", stop_reason=str(e))
        except KeyboardInterrupt:
            self.event("interrupted", "Interrumpido por el usuario (Ctrl+C). Reanuda con: "
                                      f"python run.py --resume {self.run_dir}")
            self.save_state(status="interrupted")
        finally:
            self.state["elapsed_s"] = self.state.get("elapsed_s", 0) + time.time() - t0
            self.save_state()
            summary = self._usage_summary()
            self._dashboard()
            if self.tb:
                self.tb.close()
        self._final_report(summary)
        return self.state

    def _prepare_eval_set(self) -> list[dict]:
        path = self.dirs["data"] / "eval_set.jsonl"
        if path.exists():
            self.logger.info("Set de evaluación existente reutilizado.")
            return read_jsonl(path)
        cfg = self.cfg
        self.llm.round = 0
        topics = self.state.get("topics") or cur.propose_topics(self.llm, cfg)
        self.save_state(topics=topics)
        self.event("topics", f"Taxonomía: {', '.join(topics)}", topics=topics)
        dedup = NearDuplicateIndex(float(cfg.curriculum.dedup_threshold))
        alloc = cur.allocate(int(cfg.evaluation.eval_size), {t: 1 / len(topics) for t in topics})
        tasks = cur.generate_tasks(self.llm, cfg, alloc, dedup, {}, {}, "eval", self.logger)
        items = cur.answer_tasks(self.llm, cfg, tasks, 0, self.logger)
        for it in items:
            it["split"] = "eval"
        write_jsonl(path, items)
        self.event("eval_set", f"Set de evaluación fijo creado: {len(items)} preguntas (nunca se entrena con ellas).",
                   n=len(items))
        return items

    def _baseline(self, eval_set: list[dict]) -> None:
        if self.state["phase"].get("0") == "evaluated":
            return
        ck = self._ckpt(0)
        self.student.save_adapter(ck)  # LoRA recién inicializado (B=0) ≡ modelo base
        self.llm.round = 0
        summary, rows = evaluate(self.student, self.llm, self.cfg, eval_set, 0, self.logger)
        write_jsonl(self.dirs["eval_details"] / "round_000.jsonl", rows)
        self.eval_log.write(summary)
        self._tb_eval(summary)
        write_json(ck / "meta.json", {"round": 0, "score": summary["mean_score"], "decision": "base",
                                       "pass_rate": summary["pass_rate"]})
        self.round_log.write({"round": 0, "decision": "base", "mean_score": summary["mean_score"],
                              "pass_rate": summary["pass_rate"], "new_samples": 0, "train_samples": 0,
                              "cost_usd": round(self.llm.spent, 4)})
        shutil.rmtree(self.dirs["checkpoints"] / "best", ignore_errors=True)
        shutil.copytree(ck, self.dirs["checkpoints"] / "best")
        self.state["phase"]["0"] = "evaluated"
        self.save_state(best_round=0, current_round=0, best_score=summary["mean_score"], last_eval=summary)
        self.event("baseline", f"Nota base del estudiante (sin entrenar): {summary['mean_score']:.2f}/10 · "
                               f"aprobación {summary['pass_rate']:.0%}", score=summary["mean_score"])
        self._dashboard()

    def _tb_eval(self, s: dict) -> None:
        if self.tb:
            self.tb.add_scalar("eval/mean_score", s["mean_score"], s["round"])
            self.tb.add_scalar("eval/pass_rate", s["pass_rate"], s["round"])
            self.tb.add_scalar("openai/cum_cost_usd", self.llm.spent, s["round"])

    def _loop(self, eval_set: list[dict], t0: float) -> None:
        cfg = self.cfg
        a = cfg.autonomy
        topics = self.state["topics"]
        dedup = NearDuplicateIndex(float(cfg.curriculum.dedup_threshold))
        pool: list[dict] = []
        for x in eval_set:
            dedup.add(x["instruction"])
        for f in sorted(self.dirs["data"].glob("round_*.jsonl")):
            for x in read_jsonl(f):
                dedup.add(x["instruction"])
                pool.append(x)

        start = max(1, int(self.state["round"]) + (1 if self.state["phase"].get(str(self.state["round"])) == "evaluated" else 0))
        for r in range(start, int(a.max_rounds) + 1):
            reason = self._stop_reason(t0)
            if reason:
                self.event("stop", f"Fin del bucle autónomo: {reason}.")
                self.save_state(status="finished", stop_reason=reason)
                return
            self.save_state(round=r)
            self.llm.round = r
            phase = self.state["phase"].get(str(r))
            t_round = time.time()
            spent_before = self.llm.spent

            # 1) Currículo adaptativo + respuestas del profesor
            data_path = self.dirs["data"] / f"round_{r:03d}.jsonl"
            if data_path.exists():
                new = read_jsonl(data_path)
                pool_old = [x for x in pool if x.get("round") != r]
            else:
                last = self.state.get("last_eval") or {}
                weights = cur.topic_weights(topics, last.get("by_topic"), float(cfg.curriculum.weakness_gamma),
                                            float(cfg.curriculum.min_topic_weight))
                alloc = cur.allocate(int(cfg.curriculum.tasks_per_round), weights)
                by_topic: dict[str, list[str]] = defaultdict(list)
                for x in pool:
                    by_topic[x["topic"]].append(x["instruction"])
                self.event("curriculum", f"Ronda {r}: currículo → " +
                           ", ".join(f"{t} {n}" for t, n in alloc.items()), alloc=alloc, weights=weights)
                tasks = cur.generate_tasks(self.llm, cfg, alloc, dedup, by_topic, last.get("weaknesses", {}),
                                           "train", self.logger)
                new = cur.answer_tasks(self.llm, cfg, tasks, r, self.logger)
                write_jsonl(data_path, new)
                pool_old = list(pool)
                pool.extend(new)
                self.event("data", f"Ronda {r}: {len(new)} ejemplos nuevos del profesor (pool total {len(pool)}).",
                           new=len(new), pool=len(pool))
            if not new:
                self.event("warning", f"Ronda {r}: no se obtuvieron datos nuevos; se termina.")
                self.save_state(status="finished", stop_reason="sin datos nuevos")
                return

            # 2) Entrenamiento LoRA partiendo SIEMPRE del mejor checkpoint
            ck = self._ckpt(r)
            if phase in ("trained", "evaluated") and (ck / "adapter_model.safetensors").exists():
                self.student.load_adapter_weights(ck)
                tr = read_json(ck / "meta.json", {}).get("train", {})
            else:
                parent = self.state.get("current_round", self.state["best_round"])
                self.student.load_adapter_weights(self._ckpt(parent))
                mix = cur.build_training_mix(new, pool_old, float(cfg.training.replay_ratio), int(cfg.run.seed) + r)
                tr = train_round(self.student, mix, cfg, r, int(self.state["global_step"]), self.train_log, self.tb,
                                 self.stop_file, self.logger)
                self.state["global_step"] = tr["global_step"]
                self.state["train_seconds"] = self.state.get("train_seconds", 0) + tr["seconds"]
                self.student.save_adapter(ck)
                write_json(ck / "meta.json", {"round": r, "train": tr, "parent": parent})
                self.state["phase"][str(r)] = "trained"
                self.save_state()
                self.event("trained", f"Ronda {r}: entrenada en {fmt_secs(tr['seconds'])} · pérdida "
                                      f"{tr['loss_first']} → {tr['loss_last']} · checkpoint {ck.name}", **tr)
                if tr["interrupted"]:
                    self.save_state(status="stopped", stop_reason="archivo STOP")
                    return

            # 3) Evaluación con juez
            summary, rows = evaluate(self.student, self.llm, cfg, eval_set, r, self.logger)
            write_jsonl(self.dirs["eval_details"] / f"round_{r:03d}.jsonl", rows)
            self.eval_log.write(summary)
            self._tb_eval(summary)

            # 4) Promover o revertir
            best = self.state["best_score"]
            delta = summary["mean_score"] - best
            if delta >= float(cfg.evaluation.min_delta):
                decision = "promovido"  # nuevo mejor checkpoint
                shutil.rmtree(self.dirs["checkpoints"] / "best", ignore_errors=True)
                shutil.copytree(ck, self.dirs["checkpoints"] / "best")
                self.save_state(best_round=r, best_score=summary["mean_score"], current_round=r,
                                rounds_without_improvement=0)
            elif delta >= -float(cfg.evaluation.rollback_tolerance):
                decision = "aceptado"   # no supera al mejor, pero no empeora más que la tolerancia: se sigue desde aquí
                self.save_state(current_round=r,
                                rounds_without_improvement=self.state["rounds_without_improvement"] + 1)
            else:
                decision = "revertido"  # empeoró: se vuelve al mejor checkpoint
                self.student.load_adapter_weights(self._ckpt(self.state["best_round"]))
                self.save_state(current_round=self.state["best_round"],
                                rounds_without_improvement=self.state["rounds_without_improvement"] + 1)
            meta = read_json(ck / "meta.json", {})
            meta.update({"score": summary["mean_score"], "pass_rate": summary["pass_rate"], "delta": round(delta, 4),
                         "decision": decision})
            write_json(ck / "meta.json", meta)
            self.state["phase"][str(r)] = "evaluated"
            self.save_state(last_eval=summary)
            self.round_log.write({"round": r, "decision": decision, "mean_score": summary["mean_score"],
                                  "pass_rate": summary["pass_rate"], "delta": round(delta, 4),
                                  "new_samples": len(new), "train_samples": tr.get("examples"),
                                  "loss_last": tr.get("loss_last"), "train_seconds": tr.get("seconds"),
                                  "round_cost_usd": round(self.llm.spent - spent_before, 4),
                                  "cost_usd": round(self.llm.spent, 4), "round_seconds": round(time.time() - t_round, 1)})
            self.event("eval", f"Ronda {r}: nota {summary['mean_score']:.2f} (Δ {delta:+.2f}) → {decision.upper()} · "
                               f"mejor = ronda {self.state['best_round']} ({self.state['best_score']:.2f}) · "
                               f"gasto acumulado {fmt_usd(self.llm.spent)}", decision=decision)
            self._usage_summary()
            self._dashboard()
        self.event("stop", f"Fin del bucle autónomo: máximo de rondas ({a.max_rounds}).")
        self.save_state(status="finished", stop_reason="máximo de rondas")

    def _final_report(self, summary: dict) -> None:
        s = self.state
        tot = summary.get("total", {})
        self.logger.info("=" * 70)
        self.logger.info(f"ESTADO: {s['status']} · motivo: {s.get('stop_reason', '-')}")
        if s.get("best_round") is not None:
            self.logger.info(f"Mejor checkpoint: ronda {s['best_round']} · nota {s['best_score']:.2f}/10 → "
                             f"{self.dirs['checkpoints'] / 'best'}")
        self.logger.info(f"API: {int(tot.get('calls', 0))} llamadas · {int(tot.get('input_tokens', 0)):,} tokens entrada · "
                         f"{int(tot.get('output_tokens', 0)):,} salida · {fmt_usd(tot.get('cost_usd', 0))}")
        self.logger.info(f"Dashboard: {self.run_dir / 'dashboard.html'}")
        self.logger.info("=" * 70)

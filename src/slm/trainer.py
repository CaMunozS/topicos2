"""ENTRENAMIENTO SFT con LoRA (bucle propio en PyTorch, transparente y registrable).

Por cada paso de optimización se registra: pérdida, tasa de aprendizaje, norma del
gradiente, tokens/segundo, memoria y época → train_metrics.jsonl (+ TensorBoard opcional).
"""
from __future__ import annotations

import contextlib
import math
import random
import time
from pathlib import Path

import torch
from transformers import get_cosine_schedule_with_warmup

from .student import memory_gb


def _batches(examples: list[dict], bs: int, pad_id: int, rng: random.Random):
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        chunk = [examples[j] for j in idx[i : i + bs]]
        L = max(len(e["input_ids"]) for e in chunk)
        ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
        lab = torch.full((len(chunk), L), -100, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for k, e in enumerate(chunk):
            n = len(e["input_ids"])
            ids[k, :n] = torch.tensor(e["input_ids"])
            lab[k, :n] = torch.tensor(e["labels"])
            att[k, :n] = 1
        yield ids, lab, att


def train_round(student, records: list[dict], cfg, round_idx: int, global_step: int, metrics_log,
                tb_writer, stop_file: Path, logger) -> dict:
    tcfg = cfg.training
    model, tok = student.model, student.tokenizer
    tok.padding_side = "right"
    max_len = int(cfg.student.max_seq_len)

    examples = [e for r in records if (e := student.encode_example(r["instruction"], r["response"], max_len))]
    skipped = len(records) - len(examples)
    if not examples:
        raise RuntimeError("No hay ejemplos de entrenamiento válidos (¿max_seq_len demasiado corto?).")

    bs, accum, epochs = int(tcfg.batch_size), int(tcfg.grad_accum), int(tcfg.epochs_per_round)
    micro_per_epoch = math.ceil(len(examples) / bs)
    steps_per_epoch = math.ceil(micro_per_epoch / accum)
    total_steps = steps_per_epoch * epochs
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=float(tcfg.learning_rate), weight_decay=float(tcfg.weight_decay))
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(float(tcfg.warmup_ratio) * total_steps)), total_steps)

    dev = model.device
    amp = (torch.autocast("cuda", dtype=student._dtype()) if student.device == "cuda" and student._dtype() != torch.float32
           else contextlib.nullcontext())
    model.train()
    model.config.use_cache = False
    rng = random.Random(int(cfg.run.seed) + round_idx)
    if student.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    logger.info(f"Ronda {round_idx}: entrenando con {len(examples)} ejemplos ({skipped} omitidos) · "
                f"{epochs} épocas · {total_steps} pasos (lote efectivo {bs * accum}).")
    t_start = time.perf_counter()
    step, tokens_total, losses, interrupted = 0, 0, [], False
    for epoch in range(epochs):
        opt.zero_grad(set_to_none=True)
        acc_loss, acc_tok, micro, t_step = 0.0, 0, 0, time.perf_counter()
        for ids, lab, att in _batches(examples, bs, tok.pad_token_id, rng):
            ids, lab, att = ids.to(dev), lab.to(dev), att.to(dev)
            n_tok = int((lab != -100).sum())
            with amp:
                out = model(input_ids=ids, attention_mask=att, labels=lab)
            loss = out.loss.float()
            (loss / accum).backward()
            acc_loss += loss.item()
            acc_tok += n_tok
            micro += 1
            if micro % accum == 0 or micro == micro_per_epoch:
                n_micro = accum if micro % accum == 0 else micro % accum
                gnorm = torch.nn.utils.clip_grad_norm_(params, float(tcfg.max_grad_norm)).item()
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                global_step += 1
                dt = time.perf_counter() - t_step
                row = {"round": round_idx, "epoch": epoch + 1, "step": step, "global_step": global_step,
                       "loss": round(acc_loss / n_micro, 5), "lr": sched.get_last_lr()[0], "grad_norm": round(gnorm, 4),
                       "tokens": acc_tok, "tokens_per_s": round(acc_tok / max(dt, 1e-6), 1),
                       "mem_gb": round(memory_gb(), 3), "elapsed_s": round(time.perf_counter() - t_start, 2)}
                metrics_log.write(row)
                losses.append(row["loss"])
                tokens_total += acc_tok
                if tb_writer is not None:
                    tb_writer.add_scalar("train/loss", row["loss"], global_step)
                    tb_writer.add_scalar("train/lr", row["lr"], global_step)
                    tb_writer.add_scalar("train/grad_norm", row["grad_norm"], global_step)
                    tb_writer.add_scalar("train/tokens_per_s", row["tokens_per_s"], global_step)
                if step % int(tcfg.log_every) == 0:
                    logger.info(f"  r{round_idx} ép {epoch+1} paso {step}/{total_steps} · pérdida {row['loss']:.4f} · "
                                f"lr {row['lr']:.2e} · {row['tokens_per_s']:.0f} tok/s · {row['mem_gb']:.1f} GB")
                acc_loss, acc_tok, t_step = 0.0, 0, time.perf_counter()
                if stop_file.exists():
                    interrupted = True
                    break
        if interrupted:
            logger.warning("Archivo STOP detectado: se corta el entrenamiento de esta ronda.")
            break
    model.eval()
    k = max(1, len(losses) // 5)
    return {"examples": len(examples), "skipped": skipped, "steps": step, "global_step": global_step,
            "loss_first": round(sum(losses[:k]) / k, 5) if losses else None,
            "loss_last": round(sum(losses[-k:]) / k, 5) if losses else None,
            "train_tokens": tokens_total, "seconds": round(time.perf_counter() - t_start, 2),
            "peak_mem_gb": round(memory_gb(), 3), "interrupted": interrupted}

"""ESTUDIANTE: modelo Llama local (Hugging Face) + adaptadores LoRA.

- Descarga el modelo desde Hugging Face la primera vez (queda en caché ~/.cache/huggingface).
- Detecta el hardware: CUDA (NVIDIA), MPS (Apple Silicon) o CPU.
- En CUDA usa QLoRA (4 bits) automáticamente si la GPU tiene poca memoria.
- Solo se entrenan los adaptadores LoRA (≈1 % de los parámetros); el modelo base queda congelado.
- Modo `tiny-random-llama`: un Llama diminuto creado localmente (sin descarga) para pruebas.
"""
from __future__ import annotations

import contextlib
import os
import platform
from pathlib import Path

import torch

from .utils import strip_think

TINY_ID = "tiny-random-llama"
TINY_CHAT_TEMPLATE = (
    "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m['role'] }}<|end_header_id|>\n\n"
    "{{ m['content'] }}<|eot_id|>{% endfor %}"
    "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
)


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def system_info() -> dict:
    import transformers

    info = {
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "device": detect_device(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import peft

        info["peft"] = peft.__version__
    except Exception:
        pass
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["vram_gb"] = round(p.total_memory / 1e9, 1)
        info["bf16"] = torch.cuda.is_bf16_supported()
    return info


def _mps_supports_bf16() -> bool:
    """bfloat16 en MPS requiere macOS 14+; si no, se usa float16."""
    try:
        x = torch.ones(2, 2, dtype=torch.bfloat16, device="mps")
        _ = (x @ x).sum().item()
        return True
    except Exception:
        return False


def _dtype_kwarg() -> str:
    """transformers ≥ 4.56 usa `dtype`; versiones anteriores, `torch_dtype`."""
    import transformers

    try:
        major, minor = (int(v) for v in transformers.__version__.split(".")[:2])
    except ValueError:
        return "dtype"
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def estimate_params(config) -> int | None:
    """Parámetros aproximados de un modelo tipo Llama a partir de su config (sin descargar pesos)."""
    try:
        h, L, inter, V = config.hidden_size, config.num_hidden_layers, config.intermediate_size, config.vocab_size
        heads = config.num_attention_heads
        kv = getattr(config, "num_key_value_heads", heads) or heads
        hd = getattr(config, "head_dim", None) or h // heads
        attn = h * heads * hd * 2 + h * kv * hd * 2
        mlp = 3 * h * inter
        emb = V * h * (1 if getattr(config, "tie_word_embeddings", False) else 2)
        return int(L * (attn + mlp) + emb)
    except Exception:
        return None


def total_ram_gb() -> float | None:
    try:
        import psutil

        return psutil.virtual_memory().total / 1e9
    except Exception:
        return None


def memory_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e9
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            return torch.mps.current_allocated_memory() / 1e9
        except Exception:
            return 0.0
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e9
    except Exception:
        return 0.0


class Student:
    def __init__(self, cfg, run_dir: Path, logger):
        self.cfg = cfg
        self.scfg = cfg.student
        self.run_dir = Path(run_dir)
        self.logger = logger
        self.device = detect_device()
        self.model = None
        self.tokenizer = None
        self.end_of_turn = None
        self.quantized = False

    # ------------------------------------------------------------ carga
    def _dtype(self) -> torch.dtype:
        if self.scfg.model_id == TINY_ID:
            return torch.float32  # el modelo de prueba es diminuto: float32 es estable en cualquier dispositivo
        want = str(self.scfg.get("dtype", "auto")).lower()
        if want in ("bfloat16", "bf16"):
            return torch.bfloat16
        if want in ("float16", "fp16"):
            return torch.float16
        if want in ("float32", "fp32"):
            return torch.float32
        if self.device == "cuda":
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if self.device == "mps":
            return torch.bfloat16 if _mps_supports_bf16() else torch.float16
        return torch.float32

    def _want_4bit(self) -> bool:
        v = self.scfg.get("load_in_4bit", "auto")
        if self.device != "cuda":
            if v is True:
                self.logger.warning("4 bits (bitsandbytes) solo funciona con GPU NVIDIA; se carga sin cuantizar.")
            return False
        try:
            import bitsandbytes  # noqa: F401
        except ImportError:
            if v is True:
                self.logger.warning("bitsandbytes no está instalado; se carga sin cuantizar.")
            return False
        if v == "auto":
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return vram < float(self.scfg.get("auto_4bit_below_vram_gb", 20))
        return bool(v)

    def load(self, tiny_corpus: list[str] | None = None) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_id = self.scfg.model_id
        dtype = self._dtype()
        if model_id == TINY_ID:
            base_dir = self._ensure_tiny_base(tiny_corpus or [])
            src, token = str(base_dir), None
        else:
            src, token = model_id, self.scfg.get("hf_token")
        self.logger.info(f"Cargando estudiante '{model_id}' en {self.device} ({str(dtype).replace('torch.', '')})"
                         " — la primera vez se descarga desde Hugging Face…")
        self.tokenizer = AutoTokenizer.from_pretrained(src, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        vocab = self.tokenizer.get_vocab()
        self.end_of_turn = self.scfg.get("end_of_turn") or ("<|eot_id|>" if "<|eot_id|>" in vocab else self.tokenizer.eos_token)

        self.quantized = self._want_4bit()
        self._memory_preflight(src, token, dtype)
        kwargs = {_dtype_kwarg(): dtype, "token": token}
        if self.quantized:
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = {"": 0}
            self.logger.info("QLoRA activado: modelo base en 4 bits (NF4).")
        elif self.device == "cuda":
            kwargs["device_map"] = {"": 0}
        model = AutoModelForCausalLM.from_pretrained(src, **kwargs)
        if not self.quantized and self.device != "cuda":
            model = model.to(self.device)
        self.model = self._wrap_lora(model)

    def _memory_preflight(self, src: str, token, dtype) -> None:
        """Evita el típico 'out of memory' en Mac/CPU: estima la memoria antes de cargar el modelo."""
        if self.scfg.model_id == TINY_ID or self.scfg.get("skip_memory_check") or self.device == "cuda":
            return
        from transformers import AutoConfig

        try:
            conf = AutoConfig.from_pretrained(src, token=token)
        except Exception:
            return
        n = estimate_params(conf)
        ram = total_ram_gb()
        if not n or not ram:
            return
        bytes_per = 4 if dtype == torch.float32 else 2
        need = n * bytes_per / 1e9 + 3.0  # pesos + LoRA/optimizador/activaciones
        usable = ram * (0.70 if self.device == "mps" else 0.85)
        self.logger.info(f"Memoria: el estudiante ({n/1e9:.2f} B parámetros) necesita ≈ {need:.1f} GB; "
                         f"este equipo tiene {ram:.0f} GB (≈ {usable:.1f} GB utilizables).")
        if need > usable:
            raise SystemExit(
                f"\n[MEMORIA INSUFICIENTE] '{self.scfg.model_id}' necesita ≈ {need:.1f} GB y este equipo tiene ≈ {usable:.1f} GB "
                f"utilizables ({ram:.0f} GB en total).\nOpciones:\n"
                "  1) Un estudiante más pequeño de arquitectura Llama (sin registro):\n"
                "       python run.py --model HuggingFaceTB/SmolLM2-1.7B-Instruct\n"
                "  2) Llama 3.2 1B de Meta (requiere HF_TOKEN en .env y aceptar la licencia en Hugging Face):\n"
                "       python run.py --model meta-llama/Llama-3.2-1B-Instruct\n"
                "  3) Usar una GPU NVIDIA de 12 GB o más (QLoRA automático), p. ej. en la nube.\n"
                "  (Si igual quieres intentarlo: student.skip_memory_check: true en config.yaml)\n")

    def _ensure_tiny_base(self, corpus: list[str]) -> Path:
        """Crea (una sola vez por corrida) un Llama diminuto + tokenizador BPE entrenado localmente."""
        base_dir = self.run_dir / "base_model"
        if (base_dir / "config.json").exists():
            return base_dir
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        specials = ["<|begin_of_text|>", "<|eot_id|>", "<|pad|>", "<|start_header_id|>", "<|end_header_id|>"]
        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=int(self.scfg.get("tiny_vocab", 3000)), special_tokens=specials,
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tok.train_from_iterator(corpus or ["hola mundo"], trainer)
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                       pad_token="<|pad|>",
                                       additional_special_tokens=["<|start_header_id|>", "<|end_header_id|>"])
        fast.chat_template = TINY_CHAT_TEMPLATE
        t = self.scfg.get("tiny", {})
        conf = LlamaConfig(vocab_size=len(fast), hidden_size=int(t.get("hidden_size", 256)),
                           intermediate_size=int(t.get("intermediate_size", 688)),
                           num_hidden_layers=int(t.get("layers", 4)), num_attention_heads=int(t.get("heads", 4)),
                           num_key_value_heads=int(t.get("kv_heads", 2)), max_position_embeddings=2048,
                           bos_token_id=fast.bos_token_id, eos_token_id=fast.eos_token_id,
                           pad_token_id=fast.pad_token_id, tie_word_embeddings=False)
        torch.manual_seed(int(self.cfg.run.seed))
        model = LlamaForCausalLM(conf)
        base_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(base_dir)
        fast.save_pretrained(base_dir)
        self.logger.info(f"Modelo diminuto de prueba creado en {base_dir} ({sum(p.numel() for p in model.parameters())/1e6:.1f} M parámetros).")
        return base_dir

    def _wrap_lora(self, model):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        tcfg = self.cfg.training
        if self.quantized:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(tcfg.gradient_checkpointing))
        if tcfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        lc = self.cfg.lora
        conf = LoraConfig(r=int(lc.r), lora_alpha=int(lc.alpha), lora_dropout=float(lc.dropout),
                          target_modules=list(lc.target_modules), task_type="CAUSAL_LM",
                          modules_to_save=list(lc.get("modules_to_save") or []) or None)
        model = get_peft_model(model, conf)
        # Solo las matrices LoRA (A y B) van en float32 para un entrenamiento estable. Los módulos completos
        # (modules_to_save) conservan el dtype del modelo base: si se pasan a float32, sus salidas chocan con
        # las capas en bfloat16 ("expected m1 and m2 to have the same dtype").
        for name, p in model.named_parameters():
            if p.requires_grad and "lora_" in name and p.dtype != torch.float32:
                p.data = p.data.float()
        trainable, total = self.param_counts(model)
        self.logger.info(f"LoRA r={lc.r}: {trainable/1e6:.2f} M parámetros entrenables de {total/1e6:.0f} M "
                         f"({100*trainable/max(total,1):.2f} %).")
        return model

    @staticmethod
    def param_counts(model) -> tuple[int, int]:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        return trainable, total

    # ---------------------------------------------------------- formato
    def system_prompt(self) -> str:
        sp = self.scfg.get("system_prompt", "auto")
        if sp != "auto":
            return sp or ""
        if "nemotron" in str(self.scfg.model_id).lower():
            return "detailed thinking off"  # Nemotron: apaga el razonamiento largo (<think>)
        lang = self.cfg.domain.get("language", "español")
        return f"Eres un asistente experto. Responde en {lang}, de forma clara y precisa."

    def _messages(self, instruction: str) -> list[dict]:
        msgs = []
        sp = self.system_prompt()
        if sp:
            msgs.append({"role": "system", "content": sp})
        msgs.append({"role": "user", "content": instruction})
        return msgs

    def prompt_text(self, instruction: str) -> str:
        text = self.tokenizer.apply_chat_template(self._messages(instruction), tokenize=False, add_generation_prompt=True)
        return text + (self.scfg.get("assistant_prefill") or "")

    def encode_example(self, instruction: str, response: str, max_len: int) -> dict | None:
        """input_ids + labels con la instrucción ENMASCARADA (-100): solo se aprende la respuesta."""
        p_ids = self.tokenizer(self.prompt_text(instruction), add_special_tokens=False)["input_ids"]
        r_ids = self.tokenizer(response.strip() + self.end_of_turn, add_special_tokens=False)["input_ids"]
        if len(p_ids) >= max_len // 2:
            return None
        r_ids = r_ids[: max_len - len(p_ids)]
        return {"input_ids": p_ids + r_ids, "labels": [-100] * len(p_ids) + r_ids}

    # ------------------------------------------------------- generación
    def _stop_ids(self) -> list[int]:
        ids = {self.tokenizer.eos_token_id}
        eot = self.tokenizer.convert_tokens_to_ids(self.end_of_turn)
        if isinstance(eot, int) and eot >= 0:
            ids.add(eot)
        return [i for i in ids if i is not None]

    @torch.no_grad()
    def generate(self, instructions: list[str], max_new_tokens: int | None = None, batch_size: int | None = None,
                 use_adapter: bool = True) -> list[str]:
        max_new_tokens = int(max_new_tokens or self.scfg.max_new_tokens)
        batch_size = int(batch_size or self.scfg.gen_batch_size)
        model = self.model
        was_training = model.training
        model.eval()
        model.config.use_cache = True
        self.tokenizer.padding_side = "left"
        outs: list[str] = []
        ctx = model.disable_adapter() if not use_adapter else contextlib.nullcontext()
        with ctx:
            for i in range(0, len(instructions), batch_size):
                chunk = instructions[i : i + batch_size]
                enc = self.tokenizer([self.prompt_text(x) for x in chunk], return_tensors="pt", padding=True,
                                     add_special_tokens=False)
                enc = {k: enc[k].to(model.device) for k in ("input_ids", "attention_mask")}  # sin token_type_ids
                gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                     repetition_penalty=float(self.scfg.get("repetition_penalty", 1.0)),
                                     pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self._stop_ids())
                new = gen[:, enc["input_ids"].shape[1]:]
                for row in self.tokenizer.batch_decode(new, skip_special_tokens=True):
                    outs.append(strip_think(row) if self.scfg.get("strip_think", True) else row.strip())
        self.tokenizer.padding_side = "right"
        model.config.use_cache = False
        if was_training:
            model.train()
        return outs

    # -------------------------------------------------- checkpoints LoRA
    def save_adapter(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)

    def load_adapter_weights(self, path: Path) -> None:
        """Carga pesos LoRA guardados en el modelo ya envuelto (para revertir al mejor checkpoint)."""
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        path = Path(path)
        f = path / "adapter_model.safetensors"
        if not f.exists():
            raise FileNotFoundError(f)
        sd = load_file(str(f), device=str(self.model.device) if self.device != "mps" else "cpu")
        set_peft_model_state_dict(self.model, sd)

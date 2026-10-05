"""Pruebas de los caminos 'reales' sin red ni GPU:

- Un Llama local pequeño con la MISMA plantilla de chat y tokens especiales que Nemotron-Nano-4B.
- Un servidor falso compatible con la API de OpenAI (Responses y Chat Completions) en localhost.
- Precisión bfloat16 (la que se usa en Mac/MPS y GPUs modernas) + gradient checkpointing + LoRA.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slm.config import load_config  # noqa: E402
from slm.orchestrator import Orchestrator  # noqa: E402
from slm.simulate import MockTeacher, corpus  # noqa: E402

NEMOTRON_TEMPLATE = (
    "{{- '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\\n\\n' -}}"
    "{%- if messages[0].role == 'system' and messages[0].content != '' -%}{{- messages[0].content -}}"
    "{%- else -%}{{- 'detailed thinking off' -}}{%- endif %}{{- '<|eot_id|>' -}}"
    "{%- for message in messages -%}{%- if (message.role == 'user') -%}"
    "{{- '<|start_header_id|>user<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' -}}"
    "{%- elif message.role == 'assistant' -%}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' + message.content + '<|eot_id|>' -}}"
    "{%- endif %}{%- endfor %}{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' -}}{%- endif %}"
)


@pytest.fixture(scope="module")
def local_llama(tmp_path_factory):
    """Llama pequeño guardado en disco, cargado con AutoModel/AutoTokenizer igual que uno de Hugging Face."""
    import torch
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("fake-nemotron")
    specials = ["<|begin_of_text|>", "<|end_of_text|>", "<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus(), trainers.BpeTrainer(vocab_size=2000, special_tokens=specials,
                                                          initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                   additional_special_tokens=["<|start_header_id|>", "<|end_header_id|>", "<|end_of_text|>"])
    fast.chat_template = NEMOTRON_TEMPLATE  # sin pad_token, como Nemotron
    conf = LlamaConfig(vocab_size=len(fast), hidden_size=128, intermediate_size=384, num_hidden_layers=2,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=32, max_position_embeddings=1024,
                       bos_token_id=fast.bos_token_id, eos_token_id=fast.convert_tokens_to_ids("<|end_of_text|>"),
                       tie_word_embeddings=False)
    torch.manual_seed(0)
    LlamaForCausalLM(conf).to(torch.bfloat16).save_pretrained(d)
    fast.save_pretrained(d)
    return str(d)


@pytest.fixture(scope="module")
def fake_openai():
    """Servidor local que imita /v1/responses y /v1/chat/completions (y rechaza 'reasoning' la 1.ª vez)."""
    mt = MockTeacher(1)
    state = {"n": 0}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            out = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["n"] += 1
            if ("reasoning" in body or "reasoning_effort" in body) and state["n"] == 1:
                return self._send(400, {"error": {"message": "Unsupported parameter: 'reasoning' is not supported with this model.",
                                                  "type": "invalid_request_error", "param": "reasoning", "code": None}})
            if self.path.endswith("/responses"):
                name = body["text"]["format"]["name"]
                txt = json.dumps(mt(name, body.get("instructions", ""), body["input"]), ensure_ascii=False)
                return self._send(200, {
                    "id": "resp_1", "object": "response", "created_at": 0, "status": "completed", "model": body["model"],
                    "output": [{"type": "message", "id": "m1", "status": "completed", "role": "assistant",
                                "content": [{"type": "output_text", "text": txt, "annotations": []}]}],
                    "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
                    "usage": {"input_tokens": 120, "input_tokens_details": {"cached_tokens": 40}, "output_tokens": 60,
                              "output_tokens_details": {"reasoning_tokens": 12}, "total_tokens": 180}})
            msgs = body["messages"]
            name = body["response_format"]["json_schema"]["name"]
            txt = json.dumps(mt(name, msgs[0]["content"], msgs[1]["content"]), ensure_ascii=False)
            return self._send(200, {
                "id": "c1", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": txt}}],
                "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180,
                          "prompt_tokens_details": {"cached_tokens": 0}, "completion_tokens_details": {"reasoning_tokens": 0}}})

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


def _cfg(model_path, base_url, monkeypatch, **over):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", base_url)
    cfg = load_config(ROOT / "config.yaml", simulate=False)
    cfg.student.model_id = model_path
    cfg.student.max_seq_len = 256
    cfg.student.max_new_tokens = 48
    cfg.domain.topics = ["RAG", "Ajuste fino"]
    cfg.curriculum.tasks_per_round = 12
    cfg.curriculum.min_answer_chars = 20
    cfg.curriculum.dedup_threshold = 0.97
    cfg.evaluation.eval_size = 4
    cfg.training.epochs_per_round = 1
    cfg.training.batch_size = 2
    cfg.training.grad_accum = 2
    cfg.autonomy.max_rounds = 2
    cfg.openai.max_workers = 2
    cfg.monitoring.tensorboard = False
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    return cfg


def _check(run):
    assert (run / "checkpoints/round_002/adapter_model.safetensors").exists()
    usage = [json.loads(l) for l in (run / "logs/openai_usage.jsonl").read_text().splitlines()]
    assert any(u["status"] == "ok" and u["cost_usd"] > 0 for u in usage)
    assert (run / "dashboard.html").exists()


def test_bf16_gradient_checkpointing_responses_api(local_llama, fake_openai, tmp_path, monkeypatch):
    cfg = _cfg(local_llama, fake_openai, monkeypatch, student__dtype="bfloat16", training__gradient_checkpointing=True)
    state = Orchestrator(cfg, tmp_path / "r1").run()
    assert state["status"] == "finished"
    _check(tmp_path / "r1")
    # la plantilla de Nemotron debe recibir "detailed thinking off"
    assert json.loads((tmp_path / "r1/config_snapshot.json").read_text())["student"]["system_prompt"] == "auto"


def test_bf16_modules_to_save_chat_api(local_llama, fake_openai, tmp_path, monkeypatch):
    cfg = _cfg(local_llama, fake_openai, monkeypatch, student__dtype="bfloat16", openai__api_style="chat",
               lora__modules_to_save=["lm_head"], training__gradient_checkpointing=False)
    state = Orchestrator(cfg, tmp_path / "r2").run()
    assert state["status"] == "finished"
    _check(tmp_path / "r2")


def test_float32_and_resume(local_llama, fake_openai, tmp_path, monkeypatch):
    cfg = _cfg(local_llama, fake_openai, monkeypatch, student__dtype="float32")
    Orchestrator(cfg, tmp_path / "r3").run()
    cfg.autonomy.max_rounds = 3
    state = Orchestrator(cfg, tmp_path / "r3").run()
    assert state["status"] == "finished"
    assert (tmp_path / "r3/checkpoints/round_003/meta.json").exists()


def test_prompt_format_nemotron(local_llama, tmp_path):
    from slm.student import Student
    from slm.utils import get_logger

    cfg = load_config(ROOT / "config.yaml", simulate=False)
    cfg.student.model_id = local_llama
    cfg.training.gradient_checkpointing = False
    st = Student(cfg, tmp_path, get_logger("t"))
    st.load()
    p = st.prompt_text("Hola")
    assert p.startswith("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\ndetailed thinking off<|eot_id|>")
    assert p.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n")
    ex = st.encode_example("Hola", "Respuesta", 256)
    n_prompt = sum(1 for x in ex["labels"] if x == -100)
    assert ex["labels"][-1] == st.tokenizer.convert_tokens_to_ids("<|eot_id|>") and n_prompt > 5

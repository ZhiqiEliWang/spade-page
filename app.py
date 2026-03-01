from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from queue import Empty
from threading import Lock, Thread
from typing import Any, Dict, Iterator, Tuple

import gradio as gr
import torch
from transformers import pipeline as hf_pipeline
from transformers import TextIteratorStreamer
from transformers import __version__ as transformers_version


DETECTOR_MODEL_ID = os.getenv(
    "DETECTOR_MODEL_ID",
    "ZhiqiEliWang/qwen3_0.6b_psyscam_romance_ephishllm",
)
EXPLAINER_MODEL_ID = os.getenv(
    "EXPLAINER_MODEL_ID",
    "ZhiqiEliWang/qwen3_0.6b_explainer",
)
STOP_TOKEN = "<|im_end|>"
NUM_CTX = 4096
TEMPERATURE = 0.6
TOP_K = 20
TOP_P = 0.95
MAX_NEW_TOKENS_DETECTOR = int(os.getenv("MAX_NEW_TOKENS_DETECTOR", "2048"))
MAX_NEW_TOKENS_EXPLAINER = int(os.getenv("MAX_NEW_TOKENS_EXPLAINER", "512"))
USER_PLACEHOLDER = "<<__SPADE_USER_PROMPT__>>"


def _default_kv_cache_dir() -> Path:
    data_dir = Path("/data")
    if data_dir.exists() and os.access(data_dir, os.W_OK):
        return data_dir / "spade_kv_cache"
    return Path("/tmp/spade_kv_cache")


KV_CACHE_DIR = Path(os.getenv("KV_CACHE_DIR", str(_default_kv_cache_dir())))
ENABLE_DISK_KV_CACHE = os.getenv("ENABLE_DISK_KV_CACHE", "1") == "1"
WARMUP_ON_STARTUP = os.getenv("WARMUP_ON_STARTUP", "1") == "1"
KV_CACHE_SCHEMA_VERSION = os.getenv("KV_CACHE_SCHEMA_VERSION", "2")
# FORCE_CLEAN_KV_CACHE_ON_STARTUP = os.getenv("FORCE_CLEAN_KV_CACHE_ON_STARTUP", "0") == "1"
FORCE_CLEAN_KV_CACHE_ON_STARTUP = "1"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("spade")
_PROMPT_TEMPLATE_CACHE: Dict[Tuple[int, str, bool], str] = {}
_PREFIX_KV_CACHE: Dict[Tuple[int, str, bool], Tuple[Any, int, str]] = {}
_PROMPT_TEMPLATE_CACHE_LOCK = Lock()
_PREFIX_KV_CACHE_LOCK = Lock()
_MODEL_LOCKS: Dict[int, Lock] = {}
_MODEL_LOCKS_LOCK = Lock()

CUSTOM_CSS = """
:root {
  --bg: #f4f5f7;
  --surface: #ffffff;
  --text: #111827;
  --muted: #6b7280;
  --accent: #245fa8;
  --accent-hover: #1f4f8a;
  --border: #d9dde3;
  --focus: #245fa8;
  --focus-ring: rgba(36, 95, 168, 0.18);
}

html, body, .gradio-container {
  background: var(--bg) !important;
  color: var(--text) !important;
}

.gradio-container {
  font-family: "Helvetica Neue", Helvetica, Arial, sans-serif !important;
  font-size: 15px !important;
  line-height: 1.45 !important;
  color-scheme: light !important;
  --body-background-fill: var(--bg) !important;
  --body-background-fill-dark: var(--bg) !important;
  --body-text-color: var(--text) !important;
  --body-text-color-dark: var(--text) !important;
  --body-text-color-subdued: var(--muted) !important;
  --body-text-color-subdued-dark: var(--muted) !important;
  --background-fill-primary: var(--bg) !important;
  --background-fill-primary-dark: var(--bg) !important;
  --background-fill-secondary: var(--surface) !important;
  --background-fill-secondary-dark: var(--surface) !important;
  --block-background-fill: var(--surface) !important;
  --block-background-fill-dark: var(--surface) !important;
  --block-border-color: var(--border) !important;
  --block-border-color-dark: var(--border) !important;
  --block-title-text-color: var(--text) !important;
  --block-title-text-color-dark: var(--text) !important;
  --block-label-text-color: var(--text) !important;
  --block-label-text-color-dark: var(--text) !important;
  --input-background-fill: #ffffff !important;
  --input-background-fill-dark: #ffffff !important;
  --input-border-color: var(--border) !important;
  --input-border-color-dark: var(--border) !important;
  --input-placeholder-color: var(--muted) !important;
  --input-placeholder-color-dark: var(--muted) !important;
  --button-primary-background-fill: var(--accent) !important;
  --button-primary-background-fill-dark: var(--accent) !important;
  --button-primary-background-fill-hover: var(--accent-hover) !important;
  --button-primary-background-fill-hover-dark: var(--accent-hover) !important;
  --button-primary-border-color: var(--accent) !important;
  --button-primary-border-color-dark: var(--accent) !important;
  --button-primary-text-color: #ffffff !important;
  --button-primary-text-color-dark: #ffffff !important;
  --code-background-fill: #fbfbfc !important;
  --code-background-fill-dark: #fbfbfc !important;
}

body.dark .gradio-container,
.dark .gradio-container,
[data-theme="dark"] .gradio-container {
  color-scheme: light !important;
}

.gradio-container h1,
.gradio-container h2,
.gradio-container h3,
.gradio-container label {
  letter-spacing: -0.01em;
}

#app-shell {
  max-width: 1080px;
  margin: 0 auto;
  padding: 2rem 1rem 2.5rem;
}

.hero {
  margin-bottom: 1rem;
  padding: 0;
}

.hero h1 {
  margin: 0 0 0.5rem;
  font-size: 30px;
  font-weight: 600;
  color: var(--text) !important;
}

.hero-subtitle {
  margin: 0;
  color: var(--muted) !important;
  font-size: 15px;
}

.hero-meta {
  display: grid;
  gap: 0.35rem;
  margin-top: 0.85rem;
}

.hero-meta p {
  margin: 0;
  color: var(--muted) !important;
  font-size: 14px;
}

.hero-meta span {
  color: var(--text) !important;
  font-weight: 600;
  margin-right: 0.4rem;
}

.hero-meta code {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 0.08rem 0.32rem;
  background: #f7f8fa;
  color: #1f2937;
}

.section-card {
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  background: var(--surface) !important;
  box-shadow: 0 1px 2px rgba(17, 24, 39, 0.04) !important;
  padding: 0.9rem !important;
  margin-top: 0.95rem;
}

.input-card,
.examples-card {
  margin-top: 1rem;
}

#run-btn {
  margin-top: 0.6rem;
  border: 1px solid var(--accent) !important;
  background: var(--accent) !important;
  color: #fff !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  min-height: 40px !important;
}

#run-btn:hover {
  background: var(--accent-hover) !important;
}

#run-btn:focus-visible {
  outline: none !important;
  box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

#outputs-row {
  gap: 1rem;
}

.output-left,
.output-right {
  min-height: 300px;
}

.output-left .cm-editor,
.output-left .cm-scroller,
.output-right .prose,
.output-right .markdown {
  background: #fbfbfc !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text) !important;
}

.output-left .cm-scroller,
.output-left .cm-content,
.output-left .cm-line {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.output-right .prose,
.output-right .markdown {
  white-space: pre-wrap !important;
  overflow-wrap: anywhere !important;
  word-break: break-word !important;
}

.gradio-container .prose,
.gradio-container .prose *,
.gradio-container .markdown,
.gradio-container .markdown * {
  color: var(--text) !important;
}

.output-left .wrap,
.output-right .wrap {
  min-height: 240px;
}

.output-left label,
.output-right label {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 15px !important;
}

.examples-card .label-wrap span {
  color: var(--text) !important;
  font-weight: 600 !important;
  font-size: 15px !important;
}

.examples-card .dataset-item {
  color: var(--muted) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: #fafbfc !important;
}

.gradio-container textarea,
.gradio-container input[type="text"] {
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  background: #ffffff !important;
  color: var(--text) !important;
}

.gradio-container textarea:focus,
.gradio-container input[type="text"]:focus {
  border-color: var(--focus) !important;
  box-shadow: 0 0 0 3px var(--focus-ring) !important;
}

.gradio-container .message code,
.gradio-container code {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace !important;
}

@media (max-width: 900px) {
  #app-shell {
    padding: 1.15rem 0.75rem 1.5rem;
  }

  .hero h1 {
    font-size: 24px;
  }

  #outputs-row {
    display: flex;
    flex-direction: column;
  }
}
"""

DETECTOR_SYSTEM_PROMPT = """You are an expert in psychological manipulation and fraud detection.\n\nTASK: Analyze the message through the lens of persuasion techniques to determine if it's a scam.\n\nANALYTICAL FRAMEWORK - Psychological Techniques (PTs):\nThese are common persuasion methods used in both legitimate communication and scams. \nThe key is HOW they're deployed - legitimately or deceptively.\n\nAuthority and Impersonation: Authority and Impersonation | Tend to obey authorities and credible individuals | Person claimed to be calling for Finance America, claiming our home warranty was expired\nPhantom Riches: Phantom Riches | Visceral triggers of desire that override rationality | Your phone Number was randomly selected from the US database and you have won 18,087.71\nFear and Intimidation: Fear and Intimidation | Fear of loss and penalties | You will be arrested!\nLiking: Liking | Preference for saying \u201cyes\u201d to people they like | I am always available to help, and it\u2019s my pleasure to answer any questions you may have\nUrgency and Scarcity: Urgency and Scarcity | Sense of urgency and scarcity assign more value to items | We are currently in urgent need of 100 employees\nPretext and Trust: Pretext and Trust | Tendency to trust credible individuals | This is an urgent message for [MY NAME]. I\u2019m calling regarding a complaint scheduled to be filed out of [Our County Name]\nReciprocity: Reciprocity | Tendency to feel obliged to repay favors from others | We will send you a check to purchase equipment such as new apple laptop and iphone 14 and software\nConsistency: Consistency | Tendency to behave consistently with past behaviors | Starts with small asks (fill a form) and escalate to big asks (invest money)\nSocial Proof: Social Proof | Tendency to refer majority\u2019s behavior to guide own actions | Your resume has been recommended by many online recruitment companies\n\n\nANALYSIS METHOD:\nFor each PT you identify, ask:\n1. Is this technique present? (What specific evidence?)\n2. What is the apparent intent? (Inform, persuade, or deceive?)\n3. Is there verification possible? (Can claims be checked?)\n4. What action is requested? (Reasonable vs suspicious?)\n\nCLASSIFICATION PRINCIPLE:\nA scam typically combines multiple PTs to create a deceptive narrative that:\n- Cannot be verified through official channels\n- Requests irreversible actions (money, credentials)\n- Benefits from victim's emotional response over logical thinking\n\nLegitimate messages may use PTs but:\n- Can be verified independently\n- Follow normal business practices\n- Allow time for consideration\n\nAnalyze the message below. Output JSON with:\n- 'features': {PT_name: evidence_snippet} for all PTs (empty string if absent)\n- 'scam': 1 if deceptive pattern detected, 0 if legitimate use of persuasion\n"""
EXPLAINER_SYSTEM_PROMPT = """You are an expert at explaining scam detection decisions. Given a message with extracted psychological cues (PTs) and a scam classification, generate a concise explanation.

Output format:
- Write 2–3 cue lines: <Cue>: "<≤3-word quote>" → <plain meaning>.
- End with one Summary sentence describing the manipulation mechanism (no advice).

Allowed cues: Authority, Fear, Urgency, Pretext, Consistency, Reciprocity, Liking, Phantom Riches, Social Proof.

Output only the explanation, no extra text."""

@lru_cache(maxsize=1)
def get_models() -> Tuple[Any, Any]:
    has_cuda = torch.cuda.is_available()
    logger.info("Loading models. CUDA available: %s", has_cuda)

    pipeline_kwargs: Dict[str, Any] = {"trust_remote_code": True}
    if has_cuda:
        pipeline_kwargs["device_map"] = "auto"
        pipeline_kwargs["torch_dtype"] = torch.float16
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        # Explicit CPU mode to keep behavior stable on non-GPU Spaces.
        pipeline_kwargs["device"] = -1
        logger.info("Using CPU mode.")

    detector = hf_pipeline(
        "text-generation",
        model=DETECTOR_MODEL_ID,
        **pipeline_kwargs,
    )
    explainer = hf_pipeline(
        "text-generation",
        model=EXPLAINER_MODEL_ID,
        **pipeline_kwargs,
    )
    logger.info("Models loaded.")
    return detector, explainer


def _extract_text(generation_output: Any) -> str:
    if isinstance(generation_output, list) and generation_output:
        first = generation_output[0]
        if isinstance(first, dict):
            generated = first.get("generated_text", "")
            if isinstance(generated, list) and generated:
                last = generated[-1]
                if isinstance(last, dict):
                    return str(last.get("content", "")).strip()
            return str(generated).strip()
    if isinstance(generation_output, str):
        return generation_output.strip()
    return str(generation_output).strip()


def _extract_json_object(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    candidate = match.group(0)
    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass
    return {}


def _build_prompt(generator: Any, system_prompt: str, user_prompt: str, thinking: bool) -> str:
    tokenizer = generator.tokenizer
    cache_key = (id(tokenizer), system_prompt, thinking)
    with _PROMPT_TEMPLATE_CACHE_LOCK:
        template = _PROMPT_TEMPLATE_CACHE.get(cache_key)
        if template is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PLACEHOLDER},
            ]
            template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
            _PROMPT_TEMPLATE_CACHE[cache_key] = template
            logger.info("Cached prompt template for tokenizer id=%s", id(tokenizer))

    return template.replace(USER_PLACEHOLDER, user_prompt, 1)


def _move_inputs_to_generator_device(generator: Any, encoded: Any) -> Any:
    device = getattr(generator, "device", None)
    if device is None:
        return encoded
    # device_map="auto" commonly uses cuda:0 as entry device for inputs.
    if str(device) == "cpu":
        return encoded
    return encoded.to(device)


def _get_generator_device(generator: Any) -> torch.device:
    device = getattr(generator, "device", None)
    if device is None:
        return torch.device("cpu")
    return torch.device(str(device))


def _tensor_tree_map(obj: Any, fn: Any) -> Any:
    if torch.is_tensor(obj):
        return fn(obj)
    if isinstance(obj, tuple):
        return tuple(_tensor_tree_map(item, fn) for item in obj)
    if isinstance(obj, list):
        return [_tensor_tree_map(item, fn) for item in obj]
    if isinstance(obj, dict):
        return {k: _tensor_tree_map(v, fn) for k, v in obj.items()}
    return obj


def _move_past_key_values_to_device(past_key_values: Any, device: torch.device) -> Any:
    return _tensor_tree_map(past_key_values, lambda t: t.to(device))


def _cpu_clone_past_key_values(past_key_values: Any) -> Any:
    return _tensor_tree_map(past_key_values, lambda t: t.detach().to("cpu"))


def _clone_past_key_values_for_inference(past_key_values: Any) -> Any:
    # Never pass shared cache tensors directly into generate(); they may be mutated in-place.
    return _tensor_tree_map(past_key_values, lambda t: t.detach().clone())


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sanitize_key(key: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", key)


def _get_model_lock(model: Any) -> Lock:
    model_id = id(model)
    with _MODEL_LOCKS_LOCK:
        lock = _MODEL_LOCKS.get(model_id)
        if lock is None:
            lock = Lock()
            _MODEL_LOCKS[model_id] = lock
        return lock


def _get_prompt_parts(generator: Any, system_prompt: str, thinking: bool) -> Tuple[str, str]:
    tokenizer = generator.tokenizer
    cache_key = (id(tokenizer), system_prompt, thinking)
    with _PROMPT_TEMPLATE_CACHE_LOCK:
        template = _PROMPT_TEMPLATE_CACHE.get(cache_key)
        if template is None:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": USER_PLACEHOLDER},
            ]
            template = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking
            )
            _PROMPT_TEMPLATE_CACHE[cache_key] = template
            logger.info("Cached prompt template for tokenizer id=%s", id(tokenizer))

    if USER_PLACEHOLDER not in template:
        return template, ""
    return template.split(USER_PLACEHOLDER, 1)


def _disk_cache_paths(generator: Any, system_prompt: str, thinking: bool) -> Tuple[Path, Path]:
    model_name = getattr(generator.model.config, "_name_or_path", "unknown_model")
    tokenizer_name = getattr(generator.tokenizer, "name_or_path", "unknown_tokenizer")
    prompt_hash = _sha256_text(system_prompt)
    thinking_tag = "thinking1" if thinking else "thinking0"
    schema_tag = f"schema{KV_CACHE_SCHEMA_VERSION}"
    base_name = _sanitize_key(
        f"{model_name}__{tokenizer_name}__{schema_tag}__{thinking_tag}__{prompt_hash[:16]}"
    )
    return KV_CACHE_DIR / f"{base_name}.pt", KV_CACHE_DIR / f"{base_name}.meta.json"


def _load_prefix_kv_from_disk(
    generator: Any,
    system_prompt: str,
    prefix_hash: str,
    suffix_hash: str,
    thinking: bool,
) -> Tuple[Any, int] | None:
    if not ENABLE_DISK_KV_CACHE:
        return None

    pt_path, meta_path = _disk_cache_paths(generator, system_prompt, thinking)
    if not pt_path.exists() or not meta_path.exists():
        return None

    try:
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        expected = {
            "kv_cache_schema_version": KV_CACHE_SCHEMA_VERSION,
            "transformers_version": transformers_version,
            "system_prompt_hash": _sha256_text(system_prompt),
            "prefix_hash": prefix_hash,
            "suffix_hash": suffix_hash,
            "thinking": thinking,
        }
        for key, expected_value in expected.items():
            if meta.get(key) != expected_value:
                logger.info("[DEBUG] Disk KV cache metadata mismatch on %s; rebuilding cache.", key)
                return None

        payload = torch.load(pt_path, map_location="cpu")
        past_key_values = payload.get("past_key_values")
        prefix_len = int(payload.get("prefix_len", 0))
        if past_key_values is None or prefix_len <= 0:
            return None

        runtime_device = _get_generator_device(generator)
        past_key_values = _move_past_key_values_to_device(past_key_values, runtime_device)
        logger.info("[DEBUG] Loaded prefix KV cache from disk: %s", pt_path)
        return past_key_values, prefix_len
    except Exception as exc:
        logger.warning("[DEBUG] Failed to load disk KV cache (%s): %s", pt_path, exc)
        return None


def _save_prefix_kv_to_disk(
    generator: Any,
    system_prompt: str,
    prefix_hash: str,
    suffix_hash: str,
    past_key_values: Any,
    prefix_len: int,
    thinking: bool,
) -> None:
    if not ENABLE_DISK_KV_CACHE:
        return

    pt_path, meta_path = _disk_cache_paths(generator, system_prompt, thinking)
    model_name = getattr(generator.model.config, "_name_or_path", "unknown_model")
    tokenizer_name = getattr(generator.tokenizer, "name_or_path", "unknown_tokenizer")

    try:
        KV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cpu_past = _cpu_clone_past_key_values(past_key_values)
        payload = {
            "past_key_values": cpu_past,
            "prefix_len": prefix_len,
        }
        meta = {
            "created_at_unix": int(time.time()),
            "kv_cache_schema_version": KV_CACHE_SCHEMA_VERSION,
            "transformers_version": transformers_version,
            "model_name_or_path": model_name,
            "tokenizer_name_or_path": tokenizer_name,
            "system_prompt_hash": _sha256_text(system_prompt),
            "prefix_hash": prefix_hash,
            "suffix_hash": suffix_hash,
            "thinking": thinking,
        }
        torch.save(payload, pt_path)
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f)
        logger.info("[DEBUG] Saved prefix KV cache to disk: %s", pt_path)
    except Exception as exc:
        logger.warning("[DEBUG] Failed to save disk KV cache (%s): %s", pt_path, exc)


def _get_prefix_kv(generator: Any, system_prompt: str, thinking: bool) -> Tuple[Any, int, str]:
    model = generator.model
    tokenizer = generator.tokenizer
    cache_key = (id(model), system_prompt, thinking)
    with _PREFIX_KV_CACHE_LOCK:
        cached = _PREFIX_KV_CACHE.get(cache_key)
    if cached is not None:
        logger.info(
            "[DEBUG] Prefix KV cache source=memory_hit model_id=%s thinking=%s prefix_tokens=%s",
            id(model),
            thinking,
            cached[1],
        )
        return cached

    prefix, suffix = _get_prompt_parts(generator, system_prompt, thinking)
    prefix_hash = _sha256_text(prefix)
    suffix_hash = _sha256_text(suffix)
    logger.info(
        "[DEBUG] Prefix KV cache source=memory_miss model_id=%s thinking=%s; checking disk",
        id(model),
        thinking,
    )

    disk_cache = _load_prefix_kv_from_disk(
        generator,
        system_prompt,
        prefix_hash,
        suffix_hash,
        thinking,
    )
    if disk_cache is not None:
        past_key_values, prefix_len = disk_cache
        with _PREFIX_KV_CACHE_LOCK:
            _PREFIX_KV_CACHE[cache_key] = (past_key_values, prefix_len, suffix)
        logger.info(
            "[DEBUG] Prefix KV cache source=disk_hit model_id=%s thinking=%s prefix_tokens=%s",
            id(model),
            thinking,
            prefix_len,
        )
        with _PREFIX_KV_CACHE_LOCK:
            return _PREFIX_KV_CACHE[cache_key]

    logger.info(
        "[DEBUG] Prefix KV cache source=disk_miss model_id=%s thinking=%s; recomputing",
        id(model),
        thinking,
    )

    encoded_prefix = tokenizer(prefix, return_tensors="pt")
    encoded_prefix = _move_inputs_to_generator_device(generator, encoded_prefix)

    with torch.inference_mode():
        outputs = model(**encoded_prefix, use_cache=True)

    prefix_len = int(encoded_prefix["input_ids"].shape[1])
    past_key_values = outputs.past_key_values
    with _PREFIX_KV_CACHE_LOCK:
        _PREFIX_KV_CACHE[cache_key] = (past_key_values, prefix_len, suffix)
    _save_prefix_kv_to_disk(
        generator=generator,
        system_prompt=system_prompt,
        prefix_hash=prefix_hash,
        suffix_hash=suffix_hash,
        past_key_values=past_key_values,
        prefix_len=prefix_len,
        thinking=thinking,
    )
    logger.info(
        "[DEBUG] Prefix KV cache source=recompute model_id=%s thinking=%s prefix_tokens=%s",
        id(model),
        thinking,
        prefix_len,
    )
    with _PREFIX_KV_CACHE_LOCK:
        return _PREFIX_KV_CACHE[cache_key]


def _resolve_eos_ids(generator: Any) -> Any:
    tokenizer = generator.tokenizer
    eos_ids = []

    default_eos = getattr(tokenizer, "eos_token_id", None)
    if default_eos is not None:
        eos_ids.append(default_eos)

    stop_id = tokenizer.convert_tokens_to_ids(STOP_TOKEN)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if stop_id is not None and stop_id >= 0 and stop_id != unk_id and stop_id not in eos_ids:
        eos_ids.append(stop_id)

    if not eos_ids:
        return None
    if len(eos_ids) == 1:
        return eos_ids[0]
    return eos_ids


def _generate_text_stream(
    generator: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    thinking: bool,
    task_name: str = "generation",
    force_full_prompt: bool = False,
) -> Iterator[str]:
    eos_ids = _resolve_eos_ids(generator)
    pad_token_id = getattr(generator.tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(generator.tokenizer, "eos_token_id", None)

    t0 = time.perf_counter()
    model_lock = _get_model_lock(generator.model)
    retry_with_full_prompt = False

    logger.info(
        "[DEBUG] [%s] Generation started (max_new_tokens=%s, thinking=%s)",
        task_name,
        max_new_tokens,
        thinking,
    )

    with model_lock:
        generate_kwargs: Dict[str, Any]
        path_label = "prefix_kv"
        kv_cache_applied = True

        try:
            if force_full_prompt:
                raise RuntimeError("force_full_prompt enabled")

            past_key_values, prefix_len, suffix = _get_prefix_kv(generator, system_prompt, thinking)
            past_key_values = _clone_past_key_values_for_inference(past_key_values)
            dynamic_prompt = f"{user_prompt}{suffix}"
            encoded_dynamic = generator.tokenizer(dynamic_prompt, return_tensors="pt")
            encoded_dynamic = _move_inputs_to_generator_device(generator, encoded_dynamic)
            dynamic_len = int(encoded_dynamic["input_ids"].shape[1])
            if dynamic_len <= 0:
                raise RuntimeError("Dynamic prompt tokenized to 0 tokens on prefix-KV path.")

            attention_mask = torch.ones(
                (1, prefix_len + dynamic_len),
                dtype=encoded_dynamic["attention_mask"].dtype,
                device=encoded_dynamic["attention_mask"].device,
            )
            cache_position = torch.arange(
                prefix_len,
                prefix_len + dynamic_len,
                dtype=torch.long,
                device=encoded_dynamic["input_ids"].device,
            )
            logger.info(
                "[DEBUG] [%s] Prefix-KV input lengths: prefix_tokens=%s dynamic_tokens=%s cache_position_len=%s",
                task_name,
                prefix_len,
                dynamic_len,
                int(cache_position.numel()),
            )

            generate_kwargs = {
                "input_ids": encoded_dynamic["input_ids"],
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "cache_position": cache_position,
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "use_cache": True,
                "eos_token_id": eos_ids,
                "pad_token_id": pad_token_id,
            }
        except Exception as exc:
            path_label = "full_prompt"
            kv_cache_applied = False
            if not force_full_prompt:
                logger.warning("[DEBUG] KV-cache path failed, falling back to full prompt path: %s", exc)
            prompt = _build_prompt(generator, system_prompt, user_prompt, thinking)
            encoded_prompt = generator.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=NUM_CTX,
            )
            encoded_prompt = _move_inputs_to_generator_device(generator, encoded_prompt)
            generate_kwargs = {
                "input_ids": encoded_prompt["input_ids"],
                "attention_mask": encoded_prompt.get("attention_mask"),
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "temperature": TEMPERATURE,
                "top_k": TOP_K,
                "top_p": TOP_P,
                "use_cache": True,
                "eos_token_id": eos_ids,
                "pad_token_id": pad_token_id,
            }

        streamer = TextIteratorStreamer(
            generator.tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
            timeout=1.0,
        )
        generate_kwargs["streamer"] = streamer

        generation_error: Dict[str, Exception] = {}

        def _worker() -> None:
            try:
                with torch.inference_mode():
                    generator.model.generate(**generate_kwargs)
            except Exception as exc:
                generation_error["error"] = exc

        worker = Thread(target=_worker, daemon=True)
        worker.start()

        text = ""
        first_token_latency_ms: float | None = None
        stop_seen = False
        while True:
            try:
                chunk = next(streamer)
            except StopIteration:
                break
            except Empty:
                if worker.is_alive():
                    continue
                break

            if stop_seen:
                continue

            if first_token_latency_ms is None:
                first_token_latency_ms = (time.perf_counter() - t0) * 1000.0

            text += chunk
            if STOP_TOKEN in text:
                text = text.split(STOP_TOKEN, 1)[0]
                stop_seen = True
            yield text.strip()

        worker.join()
        if "error" in generation_error:
            elapsed = time.perf_counter() - t0
            logger.error(
                "[DEBUG] [%s] Generation failed after %.2fs (kv_cache_applied=%s, path=%s, first_token_latency_ms=%s, output_chars=%s): %s",
                task_name,
                elapsed,
                kv_cache_applied,
                path_label,
                f"{first_token_latency_ms:.1f}" if first_token_latency_ms is not None else "none",
                len(text),
                generation_error["error"],
            )
            if kv_cache_applied and not force_full_prompt:
                logger.warning("[DEBUG] [%s] Retrying generation with full_prompt path.", task_name)
                retry_with_full_prompt = True
            else:
                raise generation_error["error"]
        else:
            elapsed = time.perf_counter() - t0
            logger.info(
                "[DEBUG] [%s] Generation complete in %.2fs (max_new_tokens=%s, kv_cache_applied=%s, path=%s, first_token_latency_ms=%s, output_chars=%s)",
                task_name,
                elapsed,
                max_new_tokens,
                kv_cache_applied,
                path_label,
                f"{first_token_latency_ms:.1f}" if first_token_latency_ms is not None else "none",
                len(text),
            )

    if retry_with_full_prompt:
        yield from _generate_text_stream(
            generator=generator,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_new_tokens=max_new_tokens,
            thinking=thinking,
            task_name=f"{task_name}:full_prompt_retry",
            force_full_prompt=True,
        )
        return


def _generate_text(
    generator: Any,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
    thinking: bool,
    task_name: str = "generation",
) -> str:
    final = ""
    for partial in _generate_text_stream(
        generator,
        system_prompt,
        user_prompt,
        max_new_tokens,
        thinking,
        task_name=task_name,
    ):
        final = partial
    return final


def _build_detector_output(raw: str, empty_input: bool = False) -> Dict[str, Any]:
    if empty_input:
        return {
            "label": "invalid_input",
            "score": 0.0,
            "reasoning": "Input text is empty.",
            "raw_output": "",
        }

    parsed = _extract_json_object(raw)
    if parsed:
        parsed["raw_output"] = raw
        logger.info("Detector step completed with valid JSON.")
        return parsed

    logger.info("Detector step completed without valid JSON.")
    return {
        "label": "unknown",
        "score": None,
        "reasoning": "Detector did not return valid JSON.",
        "raw_output": raw,
    }


def run_detector_stream(text: str, task_name: str = "detector") -> Iterator[str]:
    cleaned = text.strip()
    if not cleaned:
        return

    detector, _ = get_models()
    user_prompt = f"Message: {cleaned}"
    yield from _generate_text_stream(
        detector,
        system_prompt=DETECTOR_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=MAX_NEW_TOKENS_DETECTOR,
        thinking=True,
        task_name=task_name,
    )


def run_detector(text: str) -> Dict[str, Any]:
    logger.info("Detector step started.")
    cleaned = text.strip()
    if not cleaned:
        logger.info("Detector step skipped: empty input.")
        return _build_detector_output(raw="", empty_input=True)

    raw = ""
    for partial in run_detector_stream(cleaned, task_name="detector"):
        raw = partial
    return _build_detector_output(raw=raw, empty_input=False)


def _fallback_explanation(detector_output: Dict[str, Any]) -> str:
    scam = detector_output.get("scam", detector_output.get("label", "unknown"))
    features = detector_output.get("features", {})
    non_empty_cues = []
    if isinstance(features, dict):
        for cue, evidence in features.items():
            if str(evidence).strip():
                non_empty_cues.append((cue, str(evidence).strip()))

    lines = [f"Summary: detector predicts {scam}."]
    if non_empty_cues:
        for cue, evidence in non_empty_cues[:3]:
            lines.append(f"{cue}: {evidence[:120]}")
    else:
        lines.append("No strong cues were provided by the detector output.")
    return "\n".join(lines)


def _normalize_visible_escapes(text: str) -> str:
    if not text:
        return text
    if "\\n" not in text and "\\r" not in text and "\\t" not in text:
        return text
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


def run_explainer_stream(
    detector_output: Dict[str, Any],
    simplified_prompt: bool = False,
    task_name: str = "explainer",
) -> Iterator[str]:
    _, explainer = get_models()
    user_prompt = (
        json.dumps(detector_output, ensure_ascii=True)
        if simplified_prompt
        else json.dumps(detector_output, ensure_ascii=True, indent=2)
    )
    max_tokens = (
        max(96, min(256, MAX_NEW_TOKENS_EXPLAINER))
        if simplified_prompt
        else MAX_NEW_TOKENS_EXPLAINER
    )
    yield from _generate_text_stream(
        explainer,
        system_prompt=EXPLAINER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_new_tokens=max_tokens,
        thinking=False,
        task_name=task_name,
    )


def run_explainer(text: str, detector_output: Dict[str, Any]) -> str:
    logger.info("Explainer step started.")
    del text  # explainer user prompt should be detector output only

    explanation = ""
    for partial in run_explainer_stream(detector_output, simplified_prompt=False, task_name="explainer"):
        explanation = partial
    if explanation.strip():
        return _normalize_visible_escapes(explanation)

    logger.warning("Explainer returned empty text; retrying with simplified prompt.")
    retry = ""
    for partial in run_explainer_stream(detector_output, simplified_prompt=True, task_name="explainer_retry"):
        retry = partial
    if retry.strip():
        return _normalize_visible_escapes(retry)

    logger.warning("Explainer retry also empty; using deterministic fallback explanation.")
    return _normalize_visible_escapes(_fallback_explanation(detector_output))


def pipeline(text: str) -> Iterator[Tuple[str, str]]:
    req_id = f"req-{int(time.time() * 1000)}"
    started = time.perf_counter()
    logger.info("[%s] Pipeline started.", req_id)

    detector_render = ""
    explainer_render = ""
    yield detector_render, explainer_render

    cleaned = text.strip()
    if not cleaned:
        detector_output = _build_detector_output(raw="", empty_input=True)
        detector_render = json.dumps(detector_output, ensure_ascii=True, indent=2)
        explainer_render = _normalize_visible_escapes(_fallback_explanation(detector_output))
        yield detector_render, explainer_render
        elapsed = time.perf_counter() - started
        logger.info("[%s] Pipeline finished in %.2fs", req_id, elapsed)
        return

    logger.info("[%s] Detector stream started.", req_id)
    detector_raw = ""
    for partial in run_detector_stream(cleaned, task_name=f"{req_id}:detector"):
        detector_raw = partial
        detector_render = detector_raw
        yield detector_render, explainer_render

    detector_output = _build_detector_output(raw=detector_raw, empty_input=False)
    detector_render = json.dumps(detector_output, ensure_ascii=True, indent=2)
    yield detector_render, explainer_render

    logger.info("[%s] Explainer stream started.", req_id)
    explanation = ""
    for partial in run_explainer_stream(
        detector_output,
        simplified_prompt=False,
        task_name=f"{req_id}:explainer",
    ):
        explanation = partial
        explainer_render = explanation
        yield detector_render, explainer_render

    if not explanation.strip():
        logger.warning("[%s] Explainer empty; retrying with simplified prompt.", req_id)
        retry = ""
        for partial in run_explainer_stream(
            detector_output,
            simplified_prompt=True,
            task_name=f"{req_id}:explainer_retry",
        ):
            retry = partial
            explainer_render = retry
            yield detector_render, explainer_render
        explanation = retry

    if not explanation.strip():
        logger.warning("[%s] Explainer still empty; using fallback.", req_id)
        explanation = _fallback_explanation(detector_output)
        explainer_render = explanation
        yield detector_render, explainer_render

    normalized_explanation = _normalize_visible_escapes(explanation)
    if normalized_explanation != explainer_render:
        explainer_render = normalized_explanation
        yield detector_render, explainer_render

    yield detector_render, explainer_render
    elapsed = time.perf_counter() - started
    logger.info("[%s] Pipeline finished in %.2fs", req_id, elapsed)


def _force_clean_kv_cache_dir() -> None:
    if not FORCE_CLEAN_KV_CACHE_ON_STARTUP:
        return
    if not KV_CACHE_DIR.exists():
        logger.info("[DEBUG] KV cache clean skipped: directory not found (%s).", KV_CACHE_DIR)
        return

    removed = 0
    for path in KV_CACHE_DIR.glob("*"):
        if path.suffix not in {".pt", ".json"}:
            continue
        try:
            path.unlink(missing_ok=True)
            removed += 1
        except Exception as exc:
            logger.warning("[DEBUG] Failed to remove KV cache file %s: %s", path, exc)
    logger.info("[DEBUG] Force-cleaned KV cache files on startup: removed=%s dir=%s", removed, KV_CACHE_DIR)


def warmup_prefix_kv_cache() -> None:
    if not WARMUP_ON_STARTUP:
        logger.info("Startup warmup disabled (WARMUP_ON_STARTUP=0).")
        return

    _force_clean_kv_cache_dir()
    logger.info("Startup warmup started. KV cache dir: %s", KV_CACHE_DIR)
    try:
        detector, explainer = get_models()
        _get_prefix_kv(detector, DETECTOR_SYSTEM_PROMPT, thinking=True)
        _get_prefix_kv(explainer, EXPLAINER_SYSTEM_PROMPT, thinking=False)
        logger.info("Startup warmup completed.")
    except Exception as exc:
        # Keep service available even if warmup fails.
        logger.warning("Startup warmup failed: %s", exc)


with gr.Blocks(title="SPADE Demo API", css=CUSTOM_CSS) as demo:
    with gr.Column(elem_id="app-shell"):
        gr.Markdown(
            f"""
            <div class="hero">
              <h1>SPADE Detector + Explainer</h1>
              <p class="hero-subtitle">
                A paper demo for psychological scam detection and explanation.
                The detector output appears at bottom-left and the explainer output at bottom-right.
              </p>
              <div class="hero-meta">
                <p><span>Detector model:</span> <code>{DETECTOR_MODEL_ID}</code></p>
                <p><span>Explainer model:</span> <code>{EXPLAINER_MODEL_ID}</code></p>
                <p><span>Runtime note:</span> this Space is currently running on CPU, so inference is slower than GPU.</p>
              </div>
            </div>
            """
        )

        with gr.Group(elem_classes=["section-card", "input-card"]):
            input_text = gr.Textbox(
                label="Input message x",
                lines=6,
                placeholder="Paste or type a message to analyze...",
            )
            run_btn = gr.Button("Run Pipeline", elem_id="run-btn")

        with gr.Row(elem_id="outputs-row", equal_height=True):
            with gr.Column(scale=1, min_width=360):
                with gr.Group(elem_classes=["section-card", "output-left"]):
                    detector_json = gr.Code(label="Detector Output", language="json")
            with gr.Column(scale=1, min_width=360):
                with gr.Group(elem_classes=["section-card", "output-right"]):
                    explainer_md = gr.Markdown(label="Explainer Output")

        with gr.Group(elem_classes=["section-card", "examples-card"]):
            gr.Examples(
                examples=[
                    "this is Oscar Walden with location services contacting you in reference to a pending claim being issued against your name requesting a signature I do need to make your work phone number QJR19680 is finalized there are no longer being an opportunity to contact the office processing your claim this sort of location requires a signature service to take place at your home worker just due to the Sonia and we're getting this matter I'm providing with the filing parties information one last time the number to contact is 877-595-5588 if the filing party isn't contacted I have no choice but to move forward with your order location you need to be available to provide a signature",
                ],
                inputs=input_text,
            )

    run_btn.click(
        fn=pipeline,
        inputs=input_text,
        outputs=[detector_json, explainer_md],
        api_name="pipeline",
    )


if __name__ == "__main__":
    warmup_prefix_kv_cache()
    demo.queue(default_concurrency_limit=8, max_size=64).launch()

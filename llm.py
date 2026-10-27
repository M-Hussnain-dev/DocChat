"""Streaming chat clients: Gemini (free key), any OpenAI-compatible API (Groq, OpenRouter, ...), and local Ollama.
Gemini is tuned for speed: thinking is switched off, one connection is reused, the last working model is tried first,
and models are discovered from your key with automatic fallback when one is retired or rate-limited."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Iterator

import httpx

PROVIDERS = ["Extractive (no AI, always free)", "Gemini (free key)", "OpenAI-compatible (Groq, OpenRouter)", "Ollama (local)"]
PRESETS = {  # model names change over time: edit them in the sidebar if a provider renames one
    "Groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "OpenRouter": ("https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct:free"),
    "Custom": ("", ""),
}
GEMINI_BASE = "https://generativelanguage.googleapis.com"
GEMINI_SPEEDS = ["Fast", "Accurate"]
GEMINI_FALLBACK = {"Fast": ("gemini-3.1-flash-lite", "gemini-2.5-flash-lite", "gemini-3.6-flash", "gemini-2.5-flash"),
                   "Accurate": ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash")}
GEMINI_SKIP = ("tts", "image", "live", "audio", "embedding", "robotics", "computer", "native", "learnlm", "gemma")
RETRY_CODES = (404, 429, 500, 503)
_client = httpx.Client(timeout=httpx.Timeout(120, connect=10), limits=httpx.Limits(max_keepalive_connections=8))
_models_cache: dict[str, tuple[float, list]] = {}
_good: dict[tuple, str] = {}
_no_think: set[str] = set()   # models that rejected the thinking setting: do not ask again


class LLMError(Exception):
    def __init__(self, msg: str, code: int = 0):
        super().__init__(msg)
        self.code = code


@dataclass(frozen=True)
class LLM:
    provider: str = PROVIDERS[0]
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    speed: str = "Fast"

    @property
    def enabled(self) -> bool:
        return not self.provider.startswith("Extractive")


def _error(code: int, body: str) -> str:
    try:
        j = json.loads(body)
        msg = j.get("error", j)
        msg = msg.get("message", msg) if isinstance(msg, dict) else msg
    except Exception:
        msg = body[:200]
    hint = {401: "Check your API key.", 403: "The key may lack access to this model.", 404: "Model or address not found: check the model name.",
            429: "Rate limit reached. Wait a minute or use another model."}.get(code, "")
    return f"{code}: {str(msg)[:220]} {hint}".strip()


def thinking_config(name: str) -> dict | None:
    """Ask Gemini not to 'think' first: much faster, and plenty for answers grounded in your documents."""
    if "2.5" in name:
        return {"thinkingBudget": 0}
    if name.startswith("gemini-3"):
        return {"thinkingLevel": "low" if "pro" in name else "minimal"}
    return None


def gemini_candidates(key: str, base: str = GEMINI_BASE, speed: str = "Fast") -> tuple[str, ...]:
    """Models to try, best first. Fast prefers Flash-Lite, Accurate the full Flash. Stable before preview."""
    names: list[str] = []
    hit = _models_cache.get(key)
    if hit and time.time() - hit[0] < 3600:
        names = hit[1]
    else:
        try:
            r = _client.get(f"{base.rstrip('/')}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=15)
            if r.status_code in (400, 401, 403):
                raise LLMError(_error(r.status_code, r.text), r.status_code)
            if r.status_code == 200:
                for m in r.json().get("models", []):
                    n = m.get("name", "").removeprefix("models/")
                    if "generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini") and "flash" in n \
                            and not any(x in n for x in GEMINI_SKIP):
                        names.append(n)
                _models_cache[key] = (time.time(), names)
        except httpx.HTTPError:
            pass

    def rank(n: str):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (bool(re.search(r"preview|exp", n)), ("lite" in n) == (speed == "Accurate"), -(float(v.group(1)) if v else 0.0), n)

    out = sorted(set(names), key=rank)[:5] or list(GEMINI_FALLBACK.get(speed, GEMINI_FALLBACK["Fast"]))
    good = _good.get((key, speed))
    if good in out:  # remember what worked last time: no wasted attempts
        out.remove(good)
        out.insert(0, good)
    return tuple(out)


def _lines(url: str, headers: dict, payload: dict) -> Iterator[str]:
    try:
        with _client.stream("POST", url, headers=headers, json=payload) as r:
            if r.status_code != 200:
                raise LLMError(_error(r.status_code, r.read().decode("utf-8", "replace")), r.status_code)
            yield from r.iter_lines()
    except httpx.HTTPError as e:
        raise LLMError(f"Could not reach the AI service ({type(e).__name__}).")


def parse_openai(line: str) -> str:
    if not line.startswith("data:"):
        return ""
    d = line[5:].strip()
    if not d or d == "[DONE]":
        return ""
    try:
        ch = (json.loads(d).get("choices") or [{}])[0]
    except ValueError:
        return ""
    return (ch.get("delta") or {}).get("content") or ""


def parse_gemini(line: str) -> str:
    if not line.startswith("data:"):
        return ""
    try:
        j = json.loads(line[5:].strip())
    except ValueError:
        return ""
    block = (j.get("promptFeedback") or {}).get("blockReason")
    if block:
        raise LLMError(f"Gemini blocked this request ({block}).")
    cands = j.get("candidates") or [{}]
    return "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts", []))


def parse_ollama(line: str) -> str:
    try:
        j = json.loads(line)
    except ValueError:
        return ""
    if j.get("error"):
        raise LLMError(str(j["error"])[:200])
    return (j.get("message") or {}).get("content", "")


def _gemini_stream(cfg: LLM, system: str, messages: list[dict], info: dict | None) -> Iterator[str]:
    if not cfg.api_key:
        raise LLMError("Add your free Gemini API key in the sidebar.")
    base = (cfg.base_url or GEMINI_BASE).rstrip("/")
    names = [cfg.model] if cfg.model and cfg.model != "auto" else list(gemini_candidates(cfg.api_key, base, cfg.speed))
    contents = [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages]
    last: LLMError = LLMError("No Gemini model could answer.")
    for name in names:  # retired or rate-limited model: try the next one before any text was sent
        for thinking in ((True, False) if thinking_config(name) and name not in _no_think else (False,)):
            cfg_json: dict = {"temperature": 0.2}
            if thinking:
                cfg_json["thinkingConfig"] = thinking_config(name)
            payload = {"systemInstruction": {"parts": [{"text": system}]}, "contents": contents, "generationConfig": cfg_json}
            started = False
            try:
                for line in _lines(f"{base}/v1beta/models/{name}:streamGenerateContent?alt=sse", {"x-goog-api-key": cfg.api_key}, payload):
                    piece = parse_gemini(line)
                    if piece:
                        if not started and info is not None:
                            info["model"] = name
                        started = True
                        _good[(cfg.api_key, cfg.speed)] = name
                        yield piece
                if started:
                    return
                last = LLMError("The AI service returned an empty answer.")
                break
            except LLMError as e:
                if started:
                    raise
                if e.code == 400 and thinking and "thinking" in str(e).lower():
                    _no_think.add(name)
                    continue  # this model rejects the setting: retry without it
                last = e
                if e.code not in RETRY_CODES:
                    raise
                break
    raise last


def stream_chat(cfg: LLM, system: str, messages: list[dict], info: dict | None = None) -> Iterator[str]:
    """messages: [{'role': 'user'|'assistant', 'content': str}]. Yields text pieces as they arrive.
    info['model'] is filled with the model that answered."""
    if cfg.provider.startswith("Gemini"):
        yield from _gemini_stream(cfg, system, messages, info)
        return
    if cfg.provider.startswith("OpenAI"):
        if not cfg.api_key or not cfg.base_url:
            raise LLMError("Add the API key and address for your provider in the sidebar.")
        url = f"{cfg.base_url.rstrip('/')}/chat/completions"
        payload = {"model": cfg.model, "stream": True, "temperature": 0.2,
                   "messages": [{"role": "system", "content": system}] + messages}
        parse, headers = parse_openai, {"Authorization": f"Bearer {cfg.api_key}"}
    elif cfg.provider.startswith("Ollama"):
        url = f"{(cfg.base_url or 'http://localhost:11434').rstrip('/')}/api/chat"
        payload = {"model": cfg.model, "stream": True, "options": {"temperature": 0.2},
                   "messages": [{"role": "system", "content": system}] + messages}
        parse, headers = parse_ollama, {}
    else:
        raise LLMError("No AI model selected.")
    if info is not None:
        info["model"] = cfg.model
    got = False
    for line in _lines(url, headers, payload):
        piece = parse(line)
        if piece:
            got = True
            yield piece
    if not got:
        raise LLMError("The AI service returned an empty answer.")


def complete(cfg: LLM, system: str, messages: list[dict]) -> str:
    return "".join(stream_chat(cfg, system, messages))


def ollama_models(host: str = "http://localhost:11434") -> list[str]:
    try:
        r = httpx.get(f"{host.rstrip('/')}/api/tags", timeout=1.5)
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return []

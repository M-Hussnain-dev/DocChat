"""Streaming chat clients: Gemini (free key), any OpenAI-compatible API (Groq, OpenRouter, ...), and local Ollama."""
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


GEMINI_FALLBACK = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.1-flash-lite", "gemini-2.5-flash")
GEMINI_SKIP = ("tts", "image", "live", "audio", "embedding", "robotics", "computer", "native", "learnlm", "gemma")
_models_cache: dict[str, tuple[float, tuple]] = {}


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


def gemini_candidates(key: str, base: str = GEMINI_BASE) -> tuple[str, ...]:
    """Flash models this key can use (model names change often): stable first, full before lite, newest first."""
    hit = _models_cache.get(key)
    if hit and time.time() - hit[0] < 3600:
        return hit[1]
    try:
        r = httpx.get(f"{base.rstrip('/')}/v1beta/models", params={"pageSize": 200}, headers={"x-goog-api-key": key}, timeout=15)
    except httpx.HTTPError:
        return GEMINI_FALLBACK
    if r.status_code in (400, 401, 403):
        raise LLMError(_error(r.status_code, r.text), r.status_code)
    if r.status_code != 200:
        return GEMINI_FALLBACK
    names = []
    for m in r.json().get("models", []):
        n = m.get("name", "").removeprefix("models/")
        if "generateContent" in m.get("supportedGenerationMethods", []) and n.startswith("gemini") and "flash" in n \
                and not any(x in n for x in GEMINI_SKIP):
            names.append(n)

    def rank(n: str):
        v = re.search(r"gemini-(\d+(?:\.\d+)?)", n)
        return (bool(re.search(r"preview|exp", n)), "lite" in n, -(float(v.group(1)) if v else 0.0), n)

    out = tuple(sorted(set(names), key=rank)[:5]) or GEMINI_FALLBACK
    _models_cache[key] = (time.time(), out)
    return out


def _lines(url: str, headers: dict, payload: dict) -> Iterator[str]:
    try:
        with httpx.stream("POST", url, headers=headers, json=payload, timeout=httpx.Timeout(120, connect=10)) as r:
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


def stream_chat(cfg: LLM, system: str, messages: list[dict]) -> Iterator[str]:
    """messages: [{'role': 'user'|'assistant', 'content': str}]. Yields text pieces as they arrive."""
    if cfg.provider.startswith("Gemini"):
        if not cfg.api_key:
            raise LLMError("Add your free Gemini API key in the sidebar.")
        base = (cfg.base_url or GEMINI_BASE).rstrip("/")
        names = [cfg.model] if cfg.model and cfg.model != "auto" else list(gemini_candidates(cfg.api_key, base))
        payload = {"systemInstruction": {"parts": [{"text": system}]},
                   "contents": [{"role": "user" if m["role"] == "user" else "model", "parts": [{"text": m["content"]}]} for m in messages],
                   "generationConfig": {"temperature": 0.2}}
        last: LLMError = LLMError("No Gemini model could answer.")
        for name in names:  # retired or rate-limited model: try the next one before any text was sent
            started = False
            try:
                for line in _lines(f"{base}/v1beta/models/{name}:streamGenerateContent?alt=sse",
                                   {"x-goog-api-key": cfg.api_key}, payload):
                    piece = parse_gemini(line)
                    if piece:
                        started = True
                        yield piece
                if started:
                    return
                last = LLMError("The AI service returned an empty answer.")
            except LLMError as e:
                if started or e.code not in (404, 429, 500, 503):
                    raise
                last = e
        raise last
    elif cfg.provider.startswith("OpenAI"):
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

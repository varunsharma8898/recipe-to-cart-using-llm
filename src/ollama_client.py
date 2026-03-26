"""
Minimal Ollama API client for ingredient extraction, normalisation, re-ranking.
Uses POST /api/chat with stream=false.

Model selection: config/ollama.yaml sets the default model. The environment variable
OLLAMA_MODEL overrides it for the process (e.g. for multi-model evaluation).
"""
import os
from pathlib import Path
from typing import Any, Optional

import requests


def get_ollama_config() -> dict[str, Any]:
    """Load config from config/ollama.yaml; fallback to defaults. OLLAMA_MODEL env overrides model."""
    root = Path(__file__).resolve().parent.parent
    config_path = root / "config" / "ollama.yaml"
    defaults = {"base_url": "http://localhost:11434", "model": "gemma3:4b", "timeout": 120}
    if not config_path.exists():
        cfg = dict(defaults)
    else:
        try:
            import yaml
            with open(config_path) as f:
                cfg = yaml.safe_load(f) or {}
            cfg = {**defaults, **cfg}
        except Exception:
            cfg = dict(defaults)
    if os.environ.get("OLLAMA_MODEL"):
        cfg["model"] = os.environ["OLLAMA_MODEL"]
    return cfg


def chat(
    messages: list[dict[str, str]],
    *,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: Optional[int] = None,
) -> str:
    """
    Send a chat request to Ollama and return the assistant reply text.

    Parameters
    ----------
    messages : list[dict]
        List of {"role": "user"|"system"|"assistant", "content": "..."}.
    model : str, optional
        Override config model.
    base_url : str, optional
        Override config base_url (e.g. http://localhost:11434).
    timeout : int, optional
        Request timeout in seconds.

    Returns
    -------
    str
        message.content from the response. Empty string on failure.
    """
    cfg = get_ollama_config()
    url = (base_url or cfg["base_url"]).rstrip("/") + "/api/chat"
    payload = {
        "model": model or cfg["model"],
        "messages": messages,
        "stream": False,
    }
    try:
        r = requests.post(
            url,
            json=payload,
            timeout=timeout or cfg.get("timeout", 120),
        )
        r.raise_for_status()
        data = r.json()
        msg = data.get("message") or {}
        return (msg.get("content") or "").strip()
    except Exception as e:
        raise RuntimeError(f"Ollama chat failed: {e}") from e


def generate(prompt: str, **kwargs: Any) -> str:
    """
    Single-turn generate: one user message. Convenience around chat().
    """
    return chat([{"role": "user", "content": prompt}], **kwargs)

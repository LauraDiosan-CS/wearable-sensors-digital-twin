"""Optional local LLM (Ollama) that the Clinical agent uses to rewrite its report as plain text.

Switched off unless the environment variable CARDIACTWIN_OLLAMA_MODEL names an installed Ollama model, e.g.
    CARDIACTWIN_OLLAMA_MODEL=llama3.2
OLLAMA_HOST changes the server address (default http://localhost:11434). Only the Python standard library is used.
"""
import json
import os
import urllib.request
from collections.abc import Callable

type LLM = Callable[[str, str], str]  # (instructions, text) -> answer

MODEL_ENV = "CARDIACTWIN_OLLAMA_MODEL"
DEFAULT_HOST = "http://localhost:11434"
TIMEOUT_S = 120


def ollama(model: str, host: str = DEFAULT_HOST) -> LLM:
    """An LLM function that asks the given Ollama model (temperature 0, so answers are repeatable)."""
    host = host if "://" in host else f"http://{host}"

    def ask(instructions: str, text: str) -> str:
        body = json.dumps({
            "model": model,
            "stream": False,
            "options": {"temperature": 0},
            "messages": [{"role": "system", "content": instructions},
                         {"role": "user", "content": text}],
        }).encode()
        request = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            return str(json.load(response)["message"]["content"])

    return ask


def from_environment() -> LLM | None:
    """The Ollama model named in CARDIACTWIN_OLLAMA_MODEL, or None if it is not set."""
    model = os.environ.get(MODEL_ENV)
    return ollama(model, os.environ.get("OLLAMA_HOST", DEFAULT_HOST)) if model else None

"""
Minimal Ollama HTTP client for the RAG chatbot.

Ollama exposes a local HTTP API on port 11434. This wrapper makes a
single ``/api/generate`` call so the rest of the pipeline does not need
to depend on the optional ``ollama`` Python package.

Usage::

    from ollama_client import OllamaGenerator
    gen = OllamaGenerator(model="phi3:3.8b")
    print(gen.generate("You are helpful.\nUser: hi\nAssistant:"))

If Ollama is not running, ``OllamaGenerator.available()`` returns
False and the pipeline can fall back to the extractive composer.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class OllamaGenerator:
    model: str = "phi3:3.8b"
    host: str = "http://localhost:11434"
    temperature: float = 0.2
    num_predict: int = 350  # max tokens to generate

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2):
                return True
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            return False

    def generate(self, prompt: str) -> str:
        data = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.host}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body.get("response", "").strip()


if __name__ == "__main__":
    gen = OllamaGenerator()
    if not gen.available():
        print("Ollama not running at http://localhost:11434.")
        print("Start it with: ollama serve   (and: ollama pull phi3:3.8b)")
    else:
        out = gen.generate(
            "You are a Shakespeare assistant.\n"
            "Question: Who is Hamlet?\n"
            "Answer:"
        )
        print(out)

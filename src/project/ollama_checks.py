"""Independent checks of the exact installed local Ollama models."""

import argparse
import json
import math
from pathlib import Path
import time
import traceback
from urllib.request import Request, urlopen

ENDPOINT = "http://localhost:11434"
CHAT_MODEL = "qwen3.8:27b"
EMBEDDING_MODEL = "nomic-embed-text:latest"
SENTENCE = "Example Corp reported revenue of EUR 500 million."


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = Request(ENDPOINT + path, data=data, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=600) as response:
        return json.load(response)


def check_models():
    installed = {model["name"] for model in request("/api/tags")["models"]}
    missing = {CHAT_MODEL, EMBEDDING_MODEL} - installed
    if missing:
        raise RuntimeError(f"Required local models missing: {sorted(missing)}")
    return sorted(installed)


def check_chat():
    response = request("/api/chat", {
        "model": CHAT_MODEL, "stream": False, "think": False,
        "messages": [{"role": "user", "content": f"Extract the company and revenue from: {SENTENCE} Return valid JSON only."}],
        "format": {
            "type": "object",
            "properties": {"company": {"type": "string"}, "revenue": {"type": "string"}},
            "required": ["company", "revenue"], "additionalProperties": False,
        },
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 128},
    })
    raw = response["message"]["content"]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Qwen JSON parsing failed: {error}; raw response: {raw!r}") from error
    if not isinstance(parsed, dict) or not all(isinstance(parsed.get(k), str) and parsed[k] for k in ("company", "revenue")):
        raise RuntimeError(f"Unexpected extraction response: {raw!r}")
    return parsed


def check_embeddings():
    response = request("/api/embed", {"model": EMBEDDING_MODEL, "input": SENTENCE})
    vector = response["embeddings"][0]
    if not vector or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in vector):
        raise RuntimeError("Ollama did not return a finite numeric embedding vector")
    return {"model": EMBEDDING_MODEL, "dimensions": len(vector), "numeric": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("check", choices=["chat", "embeddings", "all"], default="all", nargs="?")
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[2] / "outputs/experiments/ollama-preflight"
    output.mkdir(parents=True, exist_ok=True)
    result = {"endpoint": ENDPOINT, "chat_model": CHAT_MODEL, "embedding_model": EMBEDDING_MODEL}
    started = time.perf_counter()
    try:
        result["installed_models"] = check_models()
        print("Installed models:", result["installed_models"], flush=True)
        if args.check in ("embeddings", "all"):
            result["embedding_check"] = check_embeddings()
            print("Embedding check:", result["embedding_check"], flush=True)
        if args.check in ("chat", "all"):
            result["chat_check"] = check_chat()
            print("Chat check:", result["chat_check"], flush=True)
        result["status"] = "complete"
    except Exception as error:
        result.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        raise
    finally:
        result["runtime_seconds"] = round(time.perf_counter() - started, 3)
        (output / f"{args.check}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

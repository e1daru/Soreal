import json
import os
import socket
import sys
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse


def wait_for_tcp(url: str, label: str, timeout: float) -> None:
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    deadline = time.time() + timeout
    last_error = None

    print(f"[wait] waiting for {label} on {host}:{port}", flush=True)
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=2):
                print(f"[wait] {label} is reachable", flush=True)
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for {label} at {url}: {last_error}")


def wait_for_ollama(base_url: str, model: str, timeout: float) -> None:
    deadline = time.time() + timeout
    tags_url = f"{base_url.rstrip('/')}/api/tags"
    last_error = None

    print(f"[wait] waiting for Ollama HTTP API at {tags_url}", flush=True)
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(tags_url, timeout=5) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = [item.get("name", "") for item in payload.get("models", [])]
            print("[wait] Ollama API is reachable", flush=True)
            if model and not any(model == name or model in name for name in models):
                print(
                    f"[wait] warning: model '{model}' is not pulled yet. "
                    f"Run 'docker compose exec ollama ollama pull {model}' before using the app.",
                    flush=True,
                )
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
            time.sleep(1)

    raise TimeoutError(f"Timed out waiting for Ollama at {base_url}: {last_error}")


def main() -> int:
    timeout = float(os.getenv("SERVICE_WAIT_TIMEOUT", "180"))
    surreal_url = os.getenv("SURREAL_URL", "ws://surrealdb:8000/rpc")
    ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

    try:
        wait_for_tcp(surreal_url, "SurrealDB", timeout)
        wait_for_ollama(ollama_base_url, ollama_model, timeout)
    except TimeoutError as exc:
        print(f"[wait] {exc}", file=sys.stderr, flush=True)
        return 1

    print("[wait] dependencies are ready", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

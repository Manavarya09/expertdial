"""Live MoE routing viewer: drives llama-moe-spike in serve mode, streams routing to the browser.

    python3 app/serve.py --model models/OLMoE-1B-7B-0125-Instruct-Q4_K_M.gguf [--ngl 99] [--port 8777]

Steering is a plain text file ("layer expert sign") that the patched engine re-reads when it changes.
"""
import argparse
import json
import os
import queue
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
STEER_FILE = REPO / "work" / "live_steer.txt"

subscribers: list[queue.Queue] = []
subscribers_lock = threading.Lock()
steering: dict[tuple[int, int], str] = {}  # (layer, expert) -> "ban" | "force"
engine: subprocess.Popen | None = None
engine_lock = threading.Lock()
ready_message: dict | None = None


def broadcast(message: dict):
    with subscribers_lock:
        for q in list(subscribers):
            q.put(message)


def write_steer_file():
    lines = [f"{l} {e} {1 if state == 'force' else -1}" for (l, e), state in sorted(steering.items())]
    tmp = STEER_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
    tmp.replace(STEER_FILE)  # atomic, so the engine never reads a half-written file


def pump_engine(proc: subprocess.Popen):
    global ready_message
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue
        if message.get("type") == "ready":
            ready_message = message
        broadcast(message)


def start_engine(model: str, ngl: int, ctx: int, binary: str) -> subprocess.Popen:
    write_steer_file()
    env = {**os.environ, "MOE_MODE": "serve", "LLAMA_EXPERT_BIAS_FILE": str(STEER_FILE)}
    proc = subprocess.Popen(
        [binary, "-m", model, "-ngl", str(ngl), "-c", str(ctx)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        env=env, text=True, bufsize=1,
    )
    threading.Thread(target=pump_engine, args=(proc,), daemon=True).start()
    return proc


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def _send(self, code, body: bytes, content_type="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (ROOT / "index.html").read_bytes()
            return self._send(200, body, "text/html; charset=utf-8")
        if self.path == "/state":
            state = {"ready": ready_message, "steering": [[l, e, s] for (l, e), s in steering.items()]}
            return self._send(200, json.dumps(state).encode())
        if self.path == "/events":
            return self.stream_events()
        return self._send(404, b'{"error":"not found"}')

    def stream_events(self):
        q: queue.Queue = queue.Queue()
        with subscribers_lock:
            subscribers.append(q)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            if ready_message:
                self.wfile.write(f"data: {json.dumps(ready_message)}\n\n".encode())
                self.wfile.flush()
            while True:
                try:
                    message = q.get(timeout=15)
                    payload = f"data: {json.dumps(message)}\n\n"
                except queue.Empty:
                    payload = ": keepalive\n\n"
                self.wfile.write(payload.encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with subscribers_lock:
                if q in subscribers:
                    subscribers.remove(q)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length) or "{}")
        except json.JSONDecodeError:
            return self._send(400, b'{"error":"bad json"}')

        if self.path == "/chat":
            request = {"prompt": body.get("prompt", ""),
                       "n_predict": int(body.get("n_predict", 96)),
                       "reset": bool(body.get("reset", True))}
            with engine_lock:
                engine.stdin.write(json.dumps(request) + "\n")
                engine.stdin.flush()
            return self._send(202, b'{"ok":true}')

        if self.path == "/steer":
            if body.get("clear"):
                steering.clear()
            else:
                key = (int(body["layer"]), int(body["expert"]))
                state = body.get("state")
                if state in ("ban", "force"):
                    steering[key] = state
                else:
                    steering.pop(key, None)
            write_steer_file()
            payload = json.dumps({"steering": [[l, e, s] for (l, e), s in steering.items()]}).encode()
            broadcast({"type": "steering", "steering": [[l, e, s] for (l, e), s in steering.items()]})
            return self._send(200, payload)

        return self._send(404, b'{"error":"not found"}')


def main():
    global engine
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--ngl", type=int, default=99)
    parser.add_argument("--ctx", type=int, default=2048)
    parser.add_argument("--port", type=int, default=8777)
    parser.add_argument("--binary", default=str(REPO / "bin" / "llama-moe-spike"))
    args = parser.parse_args()

    engine = start_engine(args.model, args.ngl, args.ctx, args.binary)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"http://127.0.0.1:{args.port}  (model loading, the page will say when it's ready)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.terminate()


if __name__ == "__main__":
    main()

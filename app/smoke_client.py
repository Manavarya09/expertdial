"""Headless check: connect to the viewer, send a prompt, report routing and steering effects."""
import json
import sys
import threading
import urllib.request

BASE = "http://127.0.0.1:8777"
events, done = [], threading.Event()


def listen():
    with urllib.request.urlopen(BASE + "/events", timeout=600) as stream:
        for raw in stream:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            message = json.loads(line[6:])
            events.append(message)
            if message["type"] == "done":
                done.set()
                return


def post(path, payload):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=30).read()


threading.Thread(target=listen, daemon=True).start()
for extra in json.loads(sys.argv[1]) if len(sys.argv) > 1 else []:
    post("/steer", extra)
post("/chat", {"prompt": "Name three colors.", "n_predict": int(sys.argv[2]) if len(sys.argv) > 2 else 16})
done.wait(600)

text = "".join(e["text"] for e in events if e["type"] == "token")
tokens = [e for e in events if e["type"] == "token"]
print("output:", repr(text[:200]))
print("tokens:", len(tokens))
if tokens:
    routes = tokens[-1]["routes"]
    print("layers reported:", len(routes), "| layer 0 experts on last token:", routes.get("0"))
banned = [(s["layer"], s["expert"]) for s in (json.loads(sys.argv[1]) if len(sys.argv) > 1 else []) if s.get("state") == "ban"]
for layer, expert in banned:
    hit = sum(1 for t in tokens if expert in t["routes"].get(str(layer), []))
    print(f"banned L{layer}E{expert} appeared in {hit}/{len(tokens)} tokens")

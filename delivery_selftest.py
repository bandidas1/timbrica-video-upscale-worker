"""Self-check of the result delivery — no GPU, no network, no torch.

Why it exists: support 2026-11-01 (user 11231). An 85 s clip was upscaled and
encoded in ten minutes; then the network from RunPod to the RU origin failed,
the old loop waited 600 s per silent part, gave up on nginx's 408, never told
RunPod it was alive — and after 30 minutes the server cancelled the job and
the finished file died with the worker. `_deliver_result` replaced that loop;
this file pins what it must do, on a fake door with the server's own part
semantics (UpscaleTransfer::acceptResultPart) and a fake clock:

  python delivery_selftest.py

The handler is imported with stub modules for cv2 / numpy / torchvision /
runpod — delivery uses none of them, and CI has none of them.
"""
import importlib.util
import io
import json
import os
import sys
import tempfile
import types
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    for name in ("cv2", "numpy", "torchvision", "torchvision.transforms",
                 "torchvision.transforms.functional", "torchvision.transforms.functional_tensor"):
        sys.modules.setdefault(name, types.ModuleType(name))
    rp = types.ModuleType("runpod")
    rp.serverless = types.SimpleNamespace(progress_update=lambda *a, **k: None, start=lambda *a, **k: None)
    sys.modules["runpod"] = rp
    spec = importlib.util.spec_from_file_location("upsc_handler", os.path.join(HERE, "handler.py"))
    h = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(h)
    return h


H = _load()
MiB = 1024 * 1024


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(0.0, s)


class _Resp(io.BytesIO):
    def __init__(self, code, body):
        super().__init__(json.dumps(body).encode())
        self.status = code

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class Door:
    """The server's part door. `faults(n, method)` decides the fate of the
    n-th request: None (normal), 'drop' (nothing lands, the socket goes silent
    until the worker's stall timeout), 'lost' (the part lands, the answer
    never comes back), or an HTTP code (nothing lands, that code comes back)."""

    def __init__(self, clock, faults=None, mbps=50.0):
        self.clock, self.faults, self.mbps = clock, faults or (lambda n, m: None), mbps
        self.have = b""
        self.fid = None        # whose bytes `have` are (UpscaleTransfer: meta part_file)
        self.total = None
        self.n = 0
        self.sent = 0

    def __call__(self, url, method="GET", data=None, headers=None, timeout=1800):
        n = self.n
        self.n += 1
        fault = self.faults(n, method)
        if fault == "drop":
            self.clock.sleep(timeout)
            raise TimeoutError("timed out")
        if method == "GET":
            self.clock.sleep(0.2)
            done = self.total is not None and len(self.have) >= self.total
            asks = (headers or {}).get("X-Upsc-File", "")
            if not done and asks and self.fid and asks != self.fid:
                return _Resp(200, {"size": 0, "done": False})   # those bytes are another file's
            return _Resp(200, {"size": len(self.have), "done": done})
        if isinstance(data, (bytes, bytearray)):
            body = bytes(data)
            self.clock.sleep(len(body) / (self.mbps * MiB))
        else:
            pieces = []
            for piece in data:            # the wire: time passes piece by piece
                self.clock.sleep(len(piece) / (self.mbps * MiB))
                pieces.append(piece)
            body = b"".join(pieces)
        self.sent += len(body)
        if isinstance(fault, int):
            raise urllib.error.HTTPError(url, fault, "x", {}, io.BytesIO(json.dumps({"error": "e"}).encode()))
        off, tot = int(headers["X-Upsc-Offset"]), int(headers["X-Upsc-Total"])
        fid = headers.get("X-Upsc-File", "")
        if fid and self.fid and fid != self.fid:
            if off > 0:
                raise urllib.error.HTTPError(url, 409, "gap", {}, io.BytesIO(
                    json.dumps({"error": "part_out_of_order", "size": 0}).encode()))
            self.have = b""
        if fid:
            self.fid = fid
        self.total = tot
        if off > len(self.have):
            raise urllib.error.HTTPError(url, 409, "gap", {}, io.BytesIO(
                json.dumps({"error": "part_out_of_order", "size": len(self.have)}).encode()))
        if off + len(body) > len(self.have):
            self.have = self.have[:off] + body
        done = len(self.have) >= tot
        if fault == "lost":
            self.clock.sleep(timeout)
            raise TimeoutError("timed out")
        return _Resp(200, {"ok": True, "size": len(self.have), "done": done})


def run(size, faults=None, patience=3600, deadline_in=None, part=48 * MiB, mbps=50.0):
    clock = Clock()
    door = Door(clock, faults, mbps)
    beats = []
    with tempfile.NamedTemporaryFile(delete=False) as f:
        payload = b"\x00\x00\x00\x18ftypmp42" + os.urandom(size - 12)
        f.write(payload)
        path = f.name
    H._http = door
    try:
        out = H._deliver_result({"id": "j"}, "https://x/upscale-worker/out/j", path, part, {},
                                patience_s=patience,
                                deadline=(clock() + deadline_in) if deadline_in else None,
                                clock=clock, sleep=clock.sleep,
                                beat=lambda p: beats.append((clock(), dict(p))))
    finally:
        os.unlink(path)
    return out, door, beats, payload, clock


FAILS = []


def check(name, cond, detail=""):
    print(("ok   " if cond else "FAIL ") + name + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def max_gap(beats, end):
    ts = [t for t, _ in beats] + [end]
    return max(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else 0


SIZE = 134_558_340   # the 04.11 result of the very clip from the ticket

# 1. Clean link: three parts, no retries, the bytes on the server are the file.
out, door, beats, payload, clock = run(SIZE)
check("clean: delivered", out.get("uploaded") is True, out)
check("clean: server holds exactly the file", door.have == payload)
check("clean: no retries", out.get("delivery_tries") == 0, out)
check("clean: nothing sent twice", door.sent == SIZE, door.sent)

# 2. nginx 408 («the body stopped coming») is a hiccup, not a refusal.
out, door, beats, payload, _ = run(SIZE, faults=lambda n, m: 408 if (n == 1 and m == "PUT") else None)
check("408: retried and delivered", out.get("uploaded") is True and door.have == payload, out)
check("408: one retry counted", out.get("delivery_tries") == 1, out)

# 3. The part lands, the answer is lost: ask, do not resend what landed.
out, door, beats, payload, _ = run(SIZE, faults=lambda n, m: "lost" if n == 1 else None)
check("lost answer: delivered", out.get("uploaded") is True and door.have == payload, out)
check("lost answer: the landed part is not sent again", door.sent == SIZE, door.sent)

# 4. 410 — the job is gone on the server: stop at once, no retries.
out, door, beats, payload, _ = run(SIZE, faults=lambda n, m: 410 if m == "PUT" else None)
check("410: refused for good", str(out.get("error", "")).startswith("upload_rejected:http_410"), out)
check("410: no retries", out.get("delivery_tries") == 0 and door.n == 2, (out, door.n))

# 5. 01.11: a 35-minute black hole, then the link comes back.
def run_outage(minutes, patience=3600, deadline_in=None):
    clock = Clock()
    until = clock() + minutes * 60

    def faults(n, m):
        return "drop" if clock() < until else None

    door = Door(clock, faults)
    beats = []
    with tempfile.NamedTemporaryFile(delete=False) as f:
        payload = b"\x00\x00\x00\x18ftypmp42" + os.urandom(SIZE - 12)
        f.write(payload)
        path = f.name
    H._http = door
    try:
        out = H._deliver_result({"id": "j"}, "https://x/o", path, 48 * MiB, {}, patience_s=patience,
                                deadline=(clock() + deadline_in) if deadline_in else None,
                                clock=clock, sleep=clock.sleep, beat=lambda p: beats.append((clock(), dict(p))))
    finally:
        os.unlink(path)
    return out, door, beats, payload, clock


out, door, beats, payload, clock = run_outage(35)
check("35-min outage: delivered after the link returns", out.get("uploaded") is True and door.have == payload, out)
check("35-min outage: done within ~2 min of the link coming back",
      35 * 60 <= out.get("delivery_s", 0) <= 37 * 60, out.get("delivery_s"))
gap = max_gap(beats, clock())
check("35-min outage: RunPod hears from the worker at least every 2 min", gap <= 120, gap)
hbs = [p["hb"] for _, p in beats]
check("beats: the counter always moves", all(b > a for a, b in zip(hbs, hbs[1:])), hbs[:10])

# 6. Longer than patience: give up — but not one minute before patience.
out, door, beats, payload, clock = run_outage(200, patience=3600)
check("outage > patience: gives up", str(out.get("error", "")).startswith("upload_stalled:"), out)
check("outage > patience: not before patience", 3600 <= out.get("delivery_s", 0) <= 3600 + 200, out.get("delivery_s"))

# 7. The attempt's own deadline comes first.
out, door, beats, payload, clock = run_outage(200, patience=3600, deadline_in=1200)
check("deadline: ends BEFORE the attempt's ceiling, never a try that could outlive it",
      str(out.get("error", "")).startswith("upload_stalled:")
      and 1200 - 200 <= out.get("delivery_s", 0) <= 1200, out)

# 8. A slow but MOVING link (TSPU-like 150 KB/s): never «stalled»; beats from inside the part.
out, door, beats, payload, clock = run(20 * MiB, mbps=0.15, part=16 * MiB)
check("slow link: delivered", out.get("uploaded") is True and door.have == payload, out)
check("slow link: beats while a part is on the wire", any(p.get("wire", 0) > 0 for _, p in beats))

# 9. A 503 storm (FPM restarting) is ridden out.
out, door, beats, payload, _ = run(SIZE, faults=lambda n, m: 503 if 1 <= n <= 6 and m == "PUT" else None)
check("503 storm: delivered", out.get("uploaded") is True and door.have == payload, out)

# 10. The server already has the whole file (a restarted attempt): nothing to send.
clock = Clock()
door = Door(clock)
door.total, door.have = 100, b"x" * 100
H._http = door
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(b"x" * 100)
    p = f.name
out = H._deliver_result({"id": "j"}, "https://x/o", p, 48 * MiB, {}, clock=clock, sleep=clock.sleep, beat=lambda p: None)
os.unlink(p)
check("already delivered: no PUT", out.get("uploaded") is True and door.sent == 0, out)

# 10b. A restarted attempt meets bytes of the DEAD attempt's encode on the
# server: it must start over, never glue its tail onto another file's head.
clock = Clock()
door = Door(clock)
door.total, door.fid = SIZE + 777, "dead-attempt"
door.have = b"\x00\x00\x00\x18ftypmp42" + b"Z" * (48 * MiB - 12)
H._http = door
with tempfile.NamedTemporaryFile(delete=False) as f:
    payload = b"\x00\x00\x00\x18ftypmp42" + os.urandom(SIZE - 12)
    f.write(payload)
    p = f.name
out = H._deliver_result({"id": "j"}, "https://x/o", p, 48 * MiB, {}, clock=clock, sleep=clock.sleep, beat=lambda p: None)
os.unlink(p)
check("restart: another encode's bytes are not continued", out.get("uploaded") is True and door.have == payload, out)

# 10c. A part the server saw cut short (400 part_truncated) is the wire, not the job.
out, door, beats, payload, _ = run(SIZE, faults=lambda n, m: 400 if (n == 1 and m == "PUT") else None)
check("400 without a reason: final", str(out.get("error", "")).startswith("upload_rejected:http_400"), out)

# 11. The retry classes themselves.
check("classes", all(H._delivery_retryable(c) for c in (0, 408, 425, 429, 500, 502, 503, 504))
      and not any(H._delivery_retryable(c) for c in (400, 403, 404, 409, 410, 413, 422))
      and H._delivery_retryable(400, "part_truncated"))

print(f"\n{'FAILED: ' + ', '.join(FAILS) if FAILS else 'all delivery checks passed'}")
sys.exit(1 if FAILS else 0)

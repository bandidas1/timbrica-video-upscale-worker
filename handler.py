"""RunPod serverless worker: video upscale via Real-ESRGAN.

Contract (all transfer is out-of-band — the /run JSON stays control-plane
only, RunPod caps request bodies at 10 MiB):

  input = {
    "in_url":  signed GET  → source video bytes,
    "out_url": signed PUT  → where to upload the result mp4 (in PARTS, below),
    "scale": 2 | 4,
    "out_w", "out_h": int,         # optional: the page's size box (see _target_dims)
    "model": "general-x4v3" | "x4plus" | "flashvsr-turbo",
                                   # flashvsr-*: the restoration ENGINE (flashvsr_engine.py) —
                                   # a stream model that runs AT the box, not ×4-then-reduce
    "cq": 19,                      # NVENC constant-quality target
    "max_bps": int,                # bitrate ceiling → -maxrate (result-size bound)
    "part_bytes": int,             # size of one result part (server's proxy floor)
    "billed_duration_s": int,      # what the user paid for — we are the verifier
    "billed_h": int,
  }

  return {"ok", "uploaded", "bytes", "parts", "frames", "gpu_ms", "wall_ms",
          "encoder", "model", "fps", "src_h", "gpu", "mem_total_gb",
          "cgroup_limit_gb", "peak_rss_mb"}
  or     {"error": "..."}          # server releases the token hold

Billing-truth verification: the web tier prices the job from the CLIENT's
probe. This worker re-measures the real file and refuses (before any GPU
spend) when the actual duration/height exceeds what was billed — a mismatch
fails the job and the server refunds automatically. Honest users are never
blocked: their browser probe matches ffprobe within a rounding second.

Result delivery in PARTS (2026-09-05). The result used to go up as ONE PUT,
read whole into RAM first. Support ticket of that day: a 337 s ×4 job encoded
a 1.62 GB result and the origin's nginx refused the body at 512M — after 63
minutes of GPU — and the platform requeued the job for another full pass.
Parts of `part_bytes` (48 MiB by default) fit under every proxy on the path
(nginx 512M, Cloudflare 100 MB in front of the intl origin), stream from disk,
and a lost response is retried without re-sending what already landed: the
server answers every part with the byte count it holds, and a GET on the same
signed URL asks for it explicitly.
"""

import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request

# Before torch ever initialises CUDA in this process: the restoration engine's 4K stream died on
# a 24 GB card with 5.65 GiB «reserved but unallocated» (probe 05.11). Expandable segments change
# where the allocator puts blocks, never a computed value — the ESRGAN frame stays byte-identical.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import runpod

# torchvision >= 0.17 removed transforms.functional_tensor; basicsr 1.4.2
# still imports it. Register a shim BEFORE basicsr ever loads — this keeps
# the worker independent of any image-build sed patching.
import sys as _sys, types as _types
try:
    import torchvision.transforms.functional_tensor  # noqa: F401
except Exception:
    from torchvision.transforms import functional as _F
    _shim = _types.ModuleType('torchvision.transforms.functional_tensor')
    _shim.rgb_to_grayscale = _F.rgb_to_grayscale
    _sys.modules['torchvision.transforms.functional_tensor'] = _shim

MODELS = {
    # (weights file, netscale, is_compact)
    "general-x4v3": ("realesr-general-x4v3.pth", 4, True),
    # The same compact net WITHOUT the denoise training (Real-ESRGAN's denoise_strength 0).
    # The site's classic engine since 2026-11-06: on the upscaler bench (16 clips with an
    # original) it beats x4v3 on every fidelity axis — LPIPS x2 0.195 vs 0.236, x4 0.342 vs
    # 0.375, PSNR +1.8 / +0.6 dB, NIQE on real old clips 6.2 vs 8.6 — at the same cost (same
    # architecture); x4v3 smooths texture into a painted look. Shipped as a layer by
    # patch-flashvsr.yml (sha256-checked), baked by the Dockerfile on a rebuild.
    "general-wdn-x4v3": ("realesr-general-wdn-x4v3.pth", 4, True),
    "x4plus": ("RealESRGAN_x4plus.pth", 4, False),
}
# Stream engines: not per-frame networks but whole-video models with their own runner.
# key -> (module, mode). The module ships as its own image layer (patch-flashvsr.yml).
ENGINES = {
    "flashvsr-turbo": ("flashvsr_engine", "turbo"),
}
ENGINE_MAX_SHORT = 2160
WEIGHTS_DIR = os.environ.get("WEIGHTS_DIR", "/weights")
DEFAULT_PART_BYTES = 48 * 1024 * 1024

_upsampler_cache = {}


def _build_upsampler(model_key: str):
    if model_key in _upsampler_cache:
        return _upsampler_cache[model_key]
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from basicsr.archs.srvgg_arch import SRVGGNetCompact
    from realesrgan import RealESRGANer

    fname, netscale, compact = MODELS[model_key]
    if compact:
        net = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,
                              num_conv=32, upscale=4, act_type="prelu")
    else:
        net = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23,
                      num_grow_ch=32, scale=4)
    up = RealESRGANer(
        scale=netscale,
        model_path=os.path.join(WEIGHTS_DIR, fname),
        model=net,
        tile=512, tile_pad=10, pre_pad=0,
        half=True,
    )
    # tile_process() prints «Tile i/N» for every tile of every frame — tens of
    # thousands of lines per job in the worker's log, each a write on the hot
    # path. Its module-level print is ours to silence.
    import realesrgan.utils as _resr_utils
    _resr_utils.print = lambda *a, **k: None
    _upsampler_cache[model_key] = up
    return up


# ── One frame through the network, post-processing ON THE CARD (2026-11-04) ──
# RealESRGANer.enhance() hands the ×4 output to the CPU as FLOAT32 (a 640×480
# source: 2560×1920×3×4 B ≈ 59 MB per frame), then clamps, transposes, ×255,
# rounds and casts it in numpy, and the handler reduced it to the box with
# cv2 — all on a host CPU shared with neighbours. Measured on an RTX 4060 with
# the exact realesrgan 0.3.0 / basicsr 1.4.2 code and a real clip: 640×480 →
# 1440×1080 268–381 → 134–175 ms/frame, 1280×720 → 1920×1080 870–1295 →
# 366–558 ms/frame (×2.0–2.6), the encoder (NVENC or libx264) 1–4 ms either
# way. Prod jobs on RTX 4090 spent 480–1860 ms per 720p frame in enhance() —
# the same CPU work, starved. Here the same values are computed on the card
# (÷255 in float32 → fp16 network → clamp → area reduction → ×255 → round →
# uint8 → OpenCV's own area table) and 4.7 MB of uint8 cross the bus: the
# output is the old output (max |diff| 0). The legacy plain ×2 (no box) keeps
# its exact cv2 LANCZOS4 reduction.
_POST_ON_CARD = {"ok": True, "fallback": ""}
_AREA_TABS: dict = {}


def _area_tab(src_n: int, dst_n: int):
    """OpenCV's own INTER_AREA coverage table (computeResizeAreaTab), as a dense
    dst×src matrix: the reduction to the box on the card is then two matmuls and
    reproduces cv2.resize(…, INTER_AREA) — measured max |diff| 0 at 1440×1080 and
    1920×1080, where torch's 'area' (adaptive pooling bins) was off by up to 52."""
    W = np.zeros((dst_n, src_n), dtype=np.float32)
    scale = src_n / dst_n
    for dx in range(dst_n):
        fsx1 = dx * scale
        fsx2 = fsx1 + scale
        cell = min(scale, src_n - fsx1)
        sx2 = min(int(np.floor(fsx2)), src_n - 1)
        sx1 = min(int(np.ceil(fsx1)), sx2)
        if sx1 - fsx1 > 1e-3:
            W[dx, sx1 - 1] += (sx1 - fsx1) / cell
        W[dx, sx1:sx2] += 1.0 / cell
        if fsx2 - sx2 > 1e-3:
            W[dx, sx2] += min(min(fsx2 - sx2, 1.0), cell) / cell
    return W


def _area_to_box(out255, oh: int, ow: int):
    """out255: 1×3×H×W float, already rounded to uint8 values (as cv2 sees it)."""
    import torch
    _, _, H, W = out255.shape
    key = (H, W, oh, ow, str(out255.device))
    if key not in _AREA_TABS:
        _AREA_TABS.clear()   # one job, one shape
        _AREA_TABS[key] = (torch.from_numpy(_area_tab(H, oh)).to(out255.device),
                           torch.from_numpy(_area_tab(W, ow)).to(out255.device))
    wy, wx = _AREA_TABS[key]
    res = torch.einsum("yh,chw,xw->cyx", wy, out255[0], wx).unsqueeze(0)
    # cv2 takes its integer-factor fast path (rounding half up) when both
    # factors are whole; otherwise cvRound (half to even) — same as torch.round.
    if H % oh == 0 and W % ow == 0:
        return torch.floor(res + 0.5)
    return torch.round(res)


def _frame_bgr(upsampler, frame, ow: int, oh: int, to_box: bool, scale: int):
    if _POST_ON_CARD["ok"]:
        try:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(np.array(frame, copy=True)).to(upsampler.device, non_blocking=True)
                t = t.permute(2, 0, 1)[[2, 1, 0]].unsqueeze(0).float().div_(255.0)
                upsampler.img = t.half() if upsampler.half else t
                upsampler.mod_scale = None
                if upsampler.tile_size > 0:
                    upsampler.tile_process()
                else:
                    upsampler.process()
                out = upsampler.post_process().float().clamp_(0, 1).mul_(255.0).round_()
                if to_box and (out.shape[-1], out.shape[-2]) != (ow, oh):
                    out = _area_to_box(out, oh, ow).clamp_(0, 255)
                out = out.to(torch.uint8)[0][[2, 1, 0]]
                img = out.permute(1, 2, 0).contiguous().cpu().numpy()
            if not to_box and scale != 4:
                h, w = frame.shape[:2]
                img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LANCZOS4)
            return img
        except RuntimeError as e:
            # Out of card memory on a huge frame: the old path keeps the ×4
            # output in fp16 on the card — finish the job there, never fail it.
            _POST_ON_CARD["ok"] = False
            _POST_ON_CARD["fallback"] = f"{type(e).__name__}: {e}"[:160]
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
    out_img, _ = upsampler.enhance(frame, outscale=None if to_box else scale)
    if to_box and (out_img.shape[1], out_img.shape[0]) != (ow, oh):
        out_img = cv2.resize(out_img, (ow, oh), interpolation=cv2.INTER_AREA)
    return out_img


def _http(url: str, method: str = "GET", data=None, headers=None, timeout=1800):
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    return urllib.request.urlopen(req, timeout=timeout)


def _download(url: str, dst: str):
    with _http(url) as resp, open(dst, "wb") as f:
        while True:
            buf = resp.read(1 << 20)
            if not buf:
                break
            f.write(buf)


def _json_of(resp_or_err):
    try:
        return json.loads(resp_or_err.read().decode("utf-8", "replace") or "{}")
    except Exception:
        return {}


def _remote_size(url: str, timeout: float = 60, file_id: str = "") -> int:
    """How many bytes of the result the server already holds (GET on the
    signed result URL). -1 when unknown, -2 when the whole file is there.
    With `file_id` the server counts only bytes of THIS file (see _file_id)."""
    try:
        with _http(url, method="GET", timeout=timeout,
                   headers={"X-Upsc-File": file_id} if file_id else None) as resp:
            j = _json_of(resp)
            if j.get("done"):
                return -2
            return int(j.get("size", 0))
    except Exception:
        return -1


# ── Delivery of the finished file (2026-11-04) ──────────────────────────────
#
# WHY THIS IS NOT THE LOOP IT REPLACES. Support 2026-11-01 (user 11231, 85 s
# 720p → Full HD, 27 000 tokens): the clip was upscaled and encoded in about
# ten minutes, the page said «98 %, осталось около 2 мин», and half an hour
# later the server called the job stalled, cancelled it on RunPod — which
# stopped this worker together with the ONLY copy of the finished file — and
# refunded. The network from RunPod to the RU origin was failing that day
# (the same week the clip lane logged nginx 408s for every 50th request), and
# the old loop could not live through it:
#   · each part waited up to 600 s on a socket that had gone silent, while
#     nginx had dropped the request after 60 s — three such waits are the
#     server's 30-minute stall window, without one byte landing;
#   · 408 (nginx: «the body stopped coming»), 425 and 429 counted as a final
#     «no» and ended the delivery on the first one;
#   · nothing told RunPod — and through it the server — that the worker was
#     alive and trying, so «cannot reach you» and «hung» looked the same.
#
# The rules now:
#   · silence is detected in DELIVERY_STALL_S (90 s): past nginx's 60 s the
#     request is dead on the server's side anyway;
#   · a refusal is final only when retrying the same bytes cannot change it
#     (403 signature, 404/410 job gone, 413 size, 422 not an mp4); a broken
#     connection, 408, 425, 429 and 5xx are retried;
#   · the worker gives up only after `patience_s` WITHOUT ONE ACKNOWLEDGED
#     BYTE (the server's config sends it; the default covers an hour), and
#     never past the attempt's own deadline;
#   · every attempt, every pause and every DELIVERY_BEAT_S of a part in
#     flight report to RunPod: bytes the server holds, bytes of this part on
#     the wire, the try number and a counter that always moves. The beat
#     comes from THIS loop, not a side thread — a hung delivery stops beating,
#     a live one never does, so the server can tell the two apart.
DELIVERY_STALL_S = 90
DELIVERY_BEAT_S = 30
DELIVERY_PATIENCE_S = 3600


def _delivery_retryable(code: int, error: str = "") -> bool:
    """Worth sending the same bytes again: no answer at all (0 — reset,
    timeout, refused), 408 / 425 / 429, any 5xx, and a 400 whose reason is a
    body the server saw cut short (`part_truncated` — the wire, not the job).
    Every other 4xx is the server's final word about THIS job."""
    if code == 0 or code in (408, 425, 429) or code >= 500:
        return True
    return code == 400 and "truncated" in (error or "")


def _pause_s(attempt: int) -> float:
    """Quick on a single hiccup, at most every 30 s in a long outage."""
    return (2, 5, 10, 20)[attempt] if attempt < 4 else 30


def _file_id(path: str) -> str:
    """A fingerprint of the encoded file: size + its first and last MiB.

    A restarted attempt (RunPod requeues a job whose worker died) encodes the
    clip AGAIN — different bytes, very likely a different size — and used to
    resume at the offset the server held from the dead attempt: the head of
    one encode glued to the tail of another, an mp4 whose index points into
    the wrong bytes, sniffed «ftyp» and billed as delivered. With the id the
    server keeps only parts of the same file and tells a stranger to start
    over. Identical bytes give the same id, so a genuine resume stays one.
    """
    import hashlib

    size = os.path.getsize(path)
    h = hashlib.sha256(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(size - (1 << 20))
        h.update(f.read(1 << 20))
    return h.hexdigest()[:24]


class _CountingBody:
    """The part as an iterable: http.client sends it piece by piece, and each
    piece is a chance to beat while a slow part is still on the wire."""

    def __init__(self, data: bytes, on_piece):
        self.data, self.on_piece = data, on_piece

    def __iter__(self):
        step = 256 * 1024
        for i in range(0, len(self.data), step):
            piece = self.data[i:i + step]
            yield piece
            self.on_piece(i + len(piece))

    def __len__(self):
        return len(self.data)


# ── A progress beat must not outlive its job (2026-11-04) ──
# runpod.serverless.progress_update() posts {"status": "IN_PROGRESS", "output":
# …} to the SAME job-done URL the final result goes to, from a fire-and-forget
# thread with retries of its own (runpod 1.7.13: rp_progress.py → rp_http.py
# send_result). A beat still on the wire when the handler returns can land
# AFTER the result and turn a finished job back to IN_PROGRESS for good. Prod
# 04.11, job 13494d61…: the file was on our server and the row done 10 s after
# the start; RunPod kept IN_PROGRESS with the delivery's last beat as output
# while the worker sat IDLE and was then stopped. The delivery beats (hb) right
# before the return made that order likely. So beats go through threads we
# keep, and handler() waits for them, bounded, before RunPod gets the result.
_PROGRESS_THREADS: list = []
PROGRESS_DRAIN_S = 20.0


def _progress(job, payload) -> None:
    try:
        from runpod.serverless.modules.rp_progress import _thread_target as send
    except Exception:
        runpod.serverless.progress_update(job, payload)   # an SDK without it: as before
        return
    t = threading.Thread(target=send, args=(job, payload), daemon=True)
    t.start()
    _PROGRESS_THREADS.append(t)


def _drain_progress(limit_s: float = PROGRESS_DRAIN_S, clock=time.monotonic) -> int:
    """Wait for every beat still on the wire, at most `limit_s` in all.
    Returns how many were still in flight when the bound ran out."""
    end = clock() + limit_s
    for t in list(_PROGRESS_THREADS):
        t.join(max(0.0, end - clock()))
    alive = [t for t in _PROGRESS_THREADS if t.is_alive()]
    _PROGRESS_THREADS[:] = alive
    return len(alive)


def _deliver_result(job, url: str, src: str, part_bytes: int, extra_headers: dict | None,
                    patience_s: int = 0, deadline=None, clock=time.monotonic, sleep=time.sleep,
                    beat=None) -> dict:
    """Put the finished file on the server, in parts, through a bad network.

    Returns the job-output fragment: {"uploaded": True, "bytes", "parts",
    "delivery_tries", "delivery_s"} or {"error": ..., "bytes", "delivery_tries"}.
    `deadline` is a clock() value the delivery must end before (the attempt's
    own execution ceiling); `beat(payload)` reports to RunPod.
    """
    total = os.path.getsize(src)
    part_bytes = max(1 << 20, int(part_bytes or DEFAULT_PART_BYTES))
    parts = max(1, (total + part_bytes - 1) // part_bytes)
    patience = float(patience_s or DELIVERY_PATIENCE_S)
    fid = _file_id(src)
    started = progressed = clock()
    seq = [0]
    last_beat = [0.0]
    state = {"acked": 0, "tries": 0, "wire": 0, "last": ""}

    def _beat(force=False):
        t = clock()
        if not force and t - last_beat[0] < DELIVERY_BEAT_S:
            return
        last_beat[0] = t
        seq[0] += 1
        payload = {"pct": 98 + (1 if state["acked"] * 2 >= total else 0), "stage": "upload",
                   "sent": state["acked"], "wire": state["wire"], "total": total,
                   "try": state["tries"], "hb": seq[0]}
        if state["last"]:
            payload["last"] = state["last"][:60]
        try:
            (beat or (lambda p: _progress(job, p)))(payload)
        except Exception:
            pass   # a lost beat is not a lost file

    def _fail(reason: str) -> dict:
        _beat(force=True)
        return {"error": reason, "bytes": total, "delivery_tries": state["tries"],
                "delivery_s": int(clock() - started)}

    def _give_up_now(pause: float = 0.0) -> bool:
        t = clock()
        # Never start a try that, silent to the end, would outlive the attempt.
        return t - progressed >= patience or (
            deadline is not None and t + pause + DELIVERY_STALL_S >= deadline)

    known = _remote_size(url, timeout=DELIVERY_STALL_S, file_id=fid)
    if known == -2:
        return {"uploaded": True, "bytes": total, "parts": parts, "delivery_tries": 0, "delivery_s": 0}
    state["acked"] = max(0, known)
    _beat(force=True)
    attempt = 0
    with open(src, "rb") as f:
        while state["acked"] < total:
            offset = state["acked"]
            f.seek(offset)
            chunk = f.read(min(part_bytes, total - offset))
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(chunk)),
                "X-Upsc-Offset": str(offset),
                "X-Upsc-Total": str(total),
                "X-Upsc-Part": str(offset // part_bytes),
                "X-Upsc-Parts": str(parts),
                # How hard delivery was — rides into the job's meta sidecar.
                "X-Upsc-Tries": str(state["tries"]),
                "X-Upsc-File": fid,
            }
            if extra_headers:
                headers.update(extra_headers)

            def _on_piece(n):
                state["wire"] = n
                _beat()

            code, j = 0, {}
            try:
                with _http(url, method="PUT", data=_CountingBody(chunk, _on_piece), headers=headers,
                           timeout=DELIVERY_STALL_S) as resp:
                    code, j = resp.status, _json_of(resp)
            except urllib.error.HTTPError as e:
                code, j = e.code, _json_of(e)
            except Exception as e:
                code, j = 0, {"error": type(e).__name__}
            state["wire"] = 0

            if code == 200:
                if j.get("done"):
                    state["acked"] = total
                    break
                nxt = int(j.get("size", offset + len(chunk)))
                if nxt > offset:
                    state["acked"] = min(total, nxt)
                    progressed = clock()
                    attempt = 0
                    _beat(force=True)
                    continue
            elif code == 409 and j.get("error") == "part_out_of_order":
                # The server is the authority on where we are.
                nxt = int(j.get("size", 0))
                if nxt > state["acked"]:
                    progressed = clock()
                state["acked"] = max(0, min(total, nxt))
                continue
            elif not _delivery_retryable(code, str(j.get("error", ""))):
                state["last"] = f"http_{code}:{j.get('error', '')}"
                return _fail(f"upload_rejected:{state['last']}")

            # Transient: count it, wait, ask the server what it has, go again.
            state["tries"] += 1
            state["last"] = f"http_{code}:{j.get('error', '')}" if code else f"net:{j.get('error', '')}"
            if _give_up_now(_pause_s(attempt)):
                return _fail(f"upload_stalled:{state['last']}")
            pause_until = clock() + _pause_s(attempt)
            attempt += 1
            _beat(force=True)
            while clock() < pause_until:
                sleep(min(DELIVERY_BEAT_S, max(0.05, pause_until - clock())))
                _beat()
            known = _remote_size(url, timeout=DELIVERY_STALL_S, file_id=fid)
            _beat(force=True)
            if known == -2:
                state["acked"] = total
                break
            if known > state["acked"]:
                state["acked"] = min(total, known)
                progressed = clock()
                attempt = 0
            elif known >= 0:
                state["acked"] = known
    _beat(force=True)
    return {"uploaded": True, "bytes": total, "parts": parts, "delivery_tries": state["tries"],
            "delivery_s": int(clock() - started)}


def _target_dims(inp: dict, w: int, h: int):
    """The size the person chose on the page, or None for plain ×scale.

    The page's targets are boxes — «HD 720p», «Full HD», «До 4К» — and until
    2026-10-29 this worker only knew ×2 and ×4, so «До 4К» on a 1178×660 clip
    came out 2356×1320 instead of 3840×2152 (refund request №54). The model
    always runs ×4 and only then resizes, so the box costs the same GPU time as
    ×2: the frame leaves the network at ×4 and is reduced to the box here.

    The web tier has already validated the box; this is the worker refusing to
    trust a stale or hand-made one. Anything off — odd sides, smaller than the
    source, past ×4, a different shape (rotation read differently) — falls
    back to plain ×scale rather than failing a paid job.
    """
    try:
        tw, th = int(inp.get("out_w") or 0), int(inp.get("out_h") or 0)
    except (TypeError, ValueError):
        return None
    if tw <= 0 or th <= 0 or tw % 2 or th % 2:
        return None
    if tw < w or th < h or tw > w * 4 or th > h * 4:
        return None
    if abs(tw / th - w / h) > 0.02 * (w / h):
        return None
    return tw, th


def _ffprobe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", "-show_format", path],
        capture_output=True, text=True, timeout=120,
    )
    info = json.loads(out.stdout or "{}")
    v = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if not v:
        raise ValueError("no_video_stream")
    num, _, den = (v.get("avg_frame_rate") or "30/1").partition("/")
    try:
        fps = float(num) / float(den or 1)
    except (ValueError, ZeroDivisionError):
        fps = 30.0
    if not (0.5 < fps < 240):
        fps = 30.0
    duration = float(info.get("format", {}).get("duration") or v.get("duration") or 0)
    # Rotation lives in side data (or the legacy rotate tag), and ffmpeg's
    # decoder AUTOROTATES by default — the raw pipe delivers display-oriented
    # frames. Probe dims are the UNrotated ones, so without this swap every
    # phone portrait video reshaped into transposed garbage (proven on a
    # display_rotation=90 file: pipe emits 720×1280 while probe says 1280×720).
    rot = 0
    for sd in (v.get("side_data_list") or []):
        if sd.get("rotation") is not None:
            try:
                rot = int(round(float(sd["rotation"]))) % 360
            except (TypeError, ValueError):
                pass
    if not rot:
        try:
            rot = int((v.get("tags") or {}).get("rotate", 0)) % 360
        except (TypeError, ValueError):
            rot = 0
    w, h = int(v["width"]), int(v["height"])
    if rot in (90, 270):
        w, h = h, w
    return {
        "w": w, "h": h,
        "fps": fps, "duration": duration, "has_audio": a is not None,
    }


# NVENC session walls, measured on Ada (RTX 4060/4090, 2026-08-13): the H.264
# hardware encoder refuses either dimension above 4096 (a 2304×1296 source at
# ×2 = 4608 wide died exactly here — 26/26 historical failures were this wall),
# HEVC refuses above 8192. Same limits across the endpoint's whole GPU pool
# (Ampere 3090/A5000 and Ada 4090/L40S). The web tier sells nothing past
# HEVC_MAX_PX; the check here is the billing-truth belt for hand-crafted jobs.
H264_MAX_PX = 4096
HEVC_MAX_PX = 8192

_enc_list_cache = None


def _enc_available(name: str) -> bool:
    global _enc_list_cache
    if _enc_list_cache is None:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                                 capture_output=True, text=True, timeout=60)
            _enc_list_cache = out.stdout
        except Exception:
            _enc_list_cache = ""
    return name in _enc_list_cache


def _gpu_name() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip().splitlines()[0][:24] if out.stdout else "?"
    except Exception:
        return "?"


def _nvenc_lib() -> bool:
    """Is libnvidia-encode loadable here — the precondition of every NVENC
    session (see the Dockerfile: the `video` capability)."""
    try:
        import ctypes
        ctypes.CDLL("libnvidia-encode.so.1")
        return True
    except Exception:
        return False


def _mem_facts() -> dict:
    """What this worker may use. RunPod does not document RAM per serverless
    worker; the job result is the one durable place to learn it from."""
    facts = {}
    try:
        facts["mem_total_gb"] = round(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30, 1)
    except Exception:
        pass
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(p) as f:
                raw = f.read().strip()
            facts["cgroup_limit_gb"] = None if raw == "max" else round(int(raw) / 2**30, 1)
            break
        except Exception:
            continue
    try:
        import resource  # POSIX only; the local harness runs on Windows
        facts["peak_rss_mb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    except Exception:
        pass
    try:  # the card's own memory tells a MIG slice from the whole card (same name for both)
        import torch
        if torch.cuda.is_available():
            facts["gpu_mem_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1)
    except Exception:
        pass
    return facts


def _probe_encoder(name: str, ow: int, oh: int) -> bool:
    """1-frame null encode at the EXACT session dims. RunPod hosts are a
    lottery: one 2026-08-13 host refused every hevc_nvenc session ("No capable
    devices found" — driver/caps, not our params) while the previous worker
    encoded the same 4608×2592 fine. Only a live open on THIS host proves the
    encoder; ~1-2 s, before any GPU inference is spent."""
    args = ["ffmpeg", "-v", "error",
            "-f", "lavfi", "-i", f"color=c=black:s={ow}x{oh}:r=24:d=1",
            "-frames:v", "1", "-c:v", name]
    if name in ("h264_nvenc", "hevc_nvenc"):
        args += ["-preset", "p5"]
    args += ["-pix_fmt", "yuv420p", "-f", "null", "-"]
    try:
        out = subprocess.run(args, capture_output=True, timeout=45)
        if out.returncode != 0:
            # The reason travels to the job output: 2026-11-04 the ladder fell
            # to libx264 on an RTX 4090 host WITH libnvidia-encode loadable, and
            # nothing said why.
            _ENC_PROBE_ERRORS[name] = (out.stderr or b"").decode("utf-8", "replace").strip()[-300:]
        return out.returncode == 0
    except Exception as e:
        _ENC_PROBE_ERRORS[name] = f"{type(e).__name__}: {e}"[:300]
        return False


_ENC_PROBE_ERRORS: dict = {}


def _driver_version() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name,compute_cap",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
        return (out.stdout or "").strip().splitlines()[0][:80] if out.stdout else "?"
    except Exception:
        return "?"


def _bench_post(cfg: dict) -> dict:
    """{"probe": true, "bench_post": {"w", "h", "frames", "box": [ow, oh], "model"?}} —
    the same frames through the old path (enhance() + cv2) and _frame_bgr,
    interleaved on THIS host, so card, CPU and neighbours are the same for both.
    Synthetic frames (lavfi testsrc2): the network's cost does not depend on
    content. Reports ms/frame each way and whether the outputs are identical."""
    import torch
    w, h = int(cfg.get("w", 640)), int(cfg.get("h", 480))
    n = max(4, min(120, int(cfg.get("frames", 40))))
    ow, oh = (cfg.get("box") or [w * 2, h * 2])[:2]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=25",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, timeout=120).stdout
    model = cfg.get("model", "general-x4v3")
    if model not in MODELS:
        return {"error": "unknown_model", "model": model}
    up = _build_upsampler(model)
    ta = tb = 0.0
    same = True
    for i in range(n):
        f = np.frombuffer(raw[i * w * h * 3:(i + 1) * w * h * 3], dtype=np.uint8).reshape(h, w, 3)
        a = time.time()
        o1, _ = up.enhance(f, outscale=None)
        if (o1.shape[1], o1.shape[0]) != (ow, oh):
            o1 = cv2.resize(o1, (ow, oh), interpolation=cv2.INTER_AREA)
        o1 = np.ascontiguousarray(o1)
        torch.cuda.synchronize()
        b = time.time()
        o2 = _frame_bgr(up, f, ow, oh, True, 2)
        torch.cuda.synchronize()
        c = time.time()
        if i >= 2:   # the first frames carry cuDNN autotuning
            ta += b - a
            tb += c - b
        same = same and bool(np.array_equal(o1, o2))
    m = max(1, n - 2)
    try:
        cpus = len(os.sched_getaffinity(0))
    except Exception:
        cpus = os.cpu_count()
    return {"bench_post": True, "model": model, "gpu": _gpu_name(), "driver": _driver_version(), "cpus": cpus,
            "src": f"{w}x{h}", "box": f"{ow}x{oh}", "frames": n,
            "old_ms": round(1000 * ta / m, 1), "card_ms": round(1000 * tb / m, 1),
            "speedup": round(ta / tb, 2) if tb else None, "identical": same,
            "post_on_card": _POST_ON_CARD["ok"], "post_fallback": _POST_ON_CARD["fallback"] or None}


def _bench_engine(cfg: dict) -> dict:
    """{"probe": true, "bench_engine": {"model": "flashvsr-turbo", "w", "h", "box": [ow, oh], "frames"}} —
    a stream engine on THIS host without a job, an upload or a delivery: synthetic frames
    (lavfi testsrc2; a diffusion model's cost does not depend on content) through the engine
    twice, the second pass is the number. Answers the two things the web tier prices on and
    no machine of ours can measure — does the image run the engine on a pool card, and how
    many GPU seconds a megapixel of OUTPUT costs (`s_per_out_mpx_frame`)."""
    import importlib
    key = cfg.get("model", "flashvsr-turbo")
    if key not in ENGINES:
        return {"error": "unknown_engine", "model": key}
    w, h = int(cfg.get("w", 480)), int(cfg.get("h", 270))
    n = max(9, min(97, int(cfg.get("frames", 33))))
    ow, oh = (cfg.get("box") or [w * 4, h * 4])[:2]
    raw = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", f"testsrc2=size={w}x{h}:rate=25",
                          "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         capture_output=True, timeout=120).stdout
    frames = [np.frombuffer(raw[i * w * h * 3:(i + 1) * w * h * 3], dtype=np.uint8).reshape(h, w, 3) for i in range(n)]
    mod = importlib.import_module(ENGINES[key][0])
    mpx = ow * oh / 1e6
    # "variants": knob sets A/B'd on THIS host, in the given order (list the baseline again at
    # the end to see drift). The first variant's frames are the reference the others are
    # compared to — a knob that changes the picture beyond rounding is not a speed knob.
    variants = cfg.get("variants") or [{}]
    defaults = dict(mod.KNOBS)
    ref, warm_key, passes = None, None, []
    try:
        for var in variants:
            mod.KNOBS.update(defaults)
            mod.KNOBS.update({k: v for k, v in var.items() if k in defaults})
            de = int(var.get("decode_every", 8))
            for _ in range(1 if warm_key == de else 2):   # a new pipeline warms first
                kept = []
                t = time.time()
                facts = mod.upscale_stream(iter(frames), w, h, ow, oh, kept.append, mode=ENGINES[key][1], decode_every=de)
                s = time.time() - t
            warm_key = de
            row = {"variant": var, "s": round(s, 2), "s_per_out_mpx_frame": round(s / max(1, len(kept)) / mpx, 4), **facts}
            if ref is None:
                ref = kept
            elif len(kept) == len(ref):
                mx, mean = 0, 0.0
                for a, b in zip(kept, ref):
                    x = np.abs(a.astype(np.int16) - b.astype(np.int16))
                    mx, mean = max(mx, int(x.max())), mean + float(x.mean()) / len(ref)
                row["vs_first"] = {"max": mx, "mean": round(mean, 4)}
            passes.append(row)
    finally:
        mod.KNOBS.update(defaults)
    return {"bench_engine": True, "model": key, "gpu": _gpu_name(), "driver": _driver_version(),
            "src": f"{w}x{h}", "box": f"{ow}x{oh}", "frames": n, "passes": passes,
            "s_per_frame": round(passes[0]["s"] / max(1, passes[0]["frames_out"]), 4),
            "s_per_out_mpx_frame": passes[0]["s_per_out_mpx_frame"], **_mem_facts()}


def _encoder_probe_report() -> dict:
    """{"probe": true} — what THIS host can encode, and why not, without a
    model, a download or a token (the demucs worker's free probe, same idea)."""
    try:
        ver = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=20).stdout.splitlines()[0]
    except Exception as e:
        ver = f"? {e}"
    rep = {"probe": True, "gpu": _gpu_name(), "driver": _driver_version(), "ffmpeg": ver[:120],
           "nvenc_lib": _nvenc_lib(), "caps_env": os.environ.get("NVIDIA_DRIVER_CAPABILITIES", ""),
           "h264_nvenc_listed": _enc_available("h264_nvenc"), "hevc_nvenc_listed": _enc_available("hevc_nvenc"),
           "probes": {}}
    for name in ("h264_nvenc", "hevc_nvenc"):
        for w, h in ((256, 256), (1920, 1080)):
            _ENC_PROBE_ERRORS.pop(name, None)
            okk = _probe_encoder(name, w, h)
            rep["probes"][f"{name}@{w}x{h}"] = "ok" if okk else _ENC_PROBE_ERRORS.get(name, "failed")
    return rep


def _pick_encoder(ow: int, oh: int) -> str:
    """Hardware ladder, each rung proven on this host — software floor last.
    libx264 is slower (encode overlaps inference, which still dominates) but
    it ALWAYS opens; a paid job must never die on an encoder we could have
    avoided."""
    _ENC_PROBE_ERRORS.clear()   # a warm worker must not report the last job's reasons
    ladder = []
    if max(ow, oh) <= H264_MAX_PX and _enc_available("h264_nvenc"):
        ladder.append("h264_nvenc")
    if _enc_available("hevc_nvenc"):
        ladder.append("hevc_nvenc")
    for name in ladder:
        if _probe_encoder(name, ow, oh):
            return name
    return "libx264"


def _tail(path: str, n: int = 400) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()[-n:].strip()
    except Exception:
        return ""


def handler(job):
    """RunPod's entry: the job, then — before RunPod receives the result —
    every progress beat still in flight (see _progress)."""
    try:
        return _handle(job)
    finally:
        _drain_progress()


def _handle(job):
    t0 = time.time()
    t0_mono = time.monotonic()
    inp = job.get("input") or {}
    _POST_ON_CARD.update(ok=True, fallback="")   # a warm worker starts every job on the card
    if inp.get("probe") and inp.get("bench_post"):
        return _bench_post(inp["bench_post"])
    if inp.get("probe") and inp.get("bench_engine"):
        return _bench_engine(inp["bench_engine"])
    if inp.get("probe"):
        return _encoder_probe_report()
    in_url, out_url = inp.get("in_url"), inp.get("out_url")
    if not in_url or not out_url:
        return {"error": "missing_urls"}
    scale = 2 if int(inp.get("scale", 2)) == 2 else 4
    model_key = inp.get("model") if (inp.get("model") in MODELS or inp.get("model") in ENGINES) else "general-x4v3"
    engine = ENGINES.get(model_key)
    cq = max(10, min(35, int(inp.get("cq", 19))))
    billed_s = int(inp.get("billed_duration_s", 0))
    billed_h = int(inp.get("billed_h", 0))
    # Result-size bound the server quoted the user against. 0 = uncapped
    # (older servers); the encoder then behaves exactly as before.
    max_bps = max(0, int(inp.get("max_bps", 0) or 0))
    part_bytes = int(inp.get("part_bytes", DEFAULT_PART_BYTES) or DEFAULT_PART_BYTES)

    workdir = tempfile.mkdtemp(prefix="upsc_")
    src = os.path.join(workdir, "in.bin")
    dst = os.path.join(workdir, "out.mp4")

    _progress(job, {"pct": 1, "stage": "download"})
    try:
        _download(in_url, src)
    except Exception as e:
        code = getattr(e, "code", None) or type(e).__name__
        return {"error": f"download_failed:{code}"}

    try:
        meta = _ffprobe(src)
    except Exception:
        return {"error": "probe_failed"}

    # Billing truth: refuse before GPU spend if the client understated.
    if billed_s and meta["duration"] > billed_s * 1.02 + 2:
        return {"error": "duration_mismatch", "actual_s": round(meta["duration"], 1)}
    if billed_h and meta["h"] > billed_h * 1.05 + 8:
        return {"error": "resolution_mismatch", "actual_h": meta["h"]}

    w, h, fps = meta["w"], meta["h"], meta["fps"]
    ow, oh = w * scale, h * scale
    target = _target_dims(inp, w, h)
    if target:
        ow, oh = target
    total_frames = max(1, int(meta["duration"] * fps))

    # Fail BEFORE any GPU spend: past the HEVC wall no encoder on this fleet
    # can deliver the frame. The web tier refuses to sell these; reaching this
    # line means a stale client or a hand-crafted job.
    if max(ow, oh) > HEVC_MAX_PX:
        return {"error": "output_dims_unsupported", "out_w": ow, "out_h": oh}

    # The restoration engine is proven untiled up to 4K on 24 GB (its authors' claim, the
    # web tier sells no more): refuse past it BEFORE any GPU spend, as the HEVC wall above.
    if engine and min(ow, oh) > ENGINE_MAX_SHORT:
        return {"error": "engine_out_too_big", "out_w": ow, "out_h": oh, "model": model_key}

    upsampler = None if engine else _build_upsampler(model_key)
    encoder = _pick_encoder(ow, oh)

    # The fps filter pins the pipe to the SAME rate the encoder stamps.
    # Without it, VFR sources (phone/screen recordings) got CFR-duplicated by
    # the rawvideo muxer at the stream's nominal rate while the encoder timed
    # frames at avg_fps — a 2.96 s VFR clip came out 7.06 s (2.4× slow-motion,
    # guaranteed audio desync). Measured 2026-08-13 on a real VFR fixture.
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src,
         "-vf", f"fps={fps:.6f}",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
        stdout=subprocess.PIPE, bufsize=w * h * 3 * 4,
    )
    enc_args = ["ffmpeg", "-y", "-v", "error",
                "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
                "-r", f"{fps:.6f}", "-i", "-", "-i", src,
                "-map", "0:v", "-map", "1:a?",
                "-c:v", encoder]
    if encoder == "h264_nvenc":
        enc_args += ["-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    elif encoder == "hevc_nvenc":
        # hvc1 tag: QuickTime/Safari refuse the default hev1 sample entry.
        enc_args += ["-preset", "p5", "-rc", "vbr", "-cq", str(cq), "-b:v", "0",
                     "-tag:v", "hvc1"]
    else:
        enc_args += ["-preset", "veryfast", "-crf", str(cq)]
    if max_bps > 0:
        # Constant quality UNDER a ceiling: the same picture as before on
        # every clip the ceiling does not reach (measured 0.04–0.06 bits per
        # output pixel on real footage against a 0.10 budget), and a bounded
        # file on the ones it does — bounded is what the quote promised.
        enc_args += ["-maxrate", str(max_bps), "-bufsize", str(2 * max_bps)]
    enc_args += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", dst]
    # Encoder stderr goes to a FILE (a PIPE nobody drains deadlocks ffmpeg at
    # 64 KB) so an encode death reports its real reason instead of surfacing
    # as a bare BrokenPipeError on our next stdin write.
    enc_log = os.path.join(workdir, "enc.log")
    enc_err_f = open(enc_log, "w")
    try:
        enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE, stderr=enc_err_f)
    finally:
        enc_err_f.close()  # the child holds its own copy of the fd

    frame_bytes = w * h * 3
    counts = {"frames": 0, "last_pct": -1}
    gpu_ms = 0.0
    engine_facts = {}

    def source_frames():
        while True:
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                return
            yield np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)

    def emit(out_img):
        enc.stdin.write(np.ascontiguousarray(out_img).tobytes())
        counts["frames"] += 1
        n = counts["frames"]
        pct = 3 + int(93 * n / max(n, total_frames))
        if pct != counts["last_pct"] and (n % 15 == 0 or pct >= 96):
            counts["last_pct"] = pct
            _progress(job, {"pct": min(96, pct), "stage": "upscale"})

    try:
        if engine:
            import importlib
            mod = importlib.import_module(engine[0])
            g0 = time.time()
            engine_facts = mod.upscale_stream(source_frames(), w, h, ow, oh, emit, mode=engine[1])
            gpu_ms = (time.time() - g0) * 1000
        else:
            for frame in source_frames():
                g0 = time.time()
                # ×4 down to the box: area averaging is the reduction that does not
                # alias fine detail the model just made (on the card, _frame_bgr).
                out_img = _frame_bgr(upsampler, frame, ow, oh, bool(target), scale)
                gpu_ms += (time.time() - g0) * 1000
                emit(out_img)
    except BrokenPipeError:
        # The encoder died under us — its stderr has the real reason (NVENC
        # session-open refusals land here: the write, not the spawn, fails).
        try:
            dec.kill(); enc.kill()
        except Exception:
            pass
        return {"error": f"encode_failed:{encoder}:{_gpu_name()}:{_tail(enc_log, 180)}",
                "frames": counts["frames"], "encoder": encoder, "gpu": _gpu_name()}
    except Exception as e:
        try:
            dec.kill(); enc.kill()
        except Exception:
            pass
        return {"error": f"upscale_failed:{type(e).__name__}", "frames": counts["frames"],
                "model": model_key, "gpu": _gpu_name(), "detail": str(e)[:200]}
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass

    # Said out loud: the bar used to freeze at 97 % for as long as the encoder
    # took to flush and rewrite the file for faststart — minutes on a
    # multi-GB result — with nothing to tell "finishing" from "hung".
    _progress(job, {"pct": 97, "stage": "finalize"})
    dec.wait(timeout=60)
    try:
        enc.stdin.close()
    except Exception:
        pass
    if enc.wait(timeout=1800) != 0:
        return {"error": f"encode_failed:{encoder}:{_gpu_name()}:{_tail(enc_log, 180)}",
                "encoder": encoder, "gpu": _gpu_name()}
    frames = counts["frames"]
    if frames == 0 or not os.path.exists(dst) or os.path.getsize(dst) < 4096:
        return {"error": "empty_output", "frames": frames}

    # Delivery: see _deliver_result. The attempt's own ceiling (RunPod's
    # executionTimeout, sent by the server as `deadline_s`) bounds the wait;
    # a minute is left for the result to be posted.
    deadline_s = int(inp.get("deadline_s", 0) or 0)
    sent = _deliver_result(
        job, out_url, dst, part_bytes,
        # Encoder + GPU ride the parts as headers → the server drops them into
        # a meta.json sidecar → srv_done telemetry. RunPod's own job output
        # expires in hours; this is the only durable per-job fleet map.
        {"X-Upsc-Encoder": encoder, "X-Upsc-Gpu": _gpu_name(), "X-Upsc-Out": f"{ow}x{oh}"},
        patience_s=int(inp.get("deliver_patience_s", 0) or 0),
        deadline=(t0_mono + deadline_s - 60) if deadline_s > 0 else None,
    )
    if sent.get("error"):
        return {**sent, "frames": frames, "encoder": encoder, "gpu": _gpu_name()}
    size, parts = sent["bytes"], sent["parts"]

    return {
        "ok": True, "uploaded": True, "bytes": size, "parts": parts, "frames": frames,
        "gpu_ms": int(gpu_ms), "wall_ms": int((time.time() - t0) * 1000),
        "encoder": encoder, "model": model_key, "fps": round(fps, 3),
        "src_h": h, "gpu": _gpu_name(), "max_bps": max_bps,
        "out_w": ow, "out_h": oh, "target": bool(target),
        "delivery_tries": sent.get("delivery_tries", 0), "delivery_s": sent.get("delivery_s", 0),
        "nvenc_lib": _nvenc_lib(),
        "post_on_card": _POST_ON_CARD["ok"], "post_fallback": _POST_ON_CARD["fallback"] or None,
        "engine": engine_facts or None,
        "encoder_probe_err": {k: v[-160:] for k, v in _ENC_PROBE_ERRORS.items()} if encoder == "libx264" else None,
        **_mem_facts(),
    }


if __name__ == "__main__":
    # Guarded so the module can be IMPORTED by the local end-to-end harness
    # (real ffmpeg, real NVENC, a stub model, the real Laravel part ingest)
    # without starting the RunPod loop. The image runs `python -u /handler.py`.
    runpod.serverless.start({"handler": handler})

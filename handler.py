"""RunPod serverless worker: video upscale via Real-ESRGAN.

Contract (all transfer is out-of-band — the /run JSON stays control-plane
only, RunPod caps request bodies at 10 MiB):

  input = {
    "in_url":  signed GET  → source video bytes,
    "out_url": signed PUT  → where to upload the result mp4 (in PARTS, below),
    "scale": 2 | 4,
    "model": "general-x4v3" | "x4plus",
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
import time
import urllib.error
import urllib.request

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
    "x4plus": ("RealESRGAN_x4plus.pth", 4, False),
}
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
    _upsampler_cache[model_key] = up
    return up


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


class UploadRejected(Exception):
    """The server said no for good (4xx other than out-of-order) — retrying
    the same bytes cannot change the answer."""


def _json_of(resp_or_err):
    try:
        return json.loads(resp_or_err.read().decode("utf-8", "replace") or "{}")
    except Exception:
        return {}


def _remote_size(url: str) -> int:
    """How many bytes of the result the server already holds (GET on the
    signed result URL). -1 when unknown — the caller then starts from 0 and
    lets `part_out_of_order` correct it."""
    try:
        with _http(url, method="GET", timeout=60) as resp:
            j = _json_of(resp)
            if j.get("done"):
                return int(j.get("size", 0)) or -2
            return int(j.get("size", 0))
    except Exception:
        return -1


def _upload_parts(url: str, src: str, part_bytes: int, extra_headers: dict | None,
                  on_progress=None) -> tuple[int, int]:
    """PUT `src` to `url` in parts. Returns (bytes, parts)."""
    total = os.path.getsize(src)
    part_bytes = max(1 << 20, int(part_bytes or DEFAULT_PART_BYTES))
    parts = max(1, (total + part_bytes - 1) // part_bytes)
    offset = max(0, _remote_size(url))
    if offset == -2:  # a previous attempt already delivered the whole file
        return total, parts
    misses = 0
    with open(src, "rb") as f:
        while offset < total:
            f.seek(offset)
            chunk = f.read(min(part_bytes, total - offset))
            idx = offset // part_bytes
            headers = {
                "Content-Type": "application/octet-stream",
                "Content-Length": str(len(chunk)),
                "X-Upsc-Offset": str(offset),
                "X-Upsc-Total": str(total),
                "X-Upsc-Part": str(idx),
                "X-Upsc-Parts": str(parts),
            }
            if extra_headers:
                headers.update(extra_headers)
            try:
                with _http(url, method="PUT", data=chunk, headers=headers, timeout=600) as resp:
                    j = _json_of(resp)
                misses = 0
                if j.get("done"):
                    offset = total
                    break
                nxt = int(j.get("size", offset + len(chunk)))
                # The server is the authority on where we are — never advance
                # past what it acknowledged, never re-send what it holds.
                offset = nxt if nxt > offset else offset + len(chunk)
            except urllib.error.HTTPError as e:
                j = _json_of(e)
                if e.code == 409 and j.get("error") == "part_out_of_order":
                    offset = int(j.get("size", 0))
                    continue
                if 400 <= e.code < 500:
                    raise UploadRejected(f"http_{e.code}:{j.get('error', '')}")
                misses += 1
            except Exception:
                misses += 1
            if misses:
                if misses > 6:
                    raise RuntimeError("upload_gave_up")
                time.sleep(min(60, 2 ** misses))
                known = _remote_size(url)
                if known == -2:
                    return total, parts
                if known >= 0:
                    offset = known
            if on_progress:
                on_progress(offset, total)
    return total, parts


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
        return subprocess.run(args, capture_output=True, timeout=45).returncode == 0
    except Exception:
        return False


def _pick_encoder(ow: int, oh: int) -> str:
    """Hardware ladder, each rung proven on this host — software floor last.
    libx264 is slower (encode overlaps inference, which still dominates) but
    it ALWAYS opens; a paid job must never die on an encoder we could have
    avoided."""
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
    t0 = time.time()
    inp = job.get("input") or {}
    in_url, out_url = inp.get("in_url"), inp.get("out_url")
    if not in_url or not out_url:
        return {"error": "missing_urls"}
    scale = 2 if int(inp.get("scale", 2)) == 2 else 4
    model_key = inp.get("model") if inp.get("model") in MODELS else "general-x4v3"
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

    runpod.serverless.progress_update(job, {"pct": 1, "stage": "download"})
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
    total_frames = max(1, int(meta["duration"] * fps))

    # Fail BEFORE any GPU spend: past the HEVC wall no encoder on this fleet
    # can deliver the frame. The web tier refuses to sell these; reaching this
    # line means a stale client or a hand-crafted job.
    if max(ow, oh) > HEVC_MAX_PX:
        return {"error": "output_dims_unsupported", "out_w": ow, "out_h": oh}

    upsampler = _build_upsampler(model_key)
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
    frames = 0
    gpu_ms = 0.0
    last_pct = -1
    try:
        while True:
            raw = dec.stdout.read(frame_bytes)
            if not raw or len(raw) < frame_bytes:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
            g0 = time.time()
            out_img, _ = upsampler.enhance(frame, outscale=scale)
            gpu_ms += (time.time() - g0) * 1000
            enc.stdin.write(np.ascontiguousarray(out_img).tobytes())
            frames += 1
            pct = 3 + int(93 * frames / max(frames, total_frames))
            if pct != last_pct and (frames % 15 == 0 or pct >= 96):
                last_pct = pct
                runpod.serverless.progress_update(
                    job, {"pct": min(96, pct), "stage": "upscale"})
    except BrokenPipeError:
        # The encoder died under us — its stderr has the real reason (NVENC
        # session-open refusals land here: the write, not the spawn, fails).
        try:
            dec.kill(); enc.kill()
        except Exception:
            pass
        return {"error": f"encode_failed:{encoder}:{_gpu_name()}:{_tail(enc_log, 180)}",
                "frames": frames, "encoder": encoder, "gpu": _gpu_name()}
    except Exception as e:
        try:
            dec.kill(); enc.kill()
        except Exception:
            pass
        return {"error": f"upscale_failed:{type(e).__name__}", "frames": frames}
    finally:
        try:
            dec.stdout.close()
        except Exception:
            pass

    # Said out loud: the bar used to freeze at 97 % for as long as the encoder
    # took to flush and rewrite the file for faststart — minutes on a
    # multi-GB result — with nothing to tell "finishing" from "hung".
    runpod.serverless.progress_update(job, {"pct": 97, "stage": "finalize"})
    dec.wait(timeout=60)
    try:
        enc.stdin.close()
    except Exception:
        pass
    if enc.wait(timeout=1800) != 0:
        return {"error": f"encode_failed:{encoder}:{_gpu_name()}:{_tail(enc_log, 180)}",
                "encoder": encoder, "gpu": _gpu_name()}
    if frames == 0 or not os.path.exists(dst) or os.path.getsize(dst) < 4096:
        return {"error": "empty_output", "frames": frames}

    runpod.serverless.progress_update(job, {"pct": 98, "stage": "upload"})

    def _on_progress(sent, total):
        runpod.serverless.progress_update(
            job, {"pct": 98 + (1 if total and sent * 2 >= total else 0), "stage": "upload"})

    try:
        # Encoder + GPU ride the parts as headers → the server drops them into
        # a meta.json sidecar → srv_done telemetry. RunPod's own job output
        # expires in hours; this is the only durable per-job fleet map.
        size, parts = _upload_parts(out_url, dst, part_bytes, {
            "X-Upsc-Encoder": encoder,
            "X-Upsc-Gpu": _gpu_name(),
        }, _on_progress)
    except UploadRejected as e:
        return {"error": f"upload_rejected:{e}", "bytes": os.path.getsize(dst)}
    except Exception as e:
        return {"error": f"upload_failed:{type(e).__name__}", "bytes": os.path.getsize(dst)}

    return {
        "ok": True, "uploaded": True, "bytes": size, "parts": parts, "frames": frames,
        "gpu_ms": int(gpu_ms), "wall_ms": int((time.time() - t0) * 1000),
        "encoder": encoder, "model": model_key, "fps": round(fps, 3),
        "src_h": h, "gpu": _gpu_name(), "max_bps": max_bps,
        **_mem_facts(),
    }


if __name__ == "__main__":
    # Guarded so the module can be IMPORTED by the local end-to-end harness
    # (real ffmpeg, real NVENC, a stub model, the real Laravel part ingest)
    # without starting the RunPod loop. The image runs `python -u /handler.py`.
    runpod.serverless.start({"handler": handler})

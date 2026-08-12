"""RunPod serverless worker: video upscale via Real-ESRGAN.

Contract (all transfer is out-of-band — the /run JSON stays control-plane
only, RunPod caps request bodies at 10 MiB):

  input = {
    "in_url":  signed GET  → source video bytes,
    "out_url": signed PUT  → where to upload the result mp4,
    "scale": 2 | 4,
    "model": "general-x4v3" | "x4plus",
    "cq": 19,                      # NVENC constant-quality target
    "billed_duration_s": int,      # what the user paid for — we are the verifier
    "billed_h": int,
  }

  return {"ok", "uploaded", "bytes", "frames", "gpu_ms", "wall_ms",
          "encoder", "model", "fps", "src_h"}
  or     {"error": "..."}          # server releases the token hold

Billing-truth verification: the web tier prices the job from the CLIENT's
probe. This worker re-measures the real file and refuses (before any GPU
spend) when the actual duration/height exceeds what was billed — a mismatch
fails the job and the server refunds automatically. Honest users are never
blocked: their browser probe matches ffprobe within a rounding second.
"""

import json
import os
import subprocess
import tempfile
import time
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


def _upload(url: str, src: str, extra_headers: dict | None = None) -> int:
    size = os.path.getsize(src)
    with open(src, "rb") as f:
        # Read fully: urllib streams file objects chunked, but a known length
        # lets the receiving side verify truncation (Content-Length check).
        body = f.read()
    headers = {"Content-Type": "video/mp4", "Content-Length": str(size)}
    if extra_headers:
        headers.update(extra_headers)
    with _http(url, method="PUT", data=body, headers=headers) as resp:
        resp.read()
    return size


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
            pct = 3 + int(94 * frames / max(frames, total_frames))
            if pct != last_pct and (frames % 15 == 0 or pct >= 97):
                last_pct = pct
                runpod.serverless.progress_update(
                    job, {"pct": min(97, pct), "stage": "upscale"})
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
    try:
        # Encoder + GPU ride the PUT as headers → the server drops them into a
        # meta.json sidecar → srv_done telemetry. RunPod's own job output
        # expires in hours; this is the only durable per-job fleet map.
        size = _upload(out_url, dst, {
            "X-Upsc-Encoder": encoder,
            "X-Upsc-Gpu": _gpu_name(),
        })
    except Exception as e:
        return {"error": f"upload_failed:{type(e).__name__}"}

    return {
        "ok": True, "uploaded": True, "bytes": size, "frames": frames,
        "gpu_ms": int(gpu_ms), "wall_ms": int((time.time() - t0) * 1000),
        "encoder": encoder, "model": model_key, "fps": round(fps, 3),
        "src_h": h, "gpu": _gpu_name(),
    }


runpod.serverless.start({"handler": handler})

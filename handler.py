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


def _upload(url: str, src: str) -> int:
    size = os.path.getsize(src)
    with open(src, "rb") as f:
        # Read fully: urllib streams file objects chunked, but a known length
        # lets the receiving side verify truncation (Content-Length check).
        body = f.read()
    with _http(url, method="PUT", data=body,
               headers={"Content-Type": "video/mp4", "Content-Length": str(size)}) as resp:
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
    return {
        "w": int(v["width"]), "h": int(v["height"]),
        "fps": fps, "duration": duration, "has_audio": a is not None,
    }


def _nvenc_available() -> bool:
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=60)
        return "h264_nvenc" in out.stdout
    except Exception:
        return False


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
        return {"error": f"download_failed:{type(e).__name__}"}

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

    upsampler = _build_upsampler(model_key)
    encoder = "h264_nvenc" if _nvenc_available() else "libx264"

    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", src,
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
    else:
        enc_args += ["-preset", "veryfast", "-crf", str(cq)]
    enc_args += ["-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                 "-movflags", "+faststart", dst]
    enc = subprocess.Popen(enc_args, stdin=subprocess.PIPE)

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
        return {"error": "encode_failed"}
    if frames == 0 or not os.path.exists(dst) or os.path.getsize(dst) < 4096:
        return {"error": "empty_output", "frames": frames}

    runpod.serverless.progress_update(job, {"pct": 98, "stage": "upload"})
    try:
        size = _upload(out_url, dst)
    except Exception as e:
        return {"error": f"upload_failed:{type(e).__name__}"}

    return {
        "ok": True, "uploaded": True, "bytes": size, "frames": frames,
        "gpu_ms": int(gpu_ms), "wall_ms": int((time.time() - t0) * 1000),
        "encoder": encoder, "model": model_key, "fps": round(fps, 3),
        "src_h": h,
    }


runpod.serverless.start({"handler": handler})

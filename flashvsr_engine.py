"""FlashVSR+ — the restoration engine of the upscale worker (2026-11-05).

Model: FlashVSR+ Turbo (OpenImagingLab, Apache-2.0), a one-step diffusion video super-resolution
model on Wan2.1-1.3B (Apache-2.0). Chosen by measurement on our bench (scratch corpus: Xiph/ToS/
Commons clips, x2/x4 synthetic + real low-res) against Real-ESRGAN general-x4v3 / wdn / x4plus,
SeedVR2-3B, RealViformer and EfRLFN — see docs/why/video-upscaler-restoration-engine.md in the site
repo for the table and the crops.

Why a separate engine and not "another model in MODELS": the ESRGAN path is per frame at ×4 and
then reduced to the box; this one is a STREAM — it keeps a bounded KV cache across frames (that
is where its temporal stability comes from) and runs directly at the box size, so its cost is
the OUTPUT pixels. The handler's decoder, encoder, progress and delivery stay shared: frames in
(BGR uint8, source size) -> frames out (BGR uint8, exactly ow×oh) through `emit`.

Contract with the model's own preprocessing (inference/preprocessing/input.py, stream mode):
LQ frames are bicubic-resized (PIL, a=-0.5) to the target and scaled to [-1, 1]; sides must be
multiples of 128 — we pad bottom/right by reflection and crop the output back, so no pixel of
the picture is cropped or stretched beyond the box rounding the ESRGAN path already does.
"""
import os
import sys
import time

import numpy as np

FVP_DIR = os.environ.get("FLASHVSR_DIR", "/opt/flashvsr")
# Pure-Python deps the base image lacks (einops, imageio(+ffmpeg), safetensors, tqdm,
# flashvsr_attn) — a layer of their own, APPENDED to sys.path so nothing in the base is shadowed.
PYDEPS = os.environ.get("FLASHVSR_PYDEPS", "/opt/pydeps")
MULT = 128
_CACHE = {}


def model_dims(ow: int, oh: int) -> tuple:
    """The size the model runs at: the box rounded UP to multiples of 128."""
    return -(-ow // MULT) * MULT, -(-oh // MULT) * MULT


def _paths():
    """The model's code and the deps layer onto sys.path — BEFORE any `inference` import.
    In the image nothing is pip-installed (layers only); on a dev box the package may be,
    which is exactly how an import placed before this call hid on the laptop and M5 and
    failed on the first pool card (probe 05.11: «No module named 'inference'»)."""
    if FVP_DIR not in sys.path:
        sys.path.insert(0, FVP_DIR)
    if os.path.isdir(PYDEPS) and PYDEPS not in sys.path:
        sys.path.append(PYDEPS)


def _pipeline(mode: str, decode_every: int):
    key = (mode, decode_every)
    if key in _CACHE:
        return _CACHE[key]
    _paths()
    import torch
    from pathlib import Path
    from inference import build_inference_pipeline
    from inference.presets import model_paths_for_mode, stream_config_for_mode
    config = stream_config_for_mode(
        mode, device="cuda:0", seed=1, decode_every=decode_every,
        temporal_compression=4, color_correction=True,
    )
    # weights live where the model's own loader expects them: <FVP_DIR>/ckpts/{TCDecoder.ckpt,<mode>/…}
    paths = model_paths_for_mode(Path(FVP_DIR), mode)
    _CACHE.clear()  # one engine resident: the card holds one 1.3B model, not two
    torch.cuda.empty_cache()
    pipe = build_inference_pipeline(config, paths)
    _CACHE[key] = pipe
    return pipe


def upscale_stream(frames, w: int, h: int, ow: int, oh: int, emit, *, mode: str = "turbo",
                   decode_every: int = 8, on_progress=None) -> dict:
    """frames: iterator of BGR uint8 (h, w, 3); emit(bgr uint8 (oh, ow, 3)) for every output frame.
    Returns facts for the job record: model frames, peak memory, model size."""
    _paths()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from inference.preprocessing.stream import TemporalVideoStream
    from inference.config import TEMPORAL_DOWNSAMPLE

    pipe = _pipeline(mode, decode_every)
    OW, OH = model_dims(ow, oh)
    tc = pipe.config.temporal_compression
    chunk = decode_every * tc

    def lq_chunks():
        buf = []
        first = True
        for bgr in frames:
            rgb = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
            if (w, h) != (ow, oh):
                rgb = rgb.resize((ow, oh), Image.Resampling.BICUBIC)
            buf.append(np.asarray(rgb))
            need = chunk + 1 if first else chunk
            if len(buf) >= need:
                yield _to_tensor(buf)
                buf, first = [], False
        if buf:
            yield _to_tensor(buf)

    def _to_tensor(buf):
        t = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)  # F,3,oh,ow
        if (OH, OW) != (oh, ow):
            t = F.pad(t, (0, OW - ow, 0, OH - oh), mode="reflect" if (OW - ow < ow and OH - oh < oh) else "replicate")
        return t.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # 1,3,F,OH,OW

    out = {"frames": 0}

    class _Writer:
        def append(self, video, *, value_range="neg_one_to_one"):
            v = video[0, :, :, :oh, :ow]  # 3,F,oh,ow on the card
            if stream.eof:  # the runner already cuts the EOF padding; never emit past the source
                v = v[:, :max(0, stream.source_frames - out["frames"])]
            v = v.float().add_(1.0).mul_(127.5).clamp_(0, 255).round_().to(torch.uint8)
            v = v[[2, 1, 0]].permute(1, 2, 3, 0).contiguous().cpu().numpy()  # F,oh,ow,3 BGR
            for f in v:
                emit(f)
                out["frames"] += 1
            if on_progress:
                on_progress(out["frames"])

        def close(self):
            pass

    stream = TemporalVideoStream(lq_chunks(), tc, TEMPORAL_DOWNSAMPLE)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        pipe.runner.run(stream, None, progress=None, segment_writer=_Writer(), output_frame_limit=None)
    finally:
        stream.close()
    torch.cuda.synchronize()
    return {"engine": f"flashvsr-{mode}", "model_w": OW, "model_h": OH, "frames_out": out["frames"],
            "model_s": round(time.time() - t0, 1),
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}

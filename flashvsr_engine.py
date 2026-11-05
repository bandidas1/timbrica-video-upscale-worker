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
import contextlib
import functools
import os
import queue
import sys
import threading
import time

import numpy as np

FVP_DIR = os.environ.get("FLASHVSR_DIR", "/opt/flashvsr")
# Pure-Python deps the base image lacks (einops, imageio(+ffmpeg), safetensors, tqdm,
# flashvsr_attn) — a layer of their own, APPENDED to sys.path so nothing in the base is shadowed.
PYDEPS = os.environ.get("FLASHVSR_PYDEPS", "/opt/pydeps")
MULT = 128
_CACHE = {}

# Knobs a probe A/B's on ONE pool host (bench_engine "variants" — hosts differ, so only a pair on
# the same host is a measurement); a job runs the defaults. A knob earns its default by such a
# probe, never by reasoning — docs/why/video-upscaler-restoration-engine.md keeps the numbers.
# Pool RTX 4090, 33 frames, every variant's frames identical to the baseline (max |diff| 0):
#   1080p  baseline 32.4 s · color_on_card 21.9 · keep_dit 26.3 · both 14.6 (0.47 -> 0.21 s/MPx)
#   1440p  baseline 55.3 s · color_on_card 36.3 · both 28.7 (0.45 -> 0.24 s/MPx), peak 13.9 -> 18.6 GiB
#   color_on_card — the wavelet color correction on the card: the model's decoder hands its frames
#                   back on the host, and the correction there took 10 s of 32 at 1080p.
#   keep_dit      — leave the 1.3B DiT (and its attention cache) on the card while the decoder runs;
#                   the model moves them to the host and back every segment so that 4K fits 24 GB
#                   (6-8.5 s of a run). "auto" = keep up to the measured 1440p model area.
#   prefetch      — LQ frames resized and packed in a background thread, one chunk ahead, so the
#                   card does not wait for PIL: 1440p, 97 frames, pool PRO 6000 73.4 -> 59.5 s.
#   decode_on_card — the decoder's memory and output stay on the card (_segment_decoder); under test.
# (A thread-count knob was measured and dropped: the pool pod has no cgroup CPU quota, 64 threads.)
KNOBS = {"color_on_card": True, "keep_dit": "auto", "prefetch": True, "decode_on_card": False}
KEEP_DIT_MAX_AREA = 2560 * 1536   # model area of a 1440p box: peak 18.6 GiB on a 24 GB card
_PH = {"cur": None, "stack": []}


def cpu_quota():
    """vCPUs the container may actually use (cgroup v2 cpu.max, else v1 CFS); None = no quota."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            q, p = fh.read().split()[:2]
        if q != "max":
            return max(1, round(int(q) / int(p)))
        return None
    except Exception:
        pass
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fh:
            q = int(fh.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fh:
            p = int(fh.read())
        return max(1, round(q / p)) if q > 0 else None
    except Exception:
        return None


_RUN = {"keep_dit": False}   # the knobs resolved for the run in progress (the wrappers read it)


@contextlib.contextmanager
def _phase(name: str):
    """Exclusive wall time of a phase (a nested phase's time is not counted twice), the card
    synchronized at both edges so GPU work lands in the phase that queued it. Off unless a
    run collects (upscale_stream sets _PH["cur"]); a few syncs per 32-frame segment."""
    cur = _PH["cur"]
    if cur is None:
        yield
        return
    import torch
    torch.cuda.synchronize()
    st = _PH["stack"]
    st.append(0.0)
    t = time.perf_counter()
    try:
        yield
    finally:
        torch.cuda.synchronize()
        el = time.perf_counter() - t
        child = st.pop()
        cur[name] = cur.get(name, 0.0) + el - child
        if st:
            st[-1] += el


def _timed(name: str, fn):
    if getattr(fn, "_tmb_phase", None):
        return fn

    @functools.wraps(fn)
    def run(*a, **k):
        with _phase(name):
            return fn(*a, **k)
    run._tmb_phase = name
    return run


def _segment_decoder(dec):
    """TinyDecoder.decode_stream_segment, or — knob decode_on_card — the same steps with the
    decoder's temporal memory and its output kept on the card. The model's adapter (pinned
    b548a2cc, inference/decoders/tiny.py) moves the memory to the host and back around every
    segment and every 4-latent output chunk to the host, then concatenates and rescales the
    segment there in bf16: room for DiT on a 24 GB card at 4K. With the DiT kept (keep_dit) the
    host trips are pure cost. Same candidates, same OOM fallback order, same arithmetic (×2 −1
    in bf16 rounds the same on either device); the snapshot for an OOM retry is a card copy."""
    import torch
    orig = dec.decode_stream_segment

    def chunks_on_card(latents, lq_video, *, device, dtype, use_parallel, first):
        # TinyDecoder._decode_chunks minus the per-chunk copy to the host
        outputs = []
        for start in range(0, latents.shape[2], 4):
            stop = min(start + 4, latents.shape[2])
            cur = latents[:, :, start:stop].to(device=device, dtype=dtype)
            if first and start == 0:
                f0, n = 0, 4 * (stop - start) - 3
            else:
                f0, n = 4 * start - (3 if first else 0), 4 * (stop - start)
            lq = lq_video[:, :, f0:f0 + n].to(device=device, dtype=dtype)
            out = dec.module.decode_video(cur.transpose(1, 2), parallel=use_parallel,
                                          show_progress_bar=False, cond=lq).transpose(1, 2)
            outputs.append(out.detach())
            del cur, lq, out
        return torch.cat(outputs, dim=2)

    def segment(latents, lq_video, *, first=False):
        if not KNOBS["decode_on_card"]:
            return orig(latents, lq_video, first=first)
        from inference.decoders.base import DecodeResult
        if first:
            dec.reset()
            dec.last_auto_fallbacks = 0
        p = next(dec.module.parameters())
        device, dtype = p.device, p.dtype
        mode = dec.parallel_mode
        cands = (dec._auto_candidates(lq_video, device) if mode == "auto"
                 else [("parallel", True)] if mode == "parallel" else [("serial", False)])
        saved = [None if x is None else x.detach().to(device).clone() for x in dec.module.mem]
        for name, par in cands:
            dec.module.mem = [None if x is None else x.clone() for x in saved]
            try:
                value = chunks_on_card(latents, lq_video, device=device, dtype=dtype,
                                       use_parallel=par, first=first)
            except torch.cuda.OutOfMemoryError:
                dec.module.mem = saved
                if mode != "auto" or name == "serial":
                    raise
                dec.last_auto_fallbacks += 1
                torch.cuda.empty_cache()
                continue
            dec.last_execution_mode = name
            return DecodeResult(value.mul_(2).sub_(1), None)
        raise RuntimeError("Tiny Decoder produced no output")

    return segment


def _instrument(pipe):
    """Phase timers on the model's own seams + the knobs' switches. The model's code is not
    edited (it is a pinned upstream commit); its module attributes are wrapped once."""
    import torch
    import inference.runners.stream as RS
    from inference.postprocessing.color_correction import WaveletColorCorrector

    if not getattr(RS, "_tmb_instrumented", False):
        off, back = RS._offload_dit, RS._restore_dit

        def offload(bundle, state, device):
            if not _RUN["keep_dit"]:
                off(bundle, state, device)

        def restore(bundle, state, device):
            if not _RUN["keep_dit"]:
                back(bundle, state, device)

        class CardColor(WaveletColorCorrector):
            def __call__(self, generated, reference):
                with _phase("color"):
                    if KNOBS["color_on_card"] and generated.device.type != "cuda":
                        try:
                            return super().__call__(generated.to("cuda"), reference)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                    return super().__call__(generated, reference)

        RS._offload_dit = _timed("offload", offload)
        RS._restore_dit = _timed("restore", restore)
        RS._merge_segment_groups = _timed("merge", RS._merge_segment_groups)
        RS.WaveletColorCorrector = CardColor
        RS._tmb_instrumented = True
    b = pipe.bundle
    b.decoder.decode_stream_segment = _timed("decode", _segment_decoder(b.decoder))
    # the decoder network itself, nested in "decode": what stays in "decode" is the model's host
    # round trips around it (memory snapshot, outputs to the host, the [-1, 1] rescale there)
    b.decoder.module.decode_video = _timed("decode_net", b.decoder.module.decode_video)
    b.denoiser.forward_stream = _timed("dit", b.denoiser.forward_stream)
    pipe.runner._project_lq_chunk = _timed("proj", pipe.runner._project_lq_chunk)


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


# A self-contained C toolchain (conda-forge gcc with its own sysroot) shipped as a layer: Triton
# builds its CUDA driver module and a launcher per kernel with a C compiler at first use, and
# the worker's runtime base has none (05.11 «Failed to find C compiler»). Its own sysroot means
# the image's system libraries are not touched — the ESRGAN runtime stays what it was.
CC_PATH = os.environ.get("FLASHVSR_CC", "/opt/cc/bin/x86_64-conda-linux-gnu-gcc")
ATTN = {"kind": None, "why": ""}


def _toolchain():
    """CC for Triton, and a directory where `-lcuda` links: Triton passes -L<dir of libcuda.so.1>
    and -lcuda, and the container carries only the versioned libcuda.so.1."""
    if not os.environ.get("CC") and os.path.exists(CC_PATH):
        os.environ["CC"] = CC_PATH
    if os.environ.get("TRITON_LIBCUDA_PATH"):
        return
    try:
        import subprocess
        out = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True)
        locs = [ln.split()[-1] for ln in out.splitlines() if "libcuda.so.1" in ln]
        if not locs:
            return
        d = "/tmp/tmb-libcuda"
        os.makedirs(d, exist_ok=True)
        for name in ("libcuda.so.1", "libcuda.so"):
            p = os.path.join(d, name)
            if not os.path.lexists(p):
                os.symlink(locs[0], p)
        os.environ["TRITON_LIBCUDA_PATH"] = d
    except Exception:
        pass


def _triton_can_build() -> bool:
    import shutil
    return bool(os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang"))


def _choose_attention(MA):
    """The model's Triton stream kernel if it BUILDS here and answers like the exact path on a
    known input (the OrtEpVerify rule: a kernel is trusted by its answer, not by its creation);
    otherwise exact_block_sparse. Decided once per process; the job record says which and why."""
    if ATTN["kind"]:
        return
    import torch
    triton_fn = MA.stream_block_sparse_attention
    _toolchain()
    if triton_fn is None or not _triton_can_build():
        MA.stream_block_sparse_attention = exact_block_sparse
        ATTN.update(kind="torch", why="no kernel" if triton_fn is None else "no C compiler")
        return
    try:
        g = torch.Generator(device="cuda").manual_seed(0)
        bf = torch.bfloat16
        q = torch.randn(1, 2, 4, 128, 128, device="cuda", dtype=bf, generator=g)
        k = torch.randn(1, 2, 7, 128, 128, device="cuda", dtype=bf, generator=g)
        v = torch.randn(1, 2, 7, 128, 128, device="cuda", dtype=bf, generator=g)
        m = torch.rand(1, 2, 4, 7, device="cuda", generator=g) < 0.4
        m[..., 0] = True
        with torch.inference_mode():
            d = (triton_fn(q, k, v, m).float() - exact_block_sparse(q, k, v, m).float()).abs().max().item()
        if d < 0.05:
            ATTN.update(kind="triton", why=f"verified max|d| {d:.4f}")
        else:
            MA.stream_block_sparse_attention = exact_block_sparse
            ATTN.update(kind="torch", why=f"triton answered wrong: max|d| {d:.4f}")
    except Exception as e:  # noqa: BLE001 — a kernel that cannot build or launch must not fail a paid job
        MA.stream_block_sparse_attention = exact_block_sparse
        ATTN.update(kind="torch", why=f"triton failed: {type(e).__name__}: {str(e)[:160]}")


def exact_block_sparse(q_windows, k_windows, v_windows, block_mask, *, ensure_nonempty=False, block_m=None):
    """The stream's block-sparse attention in plain torch — the same softmax over the same keys as
    FlashVSR+'s Triton kernel and its eager reference (checked: max |Δ| 0.002 in bf16, M5 05.11).
    Layout of the model's stream: q[B,H,Qb,T,D], k/v[B,H,Kb,T,D], block_mask[B,H,Qb,Kb]. Per query
    block: gather the union of allowed key blocks, SDPA with the per-head token mask (None when
    every head allows every gathered block, so the fused kernels apply)."""
    import torch
    import torch.nn.functional as F
    B, H, Qb, T, D = q_windows.shape
    flat = block_mask.bool().reshape(B * H, Qb, -1).clone()
    empty = ~flat.any(dim=-1)
    rows = torch.arange(Qb, device=flat.device)
    cols = rows.clamp_max(max(flat.shape[-1] - 1, 0))
    eb, er = torch.where(empty)
    flat[eb, er, cols[er]] = True   # an empty row attends to its own block (as the reference)
    bm = flat.reshape(B, H, Qb, -1)
    any_bh = bm.any(dim=0).any(dim=0)
    out = torch.empty_like(q_windows)
    for qb in range(Qb):
        sel = torch.nonzero(any_bh[qb], as_tuple=False).flatten()
        n = sel.numel()
        k = k_windows.index_select(2, sel).reshape(B, H, n * T, D)
        v = v_windows.index_select(2, sel).reshape(B, H, n * T, D)
        m = bm[:, :, qb].index_select(2, sel)
        mask = None if bool(m.all()) else m.repeat_interleave(T, dim=-1)[:, :, None, :]
        out[:, :, qb] = F.scaled_dot_product_attention(q_windows[:, :, qb], k, v, attn_mask=mask)
    return out


def _pipeline(mode: str, decode_every: int):
    key = (mode, decode_every)
    if key in _CACHE:
        return _CACHE[key]
    _paths()
    import models.attention as MA
    _choose_attention(MA)
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
    _instrument(pipe)
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
    kd = KNOBS["keep_dit"]
    _RUN["keep_dit"] = (OW * OH <= KEEP_DIT_MAX_AREA) if kd == "auto" else bool(kd)
    lq_busy = {"src": 0.0, "prep": 0.0}   # producer time: on the critical path only without prefetch

    def lq_chunks():
        """The model's LQ input: frames bicubic-resized (PIL, the model's own preprocessing) and
        packed into chunks. Times itself without _phase — with prefetch it runs in another thread."""
        buf = []
        first = True
        src = iter(frames)
        while True:
            t = time.perf_counter()
            bgr = next(src, None)   # the caller's decoder (ffmpeg pipe in a job)
            lq_busy["src"] += time.perf_counter() - t
            if bgr is None:
                break
            t = time.perf_counter()
            rgb = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
            if (w, h) != (ow, oh):
                rgb = rgb.resize((ow, oh), Image.Resampling.BICUBIC)
            buf.append(np.asarray(rgb))
            need = chunk + 1 if first else chunk
            ready = len(buf) >= need
            if ready:
                packed = _to_tensor(buf)
                buf, first = [], False
            lq_busy["prep"] += time.perf_counter() - t
            if ready:
                yield packed
        if buf:
            t = time.perf_counter()
            packed = _to_tensor(buf)
            lq_busy["prep"] += time.perf_counter() - t
            yield packed

    def prefetched(gen):
        """`gen` one chunk ahead in a thread (PIL and the packing release the GIL), so the card
        runs the previous chunk meanwhile. The consumer's wait is what still costs wall time
        (wait_lq); an exception in the producer is raised here; leaving early stops it."""
        q = queue.Queue(maxsize=1)
        stop = threading.Event()
        end = object()

        def put(item):
            while not stop.is_set():
                try:
                    q.put(item, timeout=0.5)
                    return True
                except queue.Full:
                    pass
            return False

        def run():
            try:
                for item in gen:
                    if not put(item):
                        return
                put(end)
            except BaseException as e:  # noqa: BLE001 — handed to the consumer, raised there
                put(e)
            finally:
                gen.close()

        th = threading.Thread(target=run, name="flashvsr-lq-prefetch", daemon=True)
        th.start()
        try:
            while True:
                with _phase("wait_lq"):
                    item = q.get()
                if item is end:
                    return
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            th.join(timeout=30)

    def _to_tensor(buf):
        t = torch.from_numpy(np.stack(buf)).permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)  # F,3,oh,ow
        if (OH, OW) != (oh, ow):
            t = F.pad(t, (0, OW - ow, 0, OH - oh), mode="reflect" if (OW - ow < ow and OH - oh < oh) else "replicate")
        return t.permute(1, 0, 2, 3).unsqueeze(0).contiguous()  # 1,3,F,OH,OW

    out = {"frames": 0}

    class _Writer:
        def append(self, video, *, value_range="neg_one_to_one"):
            with _phase("write"):
                v = video[0, :, :, :oh, :ow]  # 3,F,oh,ow — on the card or the host, per the color knob
                if stream.eof:  # the runner already cuts the EOF padding; never emit past the source
                    v = v[:, :max(0, stream.source_frames - out["frames"])]
                v = v.float().add_(1.0).mul_(127.5).clamp_(0, 255).round_().to(torch.uint8)
                v = v[[2, 1, 0]].permute(1, 2, 3, 0).contiguous().cpu().numpy()  # F,oh,ow,3 BGR
            with _phase("emit"):  # the caller's encoder pipe (blocks when the encoder lags)
                for f in v:
                    emit(f)
                    out["frames"] += 1
                if on_progress:
                    on_progress(out["frames"])

        def close(self):
            pass

    prefetch = bool(KNOBS["prefetch"])
    stream = TemporalVideoStream(prefetched(lq_chunks()) if prefetch else lq_chunks(), tc, TEMPORAL_DOWNSAMPLE)
    torch.cuda.reset_peak_memory_stats()
    phases = {}
    _PH.update(cur=phases, stack=[])
    t0 = time.perf_counter()   # the clock the phases use: wall time can be adjusted mid-run
    try:
        pipe.runner.run(stream, None, progress=None, segment_writer=_Writer(), output_frame_limit=None)
    finally:
        _PH.update(cur=None, stack=[])
        stream.close()
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    if not prefetch:   # on the critical path: part of the wall time (never nested in another phase)
        phases.update(lq_busy)
    phases["other"] = total - sum(phases.values())
    if prefetch:       # overlapped with the card: reported, not summed
        phases.update({k + "_bg": v for k, v in lq_busy.items()})
    return {"engine": f"flashvsr-{mode}", "attn": ATTN["kind"], "attn_why": ATTN["why"],
            "model_w": OW, "model_h": OH, "frames_out": out["frames"],
            "model_s": round(total, 1),
            "phases_s": {k: round(v, 2) for k, v in sorted(phases.items(), key=lambda kv: -kv[1])},
            "threads": torch.get_num_threads(), "cpu_quota": cpu_quota(),
            "knobs": dict(KNOBS), "keep_dit": _RUN["keep_dit"], "decode_every": decode_every,
            "decoder_mode": pipe.bundle.decoder.last_execution_mode,
            "peak_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)}

# timbrica-video-upscale-worker

RunPod serverless worker for the paid server-side lane of timbrica.com//video-upscaler.
Real-ESRGAN (realesr-general-x4v3 default, x4plus switchable) fp16 tile-512,
stream decode/encode via ffmpeg (h264_nvenc, libx264 fallback), audio AAC 192k.

Transport: control-plane JSON only; the video moves over signed GET/PUT URLs
minted by the Laravel side (RunPod caps /run bodies at 10 MiB).
Billing truth: the worker re-probes the file and refuses (duration_mismatch /
resolution_mismatch) before GPU spend if the client understated what it paid for.

Deploy: RunPod console -> Serverless -> New Endpoint -> GitHub repo (this one),
GPU 4090 24GB, container disk 25 GB, executionTimeoutMs 6600000, workersMax >= 2.

## ⚠️ The live endpoint is NOT built from this image (as of 2026-08-27)

Endpoint `y796txt9qe1rcx` runs template `6zioxjrgg0`, which starts the generic
`runpod/pytorch` base and rebuilds the worker from scratch in `dockerStartCmd`
on EVERY cold boot: `apt-get install ffmpeg`, `pip install basicsr realesrgan
opencv-python-headless`, `wget` of both weight files (~70 MB) from GitHub
releases and of `handler.py` from raw.githubusercontent — and only then starts
the handler. That path was taken at launch because the REST API cannot create a
GitHub-built endpoint; nobody ever switched it over.

What it costs, measured 2026-08-27 on the live endpoint with an EMPTY queue
(RunPod `delayTime` — time before the handler sees the job):

| worker state at submit | delayTime |
|---|---|
| hot (a job right after a job) | **1.5 s** |
| cold | **73 s** |
| cold | **101 s** |
| cold | **591 s** |

Plus 2153 s on the production job behind the 2026-08-27 support ticket
(«all stops at "waiting for gpu"»). Throttling was 0 throughout, and
`workers.running` went to 1 while `jobs.inProgress` stayed 0 — the wait is the
worker building itself, not a shortage of GPUs. With `idleTimeout: 5 s` on a
lane that sees roughly one job a day, essentially every real job pays it.

Building from this Dockerfile replaces all of that with an image pull.

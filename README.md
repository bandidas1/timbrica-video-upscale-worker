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

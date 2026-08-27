# RunPod serverless worker: Real-ESRGAN video upscale (timbrica /video-upscaler paid lane)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod==1.7.* numpy opencv-python-headless \
    basicsr==1.4.2 realesrgan==0.3.0

# ⚠️ NO sed patch here on purpose. basicsr 1.4.2 imports
# torchvision.transforms.functional_tensor, removed in torchvision >= 0.17 —
# and handler.py already registers a runtime shim for it BEFORE basicsr loads
# ("this keeps the worker independent of any image-build sed patching").
#
# The line that used to sit here hardcoded
# /usr/local/lib/python3.11/dist-packages/... and ended in `|| true`, so if pip
# had put basicsr anywhere else the patch would have silently done nothing and
# the image would still have built clean. A build step that cannot fail is not
# a safeguard, it is a comment — and this one was not even needed.

# Bake the weights into the image — deterministic cold start, no runtime fetch.
RUN mkdir -p /weights \
    && wget -q -O /weights/realesr-general-x4v3.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth \
    && wget -q -O /weights/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

ENV WEIGHTS_DIR=/weights
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

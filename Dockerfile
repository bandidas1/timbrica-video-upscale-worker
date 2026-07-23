# RunPod serverless worker: Real-ESRGAN video upscale (timbrica /video-upscaler paid lane)
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir runpod==1.7.* numpy opencv-python-headless \
    basicsr==1.4.2 realesrgan==0.3.0

# basicsr 1.4.2 imports torchvision.transforms.functional_tensor, removed in
# torchvision >= 0.17 — the canonical one-line patch.
RUN sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    /usr/local/lib/python3.11/dist-packages/basicsr/data/degradations.py || true

# Bake the weights into the image — deterministic cold start, no runtime fetch.
RUN mkdir -p /weights \
    && wget -q -O /weights/realesr-general-x4v3.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth \
    && wget -q -O /weights/RealESRGAN_x4plus.pth \
       https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth

ENV WEIGHTS_DIR=/weights
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]

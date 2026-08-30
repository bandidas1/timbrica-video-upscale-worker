# RunPod serverless worker: Real-ESRGAN video upscale (timbrica /video-upscaler paid lane)
# ⚠️ CUDA 12.8, и версия здесь — весь смысл правки.
#
# RunPod наполняет свои РАЗМЕРНЫЕ пулы новым железом: в ADA_24 рядом с 4090
# приезжает «RTX PRO 6000 Blackwell MIG 1g.24gb» (sm_120). Прежняя база
# (torch 2.4 / CUDA 12.4.1) ядер под sm_120 не содержит, и воркер падает с
# «CUDA error: no kernel image is available for execution on the device».
# Замер 28.08: пять прогонов одного человека за семь минут, все с этой ошибкой.
# Список исключённых карт это лечит, но протухает с каждой новой архитектурой.
#
# База взята не из документации, а с нашего же стека: ровно на ней работает
# demucs-воркер (torch 2.7.1 / cu12.8), и seed-vc/SoulX стоят на тех же колёсах.
# ⚠️ Python в ней 3.11 — это НЕ косметика: basicsr 1.4.2 не собирается на 3.12+
# (замерено 30.08: KeyError '__version__' ещё на этапе сборки колеса), поэтому
# свежие runpod-базы на Ubuntu 24.04 сюда не годятся.
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# ⚠️ torch 2.6 перевернул torch.load на weights_only=True. Real-ESRGAN грузит
# пиклённые .pth чекпоинты через torch.load внутри RealESRGANer, поэтому без
# этого апгрейд поменял бы отказ GPU на отказ загрузки модели. Та же строка и по
# той же причине стоит у demucs-воркера.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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

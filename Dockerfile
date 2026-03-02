# LTX-2 Video Generation Worker for RunPod Serverless
# Optimized for H100 (80GB) - fastest generation
# PyTorch 2.6+ required: diffusers main uses custom_op that needs torch 2.5+ (infer_schema)

FROM runpod/pytorch:1.0.3-cu1290-torch260-ubuntu2204

# Set environment variables - all caches and temp on volume to avoid "No space left on device"
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume/huggingface
ENV HF_HUB_CACHE=/runpod-volume/huggingface/hub
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface
ENV TORCH_HOME=/runpod-volume/torch
ENV TMPDIR=/runpod-volume/tmp
ENV TEMP=/runpod-volume/tmp
ENV TMP=/runpod-volume/tmp

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git-lfs \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Create app directory
WORKDIR /app

# Install RunPod SDK
RUN pip install runpod

# LTX2Pipeline is in diffusers main; PyPI may not have it yet - install from git
# av (PyAV) required by diffusers.pipelines.ltx2.export_utils for video encoding
RUN pip install "git+https://github.com/huggingface/diffusers.git" \
    transformers accelerate safetensors huggingface_hub av

# Copy handler and entrypoint
COPY handler.py /app/handler.py
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Ensure cache/tmp dirs on volume exist at runtime (volume is mounted by RunPod)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "-u", "/app/handler.py"]

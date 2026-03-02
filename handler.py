"""
RunPod Serverless Handler for LTX-2 Video Generation

Uses Hugging Face diffusers LTX2Pipeline (from_pretrained).

Input format:
{
  "input": {
    "prompt": "A cat playing piano",
    "negative_prompt": "blurry, low quality",  # optional
    "width": 768,       # optional, default 768
    "height": 512,      # optional, default 512
    "num_frames": 97,   # optional, default 97 (~4 seconds at 24fps)
    "fps": 24,          # optional, default 24
    "guidance_scale": 4.0,      # optional, default 4.0
    "num_inference_steps": 30,   # optional, default 30
    "seed": null        # optional, random if not set
  }
}

Output format:
{
  "video": "data:video/mp4;base64,...",
  "duration_seconds": 4.04,
  "resolution": "768x512",
  "fps": 24,
  "seed": 12345,
  "generation_time_seconds": 72.5
}
"""

import os

# Force all HuggingFace cache and temp I/O onto the RunPod volume (before any hf imports)
RUNPOD_VOLUME = "/runpod-volume"
# Disable XET to use normal HTTP download (XET reconstruction can use temp space elsewhere)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
for _key, _val in (
    ("HF_HOME", f"{RUNPOD_VOLUME}/huggingface"),
    ("HF_HUB_CACHE", f"{RUNPOD_VOLUME}/huggingface/hub"),
    ("HF_XET_CACHE", f"{RUNPOD_VOLUME}/huggingface/xet"),
    ("TRANSFORMERS_CACHE", f"{RUNPOD_VOLUME}/huggingface"),
    ("TORCH_HOME", f"{RUNPOD_VOLUME}/torch"),
    ("TMPDIR", f"{RUNPOD_VOLUME}/tmp"),
    ("TEMP", f"{RUNPOD_VOLUME}/tmp"),
    ("TMP", f"{RUNPOD_VOLUME}/tmp"),
):
    os.environ.setdefault(_key, _val)
for _d in (
    f"{RUNPOD_VOLUME}/huggingface/hub",
    f"{RUNPOD_VOLUME}/huggingface/xet",
    f"{RUNPOD_VOLUME}/torch",
    f"{RUNPOD_VOLUME}/tmp",
):
    os.makedirs(_d, exist_ok=True)

import base64
import tempfile
import time
from typing import Optional

# Use volume for Python temp files too
tempfile.tempdir = os.environ.get("TMPDIR", tempfile.gettempdir())

import runpod
import torch


PIPELINE = None
HF_CACHE_DIR = f"{RUNPOD_VOLUME}/huggingface"


def load_pipeline():
    """Load LTX-2 pipeline from diffusers (cached globally)."""
    global PIPELINE

    if PIPELINE is not None:
        return PIPELINE

    print("Loading LTX-2 pipeline (diffusers)...")
    start = time.time()

    from diffusers.pipelines.ltx2 import LTX2Pipeline

    PIPELINE = LTX2Pipeline.from_pretrained(
        "Lightricks/LTX-2",
        torch_dtype=torch.bfloat16,
        cache_dir=HF_CACHE_DIR,
    )
    PIPELINE.to("cuda")

    print(f"Pipeline loaded in {time.time() - start:.1f}s")
    return PIPELINE


def generate_video(
    prompt: str,
    negative_prompt: str = "blurry, low quality, distorted, glitchy, watermark",
    width: int = 768,
    height: int = 512,
    num_frames: int = 97,
    fps: float = 24.0,
    guidance_scale: float = 4.0,
    num_inference_steps: int = 30,
    seed: Optional[int] = None,
) -> dict:
    """Generate video (and audio) from text prompt using diffusers LTX2Pipeline."""

    pipe = load_pipeline()

    if seed is None:
        seed = torch.randint(0, 2**32, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    print(f"Generating video: {prompt[:50]}...")
    print(f"  Resolution: {width}x{height}, Frames: {num_frames}")

    start = time.time()

    video_np, audio_tensor = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        frame_rate=fps,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        output_type="np",
        return_dict=False,
    )

    generation_time = time.time() - start
    print(f"Generation completed in {generation_time:.1f}s")

    from diffusers.pipelines.ltx2.export_utils import encode_video

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir=os.environ.get("TMPDIR")) as f:
        output_path = f.name

    try:
        encode_video(
            video_np[0],
            fps=fps,
            audio=audio_tensor[0].float().cpu(),
            audio_sample_rate=pipe.vocoder.config.output_sampling_rate,
            output_path=output_path,
        )
        with open(output_path, "rb") as f:
            video_bytes = f.read()
    finally:
        try:
            os.unlink(output_path)
        except Exception:
            pass

    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    duration = num_frames / fps

    return {
        "video": f"data:video/mp4;base64,{video_base64}",
        "duration_seconds": round(duration, 2),
        "resolution": f"{width}x{height}",
        "fps": fps,
        "seed": seed,
        "generation_time_seconds": round(generation_time, 2),
    }


def handler(event: dict) -> dict:
    try:
        input_data = event.get("input", {})
        prompt = input_data.get("prompt")
        if not prompt:
            return {"error": "Missing required field: prompt"}

        result = generate_video(
            prompt=prompt,
            negative_prompt=input_data.get(
                "negative_prompt",
                "blurry, low quality, distorted, glitchy, watermark",
            ),
            width=input_data.get("width", 768),
            height=input_data.get("height", 512),
            num_frames=input_data.get("num_frames", 97),
            fps=float(input_data.get("fps", 24)),
            guidance_scale=float(input_data.get("guidance_scale", 4.0)),
            num_inference_steps=input_data.get("num_inference_steps", 30),
            seed=input_data.get("seed"),
        )
        return result
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


runpod.serverless.start({"handler": handler})

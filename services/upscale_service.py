"""
Upscale Service for Shallot-CUI Bot.
Provides multi-tier AI super-resolution:
1. Fast Clean (Model Super-Resolution via Remacri/AnimeSharp with smart aspect downscaling).
2. Generative Clarity (2-stage Latent Refiner: Model Pre-Scale -> VAE Encode -> Low-Denoise KSampler -> VAE Decode).
"""

import os
import io
import json
import random
import logging
import asyncio
from typing import Optional, Tuple, Dict, Any
from PIL import Image

from config import COMFYUI_ADDRESS, PipelineDefaults
from comfy_client import ComfyClient

logger = logging.getLogger("DiscordBot.UpscaleService")

_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

# Upscale model definitions
DEFAULT_UPSCALE_MODEL = "4x_foolhardy_Remacri.pth"
ANIME_UPSCALE_MODEL = "4x-UltraSharp.pth"
PHOTO_UPSCALE_MODEL = "4x_foolhardy_Remacri.pth"

MODEL_PRESETS: Dict[str, str] = {
    "general": DEFAULT_UPSCALE_MODEL,
    "photo": PHOTO_UPSCALE_MODEL,
    "anime": ANIME_UPSCALE_MODEL,
}

DEFAULT_SCALE_FACTOR = 2.0
VALID_SCALE_FACTORS = [1.5, 2.0, 4.0]
DEFAULT_GENERATIVE_DENOISE = 0.30


def calculate_target_dimensions(width: int, height: int, scale_factor: float = 2.0, divisible_by: int = 8) -> Tuple[int, int]:
    """
    Calculates target dimensions based on scale factor, ensuring divisibility by divisible_by.
    """
    if scale_factor <= 0:
        scale_factor = DEFAULT_SCALE_FACTOR
        
    target_w = int(round((width * scale_factor) / divisible_by) * divisible_by)
    target_h = int(round((height * scale_factor) / divisible_by) * divisible_by)
    return max(divisible_by, target_w), max(divisible_by, target_h)


def calculate_latent_refiner_dimensions(
    width: int,
    height: int,
    max_side: int = 1280,
    divisible_by: int = 8
) -> Tuple[int, int]:
    """
    Calculates intermediate dimensions for the SDXL latent refiner stage.
    Maintains aspect ratio while capping the maximum dimension to `max_side`
    to prevent VRAM exhaustion and PCIe paging on consumer GPUs.
    Ensures both dimensions are divisible by `divisible_by` (typically 8 for VAE).
    """
    if width <= 0 or height <= 0:
        return 1024, 1024

    aspect = width / height
    if width >= height:
        new_w = min(width, max_side)
        new_h = int(round(new_w / aspect))
    else:
        new_h = min(height, max_side)
        new_w = int(round(new_h * aspect))

    # Snap to divisible_by
    new_w = max(divisible_by, int(round(new_w / divisible_by) * divisible_by))
    new_h = max(divisible_by, int(round(new_h / divisible_by) * divisible_by))
    return new_w, new_h


async def get_image_dimensions_async(image_bytes: bytes) -> Tuple[int, int]:
    """Extracts width and height of an image asynchronously without blocking the event loop."""
    def _read_size():
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.size  # (width, height)
    return await asyncio.to_thread(_read_size)


def build_fast_upscale_workflow(
    image_filename: str,
    target_width: int,
    target_height: int,
    model_name: str = DEFAULT_UPSCALE_MODEL,
    filename_prefix: str = "Upscale_FastClean"
) -> Dict[str, Any]:
    """
    Constructs a ComfyUI workflow for single-pass AI model super-resolution with bicubic/lanczos fit.
    """
    return {
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        },
        "2": {
            "inputs": {
                "model_name": model_name
            },
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"}
        },
        "3": {
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["1", 0]
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "Upscale With Model"}
        },
        "4": {
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": target_width,
                "height": target_height,
                "crop": "disabled"
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Scale to Target Dimensions"}
        },
        "5": {
            "inputs": {
                "images": ["4", 0],
                "filename_prefix": filename_prefix
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Upscaled Image"}
        }
    }


def build_generative_upscale_workflow(
    image_filename: str,
    target_width: int,
    target_height: int,
    prompt: str,
    negative_prompt: str = "low quality, blurry, distorted, deformed anatomy, bad face, artifacts",
    checkpoint: str = PipelineDefaults.DEFAULT_SDXL_CHECKPOINT,
    model_name: str = DEFAULT_UPSCALE_MODEL,
    seed: Optional[int] = None,
    denoise: float = DEFAULT_GENERATIVE_DENOISE,
    filename_prefix: str = "Upscale_GenerativeClarity",
    orig_width: Optional[int] = None,
    orig_height: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Constructs a 2-stage generative refiner workflow:
    1. Fits input image to intermediate latent dimensions (max side 1280px) to prevent VRAM choking.
    2. VAE Encode to latents.
    3. KSampler low-denoise refiner (0.25 - 0.35) guided by text conditioning.
    4. VAE Decode.
    5. High-fidelity AI model upscale (Remacri / UltraSharp) on the refined pixels.
    6. Lanczos scale to final target dimensions.
    """
    if seed is None:
        seed = random.randint(1, 1125899906842624)

    ref_w = orig_width if orig_width and orig_width > 0 else target_width
    ref_h = orig_height if orig_height and orig_height > 0 else target_height
    inter_w, inter_h = calculate_latent_refiner_dimensions(ref_w, ref_h, max_side=1280, divisible_by=8)

    return {
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Input Image"}
        },
        "2": {
            "inputs": {
                "model_name": model_name
            },
            "class_type": "UpscaleModelLoader",
            "_meta": {"title": "Load Upscale Model"}
        },
        "4": {
            "inputs": {
                "image": ["1", 0],
                "upscale_method": "lanczos",
                "width": inter_w,
                "height": inter_h,
                "crop": "disabled"
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Fit to Intermediate Latent Canvas"}
        },
        "5": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": "Load Diffusion Checkpoint"}
        },
        "6": {
            "inputs": {
                "text": prompt or "high quality, detailed masterpiece",
                "clip": ["5", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        "7": {
            "inputs": {
                "text": negative_prompt or "low quality, blurry, distorted",
                "clip": ["5", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        "8": {
            "inputs": {
                "pixels": ["4", 0],
                "vae": ["5", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode Refiner Image"}
        },
        "9": {
            "inputs": {
                "seed": seed,
                "steps": 20,
                "cfg": 4.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": denoise,
                "model": ["5", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["8", 0]
            },
            "class_type": "KSampler",
            "_meta": {"title": "Generative Refinement KSampler"}
        },
        "10": {
            "inputs": {
                "samples": ["9", 0],
                "vae": ["5", 2]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAE Decode"}
        },
        "3": {
            "inputs": {
                "upscale_model": ["2", 0],
                "image": ["10", 0]
            },
            "class_type": "ImageUpscaleWithModel",
            "_meta": {"title": "Model Super-Resolution"}
        },
        "12": {
            "inputs": {
                "image": ["3", 0],
                "upscale_method": "lanczos",
                "width": target_width,
                "height": target_height,
                "crop": "disabled"
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Scale to Final Target Dimensions"}
        },
        "11": {
            "inputs": {
                "images": ["12", 0],
                "filename_prefix": filename_prefix
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Save Generative Upscale"}
        }
    }


async def execute_upscale_core(
    image_bytes: bytes,
    filename: str,
    scale_factor: float = DEFAULT_SCALE_FACTOR,
    mode: str = "fast",
    style: str = "general",
    prompt: str = "",
    negative_prompt: str = "",
    checkpoint: str = PipelineDefaults.DEFAULT_SDXL_CHECKPOINT,
    client: Optional[ComfyClient] = None
) -> Dict[str, Any]:
    """
    Executes the upscale operation in ComfyUI, returning upscaled image bytes and metadata.
    Decoupled from Discord interaction.
    """
    active_client = client or _comfy_client
    
    # 1. Inspect original dimensions
    orig_w, orig_h = await get_image_dimensions_async(image_bytes)
    target_w, target_h = calculate_target_dimensions(orig_w, orig_h, scale_factor=scale_factor, divisible_by=8)
    
    # 2. Upload input image to ComfyUI
    clean_upload_name = f"upscale_input_{random.randint(10000, 99999)}_{filename}"
    upload_result = await active_client.upload_image(image_bytes, clean_upload_name, subfolder="_bot_temp")
    if upload_result:
        sub = upload_result.get("subfolder")
        uploaded_name = f"{sub}/{upload_result['name']}" if sub else upload_result.get("name")
    else:
        uploaded_name = clean_upload_name

    # 3. Pick upscale model
    chosen_model = MODEL_PRESETS.get(style.lower(), DEFAULT_UPSCALE_MODEL)
    
    # 4. Build workflow
    if mode.lower() in ["generative", "clarity", "refine"]:
        workflow = build_generative_upscale_workflow(
            image_filename=uploaded_name,
            target_width=target_w,
            target_height=target_h,
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            model_name=chosen_model,
            orig_width=orig_w,
            orig_height=orig_h,
        )
    else:
        workflow = build_fast_upscale_workflow(
            image_filename=uploaded_name,
            target_width=target_w,
            target_height=target_h,
            model_name=chosen_model,
        )
        
    # 5. Execute generation
    images = await active_client.generate(workflow, timeout=14400)
    if not images:
        raise RuntimeError("ComfyUI generation completed but returned no output image.")
        
    return {
        "image_bytes": images[0],
        "orig_width": orig_w,
        "orig_height": orig_h,
        "target_width": target_w,
        "target_height": target_h,
        "scale_factor": scale_factor,
        "mode": mode,
        "model": chosen_model,
        "filename": filename,
    }

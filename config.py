"""
Centralized Configuration & Constants for Shallot-CUI Bot.
"""

import os
import logging
from dotenv import load_dotenv
from discord import app_commands

load_dotenv()

# Configure Logging Level
LOG_LEVEL_STR = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL = getattr(logging, LOG_LEVEL_STR, logging.INFO)

# Environment Variables
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMFYUI_ADDRESS = os.getenv("COMFYUI_ADDRESS", "127.0.0.1:8188")
COMFYUI_CHECKPOINT = os.getenv("COMFYUI_CHECKPOINT", "waiIllustriousSDXL_v170.safetensors")
DEFAULT_NEGATIVE_PROMPT = os.getenv("DEFAULT_NEGATIVE_PROMPT", "blurry, low quality, distorted")
IMAGE_SAVE_PREFIX = os.getenv("IMAGE_SAVE_PREFIX", "Discord Bot/")
COMFYUI_BATCH_PATH = os.getenv("COMFYUI_BATCH_PATH", r"C:\ComfyUI\run_nvidia_gpu.bat")
VRAM_CAUTION_THRESHOLD_PERCENT = float(os.getenv("VRAM_CAUTION_THRESHOLD_PERCENT", "85.0"))
VRAM_MIN_FREE_GB = float(os.getenv("VRAM_MIN_FREE_GB", "2.0"))

# Curated SDXL Checkpoint choices for all SDXL workflows
SDXL_CHECKPOINT_CHOICES = [
    app_commands.Choice(name="🎨 [SDXL/Illu] Wai Illustrious SDXL v1.70 (Recommended Default)", value="waiIllustriousSDXL_v170.safetensors"),
    app_commands.Choice(name="📸 [SDXL/Real] RealVisXL V4.0 (Photorealistic)", value="RealVisXL_V4.0.safetensors"),
    app_commands.Choice(name="📸 [SDXL/Real] Juggernaut XL (Balanced Realism)", value="juggernautXL_ragnarok.safetensors"),
    app_commands.Choice(name="🎬 [SDXL/Real] Copax Timeless XL (Cinematic)", value="CopaxTimeLessXL.safetensors"),
    app_commands.Choice(name="📸 [SDXL/Real] Ultra Realistic XL v2.5", value="ultraRealisticByStable_v25.safetensors"),
    app_commands.Choice(name="🎨 [SDXL/Illu] Hyphoria Real Illu v0.9", value="hyphoriaRealIllu_v09.safetensors"),
    app_commands.Choice(name="🎨 [SDXL/Illu] Hyphoria NAI", value="hyphoriaIlluNAI_v001.safetensors"),
    app_commands.Choice(name="🎨 [SDXL/Illu] Illustrious Realism v1.0", value="illustriousRealismBy_v10VAE.safetensors"),
    app_commands.Choice(name="📸 [SDXL/Real] Big Lust v1.6 (SDXL Realism)", value="bigLust_v16.safetensors"),
    app_commands.Choice(name="📸 [SDXL/Real] Lustify v1.0 (SDXL Realism)", value="lustifySDXLNSFWSFW_v10.safetensors"),
    app_commands.Choice(name="🦄 [SDXL/Pony] Pony Diffusion V6 XL", value="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"),
    app_commands.Choice(name="⚡ [SDXL/Fast] RealVisXL V5.0 Lightning (Ultra Fast)", value="RealVisXL_V5.0_Lightning_fp16.safetensors"),
    app_commands.Choice(name="🦊 [SDXL/Illu] Nova Furry", value="novaFurryXL_ilV180A.safetensors"),
]

# Consolidated Enhancements choices
SDXL_ENHANCEMENT_CHOICES = [
    app_commands.Choice(name="🧠 Smart Art Director (Subject-Harmonized Prompt & Style)", value="smart"),
    app_commands.Choice(name="✨ Magic Prompt (Studio Lighting & Cinematic Expansion)", value="magic"),
    app_commands.Choice(name="🧠+✨ Smart Art Director + Magic Prompt", value="smart+magic"),
    app_commands.Choice(name="🚫 Disable FreeU (Pure Checkpoint Sampling)", value="no_freeu"),
]

FLUX_ENHANCEMENT_CHOICES = [
    app_commands.Choice(name="🧠 Smart Art Director (Subject-Harmonized)", value="smart"),
    app_commands.Choice(name="✨ Magic Prompt (Studio Lighting)", value="magic"),
    app_commands.Choice(name="🧠+✨ Smart Art Director + Magic Prompt", value="smart+magic"),
]

ICO_ENHANCEMENT_CHOICES = [
    app_commands.Choice(name="🔳 Square Corners (Disable Curved Edges)", value="square"),
    app_commands.Choice(name="✨ Magic Prompt Enhancer", value="magic"),
]

# Checkpoint-specific configurations & optimal generation parameters for photorealism and LoRA compatibility
CHECKPOINT_CONFIGS = {
    "RealVisXL_V4.0.safetensors": {
        "display_name": "RealVisXL V4.0",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 4.5,
        "negative_addon": "cgi, 3d, render, illustration, painting, cartoon, waxy skin, distorted eyes, bad anatomy",
    },
    "juggernautXL_ragnarok.safetensors": {
        "display_name": "Juggernaut XL (Ragnarok)",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "cgi, 3d, render, cartoon, deformed, lowres, bad anatomy, bad hands",
    },
    "CopaxTimeLessXL.safetensors": {
        "display_name": "Copax Timeless XL",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.5,
        "negative_addon": "cgi, 3d, cartoon, anime, bad lighting, low quality",
    },
    "ultraRealisticByStable_v25.safetensors": {
        "display_name": "Ultra Realistic XL v2.5",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "cgi, 3d, render, bad lighting, deformed, plastic skin",
    },
    "hyphoriaRealIllu_v09.safetensors": {
        "display_name": "Hyphoria Real Illu v0.9",
        "architecture": "sdxl",
        "sub_type": "illustrious",
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "steps": 28,
        "cfg": 6.0,
        "negative_addon": "bad quality, blurry, cgi, illustration",
    },
    "hyphoriaIlluNAI_v001.safetensors": {
        "display_name": "Hyphoria NAI",
        "architecture": "sdxl",
        "sub_type": "illustrious",
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "steps": 28,
        "cfg": 6.0,
        "negative_addon": "bad quality, blurry, cgi, illustration",
    },
    "illustriousRealismBy_v10VAE.safetensors": {
        "display_name": "Illustrious Realism v1.0",
        "architecture": "sdxl",
        "sub_type": "illustrious",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "anime, drawing, cartoon, cgi, lowres",
    },
    "bigLust_v16.safetensors": {
        "display_name": "Big Lust v1.6 (SDXL Realism)",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 28,
        "cfg": 4.0,
        "negative_addon": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, cgi, 3d render, lowres",
    },
    "lustifySDXLNSFWSFW_v10.safetensors": {
        "display_name": "Lustify v1.0 (SDXL Realism)",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 4.5,
        "negative_addon": "worst quality, low quality, blurry, bad anatomy, distorted face, plastic skin, cgi, 3d",
    },
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors": {
        "display_name": "Pony Diffusion V6 XL",
        "architecture": "sdxl",
        "sub_type": "pony",
        "sampler_name": "euler_ancestral",
        "scheduler": "karras",
        "steps": 25,
        "cfg": 6.0,
        "negative_addon": "score_6, score_5, score_4, rating_explicit, worst quality, low quality, blurry, bad anatomy",
    },
    "RealVisXL_V5.0_Lightning_fp16.safetensors": {
        "display_name": "RealVisXL V5.0 Lightning (Ultra Fast)",
        "architecture": "sdxl",
        "sub_type": "realistic",
        "sampler_name": "dpmpp_sde",
        "scheduler": "karras",
        "steps": 6,
        "cfg": 1.8,
        "negative_addon": "worst quality, low quality, normal quality, lowres, monochrome, grayscale, cgi, 3d",
    },
    "waiIllustriousSDXL_v170.safetensors": {
        "display_name": "Wai Illustrious SDXL v1.70",
        "architecture": "sdxl",
        "sub_type": "illustrious",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "anime, anime girl, manga, comic, cartoon, cel shaded, lineart, drawing, illustration, 2d, 3d cgi render, sketch, anime face, big eyes, flat shading, bad quality, blurry, distorted anatomy, bad hands, lowres",
    }
}

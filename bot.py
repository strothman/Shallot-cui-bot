import os
import sys
import socket
import io
import json
import random
import logging
import asyncio
import copy
import re
import subprocess
import time
import discord
import aiohttp
from PIL import Image
from discord import app_commands
from discord.ext import commands, tasks
from dotenv import load_dotenv
from datetime import datetime
from comfy_client import ComfyClient, StasisInterruptException
from error_handler import error_handler, ErrorCategory, ErrorSeverity, AutoFixAction, AutoFixResult
import db

# Refactored modular imports
from parsers import (
    parse_aspect_ratio,
    truncate_prompt,
    find_common_prefix,
    find_common_suffix,
    clean_quadrant_prompts,
    parse_loras,
    validate_workflow_loras,
    apply_loras_to_workflow,
    parse_seed,
    parse_stylize,
    generate_dynamic_style,
    parse_sref,
    parse_cref,
    apply_ipadapter_to_workflow,
    expand_dynamic_prompt,
    parse_magic_prompt,
    parse_smart_prompt,
    apply_smart_magic_and_sref,
    apply_magic_enhancement,
    extract_positive_prompt,
    clean_midjourney_flags,
    LOCKED_STYLE_PRESETS,
    build_scapes_prompt,
    calculate_wan_dimensions,
    apply_face_detailer_to_workflow,
)
from image_utils import (
    crop_to_aspect_ratio,
    create_grid,
    embed_metadata,
    calculate_outpaint_padding,
    save_quadrant_images,
    get_quadrant_bytes,
    crop_quadrant_from_grid_bytes,
    format_image_filename,
    get_dated_save_prefix,
    create_windows_ico_bytes,
    save_ico_file,
    apply_rounded_corners_to_bytes,
    convert_image_to_ico,
    upscale_isolated_image,
    boost_image_vibrancy_and_contrast,
    get_checkpoint_abbrev,
)
from views import (
    GridButtons,
    UpscaleButtons,
    IsolatedImageButtons,
    DescribeButtons,
    BlendButtons,
    build_blend_embed,
    EditBlendPromptModal,
    StasisControlsView,
    StasisPausedView,
    CustomSrefModal,
    SavedSrefSelectView,
    StudyButtons,
    StudyImagineModal,
    EditStyleModal,
    StylePaginationView,
    EditPromptModal,
    PromptPaginationView,
    AdoptButtons,
    EditAdoptPromptModal,
    LoraBuildGridButtons,
    LoraBuildStatusView,
    VideoPromptModal,
)
import lora_dataset
import model_architecture


# Load environment variables
load_dotenv()

# Configure Logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("discord.client").setLevel(logging.ERROR)
logging.getLogger("discord.gateway").setLevel(logging.ERROR)
logging.getLogger("discord.ext.commands").setLevel(logging.ERROR)
logger = logging.getLogger("DiscordBot")
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
COMFYUI_ADDRESS = os.getenv("COMFYUI_ADDRESS", "127.0.0.1:8188")
COMFYUI_CHECKPOINT = os.getenv("COMFYUI_CHECKPOINT", "waiIllustriousSDXL_v170.safetensors")
DEFAULT_NEGATIVE_PROMPT = os.getenv("DEFAULT_NEGATIVE_PROMPT", "blurry, low quality, distorted")
IMAGE_SAVE_PREFIX = os.getenv("IMAGE_SAVE_PREFIX", "Discord Bot/")
COMFYUI_BATCH_PATH = os.getenv("COMFYUI_BATCH_PATH", r"C:\ComfyUI\run_nvidia_gpu.bat")
VRAM_CAUTION_THRESHOLD_PERCENT = float(os.getenv("VRAM_CAUTION_THRESHOLD_PERCENT", "85.0"))
VRAM_MIN_FREE_GB = float(os.getenv("VRAM_MIN_FREE_GB", "2.0"))
from config import BOT_OWNER_ID, is_authorized_admin

# Curated SDXL Checkpoint choices for all SDXL workflows (keeps lists clean and free of non-SDXL models)
SDXL_CHECKPOINT_CHOICES = [
    app_commands.Choice(name="Wai Illustrious SDXL v1.70 (Recommended Default)", value="waiIllustriousSDXL_v170.safetensors"),
    app_commands.Choice(name="RealVisXL V4.0 (Photorealistic)", value="RealVisXL_V4.0.safetensors"),
    app_commands.Choice(name="Juggernaut XL (Balanced Realism)", value="juggernautXL_ragnarok.safetensors"),
    app_commands.Choice(name="Copax Timeless XL (Cinematic)", value="CopaxTimeLessXL.safetensors"),
    app_commands.Choice(name="Ultra Realistic XL v2.5", value="ultraRealisticByStable_v25.safetensors"),
    app_commands.Choice(name="Hyphoria Real Illu v0.9", value="hyphoriaRealIllu_v09.safetensors"),
    app_commands.Choice(name="Hyphoria NAI", value="hyphoriaIlluNAI_v001.safetensors"),
    app_commands.Choice(name="Illustrious Realism v1.0", value="illustriousRealismBy_v10VAE.safetensors"),
    app_commands.Choice(name="Big Lust v1.6 (SDXL Realism)", value="bigLust_v16.safetensors"),
    app_commands.Choice(name="Lustify v1.0 (SDXL Realism)", value="lustifySDXLNSFWSFW_v10.safetensors"),
    app_commands.Choice(name="Pony Diffusion V6 XL", value="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"),
    app_commands.Choice(name="RealVisXL V5.0 Lightning (Ultra Fast)", value="RealVisXL_V5.0_Lightning_fp16.safetensors"),
    app_commands.Choice(name="Nova Furry", value="novaFurryXL_ilV180A.safetensors"),
]

# Consolidated Enhancements choices (replaces multiple True/False toggles with a clean dropdown)
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
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 4.5,
        "negative_addon": "cgi, 3d, render, illustration, painting, cartoon, waxy skin, distorted eyes, bad anatomy",
    },
    "juggernautXL_ragnarok.safetensors": {
        "display_name": "Juggernaut XL (Ragnarok)",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "cgi, 3d, render, cartoon, deformed, lowres, bad anatomy, bad hands",
    },
    "CopaxTimeLessXL.safetensors": {
        "display_name": "Copax Timeless XL",
        "sampler_name": "dpmpp_2m",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.5,
        "negative_addon": "cgi, 3d, cartoon, anime, bad lighting, low quality",
    },
    "ultraRealisticByStable_v25.safetensors": {
        "display_name": "Ultra Realistic XL v2.5",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "cgi, 3d, render, bad lighting, deformed, plastic skin",
    },
    "hyphoriaRealIllu_v09.safetensors": {
        "display_name": "Hyphoria Real Illu v0.9",
        "sampler_name": "euler_ancestral",
        "scheduler": "normal",
        "steps": 28,
        "cfg": 6.0,
        "negative_addon": "bad quality, blurry, cgi, illustration",
    },
    "illustriousRealismBy_v10VAE.safetensors": {
        "display_name": "Illustrious Realism v1.0",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "anime, drawing, cartoon, cgi, lowres",
    },
    "bigLust_v16.safetensors": {
        "display_name": "Big Lust v1.6 (SDXL Realism)",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 28,
        "cfg": 4.0,
        "negative_addon": "worst quality, low quality, bad anatomy, bad hands, missing fingers, extra digits, cgi, 3d render, lowres",
    },
    "lustifySDXLNSFWSFW_v10.safetensors": {
        "display_name": "Lustify v1.0 (SDXL Realism)",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 4.5,
        "negative_addon": "worst quality, low quality, blurry, bad anatomy, distorted face, plastic skin, cgi, 3d",
    },
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors": {
        "display_name": "Pony Diffusion V6 XL",
        "sampler_name": "euler_ancestral",
        "scheduler": "karras",
        "steps": 25,
        "cfg": 6.0,
        "negative_addon": "score_6, score_5, score_4, rating_explicit, worst quality, low quality, blurry, bad anatomy",
    },
    "RealVisXL_V5.0_Lightning_fp16.safetensors": {
        "display_name": "RealVisXL V5.0 Lightning (Ultra Fast)",
        "sampler_name": "dpmpp_sde",
        "scheduler": "karras",
        "steps": 6,
        "cfg": 1.8,
        "negative_addon": "worst quality, low quality, normal quality, lowres, monochrome, grayscale, cgi, 3d",
    },
    "waiIllustriousSDXL_v170.safetensors": {
        "display_name": "Wai Illustrious SDXL v1.70",
        "sampler_name": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 30,
        "cfg": 5.0,
        "negative_addon": "anime, anime girl, manga, comic, cartoon, cel shaded, lineart, drawing, illustration, 2d, 3d cgi render, sketch, anime face, big eyes, flat shading, bad quality, blurry, distorted anatomy, bad hands, lowres",
    }
}

# Global process tracker for ComfyUI server
comfy_process = None

def check_gpu_vram_caution() -> tuple[bool, dict]:
    """
    Checks GPU VRAM usage using nvidia-smi.
    Returns (is_caution, stats) where is_caution is True if VRAM usage is above threshold or free VRAM is below minimum.
    """
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        lines = res.stdout.strip().splitlines()
        for line in lines:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                name = parts[0]
                used_mb = float(parts[1])
                total_mb = float(parts[2])
                free_mb = float(parts[3])

                used_gb = used_mb / 1024.0
                total_gb = total_mb / 1024.0
                free_gb = free_mb / 1024.0
                percent = (used_mb / total_mb * 100.0) if total_mb > 0 else 0.0

                stats = {
                    "name": name,
                    "used_gb": used_gb,
                    "total_gb": total_gb,
                    "free_gb": free_gb,
                    "percent_used": percent
                }

                if percent >= VRAM_CAUTION_THRESHOLD_PERCENT or free_gb < VRAM_MIN_FREE_GB:
                    return True, stats
                return False, stats
    except Exception as e:
        logger.debug(f"Failed to check VRAM via nvidia-smi: {e}")

    return False, {}

# Initialize ComfyUI client
comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)


# Setup Bot
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

# Active generations SQLite Proxy
class ActiveGenerationsProxy:
    def __getitem__(self, key):
        val = db.get_generation(str(key))
        if val is None:
            raise KeyError(key)
        return val

    def get(self, key, default=None):
        val = db.get_generation(str(key))
        return val if val is not None else default

    def __setitem__(self, key, value):
        db.save_generation(str(key), value)

    def __contains__(self, key):
        return db.get_generation(str(key)) is not None

active_generations = ActiveGenerationsProxy()

def load_generations():
    db.init_db()

def get_generation(generation_id: str) -> dict:
    """Get generation data by ID from SQLite database proxy."""
    return active_generations.get(generation_id)

def save_generations():
    pass

def cleanup_orphaned_quadrants():
    db.cleanup_orphaned_quadrants()

SETTINGS_FILE = "settings.json"
settings = {"variation_mode": "high"}

def load_settings():
    global settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r") as f:
                settings = json.load(f)
            logger.info(f"Loaded {len(settings)} configuration setting(s) from {SETTINGS_FILE}.")
            logger.debug(f"Settings contents: {settings}")
        except Exception as e:
            logger.error(f"Error loading settings file: {e}")
            settings = {"variation_mode": "high"}
    else:
        settings = {"variation_mode": "high"}

def save_settings():
    global settings
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving settings file: {e}")

async def safe_defer(interaction: discord.Interaction, thinking: bool = False, ephemeral: bool = False):
    """Safely defers an interaction response without raising 404 Unknown Interaction errors if token expired."""
    if not interaction.response.is_done():
        try:
            await interaction.response.defer(thinking=thinking, ephemeral=ephemeral)
        except (discord.NotFound, discord.HTTPException) as e:
            logger.debug(f"Interaction defer skipped or expired: {e}")


async def _update_button_state(interaction, custom_id, style, disabled=True):
    """Helper to update a button's style and disabled state on the original message."""
    if not interaction.message:
        return
    try:
        msg = interaction.message
        view = discord.ui.View.from_message(msg)
        updated = False
        for child in view.children:
            if isinstance(child, discord.ui.Button) and child.custom_id == custom_id:
                child.style = style
                child.disabled = disabled
                updated = True
                break
        if not updated:
            return

        if not interaction.response.is_done():
            try:
                await interaction.response.edit_message(view=view)
                return
            except Exception:
                pass
        else:
            try:
                await interaction.edit_original_response(view=view)
                return
            except Exception:
                pass
        
        # Fallback: Edit the channel message directly
        try:
            await msg.edit(view=view)
        except Exception as edit_err:
            logger.debug(f"Could not edit message directly: {edit_err}")
    except Exception as e:
        logger.debug(f"Could not update button style: {e}")


def create_progress_bar(value: int, max_val: int, length: int = 10) -> str:
    """Renders a text progress bar with percentage and step count."""
    if max_val <= 0:
        percent = 0
    else:
        percent = min(100, int((value / max_val) * 100))
    filled = int(round((percent / 100) * length))
    bar = "█" * filled + "░" * (length - filled)
    return f"`[{bar}] {percent}%` (Step {value}/{max_val})"

def update_console_title(status_text: str = None):
    """Updates the Windows Command Prompt / Terminal window title bar with live status."""
    if not status_text:
        full_title = "Shallot-CUI Bot"
    else:
        full_title = f"Shallot-CUI Bot | {status_text}"
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetConsoleTitleW(full_title)
        sys.stdout.write(f"\x1b]2;{full_title}\x07")
        sys.stdout.flush()
    except Exception:
        pass


async def update_bot_presence(status_text: str = None):
    """Updates bot activity presence in the server user sidebar and console window title."""
    update_console_title(status_text)
    if not bot.is_ready():
        return
    try:
        if status_text:
            activity = discord.Activity(type=discord.ActivityType.custom, name="Custom Status", state=status_text)
            await bot.change_presence(activity=activity)
        else:
            activity = discord.Activity(type=discord.ActivityType.custom, name="Custom Status", state="Processing 0 jobs")
            await bot.change_presence(activity=activity)
    except Exception as e:
        logger.debug(f"Failed to update bot presence: {e}")


async def send_followup_fallback(interaction, content=None, embed=None, file=None, files=None, view=None, ephemeral=False):
    """Sends a follow-up message using interaction, with fallback to channel.send if expired."""
    kwargs = {}
    if content is not None:
        kwargs["content"] = content
    if embed is not None:
        kwargs["embed"] = embed
    if file is not None:
        kwargs["file"] = file
    if files is not None:
        kwargs["files"] = files
    if view is not None:
        kwargs["view"] = view
    try:
        return await interaction.followup.send(**kwargs, ephemeral=ephemeral)
    except (discord.HTTPException, discord.NotFound) as hex:
        if getattr(hex, 'code', None) in [50027, 10062, 10015] or getattr(hex, 'status', None) in [404, 400] or isinstance(hex, discord.NotFound):
            logger.info(f"Interaction token expired ({getattr(hex, 'code', '404')}). Falling back to channel.send.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    channel = await bot.fetch_channel(interaction.channel_id)
                except Exception:
                    pass
            if not channel:
                return None
            if file:
                file.fp.seek(0)
            if files:
                for f in files:
                    f.fp.seek(0)
            tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
            if tag:
                if "content" in kwargs and kwargs["content"]:
                    if interaction.user.mention not in kwargs["content"]:
                        kwargs["content"] = f"{tag}{kwargs['content']}"
                else:
                    kwargs["content"] = tag.strip()
            return await channel.send(**kwargs)
        else:
            raise hex

async def send_error_fallback(interaction, message):
    """Sends an error message using interaction, with fallback to channel.send if expired."""
    # Ensure error message fits within Discord's 2000 character limit
    if len(message) > 1980:
        message = message[:1977] + "..."
    try:
        await interaction.followup.send(message, ephemeral=True)
    except (discord.HTTPException, discord.NotFound) as hex:
        logger.info(f"Interaction token expired ({getattr(hex, 'code', '404')}) during error report. Falling back to channel.send.")
        try:
            channel = interaction.channel
            if not channel and interaction.channel_id:
                channel = await bot.fetch_channel(interaction.channel_id)
            if channel:
                tag = f"{interaction.user.mention} " if (interaction and interaction.user) else ""
                await channel.send(f"{tag}❌ {message.replace('❌ ', '')}")
        except Exception as e:
            logger.debug(f"Failed channel.send fallback in send_error_fallback: {e}")

async def download_image(url: str) -> bytes:
    """Downloads image bytes from a remote URL using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise Exception(f"HTTP {resp.status} fetching image from {url}")

async def edit_original_fallback(interaction, content=None, embed=None, view=None):
    """Edits the original interaction response, with fallback to channel.send if expired."""
    edit_kwargs = {}
    send_kwargs = {}
    if content is not None:
        edit_kwargs["content"] = content
        send_kwargs["content"] = content
    if embed is not None:
        edit_kwargs["embed"] = embed
        send_kwargs["embed"] = embed
    if view is not None:
        edit_kwargs["view"] = view
        send_kwargs["view"] = view
    else:
        edit_kwargs["view"] = None

    try:
        await interaction.edit_original_response(**edit_kwargs)
    except (discord.HTTPException, discord.NotFound) as hex:
        if getattr(hex, 'code', None) in [50027, 10062, 10015] or getattr(hex, 'status', None) in [404, 400] or isinstance(hex, discord.NotFound):
            logger.info(f"Interaction token expired ({getattr(hex, 'code', '404')}) during edit. Falling back to channel.send.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    channel = await bot.fetch_channel(interaction.channel_id)
                except Exception:
                    pass
            if channel:
                tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
                if "content" in send_kwargs:
                    send_kwargs["content"] = f"{tag}{send_kwargs['content']}"
                else:
                    send_kwargs["content"] = f"{tag}Image Description Complete"
                await channel.send(**send_kwargs)
        else:
            raise hex


async def edit_message_fallback(interaction, message_id, content=None, embed=None, file=None, view=None):
    """Edits a status message by ID. Falls back to channel.send if interaction token expired/fails."""
    chan_id = interaction.channel_id
    edit_kwargs = {}
    send_kwargs = {}
    if content is not None:
        edit_kwargs["content"] = content
        send_kwargs["content"] = content
    if embed is not None:
        edit_kwargs["embed"] = embed
        send_kwargs["embed"] = embed
    if view is not None:
        edit_kwargs["view"] = view
        send_kwargs["view"] = view
    else:
        edit_kwargs["view"] = None

    if chan_id and message_id:
        try:
            channel = interaction.channel or await bot.fetch_channel(chan_id)
            message = await channel.fetch_message(message_id)
            if message:
                if file:
                    file.fp.seek(0)
                    edit_kwargs["attachments"] = [file]
                await message.edit(**edit_kwargs)
                return
        except Exception as e:
            logger.info(f"Could not edit message via Bot API ({e}). Falling back to interaction/channel send.")
    
    try:
        if file:
            file.fp.seek(0)
            edit_kwargs["attachments"] = [file]
        await interaction.followup.edit_message(message_id, **edit_kwargs)
    except (discord.HTTPException, discord.NotFound) as hex:
        if getattr(hex, 'code', None) in [50027, 10062, 10015] or getattr(hex, 'status', None) in [404, 400] or isinstance(hex, discord.NotFound):
            logger.info(f"Interaction token expired during message edit. Sending new message to channel.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    channel = await bot.fetch_channel(interaction.channel_id)
                except Exception:
                    pass
            if channel:
                if file:
                    file.fp.seek(0)
                    send_kwargs["file"] = file
                await channel.send(**send_kwargs)
        else:
            raise hex


async def complete_grid_generation(interaction, generation_id, images, gen_data, status_message_id=None, timing_data=None):
    prompt = gen_data.get("prompt", "")
    display_prompt = gen_data.get("original_prompt", prompt)
    neg_prompt = gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    seed = gen_data.get("seed", 0)
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    selected_model = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", 4.0)
    is_magic = gen_data.get("is_magic", False)
    sref_info = gen_data.get("sref_info")
    sref_image_name = gen_data.get("sref_image")
    sref_weight = gen_data.get("sref_weight", 0.6)
    cref_image_name = gen_data.get("cref_image")
    cref_weight = gen_data.get("cref_weight", 1.0)
    expanded_prompts = gen_data.get("expanded_prompts", [])
    prepend_quality = gen_data.get("prepend_quality", True)

    save_quadrant_images(generation_id, images)

    sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
    grid_file_io = await asyncio.to_thread(create_grid, images, prompt, neg_prompt, seed, width, height)
    is_ico = gen_data.get("is_ico", False)
    is_flux = gen_data.get("is_flux", False)
    is_junji = gen_data.get("is_junji", False)
    ckpt_abbrev = get_checkpoint_abbrev(selected_model)
    if is_ico:
        grid_prefix = f"icon_{ckpt_abbrev}_grid"
    elif is_junji:
        grid_prefix = f"junji_{ckpt_abbrev}_grid"
    elif is_flux:
        grid_prefix = f"flux_{ckpt_abbrev}_grid"
    else:
        grid_prefix = f"grid_{ckpt_abbrev}"
    file = discord.File(fp=grid_file_io, filename=format_image_filename(grid_prefix, seed, "jpg", sref=sref_code))
    
    desc_parts = [f"**Prompt:** {truncate_prompt(display_prompt, 250)}", f"**Model:** {selected_model}", f"**Seed:** {seed}", f"**Size:** {width}x{height}"]
    if "{" in display_prompt and "}" in display_prompt and expanded_prompts:
        desc_parts.append("\n**Selected Quadrant Prompts:**")
        cleaned_eps = clean_quadrant_prompts(expanded_prompts, display_prompt)
        for idx, clean_ep in enumerate(cleaned_eps):
            if len(clean_ep) > 120:
                clean_ep = clean_ep[:117] + "..."
            desc_parts.append(f"* **Q{idx+1}:** {clean_ep}")
    if sref_info and "code" in sref_info:
        desc_parts.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
    if cref_image_name:
        desc_parts.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")
    jump_url = gen_data.get("jump_url")
    if jump_url:
        desc_parts.append(f"**Original Post:** [Jump to Midjourney Message]({jump_url})")
    if cfg != 4.0:
        desc_parts.append(f"**CFG:** {cfg:.1f}")
    if is_magic:
        desc_parts.append("**Magic Prompt:** ✨ Enabled")
    if sref_image_name:
        desc_parts.append(f"**Style Ref:** ✅ (weight: {sref_weight})")
    if not prepend_quality:
        desc_parts.append("**Mode:** Raw")
    if gen_data.get("is_face_detailer") and not gen_data.get("is_flux"):
        desc_parts.append("**Face Detailer:** ✨ Active (High Precision)")
    
    user_name = interaction.user.name if (interaction and interaction.user) else "User"
    user_id = interaction.user.id if (interaction and interaction.user) else 0

    if gen_data.get("is_lora_build"):
        title_txt = "🎨 LoRA Dataset Grid Complete"
        session_id = gen_data.get("lora_session_id", "")
        view = LoraBuildGridButtons(generation_id, session_id)
    else:
        title_txt = "Flux Grid Complete" if gen_data.get("is_flux") else ("Windows 11 Icon Grid Complete" if gen_data.get("is_ico") else "Image Generation Complete")
        has_sref = sref_info is not None and "code" in sref_info
        view = GridButtons(generation_id, has_sref=has_sref)

    embed = discord.Embed(
        title=title_txt, 
        description="\n".join(desc_parts)
    )
    
    timing_text = ""
    if timing_data:
        elapsed = timing_data.get("elapsed_time", 0.0)
        init_sec = timing_data.get("init_seconds", 0.0)
        samp_sec = timing_data.get("sampling_seconds", 0.0)
        post_sec = timing_data.get("post_seconds", 0.0)
        timing_text = f" • Rendered in {elapsed:.1f}s (Init: {init_sec:.1f}s | Sample: {samp_sec:.1f}s | Post: {post_sec:.1f}s)"

    embed.set_footer(text=f"Requested by {user_name} (ID: {user_id}){timing_text}")
    
    tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
    content = f"{tag}**Imagine:** {truncate_prompt(display_prompt, 100)}"

    # In-Place Message Transformation: Edit the original progress status message directly
    if status_message_id and interaction:
        try:
            await edit_message_fallback(interaction, status_message_id, content=content, embed=embed, file=file, view=view)
            return
        except Exception as edit_err:
            logger.info(f"Could not transform status message in-place ({edit_err}). Falling back to new message delivery.")

    posted = False
    if interaction and interaction.channel_id:
        try:
            channel = interaction.channel or await bot.fetch_channel(interaction.channel_id)
            if channel:
                file.fp.seek(0)
                await channel.send(content=content, embed=embed, file=file, view=view)
                posted = True
        except Exception as send_err:
            logger.warning(f"Could not send message via channel.send ({send_err}). Falling back to followup.")

    if not posted and interaction:
        await send_followup_fallback(interaction, content=content, embed=embed, file=file, view=view)



async def handle_upscale(interaction: discord.Interaction, generation_id: str, index: int, force_new_seed: bool = False, upscale_scale: str = "1.25"):
    # Defer response as image generation might take time
    await safe_defer(interaction, thinking=True)

    # Immediately mark the button as pending (blue + disabled)
    clicked_custom_id = f"upscale:{generation_id}:{index}"
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        # Revert button since we can't proceed
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find generation session data. It may have expired or the bot was restarted.", ephemeral=True)
        return
        
    original_prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    prompt = gen_data.get("prompt", "")
    neg_prompt = gen_data.get("neg_prompt", gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    base_seed = gen_data.get("seed", 12345)
    target_seed = random.randint(1, 1125899906842624) if force_new_seed else (base_seed + index - 1)
    
    # Deterministically expand prompt for this specific quadrant using its original seed
    orig_quadrant_seed = base_seed + index - 1
    expanded_prompt = expand_dynamic_prompt(prompt, random.Random(orig_quadrant_seed))
    
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", 4.0)
    is_flux = gen_data.get("is_flux") or (checkpoint and "flux" in str(checkpoint).lower())
    loras = gen_data.get("loras")
    if not loras:
        _, loras = parse_loras(original_prompt or prompt, is_flux=is_flux)
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    sref_image = gen_data.get("sref_image")
    sref_weight = gen_data.get("sref_weight", 0.6)

    try:
        upscale_factor = float(upscale_scale)
    except (ValueError, TypeError):
        upscale_factor = 1.25

    # 1. Retrieve quadrant bytes from cache
    q_bytes = get_quadrant_bytes(generation_id, index)
    q_filename = None

    if q_bytes:
        try:
            upload_result = await comfy_client.upload_image(q_bytes, f"upscale_input_{generation_id}_{index}.png")
            q_filename = upload_result.get("name")
            logger.info(f"Uploaded quadrant {index} for detailed upscale: {q_filename}")
        except Exception as e:
            logger.error(f"Failed to upload quadrant image to ComfyUI for detail upscale: {e}")

    # 2. Select detail upscale workflow
    if is_flux:
        detail_wf = "workflows/flux_lowres.json"
    elif q_filename:
        detail_wf = "workflows/img_highres_sref_detail.json" if sref_image else "workflows/img_highres_detail.json"
    else:
        detail_wf = "workflows/txt2img_sref_highres.json" if sref_image else "workflows/txt2img_highres.json"

    try:
        with open(detail_wf, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow file: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Failed to load upscale workflow file.", ephemeral=True)
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    cref_image = gen_data.get("cref_image")
    cref_weight = gen_data.get("cref_weight", 0.80)
    if cref_image and not is_flux:
        workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=cref_weight, node_prefix="cref")

    sref_info = gen_data.get("sref_info")
    sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""

    try:
        if is_flux:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            workflow["6"]["inputs"]["text"] = expanded_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt

            if q_filename:
                workflow["30"] = {
                    "inputs": {"image": q_filename},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Input Image"}
                }
                scaled_w = int(round(width * upscale_factor / 64) * 64)
                scaled_h = int(round(height * upscale_factor / 64) * 64)
                workflow["34_pixel_scale"] = {
                    "inputs": {
                        "image": ["30", 0],
                        "upscale_method": "lanczos",
                        "width": scaled_w,
                        "height": scaled_h,
                        "crop": "disabled"
                    },
                    "class_type": "ImageScale",
                    "_meta": {"title": "Pixel Scale Image"}
                }
                workflow["31_flux_vae"] = {
                    "inputs": {
                        "pixels": ["34_pixel_scale", 0],
                        "vae": ["3", 0]
                    },
                    "class_type": "VAEEncode",
                    "_meta": {"title": "Flux VAE Encode Upscaled"}
                }
                denoise_val = 0.26 if upscale_factor <= 1.3 else 0.35
                if "11" in workflow:
                    workflow["11"]["inputs"]["seed"] = target_seed
                    workflow["11"]["inputs"]["steps"] = 20
                    workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0
                    workflow["11"]["inputs"]["denoise"] = denoise_val
                    workflow["11"]["inputs"]["latent_image"] = ["31_flux_vae", 0]
                    logger.info(f"FLUX detail upscale applied via Pixel Scale + VAE (scale: {upscale_factor}x, denoise: {denoise_val})")
            else:
                if "11" in workflow:
                    workflow["11"]["inputs"]["seed"] = target_seed
                    workflow["11"]["inputs"]["steps"] = 20
                    workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0

            seed_suffix = f"_seed{target_seed}"
            workflow["9"]["class_type"] = "SaveImage"
            workflow["9"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('flux')}Flux_Upscale{seed_suffix}{sref_suffix}"
        else:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["6"]["inputs"]["text"] = expanded_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt

            # Detail upscale specific configuration
            try:
                upscale_factor = float(upscale_scale)
            except ValueError:
                upscale_factor = 1.25

            if "10" in workflow and workflow["10"].get("class_type") == "LatentUpscaleBy":
                workflow["10"]["inputs"]["scale_by"] = upscale_factor

            is_blend = gen_data.get("is_blend", False) or "caption" in gen_data or "uploaded_image_name" in gen_data
            is_ico = gen_data.get("is_ico", False)
            is_junji = gen_data.get("is_junji", False)
            is_flux = gen_data.get("is_flux", False)

            if is_blend:
                subfolder = "blend"
            elif is_ico:
                subfolder = "ico"
            elif is_junji:
                subfolder = "junji"
            elif is_flux:
                subfolder = "flux"
            else:
                subfolder = "imagine"
            prefix_tag = "BlendHighRes" if is_blend else "DetailHighRes"

            seed_suffix = f"_seed{target_seed}"
            if q_filename and "30" in workflow:
                workflow["30"]["inputs"]["image"] = q_filename
                workflow["11"]["inputs"]["seed"] = target_seed
                workflow["11"]["inputs"]["cfg"] = cfg
                workflow["11"]["inputs"]["denoise"] = 0.55
                workflow["13"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix(subfolder)}{prefix_tag}{seed_suffix}{sref_suffix}"
            else:
                workflow["3"]["inputs"]["seed"] = target_seed
                workflow["3"]["inputs"]["cfg"] = cfg
                workflow["5"]["inputs"]["width"] = width
                workflow["5"]["inputs"]["height"] = height
                workflow["11"]["inputs"]["seed"] = target_seed
                workflow["11"]["inputs"]["cfg"] = cfg
                workflow["13"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix(subfolder)}{prefix_tag}{seed_suffix}{sref_suffix}"

            if sref_image and "21" in workflow:
                workflow["21"]["inputs"]["image"] = sref_image
                if "23" in workflow:
                    workflow["23"]["inputs"]["weight"] = sref_weight
    except KeyError as e:
        logger.error(f"Invalid workflow structure: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Upscale workflow structure is invalid or mismatched.", ephemeral=True)
        return

    status_lbl = "Detail Reprocessing & Upscaling" if q_filename else "Latent Upscaling"
    await interaction.followup.send(f"{status_lbl} Image {index} (Seed: {target_seed}, Denoise: 0.55)...", ephemeral=True)
    
    try:
        images = await comfy_client.generate(workflow, timeout=14400)
        if not images:
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
            await interaction.followup.send("ComfyUI did not return any image.", ephemeral=True)
            return
        
        out_w = int(round((width * upscale_factor) / 64) * 64)
        out_h = int(round((height * upscale_factor) / 64) * 64)
        highres_file_io = embed_metadata(images[0], expanded_prompt, neg_prompt, target_seed, out_w, out_h)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        ckpt_abbrev = get_checkpoint_abbrev(checkpoint)
        if is_blend:
            upscale_prefix = f"blend_{ckpt_abbrev}_upscale_{index}"
        elif is_ico:
            upscale_prefix = f"icon_{ckpt_abbrev}_upscale_{index}"
        elif is_junji:
            upscale_prefix = f"junji_{ckpt_abbrev}_upscale_{index}"
        elif is_flux:
            upscale_prefix = f"flux_{ckpt_abbrev}_upscale_{index}"
        else:
            upscale_prefix = f"upscale_{ckpt_abbrev}_{index}"
        file = discord.File(fp=highres_file_io, filename=format_image_filename(upscale_prefix, target_seed, "png", sref=sref_code))

        sref_info = gen_data.get("sref_info")
        desc_lines = [
            f"**Prompt:** {truncate_prompt(expanded_prompt, 250)}",
            f"**Model:** {checkpoint}",
            f"**Detail Denoise:** 0.55",
            f"**Seed:** {target_seed}",
            f"**Size:** {out_w}x{out_h}"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image:
            desc_lines.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")
        jump_url = gen_data.get("jump_url")
        if jump_url:
            desc_lines.append(f"**Original Post:** [Jump to Midjourney Message]({jump_url})")

        title_txt = f"Detailed High-Res Blend Upscale Image {index}" if is_blend else f"Detailed High-Res Upscale Image {index}"
        embed = discord.Embed(
            title=title_txt, 
            description="\n".join(desc_lines)
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
        has_sref = sref_info is not None and "code" in sref_info
        view = UpscaleButtons(generation_id, index, upscale_scale=upscale_scale, has_sref=has_sref)
        content_prefix = f"**Blend Upscale Image {index}:**" if is_blend else f"**Upscale Image {index}:**"
        await send_followup_fallback(interaction, content=f"{content_prefix} {truncate_prompt(expanded_prompt, 100)}", embed=embed, file=file, view=view)

        # Mark the button as completed (green + disabled)
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)

    except Exception as e:
        entry = error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="handle_upscale",
            source_file="bot.py",
            severity=ErrorSeverity.ERROR,
            context={"generation_id": generation_id, "index": index, "checkpoint": checkpoint, "width": width, "height": height}
        )
        recipe = error_handler.find_recipe(str(e), category=ErrorCategory.WORKFLOW)
        if recipe and recipe.action == AutoFixAction.RETRY_REDUCED_RES and error_handler.can_retry(recipe, generation_id):
            error_handler.record_retry(recipe, generation_id)
            await send_error_fallback(interaction, "⚠️ Generation encountered an issue. Retrying with auto-fix adjustment (reduced resolution)...")
            try:
                # Reduce resolution by 25% for retry
                if "5" in workflow:
                    workflow["5"]["inputs"]["width"] = int(width * 0.75)
                    workflow["5"]["inputs"]["height"] = int(height * 0.75)
                retry_images = await comfy_client.generate(workflow, timeout=14400)
                if retry_images:
                    out_w, out_h = int(round((width * upscale_factor) / 64) * 64), int(round((height * upscale_factor) / 64) * 64)
                    highres_file_io = embed_metadata(retry_images[0], original_prompt, neg_prompt, target_seed, out_w, out_h)
                    file = discord.File(fp=highres_file_io, filename=format_image_filename(f"upscale_{index}", target_seed, "png"))
                    embed = discord.Embed(
                        title=f"Detailed High-Res Upscale Image {index} (Auto-Fixed)", 
                        description=f"**Prompt:** {truncate_prompt(original_prompt, 250)}\n**Model:** {checkpoint}\n**Detail Denoise:** 0.55\n**Seed:** {target_seed}\n**Size:** {out_w}x{out_h}\n*Note: Reduced resolution auto-fix applied.*"
                    )
                    embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
                    has_sref = sref_info is not None and "code" in sref_info
                    view = UpscaleButtons(generation_id, index, upscale_scale=upscale_scale, has_sref=has_sref)
                    await send_followup_fallback(interaction, content=f"**Upscale Image {index} (Auto-Fixed):** {truncate_prompt(original_prompt, 100)}", embed=embed, file=file, view=view)
                    error_handler.log_error_with_fix(entry, action=AutoFixAction.RETRY_REDUCED_RES, result=AutoFixResult.SUCCESS, detail="Retried successfully with 75% resolution")
                    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)
                    return
            except Exception as retry_e:
                error_handler.log_error_with_fix(entry, action=AutoFixAction.RETRY_REDUCED_RES, result=AutoFixResult.FAILED, detail=f"Retry failed: {retry_e}")

        logger.error(f"Error generating upscale: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
        await send_error_fallback(interaction, f"An error occurred during generation: {e}")


async def handle_isolate(interaction: discord.Interaction, generation_id: str, index: int):
    """Instantly extracts the cached quadrant image (1.0x scale) using Pillow and posts it to Discord."""
    # Defer response as we need to load/send the image
    await safe_defer(interaction)

    clicked_custom_id = f"upscale:{generation_id}:{index}"
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    q_bytes = get_quadrant_bytes(generation_id, index)
    if not q_bytes:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find the cached quadrant image. You may need to re-roll the grid.", ephemeral=True)
        return

    original_prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    prompt = gen_data.get("prompt", "")
    neg_prompt = gen_data.get("neg_prompt", gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    base_seed = gen_data.get("seed", 12345)
    target_seed = base_seed + index - 1
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    sref_info = gen_data.get("sref_info")

    # Automatically upscale low-res grid quadrant slice to full HD resolution
    q_bytes, width, height = upscale_isolated_image(q_bytes, target_w=width, target_h=height)

    try:
        # Embed metadata to the isolated PNG
        metadata_io = embed_metadata(q_bytes, original_prompt, neg_prompt, target_seed, width, height)
        png_bytes = metadata_io.getvalue()

        # Save copy to ComfyUI output highres / flux / blend folder inside MM/DD
        try:
            comfy_output_path = os.getenv("COMFYUI_OUTPUT_PATH", "C:/ComfyUI/ComfyUI/output")
            now = datetime.now()
            mm = now.strftime("%m")
            dd = now.strftime("%d")
            time_str = now.strftime("%Y%m%d_%H%M%S")
            is_blend = gen_data.get("is_blend", False) or "caption" in gen_data or "uploaded_image_name" in gen_data
            is_ico = gen_data.get("is_ico", False)
            is_junji = gen_data.get("is_junji", False)
            if is_blend:
                sub = "blend"
            elif is_ico:
                sub = "ico"
            elif is_junji:
                sub = "junji"
            elif gen_data.get("is_flux") or "flux" in str(gen_data.get("checkpoint", "")).lower():
                sub = "flux"
            else:
                sub = "imagine"

            target_dir = os.path.join(comfy_output_path, "Discord Bot", mm, dd, sub)
            os.makedirs(target_dir, exist_ok=True)
            sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
            sref_suffix = f"_sref{sref_code}" if sref_code else ""
            ckpt_abbrev = get_checkpoint_abbrev(checkpoint)

            if is_blend:
                filename = f"blend_{ckpt_abbrev}_{index}_seed{target_seed}{sref_suffix}_{time_str}.png"
            elif is_ico:
                filename = f"icon_{ckpt_abbrev}_{index}_seed{target_seed}{sref_suffix}_{time_str}.png"
            elif is_junji:
                filename = f"junji_{ckpt_abbrev}_{index}_seed{target_seed}{sref_suffix}_{time_str}.png"
            elif gen_data.get('is_flux') or "flux" in str(checkpoint).lower():
                filename = f"flux_{ckpt_abbrev}_{index}_seed{target_seed}{sref_suffix}_{time_str}.png"
            else:
                filename = f"imagine_{ckpt_abbrev}_{index}_seed{target_seed}{sref_suffix}_{time_str}.png"

            dest_path = os.path.join(target_dir, filename)
            with open(dest_path, "wb") as f:
                f.write(png_bytes)
            logger.info(f"Saved copy of isolated image {index} to {dest_path}")
        except Exception as save_err:
            logger.error(f"Failed to save isolated image copy to output directory: {save_err}")

        # Reset pointer for discord upload
        metadata_io.seek(0)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        if is_blend:
            iso_prefix = f"blend_{ckpt_abbrev}_{index}"
        elif is_ico:
            iso_prefix = f"icon_{ckpt_abbrev}_{index}"
        elif is_junji:
            iso_prefix = f"junji_{ckpt_abbrev}_{index}"
        elif gen_data.get('is_flux') or "flux" in str(checkpoint).lower():
            iso_prefix = f"flux_{ckpt_abbrev}_{index}"
        else:
            iso_prefix = f"imagine_{ckpt_abbrev}_{index}"
        file = discord.File(fp=metadata_io, filename=format_image_filename(iso_prefix, target_seed, "png", sref=sref_code))

        files_to_send = []
        rounded_corners = gen_data.get("rounded_corners", True)
        if is_ico and rounded_corners:
            iso_png_bytes = apply_rounded_corners_to_bytes(png_bytes)
            iso_png_file = discord.File(fp=io.BytesIO(iso_png_bytes), filename=format_image_filename(iso_prefix, target_seed, "png", sref=sref_code))
            files_to_send.append(iso_png_file)
        else:
            files_to_send.append(file)

        if is_ico:
            ico_bytes = create_windows_ico_bytes(png_bytes, rounded_corners=rounded_corners)
            if ico_bytes:
                ico_filename = format_image_filename(f"icon_{ckpt_abbrev}_{index}", target_seed, "ico", sref=sref_code)
                save_ico_file(ico_bytes, ico_filename)
                ico_file = discord.File(fp=io.BytesIO(ico_bytes), filename=ico_filename)
                files_to_send.append(ico_file)

        desc_lines = [
            f"**Prompt:** {truncate_prompt(original_prompt, 250)}",
            f"**Model:** {checkpoint}",
            f"**Seed:** {target_seed}",
            f"**Size:** {width}x{height}"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")

        if is_blend:
            title_txt = f"Blended Image {index}"
            content_txt = f"**Blended Image {index}:** {truncate_prompt(original_prompt, 100)}"
        elif gen_data.get("is_ico"):
            title_txt = f"Isolated Windows Icon {index}"
            content_txt = f"**Isolated Icon {index}:** {truncate_prompt(original_prompt, 100)}"
        else:
            title_txt = f"Isolated Image {index}"
            content_txt = f"**Isolate Image {index}:** {truncate_prompt(original_prompt, 100)}"

        embed = discord.Embed(
            title=title_txt,
            description="\n".join(desc_lines)
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
        
        has_sref = sref_info is not None and "code" in sref_info
        view = IsolatedImageButtons(generation_id, index, has_sref=has_sref)
        await send_followup_fallback(interaction, content=content_txt, embed=embed, files=files_to_send, view=view)
        
        # Mark grid button as completed (green + disabled)
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)
    except Exception as e:
        logger.error(f"Error isolating image: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
        await send_error_fallback(interaction, f"An error occurred while isolating the image: {e}")







async def handle_variation(interaction: discord.Interaction, generation_id: str, index: int, denoise_override: float = None, variation_type: str = None):
    """Generate a new 4-image grid as a TRUE img2img variation of the selected quadrant."""
    await safe_defer(interaction, thinking=True)

    # Immediately mark the button as pending (blue + disabled)
    clicked_custom_id = interaction.data.get("custom_id", "")
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    old_seed = gen_data["seed"]
    variation_depth = gen_data.get("variation_depth", 0) + 1
    new_base_seed = (old_seed + index - 1) + (variation_depth * 1000)

    prompt = gen_data.get("prompt", "")
    
    # Deterministically expand prompt for this specific quadrant using its original seed
    orig_quadrant_seed = old_seed + index - 1
    expanded_prompt = expand_dynamic_prompt(prompt, random.Random(orig_quadrant_seed))
    
    original_prompt = gen_data.get("original_prompt", prompt)
    neg_prompt = gen_data.get("neg_prompt", gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", 4.0)
    is_flux = gen_data.get("is_flux") or (checkpoint and "flux" in str(checkpoint).lower())
    loras = gen_data.get("loras")
    if not loras:
        _, loras = parse_loras(original_prompt or prompt, is_flux=is_flux)
    sref_image = gen_data.get("sref_image")
    sref_weight = gen_data.get("sref_weight", 0.6)

    # 1. Retrieve quadrant bytes from cache
    q_bytes = get_quadrant_bytes(generation_id, index)
    q_filename = None

    if q_bytes:
        try:
            upload_res = await comfy_client.upload_image(q_bytes, f"var_input_{generation_id}_{index}.png")
            q_filename = upload_res.get("name")
            logger.info(f"Uploaded quadrant {index} for img2img variation: {q_filename}")
        except Exception as e:
            logger.error(f"Failed to upload quadrant image to ComfyUI: {e}")

    # 2. Select workflow (Img2Img if quadrant image available, otherwise fallback to txt2img)
    if is_flux:
        lowres_wf = "workflows/flux_lowres.json"
    elif q_filename:
        lowres_wf = "workflows/img2img_sref_lowres.json" if sref_image else "workflows/img2img_lowres.json"
    else:
        lowres_wf = "workflows/txt2img_sref_lowres.json" if sref_image else "workflows/txt2img_lowres.json"

    try:
        with open(lowres_wf, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading low-res workflow for variation: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Failed to load generation workflow template.", ephemeral=True)
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    sref_info = gen_data.get("sref_info")
    sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""
    denoise_val = 0.55
    try:
        if is_flux:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = new_base_seed
                workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0
            workflow["6"]["inputs"]["text"] = expanded_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)

            if q_filename:
                workflow["30"] = {
                    "inputs": {"image": q_filename},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Input Image"}
                }
                workflow["31_flux_vae"] = {
                    "inputs": {
                        "pixels": ["30", 0],
                        "vae": ["3", 0]
                    },
                    "class_type": "VAEEncode",
                    "_meta": {"title": "Flux VAE Encode Input"}
                }
                if "11" in workflow:
                    workflow["11"]["inputs"]["latent_image"] = ["31_flux_vae", 0]
                    denoise_val = denoise_override if denoise_override is not None else 0.85
                    workflow["11"]["inputs"]["denoise"] = denoise_val
                    logger.info(f"FLUX variation ({variation_type or 'img2img'}) applied (denoise: {denoise_val:.2f})")
        else:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["3"]["inputs"]["seed"] = new_base_seed
            workflow["3"]["inputs"]["cfg"] = cfg
            workflow["6"]["inputs"]["text"] = expanded_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)

            # Img2Img specific configuration
            if q_filename and "30" in workflow:
                workflow["30"]["inputs"]["image"] = q_filename
                if denoise_override is not None:
                    denoise_val = denoise_override
                else:
                    current_mode = settings.get("variation_mode", "high")
                    denoise_val = 0.95 if current_mode == "very_high" else 0.55
                workflow["3"]["inputs"]["denoise"] = denoise_val
            elif "5" in workflow:
                workflow["5"]["inputs"]["width"] = width
                workflow["5"]["inputs"]["height"] = height

            if sref_image and "21" in workflow:
                workflow["21"]["inputs"]["image"] = sref_image
                if "23" in workflow:
                    workflow["23"]["inputs"]["weight"] = sref_weight
    except KeyError as e:
        logger.error(f"Invalid workflow structure for variation: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Workflow template has an invalid structure.", ephemeral=True)
        return

    cref_image = gen_data.get("cref_image")
    cref_weight = gen_data.get("cref_weight", 0.80)
    sref_info = gen_data.get("sref_info")
    if cref_image and not is_flux:
        workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=cref_weight, node_prefix="cref")

    # Store new generation in cache
    new_gen_id = str(random.randint(100000, 999999))
    active_generations[new_gen_id] = {
        "prompt": expanded_prompt,
        "original_prompt": expanded_prompt,
        "neg_prompt": neg_prompt,
        "negative_prompt": neg_prompt,
        "seed": new_base_seed,
        "width": width,
        "height": height,
        "loras": loras,
        "checkpoint": checkpoint,
        "cfg": cfg,
        "variation_depth": variation_depth,
        "sref_image": sref_image,
        "sref_weight": sref_weight,
        "sref_info": sref_info,
        "cref_image": cref_image,
        "cref_weight": cref_weight,
        "is_ico": gen_data.get("is_ico", False),
        "is_junji": gen_data.get("is_junji", False),
        "is_blend": gen_data.get("is_blend", False)
    }
    save_generations()

    try:
        status_msg = f"Generating visual variation of image {index} (Denoise: {denoise_val:.2f}, Seed: {new_base_seed})..." if q_filename else f"Generating variation of image {index} (Seed: {new_base_seed})..."
        await interaction.followup.send(status_msg, ephemeral=True)

        if "5" in workflow:
            workflow["5"]["inputs"]["batch_size"] = 1

        tasks = []
        for i in range(4):
            wf_copy = copy.deepcopy(workflow)
            wf_copy["3"]["inputs"]["seed"] = new_base_seed + i
            wf_copy["6"]["inputs"]["text"] = expanded_prompt
            tasks.append(comfy_client.generate(wf_copy))

        results = await asyncio.gather(*tasks)
        images = [r[0] for r in results]

        if len(images) < 4:
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
            await interaction.followup.send(f"Expected 4 images, got {len(images)}.", ephemeral=True)
            return

        # Cache the new quadrant images for future variations
        save_quadrant_images(new_gen_id, images)

        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        grid_file_io = await asyncio.to_thread(create_grid, images, expanded_prompt, neg_prompt, new_base_seed, width, height)
        file = discord.File(fp=grid_file_io, filename=format_image_filename("variation_grid", new_base_seed, "jpg", sref=sref_code))

        if q_filename:
            if variation_type:
                var_type = f"{variation_type} Variation"
            else:
                current_mode = settings.get("variation_mode", "high")
                mode_lbl = "Very High" if current_mode == "very_high" else "High"
                var_type = f"{mode_lbl} Variation"
        else:
            var_type = "Seed Variation"
        desc_lines = [
            f"**Prompt:** {truncate_prompt(original_prompt, 250)}",
            f"**Model:** {checkpoint}",
            f"**Seed:** {new_base_seed}",
            f"**Size:** {width}x{height}",
            f"**Denoise:** {denoise_val:.2f}"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image:
            desc_lines.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")

        embed = discord.Embed(
            title=f"{var_type} of Image {index} (Depth {variation_depth})",
            description="\n".join(desc_lines)
        )
        has_sref = sref_info is not None and "code" in sref_info
        view = GridButtons(new_gen_id, has_sref=has_sref)

        await send_followup_fallback(interaction, content=f"**{var_type} (Image {index}):** {truncate_prompt(original_prompt, 100)}", embed=embed, file=file, view=view)
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)
    except Exception as e:
        logger.error(f"Error generating variation: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
        await send_error_fallback(interaction, f"An error occurred during variation: {e}")

async def handle_reroll(interaction: discord.Interaction, generation_id: str):
    """Re-roll: same prompt/settings, brand new random seed."""
    await safe_defer(interaction, thinking=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    new_seed = random.randint(1, 1125899906842624)
    prompt = gen_data.get("prompt", "")
    original_prompt = gen_data.get("original_prompt", prompt)
    neg_prompt = gen_data.get("neg_prompt", gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", 4.0)
    sref_image = gen_data.get("sref_image")
    sref_weight = gen_data.get("sref_weight", 0.6)
    sref_info = gen_data.get("sref_info")
    cref_image = gen_data.get("cref_image")
    cref_weight = gen_data.get("cref_weight", 0.80)

    is_com = gen_data.get("is_com", False)
    is_sdxl_powerhouse = gen_data.get("is_sdxl_powerhouse", False)
    is_flux = is_com or gen_data.get("is_flux") or (checkpoint and "flux" in str(checkpoint).lower())

    loras = gen_data.get("loras")
    if not loras:
        _, loras = parse_loras(original_prompt or prompt, is_flux=is_flux)

    if is_com:
        lowres_wf = "workflows/com_flux_gguf.json"
    elif is_sdxl_powerhouse:
        lowres_wf = "workflows/sdxl_powerhouse_2stage.json"
    elif is_flux:
        lowres_wf = "workflows/flux_lowres.json"
    elif sref_image:
        lowres_wf = "workflows/txt2img_sref_lowres.json"
    else:
        lowres_wf = "workflows/txt2img_lowres.json"
    try:
        with open(lowres_wf, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow '{lowres_wf}' for reroll: {e}")
        await interaction.followup.send("Failed to load generation workflow template.", ephemeral=True)
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    if cref_image and not is_com:
        workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=cref_weight, node_prefix="cref")

    try:
        sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""
        if is_com:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = new_seed
                workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0
            if "13" in workflow:
                workflow["13"]["inputs"]["guidance"] = gen_data.get("guidance", 3.5)
            workflow["6"]["inputs"]["text"] = prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)
        elif is_sdxl_powerhouse:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            if "3" in workflow:
                workflow["3"]["inputs"]["seed"] = new_seed
                workflow["3"]["inputs"]["cfg"] = cfg
            if "15" in workflow:
                workflow["15"]["inputs"]["seed"] = new_seed
                workflow["15"]["inputs"]["cfg"] = cfg
            workflow["6"]["inputs"]["text"] = prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)
        elif is_flux:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = new_seed
                workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0
            workflow["6"]["inputs"]["text"] = prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)
        else:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            workflow["3"]["inputs"]["seed"] = new_seed
            workflow["3"]["inputs"]["cfg"] = cfg
            workflow["6"]["inputs"]["text"] = prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)
            if sref_image and "21" in workflow:
                workflow["21"]["inputs"]["image"] = sref_image
                if "23" in workflow:
                    workflow["23"]["inputs"]["weight"] = sref_weight

        if gen_data.get("is_face_detailer") and not is_flux:
            workflow = apply_face_detailer_to_workflow(workflow, seed=new_seed, cfg=cfg)
    except KeyError as e:
        logger.error(f"Invalid workflow structure for reroll: {e}")
        await interaction.followup.send("Workflow template has an invalid structure.", ephemeral=True)
        return

    new_gen_id = str(random.randint(100000, 999999))
    active_generations[new_gen_id] = {
        "prompt": prompt,
        "original_prompt": original_prompt,
        "neg_prompt": neg_prompt,
        "negative_prompt": neg_prompt,
        "seed": new_seed,
        "width": width,
        "height": height,
        "loras": loras,
        "checkpoint": checkpoint,
        "cfg": cfg,
        "variation_depth": 0,
        "sref_image": sref_image,
        "sref_weight": sref_weight,
        "sref_info": sref_info,
        "cref_image": cref_image,
        "cref_weight": cref_weight,
        "is_ico": gen_data.get("is_ico", False),
        "is_flux": is_flux,
        "is_com": is_com,
        "is_sdxl_powerhouse": is_sdxl_powerhouse,
        "guidance": gen_data.get("guidance", 3.5),
        "freeu": gen_data.get("freeu", True),
        "is_face_detailer": gen_data.get("is_face_detailer", False),
        "is_junji": gen_data.get("is_junji", False),
        "is_blend": gen_data.get("is_blend", False)
    }
    save_generations()

    try:
        await interaction.followup.send(f"Re-rolling with new seed {new_seed}...", ephemeral=True)

        workflow["5"]["inputs"]["batch_size"] = 1
        tasks = []
        expanded_prompts = []
        cleaned_prompt, is_magic = parse_magic_prompt(prompt)
        for i in range(4):
            wf_copy = copy.deepcopy(workflow)
            q_seed = new_seed + i
            q_rng = random.Random(q_seed)
            q_prompt = expand_dynamic_prompt(cleaned_prompt, q_rng)
            if is_magic:
                q_prompt = apply_magic_enhancement(q_prompt, q_seed)
            
            expanded_prompts.append(q_prompt)
            seed_node = "11" if is_flux else "3"
            if seed_node in wf_copy:
                wf_copy[seed_node]["inputs"]["seed"] = q_seed
            if is_sdxl_powerhouse and "15" in wf_copy:
                wf_copy["15"]["inputs"]["seed"] = q_seed
            if "85" in wf_copy:
                wf_copy["85"]["inputs"]["seed"] = q_seed
            wf_copy["6"]["inputs"]["text"] = q_prompt
            tasks.append(comfy_client.generate(wf_copy))

        results = await asyncio.gather(*tasks)
        images = [r[0] for r in results]

        if len(images) < 4:
            await interaction.followup.send(f"Expected 4 images, got {len(images)}.", ephemeral=True)
            return

        # Cache quadrant images for future visual variations
        save_quadrant_images(new_gen_id, images)

        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        grid_file_io = await asyncio.to_thread(create_grid, images, prompt, neg_prompt, new_seed, width, height)
        file = discord.File(fp=grid_file_io, filename=format_image_filename("reroll_grid", new_seed, "jpg", sref=sref_code))

        desc_parts = [f"**Prompt:** {truncate_prompt(original_prompt, 250)}", f"**Model:** {checkpoint}", f"**Seed:** {new_seed}", f"**Size:** {width}x{height}"]
        if sref_info and "code" in sref_info:
            desc_parts.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image:
            desc_parts.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")
        if "{" in original_prompt and "}" in original_prompt:
            desc_parts.append("\n**Selected Quadrant Prompts:**")
            cleaned_eps = clean_quadrant_prompts(expanded_prompts, original_prompt)
            for idx, clean_ep in enumerate(cleaned_eps):
                if len(clean_ep) > 120:
                    clean_ep = clean_ep[:117] + "..."
                desc_parts.append(f"* **Q{idx+1}:** {clean_ep}")

        reroll_title = "Re-rolled Windows 11 Icon Grid" if gen_data.get("is_ico") else "Re-rolled Generation"
        embed = discord.Embed(
            title=reroll_title,
            description="\n".join(desc_parts)
        )
        has_sref = sref_info is not None and "code" in sref_info
        view = GridButtons(new_gen_id, has_sref=has_sref)

        await send_followup_fallback(interaction, content=f"**Re-roll:** {truncate_prompt(original_prompt, 100)}", embed=embed, file=file, view=view)
    except Exception as e:
        logger.error(f"Error during reroll: {e}")
        await send_error_fallback(interaction, f"An error occurred during reroll: {e}")

async def handle_favorite_style(interaction: discord.Interaction, generation_id: str):
    """Saves the style reference from the generation to the user's favorites."""
    await safe_defer(interaction, ephemeral=True)
    
    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return
        
    sref_info = gen_data.get("sref_info")
    if not sref_info or "code" not in sref_info:
        await interaction.followup.send("No style reference code was found for this generation.", ephemeral=True)
        return
        
    code = sref_info["code"]
    name = sref_info.get("name", f"Style {code}")
    prompt = sref_info.get("prompt", "")
    
    # Save to user's favorites
    db.add_favorite_style(interaction.user.id, code, name, prompt)
    
    await interaction.followup.send(f"⭐ Saved style **{name}** (`{code}`) to your favorites!", ephemeral=True)

async def handle_favorite_prompt(interaction: discord.Interaction, generation_id: str):
    """Saves the prompt from the generation to the user's favorite prompts."""
    await safe_defer(interaction, ephemeral=True)
    
    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return
        
    prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    if not prompt:
        await interaction.followup.send("No prompt text was found for this generation.", ephemeral=True)
        return
        
    name_preview = truncate_prompt(prompt, 95)
    db.add_favorite_prompt(interaction.user.id, name_preview, prompt)
    
    await interaction.followup.send(f"⭐ Saved prompt **\"{name_preview}\"** to your favorite prompts!", ephemeral=True)
async def handle_outpaint(interaction: discord.Interaction, generation_id: str, index: int, target_ratio: str):
    """Outpaint an image to expanding aspect ratio (16:9, 21:9) or Zoom Out (1.5x, 2.0x)."""
    await safe_defer(interaction, thinking=True)

    # Immediately mark the button as pending (blue + disabled)
    clicked_custom_id = interaction.data.get("custom_id", "")
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id) or {}
    raw_prompt = gen_data.get("prompt", "")
    original_prompt = gen_data.get("original_prompt", raw_prompt)

    # Clean flags from prompt string for clean CLIP encoding
    cleaned_p, _ = parse_magic_prompt(raw_prompt)
    cleaned_p, _ = parse_seed(cleaned_p)
    cleaned_p, cfg_val, _ = parse_stylize(cleaned_p)
    cleaned_p, _, _, _ = parse_sref(cleaned_p)
    cleaned_p, _, _ = parse_aspect_ratio(cleaned_p, COMFYUI_CHECKPOINT)

    # Deterministically expand prompt for this specific quadrant using its original seed
    orig_quadrant_seed = gen_data.get("seed", 0) + index - 1
    expanded_p = expand_dynamic_prompt(cleaned_p, random.Random(orig_quadrant_seed))
    expanded_original = expand_dynamic_prompt(original_prompt, random.Random(orig_quadrant_seed))

    neg_prompt = gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", cfg_val if cfg_val is not None else 4.5)
    loras = gen_data.get("loras")
    if not loras:
        _, loras = parse_loras(original_prompt or prompt, is_flux=False)
    seed = random.randint(1, 1125899906842624)

    # 1. Obtain input image bytes (from quadrant cache or message attachment)
    q_bytes = get_quadrant_bytes(generation_id, index)
    if not q_bytes and interaction.message and interaction.message.attachments:
        try:
            att = interaction.message.attachments[0]
            q_bytes = await download_image(att.url)
        except Exception as e:
            logger.error(f"Error fetching attachment for outpaint: {e}")

    if not q_bytes:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not locate image data to outpaint.", ephemeral=True)
        return

    # 2. Calculate padding parameters and rescale base image to SDXL 1024 resolution if needed
    try:
        left, top, right, bottom, input_img_bytes, out_w, out_h = calculate_outpaint_padding(q_bytes, target_ratio)
    except Exception as e:
        logger.error(f"Error calculating outpaint padding: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send(f"Failed to calculate outpaint canvas: {e}", ephemeral=True)
        return

    # 3. Upload base image to ComfyUI
    try:
        up_res = await comfy_client.upload_image(input_img_bytes, f"outpaint_input_{generation_id}_{index}.png")
        img_filename = up_res.get("name")
    except Exception as e:
        logger.error(f"Error uploading image to ComfyUI: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send(f"Failed to upload image to ComfyUI: {e}", ephemeral=True)
        return

    # 4. Load outpaint workflow
    workflow_path = "workflows/outpaint_lowres.json"
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading outpaint workflow: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Failed to load outpaint workflow template.", ephemeral=True)
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    sref_info = gen_data.get("sref_info")
    sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""

    try:
        workflow["4"]["inputs"]["ckpt_name"] = checkpoint
        workflow["6"]["inputs"]["text"] = expanded_p
        workflow["7"]["inputs"]["text"] = neg_prompt
        workflow["30"]["inputs"]["image"] = img_filename
        
        # Configure ImagePadForOutpaint node (Node 31)
        workflow["31"]["inputs"]["left"] = left
        workflow["31"]["inputs"]["top"] = top
        workflow["31"]["inputs"]["right"] = right
        workflow["31"]["inputs"]["bottom"] = bottom
        workflow["31"]["inputs"]["feathering"] = 64
        
        # KSampler configuration (Node 3)
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["cfg"] = cfg
        workflow["3"]["inputs"]["denoise"] = 0.80
        workflow["9"]["class_type"] = "PreviewImage"
        workflow["9"]["inputs"].pop("filename_prefix", None)

        # Apply IP-Adapter style steering using the input image
        try:
            workflow = build_blend_workflow([img_filename], expanded_p, neg_prompt, checkpoint, out_w, out_h, seed, cfg, workflow_template=workflow)
        except Exception as ip_err:
            logger.error(f"Failed to chain IP-Adapter for outpaint: {ip_err}")

        cref_image = gen_data.get("cref_image")
        cref_weight = gen_data.get("cref_weight", 0.80)
        if cref_image:
            workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=cref_weight, node_prefix="cref")
    except KeyError as e:
        logger.error(f"Invalid outpaint workflow structure: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Outpaint workflow template has an invalid structure.", ephemeral=True)
        return

    await interaction.followup.send(f"Outpainting image {index} to {target_ratio} ({out_w}x{out_h}, Seed: {seed})...", ephemeral=True)

    try:
        images = await comfy_client.generate(workflow, timeout=14400)
        if not images:
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
            await interaction.followup.send("ComfyUI did not return an outpainted image.", ephemeral=True)
            return

        sref_info = gen_data.get("sref_info")
        new_gen_id = str(random.randint(100000, 999999))
        active_generations[new_gen_id] = {
            "prompt": expanded_p,
            "original_prompt": expanded_original,
            "negative_prompt": neg_prompt,
            "seed": seed,
            "width": out_w,
            "height": out_h,
            "loras": loras,
            "checkpoint": checkpoint,
            "cfg": cfg,
            "variation_depth": 0,
            "sref_info": sref_info,
            "cref_image": cref_image,
            "cref_weight": cref_weight
        }
        save_generations()

        # Cache outpainted image for future operations
        save_quadrant_images(new_gen_id, [images[0]])

        out_file_io = embed_metadata(images[0], expanded_original, neg_prompt, seed, out_w, out_h)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        file = discord.File(fp=out_file_io, filename=format_image_filename(f"outpaint_{target_ratio.replace(':', '_')}", seed, "png", sref=sref_code))

        desc_lines = [
            f"**Prompt:** {truncate_prompt(expanded_original, 250)}",
            f"**Model:** {checkpoint}",
            f"**Target Size:** {out_w}x{out_h}",
            f"**Seed:** {seed}"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image:
            desc_lines.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")

        embed = discord.Embed(
            title=f"Outpainted Canvas ({target_ratio})",
            description="\n".join(desc_lines)
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
        view = UpscaleButtons(new_gen_id, 1)

        await send_followup_fallback(interaction, content=f"**Outpaint ({target_ratio}):** {truncate_prompt(expanded_original, 100)}", embed=embed, file=file, view=view)
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)
    except Exception as e:
        logger.error(f"Error generating outpaint: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
        await send_error_fallback(interaction, f"An error occurred during outpainting: {e}")


async def handle_change_sref(interaction: discord.Interaction, generation_id: str, index: int, new_sref_str: str):
    """
    Re-renders an isolated image keeping the exact prompt, seed, checkpoint, size, and settings,
    but applying the new specified --sref.
    """
    await safe_defer(interaction, thinking=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    # 1. Retrieve isolated image base seed & settings
    base_seed = gen_data["seed"]
    target_seed = base_seed + index - 1
    
    raw_prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    
    # Strip any existing --sref from prompt string
    cleaned_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+[^\s]+(?:\s*\([^)]*\))?', '', raw_prompt, flags=re.IGNORECASE).strip()
    
    # Format new prompt with the new --sref
    if new_sref_str.lower().startswith("--sref"):
        new_full_prompt = f"{cleaned_prompt} {new_sref_str}"
    else:
        new_full_prompt = f"{cleaned_prompt} --sref {new_sref_str}"

    neg_prompt = gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    width = gen_data.get("width", 512)
    height = gen_data.get("height", 512)
    loras = gen_data.get("loras", [])
    cfg = gen_data.get("cfg", 4.0)

    # 2. Parse flags & sref from new_full_prompt
    parsed_p, magic_flag = parse_magic_prompt(new_full_prompt)
    is_magic = magic_flag
    
    parsed_p, _ = parse_seed(parsed_p)
    parsed_p, cfg_val, prepend_quality = parse_stylize(parsed_p)
    parsed_p, sref_url, sref_weight, sref_info = parse_sref(parsed_p)
    parsed_p, cref_url, cref_weight = parse_cref(parsed_p)
    parsed_p, w_parsed, h_parsed = parse_aspect_ratio(parsed_p, checkpoint)
    parsed_p, loras_parsed = parse_loras(parsed_p)
    
    if loras_parsed:
        loras = loras_parsed

    # Deterministically expand dynamic wildcards for target_seed
    expanded_p = expand_dynamic_prompt(parsed_p, random.Random(target_seed))
    if is_magic:
        expanded_p = apply_magic_enhancement(expanded_p, target_seed)

    if prepend_quality and not expanded_p.startswith("masterpiece"):
        expanded_p = f"masterpiece, best quality, absurdres. {expanded_p}"

    # Handle sref image if URL provided
    sref_image_name = None
    if sref_url:
        try:
            sref_bytes = await download_image(sref_url)
            upload_res = await comfy_client.upload_image(sref_bytes, "sref_from_url.png")
            sref_image_name = upload_res.get("name")
        except Exception as e:
            logger.error(f"Failed to fetch style reference URL: {e}")

    is_flux = gen_data.get("is_flux") or (checkpoint and "flux" in checkpoint.lower())

    # Load low-res workflow template
    if is_flux:
        wf_path = "workflows/flux_lowres.json"
    elif sref_image_name:
        wf_path = "workflows/txt2img_sref_lowres.json"
    else:
        wf_path = "workflows/txt2img_lowres.json"
    try:
        with open(wf_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow template for --sref change: {e}")
        await interaction.followup.send("Failed to load generation workflow template.", ephemeral=True)
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""
    try:
        if is_flux:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            workflow["5"]["inputs"]["batch_size"] = 1
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = target_seed
                workflow["11"]["inputs"]["cfg"] = cfg if cfg_val is None else cfg_val
            workflow["6"]["inputs"]["text"] = expanded_p
            workflow["7"]["inputs"]["text"] = neg_prompt
            seed_suffix = f"_seed{target_seed}"
            workflow["9"]["class_type"] = "SaveImage"
            workflow["9"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('flux')}Flux_SrefChange{seed_suffix}{sref_suffix}"
        else:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            workflow["5"]["inputs"]["batch_size"] = 1
            workflow["3"]["inputs"]["seed"] = target_seed
            workflow["3"]["inputs"]["cfg"] = cfg if cfg_val is None else cfg_val
            workflow["6"]["inputs"]["text"] = expanded_p
            workflow["7"]["inputs"]["text"] = neg_prompt
            seed_suffix = f"_seed{target_seed}"
            workflow["9"]["class_type"] = "SaveImage"
            workflow["9"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('imagine')}SrefChange{seed_suffix}{sref_suffix}"

            if sref_image_name and "21" in workflow:
                workflow["21"]["inputs"]["image"] = sref_image_name
                if "23" in workflow:
                    workflow["23"]["inputs"]["weight"] = sref_weight

        cref_image = gen_data.get("cref_image")
        if cref_image and not is_flux:
            workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=gen_data.get("cref_weight", 0.80), node_prefix="cref")
    except KeyError as e:
        logger.error(f"Invalid workflow structure for --sref change: {e}")
        await interaction.followup.send("Workflow template has an invalid structure.", ephemeral=True)
        return

    sref_lbl = f"--sref {sref_info['code']} ({sref_info['name']})" if sref_info and "code" in sref_info else new_sref_str

    try:
        images = await comfy_client.generate(workflow, timeout=14400)
        if not images:
            await interaction.followup.send("ComfyUI did not return an image.", ephemeral=True)
            return

        new_gen_id = str(random.randint(100000, 999999))
        active_generations[new_gen_id] = {
            "prompt": expanded_p,
            "original_prompt": new_full_prompt,
            "negative_prompt": neg_prompt,
            "seed": target_seed,
            "width": width,
            "height": height,
            "loras": loras,
            "checkpoint": checkpoint,
            "cfg": cfg,
            "variation_depth": 0,
            "sref_info": sref_info,
            "cref_image": gen_data.get("cref_image")
        }
        save_generations()

        # Save quadrant cache for new_gen_id at index 1 so variations / upscales work on it
        save_quadrant_images(new_gen_id, [images[0]])

        metadata_io = embed_metadata(images[0], new_full_prompt, neg_prompt, target_seed, width, height)
        file_code = str(sref_info['code']) if sref_info and "code" in sref_info else "custom"
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        file = discord.File(fp=metadata_io, filename=format_image_filename(f"isolated_sref_{file_code}", target_seed, "png", sref=sref_code))

        desc_lines = [
            f"**Prompt:** {truncate_prompt(new_full_prompt, 250)}",
            f"**Model:** {checkpoint}",
            f"**Seed:** {target_seed}",
            f"**Size:** {width}x{height}"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        elif sref_url:
            desc_lines.append(f"**Style Reference:** {sref_url}")

        embed = discord.Embed(
            title=f"Isolated Image (New --sref)",
            description="\n".join(desc_lines)
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")

        has_sref = (sref_info is not None and "code" in sref_info) or sref_url is not None
        view = IsolatedImageButtons(new_gen_id, 1, has_sref=has_sref)
        await send_followup_fallback(interaction, content=f"**New --sref Image:** {truncate_prompt(new_full_prompt, 100)}", embed=embed, file=file, view=view)
    except Exception as e:
        logger.error(f"Error changing --sref: {e}")
async def handle_copy_prompt(interaction: discord.Interaction, generation_id: str):
    """Sends an ephemeral message containing the 100% full, un-truncated prompt for easy viewing and copying."""
    try:
        gen_data = get_generation(generation_id)
        if not gen_data:
            if not interaction.response.is_done():
                await interaction.response.send_message("❌ Generation session data not found or expired.", ephemeral=True)
            else:
                await interaction.followup.send("❌ Generation session data not found or expired.", ephemeral=True)
            return

        prompt_text = gen_data.get("display_prompt") or gen_data.get("prompt") or gen_data.get("caption") or gen_data.get("detailed_caption")
        if not prompt_text:
            if not interaction.response.is_done():
                await interaction.response.send_message("❌ Prompt text not available.", ephemeral=True)
            else:
                await interaction.followup.send("❌ Prompt text not available.", ephemeral=True)
            return

        header = "📋 **Full Prompt:**\n"
        if len(header) + len(prompt_text) + 10 <= 1980:
            formatted = f"{header}```\n{prompt_text}\n```"
            if not interaction.response.is_done():
                await interaction.response.send_message(formatted, ephemeral=True)
            else:
                await interaction.followup.send(formatted, ephemeral=True)
        else:
            # For prompts exceeding Discord's 2,000-character limit, send an attached .txt file + truncated preview
            file = discord.File(io.BytesIO(prompt_text.encode('utf-8')), filename="full_prompt.txt")
            truncated_preview = prompt_text[:1800] + "..."
            msg_content = f"📋 **Full Prompt** *(Exceeds Discord 2,000-character limit - complete text attached as file)*:\n```\n{truncated_preview}\n```"
            if not interaction.response.is_done():
                await interaction.response.send_message(msg_content, file=file, ephemeral=True)
            else:
                await interaction.followup.send(msg_content, file=file, ephemeral=True)
    except Exception as e:
        logger.error(f"Error in handle_copy_prompt: {e}")
        try:
            if not interaction.response.is_done():
                await interaction.response.send_message(f"❌ Failed to copy prompt: {e}", ephemeral=True)
            else:
                await interaction.followup.send(f"❌ Failed to copy prompt: {e}", ephemeral=True)
        except Exception:
            pass


async def fetch_comfyui_queue():
    """Fetch current queue status from ComfyUI REST API."""
    try:
        url = f"http://{COMFYUI_ADDRESS}/queue"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None

async def fetch_comfyui_system_stats():
    """Fetch system stats from ComfyUI REST API."""
    try:
        url = f"http://{COMFYUI_ADDRESS}/system_stats"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None

@tasks.loop(seconds=10)
async def update_presence():
    """Update bot presence/status and console title with ComfyUI queue info."""
    try:
        queue = await fetch_comfyui_queue()
        if queue is None:
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name="ComfyUI (offline)"
            )
            await bot.change_presence(status=discord.Status.dnd, activity=activity)
            update_console_title("ComfyUI (offline)")
            return

        running = len(queue.get("queue_running", []))
        pending = len(queue.get("queue_pending", []))

        if running > 0:
            status_text = f"Processing {running} job{'s' if running != 1 else ''}"
            if pending > 0:
                status_text += f" | {pending} queued"
            activity = discord.Activity(
                type=discord.ActivityType.playing,
                name=status_text
            )
            await bot.change_presence(status=discord.Status.online, activity=activity)
            update_console_title(status_text)
        elif pending > 0:
            status_text = f"{pending} pending job{'s' if pending != 1 else ''}"
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name=status_text
            )
            await bot.change_presence(status=discord.Status.online, activity=activity)
            update_console_title(status_text)
        else:
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name="Ready ✓ | /imagine"
            )
            await bot.change_presence(status=discord.Status.online, activity=activity)
            update_console_title("Ready ✓")
    except Exception as e:
        logger.debug(f"Presence update failed: {e}")

@update_presence.before_loop
async def before_update_presence():
    await bot.wait_until_ready()

@bot.event
async def on_ready():
    # on_ready fires on EVERY reconnect, not just startup.
    # Only load from disk on first connect to avoid wiping in-memory data.
    if not hasattr(bot, '_initial_ready_done'):
        bot._initial_ready_done = True
        load_generations()
        load_settings()
        # Ensure standard ComfyUI output subfolders exist
        comfy_out = os.getenv("COMFYUI_OUTPUT_PATH", "C:/ComfyUI/ComfyUI/output")
        if os.path.exists(comfy_out):
            try:
                os.makedirs(os.path.join(comfy_out, "exp_data"), exist_ok=True)
            except Exception:
                pass

        await comfy_client.start()
        
        # ComfyUI status check
        try:
            if await comfy_client.is_online():
                logger.info(f"🟢 ComfyUI server is ONLINE at {COMFYUI_ADDRESS}.")
                
                # Run dependency check — report missing LoRAs, checkpoints, upscalers, etc.
                try:
                    dep_result = await comfy_client.check_dependencies()
                    if dep_result["ok"]:
                        logger.info("✅ All model dependencies are satisfied — no missing files detected.")
                    else:
                        missing = dep_result["missing"]
                        total_missing = sum(len(v) for v in missing.values())
                        logger.warning(f"⚠️  MISSING DEPENDENCIES: {total_missing} model file(s) not found in ComfyUI.")
                        
                        CATEGORY_LABELS = {
                            "checkpoints": "🔷 Checkpoints (models/checkpoints/)",
                            "loras":       "🔶 LoRAs (models/loras/)",
                            "vae":         "🟣 VAE Models (models/vae/)",
                            "unets":       "🟠 UNET / Diffusion Models (models/diffusion_models/ or models/unet/)",
                            "clip":        "🔵 CLIP / Text Encoders (models/clip/ or models/clip_vision/)",
                            "upscale_models": "🟢 Upscale Models (models/upscale_models/)",
                            "rife":        "🟡 RIFE Interpolation (custom_nodes/*/ckpts/)",
                        }
                        
                        for category, models in missing.items():
                            label = CATEGORY_LABELS.get(category, category)
                            logger.warning(f"  {label}")
                            for filename, sources in models.items():
                                src_str = ", ".join(sources)
                                logger.warning(f"    ❌ {filename}  (used by: {src_str})")
                        
                        logger.warning("──────────────────────────────────────────────────")
                        logger.warning("Download and place the missing files in the corresponding ComfyUI model folders, then restart ComfyUI.")
                except Exception as dep_err:
                    logger.debug(f"Dependency check skipped: {dep_err}")
            else:
                logger.info(f"ℹ️ ComfyUI server is currently OFFLINE at {COMFYUI_ADDRESS}. Use /cui-start in Discord to launch it.")
        except Exception:
            logger.info(f"ℹ️ ComfyUI server is currently OFFLINE at {COMFYUI_ADDRESS}. Use /cui-start in Discord to launch it.")
        try:
            # Register commands globally
            synced = await bot.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Error syncing tree: {e}")

        # Attempt to set global username to Shallot-CUI Bot if not already set
        try:
            if bot.user and bot.user.name != "Shallot-CUI Bot":
                await bot.user.edit(username="Shallot-CUI Bot")
                logger.info("Updated bot username to 'Shallot-CUI Bot'")
        except Exception as name_err:
            logger.debug(f"Username auto-update skipped: {name_err}")
    else:
        # Reconnect — just re-establish ComfyUI websocket
        logger.info("Discord reconnected (on_ready fired again). Re-establishing ComfyUI WebSocket.")
        try:
            await comfy_client.start()
        except Exception as e:
            logger.warning(f"Failed to restart ComfyUI client on reconnect: {e}")
    logger.info(f"Bot connected as {bot.user}")
    # Start the presence update loop
    if not update_presence.is_running():
        update_presence.start()

@bot.tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    if isinstance(error, app_commands.CommandOnCooldown):
        try:
            await interaction.response.send_message(
                f"⏳ Slow down! You are on cooldown. Try again in {error.retry_after:.1f}s.",
                ephemeral=True
            )
        except Exception:
            try:
                await interaction.followup.send(
                    f"⏳ Slow down! You are on cooldown. Try again in {error.retry_after:.1f}s.",
                    ephemeral=True
                )
            except Exception:
                pass
    else:
        logger.error(f"Unhandled tree error: {error}", exc_info=error)
        if not interaction.response.is_done():
            try:
                await interaction.response.send_message("❌ An unexpected error occurred.", ephemeral=True)
            except Exception:
                pass
        else:
            try:
                await interaction.followup.send("❌ An unexpected error occurred.", ephemeral=True)
            except Exception:
                pass


@bot.event
async def on_interaction(interaction: discord.Interaction):
    # Process button clicks for views on old messages
    if interaction.type == discord.InteractionType.component:
        custom_id = interaction.data.get("custom_id", "")
        if custom_id.startswith("stasis_pause:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    user_id = int(parts[2])
                    await handle_stasis_pause(interaction, generation_id, user_id)
                except ValueError:
                    pass
        elif custom_id.startswith("stasis_resume:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    user_id = int(parts[2])
                    await handle_stasis_resume(interaction, generation_id, user_id)
                except ValueError:
                    pass
        elif custom_id.startswith("upscale:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    await handle_isolate(interaction, generation_id, index)
                except ValueError:
                    pass
        elif custom_id.startswith("variation:"):
            # Format: variation:{gen_id}:{idx} or legacy variation:{strength}:{gen_id}:{idx}
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[-2]
                try:
                    index = int(parts[-1])
                    await handle_variation(interaction, generation_id, index)
                except ValueError:
                    pass
        elif custom_id.startswith("vary_subtle:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    await handle_variation(interaction, generation_id, index, denoise_override=0.70, variation_type="Subtle")
                except ValueError:
                    pass
        elif custom_id.startswith("vary_strong:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    await handle_variation(interaction, generation_id, index, denoise_override=0.95, variation_type="Strong")
                except ValueError:
                    pass
        elif custom_id.startswith("reroll:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                generation_id = parts[1]
                await handle_reroll(interaction, generation_id)
        elif custom_id.startswith("fav_style:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                generation_id = parts[1]
                await handle_favorite_style(interaction, generation_id)
        elif custom_id.startswith("fav_prompt:"):
            parts = custom_id.split(":")
            if len(parts) == 2:
                generation_id = parts[1]
                await handle_favorite_prompt(interaction, generation_id)
        elif custom_id.startswith("copy_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                generation_id = parts[1]
                await handle_copy_prompt(interaction, generation_id)
        elif custom_id.startswith("upscale_run:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    scale = parts[3]
                    await handle_upscale(interaction, generation_id, index, upscale_scale=scale)
                except ValueError:
                    pass
        elif custom_id.startswith("upscale_redo:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    scale = parts[3] if len(parts) >= 4 else "1.25"
                    await handle_upscale(interaction, generation_id, index, force_new_seed=True, upscale_scale=scale)
                except ValueError:
                    pass
        elif custom_id.startswith("outpaint:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    target_ratio = parts[3]
                    if len(parts) == 5:
                        target_ratio = f"{parts[3]}:{parts[4]}"
                    await handle_outpaint(interaction, generation_id, index, target_ratio)
                except ValueError:
                    pass
        elif custom_id.startswith("sref_change_random:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    new_code = str(random.randint(100000, 999999))
                    await handle_change_sref(interaction, generation_id, index, new_code)
                except ValueError:
                    pass
        elif custom_id.startswith("sref_change_saved:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    favorites = db.get_favorite_styles(interaction.user.id)
                    if not favorites:
                        await interaction.response.send_message("⭐ You don't have any saved favorite styles yet! Save some using the **Favorite Style** button on generations, or using `/my_prompts`.", ephemeral=True)
                    else:
                        view = SavedSrefSelectView(generation_id, index, favorites, select_callback=handle_change_sref)
                        await interaction.response.send_message("⭐ **Select a saved style to apply to this image (same prompt & seed):**", view=view, ephemeral=True)
                except ValueError:
                    pass
        elif custom_id.startswith("sref_change_custom:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                generation_id = parts[1]
                try:
                    index = int(parts[2])
                    await interaction.response.send_modal(CustomSrefModal(generation_id, index, on_submit_callback=handle_change_sref))
                except ValueError:
                    pass
        elif custom_id.startswith("gen_desc:"):
            parts = custom_id.split(":")
            # Format: gen_desc:{gen_id}:{desc_type}:{ar_x}:{ar_y}:{sr|nosr}:{oga|nooga}:{hyphoria|default}
            if len(parts) >= 3:
                generation_id = parts[1]
                desc_type = parts[2]
                if len(parts) == 8: # ["gen_desc", "id", "caption", "21", "9", "sr", "oga", "hyphoria"]
                    ar = f"{parts[3]}:{parts[4]}"
                    use_sr = parts[5]
                    use_oga = (parts[6] == "oga")
                    model_choice = parts[7]
                elif len(parts) == 7: # ["gen_desc", "id", "caption", "16:9", "sr", "oga", "hyphoria"]
                    ar = parts[3]
                    use_sr = parts[4]
                    use_oga = (parts[5] == "oga")
                    model_choice = parts[6]
                else:
                    ar = "16:9"
                    use_sr = "nosr"
                    use_oga = False
                    model_choice = "hyphoria"

                await safe_defer(interaction)
                await handle_generate_described(interaction, generation_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice)
        elif custom_id.startswith("set_desc_ar:") or custom_id.startswith("toggle_desc_sr:") or custom_id.startswith("toggle_desc_oga:") or custom_id.startswith("toggle_desc_model:"):
            parts = custom_id.split(":")
            if len(parts) >= 4:
                generation_id = parts[1]
                if len(parts) == 7: # e.g. ["set_desc_ar", "id", "21", "9", "sr", "oga", "hyphoria"]
                    new_ar = f"{parts[2]}:{parts[3]}"
                    new_sr = parts[4]
                    new_oga = (parts[5] == "oga")
                    new_model = parts[6]
                elif len(parts) == 6: # e.g. ["set_desc_ar", "id", "16:9", "sr", "oga", "hyphoria"]
                    new_ar = parts[2]
                    new_sr = parts[3]
                    new_oga = (parts[4] == "oga")
                    new_model = parts[5]
                else:
                    new_ar = parts[2]
                    new_sr = "nosr"
                    new_oga = False
                    new_model = "hyphoria"
                    new_model = "hyphoria"
                await handle_update_describe_view(interaction, generation_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model)
        elif custom_id.startswith("edit_blend_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id)
                if not gen_data:
                    await interaction.response.send_message("⚠️ Blend session data expired.", ephemeral=True)
                    return
                current_cap = gen_data.get("caption", "")
                current_det = gen_data.get("detailed_caption", "")
                current_extra = gen_data.get("extra_details", "")
                modal = EditBlendPromptModal(
                    generation_id=gen_id,
                    current_caption=current_cap,
                    current_detailed=current_det,
                    current_extra=current_extra,
                    on_submit_callback=handle_submit_edit_blend_prompts
                )
                await interaction.response.send_modal(modal)
        elif custom_id.startswith("set_blend_ar:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) == 2 and interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_ar=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("set_blend_model:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_model=val)
        elif custom_id.startswith("set_blend_comp:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) == 2 and interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_comp=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("toggle_blend_sr:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) == 3:
                val = parts[2]
                await handle_update_blend_view(interaction, gen_id, new_sr=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("toggle_blend_oga:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) == 3:
                val = (parts[2] == "oga")
                await handle_update_blend_view(interaction, gen_id, new_oga=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("toggle_blend_model:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("toggle_blend_sref:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if len(parts) == 3:
                val = parts[2]
                await handle_update_blend_view(interaction, gen_id, new_sref=val)
            elif len(parts) >= 8:
                new_ar = f"{parts[2]}:{parts[3]}" if len(parts) == 9 else parts[2]
                new_sr = parts[4] if len(parts) == 9 else parts[3]
                new_oga = (parts[5] == "oga") if len(parts) == 9 else (parts[4] == "oga")
                new_model = parts[6] if len(parts) == 9 else parts[5]
                new_comp = parts[7] if len(parts) == 9 else parts[6]
                new_sref = parts[8] if len(parts) == 9 else parts[7]
                await handle_update_blend_view(interaction, gen_id, new_ar=new_ar, new_sr=new_sr, new_oga=new_oga, new_model=new_model, new_comp=new_comp, new_sref=new_sref)
        elif custom_id.startswith("blend_desc:"):
            parts = custom_id.split(":")
            if len(parts) == 3:
                generation_id = parts[1]
                desc_type = parts[2]
                gen_data = get_generation(generation_id) or {}
                ar = gen_data.get("ar", "16:9")
                use_sr = gen_data.get("sr", True)
                use_oga = gen_data.get("oga", False)
                model_choice = gen_data.get("model_choice", "wai")
                comp_strength = gen_data.get("comp_strength", "style")
                use_sref = gen_data.get("sref_rand", "nosref")
                await safe_defer(interaction)
                await handle_generate_blended(interaction, generation_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice, comp_strength=comp_strength, use_sref_rand=use_sref)
            elif len(parts) >= 9:
                generation_id = parts[1]
                desc_type = parts[2]
                if len(parts) == 10:
                    ar = f"{parts[3]}:{parts[4]}"
                    use_sr = parts[5]
                    use_oga = (parts[6] == "oga")
                    model_choice = parts[7]
                    comp_strength = parts[8]
                    use_sref = parts[9]
                else:
                    ar = parts[3]
                    use_sr = parts[4]
                    use_oga = (parts[5] == "oga")
                    model_choice = parts[6]
                    comp_strength = parts[7]
                    use_sref = parts[8] if len(parts) > 8 else "nosref"
                await safe_defer(interaction)
                await handle_generate_blended(interaction, generation_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice, comp_strength=comp_strength, use_sref_rand=use_sref)
        elif custom_id.startswith("adopt_imagine:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data and "prompt" in data:
                    await safe_defer(interaction, thinking=True)
                    ref_url = data.get("image_url")
                    ref_weight = data.get("cref_weight", 0.20)
                    jump_url = data.get("jump_url")
                    prompt_str = data["prompt"]
                    if data.get("ogarla"):
                        if "ogarla" not in prompt_str.lower() and "oga" not in prompt_str.lower():
                            prompt_str = f"ogarla, {prompt_str} --ogarla.75"
                    if data.get("random_sref"):
                        if "--sref" not in prompt_str.lower():
                            prompt_str = f"{prompt_str} --sref random"
                    sr_w = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
                    sr_val = f"--sr {sr_w:.2f}" if sr_w > 0.0 else None
                    await execute_imagine(
                        interaction,
                        prompt=prompt_str,
                        semi_realism=sr_val,
                        reference_image_url=ref_url,
                        reference_image_weight=ref_weight,
                        original_post_url=jump_url
                    )
                else:
                    await interaction.response.send_message("⚠️ Adopted post data expired or not found.", ephemeral=True)
        elif custom_id.startswith("adopt_flux:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data and "prompt" in data:
                    await safe_defer(interaction, thinking=True)
                    ref_url = data.get("image_url")
                    ref_weight = data.get("cref_weight", 0.20)
                    jump_url = data.get("jump_url")
                    prompt_str = data["prompt"]
                    if data.get("ogarla"):
                        if "ogarla" not in prompt_str.lower() and "oga" not in prompt_str.lower():
                            prompt_str = f"ogarla, {prompt_str} --ogarla.75"
                    if data.get("random_sref"):
                        if "--sref" not in prompt_str.lower():
                            prompt_str = f"{prompt_str} --sref random"
                    sr_w = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
                    sr_val = f"--sr {sr_w:.2f}" if sr_w > 0.0 else None
                    await execute_imagine(
                        interaction,
                        prompt=prompt_str,
                        semi_realism=sr_val,
                        is_flux=True,
                        reference_image_url=ref_url,
                        reference_image_weight=ref_weight,
                        original_post_url=jump_url
                    )
                else:
                    await interaction.response.send_message("⚠️ Adopted post data expired or not found.", ephemeral=True)
        elif custom_id.startswith("adopt_toggle_oga:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    new_oga = not data.get("ogarla", False)
                    data["ogarla"] = new_oga
                    db.save_generation(adopt_id, data)
                    
                    cw = data.get("cref_weight", 0.20)
                    sr = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    rnd = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=new_oga, cref_weight=cw, semi_realism_weight=sr, random_sref_on=rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
        elif custom_id.startswith("adopt_toggle_sr:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    curr_sr = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
                    sr_weights = [0.00, 0.40, 0.60, 0.80, 1.00]
                    idx = 0
                    min_diff = 999
                    for i, w in enumerate(sr_weights):
                        if abs(w - curr_sr) < min_diff:
                            min_diff = abs(w - curr_sr)
                            idx = i
                    next_idx = (idx + 1) % len(sr_weights)
                    new_sr = sr_weights[next_idx]
                    data["semi_realism_weight"] = new_sr
                    data["semi_realism"] = (new_sr > 0.0)
                    db.save_generation(adopt_id, data)
                    
                    cw = data.get("cref_weight", 0.20)
                    oga = data.get("ogarla", False)
                    rnd = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga, cref_weight=cw, semi_realism_weight=new_sr, random_sref_on=rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
        elif custom_id.startswith("adopt_toggle_sref:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    new_rnd = not data.get("random_sref", False)
                    data["random_sref"] = new_rnd
                    db.save_generation(adopt_id, data)
                    
                    cw = data.get("cref_weight", 0.20)
                    oga = data.get("ogarla", False)
                    sr = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga, cref_weight=cw, semi_realism_weight=sr, random_sref_on=new_rnd)
                    await interaction.response.edit_message(view=view)
                else:
                    await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
        elif custom_id.startswith("adopt_cycle_cw:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data:
                    curr_cw = data.get("cref_weight", 0.20)
                    weights = [0.20, 0.40, 0.60, 0.80, 1.00]
                    idx = 0
                    min_diff = 999
                    for i, w in enumerate(weights):
                        if abs(w - curr_cw) < min_diff:
                            min_diff = abs(w - curr_cw)
                            idx = i
                    next_idx = (idx + 1) % len(weights)
                    new_cw = weights[next_idx]
                    data["cref_weight"] = new_cw
                    db.save_generation(adopt_id, data)

                    oga_on = data.get("ogarla", False)
                    sr_w = data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0)
                    rnd_on = data.get("random_sref", False)
                    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga_on, cref_weight=new_cw, semi_realism_weight=sr_w, random_sref_on=rnd_on)
                    await interaction.response.edit_message(view=view)
        elif custom_id.startswith("adopt_edit_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                if data and "prompt" in data:
                    modal = EditAdoptPromptModal(adopt_id, current_prompt=data["prompt"], on_submit_callback=handle_submit_edit_adopt_prompt)
                    await interaction.response.send_modal(modal)
                else:
                    await interaction.response.send_message("⚠️ Adopted post data expired or not found.", ephemeral=True)
        elif custom_id.startswith("adopt_copy:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                prompt_text = data.get("prompt", "") if data else ""
                if prompt_text:
                    await interaction.response.send_message(
                        content=f"📋 **Adopted Prompt:**\n```{prompt_text}```",
                        ephemeral=True
                    )
                else:
                    await interaction.response.send_message("⚠️ Prompt not found.", ephemeral=True)
        elif custom_id.startswith("adopt_save:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                adopt_id = parts[1]
                data = db.get_generation(adopt_id)
                prompt_text = data.get("prompt", "") if data else ""
                if prompt_text:
                    short_name = prompt_text[:30].strip() + ("..." if len(prompt_text) > 30 else "")
                    db.add_favorite_prompt(interaction.user.id, short_name, prompt_text)
                    await interaction.response.send_message(f"⭐ Saved prompt to your favorites (`/my_prompts`)!", ephemeral=True)
                else:
                    await interaction.response.send_message("⚠️ Prompt not found.", ephemeral=True)
        elif custom_id.startswith("lora_add:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                quadrant = int(parts[2])
                q_bytes = get_quadrant_bytes(gen_id, quadrant)
                if not q_bytes:
                    await interaction.response.send_message("❌ Could not retrieve quadrant image.", ephemeral=True)
                    return
                session = db.get_active_dataset_session(interaction.user.id)
                if not session:
                    await interaction.response.send_message("❌ No active LoRA dataset session found! Start one with `/lora-build start`.", ephemeral=True)
                    return
                session_id = session["session_id"]
                img_id, img_path = lora_dataset.save_image_to_dataset(session_id, q_bytes)
                all_imgs = db.get_dataset_images(session_id)
                await interaction.response.send_message(
                    f"✅ **Added Q{quadrant} to dataset session `{session['name']}`!**\n"
                    f"📁 **Image #{img_id}** saved (1024x1024 PNG).\n"
                    f"📊 Total images in session: **{len(all_imgs)}**\n"
                    f"💡 *Tip: Run `/lora-build status` or click `📊 Dataset Status` to view.*",
                    ephemeral=True
                )
        elif custom_id.startswith("lora_desc_add:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                quadrant = int(parts[2])
                q_bytes = get_quadrant_bytes(gen_id, quadrant)
                if not q_bytes:
                    await interaction.response.send_message("❌ Could not retrieve quadrant image.", ephemeral=True)
                    return
                session = db.get_active_dataset_session(interaction.user.id)
                if not session:
                    await interaction.response.send_message("❌ No active LoRA dataset session found! Start one with `/lora-build start`.", ephemeral=True)
                    return
                await interaction.response.defer(ephemeral=True)
                session_id = session["session_id"]
                img_id, img_path = lora_dataset.save_image_to_dataset(session_id, q_bytes)
                caption = await lora_dataset.auto_caption_dataset_image(comfy_client, img_path, session["trigger_word"])
                if caption:
                    db.update_image_caption(img_id, caption)
                    txt_path = os.path.splitext(img_path)[0] + ".txt"
                    with open(txt_path, "w", encoding="utf-8") as f:
                        f.write(caption)
                all_imgs = db.get_dataset_images(session_id)
                await interaction.followup.send(
                    f"🏷️ **Described & Added Q{quadrant} to `{session['name']}`!**\n"
                    f"📁 **Image #{img_id}** (Total: **{len(all_imgs)}** images)\n"
                    f"📝 **Caption:**\n```{caption or 'No caption generated'}```",
                    ephemeral=True
                )
        elif custom_id.startswith("lora_idea:"):
            parts = custom_id.split(":")
            session_id = parts[1] if len(parts) >= 2 else None
            session = db.get_dataset_session(session_id) if session_id else db.get_active_dataset_session(interaction.user.id)
            tw = session.get("trigger_word", "character") if session else "character"
            random_shot = random.choice(lora_dataset.LORA_SHOT_MATRIX)
            shot_name = random_shot["name"]
            shot_prompt = random_shot["template"].format(trigger_word=tw)
            await interaction.response.send_message(
                f"💡 **Suggested Next Shot: {shot_name}**\n"
                f"```{shot_prompt}```\n"
                f"Copy this into `/lora-build generate prompt:` or `/imagine`!",
                ephemeral=True
            )
        elif custom_id.startswith("lora_status:"):
            parts = custom_id.split(":")
            session_id = parts[1] if len(parts) >= 2 else None
            session = db.get_dataset_session(session_id) if session_id else db.get_active_dataset_session(interaction.user.id)
            if not session:
                await interaction.response.send_message("❌ No active dataset session found. Start one with `/lora-build start`!", ephemeral=True)
                return
            images = db.get_dataset_images(session["session_id"])
            captioned_count = sum(1 for img in images if img.get("caption"))
            embed = discord.Embed(
                title=f"🎨 LoRA Dataset Session: {session['name']}",
                description=(
                    f"**Session ID:** `{session['session_id']}`\n"
                    f"**Trigger Word:** `{session['trigger_word']}`\n"
                    f"**Total Images:** `{len(images)}` (Recommended: 15–30)\n"
                    f"**Captions Generated:** `{captioned_count} / {len(images)}`\n"
                    f"**Target Size:** `1024x1024` (SDXL Native)"
                ),
                color=discord.Color.blue()
            )
            view = LoraBuildStatusView(session["session_id"])
            await interaction.response.send_message(embed=embed, view=view, ephemeral=True)
        elif custom_id.startswith("lora_suggest:"):
            parts = custom_id.split(":")
            session_id = parts[1] if len(parts) >= 2 else None
            session = db.get_dataset_session(session_id) if session_id else db.get_active_dataset_session(interaction.user.id)
            if not session:
                await interaction.response.send_message("❌ No active dataset session found.", ephemeral=True)
                return
            suggestions = lora_dataset.generate_suggested_prompts(session["session_id"], session["trigger_word"], count=5)
            lines = []
            for idx, s in enumerate(suggestions, 1):
                lines.append(f"**Option {idx}:**\n```{s}```")
            embed = discord.Embed(
                title=f"💡 LoRA Shot Suggestions for `{session['name']}`",
                description="\n".join(lines),
                color=discord.Color.gold()
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
        elif custom_id.startswith("lora_describe_all:"):
            parts = custom_id.split(":")
            session_id = parts[1] if len(parts) >= 2 else None
            session = db.get_dataset_session(session_id) if session_id else db.get_active_dataset_session(interaction.user.id)
            if not session:
                await interaction.response.send_message("❌ No active dataset session found.", ephemeral=True)
                return
            images = db.get_dataset_images(session["session_id"])
            if not images:
                await interaction.response.send_message("❌ No images in this dataset session to describe.", ephemeral=True)
                return
            await interaction.response.defer(ephemeral=False)
            status_msg = await interaction.followup.send(f"⏳ Running Florence-2 auto-captioning on **{len(images)}** images in session `{session['name']}`... Please wait.")
            
            async def progress_cb(current, total):
                try:
                    await status_msg.edit(content=f"⏳ Running Florence-2: Captioning image **{current}/{total}**...")
                except Exception:
                    pass

            stats = await lora_dataset.batch_caption_session(comfy_client, session["session_id"], session["trigger_word"], progress_callback=progress_cb)
            await status_msg.edit(content=f"✅ **Florence-2 Captioning Complete!**\nDescribed **{stats['processed']}** images (Failed: {stats['failed']}). Trigger word `{session['trigger_word']}` injected into all `.txt` caption files.")
        elif custom_id.startswith("lora_export:"):
            parts = custom_id.split(":")
            session_id = parts[1] if len(parts) >= 2 else None
            session = db.get_dataset_session(session_id) if session_id else db.get_active_dataset_session(interaction.user.id)
            if not session:
                await interaction.response.send_message("❌ No active dataset session found.", ephemeral=True)
                return
            images = db.get_dataset_images(session["session_id"])
            if not images:
                await interaction.response.send_message("❌ No images in this dataset to export.", ephemeral=True)
                return
            await interaction.response.defer(ephemeral=False)
            try:
                zip_path, img_count = lora_dataset.export_dataset_zip(session["session_id"])
                file = discord.File(zip_path, filename=os.path.basename(zip_path))
                await interaction.followup.send(
                    content=(
                        f"📦 **SDXL Character LoRA Dataset Exported!**\n"
                        f"👤 **Character:** `{session['name']}`\n"
                        f"🏷️ **Trigger Word:** `{session['trigger_word']}`\n"
                        f"🖼️ **Images Included:** `{img_count}` (1024x1024 PNG + TXT captions)\n"
                        f"📁 **Folder Format:** `10_{session['trigger_word'].replace(' ', '_')}`\n\n"
                        f"✨ Ready to train directly in Kohya_ss, OneTrainer, or Civitai!"
                    ),
                    file=file
                )
            except Exception as exp_err:
                logger.error(f"Error exporting dataset: {exp_err}")
                await interaction.followup.send(f"❌ Failed to export dataset ZIP: {exp_err}")


@bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent):
    if payload.user_id == bot.user.id:
        return
        
    if str(payload.emoji) in ["❌", "x", "X"]:
        try:
            channel = bot.get_channel(payload.channel_id)
            if channel is None:
                channel = await bot.fetch_channel(payload.channel_id)
            message = await channel.fetch_message(payload.message_id)
            
            if message.author.id == bot.user.id:
                allowed_ids = set()

                # 1. Check mentioned users in message content
                for m in message.mentions:
                    allowed_ids.add(m.id)

                # 2. Check user tags/mentions in message content (e.g. <@123456789>)
                if message.content:
                    for uid in re.findall(r'<@!?(\d+)>', message.content):
                        allowed_ids.add(int(uid))

                # 3. Check interaction author
                meta = getattr(message, 'interaction_metadata', None)
                if meta and hasattr(meta, 'user') and meta.user:
                    allowed_ids.add(meta.user.id)
                elif hasattr(message, 'interaction') and message.interaction and hasattr(message.interaction, 'user') and message.interaction.user:
                    allowed_ids.add(message.interaction.user.id)

                # 4. Check embed footer ID
                if message.embeds:
                    for emb in message.embeds:
                        if emb.footer and emb.footer.text:
                            for match in re.finditer(r'(?:ID:\s*|id:\s*|user:\s*)(\d+)', emb.footer.text, flags=re.IGNORECASE):
                                allowed_ids.add(int(match.group(1)))

                # 5. Check permissions for moderator/admin override
                is_moderator = False
                if payload.guild_id:
                    guild = bot.get_guild(payload.guild_id) or await bot.fetch_guild(payload.guild_id)
                    member = guild.get_member(payload.user_id) if guild else None
                    if member is None and guild:
                        try:
                            member = await guild.fetch_member(payload.user_id)
                        except Exception:
                            pass
                    if member and channel:
                        permissions = channel.permissions_for(member)
                        is_moderator = permissions.manage_messages or permissions.administrator

                if payload.user_id in allowed_ids or is_moderator:
                    await message.delete()
                    logger.info(f"Deleted message {message.id} after ❌ reaction from user {payload.user_id}")
        except Exception as e:
            logger.error(f"Error in reaction delete handler: {e}")
async def on_close():
    await comfy_client.stop()

async def execute_imagine(interaction: discord.Interaction, prompt: str, negative_prompt: str = None, checkpoint: str = None, style_reference: discord.Attachment = None, magic_prompt: bool = False, favorite_style: str = None, semi_realism: str = None, aspect_ratio: str = None, ogarla: str = None, is_ico: bool = False, rounded_corners: bool = True, is_flux: bool = False, is_com: bool = False, is_sdxl_powerhouse: bool = False, guidance: float = 3.5, freeu: bool = True, smart: bool = False, enhancements: str = None, reference_image_url: str = None, reference_image_weight: float = None, original_post_url: str = None, is_lora_build: bool = False, lora_session_id: str = None, cref_image_name_override: str = None, is_face_detailer: bool = False, is_junji: bool = False):
    if enhancements:
        enh_val = str(enhancements).lower()
        if "smart" in enh_val or enh_val == "all":
            smart = True
        if "magic" in enh_val or enh_val == "all":
            magic_prompt = True
        if "no_freeu" in enh_val or "disable_freeu" in enh_val:
            freeu = False
        if "square" in enh_val or "no_curved" in enh_val:
            rounded_corners = False

    if semi_realism:
        prompt = re.sub(r'\bSemi-realism(?:,\s*masterpiece,\s*best quality,\s*absurdres\.?)?,?\s*', '', prompt, flags=re.IGNORECASE).strip()
    if ogarla:
        prompt = re.sub(r'\bogarla,?\s*', '', prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r'^[,\s]+', '', prompt).strip()

    prefixes = []
    if semi_realism:
        prefixes.append("Semi-realism, masterpiece, best quality, absurdres.")
    if ogarla:
        prefixes.append("ogarla")

    if prefixes:
        prefix_str = " ".join(p if p.endswith(".") else f"{p}," for p in prefixes)
        prompt = f"{prefix_str} {prompt}"

    if semi_realism:
        sr_val = semi_realism if semi_realism.startswith("--") else f"--{semi_realism}"
        prompt = f"{prompt} {sr_val}"

    if ogarla:
        oga_val = ogarla if ogarla.startswith("--") else f"--{ogarla}"
        prompt = f"{prompt} {oga_val}"

    if aspect_ratio:
        prompt = f"{prompt} --ar {aspect_ratio}"

    if favorite_style:
        fav_str = str(favorite_style).strip()
        if "batch" in fav_str.lower():
            prompt = f"{prompt} --sref {fav_str}"
        elif "random" in fav_str.lower():
            prompt = f"{prompt} --sref random"
        else:
            prompt = f"{prompt} --sref {fav_str}"

    neg_prompt = negative_prompt or db.get_negative_prompt(interaction.user.id if interaction and interaction.user else 0)
    if is_com:
        is_flux = True
        if checkpoint and ("safetensors" in str(checkpoint).lower() and "flux" not in str(checkpoint).lower()):
            selected_model = "flux1-dev-Q4_K_S.gguf"
        else:
            selected_model = checkpoint or "flux1-dev-Q4_K_S.gguf"
    elif is_flux or (checkpoint and "flux" in str(checkpoint).lower()):
        is_flux = True
        selected_model = checkpoint if (checkpoint and "flux" in str(checkpoint).lower()) else "flux1-dev-Q4_K_S.gguf"
    else:
        if checkpoint and ("gguf" in str(checkpoint).lower() or "ltx" in str(checkpoint).lower() or "wan" in str(checkpoint).lower()):
            selected_model = COMFYUI_CHECKPOINT
        else:
            selected_model = checkpoint or COMFYUI_CHECKPOINT

    # --- Parse all flags from prompt string ---
    cleaned_prompt, smart_flag = parse_smart_prompt(prompt)
    is_smart = smart_flag or (smart is True)

    cleaned_prompt, magic_flag = parse_magic_prompt(cleaned_prompt)
    is_magic = magic_flag or (magic_prompt is True)

    cleaned_prompt, user_seed = parse_seed(cleaned_prompt)
    seed = user_seed if user_seed is not None else random.randint(1, 1125899906842624)
    
    cleaned_prompt, cfg, prepend_quality = parse_stylize(cleaned_prompt)

    if is_flux:
        cleaned_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+[^\s]+(?:\s*\([^)]*\))?', '', cleaned_prompt, flags=re.IGNORECASE).strip()
        sref_url = None
        sref_weight = 1.0
        sref_info = None
    else:
        cleaned_prompt, sref_url, sref_weight, sref_info = parse_sref(cleaned_prompt)

    if is_smart:
        smart_expanded, rec_sref = apply_smart_magic_and_sref(cleaned_prompt, is_flux=is_flux)
        cleaned_prompt = smart_expanded
        if rec_sref and not sref_info and not sref_url and not favorite_style and not is_flux:
            _, sref_url, sref_weight, sref_info = parse_sref(f"--sref {rec_sref}")

    cleaned_prompt, cref_url, cref_weight = parse_cref(cleaned_prompt)
    if reference_image_weight is not None:
        cref_weight = reference_image_weight
    cleaned_prompt, width, height = parse_aspect_ratio(cleaned_prompt, selected_model)
    cleaned_prompt, loras = parse_loras(cleaned_prompt, is_flux=is_flux)

    # Check if a batch of styles (5, 10, 15) is requested
    batch_count = 1
    if sref_info and sref_info.get("batch_count", 1) > 1:
        batch_count = sref_info["batch_count"]
    elif favorite_style and "batch" in str(favorite_style).lower():
        b_match = re.search(r'batch[:\s]*(\d+)', str(favorite_style), flags=re.IGNORECASE)
        if b_match:
            batch_count = int(b_match.group(1))

    if batch_count > 1:
        favorites = db.get_favorite_styles(interaction.user.id)
        style_codes = []
        fav_count = 0
        if favorites:
            sample_size = min(len(favorites), batch_count)
            chosen_favs = random.sample(favorites, k=sample_size)
            for fav in chosen_favs:
                style_codes.append(str(fav["style_code"]))
                fav_count += 1
        needed_random = batch_count - len(style_codes)
        for _ in range(needed_random):
            style_codes.append(str(random.randint(100000, 999999)))

        await interaction.followup.send(
            f"🚀 **Batch Queuing {batch_count} Style Generations** ({fav_count} sampled from your {len(favorites)} saved `/styles` + {needed_random} random style codes)..."
        )

        base_p = re.sub(r'[-\u2014\u2013]{1,2}sref\s+[^\s]+', '', prompt, flags=re.IGNORECASE).strip()
        for code in style_codes:
            sub_prompt = f"{base_p} --sref {code}"
            await execute_imagine(interaction, sub_prompt, negative_prompt, checkpoint, style_reference, magic_prompt, favorite_style=None, semi_realism=None, aspect_ratio=None, ogarla=None)
        return
    
    if prepend_quality:
        cleaned_prompt = f"masterpiece, best quality, absurdres. {cleaned_prompt}"
    
    # --- Handle style reference (attachment or URL) ---
    sref_image_name = None
    if style_reference and style_reference.content_type and style_reference.content_type.startswith("image/"):
        try:
            sref_bytes = await style_reference.read()
            upload_result = await comfy_client.upload_image(sref_bytes, f"sref_{style_reference.filename}")
            sref_image_name = upload_result.get("name")
            logger.info(f"Style reference uploaded from attachment: {sref_image_name}")
        except Exception as e:
            logger.error(f"Failed to upload style reference attachment: {e}")
            await interaction.followup.send(f"Failed to upload style reference image: {e}")
            return
    elif sref_url:
        try:
            sref_bytes = await download_image(sref_url)
            upload_result = await comfy_client.upload_image(sref_bytes, "sref_from_url.png")
            sref_image_name = upload_result.get("name")
            logger.info(f"Style reference uploaded from URL: {sref_image_name}")
        except Exception as e:
            logger.error(f"Failed to download/upload style reference from URL: {e}")
            await interaction.followup.send(f"Failed to fetch style reference image: {e}")
            return

    # --- Handle character reference (--cref URL or reference_image_url from adopted post) ---
    cref_image_name = cref_image_name_override
    target_cref_url = reference_image_url or cref_url
    if not cref_image_name and target_cref_url:
        try:
            cref_bytes = await download_image(target_cref_url)
            if not cref_bytes or len(cref_bytes) < 100:
                raise ValueError("Downloaded image payload is empty or blank.")
            upload_result = await comfy_client.upload_image(cref_bytes, "cref_from_url.png")
            cref_image_name = upload_result.get("name")
            logger.info(f"Character reference uploaded from URL: {cref_image_name}")
        except Exception as e:
            if reference_image_url and cref_url and reference_image_url != cref_url:
                try:
                    fallback_bytes = await download_image(cref_url)
                    if fallback_bytes and len(fallback_bytes) >= 100:
                        upload_result = await comfy_client.upload_image(fallback_bytes, "cref_from_url.png")
                        cref_image_name = upload_result.get("name")
                except Exception:
                    cref_image_name = None

            if not cref_image_name:
                logger.warning(f"Failed to fetch character reference URL '{target_cref_url}': {e}. Falling back to Ogarla LoRA (--ogarla.75)!")
                cref_image_name = None
                if "ogarla" not in cleaned_prompt.lower() and "oga" not in cleaned_prompt.lower():
                    cleaned_prompt = f"ogarla, {cleaned_prompt} --ogarla.75"
    
    cleaned_prompt, loras = parse_loras(cleaned_prompt, is_flux=is_flux)

    use_reference_img2img = bool(reference_image_url and cref_image_name and not is_flux)

    # --- Load the appropriate workflow ---
    if is_com:
        workflow_path = "workflows/com_flux_gguf.json"
    elif is_sdxl_powerhouse:
        workflow_path = "workflows/sdxl_powerhouse_2stage.json"
    elif is_flux:
        workflow_path = "workflows/flux_lowres.json"
    elif use_reference_img2img:
        workflow_path = "workflows/img2img_sref_lowres.json" if sref_image_name else "workflows/img2img_lowres.json"
    elif sref_image_name:
        workflow_path = "workflows/txt2img_sref_lowres.json"
    else:
        workflow_path = "workflows/txt2img_lowres.json"
    
    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow '{workflow_path}': {e}")
        await interaction.followup.send("Failed to load generation workflow template.")
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""
    try:
        if is_com:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = selected_model
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            is_schnell = "schnell" in str(selected_model).lower()
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = seed
                workflow["11"]["inputs"]["steps"] = 4 if is_schnell else 16
                workflow["11"]["inputs"]["cfg"] = 1.0 if is_schnell else (cfg if cfg != 4.0 else 1.0)
            if "13" in workflow:
                workflow["13"]["inputs"]["guidance"] = 1.0 if is_schnell else (guidance if guidance is not None else 3.5)
            workflow["6"]["inputs"]["text"] = cleaned_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            if "9" in workflow:
                workflow["9"]["class_type"] = "PreviewImage"
                workflow["9"]["inputs"].pop("filename_prefix", None)
        elif is_sdxl_powerhouse:
            workflow["4"]["inputs"]["ckpt_name"] = selected_model
            if "5" in workflow:
                workflow["5"]["inputs"]["width"] = width
                workflow["5"]["inputs"]["height"] = height

            ckpt_cfg = CHECKPOINT_CONFIGS.get(selected_model, {})
            stage1_steps = 25
            stage1_cfg = cfg if cfg != 4.0 else 5.0
            stage1_sampler = "dpmpp_2m_sde"
            stage1_scheduler = "karras"
            if ckpt_cfg:
                if "cfg" in ckpt_cfg:
                    stage1_cfg = ckpt_cfg["cfg"]
                if "sampler_name" in ckpt_cfg:
                    stage1_sampler = ckpt_cfg["sampler_name"]
                if "scheduler" in ckpt_cfg:
                    stage1_scheduler = ckpt_cfg["scheduler"]
                if "steps" in ckpt_cfg:
                    stage1_steps = ckpt_cfg["steps"]
                if ckpt_cfg.get("negative_addon"):
                    neg_prompt = f"{neg_prompt}, {ckpt_cfg['negative_addon']}"

            # FreeU bypass if disabled
            if not freeu and "20" in workflow:
                fallback_model_src = ["76", 0] if "76" in workflow else ["4", 0]
                if "3" in workflow:
                    workflow["3"]["inputs"]["model"] = fallback_model_src
                if "15" in workflow:
                    workflow["15"]["inputs"]["model"] = fallback_model_src

            # Stage 1 KSampler
            if "3" in workflow:
                workflow["3"]["inputs"]["seed"] = seed
                workflow["3"]["inputs"]["steps"] = stage1_steps
                workflow["3"]["inputs"]["cfg"] = stage1_cfg
                workflow["3"]["inputs"]["sampler_name"] = stage1_sampler
                workflow["3"]["inputs"]["scheduler"] = stage1_scheduler

            # Stage 2 KSampler (Refiner)
            if "15" in workflow:
                workflow["15"]["inputs"]["seed"] = seed
                workflow["15"]["inputs"]["steps"] = max(10, int(stage1_steps * 0.6))
                workflow["15"]["inputs"]["cfg"] = stage1_cfg
                workflow["15"]["inputs"]["sampler_name"] = stage1_sampler
                workflow["15"]["inputs"]["scheduler"] = stage1_scheduler
                workflow["15"]["inputs"]["denoise"] = 0.48

            workflow["6"]["inputs"]["text"] = cleaned_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            if "9" in workflow:
                workflow["9"]["class_type"] = "PreviewImage"
                workflow["9"]["inputs"].pop("filename_prefix", None)
        elif is_flux:
            if "1" in workflow:
                workflow["1"]["inputs"]["unet_name"] = selected_model
            workflow["5"]["inputs"]["width"] = width
            workflow["5"]["inputs"]["height"] = height
            if "11" in workflow:
                workflow["11"]["inputs"]["seed"] = seed
                workflow["11"]["inputs"]["steps"] = 12
                workflow["11"]["inputs"]["cfg"] = cfg if cfg != 4.0 else 1.0
            workflow["6"]["inputs"]["text"] = cleaned_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)

            if use_reference_img2img:
                workflow["30"] = {
                    "inputs": {"image": cref_image_name},
                    "class_type": "LoadImage",
                    "_meta": {"title": "Load Reference Image"}
                }
                workflow["32_up_model"] = {
                    "inputs": {"model_name": "2x-ESRGAN.pth"},
                    "class_type": "UpscaleModelLoader",
                    "_meta": {"title": "Pre-Upscale Model Loader"}
                }
                workflow["33_up_img"] = {
                    "inputs": {
                        "upscale_model": ["32_up_model", 0],
                        "image": ["30", 0]
                    },
                    "class_type": "ImageUpscaleWithModel",
                    "_meta": {"title": "Pre-Upscale Reference Image"}
                }
                workflow["34_resize"] = {
                    "inputs": {
                        "image": ["33_up_img", 0],
                        "upscale_method": "lanczos",
                        "width": width,
                        "height": height,
                        "crop": "disabled"
                    },
                    "class_type": "ImageScale",
                    "_meta": {"title": "Resize Upscaled Reference"}
                }
                workflow["31_flux_vae"] = {
                    "inputs": {
                        "pixels": ["34_resize", 0],
                        "vae": ["3", 0]
                    },
                    "class_type": "VAEEncode",
                    "_meta": {"title": "Flux VAE Encode Reference"}
                }
                if "11" in workflow:
                    workflow["11"]["inputs"]["latent_image"] = ["31_flux_vae", 0]
                    denoise_val = max(0.50, min(0.80, 0.82 - ((cref_weight if cref_weight is not None else 0.20) * 0.30)))
                    workflow["11"]["inputs"]["denoise"] = denoise_val
                    logger.info(f"FLUX adopted image reference applied via img2img with AI Pre-Upscale (denoise: {denoise_val:.2f}, cref_weight: {cref_weight:.2f})")
        else:
            workflow["4"]["inputs"]["ckpt_name"] = selected_model
            if "5" in workflow:
                workflow["5"]["inputs"]["width"] = width
                workflow["5"]["inputs"]["height"] = height
            
            # Check for per-checkpoint custom configurations (samplers, CFG, steps, negative addons)
            ckpt_cfg = CHECKPOINT_CONFIGS.get(selected_model, {})
            if ckpt_cfg:
                if cfg == 4.0 and "cfg" in ckpt_cfg:
                    cfg = ckpt_cfg["cfg"]
                if "3" in workflow:
                    if "sampler_name" in ckpt_cfg:
                        workflow["3"]["inputs"]["sampler_name"] = ckpt_cfg["sampler_name"]
                    if "scheduler" in ckpt_cfg:
                        workflow["3"]["inputs"]["scheduler"] = ckpt_cfg["scheduler"]
                    if "steps" in ckpt_cfg:
                        workflow["3"]["inputs"]["steps"] = ckpt_cfg["steps"]
                if ckpt_cfg.get("negative_addon"):
                    neg_prompt = f"{neg_prompt}, {ckpt_cfg['negative_addon']}"

            workflow["3"]["inputs"]["seed"] = seed
            workflow["3"]["inputs"]["cfg"] = cfg
            workflow["6"]["inputs"]["text"] = cleaned_prompt
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)
            
            if use_reference_img2img:
                if "30" in workflow:
                    workflow["30"]["inputs"]["image"] = cref_image_name

                    # Pre-Upscale reference image using 2x-ESRGAN.pth for lightweight, fast & sharp upscaling
                    workflow["32_up_model"] = {
                        "inputs": {
                            "model_name": "2x-ESRGAN.pth"
                        },
                        "class_type": "UpscaleModelLoader",
                        "_meta": {"title": "Pre-Upscale Model Loader"}
                    }
                    workflow["33_up_img"] = {
                        "inputs": {
                            "upscale_model": ["32_up_model", 0],
                            "image": ["30", 0]
                        },
                        "class_type": "ImageUpscaleWithModel",
                        "_meta": {"title": "Pre-Upscale Reference Image"}
                    }
                    workflow["34_resize"] = {
                        "inputs": {
                            "image": ["33_up_img", 0],
                            "upscale_method": "lanczos",
                            "width": width,
                            "height": height,
                            "crop": "disabled"
                        },
                        "class_type": "ImageScale",
                        "_meta": {"title": "Resize Upscaled Reference"}
                    }
                    if "31" in workflow:
                        workflow["31"]["inputs"]["pixels"] = ["34_resize", 0]

                if "3" in workflow:
                    # SDXL denoise scaling: weight 0.20 -> denoise 0.55 (retains exact vibrant source colors & contrast)
                    denoise_val = max(0.40, min(0.60, 0.61 - ((cref_weight if cref_weight is not None else 0.20) * 0.28)))
                    workflow["3"]["inputs"]["denoise"] = denoise_val
                    logger.info(f"SDXL adopted image reference applied via img2img with AI Pre-Upscale (denoise: {denoise_val:.2f}, cref_weight: {cref_weight:.2f})")

            if sref_image_name and "21" in workflow:
                workflow["21"]["inputs"]["image"] = sref_image_name
                if "23" in workflow:
                    workflow["23"]["inputs"]["weight"] = sref_weight
    except KeyError as e:
        logger.error(f"Invalid low-res workflow structure: {e}")
        await interaction.followup.send("Generation workflow template has an invalid structure.")
        return

    # Chain character reference IP-Adapter if provided
    if cref_image_name and not is_flux:
        if use_reference_img2img:
            ip_w = max(0.20, min(0.85, (cref_weight if cref_weight is not None else 0.20) * 0.75))
            workflow = apply_ipadapter_to_workflow(workflow, cref_image_name, weight=ip_w, preset="PLUS (high strength)", node_prefix="cref")
        else:
            workflow = apply_ipadapter_to_workflow(workflow, cref_image_name, weight=cref_weight, preset="PLUS (high strength)", node_prefix="cref")

    if is_face_detailer and not is_flux and not is_com:
        workflow = apply_face_detailer_to_workflow(
            workflow,
            seed=seed,
            cfg=cfg,
            sampler_name=ckpt_cfg.get("sampler_name", "dpmpp_2m") if 'ckpt_cfg' in locals() and ckpt_cfg else "dpmpp_2m",
            scheduler=ckpt_cfg.get("scheduler", "karras") if 'ckpt_cfg' in locals() and ckpt_cfg else "karras",
            steps=20,
            denoise=0.40,
            guide_size=512
        )

    display_prompt = prompt
    if sref_info and "code" in sref_info:
        display_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+random', f"--sref {sref_info['code']}", prompt, flags=re.IGNORECASE)

    generation_id = str(random.randint(100000, 999999))
    
    # 1. Build workflows list first
    if "5" in workflow:
        workflow["5"]["inputs"]["batch_size"] = 1
    workflows_list = []
    expanded_prompts = []
    for i in range(4):
        wf_copy = copy.deepcopy(workflow)
        q_seed = seed + i
        q_rng = random.Random(q_seed)
        q_prompt = expand_dynamic_prompt(cleaned_prompt, q_rng)
        if is_magic:
            q_prompt = apply_magic_enhancement(q_prompt, q_seed)
        
        expanded_prompts.append(q_prompt)
        seed_node = "11" if is_flux else "3"
        if seed_node in wf_copy:
            wf_copy[seed_node]["inputs"]["seed"] = q_seed
        if is_sdxl_powerhouse and "15" in wf_copy:
            wf_copy["15"]["inputs"]["seed"] = q_seed
        if "85" in wf_copy:
            wf_copy["85"]["inputs"]["seed"] = q_seed
        wf_copy["6"]["inputs"]["text"] = q_prompt
        workflows_list.append(wf_copy)

    # 2. Save complete metadata to DB
    active_generations[generation_id] = {
            "prompt": display_prompt,
            "original_prompt": display_prompt,
            "expanded_prompt": cleaned_prompt,
            "neg_prompt": neg_prompt,
            "negative_prompt": neg_prompt,
            "seed": seed,
            "width": width,
            "height": height,
            "loras": loras,
            "checkpoint": selected_model,
            "sref_info": sref_info,
            "sref_weight": sref_weight,
            "cref_image": cref_image_name,
            "cref_weight": cref_weight,
            "expanded_prompts": expanded_prompts,
            "cfg": cfg,
            "status": "pending",
            "timestamp": datetime.now().isoformat(),
            "is_ico": is_ico,
            "rounded_corners": rounded_corners,
            "is_flux": is_flux,
            "is_com": is_com,
            "is_sdxl_powerhouse": is_sdxl_powerhouse,
            "guidance": guidance,
            "freeu": freeu,
            "jump_url": original_post_url,
            "is_lora_build": is_lora_build,
            "lora_session_id": lora_session_id,
            "is_face_detailer": is_face_detailer,
            "is_junji": is_junji
    }
    save_generations()

    flags_info = f"Seed: {seed}, Model: {selected_model}, Size: {width}x{height}, CFG: {cfg:.1f}"
    if sref_image_name:
        flags_info += f", SREF weight: {sref_weight}"
    if cref_image_name:
        flags_info += f", CREF weight: {cref_weight}"
    if is_face_detailer and not is_flux:
        flags_info += ", Face Detailer: Enabled"

    try:
        # Send initial status message
        msg = await send_followup_fallback(
            interaction,
            content=f"Job submitted (Seed: {seed}) — Queuing: '{truncate_prompt(display_prompt, 80)}'..."
        )
        
        # Save message ID
        if msg:
            gen_data = active_generations[generation_id]
            gen_data["message_id"] = msg.id
            active_generations[generation_id] = gen_data
            save_generations()

        last_grid_prog_time = [0.0]
        total_wfs = len(workflows_list)

        def make_grid_progress_cb(wf_idx: int):
            async def on_grid_progress(val, max_val):
                now = asyncio.get_event_loop().time()
                if (now - last_grid_prog_time[0] >= 1.5 or val >= max_val) and max_val > 0:
                    last_grid_prog_time[0] = now
                    total_steps = max_val * total_wfs
                    current_total_step = (wf_idx * max_val) + val
                    pct = min(100, int((current_total_step / total_steps) * 100)) if total_steps > 0 else 0
                    filled = int(round((pct / 100) * 10))
                    bar = "█" * filled + "░" * (10 - filled)
                    img_num = wf_idx + 1
                    prog_content = f"🎨 **Generating Images...**\n`[{bar}] {pct}%` (Image {img_num}/{total_wfs} • Step {val}/{max_val})\n*Model:* `{selected_model}`"
                    try:
                        if msg:
                            await edit_message_fallback(interaction, msg.id, content=prog_content)
                    except Exception:
                        pass
            return on_grid_progress

        start_time = time.perf_counter()
        if is_flux:
            results = []
            for idx, wf in enumerate(workflows_list):
                res = await comfy_client.generate(wf, generation_id=generation_id, progress_callback=make_grid_progress_cb(idx))
                results.append(res)
        else:
            tasks = [comfy_client.generate(wf, generation_id=generation_id, progress_callback=make_grid_progress_cb(idx)) for idx, wf in enumerate(workflows_list)]
            results = await asyncio.gather(*tasks)

        elapsed_time = time.perf_counter() - start_time
        t_breakdown = comfy_client.get_execution_timing()
        init_sec = t_breakdown.get("init_duration", 0.0)
        sample_sec = t_breakdown.get("sampling_duration", 0.0)
        post_sec = t_breakdown.get("post_duration", 0.0)

        # Detect command label (com vs sdxl vs junji vs ico vs flux vs imagine)
        if is_ico:
            cmd_label = "ico"
        elif is_com:
            cmd_label = "com"
        elif is_sdxl_powerhouse:
            cmd_label = "sdxl"
        elif "junji" in str(prompt).lower() or "ito" in str(prompt).lower() or "scapes" in str(prompt).lower():
            cmd_label = "junji"
        elif is_flux:
            cmd_label = "flux"
        elif is_face_detailer:
            cmd_label = "imagine_det"
        else:
            cmd_label = "imagine"

        db.record_generation_metric(
            command=cmd_label,
            duration_seconds=elapsed_time,
            init_seconds=init_sec,
            sampling_seconds=sample_sec,
            post_seconds=post_sec,
            model_name=selected_model,
            steps=20 if not is_flux else 4,
            resolution=f"{width}x{height} (4x)",
            status="success",
            user_id=interaction.user.id if interaction and interaction.user else None,
            metadata={"cfg": cfg, "is_flux": is_flux, "is_ico": is_ico, "is_com": is_com, "is_sdxl_powerhouse": is_sdxl_powerhouse, "is_face_detailer": is_face_detailer}
        )

        timing_data = {
            "elapsed_time": elapsed_time,
            "init_seconds": init_sec,
            "sampling_seconds": sample_sec,
            "post_seconds": post_sec
        }

        raw_images = [r[0] for r in results if r and len(r) > 0]
        
        # Apply automatic color vibrancy & contrast boost for realistic models (bypassed for anime/illustrious to prevent oversaturation)
        is_anime_model = any(k in str(selected_model).lower() for k in ["illustrious", "wai", "hyphoria", "anime", "nai", "furry", "pony"])
        if (use_reference_img2img or reference_image_url or cref_image_name) and not is_anime_model:
            images = [boost_image_vibrancy_and_contrast(img_bytes, saturation=1.22, contrast=1.08) for img_bytes in raw_images]
        else:
            images = raw_images
        
        if len(images) < 4:
            await send_followup_fallback(interaction, content=f"Expected 4 images from generation, but only got {len(images)}.")
            return

        # Fetch latest gen_data in case it was updated (e.g. prompt_ids added)
        gen_data = get_generation(generation_id) or gen_data
        await complete_grid_generation(interaction, generation_id, images, gen_data, status_message_id=msg.id, timing_data=timing_data)
        
        gen_data = get_generation(generation_id) or gen_data
        gen_data["status"] = "completed"
        active_generations[generation_id] = gen_data
        save_generations()

    except StasisInterruptException:
        logger.info(f"Generation {generation_id} was paused and put into stasis.")
        return
    except Exception as e:
        elapsed_time = time.perf_counter() - start_time if 'start_time' in locals() else 0.0
        t_breakdown = comfy_client.get_execution_timing()
        db.record_generation_metric(
            command="imagine",
            duration_seconds=elapsed_time,
            init_seconds=t_breakdown.get("init_duration", 0.0),
            sampling_seconds=t_breakdown.get("sampling_duration", 0.0),
            post_seconds=t_breakdown.get("post_duration", 0.0),
            model_name=selected_model,
            steps=20,
            resolution=f"{width}x{height}",
            status="error",
            error_message=str(e),
            user_id=interaction.user.id if interaction and interaction.user else None
        )
        error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="imagine",
            source_file="bot.py",
            severity=ErrorSeverity.ERROR,
            context={"prompt": prompt, "checkpoint": selected_model, "width": width, "height": height}
        )
        logger.error(f"Error executing imagine command: {e}")
        await send_error_fallback(interaction, f"An error occurred while generating images: {e}")

@bot.tree.command(name="imagine", description="Generate a 2x2 grid of images with ComfyUI.")
@app_commands.describe(
    prompt="The prompt to generate images from (supports --smart, --magic, --ar, --seed, --s, --raw, --sref, --sr)", 
    checkpoint="The checkpoint model to use",
    enhancements="⚡ Enhancements preset (Turbo, Smart Art Director, Magic Prompt)",
    aspect_ratio="Aspect ratio for generated images (--ar)",
    semi_realism="Select Semi-realism LoRA strength (--sr weight)",
    ogarla="Select Ogarla LoRA strength (--ogarla weight)",
    favorite_style="Apply one of your saved favorite styles",
    favorite_prompt="Apply one of your saved favorite prompts",
    style_reference="An image to use as style reference (--sref)"
)
@app_commands.choices(
    checkpoint=SDXL_CHECKPOINT_CHOICES,
    enhancements=SDXL_ENHANCEMENT_CHOICES,
    semi_realism=[
        app_commands.Choice(name="--sr.50", value="--sr.50"),
        app_commands.Choice(name="--sr.60", value="--sr.60"),
        app_commands.Choice(name="--sr.70", value="--sr.70"),
        app_commands.Choice(name="--sr.80", value="--sr.80"),
        app_commands.Choice(name="--sr.90", value="--sr.90"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ],
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
    ]
)
async def imagine(
    interaction: discord.Interaction, 
    prompt: str, 
    checkpoint: str = None, 
    enhancements: str = None,
    aspect_ratio: str = None,
    semi_realism: str = None,
    ogarla: str = None,
    favorite_style: str = None,
    favorite_prompt: str = None,
    style_reference: discord.Attachment = None
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    # Defer response since generation takes time
    await safe_defer(interaction, thinking=True)
    await execute_imagine(interaction, prompt, None, checkpoint, style_reference, favorite_style=favorite_style, semi_realism=semi_realism, aspect_ratio=aspect_ratio, ogarla=ogarla, enhancements=enhancements)

@imagine.autocomplete('favorite_style')
async def imagine_favorite_style_autocomplete(interaction: discord.Interaction, current: str):
    choices = []
    random_label = "🎲 Random (--sref random)"
    if not current or current.lower() in "random" or current.lower() in random_label.lower():
        choices.append(app_commands.Choice(name=random_label, value="random"))

    favorites = db.get_favorite_styles(interaction.user.id)
    for fav in favorites:
        label = f"{fav['style_name']} ({fav['style_code']})"
        if current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=str(fav['style_code'])))
    return choices[:25]

@imagine.autocomplete('favorite_prompt')
async def imagine_favorite_prompt_autocomplete(interaction: discord.Interaction, current: str):
    prompts = db.get_favorite_prompts(interaction.user.id)
    choices = []
    for item in prompts:
        label = f"📌 {item['prompt_name']}".strip()
        if not current or current.lower() in label.lower() or current.lower() in item['prompt_text'].lower():
            choices.append(app_commands.Choice(name=label[:100], value=str(item['id'])))
    return choices[:25]


@bot.tree.command(name="imagine_det", description="✨ Generate a 2x2 grid of images with Face Detailer enabled (SDXL).")
@app_commands.describe(
    prompt="The prompt to generate images from (supports --smart, --magic, --ar, --seed, --s, --raw, --sref, --sr)", 
    checkpoint="The checkpoint model to use",
    enhancements="⚡ Enhancements preset (Turbo, Smart Art Director, Magic Prompt)",
    aspect_ratio="Aspect ratio for generated images (--ar)",
    semi_realism="Select Semi-realism LoRA strength (--sr weight)",
    ogarla="Select Ogarla LoRA strength (--ogarla weight)",
    favorite_style="Apply one of your saved favorite styles",
    favorite_prompt="Apply one of your saved favorite prompts",
    style_reference="An image to use as style reference (--sref)"
)
@app_commands.choices(
    checkpoint=SDXL_CHECKPOINT_CHOICES,
    enhancements=SDXL_ENHANCEMENT_CHOICES,
    semi_realism=[
        app_commands.Choice(name="--sr.50", value="--sr.50"),
        app_commands.Choice(name="--sr.60", value="--sr.60"),
        app_commands.Choice(name="--sr.70", value="--sr.70"),
        app_commands.Choice(name="--sr.80", value="--sr.80"),
        app_commands.Choice(name="--sr.90", value="--sr.90"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ],
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
    ]
)
async def imagine_det(
    interaction: discord.Interaction, 
    prompt: str, 
    checkpoint: str = None, 
    enhancements: str = None,
    aspect_ratio: str = None,
    semi_realism: str = None,
    ogarla: str = None,
    favorite_style: str = None,
    favorite_prompt: str = None,
    style_reference: discord.Attachment = None
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    # Defer response since generation takes time
    await safe_defer(interaction, thinking=True)
    await execute_imagine(
        interaction, 
        prompt, 
        None, 
        checkpoint, 
        style_reference, 
        favorite_style=favorite_style, 
        semi_realism=semi_realism, 
        aspect_ratio=aspect_ratio, 
        ogarla=ogarla, 
        enhancements=enhancements,
        is_face_detailer=True
    )

imagine_det.autocomplete('favorite_style')(imagine_favorite_style_autocomplete)
imagine_det.autocomplete('favorite_prompt')(imagine_favorite_prompt_autocomplete)


@bot.tree.command(name="junji", description="🖤 Generate Junji Ito / Martine Johanna style horror art & dark fantasy landscapes!")
@app_commands.describe(
    prompt="Your scene, subjects, or concepts (prompt builder will weave into selected style)",
    style="Curated master aesthetic preset",
    checkpoint="The checkpoint model to use",
    enhancements="⚡ Enhancements preset (Turbo, Smart Art Director, Magic Prompt)",
    aspect_ratio="Canvas framing (--ar)",
    semi_realism="Select Semi-realism LoRA strength (--sr weight)",
    subject_type="Balance composition toward figures or surrounding scenery",
    ogarla="Select Ogarla LoRA strength (--ogarla weight)",
    favorite_style="Apply one of your saved favorite styles",
    favorite_prompt="Apply one of your saved favorite prompts",
    style_reference="An image to use as style reference (--sref)"
)
@app_commands.choices(
    checkpoint=SDXL_CHECKPOINT_CHOICES,
    enhancements=SDXL_ENHANCEMENT_CHOICES,
    style=[
        app_commands.Choice(name="Junji Ito Manga (Pure Line Art)", value="junji_manga_pure"),
        app_commands.Choice(name="Junji Ito Dark Horror (Deep Shadows)", value="junji_dark_horror"),
        app_commands.Choice(name="Martine Johanna Vibrant (Chromatic)", value="martine_vibrant"),
        app_commands.Choice(name="Martine Johanna Pastel (Melancholic)", value="martine_pastel"),
        app_commands.Choice(name="Junji Ito + Martine Johanna Hybrid Blend", value="ito_johanna_fusion"),
        app_commands.Choice(name="Dark Fantasy Landscape", value="dark_fantasy_landscape"),
        app_commands.Choice(name="Cyberpunk Cityscape", value="cyberpunk_cityscape"),
        app_commands.Choice(name="Ethereal Fine Art Portrait", value="ethereal_portrait"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="Ultrawide (21:9)", value="ultrawide"),
        app_commands.Choice(name="Landscape (16:9)", value="landscape"),
        app_commands.Choice(name="Taskbar Fit (16:9.3)", value="taskbar"),
        app_commands.Choice(name="iPad (10:7)", value="ipad"),
        app_commands.Choice(name="Portrait (3:5)", value="portrait_3_5"),
        app_commands.Choice(name="Tall Portrait (9:16)", value="portrait"),
    ],
    semi_realism=[
        app_commands.Choice(name="--sr.50", value="--sr.50"),
        app_commands.Choice(name="--sr.60", value="--sr.60"),
        app_commands.Choice(name="--sr.70", value="--sr.70"),
        app_commands.Choice(name="--sr.80", value="--sr.80"),
        app_commands.Choice(name="--sr.90", value="--sr.90"),
    ],
    subject_type=[
        app_commands.Choice(name="Scenery / Environment Focus", value="scenery"),
        app_commands.Choice(name="Character / Figure Focus", value="character"),
    ],
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
        app_commands.Choice(name="--ogarla.90", value="--ogarla.90"),
    ]
)
async def junji(
    interaction: discord.Interaction,
    prompt: str,
    style: str,
    checkpoint: str = None,
    enhancements: str = None,
    aspect_ratio: str = "landscape",
    semi_realism: str = None,
    subject_type: str = "scenery",
    ogarla: str = None,
    favorite_style: str = None,
    favorite_prompt: str = None,
    style_reference: discord.Attachment = None
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    await safe_defer(interaction, thinking=True)
    
    sref_url = style_reference.url if style_reference else None
    
    scapes_info = build_scapes_prompt(
        user_prompt=prompt,
        style=style,
        secondary_style=None,
        mode=aspect_ratio,
        subject_type=subject_type,
        sref_url=sref_url
    )
    
    enriched_prompt = scapes_info["final_prompt"]
    logger.info(f"/junji command triggered by {interaction.user.name}: style='{scapes_info['style_name']}' prompt='{enriched_prompt}'")
    
    await execute_imagine(
        interaction,
        prompt=enriched_prompt,
        negative_prompt=scapes_info["negative_additions"],
        checkpoint=checkpoint,
        style_reference=style_reference,
        favorite_style=favorite_style,
        semi_realism=semi_realism,
        aspect_ratio=None,
        ogarla=ogarla,
        enhancements=enhancements,
        is_junji=True
    )

junji.autocomplete('favorite_style')(imagine_favorite_style_autocomplete)
junji.autocomplete('favorite_prompt')(imagine_favorite_prompt_autocomplete)



@bot.tree.command(name="ico", description="🎨 Generate a 2x2 grid of Windows 11 icons (.ico) or convert an image to .ico")
@app_commands.describe(
    prompt="Prompt to generate icons from (optional if image attached)",
    image="Upload an existing image file to convert directly to .ico",
    checkpoint="The checkpoint model to use",
    enhancements="⚡ Icon enhancements preset (Turbo, Square Corners, Magic Prompt)",
    semi_realism="Select Semi-realism LoRA strength (--sr weight)",
    ogarla="Select Ogarla LoRA strength (--ogarla weight)",
    favorite_style="Apply one of your saved favorite styles",
    favorite_prompt="Apply one of your saved favorite prompts",
    style_reference="An image to use as style reference (--sref)",
    negative_prompt="Negative prompt parameters"
)
@app_commands.choices(
    checkpoint=SDXL_CHECKPOINT_CHOICES,
    enhancements=ICO_ENHANCEMENT_CHOICES,
    semi_realism=[
        app_commands.Choice(name="--sr.50", value="--sr.50"),
        app_commands.Choice(name="--sr.60", value="--sr.60"),
        app_commands.Choice(name="--sr.70", value="--sr.70"),
        app_commands.Choice(name="--sr.80", value="--sr.80"),
        app_commands.Choice(name="--sr.90", value="--sr.90"),
    ],
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
    ]
)
async def ico_command(
    interaction: discord.Interaction, 
    prompt: str = None, 
    image: discord.Attachment = None,
    checkpoint: str = None, 
    enhancements: str = None,
    semi_realism: str = None,
    ogarla: str = None,
    favorite_style: str = None,
    favorite_prompt: str = None,
    style_reference: discord.Attachment = None, 
    negative_prompt: str = None
):
    if image is not None:
        await safe_defer(interaction, thinking=True)
        try:
            raw_bytes = await image.read()
            rounded = not (enhancements and "square" in str(enhancements).lower())
            png_bytes, ico_bytes = convert_image_to_ico(raw_bytes, rounded_corners=rounded)
            if not png_bytes or not ico_bytes:
                await send_error_fallback(interaction, "Failed to process attached image into an icon.")
                return

            seed = random.randint(100000, 999999)
            ico_filename = format_image_filename("converted_icon", seed, "ico")
            png_filename = format_image_filename("converted_icon", seed, "png")
            save_ico_file(ico_bytes, ico_filename)

            png_file = discord.File(fp=io.BytesIO(png_bytes), filename=png_filename)
            ico_file = discord.File(fp=io.BytesIO(ico_bytes), filename=ico_filename)

            embed = discord.Embed(
                title="Windows 11 Icon Conversion Complete",
                description=f"**Source File:** `{image.filename}`\n**Resolution:** 1024x1024 (7 ICO Layers: 16x16 → 256x256)\n**Curved Edges:** {'✅ Enabled' if rounded else '❌ Square'}"
            )
            embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
            await send_followup_fallback(interaction, content=f"**Converted Icon:** `{image.filename}`", embed=embed, files=[png_file, ico_file])
            return
        except Exception as e:
            logger.error(f"Error converting attached image to ICO: {e}")
            await send_error_fallback(interaction, f"An error occurred while converting the image to ICO: {e}")
            return

    if not prompt:
        await safe_defer(interaction, ephemeral=True)
        await interaction.followup.send("Please provide either a `prompt` to generate a new icon grid or attach an `image` to convert!", ephemeral=True)
        return

    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    await safe_defer(interaction, thinking=True)
    await execute_imagine(
        interaction,
        prompt=prompt,
        negative_prompt=negative_prompt,
        checkpoint=checkpoint,
        style_reference=style_reference,
        favorite_style=favorite_style,
        semi_realism=semi_realism,
        aspect_ratio="1:1",
        ogarla=ogarla,
        is_ico=True,
        enhancements=enhancements
    )

@bot.tree.command(name="flux", description="✨ Generate high quality AI images with Flux1-Dev!")
@app_commands.describe(
    prompt="The prompt to generate Flux images from",
    ogarla="Select Ogarla Flux LoRA strength (--ogarla weight)",
    aspect_ratio="Aspect ratio for generated images (--ar)",
    favorite_prompt="Apply one of your saved favorite prompts",
    magic_prompt="Enable Magic Prompt enhancer (--magic / --mp)",
    smart="Enable Smart Art Director (Subject-harmonized 12B Magic Prompt)"
)
@app_commands.choices(
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
        app_commands.Choice(name="--ogarla.90", value="--ogarla.90"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ]
)
async def flux_command(
    interaction: discord.Interaction, 
    prompt: str, 
    ogarla: str = None,
    aspect_ratio: str = None,
    favorite_prompt: str = None,
    magic_prompt: bool = False,
    smart: bool = False
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    await safe_defer(interaction, thinking=True)
    await execute_imagine(
        interaction,
        prompt=prompt,
        negative_prompt=None,
        checkpoint="flux1-dev-Q4_K_S.gguf",
        style_reference=None,
        magic_prompt=magic_prompt,
        favorite_style=None,
        semi_realism=None,
        aspect_ratio=aspect_ratio,
        ogarla=ogarla,
        is_flux=True,
        smart=smart
    )

flux_command.autocomplete('favorite_prompt')(imagine_favorite_prompt_autocomplete)

@bot.tree.command(name="com", description="🚀 Community Popular: 12B Flow-Matching (Flux.1 GGUF) with Guidance control on 8GB VRAM!")
@app_commands.describe(
    prompt="Text prompt to generate (supports wildcards {a|b|c}, --smart, --magic, etc.)",
    model_type="Community Flow-Matching UNET (GGUF Quantized for 8GB VRAM)",
    guidance="Flux Guidance scale (1.0 - 10.0, default 3.5)",
    ogarla="Select Ogarla Flux LoRA strength (--ogarla weight)",
    enhancements="🧠 Enhancements preset (Smart Art Director, Magic Prompt)",
    aspect_ratio="Image canvas shape (--ar)",
    favorite_prompt="Apply one of your saved favorite prompts",
    seed="Optional fixed seed for reproducibility"
)
@app_commands.choices(
    model_type=[
        app_commands.Choice(name="Flux.1 Dev GGUF Q4 (General Masterpiece - Recommended)", value="flux1-dev-Q4_K_S.gguf"),
        app_commands.Choice(name="Flux.1 Schnell GGUF Q4 (4-Step Turbo / Instant)", value="flux1-schnell-Q4_K_S.gguf"),
        app_commands.Choice(name="FluxedUp NSFW GGUF Q4 (Community Fine-Tune)", value="fluxedUpFluxNSFW_71Q4GGUF.gguf"),
    ],
    enhancements=FLUX_ENHANCEMENT_CHOICES,
    ogarla=[
        app_commands.Choice(name="--ogarla.60", value="--ogarla.60"),
        app_commands.Choice(name="--ogarla.70", value="--ogarla.70"),
        app_commands.Choice(name="--ogarla.80", value="--ogarla.80"),
        app_commands.Choice(name="--ogarla.90", value="--ogarla.90"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ]
)
async def com_command(
    interaction: discord.Interaction, 
    prompt: str, 
    model_type: str = "flux1-dev-Q4_K_S.gguf",
    guidance: float = 3.5,
    ogarla: str = None,
    enhancements: str = None,
    aspect_ratio: str = None,
    favorite_prompt: str = None,
    seed: int = None
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    if seed is not None:
        prompt = f"{prompt} --seed {seed}"

    await safe_defer(interaction, thinking=True)
    await execute_imagine(
        interaction,
        prompt=prompt,
        negative_prompt=None,
        checkpoint=model_type,
        style_reference=None,
        aspect_ratio=aspect_ratio,
        ogarla=ogarla,
        is_flux=True,
        is_com=True,
        guidance=guidance,
        enhancements=enhancements
    )

com_command.autocomplete('favorite_prompt')(imagine_favorite_prompt_autocomplete)


@bot.tree.command(name="sdxl", description="⚡ 2-Stage Powerhouse SDXL: FreeU V2 + Base Latent + 1.35x Latent Upscale Refiner for 8GB VRAM!")
@app_commands.describe(
    prompt="Text prompt to generate (supports wildcards {a|b|c}, --smart, --magic, etc.)",
    checkpoint="SDXL Checkpoint model (default: Wai Illustrious SDXL v1.70)",
    enhancements="⚡ Enhancements preset (Turbo, Smart Art Director, Magic Prompt, FreeU)",
    aspect_ratio="Image canvas shape (--ar)",
    semi_realism="Semi-Realism LoRA weight preset",
    ogarla="Ogarla Character LoRA weight preset",
    favorite_style="Saved --sref style code from your library",
    favorite_prompt="Apply one of your saved favorite prompts",
    seed="Optional fixed seed for reproducibility"
)
@app_commands.choices(
    checkpoint=SDXL_CHECKPOINT_CHOICES,
    enhancements=SDXL_ENHANCEMENT_CHOICES,
    semi_realism=[
        app_commands.Choice(name="✨ Semi-Realism (.60 - Light)", value="sr.60"),
        app_commands.Choice(name="✨ Semi-Realism (.70 - Medium)", value="sr.70"),
        app_commands.Choice(name="✨ Semi-Realism (.80 - High)", value="sr.80"),
        app_commands.Choice(name="✨ Semi-Realism (.90 - Maximum)", value="sr.90"),
    ],
    ogarla=[
        app_commands.Choice(name="🌿 Ogarla (.60 - Light)", value="ogarla.60"),
        app_commands.Choice(name="🌿 Ogarla (.70 - Medium)", value="ogarla.70"),
        app_commands.Choice(name="🌿 Ogarla (.80 - High)", value="ogarla.80"),
        app_commands.Choice(name="🌿 Ogarla (.90 - Maximum)", value="ogarla.90"),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="1:1 (Square)", value="1:1"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ]
)
async def sdxl_command(
    interaction: discord.Interaction, 
    prompt: str, 
    checkpoint: str = "waiIllustriousSDXL_v170.safetensors",
    enhancements: str = None,
    aspect_ratio: str = None,
    semi_realism: str = None,
    ogarla: str = None,
    favorite_style: str = None,
    favorite_prompt: str = None,
    seed: int = None
):
    if favorite_prompt:
        clean_fav = favorite_prompt.replace("📌", "").strip()
        fav_text = None
        user_prompts = db.get_favorite_prompts(interaction.user.id)
        for item in user_prompts:
            p_id = str(item['id'])
            p_name = item['prompt_name'].strip()
            p_full = item['prompt_text'].strip()
            
            if (p_id == favorite_prompt or p_id == clean_fav or 
                p_name == favorite_prompt or p_name == clean_fav or
                p_name.lower() == clean_fav.lower() or
                p_full == clean_fav or p_full.lower() == clean_fav.lower() or
                (len(clean_fav) >= 10 and p_name.lower().startswith(clean_fav.lower()[:30])) or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                fav_text = item['prompt_text']
                break

        if not fav_text:
            fav_text = clean_fav

        prompt = f"{prompt} {fav_text}".strip() if prompt else fav_text

    if seed is not None:
        prompt = f"{prompt} --seed {seed}"

    await safe_defer(interaction, thinking=True)
    await execute_imagine(
        interaction,
        prompt=prompt,
        negative_prompt=None,
        checkpoint=checkpoint,
        style_reference=None,
        favorite_style=favorite_style,
        semi_realism=semi_realism,
        aspect_ratio=aspect_ratio,
        ogarla=ogarla,
        is_sdxl_powerhouse=True,
        enhancements=enhancements
    )

sdxl_command.autocomplete('favorite_prompt')(imagine_favorite_prompt_autocomplete)
sdxl_command.autocomplete('favorite_style')(imagine_favorite_style_autocomplete)

@bot.tree.command(name="save_prompt", description="Save a custom prompt to your favorite prompts.")
@app_commands.describe(
    name="Short nickname/label for this prompt",
    prompt="The prompt text to save"
)
async def save_prompt_command(interaction: discord.Interaction, name: str, prompt: str):
    db.add_favorite_prompt(interaction.user.id, name, prompt)
    await interaction.response.send_message(f"⭐ Saved prompt **\"{name}\"** to your favorites!", ephemeral=True)

@bot.tree.command(name="my_prompts", description="View and manage your saved favorite prompts.")
async def my_prompts_command(interaction: discord.Interaction):
    prompts = db.get_favorite_prompts(interaction.user.id)
    if not prompts:
        await interaction.response.send_message("You have no saved favorite prompts yet! Click **⭐ Favorite Prompt** on any generation or use `/save_prompt`.", ephemeral=True)
        return
        
    async def prompt_imagine_callback(inter: discord.Interaction, prompt_text: str):
        await execute_imagine(inter, prompt_text)

    view = PromptPaginationView(interaction.user.id, prompts, per_page=5, imagine_callback=prompt_imagine_callback)
    await interaction.response.send_message(embed=view.build_embed(), view=view, ephemeral=True)

@bot.tree.command(name="edit_prompt", description="Edit a saved favorite prompt's name or text.")
@app_commands.describe(prompt_id="The ID of the prompt to edit (check /my_prompts)")
async def edit_prompt_command(interaction: discord.Interaction, prompt_id: int):
    prompts = db.get_favorite_prompts(interaction.user.id)
    selected = next((p for p in prompts if p["id"] == prompt_id), None)
    if not selected:
        await interaction.response.send_message(f"Prompt ID **{prompt_id}** is not in your favorites list.", ephemeral=True)
        return
        
    modal = EditPromptModal(interaction.user.id, selected)
    await interaction.response.send_modal(modal)

@edit_prompt_command.autocomplete('prompt_id')
async def edit_prompt_autocomplete(interaction: discord.Interaction, current: str):
    prompts = db.get_favorite_prompts(interaction.user.id)
    choices = []
    for p in prompts:
        label = f"{p['prompt_name']} (ID: {p['id']})"
        if current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=p['id']))
    return choices[:25]

@bot.tree.command(name="delete_prompt", description="Delete a saved prompt from your favorites.")
@app_commands.describe(prompt_id="The ID of the prompt to delete (check /my_prompts)")
async def delete_prompt_command(interaction: discord.Interaction, prompt_id: int):
    db.remove_favorite_prompt(interaction.user.id, prompt_id)
    await interaction.response.send_message(f"🗑️ Deleted prompt ID **{prompt_id}** from your favorites.", ephemeral=True)

@delete_prompt_command.autocomplete('prompt_id')
async def delete_prompt_autocomplete(interaction: discord.Interaction, current: str):
    prompts = db.get_favorite_prompts(interaction.user.id)
    choices = []
    for p in prompts:
        label = f"{p['prompt_name']} (ID: {p['id']})"
        if current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=p['id']))
    return choices[:25]


@bot.tree.command(name="upscale", description="Upscale an uploaded image to 1920px (long side).")
@app_commands.describe(
    image="The image file you want to upscale"
)
async def upscale(interaction: discord.Interaction, image: discord.Attachment):
    # Defer response since upscaling takes time
    await safe_defer(interaction, thinking=True)
    
    # Check if attachment is an image
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("Please upload a valid image file (PNG/JPG).")
        return
        
    try:
        # Download image from Discord
        image_bytes = await image.read()
        
        # Upload image to ComfyUI
        logger.info(f"Uploading image {image.filename} to ComfyUI...")
        upload_result = await comfy_client.upload_image(image_bytes, image.filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await interaction.followup.send("Failed to upload the image to ComfyUI server.")
            return
            
        logger.info(f"Image uploaded successfully. ComfyUI filename: {uploaded_name}")
        
        # Load upscaler workflow
        workflow_path = "workflows/UPSCALER TO 1920 v2_api.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading upscale workflow file: {e}")
            await interaction.followup.send("Failed to load upscale workflow template.")
            return
            
        # Configure upscale workflow parameters
        # Node "1" is LoadImage in UPSCALER TO 1920 v2.json
        workflow["1"]["inputs"]["image"] = uploaded_name
        
        # Node "6" is SaveImageExtended
        # Apply save folder prefix
        workflow["6"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('upscale')}v1920_"
        
        # Run workflow
        logger.info(f"Executing upscaler workflow for {uploaded_name}...")
        images = await comfy_client.generate(workflow, timeout=14400)
        
        if not images:
            await interaction.followup.send("ComfyUI did not return any upscaled image.")
            return
            
        # Send the upscaled image back to Discord
        upscaled_file_io = io.BytesIO(images[0])
        file = discord.File(fp=upscaled_file_io, filename=f"upscaled_{image.filename}")
        
        embed = discord.Embed(
            title="Image Upscale Complete", 
            description=f"Upscaled to 1920px (long side) using `4x_foolhardy_Remacri.pth` model."
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
        
        await interaction.followup.send(embed=embed, file=file)
    except Exception as e:
        logger.error(f"Error executing upscale command: {e}")


async def execute_video_core(
    interaction: discord.Interaction,
    image_bytes: bytes,
    filename: str,
    prompt: str,
    duration: int = 5,
    smoothness: str = "smooth",
    audio: bool = True,
    audio_prompt: str = None,
    seed: int = None
):
    """Core logic to generate a Wan 2.2 video from raw image bytes and user settings."""
    try:
        # Open image with Pillow to auto-detect original width & height (aspect ratio)
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size

        # Calculate 8GB VRAM optimized dimensions carrying over original aspect ratio
        target_area = settings.get("wan_target_area", 399360)
        width, height = calculate_wan_dimensions(orig_w, orig_h, target_area=target_area)

        # Upload image to ComfyUI
        logger.info(f"Uploading image {filename} ({orig_w}x{orig_h}) to ComfyUI for video generation...")
        upload_result = await comfy_client.upload_image(image_bytes, filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await send_followup_fallback(interaction, content="Failed to upload the image to ComfyUI server.")
            return

        # Load Wan 2.2 Image to Video workflow template
        workflow_path = "workflows/wan22_i2v.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Wan 2.2 workflow template: {e}")
            await send_followup_fallback(interaction, content="Failed to load Wan 2.2 workflow template.")
            return

        # Resolve video generation parameters
        video_seed = seed if seed is not None else random.randint(1, 1125899906842624)
        wan_high_gguf = settings.get("wan_high_gguf", r"gguf\dasiwaWAN22I2V14B_midnightflirtHigh-Q3_K_M.gguf")
        wan_low_gguf = settings.get("wan_low_gguf", r"gguf\dasiwaWAN22I2V14B_midnightflirtLow-Q3_K_M.gguf")
        wan_clip = settings.get("wan_clip", "nsfw_wan_umt5-xxl_fp8_scaled.safetensors")
        wan_clip_vision = settings.get("wan_clip_vision", "clip_vision_h.safetensors")
        wan_vae = settings.get("wan_vae", "wan_2.1_vae.safetensors")
        wan_steps = settings.get("wan_steps", 6)
        wan_cfg = settings.get("wan_cfg", 1.0)
        wan_shift = settings.get("wan_shift", 8.0)
        
        # 8GB VRAM Safe Duration & Frame Scaling:
        # Keep diffusion model frames bounded at 81 frames to avoid 64k token VRAM thrashing/freezing,
        # using high-quality accelerated RIFE interpolation for extended durations (10s = 4x RIFE -> 324 frames @ 32 FPS).
        wan_frames = settings.get("wan_video_frames", 81)
        rife_ckpt = settings.get("rife_ckpt", "rife49.pth")
        wan_fps = settings.get("wan_video_fps", 32)

        if duration == 10:
            duration_sec = 10.0
            if smoothness == "fast":
                rife_multiplier = 2
                out_fps = 16
                use_rife = True
            else:
                rife_multiplier = 4
                out_fps = wan_fps
                use_rife = True
            total_output_frames = wan_frames * rife_multiplier
        elif duration == 5:
            duration_sec = 5.0
            if smoothness == "fast":
                rife_multiplier = 1
                out_fps = 16
                use_rife = False
            else:
                rife_multiplier = 2
                out_fps = wan_fps
                use_rife = True
            total_output_frames = wan_frames * rife_multiplier
        else:
            duration_sec = float(duration)
            rife_multiplier = 2
            out_fps = wan_fps
            use_rife = (smoothness != "fast")
            total_output_frames = wan_frames * rife_multiplier

        # Configure workflow parameters
        if "1" in workflow:
            workflow["1"]["inputs"]["image"] = uploaded_name
        if "2" in workflow:
            workflow["2"]["inputs"]["unet_name"] = wan_high_gguf
        if "21" in workflow:
            workflow["21"]["inputs"]["unet_name"] = wan_low_gguf
        if "22" in workflow:
            workflow["22"]["inputs"]["shift"] = wan_shift
        if "23" in workflow:
            workflow["23"]["inputs"]["shift"] = wan_shift
        if "4" in workflow:
            workflow["4"]["inputs"]["clip_name"] = wan_clip
        if "108" in workflow:
            workflow["108"]["inputs"]["clip_name"] = wan_clip_vision
        if "5" in workflow:
            workflow["5"]["inputs"]["vae_name"] = wan_vae
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = DEFAULT_NEGATIVE_PROMPT
        if "8" in workflow:
            workflow["8"]["inputs"]["width"] = width
            workflow["8"]["inputs"]["height"] = height
            workflow["8"]["inputs"]["length"] = wan_frames
        if "3" in workflow:
            workflow["3"]["inputs"]["noise_seed"] = video_seed
            workflow["3"]["inputs"]["steps"] = wan_steps
            workflow["3"]["inputs"]["cfg"] = wan_cfg
        if "31" in workflow:
            workflow["31"]["inputs"]["steps"] = wan_steps
            workflow["31"]["inputs"]["cfg"] = wan_cfg
        
        # Configure RIFE & Video Output based on chosen smoothness & duration
        seed_suffix = f"_seed{video_seed}"
        if not use_rife:
            if "9" in workflow:
                workflow["9"]["inputs"]["images"] = ["10", 0]
                workflow["9"]["inputs"]["frame_rate"] = out_fps
                workflow["9"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('video')}Wan22_I2V_Fast{seed_suffix}"
            if "75" in workflow:
                del workflow["75"]
        else:
            if "75" in workflow:
                workflow["75"]["inputs"]["ckpt_name"] = rife_ckpt
                workflow["75"]["inputs"]["multiplier"] = rife_multiplier
                workflow["75"]["inputs"]["dtype"] = "float16"
                workflow["75"]["inputs"]["fast_mode"] = True
                workflow["75"]["inputs"]["ensemble"] = False
                workflow["75"]["inputs"]["clear_cache_after_n_frames"] = 20
            if "9" in workflow:
                workflow["9"]["inputs"]["images"] = ["75", 0]
                workflow["9"]["inputs"]["filename_prefix"] = f"{get_dated_save_prefix('video')}Wan22_I2V{seed_suffix}"
                if "frame_rate" in workflow["9"]["inputs"]:
                    workflow["9"]["inputs"]["frame_rate"] = out_fps

        # Configure MMAudio Video-to-Audio Foley Synthesis if enabled
        enable_audio = settings.get("enable_video_audio", True) if audio is None else audio
        if enable_audio and "150" in workflow and "151" in workflow and "152" in workflow:
            mmaudio_model = settings.get("mmaudio_model", "mmaudio_large_44k_v2_fp16.safetensors")
            mmaudio_vae = settings.get("mmaudio_vae", "mmaudio_vae_44k_fp16.safetensors")
            mmaudio_synch = settings.get("mmaudio_synchformer", "mmaudio_synchformer_fp16.safetensors")
            mmaudio_clip = settings.get("mmaudio_clip", "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors")
            mmaudio_steps = settings.get("mmaudio_steps", 25)
            mmaudio_cfg = settings.get("mmaudio_cfg", 4.5)

            workflow["150"]["inputs"]["mmaudio_model"] = mmaudio_model
            workflow["151"]["inputs"]["vae_model"] = mmaudio_vae
            workflow["151"]["inputs"]["synchformer_model"] = mmaudio_synch
            workflow["151"]["inputs"]["clip_model"] = mmaudio_clip

            # Foley prompt: use custom audio_prompt if provided, else use motion prompt
            foley_text = audio_prompt.strip() if audio_prompt and audio_prompt.strip() else prompt
            workflow["152"]["inputs"]["prompt"] = foley_text
            workflow["152"]["inputs"]["duration"] = float(duration_sec)
            workflow["152"]["inputs"]["seed"] = video_seed
            workflow["152"]["inputs"]["steps"] = mmaudio_steps
            workflow["152"]["inputs"]["cfg"] = mmaudio_cfg
            
            # Connect image source for MMAudio based on smoothness mode
            if not use_rife:
                workflow["152"]["inputs"]["images"] = ["10", 0]
            else:
                workflow["152"]["inputs"]["images"] = ["75", 0]

            if "9" in workflow:
                workflow["9"]["inputs"]["audio"] = ["152", 0]
        else:
            # Clean up MMAudio nodes if audio disabled
            for nid in ["150", "151", "152"]:
                if nid in workflow:
                    del workflow[nid]
            if "9" in workflow and "audio" in workflow["9"]["inputs"]:
                del workflow["9"]["inputs"]["audio"]

        # Setup live progress callback for Discord server presence status & chat embed updates
        last_update_time = [0.0]
        status_msg = [None]
        total_steps_done = [0]
        last_val = [0]
        expected_total = wan_steps if wan_steps > 0 else 6

        # Send immediate initial progress embed so user sees instant feedback
        init_bar = create_progress_bar(0, expected_total)
        init_embed = discord.Embed(
            title="🎬 Generating Wan 2.2 Video...",
            description=(
                f"**Motion Prompt:** {prompt}\n"
                f"**Progress:** {init_bar}\n"
                f"**Duration:** {duration_sec:.1f}s ({wan_frames} frames @ {wan_fps} FPS)\n"
                f"**Scaled Size:** {width}x{height} (Aspect Ratio Preserved)\n"
                f"**Model:** `{os.path.basename(wan_high_gguf)}`"
            ),
            color=discord.Color.gold()
        )
        init_embed.set_footer(text="⏳ Initializing & Loading Wan 2.2 GGUF models into VRAM...")
        try:
            status_msg[0] = await send_followup_fallback(interaction, embed=init_embed)
        except Exception:
            pass

        async def on_video_progress(val, max_val):
            # Handle audio synthesis stage (Flow Matching 25 steps)
            if max_val == 25:
                presence_str = f"🔊 Video Audio: Step {val}/25"
                asyncio.create_task(update_bot_presence(presence_str))
                now = asyncio.get_event_loop().time()
                if now - last_update_time[0] >= 1.2 or val == max_val:
                    last_update_time[0] = now
                    audio_bar = create_progress_bar(val, max_val)
                    prog_embed = discord.Embed(
                        title="🎬 Generating Wan 2.2 Video (Audio Foley)...",
                        description=(
                            f"**Motion Prompt:** {prompt}\n"
                            f"**Progress:** {audio_bar}\n"
                            f"**Duration:** {duration_sec:.1f}s ({total_output_frames} frames @ {out_fps} FPS)\n"
                            f"**Scaled Size:** {width}x{height} (Aspect Ratio Preserved)\n"
                            f"**Model:** `{os.path.basename(wan_high_gguf)}`"
                        ),
                        color=discord.Color.gold()
                    )
                    prog_embed.set_footer(text="🔊 Synthesizing synchronized Foley audio track...")
                    try:
                        if status_msg[0]:
                            await status_msg[0].edit(embed=prog_embed)
                    except Exception:
                        pass
                return

            # Detect transition from Stage 1 (High Noise KSampler) to Stage 2 (Low Noise KSampler)
            if val < last_val[0]:
                total_steps_done[0] += last_val[0]
            last_val[0] = val

            current_step = total_steps_done[0] + val
            percent = min(100, int((current_step / expected_total) * 100)) if expected_total > 0 else 0
            presence_str = f"🎬 Video: {percent}% (Step {current_step}/{expected_total})"
            
            # Update sidebar user status & console title
            asyncio.create_task(update_bot_presence(presence_str))
            
            # Update chat embed progress bar (debounced to 1.2s to prevent rate limits)
            now = asyncio.get_event_loop().time()
            if now - last_update_time[0] >= 1.2 or current_step >= expected_total:
                last_update_time[0] = now
                bar = create_progress_bar(min(current_step, expected_total), expected_total)
                stage_name = "🎨 Low Noise KSampler (Stage 2/2)" if current_step > (expected_total // 2) else "🔄 High Noise KSampler (Stage 1/2)"
                prog_embed = discord.Embed(
                    title="🎬 Generating Wan 2.2 Video...",
                    description=(
                        f"**Motion Prompt:** {prompt}\n"
                        f"**Progress:** {bar}\n"
                        f"**Duration:** {duration_sec:.1f}s ({total_output_frames} frames @ {out_fps} FPS)\n"
                        f"**Scaled Size:** {width}x{height} (Aspect Ratio Preserved)\n"
                        f"**Model:** `{os.path.basename(wan_high_gguf)}`"
                    ),
                    color=discord.Color.gold()
                )
                prog_embed.set_footer(text=f"{stage_name} — Rendering frames on GPU...")
                try:
                    if status_msg[0] is None:
                        status_msg[0] = await send_followup_fallback(interaction, embed=prog_embed)
                    else:
                        await status_msg[0].edit(embed=prog_embed)
                except Exception:
                    pass

        logger.info(f"Executing high-speed GGUF Wan 2.2 I2V workflow ({width}x{height}, {wan_frames} frames @ {wan_fps} fps, seed {video_seed})...")
        start_time = time.perf_counter()
        try:
            outputs = await comfy_client.generate(workflow, timeout=14400, progress_callback=on_video_progress)
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            init_sec = t_breakdown.get("init_duration", 0.0)
            sample_sec = t_breakdown.get("sampling_duration", 0.0)
            post_sec = t_breakdown.get("post_duration", 0.0)

            db.record_generation_metric(
                command="video",
                duration_seconds=elapsed_time,
                init_seconds=init_sec,
                sampling_seconds=sample_sec,
                post_seconds=post_sec,
                model_name=wan_high_gguf,
                steps=wan_steps,
                resolution=f"{width}x{height}",
                status="success",
                user_id=interaction.user.id if interaction.user else None,
                metadata={"frames": wan_frames, "fps": wan_fps, "smoothness": smoothness}
            )
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            db.record_generation_metric(
                command="video",
                duration_seconds=elapsed_time,
                init_seconds=t_breakdown.get("init_duration", 0.0),
                sampling_seconds=t_breakdown.get("sampling_duration", 0.0),
                post_seconds=t_breakdown.get("post_duration", 0.0),
                model_name=wan_high_gguf,
                steps=wan_steps,
                resolution=f"{width}x{height}",
                status="error",
                error_message=str(e),
                user_id=interaction.user.id if interaction.user else None
            )
            raise
        finally:
            await update_bot_presence(None)
            if status_msg[0]:
                try:
                    await status_msg[0].delete()
                except Exception:
                    pass

        if not outputs or not isinstance(outputs, list):
            await send_followup_fallback(interaction, content="ComfyUI did not return any video output.")
            return

        video_bytes = outputs[0]
        video_file_io = io.BytesIO(video_bytes)
        
        file = discord.File(fp=video_file_io, filename=format_image_filename("wan22_video", video_seed, "mp4"))

        audio_info = "🔊 `MMAudio (44.1kHz Synced Foley)`" if (enable_audio and "152" in workflow) else "🔇 `Disabled`"

        embed = discord.Embed(
            title="🎬 Wan 2.2 Fast GGUF Video Generation Complete",
            description=(
                f"**Motion Prompt:** {prompt}\n"
                f"**Duration:** {duration_sec:.1f}s ({total_output_frames} frames @ {out_fps} FPS)\n"
                f"**Audio Track:** {audio_info}\n"
                f"**Render Time:** `{elapsed_time:.1f}s` (Init: `{init_sec:.1f}s` | Sample: `{sample_sec:.1f}s` | Post: `{post_sec:.1f}s`)\n"
                f"**Original Size:** {orig_w}x{orig_h}\n"
                f"**8GB VRAM Scaled Size:** {width}x{height} (Aspect Ratio Preserved)\n"
                f"**High/Low GGUF Models:** Q3_K_M (Shift 8.0, 6 Steps, CFG 1.0)\n"
                f"**Seed:** {video_seed}"
            ),
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id}) • Rendered in {elapsed_time:.1f}s")

        tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
        await send_followup_fallback(interaction, content=tag, embed=embed, file=file)
    except Exception as e:
        logger.error(f"Error executing video core generation: {e}")
        await send_error_fallback(interaction, f"An error occurred during video generation: {e}")


@bot.tree.command(name="video", description="Generate a 5s or 10s video with synced audio from an image using Wan 2.2.")
@app_commands.describe(
    image="The source image file you want to animate",
    prompt="Text prompt describing the desired video motion or action",
    duration="Video duration in seconds (5 or 10 seconds, default 5)",
    smoothness="Motion smoothing speed mode (Smooth 32 FPS vs Fast 16 FPS)",
    audio="Generate synchronized audio/sound effects (default: True)",
    audio_prompt="Optional custom sound effects / Foley prompt (defaults to motion prompt)",
    seed="Optional seed for generation reproducibility"
)
@app_commands.choices(
    duration=[
        app_commands.Choice(name="5 seconds (81 frames)", value=5),
        app_commands.Choice(name="10 seconds (161 frames)", value=10),
    ],
    smoothness=[
        app_commands.Choice(name="🎬 Smooth (32 FPS - Accelerated RIFE)", value="smooth"),
        app_commands.Choice(name="⚡ Ultra Fast (16 FPS - Native / No RIFE)", value="fast"),
    ]
)
async def video_command(
    interaction: discord.Interaction,
    image: discord.Attachment,
    prompt: str,
    duration: int = 5,
    smoothness: str = "smooth",
    audio: bool = True,
    audio_prompt: str = None,
    seed: int = None
):
    # Defer response since video generation takes time
    await safe_defer(interaction, thinking=True)

    # Check if attachment is a valid image
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("Please upload a valid image file (PNG/JPG/WEBP).")
        return

    try:
        image_bytes = await image.read()
        await execute_video_core(
            interaction=interaction,
            image_bytes=image_bytes,
            filename=image.filename,
            prompt=prompt,
            duration=duration,
            smoothness=smoothness,
            audio=audio,
            audio_prompt=audio_prompt,
            seed=seed
        )
    except Exception as e:
        logger.error(f"Error executing video command: {e}")
        await send_error_fallback(interaction, f"An error occurred during video generation: {e}")


async def execute_animate_message(interaction: discord.Interaction, message: discord.Message):
    """Context menu command handler to prompt and animate any message image into a video."""
    image_url = None
    filename = None

    if message.attachments:
        for att in message.attachments:
            if (att.content_type and att.content_type.startswith("image/")) or att.url.split('?')[0].lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_url = att.url
                filename = att.filename
                break
        if not image_url and message.attachments:
            image_url = message.attachments[0].url
            filename = message.attachments[0].filename

    if not image_url and message.embeds:
        for emb in message.embeds:
            if emb.image and emb.image.url:
                image_url = emb.image.url
                filename = f"video_{message.id}.png"
                break
            elif emb.thumbnail and emb.thumbnail.url:
                image_url = emb.thumbnail.url
                filename = f"video_{message.id}.png"
                break

    if not image_url:
        await interaction.response.send_message("❌ No valid image found in that message to animate.", ephemeral=True)
        return

    parsed = parse_adopted_post(message)
    initial_prompt = parsed.get("clean_prompt") if parsed and parsed.get("clean_prompt") != "No prompt text found" else ""

    async def on_modal_submit(interaction: discord.Interaction, prompt: str, duration_str: str, smoothness_str: str, seed_str: str):
        await safe_defer(interaction, thinking=True)
        try:
            # Parse duration
            duration = 10 if duration_str == "10" else 5
            
            # Parse smoothness
            smoothness = "fast" if "fast" in smoothness_str.lower() else "smooth"

            # Parse seed
            seed = int(seed_str.strip()) if (seed_str and seed_str.strip().isdigit()) else None

            # Download image bytes
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        await send_followup_fallback(interaction, content="❌ Failed to download the image from the message.")
                        return
                    img_bytes = await resp.read()

            await execute_video_core(
                interaction=interaction,
                image_bytes=img_bytes,
                filename=filename or f"video_{message.id}.png",
                prompt=prompt,
                duration=duration,
                smoothness=smoothness,
                seed=seed
            )
        except Exception as e:
            logger.error(f"Error executing animate context menu: {e}")
            await send_error_fallback(interaction, f"An error occurred during video generation: {e}")

    modal = VideoPromptModal(default_prompt=initial_prompt, on_submit_callback=on_modal_submit)
    await interaction.response.send_modal(modal)



@bot.tree.context_menu(name="Animate to Video")
async def animate_to_video_context(interaction: discord.Interaction, message: discord.Message):
    """Context menu command to animate any right-clicked image message into a Wan 2.2 video."""
    await execute_animate_message(interaction, message)



@bot.tree.command(name="ltx", description="Generate high-speed video animation using LTX-Video (optimized for 8GB VRAM).")
@app_commands.describe(
    image="The source image file you want to animate",
    prompt="Text prompt describing the desired video motion or action",
    duration="Video duration in seconds (4s, 6s, 8s, or 10s, default 4s)",
    motion_strength="Motion intensity (1 to 10, default 7)",
    seed="Optional seed for generation reproducibility"
)
@app_commands.choices(
    duration=[
        app_commands.Choice(name="4 seconds (97 frames - Fast ~35s)", value=4),
        app_commands.Choice(name="6 seconds (161 frames - ~55s)", value=6),
        app_commands.Choice(name="8 seconds (209 frames - ~75s)", value=8),
        app_commands.Choice(name="10 seconds (257 frames - ~95s)", value=10),
    ]
)
async def ltx_command(interaction: discord.Interaction, image: discord.Attachment, prompt: str, duration: int = 4, motion_strength: int = 7, seed: int = None):
    """Generate rapid video animation using LTX-Video."""
    await safe_defer(interaction, thinking=True)

    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("Please upload a valid image file (PNG/JPG/WEBP).")
        return

    try:
        image_bytes = await image.read()
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size

        # 8GB VRAM friendly dimensions snapped to 32-pixel boundaries
        calc_w, calc_h = calculate_wan_dimensions(orig_w, orig_h, target_area=393216)
        width = max(256, (calc_w // 32) * 32)
        height = max(256, (calc_h // 32) * 32)

        upload_result = await comfy_client.upload_image(image_bytes, image.filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await interaction.followup.send("Failed to upload the image to ComfyUI server.")
            return

        workflow_path = "workflows/ltx_i2v.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading LTX workflow template: {e}")
            await interaction.followup.send("Failed to load LTX-Video workflow template.")
            return

        video_seed = seed if seed is not None else random.randint(1, 1125899906842624)
        motion_strength_val = max(0.5, min(1.0, motion_strength / 10.0))

        # Length in LTX frames (must be (8 * k) + 1 at 25 fps)
        if duration == 10:
            ltx_frames = 257
        elif duration == 8:
            ltx_frames = 209
        elif duration == 6:
            ltx_frames = 161
        else:
            ltx_frames = 97
        duration_sec = (ltx_frames - 1) / 25.0

        if "1" in workflow:
            workflow["1"]["inputs"]["image"] = uploaded_name
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = DEFAULT_NEGATIVE_PROMPT
        if "8" in workflow:
            workflow["8"]["inputs"]["width"] = width
            workflow["8"]["inputs"]["height"] = height
            workflow["8"]["inputs"]["length"] = ltx_frames
            workflow["8"]["inputs"]["strength"] = motion_strength_val
        if "3" in workflow:
            workflow["3"]["inputs"]["seed"] = video_seed
            workflow["3"]["inputs"]["denoise"] = 1.0

        # Setup live progress callback
        last_update_time = [0.0]
        status_msg = [None]

        # Send immediate initial progress embed so user sees instant feedback
        init_bar = create_progress_bar(0, 25)
        init_embed = discord.Embed(
            title="⚡ Generating LTX-Video...",
            description=(
                f"**Motion Prompt:** {prompt}\n"
                f"**Progress:** {init_bar}\n"
                f"**Duration:** {duration_sec:.1f}s ({ltx_frames} frames @ 25 FPS)\n"
                f"**Motion Strength:** {motion_strength}/10\n"
                f"**Resolution:** {width}x{height}"
            ),
            color=discord.Color.teal()
        )
        init_embed.set_footer(text="⏳ Initializing & Loading LTX model into VRAM...")
        try:
            status_msg[0] = await send_followup_fallback(interaction, embed=init_embed)
        except Exception:
            pass

        async def on_ltx_progress(val, max_val):
            percent = min(100, int((val / max_val) * 100)) if max_val > 0 else 0
            presence_str = f"⚡ LTX-Video: {percent}% (Step {val}/{max_val})"
            asyncio.create_task(update_bot_presence(presence_str))

            now = asyncio.get_event_loop().time()
            if now - last_update_time[0] >= 1.2 or val == max_val:
                last_update_time[0] = now
                bar = create_progress_bar(val, max_val)
                prog_embed = discord.Embed(
                    title="⚡ Generating LTX-Video...",
                    description=(
                        f"**Motion Prompt:** {prompt}\n"
                        f"**Progress:** {bar}\n"
                        f"**Duration:** {duration_sec:.1f}s ({ltx_frames} frames @ 25 FPS)\n"
                        f"**Motion Strength:** {motion_strength}/10\n"
                        f"**Resolution:** {width}x{height}"
                    ),
                    color=discord.Color.teal()
                )
                prog_embed.set_footer(text="Sampling video frames on GPU...")
                try:
                    if status_msg[0] is None:
                        status_msg[0] = await send_followup_fallback(interaction, embed=prog_embed)
                    else:
                        await status_msg[0].edit(embed=prog_embed)
                except Exception:
                    pass

        start_time = time.perf_counter()
        try:
            outputs = await comfy_client.generate(workflow, timeout=3600, progress_callback=on_ltx_progress)
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            init_sec = t_breakdown.get("init_duration", 0.0)
            sample_sec = t_breakdown.get("sampling_duration", 0.0)
            post_sec = t_breakdown.get("post_duration", 0.0)

            db.record_generation_metric(
                command="ltx",
                duration_seconds=elapsed_time,
                init_seconds=init_sec,
                sampling_seconds=sample_sec,
                post_seconds=post_sec,
                model_name="ltx-video-2b-v0.9.1",
                steps=25,
                resolution=f"{width}x{height}",
                status="success",
                user_id=interaction.user.id if interaction.user else None,
                metadata={"duration_sec": duration_sec, "frames": ltx_frames, "motion_strength": motion_strength}
            )
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            db.record_generation_metric(
                command="ltx",
                duration_seconds=elapsed_time,
                init_seconds=t_breakdown.get("init_duration", 0.0),
                sampling_seconds=t_breakdown.get("sampling_duration", 0.0),
                post_seconds=t_breakdown.get("post_duration", 0.0),
                model_name="ltx-video-2b-v0.9.1",
                steps=25,
                resolution=f"{width}x{height}",
                status="error",
                error_message=str(e),
                user_id=interaction.user.id if interaction.user else None
            )
            raise
        finally:
            await update_bot_presence(None)
            if status_msg[0]:
                try:
                    await status_msg[0].delete()
                except Exception:
                    pass

        if not outputs or not isinstance(outputs, list):
            await send_followup_fallback(interaction, content="ComfyUI did not return any video output.")
            return

        video_bytes = outputs[0]
        video_file_io = io.BytesIO(video_bytes)
        file = discord.File(fp=video_file_io, filename=format_image_filename("ltx_video", video_seed, "mp4"))

        embed = discord.Embed(
            title="⚡ LTX-Video Generation Complete",
            description=(
                f"**Motion Prompt:** {prompt}\n"
                f"**Duration:** {duration_sec:.1f}s ({ltx_frames} frames @ 25 FPS)\n"
                f"**Render Time:** `{elapsed_time:.1f}s` (Init: `{init_sec:.1f}s` | Sample: `{sample_sec:.1f}s` | Post: `{post_sec:.1f}s`)\n"
                f"**Motion Strength:** {motion_strength}/10 (Intensity: {motion_strength_val:.2f})\n"
                f"**Scaled Size:** {width}x{height}\n"
                f"**Seed:** {video_seed}"
            ),
            color=discord.Color.teal()
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id}) • Rendered in {elapsed_time:.1f}s")
        tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
        await send_followup_fallback(interaction, content=tag, embed=embed, file=file)
    except Exception as e:
        logger.error(f"Error executing LTX video command: {e}")
        await send_error_fallback(interaction, f"An error occurred during LTX video generation: {e}")


@bot.tree.command(name="hunyuan", description="Generate high-fidelity video animation using HunyuanVideo (optimized for 8GB VRAM).")
@app_commands.describe(
    image="The source image file you want to animate",
    prompt="Text prompt describing the desired video motion or action",
    motion_strength="Motion intensity (1 to 10, default 7)",
    seed="Optional seed for generation reproducibility"
)
async def hunyuan_command(interaction: discord.Interaction, image: discord.Attachment, prompt: str, motion_strength: int = 7, seed: int = None):
    """Generate video animation using HunyuanVideo GGUF."""
    await safe_defer(interaction, thinking=True)

    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("Please upload a valid image file (PNG/JPG/WEBP).")
        return

    try:
        image_bytes = await image.read()
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size

        width, height = calculate_wan_dimensions(orig_w, orig_h, target_area=393216)

        upload_result = await comfy_client.upload_image(image_bytes, image.filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await interaction.followup.send("Failed to upload the image to ComfyUI server.")
            return

        workflow_path = "workflows/hunyuan_i2v.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading Hunyuan workflow template: {e}")
            await interaction.followup.send("Failed to load HunyuanVideo workflow template.")
            return

        video_seed = seed if seed is not None else random.randint(1, 1125899906842624)
        denoise_val = max(0.5, min(1.0, motion_strength / 10.0))

        if "1" in workflow:
            workflow["1"]["inputs"]["image"] = uploaded_name
        if "6" in workflow:
            workflow["6"]["inputs"]["text"] = prompt
        if "7" in workflow:
            workflow["7"]["inputs"]["text"] = DEFAULT_NEGATIVE_PROMPT
        if "3" in workflow:
            workflow["3"]["inputs"]["seed"] = video_seed
            workflow["3"]["inputs"]["denoise"] = denoise_val

        last_update_time = [0]
        status_msg = [None]

        async def on_hunyuan_progress(val, max_val):
            percent = min(100, int((val / max_val) * 100)) if max_val > 0 else 0
            presence_str = f"🎥 Hunyuan: {percent}% (Step {val}/{max_val})"
            asyncio.create_task(update_bot_presence(presence_str))

            now = asyncio.get_event_loop().time()
            if now - last_update_time[0] >= 1.2 or val == max_val:
                last_update_time[0] = now
                bar = create_progress_bar(val, max_val)
                prog_embed = discord.Embed(
                    title="🎥 Generating HunyuanVideo...",
                    description=(
                        f"**Motion Prompt:** {prompt}\n"
                        f"**Progress:** {bar}\n"
                        f"**Motion Strength:** {motion_strength}/10\n"
                        f"**Resolution:** {width}x{height}"
                    ),
                    color=discord.Color.purple()
                )
                try:
                    if status_msg[0] is None:
                        status_msg[0] = await send_followup_fallback(interaction, embed=prog_embed)
                    else:
                        await status_msg[0].edit(embed=prog_embed)
                except Exception:
                    pass

        start_time = time.perf_counter()
        try:
            outputs = await comfy_client.generate(workflow, timeout=3600, progress_callback=on_hunyuan_progress)
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            init_sec = t_breakdown.get("init_duration", 0.0)
            sample_sec = t_breakdown.get("sampling_duration", 0.0)
            post_sec = t_breakdown.get("post_duration", 0.0)

            db.record_generation_metric(
                command="hunyuan",
                duration_seconds=elapsed_time,
                init_seconds=init_sec,
                sampling_seconds=sample_sec,
                post_seconds=post_sec,
                model_name="hunyuanvideo_Q3_K_M",
                steps=20,
                resolution=f"{width}x{height}",
                status="success",
                user_id=interaction.user.id if interaction.user else None,
                metadata={"motion_strength": motion_strength}
            )
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy_client.get_execution_timing()
            db.record_generation_metric(
                command="hunyuan",
                duration_seconds=elapsed_time,
                init_seconds=t_breakdown.get("init_duration", 0.0),
                sampling_seconds=t_breakdown.get("sampling_duration", 0.0),
                post_seconds=t_breakdown.get("post_duration", 0.0),
                model_name="hunyuanvideo_Q3_K_M",
                steps=20,
                resolution=f"{width}x{height}",
                status="error",
                error_message=str(e),
                user_id=interaction.user.id if interaction.user else None
            )
            raise
        finally:
            await update_bot_presence(None)
            if status_msg[0]:
                try:
                    await status_msg[0].delete()
                except Exception:
                    pass

        if not outputs or not isinstance(outputs, list):
            await send_followup_fallback(interaction, content="ComfyUI did not return any video output.")
            return

        video_bytes = outputs[0]
        video_file_io = io.BytesIO(video_bytes)
        file = discord.File(fp=video_file_io, filename=format_image_filename("hunyuan_video", video_seed, "mp4"))

        embed = discord.Embed(
            title="🎥 HunyuanVideo Generation Complete",
            description=(
                f"**Motion Prompt:** {prompt}\n"
                f"**Render Time:** `{elapsed_time:.1f}s` (Init: `{init_sec:.1f}s` | Sample: `{sample_sec:.1f}s` | Post: `{post_sec:.1f}s`)\n"
                f"**Motion Strength:** {motion_strength}/10 (Denoise: {denoise_val:.2f})\n"
                f"**Scaled Size:** {width}x{height}\n"
                f"**Seed:** {video_seed}"
            ),
            color=discord.Color.purple()
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id}) • Rendered in {elapsed_time:.1f}s")
        tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
        await send_followup_fallback(interaction, content=tag, embed=embed, file=file)
    except Exception as e:
        logger.error(f"Error executing Hunyuan video command: {e}")
        await send_error_fallback(interaction, f"An error occurred during Hunyuan video generation: {e}")


@bot.tree.command(name="diagnostics", description="View bot generation metrics, speed benchmarks, and troubleshooting data.")
async def diagnostics_command(interaction: discord.Interaction):
    """Displays telemetry benchmarks, average render times, and recent error diagnostics."""
    await safe_defer(interaction, ephemeral=False)

    summary = db.get_performance_summary()
    recent = db.get_recent_metrics(limit=10)

    embed = discord.Embed(
        title="📊 Shallot-CUI Bot Performance & Telemetry Diagnostics",
        color=discord.Color.blue()
    )

    if summary:
        summary_text = ""
        for cmd, data in summary.items():
            total = data["total_runs"]
            succ = data["successes"]
            avg_d = data["avg_duration"]
            avg_i = data.get("avg_init", 0.0)
            avg_s = data.get("avg_sampling", 0.0)
            avg_p = data.get("avg_post", 0.0)
            min_d = data["min_duration"]
            max_d = data["max_duration"]
            summary_text += (
                f"• **`/{cmd}`**: Avg `{avg_d:.1f}s` (Min: `{min_d:.1f}s` / Max: `{max_d:.1f}s`) • {succ}/{total} OK\n"
                f"  ↳ *Breakdown:* Init `{avg_i:.1f}s` | Sample `{avg_s:.1f}s` | Post `{avg_p:.1f}s`\n"
            )
        embed.add_field(name="⚡ Speed Benchmarks by Generator", value=summary_text or "No metrics recorded yet.", inline=False)
    else:
        embed.add_field(name="⚡ Speed Benchmarks", value="No generation data recorded yet. Run `/imagine`, `/ltx`, or `/video` to gather benchmarks.", inline=False)

    if recent:
        recent_lines = []
        for r in recent[:6]:
            status_icon = "✅" if r["status"] == "success" else "❌"
            res = r.get("resolution") or "N/A"
            dur = r.get("duration_seconds") or 0.0
            i_sec = r.get("init_seconds") or 0.0
            s_sec = r.get("sampling_seconds") or 0.0
            p_sec = r.get("post_seconds") or 0.0
            cmd = r.get("command") or "unknown"
            created = r.get("created_at") or ""
            line = f"{status_icon} `/{cmd}` ({res}) - `{dur:.1f}s` (Init: `{i_sec:.1f}s` | Sample: `{s_sec:.1f}s` | Post: `{p_sec:.1f}s`)"
            if r["status"] != "success" and r.get("error_message"):
                err_snippet = r["error_message"][:60]
                line += f"\n  ↳ *Err:* `{err_snippet}`"
            recent_lines.append(line)
        embed.add_field(name="🕒 Recent Runs (Last 6)", value="\n".join(recent_lines), inline=False)

    embed.set_footer(text=f"Diagnostics requested by {interaction.user.name}")
    await send_followup_fallback(interaction, embed=embed)






async def handle_generate_described(interaction: discord.Interaction, generation_id: str, desc_type: str, ar: str = "16:9", use_sr: bool = True, use_oga: bool = False, model_choice: str = "hyphoria"):
    """Generates an image grid using stored caption or detailed description from /describe with chosen AR, LoRA, and model settings."""
    await safe_defer(interaction)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find description session data. It may have expired.", ephemeral=True)
        return

    base_prompt = gen_data.get("caption") if desc_type == "caption" else gen_data.get("detailed_caption")
    if not base_prompt:
        await interaction.followup.send(f"No {desc_type} prompt found in session.", ephemeral=True)
        return

    # Resolve sr_flag from use_sr parameter
    sr_flag = None
    if isinstance(use_sr, str):
        if use_sr in ["sr60", "sr.60"]:
            sr_flag = "--sr.60"
        elif use_sr in ["sr70", "sr.70"]:
            sr_flag = "--sr.70"
        elif use_sr in ["sr80", "sr.80"]:
            sr_flag = "--sr.80"
        elif use_sr in ["sr90", "sr.90"]:
            sr_flag = "--sr.90"
        elif use_sr == "sr" or use_sr.lower() in ["true", "1", "on"]:
            sr_flag = "--sr.90" if model_choice == "hyphoria" else "--sr.75"
    elif use_sr is True:
        sr_flag = "--sr.90" if model_choice == "hyphoria" else "--sr.75"

    # Build prompt string with triggers, base description, LoRA flags, and aspect ratio
    prompt_parts = []
    if sr_flag:
        prompt_parts.append("Semi-realism, masterpiece, best quality, absurdres.")
    if use_oga:
        prompt_parts.append("ogarla,")

    prompt_parts.append(base_prompt)

    if sr_flag:
        prompt_parts.append(sr_flag)
    if use_oga:
        prompt_parts.append("--ogarla.70")
    if ar:
        prompt_parts.append(f"--ar {ar}")

    full_prompt = " ".join(prompt_parts)
    selected_model = "hyphoriaIlluNAI_v001.safetensors" if model_choice == "hyphoria" else None
    await execute_imagine(interaction, prompt=full_prompt, model=selected_model)


async def handle_update_describe_view(interaction: discord.Interaction, generation_id: str, new_ar: str, new_sr = True, new_oga: bool = False, new_model: str = "hyphoria"):
    """Updates the interactive buttons on the /describe result embed when AR, SR, Ogarla, or Model toggle is clicked."""
    view = DescribeButtons(generation_id, ar=new_ar, sr=new_sr, oga=new_oga, model_choice=new_model)
    try:
        await interaction.response.edit_message(view=view)
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"Ignored expected interaction update error: {e}")


async def handle_update_blend_view(interaction: discord.Interaction, generation_id: str, new_ar: str = None, new_sr = None, new_oga: bool = None, new_model: str = None, new_comp: str = None, new_sref = None):
    """Updates the interactive buttons and settings on the /blend result embed."""
    gen_data = get_generation(generation_id) or {}
    if new_ar is not None:
        gen_data["ar"] = new_ar
    if new_sr is not None:
        gen_data["sr"] = new_sr
    if new_oga is not None:
        gen_data["oga"] = new_oga
    if new_model is not None:
        gen_data["model_choice"] = new_model
    if new_comp is not None:
        gen_data["comp_strength"] = new_comp
    if new_sref is not None:
        gen_data["sref_rand"] = new_sref

    db.save_generation(generation_id, gen_data)

    author_str = gen_data.get("author_str", interaction.user.name if (interaction and interaction.user) else "User")
    image_url = gen_data.get("image_url")
    embed = build_blend_embed(gen_data, author_str=author_str, image_url=image_url)
    
    view = BlendButtons(
        generation_id=generation_id,
        ar=gen_data.get("ar", "16:9"),
        sr=gen_data.get("sr", True),
        oga=gen_data.get("oga", False),
        model_choice=gen_data.get("model_choice", "wai"),
        comp_strength=gen_data.get("comp_strength", "style"),
        sref_rand=gen_data.get("sref_rand", "nosref")
    )
    try:
        await interaction.response.edit_message(embed=embed, view=view)
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"Ignored expected interaction update error: {e}")


async def handle_submit_edit_blend_prompts(interaction: discord.Interaction, generation_id: str, new_caption: str, new_detailed: str, extra_details: str = ""):
    """Updates the caption, detailed description, and extra details for a /blend session and edits the embed in place."""
    gen_data = get_generation(generation_id)
    if not gen_data:
        await send_error_fallback(interaction, "Blend session expired.")
        return

    gen_data["caption"] = new_caption
    gen_data["detailed_caption"] = new_detailed
    gen_data["extra_details"] = extra_details
    db.save_generation(generation_id, gen_data)

    author_str = gen_data.get("author_str", interaction.user.name if (interaction and interaction.user) else "User")
    image_url = gen_data.get("image_url")
    ar = gen_data.get("ar", "16:9")
    sr = gen_data.get("sr", True)
    oga = gen_data.get("oga", False)
    model_choice = gen_data.get("model_choice", "wai")
    comp_strength = gen_data.get("comp_strength", "style")
    sref_rand = gen_data.get("sref_rand", "nosref")

    embed = build_blend_embed(gen_data, author_str=author_str, image_url=image_url, is_edited=True)
    view = BlendButtons(
        generation_id=generation_id,
        ar=ar,
        sr=sr,
        oga=oga,
        model_choice=model_choice,
        comp_strength=comp_strength,
        sref_rand=sref_rand
    )
    await interaction.response.edit_message(embed=embed, view=view)


async def handle_generate_blended(interaction: discord.Interaction, generation_id: str, desc_type: str, ar: str = "16:9", use_sr = True, use_oga: bool = False, model_choice: str = "wai", comp_strength: str = "style", use_sref_rand = "nosref"):
    """Generates blended image grid(s) using stored caption/detailed description + uploaded base image with chosen settings."""
    await safe_defer(interaction)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find blend session data. It may have expired.", ephemeral=True)
        return

    base_prompt = gen_data.get("caption") if desc_type == "caption" else gen_data.get("detailed_caption")
    if not base_prompt:
        await interaction.followup.send(f"No {desc_type} prompt found in session.", ephemeral=True)
        return

    uploaded_image_name = gen_data.get("uploaded_image_name")
    user_extra_prompt = gen_data.get("user_prompt", "")
    extra_details = gen_data.get("extra_details", "").strip()

    # Map model choice to checkpoint filename
    model_ckpt_map = {
        "wai": "waiIllustriousSDXL_v170.safetensors",
        "illustrious_realism": "illustriousRealismBy_v10VAE.safetensors",
        "realvis": "RealVisXL_V4.0.safetensors",
        "juggernaut": "juggernautXL_ragnarok.safetensors",
        "copax": "CopaxTimeLessXL.safetensors",
        "ultra": "ultraRealisticByStable_v25.safetensors",
        "hyphoria": "hyphoriaIlluNAI_v001.safetensors",
        "nova": "novaFurryXL_ilV180A.safetensors",
        "default": "waiIllustriousSDXL_v170.safetensors"
    }
    selected_model = model_ckpt_map.get(model_choice, model_choice)
    if not selected_model or not selected_model.endswith(".safetensors"):
        selected_model = "waiIllustriousSDXL_v170.safetensors"

    # Build base prompt parts
    base_parts = []
    
    is_anime = any(k in selected_model.lower() for k in ["illustrious", "wai", "hyphoria", "nai", "furry", "anime"])
    
    if use_sr and use_sr != "nosr":
        if isinstance(use_sr, str) and use_sr.startswith("sr"):
            sr_tag = f"--{use_sr}"
        else:
            sr_tag = "--sr.90" if is_anime else "--sr.75"
        base_parts.append("Semi-realism, masterpiece, best quality.")
    else:
        sr_tag = None

    if use_oga:
        base_parts.append("ogarla,")

    base_parts.append(base_prompt)

    if extra_details:
        base_parts.append(extra_details)

    if user_extra_prompt:
        base_parts.append(user_extra_prompt)

    if sr_tag:
        base_parts.append(sr_tag)

    if use_oga:
        base_parts.append("--ogarla.70")

    if ar:
        base_parts.append(f"--ar {ar}")

    # Determine batch count from sref mode
    batch_count = 0
    if isinstance(use_sref_rand, str):
        if use_sref_rand in ["sref", "sref1"]:
            batch_count = 1
        elif use_sref_rand == "sref5":
            batch_count = 5
        elif use_sref_rand == "sref10":
            batch_count = 10
        elif use_sref_rand == "sref15":
            batch_count = 15
        elif use_sref_rand.startswith("sref") and use_sref_rand[4:].isdigit():
            batch_count = int(use_sref_rand[4:])
        elif use_sref_rand.lower() in ["true", "1", "on"]:
            batch_count = 1
    elif use_sref_rand is True:
        batch_count = 1

    if batch_count <= 1:
        prompt_parts = list(base_parts)
        if batch_count == 1:
            prompt_parts.append("--sref random")
        full_prompt = " ".join(prompt_parts)
        await execute_blend_generation(interaction, uploaded_image_name, prompt=full_prompt, model=selected_model, comp_strength=comp_strength)
    else:
        favorites = db.get_favorite_styles(interaction.user.id)
        style_codes = []
        fav_count = 0
        if favorites:
            sample_size = min(len(favorites), batch_count)
            chosen_favs = random.sample(favorites, k=sample_size)
            for fav in chosen_favs:
                style_codes.append(str(fav["style_code"]))
                fav_count += 1
        needed_random = batch_count - len(style_codes)
        for _ in range(needed_random):
            style_codes.append(str(random.randint(100000, 999999)))

        await interaction.followup.send(
            f"🚀 **Batch Queuing {batch_count} Style Generations** ({fav_count} sampled from your {len(favorites)} saved `/styles` + {needed_random} random style codes)..."
        )

        for code in style_codes:
            prompt_parts = list(base_parts)
            prompt_parts.append(f"--sref {code}")
            full_prompt = " ".join(prompt_parts)
            await execute_blend_generation(interaction, uploaded_image_name, prompt=full_prompt, model=selected_model, comp_strength=comp_strength)





async def execute_blend_generation(interaction: discord.Interaction, uploaded_image_name: str, prompt: str, negative_prompt: str = None, model: str = None, comp_strength: str = "style"):
    neg_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT
    selected_model = model or COMFYUI_CHECKPOINT
    generation_id = str(random.randint(100000, 999999))

    # --- Parse all flags from prompt string ---
    cleaned_prompt, magic_flag = parse_magic_prompt(prompt)
    is_magic = magic_flag
    
    cleaned_prompt, user_seed = parse_seed(cleaned_prompt)
    seed = user_seed if user_seed is not None else random.randint(1, 1125899906842624)
    
    cleaned_prompt, cfg, prepend_quality = parse_stylize(cleaned_prompt)
    cleaned_prompt, sref_url, sref_weight, sref_info = parse_sref(cleaned_prompt)
    cleaned_prompt, cref_url, cref_weight = parse_cref(cleaned_prompt)
    # Handle character reference URL if provided in prompt
    cref_image_name = None
    if cref_url:
        try:
            cref_bytes = await download_image(cref_url)
            if not cref_bytes or len(cref_bytes) < 100:
                raise ValueError("Downloaded image payload is empty or blank.")
            upload_result = await comfy_client.upload_image(cref_bytes, "cref_from_url.png")
            cref_image_name = upload_result.get("name")
            logger.info(f"Character reference uploaded from URL for blend: {cref_image_name}")
        except Exception as e:
            logger.warning(f"Failed to fetch character reference URL '{cref_url}' for blend: {e}. Falling back to Ogarla LoRA (--ogarla.75)!")
            cref_image_name = None
            if "ogarla" not in cleaned_prompt.lower() and "oga" not in cleaned_prompt.lower():
                cleaned_prompt = f"ogarla, {cleaned_prompt} --ogarla.75"

    cleaned_prompt, loras = parse_loras(cleaned_prompt)
    cleaned_prompt, width, height = parse_aspect_ratio(cleaned_prompt, selected_model)

    if prepend_quality:
        cleaned_prompt = f"masterpiece, best quality, absurdres. {cleaned_prompt}"

    # Load appropriate workflow template depending on composition reference mode
    use_img2img = (comp_strength != "style")
    workflow_path = "workflows/img2img_lowres.json" if use_img2img else "workflows/blend_lowres.json"

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow template for blend: {e}")
        await send_followup_fallback(interaction, content="Failed to load workflow template.")
        return

    # Chain character reference IP-Adapter if provided
    if cref_image_name:
        workflow = apply_ipadapter_to_workflow(workflow, cref_image_name, weight=cref_weight, node_prefix="cref")

    # If running in img2img mode, configure the input image and denoise parameter
    final_image_name = uploaded_image_name
    if use_img2img:
        denoise_map = {"low": 0.85, "med": 0.70, "high": 0.55}
        denoise_val = denoise_map.get(comp_strength, 0.70)
        
        # Download, crop/resize, and re-upload the image to match the target aspect ratio
        try:
            view_url = f"http://{COMFYUI_ADDRESS}/view?filename={uploaded_image_name}&type=input"
            async with aiohttp.ClientSession() as session:
                async with session.get(view_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"ComfyUI view returned status {resp.status}")
                    orig_image_bytes = await resp.read()
            
            cropped_bytes = crop_to_aspect_ratio(orig_image_bytes, width, height)
            target_filename = f"blend_crop_{generation_id}_{width}_{height}.png"
            upload_res = await comfy_client.upload_image(cropped_bytes, target_filename)
            final_image_name = upload_res.get("name", uploaded_image_name)
        except Exception as crop_err:
            logger.error(f"Failed to crop/resize input image for blend aspect ratio: {crop_err}")
            final_image_name = uploaded_image_name # Fallback

        try:
            workflow["3"]["inputs"]["denoise"] = denoise_val
            workflow["30"]["inputs"]["image"] = final_image_name
        except KeyError as e:
            logger.error(f"Invalid img2img workflow node structure: {e}")
            await send_followup_fallback(interaction, content="Workflow template structure mismatch.")
            return

    # Build dynamic IPAdapter workflow (chains models through LoRAs -> IPAdapter)
    try:
        workflow = build_blend_workflow([final_image_name], cleaned_prompt, neg_prompt, selected_model, width, height, seed, cfg, workflow_template=workflow)
        sref_suffix = f"_sref{sref_info['code']}" if sref_info and "code" in sref_info else ""
        workflow["9"]["class_type"] = "PreviewImage"
        workflow["9"]["inputs"].pop("filename_prefix", None)
    except Exception as e:
        logger.error(f"Error building blend workflow: {e}")
        await send_followup_fallback(interaction, content="Failed to build blend workflow.")
        return

    workflow = apply_loras_to_workflow(workflow, loras)

    display_prompt = prompt
    if sref_info and "code" in sref_info:
        display_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+random', f"--sref {sref_info['code']}", prompt, flags=re.IGNORECASE)

    active_generations[generation_id] = {
        "prompt": cleaned_prompt,
        "original_prompt": display_prompt,
        "negative_prompt": neg_prompt,
        "seed": seed,
        "width": width,
        "height": height,
        "loras": loras,
        "checkpoint": selected_model,
        "cfg": cfg,
        "variation_depth": 0,
        "sref_info": sref_info,
        "cref_image": cref_image_name,
        "cref_weight": cref_weight,
        "is_blend": True
    }
    save_generations()

    comp_lbls = {"style": "Style Only", "low": "Low Comp", "med": "Medium Comp", "high": "High Comp"}
    comp_info = comp_lbls.get(comp_strength, "Style Only")
    flags_info = f"Seed: {seed}, Model: {selected_model}, Size: {width}x{height}, CFG: {cfg:.1f}, Comp: {comp_info}"
    
    try:
        await send_followup_fallback(interaction, content=f"Job submitted (Seed: {seed}) — Queuing blend...")

        # Set batch size to 1 for the parallel worker tasks (if Node 5 exists)
        if "5" in workflow:
            workflow["5"]["inputs"]["batch_size"] = 1
        
        tasks = []
        expanded_prompts = []
        for i in range(4):
            wf_copy = copy.deepcopy(workflow)
            q_seed = seed + i
            q_rng = random.Random(q_seed)
            q_prompt = expand_dynamic_prompt(cleaned_prompt, q_rng)
            if is_magic:
                q_prompt = apply_magic_enhancement(q_prompt, q_seed)
            
            expanded_prompts.append(q_prompt)
            wf_copy["3"]["inputs"]["seed"] = q_seed
            wf_copy["6"]["inputs"]["text"] = q_prompt
            tasks.append(comfy_client.generate(wf_copy))
            
        results = await asyncio.gather(*tasks)
        images = [r[0] for r in results]
        
        if len(images) < 4:
            await send_followup_fallback(interaction, content=f"Expected 4 images from blend generation, but only got {len(images)}.")
            return

        save_quadrant_images(generation_id, images)

        grid_file_io = await asyncio.to_thread(create_grid, images, cleaned_prompt, neg_prompt, seed, width, height)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        file = discord.File(fp=grid_file_io, filename=format_image_filename("blend_grid", seed, "jpg", sref=sref_code))
        
        desc_parts = [f"**Steering Prompt:** {truncate_prompt(display_prompt, 250)}", f"**Model:** {selected_model}", f"**Seed:** {seed}", f"**Size:** {width}x{height}", f"**Composition Reference:** {comp_info}"]
        if "{" in display_prompt and "}" in display_prompt:
            desc_parts.append("\n**Selected Quadrant Prompts:**")
            cleaned_eps = clean_quadrant_prompts(expanded_prompts, display_prompt)
            for idx, clean_ep in enumerate(cleaned_eps):
                if len(clean_ep) > 120:
                    clean_ep = clean_ep[:117] + "..."
                desc_parts.append(f"* **Q{idx+1}:** {clean_ep}")
        if sref_info and "code" in sref_info:
            desc_parts.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image_name:
            desc_parts.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")
        if cfg != 4.5:
            desc_parts.append(f"**CFG:** {cfg:.1f}")
        if is_magic:
            desc_parts.append("**Magic Prompt:** ✨ Enabled")
        if not prepend_quality:
            desc_parts.append("**Mode:** Raw")
        
        embed = discord.Embed(
            title="Image Blend Complete", 
            description="\n".join(desc_parts)
        )
        has_sref = sref_info is not None and "code" in sref_info
        view = GridButtons(generation_id, has_sref=has_sref)
        
        await send_followup_fallback(interaction, content=f"**Blend:** {truncate_prompt(display_prompt, 100)}", embed=embed, file=file, view=view)
    except Exception as e:
        error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="execute_blend_generation",
            source_file="bot.py",
            severity=ErrorSeverity.ERROR,
            context={"uploaded_image_name": uploaded_image_name, "prompt": prompt, "model": selected_model, "width": width, "height": height}
        )
        logger.error(f"Error executing blend generation: {e}")
        await send_error_fallback(interaction, f"An error occurred while generating blend images: {e}")






async def run_study_imagine_callback(interaction: discord.Interaction, prompt: str):
    await safe_defer(interaction, thinking=True)
    await execute_imagine(interaction, prompt)


@bot.tree.command(name="study", description="Extract the positive prompt embedded in an uploaded image.")
@app_commands.describe(
    image="The image file (PNG/JPG) you want to extract the prompt from"
)
async def study(interaction: discord.Interaction, image: discord.Attachment):
    """Extracts the positive prompt used to make a previous image from PNG metadata."""
    await safe_defer(interaction, thinking=True)

    if not image.content_type or not image.content_type.startswith("image/"):
        await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)
        return

    try:
        image_bytes = await image.read()
        prompt = extract_positive_prompt(image_bytes)

        if not prompt or prompt == "NOT FOUND":
            await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)
            return

        if len(prompt) > 3900:
            formatted_prompt = prompt[:3900] + "..."
        else:
            formatted_prompt = prompt

        embed = discord.Embed(
            title="🔍 Extracted Positive Prompt",
            description=f"```\n{formatted_prompt}\n```",
            color=discord.Color.blue()
        )
        embed.set_footer(text=f"Extracted from {image.filename}")

        view = StudyButtons(prompt=prompt, imagine_callback=run_study_imagine_callback)

        await send_followup_fallback(interaction, embed=embed, view=view, ephemeral=False)
    except Exception as e:
        logger.error(f"Error in /study command: {e}")
        await send_followup_fallback(interaction, content="NOT FOUND", ephemeral=False)


def parse_adopted_post(message: discord.Message) -> dict:
    """Parses any Discord message (Midjourney, ComfyUI, user upload, etc.) into clean prompt and image details."""
    raw_content = message.content or ""
    
    if message.embeds:
        for emb in message.embeds:
            if emb.description:
                raw_content += "\n" + emb.description
            if emb.fields:
                for f in emb.fields:
                    if f.name.lower() in ["prompt", "clean prompt", "imagine", "description"]:
                        raw_content += "\n" + f.value
            elif emb.title and not raw_content:
                raw_content += "\n" + emb.title

    # Extract prompt if present
    prompt_field_match = re.search(r'(?:Prompt|Imagine)[:\s]+```(?:\w+)?\n?(.*?)```', raw_content, re.DOTALL | re.IGNORECASE)
    if prompt_field_match:
        extracted = prompt_field_match.group(1).strip()
    else:
        prompt_field_match2 = re.search(r'(?:Prompt|Imagine)[:\s]+([^\n]+)', raw_content, re.IGNORECASE)
        if prompt_field_match2:
            extracted = prompt_field_match2.group(1).strip()
        else:
            prompt_match = re.search(r'\*\*(.*?)\*\*', raw_content, re.DOTALL)
            if prompt_match:
                extracted = prompt_match.group(1).strip()
            else:
                extracted = raw_content.strip()

    clean_p = re.sub(r'\s*-\s*(?:Variations|Upscaled|Image|Remix|Pan|Zoom|Vary).*$', '', extracted, flags=re.IGNORECASE)
    clean_p = re.sub(r'\s*-\s*@.*$', '', clean_p)
    clean_p = re.sub(r'\s*by\s*<@!?\d+>.*$', '', clean_p, flags=re.IGNORECASE)
    clean_p = re.sub(r'\s*by\s*@[^\s]+.*$', '', clean_p, flags=re.IGNORECASE)
    clean_p = clean_p.strip()

    clean_p = clean_midjourney_flags(clean_p)

    image_url = None
    if message.attachments:
        for att in message.attachments:
            if att.content_type and att.content_type.startswith("image/") or att.url.split('?')[0].lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_url = att.url
                break
        if not image_url and message.attachments:
            image_url = message.attachments[0].url
    elif message.embeds:
        for emb in message.embeds:
            if emb.image and emb.image.url:
                image_url = emb.image.url
                break

    author_match = re.search(r'by\s+(<@!?\d+>|@[^\s]+)', raw_content, re.IGNORECASE)
    if author_match:
        author_str = author_match.group(1)
    else:
        author_str = f"<@{message.author.id}>"

    return {
        "raw_content": raw_content,
        "clean_prompt": clean_p or raw_content or "No prompt text found",
        "image_url": image_url,
        "author_str": author_str,
        "jump_url": message.jump_url
    }

parse_midjourney_post = parse_adopted_post


async def execute_adopt_post(interaction: discord.Interaction, message: discord.Message):
    """Core logic to adopt any Discord post or image (Midjourney, ComfyUI, user upload)."""
    await safe_defer(interaction, thinking=True)
    try:
        parsed = parse_adopted_post(message)
        adopt_id = f"adopt_{message.id}_{random.randint(1000, 9999)}"

        parsed_prompt = parsed["clean_prompt"]
        image_url = parsed["image_url"]

        florence_prompt = None
        if image_url:
            logger.info("Running Florence-2 vision model to generate SDXL prompt for adopted image...")
            florence_prompt = await run_florence_interrogate(image_url)

        if florence_prompt:
            ar_match = re.search(r'--ar\s+\d+:\d+', parsed_prompt, flags=re.IGNORECASE)
            ar_flag = f" {ar_match.group(0)}" if ar_match else ""
            final_prompt = f"{florence_prompt}{ar_flag}"
            logger.info(f"Florence-2 generated SDXL prompt: {final_prompt}")
        else:
            final_prompt = parsed_prompt

        _, _, parsed_cw = parse_cref(parsed_prompt)
        initial_cw = parsed_cw if parsed_cw is not None else 0.20
        initial_oga = "ogarla" in parsed_prompt.lower() or "oga" in parsed_prompt.lower()
        initial_sr_has = "semi-realism" in parsed_prompt.lower() or "hyper-realistic" in parsed_prompt.lower() or "octane render" in parsed_prompt.lower()
        initial_sr_w = 0.85 if initial_sr_has else 0.0
        initial_rnd = "--sref" in parsed_prompt.lower()

        db.save_generation(adopt_id, {
            "prompt": final_prompt,
            "original_prompt": parsed_prompt,
            "image_url": image_url,
            "author_str": parsed["author_str"],
            "jump_url": parsed["jump_url"],
            "message_id": str(message.id),
            "channel_id": str(message.channel.id),
            "cref_weight": initial_cw,
            "ogarla": initial_oga,
            "semi_realism": initial_sr_has,
            "semi_realism_weight": initial_sr_w,
            "random_sref": initial_rnd
        })

        is_mj = "midjourney" in str(parsed["author_str"]).lower() or "midjourney" in str(message.author.name).lower()
        title_str = "⛵ Adopted Midjourney Post" if is_mj else "⛵ Adopted Post / Image"

        embed = discord.Embed(
            title=title_str,
            description=f"```\n{final_prompt}\n```\n"
                        f"**Original Author:** {parsed['author_str']}\n"
                        f"**Source Message:** [Jump to Message]({parsed['jump_url']})",
            color=discord.Color.from_rgb(0, 168, 252)
        )

        has_image = bool(image_url)
        if has_image:
            embed.set_image(url=image_url)
            embed.set_footer(text="Florence-2 SDXL description generated! Use controls below to customize or generate.")
        else:
            embed.set_footer(text="Prompt extracted! Use the buttons below to generate or save.")

        view = AdoptButtons(adopt_id=adopt_id, ogarla_on=initial_oga, cref_weight=initial_cw, semi_realism_weight=initial_sr_w, random_sref_on=initial_rnd)
        await send_followup_fallback(interaction, embed=embed, view=view, ephemeral=False)
    except Exception as e:
        logger.error(f"Error adopting post: {e}")
        await send_followup_fallback(interaction, content=f"⚠️ Failed to parse message: {e}", ephemeral=True)


@bot.tree.context_menu(name="Adopt Post / Image")
async def adopt_post_context(interaction: discord.Interaction, message: discord.Message):
    """Context menu command to adopt any Discord image or post (ComfyUI, User Upload, Midjourney)."""
    await execute_adopt_post(interaction, message)


@bot.tree.context_menu(name="Adopt Midjourney Post")
async def adopt_midjourney_post(interaction: discord.Interaction, message: discord.Message):
    """Context menu command to adopt and take control of old Midjourney posts."""
    await execute_adopt_post(interaction, message)


async def handle_submit_edit_adopt_prompt(interaction: discord.Interaction, adopt_id: str, new_prompt: str):
    """Updates the prompt of an adopted post and edits the embed description in place."""
    data = db.get_generation(adopt_id)
    if not data:
        await interaction.response.send_message("⚠️ Adopted post session expired.", ephemeral=True)
        return

    data["prompt"] = new_prompt
    db.save_generation(adopt_id, data)

    author_str = data.get("author_str", "@Midjourney Bot")
    jump_url = data.get("jump_url", "")
    image_url = data.get("image_url")
    oga_on = data.get("ogarla", False)
    cw = data.get("cref_weight", 0.20)
    sr_w = float(data.get("semi_realism_weight", 0.85 if data.get("semi_realism") else 0.0))
    rnd_on = data.get("random_sref", False)

    embed = discord.Embed(
        title="⛵ Adopted Midjourney Post (Edited)",
        description=f"```\n{new_prompt}\n```\n"
                    f"**Original Author:** {author_str}\n"
                    f"**Source Message:** [Jump to Message]({jump_url})",
        color=discord.Color.from_rgb(0, 168, 252)
    )

    if image_url:
        embed.set_image(url=image_url)
        embed.set_footer(text="Original image captured as reference! Use controls below to toggle Ogarla or adjust Ref Weight.")
    else:
        embed.set_footer(text="Prompt updated! Use the buttons below to generate or save.")

    view = AdoptButtons(adopt_id=adopt_id, ogarla_on=oga_on, cref_weight=cw, semi_realism_weight=sr_w, random_sref_on=rnd_on)
    await interaction.response.edit_message(embed=embed, view=view)


async def run_florence_interrogate(image_url: str) -> str:
    """Downloads an image from URL and uses Florence-2 model to generate an SDXL prompt description."""
    if not image_url:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return None
                image_bytes = await resp.read()
        
        filename = f"florence_adopt_{random.randint(100000, 999999)}.png"
        upload_result = await comfy_client.upload_image(image_bytes, filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            return None
        
        # Clean 3-node Florence-2 workflow without fragile custom text nodes
        workflow = {
            "1": {
                "inputs": {"image": uploaded_name},
                "class_type": "LoadImage",
                "_meta": {"title": "Load Image"}
            },
            "2": {
                "inputs": {
                    "model": "MiaoshouAI/Florence-2-large-PromptGen-v2.0",
                    "precision": "fp16",
                    "convert_to_safetensors": True
                },
                "class_type": "DownloadAndLoadFlorence2Model",
                "_meta": {"title": "Load Florence2 Model"}
            },
            "3": {
                "inputs": {
                    "text_input": "",
                    "task": "detailed_caption",
                    "fill_mask": True,
                    "keep_model_loaded": False,
                    "max_new_tokens": 250,
                    "num_beams": 3,
                    "do_sample": False,
                    "output_mask_select": "",
                    "seed": random.randint(100000, 999999),
                    "image": ["1", 0],
                    "florence2_model": ["2", 0]
                },
                "class_type": "Florence2Run",
                "_meta": {"title": "Florence2Run"}
            },
            "4": {
                "inputs": {
                    "text": ["3", 2]
                },
                "class_type": "ShowText|pysssss",
                "_meta": {"title": "Show Text"}
            }
        }
        
        results = await comfy_client.generate(workflow, timeout=14400)
        
        caption = ""
        if isinstance(results, dict):
            for node_id in ["3", "4", "10", "9"]:
                if node_id in results:
                    n_data = results[node_id]
                    if isinstance(n_data, dict):
                        for k in ["text", "string", "caption"]:
                            if k in n_data and n_data[k]:
                                val = n_data[k]
                                caption = val[0] if isinstance(val, list) else str(val)
                                if caption:
                                    break
                    elif isinstance(n_data, list) and n_data:
                        caption = str(n_data[0])
                if caption:
                    break
        
        return caption.strip() if caption else None
    except Exception as e:
        logger.error(f"Error running Florence-2 interrogate for adopted post: {e}")
        return None


@bot.tree.command(name="describe", description="Generate captions and detailed descriptions for an image using Florence-2.")
@app_commands.describe(
    image="The image file you want to describe"
)
async def describe(interaction: discord.Interaction, image: discord.Attachment):
    # Send immediate response so the user knows it's queued and we avoid the generic Discord spinner
    await interaction.response.send_message("Analyzing image with Florence-2...", ephemeral=False)
    
    # Check if attachment is an image
    if not image.content_type or not image.content_type.startswith("image/"):
        await edit_original_fallback(interaction, content="❌ Please upload a valid image file (PNG/JPG).")
        return
        
    try:
        # Download image from Discord
        image_bytes = await image.read()
        
        # Upload image to ComfyUI
        logger.info(f"Uploading image {image.filename} to ComfyUI for description...")
        upload_result = await comfy_client.upload_image(image_bytes, image.filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await edit_original_fallback(interaction, content="❌ Failed to upload the image to ComfyUI server.")
            return
            
        logger.info(f"Image uploaded successfully. ComfyUI filename: {uploaded_name}")
        
        # Load description workflow
        workflow_path = "workflows/DESCRIBE_cuibot.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading description workflow file: {e}")
            await edit_original_fallback(interaction, content="❌ Failed to load description workflow template.")
            return
            
        # Configure description workflow parameters
        # Node "1" is LoadImage
        workflow["1"]["inputs"]["image"] = uploaded_name
        
        # Run workflow
        logger.info(f"Executing description workflow for {uploaded_name}...")
        results = await comfy_client.generate(workflow, timeout=14400)
        
        # Extract text outputs from results
        # Node "9" has standard caption (title: caption)
        # Node "10" has detailed caption (title: detailed_caption)
        caption = "No caption generated"
        detailed_caption = "No detailed caption generated"
        
        if isinstance(results, dict):
            # Extract from Node "9"
            if "9" in results and "text" in results["9"]:
                text_list = results["9"]["text"]
                if text_list:
                    caption = text_list[0]
            # Extract from Node "10"
            if "10" in results and "text" in results["10"]:
                text_list = results["10"]["text"]
                if text_list:
                    detailed_caption = text_list[0]
        else:
            await edit_original_fallback(interaction, content="❌ ComfyUI did not return the expected text description outputs.")
            return
            
        raw_caption = caption
        raw_detailed_caption = detailed_caption

        # Truncate descriptions to fit within Discord's 1024-character limit for embed fields
        if len(caption) > 1024:
            caption = caption[:1021] + "..."
        if len(detailed_caption) > 1024:
            detailed_caption = detailed_caption[:1021] + "..."

        # Store in active generations cache for interactive button clicks
        generation_id = str(random.randint(100000, 999999))
        active_generations[generation_id] = {
            "caption": raw_caption,
            "detailed_caption": raw_detailed_caption
        }
        save_generations()

        # Build embed response
        embed = discord.Embed(
            title="Image Description Complete",
            color=discord.Color.blue()
        )
        embed.set_thumbnail(url=image.url)
        embed.add_field(name="📝 Caption", value=caption, inline=False)
        embed.add_field(name="🔍 Detailed Description", value=detailed_caption, inline=False)
        embed.set_footer(text=f"Analyzed using Florence-2 model • Requested by {interaction.user.name}")
        
        view = DescribeButtons(generation_id, ar="16:9")
        await edit_original_fallback(interaction, content=None, embed=embed, view=view)

    except Exception as e:
        logger.error(f"Error executing describe command: {e}")
        await edit_original_fallback(interaction, content=f"❌ An error occurred while describing the image: {e}")




def build_blend_workflow(image_filenames: list, prompt: str, neg_prompt: str, selected_model: str, width: int, height: int, seed: int, cfg: float, workflow_template: dict = None):
    """Dynamically constructs a ComfyUI workflow that chains IP-Adapter for 1 to 5 images with smart composition scaling."""
    if workflow_template is not None:
        workflow = copy.deepcopy(workflow_template)
    else:
        workflow_path = "workflows/blend_lowres.json"
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)

    workflow["4"]["inputs"]["ckpt_name"] = selected_model
    if "5" in workflow:
        workflow["5"]["inputs"]["width"] = width
        workflow["5"]["inputs"]["height"] = height
        workflow["5"]["inputs"]["batch_size"] = 1
    workflow["3"]["inputs"]["seed"] = seed
    
    # Calibrate CFG (default to 3.5 if uncalibrated or high to avoid frying multi-conditioning)
    effective_cfg = cfg if (cfg is not None and 1.0 <= cfg <= 6.0) else 3.5
    workflow["3"]["inputs"]["cfg"] = effective_cfg
    workflow["6"]["inputs"]["text"] = prompt
    workflow["9"]["class_type"] = "PreviewImage"
    workflow["9"]["inputs"].pop("filename_prefix", None)

    # Detect if workflow is running in img2img mode (already performing latent composition)
    is_img2img = ("30" in workflow or "31" in workflow)

    # Get current model source (e.g. Checkpoint or final LoRA in chain)
    current_model_source = workflow["3"]["inputs"].get("model", ["4", 0])

    # Unified Loader node to load installed IPAdapter Plus models
    workflow["20"] = {
        "inputs": {
            "model": current_model_source,
            "preset": "PLUS (high strength)"
        },
        "class_type": "IPAdapterUnifiedLoader",
        "_meta": {
            "title": "IPAdapter Unified Loader"
        }
    }

    # Dynamically scale weight and end_at based on mode and number of images
    # If img2img is already providing composition latents, lower IP weight & stop at 0.75
    # so the final denoising steps cleanly polish skin, eyes, and details
    num_images = len(image_filenames)
    if is_img2img:
        ip_weight = 0.35 if num_images == 1 else round(min(0.25, 0.40 / num_images), 2)
        end_at = 0.75
        weight_type = "ease in-out"
    else:
        ip_weight = 0.55 if num_images == 1 else round(min(0.35, 0.60 / num_images), 2)
        end_at = 0.85
        weight_type = "linear"

    # Enforce quality negative prompt, but prevent contradictory negative terms
    # (e.g. don't ban "sketch / line art" if the user or SREF style explicitly asked for sketch/pencil/drawing)
    neg_base = neg_prompt or DEFAULT_NEGATIVE_PROMPT
    p_lower = prompt.lower()
    
    extra_neg_parts = ["overexposed", "pale", "washed out", "faded colors", "bloom", "white out"]
    sketch_keywords = ["sketch", "pencil", "drawing", "line art", "lineart", "ink", "monochrome", "charcoal", "cross-hatching", "hatching", "woodcut"]
    if not any(k in p_lower for k in sketch_keywords):
        extra_neg_parts.append("line art only, sketch")
        
    enhanced_neg = neg_base + ", " + ", ".join(extra_neg_parts)
    workflow["7"]["inputs"]["text"] = enhanced_neg

    prev_model_node = ["20", 0]

    for idx, img_name in enumerate(image_filenames):
        load_node_id = f"blend_img_{idx}"
        ip_node_id = f"blend_ip_{idx}"

        workflow[load_node_id] = {
            "inputs": {
                "image": img_name,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {
                "title": f"Load Blend Image {idx + 1}"
            }
        }

        workflow[ip_node_id] = {
            "inputs": {
                "model": prev_model_node,
                "ipadapter": ["20", 1],
                "image": [load_node_id, 0],
                "weight": ip_weight,
                "weight_type": weight_type,
                "combine_embeds": "average",
                "start_at": 0.0,
                "end_at": end_at,
                "embeds_scaling": "K+V"
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {
                "title": f"IPAdapter Blend {idx + 1}"
            }
        }

        prev_model_node = [ip_node_id, 0]

    workflow["3"]["inputs"]["model"] = prev_model_node
    return workflow


async def execute_blend_core(
    interaction: discord.Interaction, 
    image_bytes: bytes, 
    filename: str, 
    image_url: str, 
    prompt: str = None,
    style: str = None,
    secondary_style: str = None
):
    """Core execution logic for Florence-2 image blending and remixing."""
    # Process locked style preset if selected
    applied_style_name = None
    if style:
        scapes_info = build_scapes_prompt(
            user_prompt=prompt or "",
            style=style,
            secondary_style=secondary_style,
            mode=None,
            subject_type="scenery"
        )
        prompt = scapes_info["final_prompt"]
        applied_style_name = scapes_info["style_name"]

    try:
        # Upload image to ComfyUI
        safe_filename = filename or f"blend_{random.randint(100000, 999999)}.png"
        logger.info(f"Uploading image {safe_filename} to ComfyUI for blend description...")
        upload_result = await comfy_client.upload_image(image_bytes, safe_filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await edit_original_fallback(interaction, content="❌ Failed to upload the image to ComfyUI server.")
            return
            
        logger.info(f"Image uploaded successfully. ComfyUI filename: {uploaded_name}")
        
        # Load description workflow
        workflow_path = "workflows/DESCRIBE_cuibot.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading description workflow file: {e}")
            await edit_original_fallback(interaction, content="❌ Failed to load description workflow template.")
            return
            
        # Configure description workflow parameters
        # Node "1" is LoadImage
        workflow["1"]["inputs"]["image"] = uploaded_name
        
        # Run workflow
        logger.info(f"Executing description workflow for {uploaded_name}...")
        results = await comfy_client.generate(workflow, timeout=14400)
        
        # Extract text outputs from results
        caption = "No caption generated"
        detailed_caption = "No detailed caption generated"
        
        if isinstance(results, dict):
            if "9" in results and "text" in results["9"]:
                text_list = results["9"]["text"]
                if text_list:
                    caption = text_list[0]
            if "10" in results and "text" in results["10"]:
                text_list = results["10"]["text"]
                if text_list:
                    detailed_caption = text_list[0]
        else:
            await edit_original_fallback(interaction, content="❌ ComfyUI did not return the expected text description outputs.")
            return
            
        raw_caption = caption
        raw_detailed_caption = detailed_caption

        # Truncate descriptions to fit within Discord's 1024-character limit for embed fields
        if len(caption) > 1024:
            caption = caption[:1021] + "..."
        if len(detailed_caption) > 1024:
            detailed_caption = detailed_caption[:1021] + "..."

        # Store in active generations cache for interactive button clicks
        generation_id = str(random.randint(100000, 999999))
        gen_data = {
            "caption": raw_caption,
            "detailed_caption": raw_detailed_caption,
            "extra_details": prompt or "",
            "uploaded_image_name": uploaded_name,
            "image_url": image_url,
            "user_prompt": prompt or "",
            "ar": "16:9",
            "sr": True,
            "oga": False,
            "model_choice": "wai",
            "comp_strength": "style",
            "sref_rand": "nosref",
            "author_str": interaction.user.name
        }
        active_generations[generation_id] = gen_data
        db.save_generation(generation_id, gen_data)
        save_generations()

        # Build streamlined embed response
        embed = build_blend_embed(gen_data, author_str=interaction.user.name, image_url=image_url)
        view = BlendButtons(
            generation_id=generation_id,
            ar="16:9",
            sr=True,
            oga=False,
            model_choice="wai",
            comp_strength="style",
            sref_rand="nosref"
        )
        await edit_original_fallback(interaction, content=None, embed=embed, view=view)

    except Exception as e:
        logger.error(f"Error executing blend workflow: {e}")
        await edit_original_fallback(interaction, content=f"❌ An error occurred: {e}")


async def execute_blend_message(interaction: discord.Interaction, message: discord.Message):
    """Core logic to extract an image from any Discord message and initiate the interactive blend workflow."""
    await safe_defer(interaction, thinking=False, ephemeral=False)
    await edit_original_fallback(interaction, content="🔍 Inspecting message for image to blend...")

    image_url = None
    image_bytes = None
    filename = None

    # 1. Check message attachments
    if message.attachments:
        for att in message.attachments:
            if (att.content_type and att.content_type.startswith("image/")) or att.filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                image_url = att.url
                filename = att.filename
                try:
                    image_bytes = await att.read()
                except Exception as e:
                    logger.warning(f"Failed to read attachment directly: {e}")
                break
        if not image_url and message.attachments:
            image_url = message.attachments[0].url
            filename = message.attachments[0].filename
            try:
                image_bytes = await message.attachments[0].read()
            except Exception as e:
                logger.warning(f"Failed to read attachment directly: {e}")

    # 2. Check message embeds if no attachment image found
    if not image_url and message.embeds:
        for emb in message.embeds:
            if emb.image and emb.image.url:
                image_url = emb.image.url
                filename = f"blend_{message.id}.png"
                break
            elif emb.thumbnail and emb.thumbnail.url:
                image_url = emb.thumbnail.url
                filename = f"blend_{message.id}.png"
                break

    if not image_url:
        await edit_original_fallback(interaction, content="❌ No valid image found on that message to blend.")
        return

    try:
        if not image_bytes:
            async with aiohttp.ClientSession() as session:
                async with session.get(image_url) as resp:
                    if resp.status != 200:
                        await edit_original_fallback(interaction, content="❌ Failed to download the image from message.")
                        return
                    image_bytes = await resp.read()

        parsed = parse_adopted_post(message)
        initial_prompt = parsed.get("clean_prompt") if parsed and parsed.get("clean_prompt") != "No prompt text found" else None

        await execute_blend_core(
            interaction=interaction,
            image_bytes=image_bytes,
            filename=filename or f"blend_{message.id}.png",
            image_url=image_url,
            prompt=initial_prompt
        )
    except Exception as e:
        logger.error(f"Error in blend context menu: {e}")
        await edit_original_fallback(interaction, content=f"❌ Failed to process blend for message: {e}")


@bot.tree.context_menu(name="Blend Image")
async def blend_image_context(interaction: discord.Interaction, message: discord.Message):
    """Context menu command to run /blend on any right-clicked message containing an image."""
    await execute_blend_message(interaction, message)


@bot.tree.command(name="blend", description="Interactively blend or remix an image using Florence-2 description and custom styles.")
@app_commands.describe(
    image="Upload an image to blend and remix",
    prompt="Optional extra subject or scene details to add to the blended image",
    style="Primary locked artistic style preset to apply",
    secondary_style="Optional secondary locked style preset to blend with primary style"
)
@app_commands.choices(
    style=[
        app_commands.Choice(name="Junji Ito (Horror Manga Ink)", value="junji_ito"),
        app_commands.Choice(name="Martine Johanna (Pastel Surreal)", value="martine_johanna"),
        app_commands.Choice(name="Dark Fantasy Landscape", value="dark_fantasy_landscape"),
        app_commands.Choice(name="Cyberpunk Cityscape", value="cyberpunk_cityscape"),
        app_commands.Choice(name="Ethereal Fine Art Portrait", value="ethereal_portrait"),
    ],
    secondary_style=[
        app_commands.Choice(name="Junji Ito (Horror Manga Ink)", value="junji_ito"),
        app_commands.Choice(name="Martine Johanna (Pastel Surreal)", value="martine_johanna"),
        app_commands.Choice(name="Dark Fantasy Landscape", value="dark_fantasy_landscape"),
        app_commands.Choice(name="Cyberpunk Cityscape", value="cyberpunk_cityscape"),
        app_commands.Choice(name="Ethereal Fine Art Portrait", value="ethereal_portrait"),
    ]
)
async def blend(
    interaction: discord.Interaction, 
    image: discord.Attachment, 
    prompt: str = None,
    style: str = None,
    secondary_style: str = None
):
    # Safely defer interaction immediately so Discord doesn't timeout if image upload/network takes time
    await safe_defer(interaction, thinking=False, ephemeral=False)
    await edit_original_fallback(interaction, content="Analyzing image with Florence-2 for blending...")
    
    # Check if attachment is an image
    if not image.content_type or not image.content_type.startswith("image/"):
        await edit_original_fallback(interaction, content="❌ Please upload a valid image file (PNG/JPG).")
        return

    try:
        image_bytes = await image.read()
        await execute_blend_core(
            interaction=interaction,
            image_bytes=image_bytes,
            filename=image.filename,
            image_url=image.url,
            prompt=prompt,
            style=style,
            secondary_style=secondary_style
        )
    except Exception as e:
        logger.error(f"Error reading image for blend: {e}")
        await edit_original_fallback(interaction, content=f"❌ Failed to read uploaded image: {e}")



@bot.tree.command(name="queue", description="Show the current ComfyUI processing queue and system status.")
async def queue_command(interaction: discord.Interaction):
    """Display ComfyUI queue status, VRAM usage, and job details."""
    await safe_defer(interaction, ephemeral=True)

    queue = await fetch_comfyui_queue()
    stats = await fetch_comfyui_system_stats()

    if queue is None:
        embed = discord.Embed(
            title="⚫ ComfyUI Status",
            description="Could not connect to ComfyUI server.",
            color=discord.Color.dark_grey()
        )
        await interaction.followup.send(embed=embed, ephemeral=True)
        return

    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])

    # Determine embed color based on state
    if len(running) > 0:
        color = discord.Color.gold()  # Active processing
        status_emoji = "🟡"
    elif len(pending) > 0:
        color = discord.Color.orange()
        status_emoji = "🟠"
    else:
        color = discord.Color.green()
        status_emoji = "🟢"

    embed = discord.Embed(
        title=f"{status_emoji} ComfyUI Queue Status",
        color=color
    )

    # System / GPU info
    if stats:
        devices = stats.get("devices", [])
        for dev in devices:
            name = dev.get("name", "Unknown GPU").split(" : ")[0]  # Strip allocator suffix
            vram_total = dev.get("vram_total", 0)
            vram_free = dev.get("vram_free", 0)
            if vram_total > 0:
                vram_used = vram_total - vram_free
                pct = (vram_used / vram_total) * 100
                vram_bar = _progress_bar(pct)
                embed.add_field(
                    name="🖥️ GPU",
                    value=f"`{name}`\n{vram_bar} {vram_used / (1024**3):.1f} / {vram_total / (1024**3):.1f} GB ({pct:.0f}%)",
                    inline=False
                )

    # Active jobs
    if running:
        active_lines = []
        for job in running:
            prompt_id = job[1][:8] if len(job) > 1 else "?"
            prompt_json = job[2] if len(job) > 2 else {}
            prompt_text = _extract_short_prompt(prompt_json)
            active_lines.append(f"⚡ `{prompt_id}…` — {prompt_text}")
        embed.add_field(
            name=f"🔥 Active ({len(running)})",
            value="\n".join(active_lines[:5]),
            inline=False
        )
    else:
        embed.add_field(name="Active", value="None", inline=True)

    # Pending jobs
    if pending:
        pending_lines = []
        for idx, job in enumerate(pending[:5]):
            prompt_id = job[1][:8] if len(job) > 1 else "?"
            prompt_json = job[2] if len(job) > 2 else {}
            prompt_text = _extract_short_prompt(prompt_json)
            pending_lines.append(f"#{idx+1} `{prompt_id}…` — {prompt_text}")
        footer_extra = f"\n… and {len(pending) - 5} more" if len(pending) > 5 else ""
        embed.add_field(
            name=f"📋 Pending ({len(pending)})",
            value="\n".join(pending_lines) + footer_extra,
            inline=False
        )
    else:
        embed.add_field(name="Pending", value="None", inline=True)

    embed.set_footer(text=f"ComfyUI @ {COMFYUI_ADDRESS}")
    await interaction.followup.send(embed=embed, ephemeral=True)


def _progress_bar(percent, length=10):
    """Create a text-based progress bar."""
    filled = int(length * percent / 100)
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}]"


def _extract_short_prompt(prompt_dict):
    """Extract a short prompt preview from a ComfyUI workflow dict."""
    if not prompt_dict:
        return "Unknown"
    for node_id, node in prompt_dict.items():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            if text and text not in ["positive prompt placeholder", "negative prompt placeholder"]:
                if len(text) > 50:
                    text = text[:47] + "…"
                return text
    return "No text prompt"


@bot.tree.command(name="variation_mode", description="Toggle variation strength between 'High' (0.85 denoise) and 'Very High' (0.95 denoise).")
async def variation_mode_command(interaction: discord.Interaction):
    """Toggle default variation mode (High vs Very High) persistently."""
    current = settings.get("variation_mode", "high")
    new_mode = "very_high" if current == "high" else "high"
    settings["variation_mode"] = new_mode
    save_settings()
    
    mode_label = "🔥 Very High (Denoise: 0.95)" if new_mode == "very_high" else "⚡ High (Denoise: 0.85)"
    embed = discord.Embed(
        title="Variation Mode Updated",
        description=f"Variation buttons will now use **{mode_label}**.",
        color=discord.Color.green()
    )
    await interaction.response.send_message(embed=embed, ephemeral=True)
async def handle_stasis_pause(interaction: discord.Interaction, generation_id: str, user_id: int):
    # Check permissions: only original user or admin
    if interaction.user.id != user_id and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Only the user who started this generation can pause it.", ephemeral=True)
        return
        
    await interaction.response.defer()
    
    success = await comfy_client.pause_generation(generation_id)
    if success:
        gen_data = get_generation(generation_id)
        prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
        embed = discord.Embed(
            title="⏸️ Generation Paused (Stasis)",
            description=f"**Prompt:** {truncate_prompt(prompt, 250)}\n\nThis generation has been entered into stasis. You can resume it anytime by clicking the button below or using `/stasis resume {generation_id}`."
        )
        embed.set_footer(text=f"Paused by {interaction.user.name}")
        view = StasisPausedView(generation_id, user_id)
        await edit_message_fallback(interaction, interaction.message.id, embed=embed, view=view)
    else:
        await interaction.followup.send("Could not pause the generation. It may have already finished or failed.", ephemeral=True)


async def handle_stasis_resume(interaction: discord.Interaction, generation_id: str, user_id: int):
    if interaction.user.id != user_id and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Only the user who started this generation can resume it.", ephemeral=True)
        return

    await interaction.response.defer()

    gen_data = get_generation(generation_id)
    if not gen_data or gen_data.get("status") != "stasis":
        await interaction.followup.send("This generation is not in stasis or could not be found.", ephemeral=True)
        return

    # Update DB status and clear previous prompt IDs
    gen_data["status"] = "queued"
    gen_data["prompt_ids"] = []
    active_generations[generation_id] = gen_data
    save_generations()

    # Edit the message to show it is resuming
    prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    embed = discord.Embed(
        title="🔄 Resuming Generation...",
        description=f"**Prompt:** {truncate_prompt(prompt, 250)}\n\nQueuing workflows again on ComfyUI..."
    )
    view = StasisControlsView(generation_id, user_id)
    await edit_message_fallback(interaction, interaction.message.id, embed=embed, view=view)

    # Start the background task
    asyncio.create_task(run_resumed_generation(interaction, generation_id, gen_data, interaction.message.id))


async def run_resumed_generation(interaction: discord.Interaction, generation_id: str, gen_data: dict, status_message_id: int):
    try:
        workflows = gen_data.get("workflows", [])
        tasks = []
        for wf in workflows:
            tasks.append(comfy_client.generate(wf, generation_id=generation_id))
            
        results = await asyncio.gather(*tasks)
        images = [r[0] for r in results]
        
        if len(images) < len(workflows):
            raise Exception(f"Expected {len(workflows)} images, but only got {len(images)}.")
            
        # Complete
        gen_data = get_generation(generation_id) or gen_data
        await complete_grid_generation(interaction, generation_id, images, gen_data, status_message_id=status_message_id)
        
        gen_data = get_generation(generation_id) or gen_data
        gen_data["status"] = "completed"
        active_generations[generation_id] = gen_data
        save_generations()
        
    except StasisInterruptException:
        logger.info(f"Resumed generation {generation_id} was paused again.")
        return
    except Exception as e:
        error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="run_resumed_generation",
            source_file="bot.py",
            severity=ErrorSeverity.ERROR,
            context={"generation_id": generation_id}
        )
        embed = discord.Embed(
            title="❌ Generation Failed",
            description=f"An error occurred while resuming the generation: {e}"
        )
        try:
            await edit_message_fallback(interaction, status_message_id, embed=embed, view=None)
        except Exception:
            pass




# Define `/style` command group
style_group = app_commands.Group(name="style", description="Manage your favorite ComfyUI style references")

@style_group.command(name="list", description="List and manage your saved favorite style codes.")
async def style_list(interaction: discord.Interaction):
    await safe_defer(interaction, ephemeral=True)
    favorites = db.get_favorite_styles(interaction.user.id)
    
    if not favorites:
        await interaction.followup.send("You have no saved style references yet. Click `⭐ Favorite Style` on any completed image grid with a style reference!", ephemeral=True)
        return
        
    view = StylePaginationView(interaction.user.id, favorites, per_page=8)
    await interaction.followup.send(embed=view.build_embed(), view=view, ephemeral=True)

@style_group.command(name="edit", description="Edit a saved style's name or prompt.")
@app_commands.describe(code="The style reference code to edit")
async def style_edit(interaction: discord.Interaction, code: int):
    favorites = db.get_favorite_styles(interaction.user.id)
    selected = next((fav for fav in favorites if fav["style_code"] == code), None)
    
    if not selected:
        await interaction.response.send_message(f"Code `{code}` is not in your favorites list.", ephemeral=True)
        return
        
    modal = EditStyleModal(interaction.user.id, selected)
    await interaction.response.send_modal(modal)

@style_edit.autocomplete('code')
async def style_edit_autocomplete(interaction: discord.Interaction, current: str):
    favorites = db.get_favorite_styles(interaction.user.id)
    choices = []
    for fav in favorites:
        label = f"{fav['style_name']} ({fav['style_code']})"
        if current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=fav['style_code']))
    return choices[:25]

@style_group.command(name="remove", description="Remove a style code from your favorites.")
@app_commands.describe(code="The 6-digit style reference code to remove")
async def style_remove(interaction: discord.Interaction, code: int):
    await safe_defer(interaction, ephemeral=True)
    favorites = db.get_favorite_styles(interaction.user.id)
    codes = [fav["style_code"] for fav in favorites]
    
    if code not in codes:
        await interaction.followup.send(f"Code `{code}` is not in your favorites list.", ephemeral=True)
        return
        
    db.remove_favorite_style(interaction.user.id, code)
    await interaction.followup.send(f"❌ Removed style code `{code}` from your favorites.", ephemeral=True)

@style_remove.autocomplete('code')
async def style_remove_autocomplete(interaction: discord.Interaction, current: str):
    favorites = db.get_favorite_styles(interaction.user.id)
    choices = []
    for fav in favorites:
        label = f"{fav['style_name']} ({fav['style_code']})"
        if current.lower() in label.lower():
            choices.append(app_commands.Choice(name=label[:100], value=fav['style_code']))
    return choices[:25]

@style_group.command(name="batch", description="Queue 5, 10, or 15 generations with style codes from your /styles list or random.")
@app_commands.describe(
    prompt="The prompt to generate images from",
    count="Number of style generations to queue (5, 10, or 15)",
    aspect_ratio="Aspect ratio for generated images (--ar)"
)
@app_commands.choices(
    count=[
        app_commands.Choice(name="5 Styles", value=5),
        app_commands.Choice(name="10 Styles", value=10),
        app_commands.Choice(name="15 Styles", value=15),
    ],
    aspect_ratio=[
        app_commands.Choice(name="21:9 (Ultrawide)", value="21:9"),
        app_commands.Choice(name="16:9 (Widescreen)", value="16:9"),
        app_commands.Choice(name="16:9.3 (Taskbar Fit - 1920x1032)", value="1920:1032"),
        app_commands.Choice(name="10:7 (iPad)", value="10:7"),
        app_commands.Choice(name="3:5 (Portrait)", value="3:5"),
        app_commands.Choice(name="9:16 (Tall Portrait)", value="9:16"),
    ]
)
async def style_batch(interaction: discord.Interaction, prompt: str, count: int = 5, aspect_ratio: str = None):
    await safe_defer(interaction, thinking=True)
    full_prompt = f"{prompt} --sref batch:{count}"
    if aspect_ratio:
        full_prompt = f"{full_prompt} --ar {aspect_ratio}"
    await execute_imagine(interaction, prompt=full_prompt)

# Add style group to bot tree
bot.tree.add_command(style_group)


# =========================================================================
# /lora-build Slash Command Suite (SDXL Character LoRA Dataset Creator)
# =========================================================================

lora_build_group = app_commands.Group(name="lora-build", description="🎨 Build SDXL Character LoRA datasets from reference images")

@lora_build_group.command(name="start", description="Start a new character LoRA dataset session with a seed image.")
@app_commands.describe(
    image="Initial reference image (e.g. your Palworld character screenshot)",
    character_name="Name of your character (e.g. 'Palworld Adventurer')",
    trigger_word="Unique trigger token for LoRA training (e.g. 'ohwx palchar', 'sks character')"
)
async def lora_start(interaction: discord.Interaction, image: discord.Attachment, character_name: str, trigger_word: str):
    await safe_defer(interaction, thinking=True)
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("❌ Please upload a valid image file (PNG/JPG).", ephemeral=True)
        return

    session_id = f"lora_{interaction.user.id}_{int(datetime.now().timestamp())}"
    clean_tw = trigger_word.strip()
    clean_name = character_name.strip()

    # Create session in DB
    db.create_dataset_session(session_id, interaction.user.id, clean_name, clean_tw)

    # Download & save seed image as image #1
    img_bytes = await image.read()
    img_id, img_path = lora_dataset.save_image_to_dataset(session_id, img_bytes, caption=f"{clean_tw}, seed reference image")

    embed = discord.Embed(
        title="🚀 Character LoRA Dataset Session Started!",
        description=(
            f"👤 **Character:** `{clean_name}`\n"
            f"🏷️ **Trigger Word:** `{clean_tw}`\n"
            f"🆔 **Session ID:** `{session_id}`\n\n"
            f"✅ **Seed Image #1 Added** (Cropped to 1024x1024).\n\n"
            f"**Next Steps:**\n"
            f"1. Use `/lora-build generate` or click **`💡 Suggest Prompts`** to build diverse variations.\n"
            f"2. Use **`➕ Add Q1..Q4`** buttons on generated grids to add good shots to your dataset.\n"
            f"3. Run `/lora-build describe` with Florence-2 to auto-caption all images.\n"
            f"4. Run `/lora-build export` when you have 15–30 images to download your training ZIP!"
        ),
        color=discord.Color.green()
    )
    view = LoraBuildStatusView(session_id)
    await interaction.followup.send(embed=embed, view=view)


@lora_build_group.command(name="generate", description="Generate 4 new variations using session references & shot matrix ideas.")
@app_commands.describe(
    prompt="Custom prompt (leave blank to use preset shot idea)",
    preset="Choose a curated LoRA shot matrix preset (e.g. Studio Close-Up, Combat Action, Golden Hour)",
    reference_weight="Identity reference strength (default: 0.60 Medium Reference)",
    checkpoint="SDXL Checkpoint model (default: Wai Illustrious SDXL v1.70)"
)
@app_commands.choices(
    preset=[
        app_commands.Choice(name="📸 Studio Face Portrait", value="📸 Studio Face Portrait"),
        app_commands.Choice(name="😊 Cheerful Smiling Bust", value="😊 Cheerful Smiling Bust"),
        app_commands.Choice(name="⚔️ Dynamic Combat Action", value="⚔️ Dynamic Combat Action"),
        app_commands.Choice(name="🌲 Palworld Lush Wilderness", value="🌲 Palworld Lush Wilderness"),
        app_commands.Choice(name="🌅 Golden Hour Side Profile", value="🌅 Golden Hour Side Profile"),
        app_commands.Choice(name="⛺ Night Campfire Glow", value="⛺ Night Campfire Glow"),
        app_commands.Choice(name="👑 Heroic 3/4 Turn", value="👑 Heroic 3/4 Turn"),
        app_commands.Choice(name="🌧️ Moody Rain & Fog", value="🌧️ Moody Rain & Fog"),
        app_commands.Choice(name="👀 Over-The-Shoulder View", value="👀 Over-The-Shoulder View"),
        app_commands.Choice(name="🏛️ Ancient Stone Ruins", value="🏛️ Ancient Stone Ruins"),
        app_commands.Choice(name="✨ Cozy Indoor Room", value="✨ Cozy Indoor Room"),
        app_commands.Choice(name="⚡ Low-Angle Power Stance", value="⚡ Low-Angle Power Stance"),
    ],
    reference_weight=[
        app_commands.Choice(name="0.20 (Subtle Reference)", value=0.20),
        app_commands.Choice(name="0.40 (Light Reference)", value=0.40),
        app_commands.Choice(name="0.60 (Medium Reference)", value=0.60),
        app_commands.Choice(name="0.80 (Strong Reference)", value=0.80),
    ],
    checkpoint=SDXL_CHECKPOINT_CHOICES
)
async def lora_generate(
    interaction: discord.Interaction,
    prompt: str = None,
    preset: str = None,
    reference_weight: float = 0.60,
    checkpoint: str = "waiIllustriousSDXL_v170.safetensors"
):
    await safe_defer(interaction, thinking=True)
    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    tw = session["trigger_word"]
    selected_ckpt = checkpoint or COMFYUI_CHECKPOINT or "waiIllustriousSDXL_v170.safetensors"

    # Determine prompt (automatically anime Danbooru-tailored for Wai Illustrious vs photo for realistic models)
    presets_map = lora_dataset.get_preset_shot_prompts(tw, selected_ckpt)
    if preset and preset in presets_map:
        base_prompt = presets_map[preset]
        if prompt:
            final_prompt = f"{prompt.strip()}, {base_prompt}"
        else:
            final_prompt = base_prompt
    elif prompt:
        p_clean = prompt.strip()
        if tw.lower() not in p_clean.lower():
            final_prompt = f"{tw}, {p_clean}"
        else:
            final_prompt = p_clean
    else:
        is_anime = lora_dataset.is_anime_checkpoint(selected_ckpt)
        matrix = lora_dataset.LORA_SHOT_MATRIX_ANIME if is_anime else lora_dataset.LORA_SHOT_MATRIX_REALISTIC
        rand_shot = random.choice(matrix)
        final_prompt = rand_shot["template"].format(trigger_word=tw)

    # Get latest reference image from session
    images = db.get_dataset_images(session_id)
    if not images:
        await interaction.followup.send("❌ Your dataset session has no reference images. Add one using `/lora-build add`.", ephemeral=True)
        return

    ref_img_path = images[-1]["image_path"]
    if not os.path.exists(ref_img_path) and len(images) > 1:
        ref_img_path = images[0]["image_path"]

    # Upload reference image to ComfyUI
    try:
        with open(ref_img_path, "rb") as f:
            ref_bytes = f.read()
        upload_res = await comfy_client.upload_image(ref_bytes, f"lora_ref_{os.path.basename(ref_img_path)}")
        uploaded_ref_name = upload_res.get("name")
    except Exception as e:
        logger.error(f"Error uploading LoRA reference image: {e}")
        uploaded_ref_name = None

    # Trigger generation
    rw = max(0.2, min(1.0, reference_weight))
    selected_ckpt = checkpoint or COMFYUI_CHECKPOINT or "waiIllustriousSDXL_v170.safetensors"
    
    await execute_imagine(
        interaction,
        prompt=f"{final_prompt} --cref-weight {rw}",
        checkpoint=selected_ckpt,
        aspect_ratio="1:1",
        reference_image_url=None,
        is_lora_build=True,
        lora_session_id=session_id,
        cref_image_name_override=uploaded_ref_name
    )


@lora_build_group.command(name="add", description="Add an additional reference image to your active dataset session.")
@app_commands.describe(
    image="Image file to add (will be center-cropped to 1024x1024)",
    caption="Optional initial caption for this image"
)
async def lora_add(interaction: discord.Interaction, image: discord.Attachment, caption: str = None):
    await safe_defer(interaction, thinking=True)
    if not image.content_type or not image.content_type.startswith("image/"):
        await interaction.followup.send("❌ Please upload a valid image file (PNG/JPG).", ephemeral=True)
        return

    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    tw = session["trigger_word"]
    img_bytes = await image.read()
    
    cap = caption.strip() if caption else f"{tw}, character photo"
    img_id, img_path = lora_dataset.save_image_to_dataset(session_id, img_bytes, caption=cap)
    all_imgs = db.get_dataset_images(session_id)

    await interaction.followup.send(
        f"✅ **Added image to dataset session `{session['name']}`!**\n"
        f"📁 **Image #{img_id}** saved (1024x1024 PNG).\n"
        f"📊 Total images in session: **{len(all_imgs)}**"
    )


@lora_build_group.command(name="suggest", description="Get 5 tailored prompt & shot ideas for character LoRA dataset building.")
@app_commands.describe(count="Number of suggestions to generate (default 5)")
async def lora_suggest(interaction: discord.Interaction, count: int = 5):
    await safe_defer(interaction, thinking=True, ephemeral=True)
    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    tw = session["trigger_word"]
    num = max(1, min(10, count))

    suggestions = lora_dataset.generate_suggested_prompts(session_id, tw, count=num)
    lines = []
    for idx, s in enumerate(suggestions, 1):
        lines.append(f"**Shot {idx}:**\n```{s}```")

    embed = discord.Embed(
        title=f"💡 LoRA Shot Suggestions for `{session['name']}`",
        description="\n".join(lines),
        color=discord.Color.gold()
    )
    embed.set_footer(text="Use these prompts in /lora-build generate or /imagine")
    await interaction.followup.send(embed=embed, ephemeral=True)


@lora_build_group.command(name="describe", description="Auto-describe all images in your dataset session using Florence-2.")
async def lora_describe(interaction: discord.Interaction):
    await safe_defer(interaction, thinking=True)
    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    images = db.get_dataset_images(session_id)
    if not images:
        await interaction.followup.send("❌ No images in this dataset session to describe.", ephemeral=True)
        return

    status_msg = await interaction.followup.send(f"⏳ Running Florence-2 auto-captioning on **{len(images)}** images in session `{session['name']}`... Please wait.")

    async def progress_cb(current, total):
        try:
            await status_msg.edit(content=f"⏳ Running Florence-2: Captioning image **{current}/{total}** in `{session['name']}`...")
        except Exception:
            pass

    stats = await lora_dataset.batch_caption_session(comfy_client, session_id, session["trigger_word"], progress_callback=progress_cb)
    await status_msg.edit(content=f"✅ **Florence-2 Captioning Complete!**\nDescribed **{stats['processed']}** images (Failed: {stats['failed']}). Trigger word `{session['trigger_word']}` injected into all `.txt` caption files.")


@lora_build_group.command(name="status", description="View active LoRA dataset session status, image count, and captions.")
async def lora_status(interaction: discord.Interaction):
    await safe_defer(interaction, thinking=True)
    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    images = db.get_dataset_images(session_id)
    captioned_count = sum(1 for img in images if img.get("caption"))

    embed = discord.Embed(
        title=f"🎨 LoRA Dataset Session: {session['name']}",
        description=(
            f"**Session ID:** `{session['session_id']}`\n"
            f"**Trigger Word:** `{session['trigger_word']}`\n"
            f"**Total Images:** `{len(images)}` (Recommended: 15–30)\n"
            f"**Captions Generated:** `{captioned_count} / {len(images)}`\n"
            f"**Target Resolution:** `1024x1024` (SDXL Native)\n"
            f"**Folder:** `datasets/{session_id}/`"
        ),
        color=discord.Color.blue()
    )
    view = LoraBuildStatusView(session_id)
    await interaction.followup.send(embed=embed, view=view)


@lora_build_group.command(name="export", description="Package dataset images and .txt captions into a ZIP for Kohya_ss / Civitai.")
@app_commands.describe(repeats="Number of dataset repeats per epoch (default 10 for Kohya_ss)")
async def lora_export(interaction: discord.Interaction, repeats: int = 10):
    await safe_defer(interaction, thinking=True)
    session = db.get_active_dataset_session(interaction.user.id)
    if not session:
        await interaction.followup.send("❌ No active dataset session found! Start one first with `/lora-build start`.", ephemeral=True)
        return

    session_id = session["session_id"]
    images = db.get_dataset_images(session_id)
    if not images:
        await interaction.followup.send("❌ No images in this dataset to export.", ephemeral=True)
        return

    rep = max(1, min(100, repeats))
    try:
        zip_path, img_count = lora_dataset.export_dataset_zip(session_id, repeats=rep)
        file = discord.File(zip_path, filename=os.path.basename(zip_path))
        await interaction.followup.send(
            content=(
                f"📦 **SDXL Character LoRA Dataset Exported!**\n"
                f"👤 **Character:** `{session['name']}`\n"
                f"🏷️ **Trigger Word:** `{session['trigger_word']}`\n"
                f"🖼️ **Images Included:** `{img_count}` (1024x1024 PNG + TXT captions)\n"
                f"📁 **Folder Format:** `{rep}_{session['trigger_word'].replace(' ', '_')}`\n\n"
                f"✨ Ready to train directly in Kohya_ss, OneTrainer, or Civitai!"
            ),
            file=file
        )
    except Exception as exp_err:
        logger.error(f"Error exporting dataset: {exp_err}")
        await interaction.followup.send(f"❌ Failed to export dataset ZIP: {exp_err}")


@lora_build_group.command(name="list", description="List all your saved LoRA dataset sessions.")
async def lora_list(interaction: discord.Interaction):
    await safe_defer(interaction, thinking=True, ephemeral=True)
    sessions = db.get_user_dataset_sessions(interaction.user.id)
    if not sessions:
        await interaction.followup.send("You have no dataset sessions yet. Start one with `/lora-build start`!", ephemeral=True)
        return

    lines = []
    for s in sessions:
        active_tag = "🟢 **[ACTIVE]**" if s["is_active"] else "⚪"
        imgs = db.get_dataset_images(s["session_id"])
        lines.append(f"{active_tag} **{s['name']}** (Trigger: `{s['trigger_word']}`) — `{len(imgs)}` images\n`ID: {s['session_id']}`")

    embed = discord.Embed(
        title="📚 Your Character LoRA Dataset Sessions",
        description="\n\n".join(lines),
        color=discord.Color.purple()
    )
    await interaction.followup.send(embed=embed, ephemeral=True)


# Register /lora-build group
bot.tree.add_command(lora_build_group)



def terminate_existing_comfyui() -> bool:
    """Terminates any active ComfyUI processes (tracked PID or listening on port 8188)."""
    global comfy_process
    killed = False
    if comfy_process and comfy_process.returncode is None:
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(comfy_process.pid)], capture_output=True)
            else:
                comfy_process.terminate()
            comfy_process = None
            killed = True
        except Exception as e:
            logger.error(f"Error terminating tracked comfy_process: {e}")

    if os.name == 'nt':
        try:
            port = COMFYUI_ADDRESS.split(":")[-1] if ":" in COMFYUI_ADDRESS else "8188"
            out = subprocess.run(
                f'netstat -aon | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            lines = out.stdout.strip().splitlines()
            pids_to_kill = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pid = parts[-1]
                    if pid != "0":
                        pids_to_kill.add(pid)

            for pid in pids_to_kill:
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
                killed = True
        except Exception as e:
            logger.error(f"Error finding/killing process on port 8188: {e}")
    return killed


# ComfyUI Remote Control Commands
@bot.tree.command(name="cui-start", description="Start or restart a fresh instance of the local ComfyUI server.")
@app_commands.describe(force="Force start even if high VRAM usage (Tdarr) is detected")
async def cui_start_command(interaction: discord.Interaction, force: bool = False):
    global comfy_process
    if not is_authorized_admin(interaction):
        await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can start or manage the ComfyUI server.", ephemeral=True)
        return

    await safe_defer(interaction, thinking=True)

    # 1. Terminate any existing instance so we always start a completely fresh instance
    was_running = await comfy_client.is_online()
    killed = terminate_existing_comfyui()
    if was_running or killed:
        await asyncio.sleep(2)  # Give Windows socket and VRAM time to release

    # 2. Check for VRAM caution (e.g. TDARR transcoding)
    is_caution, vram_info = await asyncio.to_thread(check_gpu_vram_caution)
    if is_caution and not force:
        gpu_name = vram_info.get("name", "NVIDIA GPU")
        pct = vram_info.get("percent_used", 0)
        used = vram_info.get("used_gb", 0)
        total = vram_info.get("total_gb", 0)
        free = vram_info.get("free_gb", 0)
        embed = discord.Embed(
            title="⚠️ CAUTION: High VRAM Load Detected",
            description=(
                f"GPU **{gpu_name}** is under heavy load (likely Tdarr transcoding or external application).\n\n"
                f"• **VRAM Usage:** `{pct:.1f}%` (`{used:.2f} GB` / `{total:.2f} GB`)\n"
                f"• **Free VRAM:** `{free:.2f} GB` (Min threshold: `{VRAM_MIN_FREE_GB} GB`)\n\n"
                "⛔ **Server launch was halted** to prevent Out-Of-Memory crashes.\n\n"
                "💡 *To force launch anyway, run:* `/cui-start force:True`"
            ),
            color=discord.Color.gold()
        )
        await interaction.followup.send(embed=embed)
        return

    if not os.path.exists(COMFYUI_BATCH_PATH):
        await interaction.followup.send(f"❌ Could not find batch file at `{COMFYUI_BATCH_PATH}`. Please check configuration.")
        return

    comfy_dir = os.path.dirname(COMFYUI_BATCH_PATH)
    try:
        if os.name == 'nt':
            # Launch batch script in a MINIMIZED console window while keeping PID tracked
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = 6  # SW_MINIMIZE
            comfy_process = await asyncio.create_subprocess_exec(
                "cmd.exe", "/c", COMFYUI_BATCH_PATH,
                cwd=comfy_dir,
                creationflags=subprocess.CREATE_NEW_CONSOLE,
                startupinfo=si
            )
        else:
            comfy_process = await asyncio.create_subprocess_exec(
                COMFYUI_BATCH_PATH,
                cwd=comfy_dir
            )

        logger.info(f"Launched fresh ComfyUI server batch file (PID: {comfy_process.pid})")
        action_msg = "🔄 **Restarting ComfyUI Server (Fresh Instance)...**" if (was_running or killed) else "⏳ **Starting ComfyUI Server...**"
        await interaction.followup.send(f"{action_msg} Please wait while models load.")
    except Exception as e:
        logger.error(f"Failed to start ComfyUI server: {e}")
        await interaction.followup.send(f"❌ Failed to start ComfyUI server: `{e}`")
        return

    # Poll server online status for up to 45 seconds
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 45:
        await asyncio.sleep(3)
        if await comfy_client.is_online():
            embed = discord.Embed(
                title="🚀 ComfyUI Server Started (Fresh Instance)!",
                description=f"Server is online, clean, and responding at `http://{COMFYUI_ADDRESS}`.",
                color=discord.Color.green()
            )
            await interaction.followup.send(embed=embed)
            return

    # If 45s passed and server is still initializing
    embed = discord.Embed(
        title="⏳ ComfyUI Starting (Extended Load)",
        description=f"The process was launched (PID: {comfy_process.pid}), but the server is still initializing models. Check status again in a moment!",
        color=discord.Color.gold()
    )
    await interaction.followup.send(embed=embed)


@bot.tree.command(name="cui-stop", description="Stop the local ComfyUI server remotely.")
async def cui_stop_command(interaction: discord.Interaction):
    global comfy_process
    if not is_authorized_admin(interaction):
        await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can stop the ComfyUI server.", ephemeral=True)
        return

    await safe_defer(interaction, thinking=True)

    is_currently_online = await comfy_client.is_online()
    if not is_currently_online and (comfy_process is None or comfy_process.returncode is not None):
        # Double check port 8188 just in case
        killed = terminate_existing_comfyui()
        if not killed:
            await interaction.followup.send("🔴 **ComfyUI Server is already offline.**")
            return

    killed_any = terminate_existing_comfyui()

    # Wait up to 10 seconds for server to go offline
    start_time = asyncio.get_event_loop().time()
    while asyncio.get_event_loop().time() - start_time < 10:
        await asyncio.sleep(1)
        if not await comfy_client.is_online():
            embed = discord.Embed(
                title="🛑 ComfyUI Server Stopped",
                description="The ComfyUI server has been successfully shut down.",
                color=discord.Color.red()
            )
            await interaction.followup.send(embed=embed)
            return

    if not await comfy_client.is_online():
        embed = discord.Embed(
            title="🛑 ComfyUI Server Stopped",
            description="The ComfyUI server process was terminated.",
            color=discord.Color.red()
        )
        await interaction.followup.send(embed=embed)
    else:
        await interaction.followup.send("⚠️ Sent termination command, but server may still be shutting down. Check status again in a moment.")


@bot.tree.command(name="cui-status", description="Check the current status, GPU VRAM, and queue of the ComfyUI server.")
async def cui_status_command(interaction: discord.Interaction):
    await safe_defer(interaction, thinking=True)

    is_online = await comfy_client.is_online()

    if not is_online:
        embed = discord.Embed(
            title="🔴 ComfyUI Server Status: OFFLINE",
            description=f"The server at `http://{COMFYUI_ADDRESS}` is currently offline.\n\nUse `/cui-start` to launch the server.",
            color=discord.Color.red()
        )
        if comfy_process and comfy_process.returncode is None:
            embed.add_field(name="Process State", value=f"Process launched (PID: `{comfy_process.pid}`), initializing models...", inline=False)
        await interaction.followup.send(embed=embed)
        return

    # Server is online — fetch queue and system stats
    embed = discord.Embed(
        title="🟢 ComfyUI Server Status: ONLINE",
        description=f"Server is running and responsive at `http://{COMFYUI_ADDRESS}`.",
        color=discord.Color.green()
    )

    if comfy_process and comfy_process.returncode is None:
        embed.add_field(name="Process PID", value=f"`{comfy_process.pid}`", inline=True)

    # 1. Fetch Queue Info
    queue_data = await comfy_client.get_queue()
    if queue_data:
        running = len(queue_data.get("queue_running", []))
        pending = len(queue_data.get("queue_pending", []))
        embed.add_field(name="Active Jobs", value=f"**{running}** running", inline=True)
        embed.add_field(name="Queue Depth", value=f"**{pending}** pending", inline=True)

    # 2. Fetch GPU / VRAM Info
    stats_data = await comfy_client.get_system_stats()
    if stats_data and "devices" in stats_data:
        for idx, dev in enumerate(stats_data["devices"]):
            dev_name = dev.get("name", f"GPU {idx}")
            vram_free = dev.get("vram_free", 0)
            vram_total = dev.get("vram_total", 0)

            if vram_total > 0:
                vram_used = vram_total - vram_free
                used_gb = vram_used / (1024 ** 3)
                total_gb = vram_total / (1024 ** 3)
                percent = (vram_used / vram_total) * 100
                embed.add_field(
                    name=f"🎮 {dev_name}",
                    value=f"VRAM: `{used_gb:.2f} GB` / `{total_gb:.2f} GB` ({percent:.1f}% used)",
                    inline=False
                )

    # 3. Check background VRAM caution alert
    is_caution, vram_info = await asyncio.to_thread(check_gpu_vram_caution)
    if is_caution:
        pct = vram_info.get("percent_used", 0)
        free = vram_info.get("free_gb", 0)
        embed.add_field(
            name="⚠️ CAUTION: Heavy Background VRAM Load",
            value=f"GPU VRAM is at `{pct:.1f}%` capacity (`{free:.2f} GB` free). Tdarr or another process is heavily utilizing GPU memory.",
            inline=False
        )

    await interaction.followup.send(embed=embed)


@bot.tree.command(name="negative_show", description="🚫 View and manage your active negative prompt for /imagine.")
async def negative_show_command(interaction: discord.Interaction):
    current_neg = db.get_negative_prompt(interaction.user.id)
    is_default = (current_neg == db.DEFAULT_NEGATIVE_PROMPT)
    
    embed = discord.Embed(
        title="🚫 Active Negative Prompt (/imagine)",
        description=f"```text\n{current_neg}\n```",
        color=discord.Color.blue() if is_default else discord.Color.gold()
    )
    embed.add_field(name="Status", value="⚙️ Default System Prompt" if is_default else "🎨 Custom User Prompt", inline=True)
    embed.set_footer(text="Use /set_negative to customize or click Reset below to restore default.")

    # Interactive buttons: Edit Modal & Reset
    class NegativePromptView(discord.ui.View):
        def __init__(self):
            super().__init__(timeout=180)

        @discord.ui.button(label="✏️ Edit Negative Prompt", style=discord.ButtonStyle.primary)
        async def edit_button(self, btn_interaction: discord.Interaction, button: discord.ui.Button):
            if btn_interaction.user.id != interaction.user.id:
                await btn_interaction.response.send_message("Only the command requester can edit this.", ephemeral=True)
                return

            class EditNegModal(discord.ui.Modal, title="Edit Negative Prompt"):
                neg_input = discord.ui.TextInput(
                    label="Negative Prompt Text",
                    style=discord.TextStyle.paragraph,
                    default=current_neg,
                    max_length=1000
                )
                async def on_submit(self, modal_interaction: discord.Interaction):
                    db.set_negative_prompt(modal_interaction.user.id, self.neg_input.value)
                    new_embed = discord.Embed(
                        title="✅ Negative Prompt Updated",
                        description=f"```text\n{self.neg_input.value}\n```",
                        color=discord.Color.green()
                    )
                    await modal_interaction.response.send_message(embed=new_embed, ephemeral=True)

            await btn_interaction.response.send_modal(EditNegModal())

        @discord.ui.button(label="🔄 Reset to Default", style=discord.ButtonStyle.secondary)
        async def reset_button(self, btn_interaction: discord.Interaction, button: discord.ui.Button):
            if btn_interaction.user.id != interaction.user.id:
                await btn_interaction.response.send_message("Only the command requester can reset this.", ephemeral=True)
                return
            db.reset_negative_prompt(btn_interaction.user.id)
            reset_embed = discord.Embed(
                title="🔄 Negative Prompt Reset to Default",
                description=f"```text\n{db.DEFAULT_NEGATIVE_PROMPT}\n```",
                color=discord.Color.blue()
            )
            await btn_interaction.response.send_message(embed=reset_embed, ephemeral=True)

    await interaction.response.send_message(embed=embed, view=NegativePromptView())

@bot.tree.command(name="set_negative", description="✍️ Set a custom negative prompt for your /imagine generations.")
@app_commands.describe(prompt="The new negative prompt text to use")
async def set_negative_command(interaction: discord.Interaction, prompt: str):
    db.set_negative_prompt(interaction.user.id, prompt)
    embed = discord.Embed(
        title="✅ Negative Prompt Updated",
        description=f"All future `/imagine` generations will use:\n```text\n{prompt}\n```",
        color=discord.Color.green()
    )
    await interaction.response.send_message(embed=embed)


@bot.tree.command(name="models", description="📦 View all registered checkpoints, LoRAs, and their suited architectures.")
@app_commands.describe(
    architecture="Filter models by architecture (e.g., sdxl, flux, wan, ltx)",
    model_type="Filter by model type (checkpoint, lora, unet)"
)
@app_commands.choices(
    architecture=[
        app_commands.Choice(name="All Architectures", value="all"),
        app_commands.Choice(name="🎨 SDXL", value="sdxl"),
        app_commands.Choice(name="⚡ FLUX", value="flux"),
        app_commands.Choice(name="🎬 WAN (Video)", value="wan"),
        app_commands.Choice(name="🎥 LTX (Video)", value="ltx"),
    ],
    model_type=[
        app_commands.Choice(name="All Types", value="all"),
        app_commands.Choice(name="Checkpoints", value="checkpoint"),
        app_commands.Choice(name="LoRAs", value="lora"),
    ]
)
async def models_command(
    interaction: discord.Interaction, 
    architecture: str = "all", 
    model_type: str = "all"
):
    arch_filter = None if architecture == "all" else architecture
    type_filter = None if model_type == "all" else model_type
    
    entries = db.get_models_by_architecture(base_architecture=arch_filter, model_type=type_filter)
    if not entries:
        await interaction.response.send_message(
            f"No registered models found matching filter `architecture={architecture}`, `type={model_type}`.", 
            ephemeral=True
        )
        return

    embed = discord.Embed(
        title="📦 Registered Model & LoRA Architectures",
        description="Architecture registry ensures compatible checkpoint and LoRA pairing without tensor mismatch errors.",
        color=discord.Color.blurple()
    )

    grouped = {}
    for entry in entries:
        arch = entry.get("base_architecture", "unknown").upper()
        grouped.setdefault(arch, []).append(entry)

    for arch, items in grouped.items():
        lines = []
        for it in items[:12]:
            badge = model_architecture.get_architecture_badge(it.get("base_architecture", ""))
            mtype = "Checkpoint" if it.get("model_type") == "checkpoint" else "LoRA"
            subtype = f" ({it.get('sub_type')})" if it.get("sub_type") and it.get("sub_type") != "standard" else ""
            lines.append(f"• **{it.get('display_name')}** `{badge}` — *{mtype}{subtype}*")
        
        if len(items) > 12:
            lines.append(f"*...and {len(items) - 12} more*")

        embed.add_field(name=f"🏛️ {arch} Models ({len(items)})", value="\n".join(lines), inline=False)

    await interaction.response.send_message(embed=embed, ephemeral=True)


@bot.tree.command(name="scan_models", description="🔄 Scan local ComfyUI model directories and auto-register new models & LoRAs.")
async def scan_models_command(interaction: discord.Interaction):
    if not is_authorized_admin(interaction):
        await interaction.response.send_message("⛔ **Access Denied:** Only the bot owner or server administrators can scan and register models.", ephemeral=True)
        return

    await interaction.response.defer(ephemeral=True, thinking=True)
    
    import asyncio
    stats = await asyncio.to_thread(model_architecture.scan_and_register_comfyui_models)
    
    if not stats.get("success"):
        await interaction.followup.send(f"❌ Scan failed: {stats.get('error')}", ephemeral=True)
        return
        
    embed = discord.Embed(
        title="🔍 Model & LoRA Auto-Discovery Complete",
        description="Scanned local ComfyUI model directories and updated SQLite architecture registry.",
        color=discord.Color.green()
    )
    embed.add_field(name="📁 Files Scanned", value=str(stats.get("scanned", 0)), inline=True)
    embed.add_field(name="🏛️ Checkpoints", value=str(stats.get("checkpoints", 0)), inline=True)
    embed.add_field(name="⚡ LoRAs", value=str(stats.get("loras", 0)), inline=True)
    embed.add_field(name="🟠 UNets / Diffusion", value=str(stats.get("unets", 0)), inline=True)
    embed.add_field(name="✅ Total Registered", value=str(stats.get("total_registered", 0)), inline=True)
    
    await interaction.followup.send(embed=embed, ephemeral=True)


_instance_lock_socket = None

def acquire_instance_lock(port: int = 48123) -> bool:
    """
    Ensures only a single instance of the bot process can run on the machine at a time.
    Binds a localhost TCP socket on a dedicated lock port.
    If another instance is already active, binding fails and startup is aborted,
    preventing duplicate command queueing and double button interaction events (e.g. U1-U4).
    """
    global _instance_lock_socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", port))
        sock.listen(1)
        _instance_lock_socket = sock
        return True
    except OSError:
        try:
            sock.close()
        except Exception:
            pass
        logger.error(f"[!] Another instance of Shallot-CUI Bot is already running (port {port} in use). Startup aborted.")
        print("\n" + "=" * 72)
        print("[!] CRITICAL ERROR: Shallot-CUI Bot is ALREADY RUNNING!")
        print("Running multiple bot instances causes all prompts and button clicks (like U1)")
        print("to execute twice (duplicate generations/upscales).")
        print("Please close any existing bot console windows before launching a new one.")
        print("=" * 72 + "\n")
        return False


if __name__ == "__main__":
    if not acquire_instance_lock():
        sys.exit(1)

    if not DISCORD_TOKEN or DISCORD_TOKEN == "YOUR_DISCORD_BOT_TOKEN_HERE":
        logger.error("Please configure the DISCORD_TOKEN in the .env file before running.")
    else:
        bot.run(DISCORD_TOKEN, log_handler=None)



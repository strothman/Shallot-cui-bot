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
import gc
from collections import OrderedDict
from typing import Optional, Dict, Any, List
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
from characters import get_character_autocomplete_choices
from model_architecture import Architecture
from celebrities import (
    CELEBRITY_CHOICES_KREA2,
    get_celebrity_autocomplete_choices,
    get_celebrity_display_badge,
    get_celebrity
)

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
    parse_powerhouse_prompt,
    parse_freeu_prompt,
    apply_smart_magic_and_sref,
    apply_magic_enhancement,
    extract_positive_prompt,
    clean_midjourney_flags,
    LOCKED_STYLE_PRESETS,
    build_scapes_prompt,
    apply_face_detailer_to_workflow,
    resolve_bertflow_dimensions,
    format_krea2_prompt,
    format_sdxl_prompt,
    format_flux_prompt,
    sanitize_describe_text,
    deduplicate_intro_quality_tags,
    fuse_krea2_blend_prompt,
    get_bertflow_unet_model,
    prepare_bertflow_workflow,
)
from image_utils import (
    crop_to_aspect_ratio,
    crop_to_aspect_ratio_async,
    create_grid,
    create_grid_async,
    embed_metadata,
    embed_metadata_async,
    calculate_outpaint_padding,
    calculate_outpaint_padding_async,
    save_quadrant_images,
    save_quadrant_images_async,
    get_quadrant_bytes,
    get_quadrant_bytes_async,
    crop_quadrant_from_grid_bytes,
    crop_quadrant_from_grid_bytes_async,
    format_image_filename,
    get_dated_save_prefix,
    upscale_isolated_image,
    upscale_isolated_image_async,
    boost_image_vibrancy_and_contrast,
    boost_image_vibrancy_and_contrast_async,
    get_checkpoint_abbrev,
    detect_closest_aspect_ratio,
    detect_closest_krea_aspect_ratio,
    create_thumbnail_bytes,
    create_thumbnail_bytes_async,
    QUADRANT_CACHE_DIR,
)
from views import (
    GridButtons,
    UpscaleButtons,
    IsolatedImageButtons,
    DescribeButtons,
    BlendButtons,
    build_blend_embed,
    build_blend_complete_embed,
    build_blended_image_embed,
    EditBlendPromptModal,
    build_blend_krea_embed,
    EditBlendKreaModal,
    BlendKreaButtons,
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
    CancelGenerationView,
    RemixModal,
    BertflowButtons,
)
import model_architecture
from characters import get_character, mask_character_in_prompt, inject_trained_trigger_in_prompt, CHARACTERS

_active_architecture = None


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
from config import BOT_OWNER_ID, is_authorized_admin, PipelineDefaults, get_checkpoint_display_name

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
    app_commands.Choice(name="Pony Diffusion V6 XL", value="ponyDiffusionV6XL_v6StartWithThisOne.safetensors"),
    app_commands.Choice(name="RealVisXL V5.0 Lightning (Ultra Fast)", value="RealVisXL_V5.0_Lightning_fp16.safetensors"),
    app_commands.Choice(name="Nova Furry", value="novaFurryXL_ilV180A.safetensors"),
]

# Consolidated Enhancements choices (replaces multiple True/False toggles with a clean dropdown)
SDXL_ENHANCEMENT_CHOICES = [
    app_commands.Choice(name="👑 Ultimate Quality (Powerhouse 1.35x + Smart Director + Magic)", value="ultimate"),
    app_commands.Choice(name="🌟 Studio Duo (Smart Art Director + Magic Prompt)", value="smart+magic"),
    app_commands.Choice(name="⚡ 2-Stage Powerhouse (FreeU + 1.35x Refiner)", value="powerhouse"),
    app_commands.Choice(name="✨ Magic Prompt (Studio Lighting & Cinematic Expansion)", value="magic"),
    app_commands.Choice(name="🧠 Smart Art Director (Subject-Harmonized Prompt & Style)", value="smart"),
    app_commands.Choice(name="🚫 Pure Checkpoint (Disable FreeU Enhancer)", value="no_freeu"),
]

CHARACTER_CHOICES_SDXL = [
    app_commands.Choice(name="🎀 Cheri (Epoch 6 - Default)", value="cheri.85"),
    app_commands.Choice(name="🎀 Cheri (Epoch 4)", value="cheri4.85"),
    app_commands.Choice(name="🎀 Cheri (.70 - Light)", value="cheri.70"),
    app_commands.Choice(name="🔮 Mageill (Epoch 5 - Default)", value="mageill.85"),
    app_commands.Choice(name="🔮 Mageill (Epoch 6)", value="mageill6.85"),
    app_commands.Choice(name="🔮 Mageill (Epoch 4)", value="mageill4.85"),
    app_commands.Choice(name="🔮 Mageill (Epoch 3)", value="mageill3.85"),
    app_commands.Choice(name="🔮 Mageill (.70 - Light)", value="mageill.70"),
    app_commands.Choice(name="🌿 Ogarla (.85 - Default)", value="ogarla.85"),
    app_commands.Choice(name="🌿 Ogarla (.70 - Light)", value="ogarla.70"),
    app_commands.Choice(name="✨ Valerie (.85 - Default)", value="valerie.85"),
    app_commands.Choice(name="✨ Valerie (.70 - Light)", value="valerie.70"),
    app_commands.Choice(name="👓 Sully (.85 - Default)", value="sully.85"),
    app_commands.Choice(name="👓 Sully (.70 - Light)", value="sully.70"),
]


CHARACTER_CHOICES_KREA2 = [
    app_commands.Choice(name="🌿 Ogarla Krea 2 (.85 - Default)", value="ogarla.85"),
    app_commands.Choice(name="🌿 Ogarla Krea 2 (.70 - Light)", value="ogarla.70"),
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
bot = commands.Bot(command_prefix="!", intents=intents, max_messages=100)
bot.comfy_client = comfy_client

# Modular Services & Cogs
from services.vision_service import (
    run_vision_interrogate,
    run_florence_interrogate,
    execute_blend_core,
    execute_blend_message,
    execute_describe_core,
    handle_generate_described,
    handle_update_describe_view,
    parse_adopted_post,
    parse_midjourney_post,
    user_vision_preferences,
)
from cogs.vision_cog import VisionCog
from services.krea_service import (
    BERTFLOW_MODEL_CHOICES,
    execute_bertflow,
    handle_bertflow_reroll,
    handle_bertflow_remix,
    handle_bertflow_toggle_char,
    handle_bertflow_upscale,
    handle_update_blend_krea_view,
    handle_submit_edit_blend_krea_prompt,
    handle_generate_blend_krea,
    execute_blend_krea_core,
)
from cogs.krea_cog import KreaCog
from cogs.system_cog import SystemCog
from cogs.upscale_cog import UpscaleCog
from cogs.imagine_cog import ImagineCog
from services.generation_service import (
    ActiveGenerationsProxy,
    active_generations,
    get_generation,
    save_generations,
    build_blend_workflow,
    complete_grid_generation,
    handle_upscale,
    handle_isolate,
    handle_variation,
    handle_reroll,
    handle_favorite_style,
    handle_favorite_prompt,
    handle_cancel_generation,
    handle_remix,
    handle_outpaint,
    handle_change_sref,
    handle_copy_prompt,
    handle_stasis_pause,
    handle_stasis_resume,
    run_resumed_generation,
    handle_update_blend_view,
    handle_submit_edit_blend_prompts,
    handle_reblend,
    handle_generate_blended,
    execute_blend_generation,
    execute_imagine,
    set_comfy_client as set_generation_comfy_client,
)
set_generation_comfy_client(comfy_client)
bot.execute_imagine = execute_imagine
bot.execute_blend_generation = execute_blend_generation
bot.handle_reblend = handle_reblend
bot.handle_generate_blended = handle_generate_blended
bot.handle_update_blend_view = handle_update_blend_view

from services.upscale_service import (
    execute_upscale_core,
    build_fast_upscale_workflow,
    build_generative_upscale_workflow,
    calculate_target_dimensions,
    calculate_latent_refiner_dimensions,
)
from services.system_service import (
    SETTINGS_FILE,
    settings,
    load_settings,
    save_settings,
    get_setting,
    set_setting,
    terminate_existing_comfyui,
    fetch_comfyui_queue,
    fetch_comfyui_system_stats,
    build_queue_embed,
    build_models_embed,
    purge_vram_core,
)

_vision_cog = VisionCog(bot)
_krea_cog = KreaCog(bot)
_system_cog = SystemCog(bot)
_upscale_cog = UpscaleCog(bot)
_imagine_cog = ImagineCog(bot)
try:
    loop = asyncio.get_running_loop()
    loop.create_task(bot.add_cog(_vision_cog))
    loop.create_task(bot.add_cog(_krea_cog))
    loop.create_task(bot.add_cog(_system_cog))
    loop.create_task(bot.add_cog(_upscale_cog))
    loop.create_task(bot.add_cog(_imagine_cog))
except RuntimeError:
    asyncio.run(bot.add_cog(_vision_cog))
    asyncio.run(bot.add_cog(_krea_cog))
    asyncio.run(bot.add_cog(_system_cog))
    asyncio.run(bot.add_cog(_upscale_cog))
    asyncio.run(bot.add_cog(_imagine_cog))

# Backward-compatibility re-exports for external test suites and legacy imports
blend_image_context = _vision_cog.blend_image_context
blend_sdxl = _vision_cog.blend_sdxl
blend = _vision_cog.blend_sdxl
describe = _vision_cog.describe
bertflow = _krea_cog.bertflow
bertflow_command = _krea_cog.bertflow
blend_krea = _krea_cog.blend_krea
cui_start_command = _system_cog.cui_start
cui_stop_command = _system_cog.cui_stop
cui_status_command = _system_cog.cui_status
free_vram_command = _system_cog.free_vram
purge_vram_command = _system_cog.free_vram  # Alias pointing to free_vram
upscale = _upscale_cog.upscale
upscale_command = _upscale_cog.upscale
queue_command = _system_cog.queue_status
models_command = _system_cog.models
scan_models_command = _system_cog.scan_models
negative_command = _system_cog.negative
prompt_group = _system_cog.prompt_group
style_group = _system_cog.style_group
imagine = _imagine_cog.imagine
study = _imagine_cog.study
adopt_post_context = _imagine_cog.adopt_post_context
execute_adopt_post = _imagine_cog.execute_adopt_post
handle_submit_edit_adopt_prompt = _imagine_cog.handle_submit_edit_adopt_prompt
run_study_imagine_callback = _imagine_cog.run_study_imagine_callback

def load_generations():
    db.init_db()

def cleanup_orphaned_quadrants():
    db.cleanup_orphaned_quadrants()

# settings, load_settings, save_settings imported from services.system_service

async def safe_defer(interaction: discord.Interaction, thinking: bool = False, ephemeral: bool = False):
    """Safely defers an interaction response without crashing if connection drops, socket resets, or token expired."""
    if not interaction.response.is_done():
        for attempt in range(3):
            try:
                await interaction.response.defer(thinking=thinking, ephemeral=ephemeral)
                return
            except (discord.NotFound, discord.HTTPException) as e:
                logger.debug(f"Interaction defer skipped or expired: {e}")
                return
            except (aiohttp.ClientError, OSError, asyncio.TimeoutError) as e:
                logger.warning(f"Transient network glitch during interaction.defer (attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(0.35)
                else:
                    break
            except Exception as e:
                logger.warning(f"Unexpected defer error: {e}")
                break


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
            await sync_presence_now()
    except Exception as e:
        logger.debug(f"Failed to update bot presence: {e}")


def is_interaction_expired(interaction) -> bool:
    """Returns True if the interaction is None or exceeds Discord's 15-minute token lifetime."""
    if not interaction:
        return True
    created_at = getattr(interaction, "created_at", None)
    if isinstance(created_at, datetime):
        age = (discord.utils.utcnow() - created_at).total_seconds()
        if age >= 870:  # 14.5 minutes (Discord invalidates webhook tokens at 15m)
            return True
    return False


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

    if not is_interaction_expired(interaction):
        try:
            return await interaction.followup.send(**kwargs, ephemeral=ephemeral)
        except (discord.HTTPException, discord.NotFound, aiohttp.ClientError, OSError) as hex:
            logger.debug(f"Interaction token or socket issue ({getattr(hex, 'code', str(hex))}). Falling back to channel.send.")
    else:
        logger.debug("Interaction token reached 15m lifetime limit. Fast-routing directly to channel.send.")

    channel = getattr(interaction, "channel", None)
    if not channel and getattr(interaction, "channel_id", None):
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
    try:
        return await channel.send(**kwargs)
    except Exception as ce:
        logger.warning(f"Channel send fallback failed: {ce}")
        return None

async def send_error_fallback(interaction, message):
    """Sends an error message using interaction, with fallback to channel.send if expired."""
    # Ensure error message fits within Discord's 2000 character limit
    if len(message) > 1980:
        message = message[:1977] + "..."
    if not is_interaction_expired(interaction):
        try:
            await interaction.followup.send(message, ephemeral=True)
            return
        except (discord.HTTPException, discord.NotFound, aiohttp.ClientError, OSError) as hex:
            logger.debug(f"Interaction token or socket issue ({getattr(hex, 'code', str(hex))}) during error report. Falling back to channel.send.")
    else:
        logger.debug("Interaction token reached 15m lifetime limit during error report. Fast-routing to channel.send.")

    try:
        channel = getattr(interaction, "channel", None)
        if not channel and getattr(interaction, "channel_id", None):
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

async def edit_original_fallback(interaction, content=None, embed=None, view=None, attachments=None):
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
    if attachments is not None:
        edit_kwargs["attachments"] = attachments
        send_kwargs["files"] = attachments

    if not is_interaction_expired(interaction):
        try:
            await interaction.edit_original_response(**edit_kwargs)
            return
        except (discord.HTTPException, discord.NotFound) as hex:
            if getattr(hex, 'code', None) in [50027, 10062, 10015] or getattr(hex, 'status', None) in [404, 400] or isinstance(hex, discord.NotFound):
                logger.debug(f"Interaction token expired ({getattr(hex, 'code', '404')}) during edit. Falling back to channel.send.")
            else:
                raise hex
    else:
        logger.debug("Interaction token reached 15m lifetime limit during edit. Fast-routing to channel.send.")

    channel = getattr(interaction, "channel", None)
    if not channel and getattr(interaction, "channel_id", None):
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


async def edit_message_fallback(interaction, message_id, content=None, embed=None, file=None, view=None, allow_send_fallback=True):
    """Edits a status message by ID. Falls back to channel.send if interaction token expired/fails (unless allow_send_fallback is False)."""
    chan_id = getattr(interaction, "channel_id", None)
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
            channel = getattr(interaction, "channel", None) or await bot.fetch_channel(chan_id)
            message = await channel.fetch_message(message_id)
            if message:
                if file:
                    file.fp.seek(0)
                    edit_kwargs["attachments"] = [file]
                await message.edit(**edit_kwargs)
                return
        except Exception as e:
            if not allow_send_fallback:
                return
            logger.debug(f"Could not edit message via Bot API ({e}). Falling back to interaction/channel send.")

    if not is_interaction_expired(interaction):
        try:
            if file:
                file.fp.seek(0)
                edit_kwargs["attachments"] = [file]
            await interaction.followup.edit_message(message_id, **edit_kwargs)
            return
        except (discord.HTTPException, discord.NotFound) as hex:
            if getattr(hex, 'code', None) in [50027, 10062, 10015] or getattr(hex, 'status', None) in [404, 400] or isinstance(hex, discord.NotFound):
                if not allow_send_fallback:
                    return
                logger.debug(f"Interaction token expired during message edit. Sending new message to channel.")
            else:
                if allow_send_fallback:
                    raise hex
                return
    else:
        if not allow_send_fallback:
            return
        logger.debug("Interaction token reached 15m lifetime limit during message edit. Sending new message to channel.")

    channel = getattr(interaction, "channel", None)
    if not channel and getattr(interaction, "channel_id", None):
        try:
            channel = await bot.fetch_channel(interaction.channel_id)
        except Exception:
            pass
    if channel:
        if file:
            file.fp.seek(0)
            send_kwargs["file"] = file
        await channel.send(**send_kwargs)


# =========================================================================
# Generation Pipeline & Button Handlers — Modularized into services/generation_service.py
# (complete_grid_generation, handle_upscale, handle_isolate, handle_variation,
#  handle_reroll, handle_favorite_style, handle_favorite_prompt, handle_cancel_generation,
#  handle_remix, handle_outpaint, handle_change_sref, handle_copy_prompt)
# =========================================================================



# fetch_comfyui_queue, fetch_comfyui_system_stats imported from services.system_service

def get_effective_queue_counts(comfy_queue: Optional[dict] = None) -> tuple[int, int]:
    """
    Calculates combined active (running) and pending jobs across ComfyUI and EngineAwareQueue.
    ComfyUI processes 1 job at a time while EngineAwareQueue buffers pending jobs
    to prevent VRAM thrashing on 8GB GPUs.
    """
    running = 0
    pending = 0
    if comfy_queue:
        running = len(comfy_queue.get("queue_running", []))
        pending = len(comfy_queue.get("queue_pending", []))

    try:
        from services.engine_queue import get_engine_queue
        eq = get_engine_queue()
        if eq and eq.get_status().get("is_running", False):
            eq_status = eq.get_status()
            if eq_status.get("active_job"):
                running = max(running, 1)
            pending += eq_status.get("pending_count", 0)
    except Exception as e:
        logger.debug(f"Could not read EngineAwareQueue status: {e}")

    return running, pending


def format_presence_status_text(running: int, pending: int) -> tuple[str, discord.ActivityType]:
    """Generates the presence string and Discord activity type from queue counts."""
    if running > 0:
        status_text = f"Processing {running} job{'s' if running != 1 else ''}"
        if pending > 0:
            status_text += f" | {pending} queued"
        return status_text, discord.ActivityType.playing
    elif pending > 0:
        status_text = f"{pending} queued job{'s' if pending != 1 else ''}"
        return status_text, discord.ActivityType.watching
    else:
        return "Ready ✓ | /imagine", discord.ActivityType.watching


_last_presence_sync_time: float = 0.0
_presence_sync_task: Optional[asyncio.Task] = None

_last_generation_activity: float = time.time()
_idle_purged: bool = False
IDLE_PURGE_TIMEOUT: float = float(os.getenv("IDLE_PURGE_TIMEOUT_SECONDS", "1800"))  # 30 minutes


def touch_activity():
    """Updates the last activity timestamp and resets the idle purge state."""
    global _last_generation_activity, _idle_purged
    _last_generation_activity = time.time()
    _idle_purged = False


async def sync_presence_now():
    """Immediately synchronizes bot presence with current ComfyUI and EngineAwareQueue state."""
    global _last_presence_sync_time
    _last_presence_sync_time = time.time()
    try:
        session = getattr(comfy_client, "session", None)
        queue = await fetch_comfyui_queue(session=session)
        if queue is None:
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name="ComfyUI (offline)"
            )
            await bot.change_presence(status=discord.Status.dnd, activity=activity)
            update_console_title("ComfyUI (offline)")
            return

        running, pending = get_effective_queue_counts(queue)
        status_text, activity_type = format_presence_status_text(running, pending)

        activity = discord.Activity(
            type=activity_type,
            name=status_text
        )
        await bot.change_presence(status=discord.Status.online, activity=activity)
        update_console_title(status_text)
    except Exception as e:
        logger.debug(f"Presence update failed: {e}")


def on_engine_queue_change():
    """Throttled callback triggered on EngineAwareQueue enqueue/start/cancel/completion."""
    global _last_presence_sync_time, _presence_sync_task
    touch_activity()
    if not bot.is_ready():
        return
    now = time.time()
    # Respect Discord rate limits (minimum 3.5s interval between gateway presence updates)
    if now - _last_presence_sync_time >= 3.5:
        if _presence_sync_task and not _presence_sync_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            _presence_sync_task = loop.create_task(sync_presence_now())
        except RuntimeError:
            pass
    else:
        if _presence_sync_task is None or _presence_sync_task.done():
            delay = max(0.5, 3.5 - (now - _last_presence_sync_time))
            async def _delayed_sync():
                await asyncio.sleep(delay)
                await sync_presence_now()
            try:
                loop = asyncio.get_running_loop()
                _presence_sync_task = loop.create_task(_delayed_sync())
            except RuntimeError:
                pass


@tasks.loop(seconds=10)
async def update_presence():
    """Periodic fallback loop to update bot presence and console title with queue info."""
    await sync_presence_now()

@update_presence.before_loop
async def before_update_presence():
    await bot.wait_until_ready()

@tasks.loop(hours=6)
async def periodic_scratch_maintenance():
    """Periodically purges orphaned quadrant scratch files and vacuums SQLite database."""
    try:
        stats = await asyncio.to_thread(db.cleanup_orphaned_quadrants, 48.0)
        await asyncio.to_thread(db.vacuum_database)
        if stats.get("deleted", 0) > 0:
            reclaimed_mb = round(stats.get("reclaimed_bytes", 0) / (1024 * 1024), 2)
            logger.info(f"🧹 Scratch maintenance: evicted {stats['deleted']} stale scratch file(s) ({reclaimed_mb} MB reclaimed).")
    except Exception as e:
        logger.debug(f"Periodic scratch maintenance warning: {e}")

@periodic_scratch_maintenance.before_loop
async def before_periodic_scratch_maintenance():
    await bot.wait_until_ready()

@tasks.loop(minutes=5)
async def idle_memory_watchdog():
    """Periodically purges ComfyUI model caches from RAM/VRAM and runs garbage collection after extended inactivity."""
    global _idle_purged
    try:
        now = time.time()
        idle_seconds = now - _last_generation_activity
        if idle_seconds >= IDLE_PURGE_TIMEOUT and not _idle_purged:
            # Check if engine queue has any running or pending jobs
            running = 0
            pending = 0
            try:
                from services.engine_queue import get_engine_queue
                eq = get_engine_queue()
                if eq:
                    q_stat = eq.get_queue_status()
                    running = 1 if q_stat.get("active_job") else 0
                    pending = q_stat.get("queue_length", 0)
            except Exception:
                pass

            if running == 0 and pending == 0:
                logger.info(f"🧹 Idle memory watchdog: Bot has been idle for {int(idle_seconds // 60)}m. Purging ComfyUI RAM/VRAM cache and trimming working set...")
                try:
                    from services.system_service import purge_vram_core
                    await purge_vram_core(client=comfy_client, trim_working_set=True)
                except Exception as purge_err:
                    logger.debug(f"Idle watchdog purge_vram_core warning: {purge_err}")
                _idle_purged = True
                logger.info("✅ Idle memory purge complete: ComfyUI models evicted, Python GC run, and host RAM trimmed.")
    except Exception as e:
        logger.debug(f"Idle memory watchdog error: {e}")

@idle_memory_watchdog.before_loop
async def before_idle_memory_watchdog():
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
        try:
            from services.engine_queue import set_engine_queue_client
            _eq = set_engine_queue_client(comfy_client)
            _eq.add_change_listener(on_engine_queue_change)
            _eq.start()
            logger.info("🟢 EngineAwareQueue worker started with model-affinity scheduling.")
        except Exception as eq_err:
            logger.warning(f"EngineAwareQueue initialization warning: {eq_err}")

        try:
            from services.recovery_service import start_crash_recovery
            start_crash_recovery(bot, comfy_client)
            logger.info("🟢 Crash recovery service initialized and listening for pending jobs.")
        except Exception as rec_err:
            logger.warning(f"Crash recovery startup warning: {rec_err}")

        # Launch non-blocking background scratch maintenance on startup
        asyncio.create_task(asyncio.to_thread(db.cleanup_orphaned_quadrants, 48.0))
        
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
    touch_activity()
    logger.info(f"Bot connected as {bot.user}")
    # Start background loops
    if not update_presence.is_running():
        update_presence.start()
    if not periodic_scratch_maintenance.is_running():
        periodic_scratch_maintenance.start()
    if not idle_memory_watchdog.is_running():
        idle_memory_watchdog.start()

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
    touch_activity()
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
        elif custom_id.startswith("cancel_gen:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                generation_id = parts[1]
                await handle_cancel_generation(interaction, generation_id)
        elif custom_id.startswith("remix:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                generation_id = parts[1]
                await handle_remix(interaction, generation_id)
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
        elif custom_id.startswith("bertflow_reroll:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                await handle_bertflow_reroll(interaction, parts[1])
        elif custom_id.startswith("bertflow_remix:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                await handle_bertflow_remix(interaction, parts[1])
        elif custom_id.startswith("bertflow_toggle_char:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                await handle_bertflow_toggle_char(interaction, parts[1])
        elif custom_id.startswith("bertflow_upscale:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                await handle_bertflow_upscale(interaction, parts[1])
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
                    scale = parts[3] if len(parts) >= 4 else "2.0"
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
                await handle_update_describe_view(interaction, generation_id, new_ar, new_sr, new_oga, new_model)
        elif custom_id.startswith("set_blend_krea_ar:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                val = parts[2]
                await handle_update_blend_krea_view(interaction, gen_id, new_ar=val)
        elif custom_id.startswith("toggle_blend_krea_model:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id) or {}
                cur_model = gen_data.get("model_choice", "muse")
                next_model = "pornmaster" if "muse" in cur_model.lower() else "muse"
                await handle_update_blend_krea_view(interaction, gen_id, new_model=next_model)
        elif custom_id.startswith("toggle_blend_krea_wetness:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id) or {}
                cur_wet = float(gen_data.get("wetness", -2.0))
                next_wet = 0.0 if cur_wet == -2.0 else (1.0 if cur_wet == 0.0 else -2.0)
                await handle_update_blend_krea_view(interaction, gen_id, new_wetness=next_wet)
        elif custom_id.startswith("set_blend_krea_comp:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                if interaction.data and "values" in interaction.data:
                    val = interaction.data["values"][0]
                    await handle_update_blend_krea_view(interaction, gen_id, new_comp=val)
        elif custom_id.startswith("toggle_blend_krea_composition:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id) or {}
                cur_comp = gen_data.get("composition", "off")
                next_comp = "subtle" if cur_comp == "off" else ("medium" if cur_comp == "subtle" else ("strong" if cur_comp == "medium" else "off"))
                await handle_update_blend_krea_view(interaction, gen_id, new_comp=next_comp)
        elif custom_id.startswith("set_blend_krea_char:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                if interaction.data and "values" in interaction.data:
                    val = interaction.data["values"][0]
                    await handle_update_blend_krea_view(interaction, gen_id, new_char=val)
        elif custom_id.startswith("set_blend_krea_celeb:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                if interaction.data and "values" in interaction.data:
                    val = interaction.data["values"][0]
                    await handle_update_blend_krea_view(interaction, gen_id, new_celeb=val)
        elif custom_id.startswith("edit_blend_krea_prompt:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                gen_data = get_generation(gen_id) or {}
                cur_prompt = gen_data.get("fused_prompt") or gen_data.get("krea2_prompt") or gen_data.get("user_prompt", "")
                modal = EditBlendKreaModal(
                    generation_id=gen_id,
                    current_prompt=cur_prompt,
                    on_submit_callback=handle_submit_edit_blend_krea_prompt
                )
                try:
                    if not interaction.response.is_done():
                        await interaction.response.send_modal(modal)
                except discord.HTTPException as e:
                    if e.code != 40060:
                        logger.warning(f"Failed to send EditBlendKreaModal: {e}")
        elif custom_id.startswith("gen_blend_krea:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                gen_id = parts[1]
                await handle_generate_blend_krea(interaction, gen_id)
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
        elif custom_id.startswith("toggle_blend_sr:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                cur_sr = gen_data.get("sr", True)
                if cur_sr not in ["nosr", False, None]:
                    new_sr = "nosr"
                else:
                    new_sr = "sr75"
                await handle_update_blend_view(interaction, gen_id, new_sr=new_sr)
        elif custom_id.startswith("cycle_blend_ar:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                ar_list = ["16:9", "21:9", "10:7", "1:1", "3:5", "9:16"]
                cur_ar = gen_data.get("ar", "16:9")
                try:
                    idx = ar_list.index(cur_ar)
                    next_ar = ar_list[(idx + 1) % len(ar_list)]
                except ValueError:
                    next_ar = "16:9"
                await handle_update_blend_view(interaction, gen_id, new_ar=next_ar)
        elif custom_id.startswith("cycle_blend_comp:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                comp_list = ["style", "low", "med", "high"]
                cur_comp = gen_data.get("comp_strength", "style")
                try:
                    idx = comp_list.index(cur_comp)
                    next_comp = comp_list[(idx + 1) % len(comp_list)]
                except ValueError:
                    next_comp = "low"
                await handle_update_blend_view(interaction, gen_id, new_comp=next_comp)
        elif custom_id.startswith("toggle_blend_sref:") or custom_id.startswith("cycle_blend_style:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            gen_data = get_generation(gen_id)
            if gen_data:
                cur_sref = gen_data.get("sref_rand", "nosref")
                is_on = cur_sref in ["sref", "sref1", True] or (isinstance(cur_sref, str) and cur_sref.lower() in ["true", "1", "on"])
                next_sref = "nosref" if is_on else "sref"
                await handle_update_blend_view(interaction, gen_id, new_sref=next_sref)
        elif custom_id.startswith("switch_blend_tab:"):
            parts = custom_id.split(":")
            if len(parts) >= 3:
                gen_id = parts[1]
                tab = parts[2]
                await handle_update_blend_view(interaction, gen_id, tab=tab)
        elif custom_id.startswith("set_blend_char:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_char=val)
        elif custom_id.startswith("set_blend_sr:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_sr=val)
        elif custom_id.startswith("set_blend_style:"):
            parts = custom_id.split(":")
            gen_id = parts[1]
            if interaction.data and "values" in interaction.data:
                val = interaction.data["values"][0]
                await handle_update_blend_view(interaction, gen_id, new_sref=val)
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
                char_choice = gen_data.get("char_choice", "ogarla" if use_oga else "none")
                model_choice = gen_data.get("model_choice", "wai")
                comp_strength = gen_data.get("comp_strength", "style")
                use_sref = gen_data.get("sref_rand", "nosref")
                await safe_defer(interaction)
                await handle_generate_blended(interaction, generation_id, desc_type, ar=ar, use_sr=use_sr, use_oga=use_oga, model_choice=model_choice, comp_strength=comp_strength, use_sref_rand=use_sref, char_choice=char_choice)
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
        elif custom_id.startswith("reblend:"):
            parts = custom_id.split(":")
            if len(parts) >= 2:
                generation_id = parts[1]
                await handle_reblend(interaction, generation_id)
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

# =========================================================================
# execute_imagine — Modularized into services/generation_service.py
# =========================================================================

# /imagine and its autocompletes - Modularized into cogs/imagine_cog.py



# =========================================================================
# /bertflow (Bert's Krea 2 Photorealism Workflow) - Modularized into cogs/krea_cog.py & services/krea_service.py
# =========================================================================



# /prompt group, /free, and /purge-vram - Modularized into cogs/system_cog.py & services/system_service.py




# /upscale - Modularized into cogs/upscale_cog.py & services/upscale_service.py

# /video, /ltx, and 'Animate to Video' context menu - Modularized into cogs/video_cog.py & services/video_service.py




# /diagnostics - Modularized into cogs/system_cog.py & services/system_service.py







# handle_generate_described and handle_update_describe_view imported from services.vision_service


# handle_update_blend_view, handle_submit_edit_blend_prompts, handle_reblend imported from services.generation_service

# handle_generate_blended imported from services.generation_service
# execute_blend_generation imported from services.generation_service




# /study, Adopt Post / Image, and build_blend_workflow - Modularized into cogs/imagine_cog.py & services/generation_service.py




# /queue and /variation_mode - Modularized into cogs/system_cog.py & services/system_service.py

# handle_stasis_pause, handle_stasis_resume, run_resumed_generation imported from services.generation_service




# /style, /cui-start, /cui-stop, /cui-status, /negative, /models, /scan_models
# Modularized into cogs/system_cog.py & services/system_service.py




_instance_lock_socket = None

def acquire_instance_lock(port: int = 48123, silent: bool = False) -> bool:
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
        if not silent:
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



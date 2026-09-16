"""
Generation Service for Shallot-CUI Bot.
Handles workflow synthesis, parameter compilation, and image generation pipeline operations.
Modularized from bot.py to adhere to Rule 2 (Modular Cog & Service Architecture).
"""

import os
import io
import re
import json
import copy
import time
import random
import logging
import asyncio
from datetime import datetime
from collections import OrderedDict
from typing import Optional, List, Dict, Any, Tuple

import aiohttp
import discord
from discord import app_commands

from config import (
    COMFYUI_ADDRESS,
    COMFYUI_CHECKPOINT,
    DEFAULT_NEGATIVE_PROMPT,
    CHECKPOINT_CONFIGS,
    PipelineDefaults,
    get_checkpoint_display_name,
)
from comfy_client import ComfyClient, StasisInterruptException
import db
import model_architecture
from error_handler import (
    error_handler,
    ErrorCategory,
    ErrorSeverity,
    AutoFixAction,
    AutoFixResult,
)
from image_utils import (
    create_grid,
    get_quadrant_bytes_async,
    save_quadrant_images_async,
    upscale_isolated_image_async,
    embed_metadata_async,
    calculate_outpaint_padding_async,
    crop_to_aspect_ratio_async,
    format_image_filename,
    get_checkpoint_abbrev,
    get_dated_save_prefix,
    boost_image_vibrancy_and_contrast_async,
)
from parsers import (
    clean_quadrant_prompts,
    truncate_prompt,
    parse_seed,
    parse_stylize,
    parse_sref,
    parse_cref,
    parse_aspect_ratio,
    parse_loras,
    parse_magic_prompt,
    parse_smart_prompt,
    parse_powerhouse_prompt,
    parse_freeu_prompt,
    expand_dynamic_prompt,
    apply_magic_enhancement,
    apply_smart_magic_and_sref,
    apply_loras_to_workflow,
    apply_ipadapter_to_workflow,
    apply_face_detailer_to_workflow,
    deduplicate_intro_quality_tags,
    LOCKED_STYLE_PRESETS,
)
from characters import get_character, mask_character_in_prompt
from views import (
    GridButtons,
    UpscaleButtons,
    IsolatedImageButtons,
    RemixModal,
    CancelGenerationView,
    StasisControlsView,
    StasisPausedView,
    BlendButtons,
    EditBlendPromptModal,
    build_blend_embed,
    build_blend_complete_embed,
    build_blended_image_embed,
)
from core_helpers import (
    safe_defer,
    download_image,
    send_followup_fallback,
    send_error_fallback,
    edit_original_fallback,
    edit_message_fallback,
    _update_button_state,
    get_active_bot,
    get_active_architecture,
    set_active_architecture,
)
from services.system_service import settings

logger = logging.getLogger("DiscordBot.GenerationService")

# ComfyClient Proxy
_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

def set_comfy_client(client: ComfyClient):
    global _comfy_client
    _comfy_client = client

def get_comfy_client() -> ComfyClient:
    bot = get_active_bot()
    if bot and hasattr(bot, "comfy_client") and bot.comfy_client:
        return bot.comfy_client
    return _comfy_client

class _ComfyClientProxy:
    def __getattr__(self, name):
        client = get_comfy_client()
        return getattr(client, name)

comfy_client = _ComfyClientProxy()


# Active generations SQLite Proxy with bounded LRU in-memory caching
class ActiveGenerationsProxy:
    def __init__(self, max_size=500):
        self._cache = OrderedDict()
        self._max_size = max_size

    def __getitem__(self, key):
        k = str(key)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        val = db.get_generation(k)
        if val is None:
            raise KeyError(key)
        self._cache[k] = val
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        return val

    def get(self, key, default=None):
        k = str(key)
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        val = db.get_generation(k)
        if val is not None:
            self._cache[k] = val
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return val
        return default

    def __setitem__(self, key, value):
        k = str(key)
        self._cache[k] = value
        self._cache.move_to_end(k)
        if len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
        db.save_generation(k, value)

    def __contains__(self, key):
        k = str(key)
        if k in self._cache:
            return True
        return db.get_generation(k) is not None

active_generations = ActiveGenerationsProxy()

def get_generation(generation_id: str) -> dict:
    """Get generation data by ID from SQLite database proxy."""
    return active_generations.get(generation_id)

def save_generations():
    pass


def build_blend_workflow(
    image_filenames: list, 
    prompt: str, 
    neg_prompt: str, 
    selected_model: str, 
    width: int, 
    height: int, 
    seed: int, 
    cfg: float, 
    workflow_template: dict = None, 
    comp_strength: str = None
) -> dict:
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
    
    # Check for per-checkpoint custom configurations (samplers, CFG, steps, negative addons)
    ckpt_cfg = CHECKPOINT_CONFIGS.get(selected_model, {})
    if ckpt_cfg:
        if "sampler_name" in ckpt_cfg:
            workflow["3"]["inputs"]["sampler_name"] = ckpt_cfg["sampler_name"]
        if "scheduler" in ckpt_cfg:
            workflow["3"]["inputs"]["scheduler"] = ckpt_cfg["scheduler"]
        if "steps" in ckpt_cfg:
            workflow["3"]["inputs"]["steps"] = ckpt_cfg["steps"]
        if "cfg" in ckpt_cfg and (cfg is None or cfg == 4.0 or cfg == 3.5):
            effective_cfg = ckpt_cfg["cfg"]
        else:
            effective_cfg = cfg if (cfg is not None and 1.0 <= cfg <= 8.0) else 4.0
        if ckpt_cfg.get("negative_addon"):
            neg_prompt = f"{neg_prompt}, {ckpt_cfg['negative_addon']}" if neg_prompt else ckpt_cfg["negative_addon"]
    else:
        effective_cfg = cfg if (cfg is not None and 1.0 <= cfg <= 8.0) else 3.5

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
    num_images = len(image_filenames)
    norm_comp = str(comp_strength).lower() if comp_strength is not None else None

    if norm_comp == "style":
        ip_weight = 0.20 if num_images == 1 else round(min(0.15, 0.25 / num_images), 2)
        end_at = 0.65
        weight_type = "ease out"
    elif norm_comp == "low":
        ip_weight = 0.35 if num_images == 1 else round(min(0.25, 0.40 / num_images), 2)
        end_at = 0.75
        weight_type = "ease in-out"
    elif norm_comp == "med":
        ip_weight = 0.50 if num_images == 1 else round(min(0.35, 0.55 / num_images), 2)
        end_at = 0.80
        weight_type = "ease in-out"
    elif norm_comp == "high":
        ip_weight = 0.65 if num_images == 1 else round(min(0.45, 0.70 / num_images), 2)
        end_at = 0.85
        weight_type = "linear"
    elif is_img2img:
        ip_weight = 0.35 if num_images == 1 else round(min(0.25, 0.40 / num_images), 2)
        end_at = 0.75
        weight_type = "ease in-out"
    else:
        ip_weight = 0.55 if num_images == 1 else round(min(0.35, 0.60 / num_images), 2)
        end_at = 0.85
        weight_type = "linear"

    neg_base = neg_prompt or db.get_negative_prompt(0)
    p_lower = prompt.lower()
    
    extra_neg_parts = ["overexposed", "pale", "washed out", "faded colors", "bloom", "white out"]
    sketch_keywords = ["sketch", "pencil", "drawing", "line art", "lineart", "ink", "monochrome", "charcoal", "cross-hatching", "hatching", "woodcut"]
    if not any(k in p_lower for k in sketch_keywords):
        extra_neg_parts.append("line art only, sketch")
        
    enhanced_neg = neg_base + ", " + ", ".join(extra_neg_parts)

    if any(k in p_lower for k in ["semi-realism", "photorealistic", "realism", "--sr"]):
        enhanced_neg = re.sub(r'\b(?:photorealistic|realistic|photorealism)\b,?', '', enhanced_neg, flags=re.IGNORECASE)

    if any(k in str(selected_model).lower() for k in ["illustrious", "wai", "hyphoria", "nai", "furry", "anime"]):
        enhanced_neg = re.sub(r'\banime(?:\s+style|\s+girl)?\b,?', '', enhanced_neg, flags=re.IGNORECASE)

    enhanced_neg = re.sub(r',\s*,+', ', ', enhanced_neg).strip(' ,')
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

    await save_quadrant_images_async(generation_id, images)

    sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
    grid_file_io = await asyncio.to_thread(create_grid, images, prompt, neg_prompt, seed, width, height)
    is_flux = gen_data.get("is_flux", False)
    ckpt_abbrev = get_checkpoint_abbrev(selected_model)
    if is_flux:
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

    title_txt = "Flux Grid Complete" if gen_data.get("is_flux") else "Image Generation Complete"
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
            bot = get_active_bot()
            channel = interaction.channel or (await bot.fetch_channel(interaction.channel_id) if bot else None)
            if channel:
                file.fp.seek(0)
                await channel.send(content=content, embed=embed, file=file, view=view)
                posted = True
        except Exception as send_err:
            logger.warning(f"Could not send message via channel.send ({send_err}). Falling back to followup.")

    if not posted and interaction:
        await send_followup_fallback(interaction, content=content, embed=embed, file=file, view=view)


async def handle_upscale(interaction: discord.Interaction, generation_id: str, index: int, force_new_seed: bool = False, upscale_scale: str = "2.0"):
    await safe_defer(interaction, thinking=True)

    clicked_custom_id = f"upscale:{generation_id}:{index}"
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find generation session data. It may have expired or the bot was restarted.", ephemeral=True)
        return
        
    original_prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    prompt = gen_data.get("prompt", "")
    neg_prompt = gen_data.get("neg_prompt", gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT))
    base_seed = gen_data.get("seed", 12345)
    target_seed = random.randint(1, 1125899906842624) if force_new_seed else (base_seed + index - 1)
    
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
        upscale_factor = 2.0

    q_bytes = await get_quadrant_bytes_async(generation_id, index)
    q_filename = None

    if q_bytes:
        try:
            upload_result = await comfy_client.upload_image(q_bytes, f"upscale_input_{generation_id}_{index}.png")
            q_filename = upload_result.get("name")
            logger.info(f"Uploaded quadrant {index} for detailed upscale: {q_filename}")
        except Exception as e:
            logger.error(f"Failed to upload quadrant image to ComfyUI for detail upscale: {e}")

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
                denoise_val = PipelineDefaults.UPSCALE_DENOISE_FLUX_SUBTLE if upscale_factor <= 1.3 else PipelineDefaults.UPSCALE_DENOISE_FLUX_MODERATE
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

            if "10" in workflow and workflow["10"].get("class_type") == "LatentUpscaleBy":
                workflow["10"]["inputs"]["scale_by"] = upscale_factor

            is_blend = gen_data.get("is_blend", False) or "caption" in gen_data or "uploaded_image_name" in gen_data
            is_flux = gen_data.get("is_flux", False)

            if is_blend:
                subfolder = "blend"
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
                workflow["11"]["inputs"]["denoise"] = PipelineDefaults.UPSCALE_DENOISE_SDXL
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
    await interaction.followup.send(f"{status_lbl} Image {index} (Seed: {target_seed}, Denoise: {PipelineDefaults.UPSCALE_DENOISE_SDXL})...", ephemeral=True)
    
    try:
        images = await comfy_client.generate(workflow, timeout=14400)
        if not images:
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
            await interaction.followup.send("ComfyUI did not return any image.", ephemeral=True)
            return
        
        out_w = int(round((width * upscale_factor) / 64) * 64)
        out_h = int(round((height * upscale_factor) / 64) * 64)
        highres_file_io = await embed_metadata_async(images[0], expanded_prompt, neg_prompt, target_seed, out_w, out_h)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        ckpt_abbrev = get_checkpoint_abbrev(checkpoint)
        if is_blend:
            upscale_prefix = f"blend_{ckpt_abbrev}_upscale_{index}"
        elif is_flux:
            upscale_prefix = f"flux_{ckpt_abbrev}_upscale_{index}"
        else:
            upscale_prefix = f"upscale_{ckpt_abbrev}_{index}"
        file = discord.File(fp=highres_file_io, filename=format_image_filename(upscale_prefix, target_seed, "png", sref=sref_code))

        sref_info = gen_data.get("sref_info")
        desc_lines = [
            f"**Prompt:** {truncate_prompt(expanded_prompt, 250)}",
            f"**Model:** {checkpoint}",
            f"**Detail Denoise:** {PipelineDefaults.UPSCALE_DENOISE_SDXL}",
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

        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)

    except Exception as e:
        entry = error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="handle_upscale",
            source_file="generation_service.py",
            severity=ErrorSeverity.ERROR,
            context={"generation_id": generation_id, "index": index, "checkpoint": checkpoint, "width": width, "height": height}
        )
        recipe = error_handler.find_recipe(str(e), category=ErrorCategory.WORKFLOW)
        if recipe and recipe.action == AutoFixAction.RETRY_REDUCED_RES and error_handler.can_retry(recipe, generation_id):
            error_handler.record_retry(recipe, generation_id)
            await send_error_fallback(interaction, "⚠️ Generation encountered an issue. Retrying with auto-fix adjustment (reduced resolution)...")
            try:
                if "5" in workflow:
                    workflow["5"]["inputs"]["width"] = int(width * 0.75)
                    workflow["5"]["inputs"]["height"] = int(height * 0.75)
                retry_images = await comfy_client.generate(workflow, timeout=14400)
                if retry_images:
                    out_w, out_h = int(round((width * upscale_factor) / 64) * 64), int(round((height * upscale_factor) / 64) * 64)
                    highres_file_io = await embed_metadata_async(retry_images[0], original_prompt, neg_prompt, target_seed, out_w, out_h)
                    file = discord.File(fp=highres_file_io, filename=format_image_filename(f"upscale_{index}", target_seed, "png"))
                    embed = discord.Embed(
                        title=f"Detailed High-Res Upscale Image {index} (Auto-Fixed)", 
                        description=f"**Prompt:** {truncate_prompt(original_prompt, 250)}\n**Model:** {checkpoint}\n**Detail Denoise:** {PipelineDefaults.UPSCALE_DENOISE_SDXL}\n**Seed:** {target_seed}\n**Size:** {out_w}x{out_h}\n*Note: Reduced resolution auto-fix applied.*"
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
    await safe_defer(interaction)

    clicked_custom_id = f"upscale:{generation_id}:{index}"
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    q_bytes = await get_quadrant_bytes_async(generation_id, index)
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

    q_bytes, width, height = await upscale_isolated_image_async(q_bytes, target_w=width, target_h=height)

    try:
        metadata_io = await embed_metadata_async(q_bytes, original_prompt, neg_prompt, target_seed, width, height)
        png_bytes = metadata_io.getvalue()

        try:
            comfy_output_path = os.getenv("COMFYUI_OUTPUT_PATH", "C:/ComfyUI/ComfyUI/output")
            now = datetime.now()
            mm = now.strftime("%m")
            dd = now.strftime("%d")
            time_str = now.strftime("%Y%m%d_%H%M%S")
            is_blend = gen_data.get("is_blend", False) or "caption" in gen_data or "uploaded_image_name" in gen_data
            if is_blend:
                sub = "blend"
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

        metadata_io.seek(0)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        if is_blend:
            iso_prefix = f"blend_{ckpt_abbrev}_{index}"
        elif gen_data.get('is_flux') or "flux" in str(checkpoint).lower():
            iso_prefix = f"flux_{ckpt_abbrev}_{index}"
        else:
            iso_prefix = f"imagine_{ckpt_abbrev}_{index}"
        file = discord.File(fp=metadata_io, filename=format_image_filename(iso_prefix, target_seed, "png", sref=sref_code))

        files_to_send = [file]

        user_name = interaction.user.display_name if (interaction and interaction.user) else "User"
        user_id = interaction.user.id if (interaction and interaction.user) else None
        ref_image_url = gen_data.get("image_url")
        comp_strength = gen_data.get("comp_strength", "style")
        char_choice = gen_data.get("char_choice")
        sr_choice = gen_data.get("sr")

        embed = build_blended_image_embed(
            index=index,
            display_prompt=original_prompt,
            checkpoint=checkpoint,
            target_seed=target_seed,
            width=width,
            height=height,
            comp_strength=comp_strength,
            cfg=gen_data.get("cfg", 4.0),
            sref_info=sref_info,
            char_choice=char_choice,
            sr_choice=sr_choice,
            user_name=user_name,
            user_id=user_id,
            image_url=ref_image_url,
            is_blend=is_blend
        )

        content_txt = f"✨ **Blended Image {index}:** {truncate_prompt(original_prompt, 100)}" if is_blend else f"**Isolated Image {index}:** {truncate_prompt(original_prompt, 100)}"

        has_sref = sref_info is not None and "code" in sref_info
        view = IsolatedImageButtons(generation_id, index, has_sref=has_sref, is_blend=is_blend)
        await send_followup_fallback(interaction, content=content_txt, embed=embed, files=files_to_send, view=view)
        
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.success, disabled=True)
    except Exception as e:
        logger.error(f"Error isolating image: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
        await send_error_fallback(interaction, f"An error occurred while isolating the image: {e}")


async def handle_variation(interaction: discord.Interaction, generation_id: str, index: int, denoise_override: float = None, variation_type: str = None):
    """Generate a new 4-image grid as a TRUE img2img variation of the selected quadrant."""
    await safe_defer(interaction, thinking=True)

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

    q_bytes = await get_quadrant_bytes_async(generation_id, index)
    q_filename = None

    if q_bytes:
        try:
            upload_res = await comfy_client.upload_image(q_bytes, f"var_input_{generation_id}_{index}.png")
            q_filename = upload_res.get("name")
            logger.info(f"Uploaded quadrant {index} for img2img variation: {q_filename}")
        except Exception as e:
            logger.error(f"Failed to upload quadrant image to ComfyUI: {e}")

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
    denoise_val = PipelineDefaults.UPSCALE_DENOISE_SDXL
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
                    denoise_val = denoise_override if denoise_override is not None else PipelineDefaults.VARIATION_DENOISE_HIGH_CHANGE
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

            if q_filename and "30" in workflow:
                workflow["30"]["inputs"]["image"] = q_filename
                if denoise_override is not None:
                    denoise_val = denoise_override
                else:
                    current_mode = settings.get("variation_mode", "high")
                    denoise_val = PipelineDefaults.VARIATION_DENOISE_VERY_HIGH if current_mode == "very_high" else PipelineDefaults.VARIATION_DENOISE_SUBTLE_CHANGE
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

        await save_quadrant_images_async(new_gen_id, images)

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

    if gen_data.get("is_bertflow"):
        from services.krea_service import handle_bertflow_reroll
        await handle_bertflow_reroll(interaction, generation_id)
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
        "is_flux": is_flux,
        "is_com": is_com,
        "is_sdxl_powerhouse": is_sdxl_powerhouse,
        "guidance": gen_data.get("guidance", 3.5),
        "freeu": gen_data.get("freeu", True),
        "is_face_detailer": gen_data.get("is_face_detailer", False),
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

        await save_quadrant_images_async(new_gen_id, images)

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

        reroll_title = "Re-rolled Generation"
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


async def handle_cancel_generation(interaction: discord.Interaction, generation_id: str):
    """Cancels/interrupts an active generation in ComfyUI and updates the status message."""
    try:
        success = await comfy_client.pause_generation(generation_id)
        if success:
            await edit_original_fallback(interaction, content="🛑 **Generation cancelled by user.**", view=None)
        else:
            await interaction.response.send_message("⚠️ Job is no longer active or already completed.", ephemeral=True)
    except Exception as e:
        logger.error(f"Error handling cancel generation {generation_id}: {e}")
        await send_error_fallback(interaction, f"Failed to cancel generation: {e}")


async def handle_remix(interaction: discord.Interaction, generation_id: str):
    """Opens a modal pre-filled with the generation prompt and seed for tweaking."""
    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.response.send_message("❌ Generation session data expired.", ephemeral=True)
        return

    if gen_data.get("is_bertflow"):
        from services.krea_service import handle_bertflow_remix
        await handle_bertflow_remix(interaction, generation_id)
        return

    orig_p = gen_data.get("original_prompt") or gen_data.get("prompt") or ""
    orig_seed = gen_data.get("seed")

    async def remix_callback(inter: discord.Interaction, g_id: str, new_prompt: str, new_seed: int):
        await safe_defer(inter, thinking=True)
        p = new_prompt
        if new_seed is not None and "--seed" not in p:
            p = f"{p} --seed {new_seed}"
        await execute_imagine(
            inter,
            prompt=p,
            negative_prompt=gen_data.get("negative_prompt"),
            checkpoint=gen_data.get("checkpoint"),
            magic_prompt=False,
            is_flux=gen_data.get("is_flux", False),
            is_sdxl_powerhouse=gen_data.get("is_sdxl_powerhouse", False),
            guidance=gen_data.get("guidance", 3.5),
            freeu=gen_data.get("freeu", True),
            is_face_detailer=gen_data.get("is_face_detailer", False)
        )

    modal = RemixModal(generation_id, initial_prompt=orig_p, initial_seed=orig_seed, on_submit_callback=remix_callback)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_modal(modal)
    except discord.HTTPException as e:
        if e.code != 40060:
            logger.warning(f"Failed to send Remix modal: {e}")


async def handle_outpaint(interaction: discord.Interaction, generation_id: str, index: int, target_ratio: str):
    """Outpaint an image to expanding aspect ratio (16:9, 21:9) or Zoom Out (1.5x, 2.0x)."""
    await safe_defer(interaction, thinking=True)

    clicked_custom_id = interaction.data.get("custom_id", "")
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id) or {}
    raw_prompt = gen_data.get("prompt", "")
    original_prompt = gen_data.get("original_prompt", raw_prompt)

    cleaned_p, _ = parse_magic_prompt(raw_prompt)
    cleaned_p, _ = parse_seed(cleaned_p)
    cleaned_p, cfg_val, _ = parse_stylize(cleaned_p)
    cleaned_p, _, _, _ = parse_sref(cleaned_p)
    cleaned_p, _, _ = parse_aspect_ratio(cleaned_p, COMFYUI_CHECKPOINT)

    orig_quadrant_seed = gen_data.get("seed", 0) + index - 1
    expanded_p = expand_dynamic_prompt(cleaned_p, random.Random(orig_quadrant_seed))
    expanded_original = expand_dynamic_prompt(original_prompt, random.Random(orig_quadrant_seed))

    neg_prompt = gen_data.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT)
    checkpoint = gen_data.get("checkpoint", COMFYUI_CHECKPOINT)
    cfg = gen_data.get("cfg", cfg_val if cfg_val is not None else 4.5)
    loras = gen_data.get("loras")
    if not loras:
        _, loras = parse_loras(original_prompt or raw_prompt, is_flux=False)
    seed = random.randint(1, 1125899906842624)

    q_bytes = await get_quadrant_bytes_async(generation_id, index)
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

    try:
        left, top, right, bottom, input_img_bytes, out_w, out_h = await calculate_outpaint_padding_async(q_bytes, target_ratio)
    except Exception as e:
        logger.error(f"Error calculating outpaint padding: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send(f"Failed to calculate outpaint canvas: {e}", ephemeral=True)
        return

    try:
        up_res = await comfy_client.upload_image(input_img_bytes, f"outpaint_input_{generation_id}_{index}.png")
        img_filename = up_res.get("name")
    except Exception as e:
        logger.error(f"Error uploading image to ComfyUI: {e}")
        await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
        await interaction.followup.send(f"Failed to upload image to ComfyUI: {e}", ephemeral=True)
        return

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
        
        workflow["31"]["inputs"]["left"] = left
        workflow["31"]["inputs"]["top"] = top
        workflow["31"]["inputs"]["right"] = right
        workflow["31"]["inputs"]["bottom"] = bottom
        workflow["31"]["inputs"]["feathering"] = 64
        
        workflow["3"]["inputs"]["seed"] = seed
        workflow["3"]["inputs"]["cfg"] = cfg
        workflow["3"]["inputs"]["denoise"] = 0.80
        workflow["9"]["class_type"] = "PreviewImage"
        workflow["9"]["inputs"].pop("filename_prefix", None)

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

        await save_quadrant_images_async(new_gen_id, [images[0]])

        out_file_io = await embed_metadata_async(images[0], expanded_original, neg_prompt, seed, out_w, out_h)
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

    base_seed = gen_data["seed"]
    target_seed = base_seed + index - 1
    raw_prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    
    cleaned_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+[^\s]+(?:\s*\([^)]*\))?', '', raw_prompt, flags=re.IGNORECASE).strip()
    
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

    expanded_p = expand_dynamic_prompt(parsed_p, random.Random(target_seed))
    if is_magic:
        expanded_p = apply_magic_enhancement(expanded_p, target_seed)

    if prepend_quality and not expanded_p.startswith("masterpiece"):
        expanded_p = f"masterpiece, best quality, absurdres. {expanded_p}"

    sref_image_name = None
    if sref_url:
        try:
            sref_bytes = await download_image(sref_url)
            upload_res = await comfy_client.upload_image(sref_bytes, "sref_from_url.png")
            sref_image_name = upload_res.get("name")
        except Exception as e:
            logger.error(f"Failed to fetch style reference URL: {e}")

    is_flux = gen_data.get("is_flux") or (checkpoint and "flux" in checkpoint.lower())

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

        await save_quadrant_images_async(new_gen_id, [images[0]])

        metadata_io = await embed_metadata_async(images[0], new_full_prompt, neg_prompt, target_seed, width, height)
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


async def handle_stasis_pause(interaction: discord.Interaction, generation_id: str, user_id: int):
    if interaction.user.id != user_id and not interaction.user.guild_permissions.administrator:
        await interaction.response.send_message("Only the user who started this generation can pause it.", ephemeral=True)
        return
        
    await interaction.response.defer()
    
    success = await comfy_client.pause_generation(generation_id)
    if success:
        gen_data = get_generation(generation_id) or {}
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

    gen_data["status"] = "queued"
    gen_data["prompt_ids"] = []
    active_generations[generation_id] = gen_data
    save_generations()

    prompt = gen_data.get("original_prompt", gen_data.get("prompt", ""))
    embed = discord.Embed(
        title="🔄 Resuming Generation...",
        description=f"**Prompt:** {truncate_prompt(prompt, 250)}\n\nQueuing workflows again on ComfyUI..."
    )
    view = StasisControlsView(generation_id, user_id)
    await edit_message_fallback(interaction, interaction.message.id, embed=embed, view=view)

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
            source_file="generation_service.py",
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


async def handle_update_blend_view(interaction: discord.Interaction, generation_id: str, new_ar: str = None, new_sr = None, new_oga: bool = None, new_model: str = None, new_comp: str = None, new_sref = None, new_char: str = None, tab: str = None):
    """Updates the interactive buttons and settings on the /blend result embed."""
    gen_data = get_generation(generation_id) or {}
    if new_ar is not None:
        gen_data["ar"] = new_ar
    if new_sr is not None:
        gen_data["sr"] = new_sr
    if new_char is not None:
        gen_data["char_choice"] = new_char
        gen_data["oga"] = (new_char == "ogarla")
    elif new_oga is not None:
        gen_data["oga"] = new_oga
        if new_oga:
            gen_data["char_choice"] = "ogarla"
        elif gen_data.get("char_choice") == "ogarla":
            gen_data["char_choice"] = "none"
    if new_model is not None:
        gen_data["model_choice"] = new_model
    if new_comp is not None:
        gen_data["comp_strength"] = new_comp
    if new_sref is not None:
        gen_data["sref_rand"] = new_sref
    if tab is not None:
        gen_data["blend_tab"] = tab

    db.save_generation(generation_id, gen_data)

    author_str = gen_data.get("author_str", interaction.user.name if (interaction and interaction.user) else "User")
    image_url = gen_data.get("image_url")
    embed = build_blend_embed(gen_data, author_str=author_str, image_url=image_url)
    user_favs = db.get_favorite_styles(interaction.user.id) if (interaction and interaction.user) else []
    
    view = BlendButtons(
        generation_id=generation_id,
        ar=gen_data.get("ar", "16:9"),
        sr=gen_data.get("sr", True),
        oga=gen_data.get("oga", False),
        model_choice=gen_data.get("model_choice", "wai"),
        comp_strength=gen_data.get("comp_strength", "style"),
        sref_rand=gen_data.get("sref_rand", "nosref"),
        char_choice=gen_data.get("char_choice", "none"),
        tab=gen_data.get("blend_tab", "canvas"),
        user_favorites=user_favs
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
    char_choice = gen_data.get("char_choice", "ogarla" if oga else "none")
    tab = gen_data.get("blend_tab", "canvas")
    model_choice = gen_data.get("model_choice", "wai")
    comp_strength = gen_data.get("comp_strength", "style")
    sref_rand = gen_data.get("sref_rand", "nosref")

    embed = build_blend_embed(gen_data, author_str=author_str, image_url=image_url, is_edited=True)
    user_favs = db.get_favorite_styles(interaction.user.id) if (interaction and interaction.user) else []
    view = BlendButtons(
        generation_id=generation_id,
        ar=ar,
        sr=sr,
        oga=oga,
        model_choice=model_choice,
        comp_strength=comp_strength,
        sref_rand=sref_rand,
        char_choice=char_choice,
        tab=tab,
        user_favorites=user_favs
    )
    await interaction.response.edit_message(embed=embed, view=view)


async def handle_reblend(interaction: discord.Interaction, generation_id: str):
    """Re-opens the interactive 2-tab Blend Studio for a previously generated blend grid."""
    gen_data = get_generation(generation_id)
    if not gen_data:
        if interaction.response.is_done():
            await interaction.followup.send("❌ Could not find session data for this blend. It may have expired.", ephemeral=True)
        else:
            await interaction.response.send_message("❌ Could not find session data for this blend. It may have expired.", ephemeral=True)
        return

    author_name = interaction.user.display_name if (interaction and interaction.user) else "User"
    user_id = interaction.user.id if (interaction and interaction.user) else 0
    favs = db.get_favorite_styles(user_id) if user_id else []

    embed = build_blend_embed(
        gen_data,
        author_str=author_name,
        image_url=gen_data.get("image_url")
    )
    view = BlendButtons(
        generation_id=generation_id,
        ar=gen_data.get("ar", "16:9"),
        sr=gen_data.get("sr", True),
        oga=gen_data.get("oga", False),
        model_choice=gen_data.get("model_choice", "wai"),
        comp_strength=gen_data.get("comp_strength", "style"),
        sref_rand=gen_data.get("sref_rand", "nosref"),
        char_choice=gen_data.get("char_choice", "none"),
        tab="canvas",
        user_favorites=favs
    )
    if interaction.response.is_done():
        await interaction.followup.send(embed=embed, view=view, ephemeral=True)
    else:
        await interaction.response.send_message(embed=embed, view=view, ephemeral=True)


def _get_blend_executor():
    import sys
    bot_module = sys.modules.get("bot")
    if bot_module and hasattr(bot_module, "execute_blend_generation"):
        return getattr(bot_module, "execute_blend_generation")
    bot_inst = get_active_bot()
    if bot_inst and hasattr(bot_inst, "execute_blend_generation"):
        return getattr(bot_inst, "execute_blend_generation")
    return execute_blend_generation


async def handle_generate_blended(interaction: discord.Interaction, generation_id: str, desc_type: str, ar: str = "16:9", use_sr = True, use_oga: bool = False, model_choice: str = "wai", comp_strength: str = "style", use_sref_rand = "nosref", char_choice: str = None):
    """Generates blended image grid(s) using stored caption/detailed description + uploaded base image with chosen settings."""
    await safe_defer(interaction)

    gen_data = get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find blend session data. It may have expired.", ephemeral=True)
        return

    if desc_type == "caption":
        base_prompt = gen_data.get("caption")
    elif desc_type == "blend":
        base_prompt = gen_data.get("detailed_caption") or gen_data.get("caption") or gen_data.get("prompt")
    else:
        base_prompt = gen_data.get("detailed_caption") or gen_data.get("caption") or gen_data.get("prompt")

    if not base_prompt:
        base_prompt = gen_data.get("user_prompt", "") or gen_data.get("extra_details", "")
    if not base_prompt:
        await interaction.followup.send(f"No {desc_type} prompt found in session.", ephemeral=True)
        return

    uploaded_image_name = gen_data.get("uploaded_image_name")
    user_extra_prompt = gen_data.get("user_prompt", "")
    extra_details = gen_data.get("extra_details", "").strip()

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

    base_parts = []
    
    is_anime = any(k in selected_model.lower() for k in ["illustrious", "wai", "hyphoria", "nai", "furry", "anime"])
    
    if use_sr and use_sr != "nosr":
        if isinstance(use_sr, str) and use_sr.startswith("sr"):
            val = use_sr[2:]
            sr_tag = f"--sr.{val}" if (val.isdigit() and len(val) == 2) else f"--{use_sr}"
        else:
            sr_tag = "--sr.90" if is_anime else "--sr.75"
        base_parts.append("Semi-realism,")
    else:
        sr_tag = None

    if char_choice is None:
        char_choice = "ogarla" if use_oga else "none"

    char_tag = None
    if char_choice and char_choice != "none":
        char_flag_map = {
            "ogarla": ("ogarla,", "--ogarla.70"),
            "valerie": ("valerie,", "--valerie.85"),
            "sully": ("sully,", "--sully.85"),
            "cheri": ("cheri,", "--cheri.85"),
            "cheri_e4": ("cheri,", "--cheri4.85"),
            "mageill": ("mageill,", "--mageill.85"),
            "mageill_e6": ("mageill,", "--mageill6.85"),
            "mageill_e4": ("mageill,", "--mageill4.85"),
            "mageill_e3": ("mageill,", "--mageill3.85"),
        }
        if char_choice in char_flag_map:
            prefix, flag = char_flag_map[char_choice]
            base_parts.append(prefix)
            char_tag = flag
        else:
            char_prof = get_character(char_choice)
            if char_prof:
                weight_val = int(round(char_prof.default_weight * 100))
                flag = f"--{char_prof.id}.{weight_val}"
                prefix = f"{char_prof.id},"
                base_parts.append(prefix)
                char_tag = flag
    elif use_oga:
        base_parts.append("ogarla,")
        char_tag = "--ogarla.70"

    base_parts.append(base_prompt)

    if extra_details:
        base_parts.append(extra_details)

    if user_extra_prompt:
        base_parts.append(user_extra_prompt)

    if sr_tag:
        base_parts.append(sr_tag)

    if char_tag:
        base_parts.append(char_tag)

    if ar:
        base_parts.append(f"--ar {ar}")

    executor = _get_blend_executor()

    if isinstance(use_sref_rand, str) and use_sref_rand.startswith("saved_"):
        saved_code = use_sref_rand.replace("saved_", "")
        prompt_parts = list(base_parts)
        prompt_parts.append(f"--sref {saved_code}")
        full_prompt = " ".join(prompt_parts)
        await executor(interaction, uploaded_image_name, prompt=full_prompt, model=selected_model, comp_strength=comp_strength)
        return

    preset_style_key = None
    if isinstance(use_sref_rand, str) and use_sref_rand.startswith("preset_"):
        preset_style_key = use_sref_rand.replace("preset_", "")

    if preset_style_key and preset_style_key in LOCKED_STYLE_PRESETS:
        style_info = LOCKED_STYLE_PRESETS[preset_style_key]
        base_parts.append(style_info["positive"])
        full_prompt = " ".join(base_parts)
        preset_neg = style_info.get("negative")
        custom_neg = f"{DEFAULT_NEGATIVE_PROMPT}, {preset_neg}" if preset_neg else None
        await executor(interaction, uploaded_image_name, prompt=full_prompt, negative_prompt=custom_neg, model=selected_model, comp_strength=comp_strength)
        return

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
        await executor(interaction, uploaded_image_name, prompt=full_prompt, model=selected_model, comp_strength=comp_strength)
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
            await executor(interaction, uploaded_image_name, prompt=full_prompt, model=selected_model, comp_strength=comp_strength)


async def execute_blend_generation(interaction: discord.Interaction, uploaded_image_name: str, prompt: str, negative_prompt: str = None, model: str = None, comp_strength: str = "style"):
    user_id = interaction.user.id if (interaction and interaction.user) else 0
    neg_prompt = negative_prompt or db.get_negative_prompt(user_id)
    selected_model = model or COMFYUI_CHECKPOINT
    generation_id = str(random.randint(100000, 999999))

    cleaned_prompt, magic_flag = parse_magic_prompt(prompt)
    is_magic = magic_flag
    
    cleaned_prompt, user_seed = parse_seed(cleaned_prompt)
    seed = user_seed if user_seed is not None else random.randint(1, 1125899906842624)
    
    cleaned_prompt, cfg, prepend_quality = parse_stylize(cleaned_prompt)
    cleaned_prompt, sref_url, sref_weight, sref_info = parse_sref(cleaned_prompt)
    cleaned_prompt, cref_url, cref_weight = parse_cref(cleaned_prompt)

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
        if not re.search(r'\b(?:masterpiece|best quality)\b', cleaned_prompt, flags=re.IGNORECASE):
            cleaned_prompt = f"masterpiece, best quality, absurdres. {cleaned_prompt}"
    cleaned_prompt = deduplicate_intro_quality_tags(cleaned_prompt)

    use_img2img = (comp_strength != "style")
    workflow_path = "workflows/img2img_lowres.json" if use_img2img else "workflows/blend_lowres.json"

    try:
        with open(workflow_path, "r", encoding="utf-8") as f:
            workflow = json.load(f)
    except Exception as e:
        logger.error(f"Error loading workflow template for blend: {e}")
        await send_followup_fallback(interaction, content="Failed to load workflow template.")
        return

    if cref_image_name:
        workflow = apply_ipadapter_to_workflow(workflow, cref_image_name, weight=cref_weight, node_prefix="cref")

    final_image_name = uploaded_image_name
    if use_img2img:
        denoise_val = PipelineDefaults.VARIATION_DENOISE_MAP.get(comp_strength, PipelineDefaults.VARIATION_DENOISE_MED_CHANGE)
        
        try:
            view_url = f"http://{COMFYUI_ADDRESS}/view?filename={uploaded_image_name}&type=input"
            async with aiohttp.ClientSession() as session:
                async with session.get(view_url) as resp:
                    if resp.status != 200:
                        raise Exception(f"ComfyUI view returned status {resp.status}")
                    orig_image_bytes = await resp.read()
            
            cropped_bytes = await crop_to_aspect_ratio_async(orig_image_bytes, width, height)
            target_filename = f"blend_crop_{generation_id}_{width}_{height}.png"
            upload_res = await comfy_client.upload_image(cropped_bytes, target_filename)
            final_image_name = upload_res.get("name", uploaded_image_name)
        except Exception as crop_err:
            logger.error(f"Failed to crop/resize input image for blend aspect ratio: {crop_err}")
            final_image_name = uploaded_image_name

        try:
            workflow["3"]["inputs"]["denoise"] = denoise_val
            workflow["30"]["inputs"]["image"] = final_image_name
        except KeyError as e:
            logger.error(f"Invalid img2img workflow node structure: {e}")
            await send_followup_fallback(interaction, content="Workflow template structure mismatch.")
            return

    try:
        workflow = build_blend_workflow(
            [final_image_name], 
            cleaned_prompt, 
            neg_prompt, 
            selected_model, 
            width, 
            height, 
            seed, 
            cfg, 
            workflow_template=workflow,
            comp_strength=comp_strength
        )
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

    existing_data = get_generation(generation_id) or {}
    gen_data_record = {
        **existing_data,
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
    active_generations[generation_id] = gen_data_record
    db.save_generation(generation_id, gen_data_record)
    save_generations()

    comp_lbls = {"style": "Style Only", "low": "Low Comp", "med": "Medium Comp", "high": "High Comp"}
    comp_info = comp_lbls.get(comp_strength, "Style Only")
    flags_info = f"Seed: {seed}, Model: {selected_model}, Size: {width}x{height}, CFG: {cfg:.1f}, Comp: {comp_info}"
    
    start_time = time.time()
    status_msg = None
    try:
        status_msg = await send_followup_fallback(interaction, content=f"Job submitted (Seed: {seed}) — Queuing blend...")

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

        await save_quadrant_images_async(generation_id, images)

        grid_file_io = await asyncio.to_thread(create_grid, images, cleaned_prompt, neg_prompt, seed, width, height)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        file = discord.File(fp=grid_file_io, filename=format_image_filename("blend_grid", seed, "jpg", sref=sref_code))
        
        elapsed_time = time.time() - start_time
        user_name = interaction.user.display_name if (interaction and interaction.user) else "User"
        ref_image_url = gen_data_record.get("image_url")
        char_choice = gen_data_record.get("char_choice")
        sr_choice = gen_data_record.get("sr")

        embed = build_blend_complete_embed(
            display_prompt=display_prompt,
            selected_model=selected_model,
            seed=seed,
            width=width,
            height=height,
            comp_strength=comp_strength,
            cfg=cfg,
            sref_info=sref_info,
            cref_image_name=cref_image_name,
            cref_weight=cref_weight,
            is_magic=is_magic,
            char_choice=char_choice,
            sr_choice=sr_choice,
            user_name=user_name,
            image_url=ref_image_url,
            elapsed_time=elapsed_time,
            expanded_prompts=expanded_prompts
        )
        has_sref = sref_info is not None and "code" in sref_info
        view = GridButtons(generation_id, has_sref=has_sref, is_blend=True)
        
        tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
        content = f"{tag}**Blend:** {truncate_prompt(display_prompt, 100)}"

        transformed = False
        if status_msg and hasattr(status_msg, "id"):
            try:
                await edit_message_fallback(interaction, status_msg.id, content=content, embed=embed, file=file, view=view)
                transformed = True
            except Exception as edit_err:
                logger.info(f"Could not transform blend status message in-place ({edit_err}). Falling back to followup.")

        if not transformed:
            if file:
                file.fp.seek(0)
            await send_followup_fallback(interaction, content=content, embed=embed, file=file, view=view)
    except Exception as e:
        error_handler.log_error(
            e,
            category=ErrorCategory.WORKFLOW,
            source_function="execute_blend_generation",
            source_file="generation_service.py",
            severity=ErrorSeverity.ERROR,
            context={"uploaded_image_name": uploaded_image_name, "prompt": prompt, "model": selected_model, "width": width, "height": height}
        )
        logger.error(f"Error executing blend generation: {e}")
        await send_error_fallback(interaction, f"An error occurred while generating blend images: {e}")


async def execute_imagine(
    interaction: discord.Interaction, 
    prompt: str, 
    negative_prompt: str = None, 
    checkpoint: str = None, 
    style_reference: discord.Attachment = None, 
    magic_prompt: bool = False, 
    favorite_style: str = None, 
    semi_realism: str = None, 
    aspect_ratio: str = None, 
    ogarla: str = None, 
    is_flux: bool = False, 
    is_com: bool = False, 
    is_sdxl_powerhouse: bool = False, 
    guidance: float = 3.5, 
    freeu: bool = True, 
    smart: bool = False, 
    enhancements: str = None, 
    reference_image_url: str = None, 
    reference_image_weight: float = None, 
    original_post_url: str = None, 
    cref_image_name_override: str = None, 
    is_face_detailer: bool = False, 
    character: str = None, 
    model: str = None
):
    if model and not checkpoint:
        checkpoint = model

    target_arch = model_architecture.Architecture.FLUX if (is_flux or is_com) else model_architecture.Architecture.SDXL
    curr_arch = get_active_architecture()
    if curr_arch is not None and curr_arch != target_arch:
        logger.info(f"Switching architecture from {curr_arch} to {target_arch}. Purging ComfyUI VRAM via /free...")
        await comfy_client.free_memory()
    set_active_architecture(target_arch)

    if enhancements:
        enh_val = str(enhancements).lower()
        if "smart" in enh_val or enh_val in ["all", "ultimate"]:
            smart = True
        if "magic" in enh_val or enh_val in ["all", "ultimate"]:
            magic_prompt = True
        if "powerhouse" in enh_val or enh_val in ["all", "ultimate"]:
            is_sdxl_powerhouse = True
        if "no_freeu" in enh_val or "disable_freeu" in enh_val:
            freeu = False

    char_choice = character or ogarla
    if char_choice:
        c_flag = char_choice if char_choice.startswith("--") else f"--{char_choice}"
        prompt = f"{prompt} {c_flag}"

    if semi_realism:
        prompt = re.sub(r'\bSemi-realism(?:,\s*masterpiece,\s*best quality,\s*absurdres\.?)?,?\s*', '', prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r'^[,\s]+', '', prompt).strip()

    prefixes = []
    if semi_realism:
        prefixes.append("Semi-realism, masterpiece, best quality, absurdres.")

    if prefixes:
        prefix_str = " ".join(p if p.endswith(".") else f"{p}," for p in prefixes)
        prompt = f"{prefix_str} {prompt}"

    if semi_realism:
        sr_val = semi_realism if semi_realism.startswith("--") else f"--{semi_realism}"
        prompt = f"{prompt} {sr_val}"

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

    cleaned_prompt, smart_flag = parse_smart_prompt(prompt)
    is_smart = smart_flag or (smart is True)

    cleaned_prompt, magic_flag = parse_magic_prompt(cleaned_prompt)
    is_magic = magic_flag or (magic_prompt is True)

    cleaned_prompt, ph_flag = parse_powerhouse_prompt(cleaned_prompt)
    if ph_flag:
        is_sdxl_powerhouse = True

    cleaned_prompt, no_freeu_flag = parse_freeu_prompt(cleaned_prompt)
    if no_freeu_flag:
        freeu = False

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

        for code in style_codes:
            sub_prompt = f"{prompt} --sref {code}"
            await execute_imagine(
                interaction,
                prompt=sub_prompt,
                negative_prompt=negative_prompt,
                checkpoint=checkpoint,
                style_reference=style_reference,
                magic_prompt=magic_prompt,
                aspect_ratio=aspect_ratio,
                ogarla=ogarla,
                is_flux=is_flux,
                is_com=is_com,
                is_sdxl_powerhouse=is_sdxl_powerhouse,
                guidance=guidance,
                freeu=freeu,
                smart=smart,
                reference_image_url=reference_image_url,
                reference_image_weight=reference_image_weight,
                original_post_url=original_post_url,
                cref_image_name_override=cref_image_name_override,
                is_face_detailer=is_face_detailer,
                character=character
            )
        return

    if prepend_quality:
        if not re.search(r'\b(?:masterpiece|best quality)\b', cleaned_prompt, flags=re.IGNORECASE):
            cleaned_prompt = f"masterpiece, best quality, absurdres. {cleaned_prompt}"
    cleaned_prompt = deduplicate_intro_quality_tags(cleaned_prompt)

    sref_image_name = None
    if style_reference:
        try:
            sref_bytes = await style_reference.read()
            upload_result = await comfy_client.upload_image(sref_bytes, "sref.png")
            sref_image_name = upload_result.get("name")
            logger.info(f"Style reference uploaded: {sref_image_name}")
        except Exception as e:
            logger.error(f"Failed to upload style reference image: {e}")
            await interaction.followup.send("Failed to upload style reference image.")
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

            if not freeu and "20" in workflow:
                fallback_model_src = ["76", 0] if "76" in workflow else ["4", 0]
                if "3" in workflow:
                    workflow["3"]["inputs"]["model"] = fallback_model_src
                if "15" in workflow:
                    workflow["15"]["inputs"]["model"] = fallback_model_src

            if "3" in workflow:
                workflow["3"]["inputs"]["seed"] = seed
                workflow["3"]["inputs"]["steps"] = stage1_steps
                workflow["3"]["inputs"]["cfg"] = stage1_cfg
                workflow["3"]["inputs"]["sampler_name"] = stage1_sampler
                workflow["3"]["inputs"]["scheduler"] = stage1_scheduler

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

    display_prompt = mask_character_in_prompt(prompt)
    if sref_info and "code" in sref_info:
        display_prompt = re.sub(r'[-\u2014\u2013]{1,2}sref\s+random', f"--sref {sref_info['code']}", display_prompt, flags=re.IGNORECASE)

    generation_id = str(random.randint(100000, 999999))
    
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
        "is_flux": is_flux,
        "is_com": is_com,
        "is_sdxl_powerhouse": is_sdxl_powerhouse,
        "guidance": guidance,
        "freeu": freeu,
        "jump_url": original_post_url,
        "is_face_detailer": is_face_detailer
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
        msg = await send_followup_fallback(
            interaction,
            content=f"Job submitted (Seed: {seed}) — Queuing: '{truncate_prompt(display_prompt, 80)}'...",
            view=CancelGenerationView(generation_id)
        )
        
        if msg:
            gen_data = active_generations[generation_id]
            gen_data["message_id"] = msg.id
            active_generations[generation_id] = gen_data
            save_generations()

        last_grid_prog_time = [0.0]
        total_wfs = len(workflows_list)

        def make_grid_progress_cb(wf_idx: int):
            async def on_grid_progress(val, max_val):
                now = time.monotonic()
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
                            await edit_message_fallback(interaction, msg.id, content=prog_content, allow_send_fallback=False)
                    except Exception:
                        pass
            return on_grid_progress

        start_time = time.perf_counter()
        if is_flux:
            results = []
            for idx, wf in enumerate(workflows_list):
                res = await comfy_client.generate(
                    wf,
                    generation_id=generation_id,
                    progress_callback=make_grid_progress_cb(idx),
                    channel_id=interaction.channel_id if interaction else None,
                    message_id=msg.id if msg else None,
                    user_id=interaction.user.id if interaction and interaction.user else None,
                    command_type="flux",
                    metadata={"prompt": prompt, "checkpoint": selected_model}
                )
                results.append(res)
        else:
            tasks = [
                comfy_client.generate(
                    wf,
                    generation_id=generation_id,
                    progress_callback=make_grid_progress_cb(idx),
                    channel_id=interaction.channel_id if interaction else None,
                    message_id=msg.id if msg else None,
                    user_id=interaction.user.id if interaction and interaction.user else None,
                    command_type="imagine",
                    metadata={"prompt": prompt, "checkpoint": selected_model}
                ) for idx, wf in enumerate(workflows_list)
            ]
            results = await asyncio.gather(*tasks)

        elapsed_time = time.perf_counter() - start_time
        t_breakdown = comfy_client.get_execution_timing()
        init_sec = t_breakdown.get("init_duration", 0.0)
        sample_sec = t_breakdown.get("sampling_duration", 0.0)
        post_sec = t_breakdown.get("post_duration", 0.0)

        cmd_label = "flux" if is_flux else "imagine"

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
            metadata={"cfg": cfg, "is_flux": is_flux, "is_com": is_com, "is_sdxl_powerhouse": is_sdxl_powerhouse, "is_face_detailer": is_face_detailer}
        )

        timing_data = {
            "elapsed_time": elapsed_time,
            "init_seconds": init_sec,
            "sampling_seconds": sample_sec,
            "post_seconds": post_sec
        }

        raw_images = [r[0] for r in results if r and len(r) > 0]
        
        is_anime_model = any(k in str(selected_model).lower() for k in ["illustrious", "wai", "hyphoria", "anime", "nai", "furry", "pony"])
        if (use_reference_img2img or reference_image_url or cref_image_name) and not is_anime_model:
            images = await asyncio.gather(*[boost_image_vibrancy_and_contrast_async(img_bytes, saturation=1.22, contrast=1.08) for img_bytes in raw_images])
        else:
            images = raw_images
        
        if len(images) < 4:
            await send_followup_fallback(interaction, content=f"Expected 4 images from generation, but only got {len(images)}.")
            return

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
            source_file="generation_service.py",
            severity=ErrorSeverity.ERROR,
            context={"prompt": prompt, "checkpoint": selected_model, "width": width, "height": height}
        )
        logger.error(f"Error executing imagine command: {e}")
        await send_error_fallback(interaction, f"An error occurred while generating images: {e}")

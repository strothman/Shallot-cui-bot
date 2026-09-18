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
        val = db.get_generation(k)
        if val is not None:
            self._cache[k] = val
            self._cache.move_to_end(k)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return val
        if k in self._cache:
            self._cache.move_to_end(k)
            return self._cache[k]
        raise KeyError(key)

    def get(self, key, default=None):
        k = str(key)
        val = db.get_generation(k)
        if val is not None:
            self._cache[k] = val
            self._cache.move_to_end(k)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
            return val
        return self._cache.get(k, default)

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


# ---------------------------------------------------------------------------
# Re-exported Blend and Grid Action Functions for 100% Backward Compatibility
# ---------------------------------------------------------------------------
from services.blend_generation_service import (
    build_blend_workflow,
    handle_update_blend_view,
    handle_submit_edit_blend_prompts,
    handle_reblend,
    handle_generate_blended,
    execute_blend_generation,
)
from services.grid_actions_service import (
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
)


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
                            await edit_message_fallback(interaction, msg.id, content=prog_content, view=CancelGenerationView(generation_id), allow_send_fallback=False)
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

    except (StasisInterruptException, asyncio.CancelledError):
        logger.info(f"Generation {generation_id} was cancelled/paused.")
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

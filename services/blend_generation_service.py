"""
SDXL Blend Generation Engine, Workflow Compilation, and Synthesis Service for Shallot-CUI Bot.
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
from typing import Optional, List, Dict, Any, Tuple

import aiohttp
import discord
from PIL import Image

import db
from comfy_client import ComfyClient
from core_helpers import (
    safe_defer, 
    send_error_fallback, 
    send_followup_fallback, 
    edit_message_fallback, 
    download_image
)
from error_handler import error_handler, ErrorCategory, ErrorSeverity
from config import (
    CHECKPOINT_CONFIGS, 
    DEFAULT_NEGATIVE_PROMPT, 
    SDXL_CHECKPOINT_CHOICES,
    COMFYUI_CHECKPOINT
)
from parsers import (
    parse_aspect_ratio, 
    parse_loras, 
    apply_loras_to_workflow, 
    parse_sref,
    apply_ipadapter_to_workflow,
    LOCKED_STYLE_PRESETS
)
from characters import get_character, mask_character_in_prompt
from image_utils import (
    create_grid_async, 
    boost_image_vibrancy_and_contrast_async, 
    get_quadrant_bytes_async
)
from views import (
    BlendButtons, 
    GridButtons, 
    build_blend_embed, 
    build_blend_complete_embed, 
    build_blended_image_embed,
    EditBlendPromptModal
)
from services.system_service import settings

logger = logging.getLogger("DiscordBot.BlendGenerationService")


def get_comfy_client():
    from services.generation_service import get_comfy_client as _gcc
    return _gcc()


def get_generation(generation_id: str):
    from services.generation_service import get_generation as _gg
    return _gg(generation_id)


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

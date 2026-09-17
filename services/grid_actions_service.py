"""
Post-Generation Quadrant Isolation, Super-Resolution Upscaling, Variations, Rerolls,
and Lifecycle Stasis Controls for Shallot-CUI Bot.
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
from PIL import Image

import db
import model_architecture
from comfy_client import ComfyClient, StasisInterruptException
from error_handler import (
    error_handler, 
    ErrorCategory, 
    ErrorSeverity, 
    AutoFixAction, 
    AutoFixResult
)
from image_utils import (
    create_grid_async,
    save_quadrant_images_async,
    get_quadrant_bytes_async,
    crop_to_aspect_ratio_async,
    upscale_isolated_image_async,
    calculate_outpaint_padding_async,
    boost_image_vibrancy_and_contrast_async,
    crop_quadrant_from_grid_bytes_async
)
from parsers import (
    parse_aspect_ratio,
    parse_loras,
    apply_loras_to_workflow,
    parse_seed,
    parse_stylize,
    parse_sref,
    parse_cref,
    apply_ipadapter_to_workflow,
    expand_dynamic_prompt,
    parse_magic_prompt,
    apply_magic_enhancement,
    calculate_wan_dimensions,
    extract_positive_prompt,
    apply_face_detailer_to_workflow,
    prepare_bertflow_workflow,
    get_bertflow_unet_model,
)
from characters import get_character, mask_character_in_prompt
from views import (
    GridButtons,
    IsolatedImageButtons,
    UpscaleButtons,
    CancelGenerationView,
    StasisControlsView,
    StasisPausedView,
    SavedSrefSelectView,
    CustomSrefModal,
    RemixModal
)
from core_helpers import (
    safe_defer, 
    send_error_fallback, 
    send_followup_fallback, 
    edit_message_fallback, 
    edit_original_fallback,
    download_image
)
from config import (
    DEFAULT_NEGATIVE_PROMPT,
    CHECKPOINT_CONFIGS,
    SDXL_CHECKPOINT_CHOICES,
    SDXL_ENHANCEMENT_CHOICES,
    COMFYUI_CHECKPOINT
)
from services.system_service import settings

logger = logging.getLogger("DiscordBot.GridActionsService")


def get_comfy_client():
    from services.generation_service import get_comfy_client as _gcc
    return _gcc()


def get_generation(generation_id: str):
    from services.generation_service import get_generation as _gg
    return _gg(generation_id)


def save_generations():
    pass


async def execute_imagine(*args, **kwargs):
    from services.generation_service import execute_imagine as _ei
    return await _ei(*args, **kwargs)


async def complete_grid_generation(*args, **kwargs):
    from services.generation_service import complete_grid_generation as _cgg
    return await _cgg(*args, **kwargs)


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
    """Outpaint an image via directional panning (left, right, up, down) or expanding aspect ratio / zoom."""
    await safe_defer(interaction, thinking=True)

    clicked_custom_id = interaction.data.get("custom_id", "")
    await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.primary, disabled=True)

    gen_data = get_generation(generation_id) or {}
    if not gen_data:
        gen_data = db.get_generation(generation_id) or {}

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

    is_bertflow = bool(gen_data.get("is_bertflow") or gen_data.get("engine") == "krea2")

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

    sref_info = gen_data.get("sref_info")
    cref_image = gen_data.get("cref_image")
    cref_weight = gen_data.get("cref_weight", 0.80)

    if is_bertflow:
        # Architecture-aware Krea 2 / Bertflow outpainting
        try:
            unet_choice = gen_data.get("unet_model") or get_bertflow_unet_model()
            workflow = prepare_bertflow_workflow(
                prompt=expanded_p,
                width=out_w,
                height=out_h,
                seed=seed,
                steps=gen_data.get("steps", 8),
                unet_model=unet_choice,
                wetness_strength=gen_data.get("wetness", -2.0),
                init_image=img_filename,
                comp_strength="medium",
                character=gen_data.get("character"),
                celebrity=gen_data.get("celebrity"),
            )
        except Exception as e:
            logger.error(f"Error preparing Bertflow outpaint workflow: {e}")
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
            await interaction.followup.send(f"Failed to prepare Krea 2 outpaint workflow: {e}", ephemeral=True)
            return
    else:
        # SDXL outpaint with soft feathering and anti-ghosting denoise
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

        try:
            workflow["4"]["inputs"]["ckpt_name"] = checkpoint
            workflow["6"]["inputs"]["text"] = expanded_p
            workflow["7"]["inputs"]["text"] = neg_prompt
            workflow["30"]["inputs"]["image"] = img_filename
            
            workflow["31"]["inputs"]["left"] = left
            workflow["31"]["inputs"]["top"] = top
            workflow["31"]["inputs"]["right"] = right
            workflow["31"]["inputs"]["bottom"] = bottom
            workflow["31"]["inputs"]["feathering"] = 72  # Soft mask feathering to eliminate box seams
            
            workflow["3"]["inputs"]["seed"] = seed
            workflow["3"]["inputs"]["cfg"] = cfg
            workflow["3"]["inputs"]["denoise"] = 0.80  # Controlled denoise to prevent duplicate bodies
            workflow["9"]["class_type"] = "PreviewImage"
            workflow["9"]["inputs"].pop("filename_prefix", None)

            if cref_image:
                workflow = apply_ipadapter_to_workflow(workflow, cref_image, weight=cref_weight, node_prefix="cref")
        except KeyError as e:
            logger.error(f"Invalid outpaint workflow structure: {e}")
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.secondary, disabled=False)
            await interaction.followup.send("Outpaint workflow template has an invalid structure.", ephemeral=True)
            return

    direction_display = {
        "up": "⬆️ Pan Up",
        "down": "⬇️ Pan Down",
        "left": "⬅️ Pan Left",
        "right": "➡️ Pan Right",
        "1.5x": "🔍 Zoom 1.5x",
    }.get(str(target_ratio).lower(), f"Expand {target_ratio}")

    await interaction.followup.send(f"Outpainting ({direction_display}): `{out_w}x{out_h}` (Seed: `{seed}`)...", ephemeral=True)

    try:
        images = await comfy_client.generate(workflow, timeout=14400)
        if not images:
            await _update_button_state(interaction, clicked_custom_id, discord.ButtonStyle.danger, disabled=False)
            await interaction.followup.send("ComfyUI did not return an outpainted image.", ephemeral=True)
            return

        new_gen_id = str(random.randint(100000, 999999))
        new_gen_data = {
            "prompt": expanded_p,
            "original_prompt": expanded_original,
            "negative_prompt": neg_prompt,
            "seed": seed,
            "width": out_w,
            "height": out_h,
            "loras": loras,
            "checkpoint": checkpoint,
            "cfg": cfg,
            "is_bertflow": is_bertflow,
            "engine": "krea2" if is_bertflow else "sdxl",
            "unet_model": gen_data.get("unet_model"),
            "character": gen_data.get("character"),
            "celebrity": gen_data.get("celebrity"),
            "variation_depth": 0,
            "sref_info": sref_info,
            "cref_image": cref_image,
            "cref_weight": cref_weight
        }
        active_generations[new_gen_id] = new_gen_data
        db.save_generation(new_gen_id, new_gen_data)
        save_generations()

        await save_quadrant_images_async(new_gen_id, [images[0]])

        out_file_io = await embed_metadata_async(images[0], expanded_original, neg_prompt, seed, out_w, out_h)
        sref_code = sref_info.get("code") if (sref_info and isinstance(sref_info, dict)) else None
        safe_ratio_name = str(target_ratio).replace(':', '_').replace('.', '_')
        file = discord.File(fp=out_file_io, filename=format_image_filename(f"outpaint_{safe_ratio_name}", seed, "png", sref=sref_code))

        model_label = gen_data.get("unet_model") if is_bertflow else checkpoint
        desc_lines = [
            f"**Prompt:** {truncate_prompt(expanded_original, 250)}",
            f"**Engine:** {'Krea 2 Turbo' if is_bertflow else 'SDXL'}",
            f"**Model:** `{model_label}`",
            f"**Canvas Size:** `{out_w}x{out_h}` ({direction_display})",
            f"**Seed:** `{seed}`"
        ]
        if sref_info and "code" in sref_info:
            desc_lines.append(f"**Style Reference:** --sref {sref_info['code']} ({sref_info['name']})")
        if cref_image:
            desc_lines.append(f"**Character Reference:** --cref (weight: {cref_weight:.2f})")

        embed = discord.Embed(
            title=f"Outpainted Canvas ({direction_display})",
            description="\n".join(desc_lines),
            color=discord.Color.from_rgb(235, 140, 52) if is_bertflow else discord.Color.blurple()
        )
        embed.set_footer(text=f"Requested by {interaction.user.name} (ID: {interaction.user.id})")
        
        # IsolatedImageButtons enables recursive panning and 2x/4x upscale on the outpainted canvas
        view = IsolatedImageButtons(new_gen_id, 1, has_sref=bool(sref_info and "code" in sref_info))

        await send_followup_fallback(interaction, content=f"**Outpaint ({direction_display}):** {truncate_prompt(expanded_original, 100)}", embed=embed, file=file, view=view)
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

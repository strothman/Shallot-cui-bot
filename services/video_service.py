"""
Video & Animation Service for Shallot-CUI Bot.
Handles Wan 2.2 Image-to-Video generation (14B GGUF + RIFE interpolation),
LTX-Video high-speed animation, and video action button callbacks.
"""

import os
import io
import time
import json
import random
import logging
import asyncio
import aiohttp
import discord
from discord import app_commands
from typing import Optional, List, Dict, Any
from PIL import Image

from config import COMFYUI_ADDRESS, DEFAULT_NEGATIVE_PROMPT
from image_utils import (
    QUADRANT_CACHE_DIR,
    format_image_filename,
    get_dated_save_prefix,
)
from comfy_client import ComfyClient
import db
from model_architecture import Architecture
from parsers import parse_video_motion_flags, calculate_wan_dimensions
from services.vision_service import parse_adopted_post
from views import (
    build_video_complete_embed,
    VideoActionView,
    VideoPromptModal,
)
from core_helpers import (
    safe_defer,
    send_followup_fallback,
    send_error_fallback,
    create_progress_bar,
    update_bot_presence,
    get_active_architecture,
    set_active_architecture,
)

logger = logging.getLogger("DiscordBot.VideoService")

_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)


def get_video_settings() -> dict:
    """Loads runtime settings from settings.json with graceful fallback."""
    try:
        if os.path.exists("settings.json"):
            with open("settings.json", "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


async def execute_video_core(
    interaction: discord.Interaction,
    image_bytes: bytes,
    filename: str,
    prompt: str,
    duration: int = 10,
    smoothness: str = "smooth",
    seed: int = None,
    audio: bool = False,
    audio_prompt: str = None,
    client: Optional[ComfyClient] = None,
    **kwargs
):
    """Core logic to generate a Wan 2.2 video from raw image bytes and user settings."""
    comfy = client or _comfy_client
    settings = get_video_settings()

    active_arch = get_active_architecture()
    if active_arch is not None and active_arch != Architecture.WAN:
        logger.info(f"Switching architecture from {active_arch} to {Architecture.WAN}. Purging ComfyUI VRAM via /free...")
        await comfy.free_memory()
    set_active_architecture(Architecture.WAN)

    status_msg = [None]
    try:
        # Parse motion flags and camera directives
        clean_prompt, motion_badges, wan_prompt = parse_video_motion_flags(prompt)

        # Open image with Pillow to auto-detect original width & height (aspect ratio)
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size

        # Resolve video generation parameters
        video_seed = seed if seed is not None else random.randint(1, 1125899906842624)

        # Generate unique generation ID and cache source image for Re-roll / Remix
        gen_id = f"vid_{video_seed}_{int(time.time())}"
        try:
            os.makedirs(QUADRANT_CACHE_DIR, exist_ok=True)
            src_cache_path = os.path.join(QUADRANT_CACHE_DIR, f"{gen_id}_source.png")
            with open(src_cache_path, "wb") as f_src:
                f_src.write(image_bytes)
        except Exception as cache_err:
            logger.debug(f"Could not cache video source image: {cache_err}")

        # Calculate 8GB VRAM optimized dimensions carrying over original aspect ratio
        target_area = settings.get("wan_target_area", 399360)
        width, height = calculate_wan_dimensions(orig_w, orig_h, target_area=target_area)

        # Save metadata to DB for interactive action buttons
        db.save_generation(gen_id, {
            "type": "video",
            "prompt": clean_prompt,
            "raw_prompt": prompt,
            "duration": duration,
            "smoothness": smoothness,
            "seed": video_seed,
            "user_id": interaction.user.id if (interaction and interaction.user) else 0,
            "user_name": interaction.user.name if (interaction and interaction.user) else "User",
            "filename": filename,
            "orig_w": orig_w,
            "orig_h": orig_h,
            "width": width,
            "height": height
        })

        # Upload image to ComfyUI
        logger.info(f"Uploading image {filename} ({orig_w}x{orig_h}) to ComfyUI for video generation...")
        upload_result = await comfy.upload_image(image_bytes, filename)
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

        wan_high_gguf = settings.get("wan_high_gguf", r"gguf\dasiwaWAN22I2V14B_midnightflirtHigh-Q3_K_M.gguf")
        wan_low_gguf = settings.get("wan_low_gguf", r"gguf\dasiwaWAN22I2V14B_midnightflirtLow-Q3_K_M.gguf")
        wan_clip = settings.get("wan_clip", "nsfw_wan_umt5-xxl_fp8_scaled.safetensors")
        wan_clip_vision = settings.get("wan_clip_vision", "clip_vision_h.safetensors")
        wan_vae = settings.get("wan_vae", "wan_2.1_vae.safetensors")
        wan_steps = settings.get("wan_steps", 6)
        wan_cfg = settings.get("wan_cfg", 1.0)
        wan_shift = settings.get("wan_shift", 8.0)
        
        # Natural Speed Duration & Frame Scaling:
        # 10s runs 161 frames natively in Wan 2.2 (161 frames @ 16 FPS = 10.06s real-time motion).
        # 5s runs 81 frames natively in Wan 2.2 (81 frames @ 16 FPS = 5.06s real-time motion).
        # Fast mode outputs native 16 FPS without RIFE. Smooth mode uses 2x RIFE for silky 32 FPS.
        rife_ckpt = settings.get("rife_ckpt", "rife49.pth")
        wan_fps = settings.get("wan_video_fps", 32)

        if duration == 10:
            duration_sec = 10.0
            wan_frames = 161
            if smoothness == "fast":
                rife_multiplier = 1
                out_fps = 16
                use_rife = False
            else:
                rife_multiplier = 2
                out_fps = wan_fps
                use_rife = True
            total_output_frames = wan_frames * rife_multiplier
        elif duration == 5:
            duration_sec = 5.0
            wan_frames = 81
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
            wan_frames = 161 if duration >= 10 else 81
            rife_multiplier = 2 if smoothness != "fast" else 1
            out_fps = wan_fps if smoothness != "fast" else 16
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
            workflow["6"]["inputs"]["text"] = wan_prompt
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

        # Ensure any residual audio nodes are stripped to keep video workflow lightweight
        for nid in ["150", "151", "152"]:
            if nid in workflow:
                del workflow[nid]
        if "9" in workflow and "audio" in workflow["9"]["inputs"]:
            del workflow["9"]["inputs"]["audio"]

        # Setup live progress callback for Discord server presence status & chat embed updates
        last_update_time = [0.0]
        total_steps_done = [0]
        last_val = [0]
        expected_total = wan_steps if wan_steps > 0 else 6

        # Send immediate initial progress embed so user sees instant feedback
        init_bar = create_progress_bar(0, expected_total)
        badges_line = f"**Directives:** {' • '.join(motion_badges)}\n" if motion_badges else ""
        init_embed = discord.Embed(
            title="🎬 Generating Wan 2.2 Video...",
            description=(
                f"**Motion Prompt:** {clean_prompt}\n"
                f"{badges_line}"
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
                        f"**Motion Prompt:** {clean_prompt}\n"
                        f"{badges_line}"
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
            outputs = await comfy.generate(
                workflow,
                timeout=14400,
                progress_callback=on_video_progress,
                channel_id=interaction.channel_id if interaction else None,
                message_id=status_msg[0].id if (status_msg and status_msg[0]) else None,
                user_id=interaction.user.id if (interaction and interaction.user) else None,
                command_type="video",
                metadata={"prompt": prompt, "width": width, "height": height, "model": wan_high_gguf}
            )
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy.get_execution_timing()
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
            t_breakdown = comfy.get_execution_timing()
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
            try:
                await comfy.free_memory()
            except Exception:
                pass

        if not outputs or not isinstance(outputs, list):
            await send_followup_fallback(interaction, content="ComfyUI did not return any video output.")
            return

        video_bytes = outputs[0]
        video_file_io = io.BytesIO(video_bytes)
        
        file = discord.File(fp=video_file_io, filename=format_image_filename("wan22_video", video_seed, "mp4"))

        view = VideoActionView(
            generation_id=gen_id,
            on_reroll_cb=handle_video_reroll,
            on_remix_cb=handle_video_remix,
            on_toggle_fps_cb=handle_video_toggle_fps,
            smoothness=smoothness
        )

        embed = build_video_complete_embed(
            prompt=clean_prompt,
            duration_sec=duration_sec,
            total_output_frames=total_output_frames,
            out_fps=out_fps,
            orig_w=orig_w,
            orig_h=orig_h,
            width=width,
            height=height,
            video_seed=video_seed,
            elapsed_time=elapsed_time,
            init_sec=init_sec,
            sample_sec=sample_sec,
            post_sec=post_sec,
            motion_badges=motion_badges,
            smoothness=smoothness,
            user_name=interaction.user.name if (interaction and interaction.user) else "User",
            user_id=interaction.user.id if (interaction and interaction.user) else 0
        )

        tag = f"{interaction.user.mention}\n" if (interaction and interaction.user) else ""
        delivered = False
        if status_msg[0]:
            try:
                await status_msg[0].edit(content=tag, embed=embed, attachments=[file], view=view)
                delivered = True
            except Exception as edit_err:
                logger.debug(f"Could not edit video status message in-place: {edit_err}")
                try:
                    await status_msg[0].delete()
                except Exception:
                    pass
        if not delivered:
            await send_followup_fallback(interaction, content=tag, embed=embed, file=file, view=view)
    except Exception as e:
        logger.error(f"Error executing video core generation: {e}")
        if status_msg[0]:
            try:
                await status_msg[0].delete()
            except Exception:
                pass
        await send_error_fallback(interaction, f"An error occurred during video generation: {e}")


async def handle_video_reroll(interaction: discord.Interaction, generation_id: str):
    """Re-runs a video generation on the same source image with a new random seed."""
    await safe_defer(interaction, thinking=True)
    gen_data = db.get_generation(generation_id) or {}
    src_cache_path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_source.png")
    if not os.path.exists(src_cache_path):
        await send_error_fallback(interaction, "Original source image has expired from cache. Please re-upload via `/video`.")
        return

    with open(src_cache_path, "rb") as f_src:
        image_bytes = f_src.read()

    new_seed = random.randint(1, 1125899906842624)
    await execute_video_core(
        interaction=interaction,
        image_bytes=image_bytes,
        filename=gen_data.get("filename", "reroll_video.png"),
        prompt=gen_data.get("raw_prompt", gen_data.get("prompt", "")),
        duration=gen_data.get("duration", 5),
        smoothness=gen_data.get("smoothness", "smooth"),
        seed=new_seed
    )


async def handle_video_remix(interaction: discord.Interaction, generation_id: str):
    """Opens VideoPromptModal pre-filled with settings for remixing motion/duration."""
    gen_data = db.get_generation(generation_id) or {}
    src_cache_path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_source.png")
    if not os.path.exists(src_cache_path):
        await send_error_fallback(interaction, "Original source image has expired from cache. Please re-upload via `/video`.")
        return

    async def on_remix_submit(modal_inter, new_prompt, dur_str, smooth_str, seed_str):
        await safe_defer(modal_inter, thinking=True)
        with open(src_cache_path, "rb") as f_src:
            image_bytes = f_src.read()

        dur_val = 5 if str(dur_str).strip() == "5" else 10
        smooth_val = "fast" if str(smooth_str).strip().lower() in ("fast", "16", "native") else "smooth"
        seed_val = int(seed_str) if (seed_str and str(seed_str).strip().isdigit()) else random.randint(1, 1125899906842624)

        await execute_video_core(
            interaction=modal_inter,
            image_bytes=image_bytes,
            filename=gen_data.get("filename", "remix_video.png"),
            prompt=new_prompt,
            duration=dur_val,
            smoothness=smooth_val,
            seed=seed_val
        )

    modal = VideoPromptModal(
        default_prompt=gen_data.get("raw_prompt", gen_data.get("prompt", "")),
        default_duration=str(gen_data.get("duration", "10")),
        default_smoothness=gen_data.get("smoothness", "smooth"),
        on_submit_callback=on_remix_submit
    )
    await interaction.response.send_modal(modal)


async def handle_video_toggle_fps(interaction: discord.Interaction, generation_id: str):
    """Toggles between Smooth (32 FPS via RIFE) and Ultra Fast (16 FPS native) on the same image."""
    await safe_defer(interaction, thinking=True)
    gen_data = db.get_generation(generation_id) or {}
    src_cache_path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}_source.png")
    if not os.path.exists(src_cache_path):
        await send_error_fallback(interaction, "Original source image has expired from cache. Please re-upload via `/video`.")
        return

    with open(src_cache_path, "rb") as f_src:
        image_bytes = f_src.read()

    current_smoothness = gen_data.get("smoothness", "smooth")
    new_smoothness = "fast" if current_smoothness == "smooth" else "smooth"

    await execute_video_core(
        interaction=interaction,
        image_bytes=image_bytes,
        filename=gen_data.get("filename", "toggle_fps_video.png"),
        prompt=gen_data.get("raw_prompt", gen_data.get("prompt", "")),
        duration=gen_data.get("duration", 10),
        smoothness=new_smoothness,
        seed=gen_data.get("seed", random.randint(1, 1125899906842624))
    )


async def execute_animate_message(interaction: discord.Interaction, message: discord.Message, client: Optional[ComfyClient] = None):
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
            duration = 5 if duration_str == "5" else 10
            
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
                seed=seed,
                client=client
            )
        except Exception as e:
            logger.error(f"Error executing animate context menu: {e}")
            await send_error_fallback(interaction, f"An error occurred during video generation: {e}")

    modal = VideoPromptModal(default_prompt=initial_prompt, on_submit_callback=on_modal_submit)
    await interaction.response.send_modal(modal)


async def execute_ltx_core(
    interaction: discord.Interaction,
    image_bytes: bytes,
    filename: str,
    prompt: str,
    duration: int = 4,
    motion_strength: int = 7,
    seed: int = None,
    client: Optional[ComfyClient] = None,
    **kwargs
):
    """Generate rapid video animation using LTX-Video."""
    comfy = client or _comfy_client

    active_arch = get_active_architecture()
    if active_arch is not None and active_arch != Architecture.LTX:
        logger.info(f"Switching architecture from {active_arch} to {Architecture.LTX}. Purging ComfyUI VRAM via /free...")
        await comfy.free_memory()
    set_active_architecture(Architecture.LTX)

    status_msg = [None]
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            orig_w, orig_h = img.size

        # 8GB VRAM friendly dimensions snapped to 32-pixel boundaries
        calc_w, calc_h = calculate_wan_dimensions(orig_w, orig_h, target_area=393216)
        width = max(256, (calc_w // 32) * 32)
        height = max(256, (calc_h // 32) * 32)

        upload_result = await comfy.upload_image(image_bytes, filename)
        uploaded_name = upload_result.get("name")
        if not uploaded_name:
            await send_followup_fallback(interaction, content="Failed to upload the image to ComfyUI server.")
            return

        workflow_path = "workflows/ltx_i2v.json"
        try:
            with open(workflow_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)
        except Exception as e:
            logger.error(f"Error loading LTX workflow template: {e}")
            await send_followup_fallback(interaction, content="Failed to load LTX-Video workflow template.")
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
            outputs = await comfy.generate(workflow, timeout=3600, progress_callback=on_ltx_progress)
            elapsed_time = time.perf_counter() - start_time
            t_breakdown = comfy.get_execution_timing()
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
            t_breakdown = comfy.get_execution_timing()
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

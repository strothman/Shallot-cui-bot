"""
Anima Service for Shallot-CUI Bot.
Handles Anima 2-stage anime generation, upscale refinement, and interactive callbacks.
"""

import os
import io
import time
import random
import logging
import asyncio
import discord
from PIL import Image

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient, StasisInterruptException
import db
from parsers import (
    resolve_anima_dimensions,
    prepare_anima_workflow,
    expand_dynamic_prompt,
)
from views import (
    CancelGenerationView,
    AnimaButtons,
    RemixModal,
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

logger = logging.getLogger("DiscordBot.AnimaService")

_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

ANIMA_DEFAULT_MODEL = "dasiwaAnima_luminousLabyrinthV1.safetensors"
ANIMA_DEFAULT_NEGATIVE = "blurry, low quality, distorted, bad anatomy, artifacts"


async def execute_anima(
    interaction: discord.Interaction,
    prompt: str,
    aspect_ratio: str = "4:3",
    seed: int = None,
    steps: int = 35,
    cfg: float = 5.0,
    denoise_upscale: float = 0.3,
    negative_prompt: str = None,
    model_name: str = None,
    client: ComfyClient = None,
    status_msg_ref: list = None,
):
    """Executes Anima 2-stage high-resolution anime generation with hires ESRGAN refinement."""
    active_client = client or _comfy_client
    target_arch = "ANIMA"
    curr_arch = get_active_architecture()
    if curr_arch is not None and curr_arch != target_arch:
        logger.info(f"Switching architecture from {curr_arch} to {target_arch}. Purging ComfyUI VRAM via /free...")
        await active_client.free_memory()
    set_active_architecture(target_arch)

    cleaned_prompt, base_w, base_h, hires_w, hires_h = resolve_anima_dimensions(prompt, aspect_ratio)

    actual_seed = seed if seed is not None else random.randint(1, 1125899906842624)
    cleaned_prompt = expand_dynamic_prompt(cleaned_prompt, random.Random(actual_seed))
    active_unet = model_name or ANIMA_DEFAULT_MODEL

    logger.info(f"[/anima] Prompt: '{cleaned_prompt}' | Base: {base_w}x{base_h} -> Hires: {hires_w}x{hires_h} | Steps: {steps} | CFG: {cfg} | Seed: {actual_seed}")

    try:
        workflow = prepare_anima_workflow(
            prompt=cleaned_prompt,
            aspect_ratio=aspect_ratio,
            seed=actual_seed,
            steps=steps,
            cfg=cfg,
            denoise_upscale=denoise_upscale,
            unet_model=active_unet,
            negative_prompt=negative_prompt or ANIMA_DEFAULT_NEGATIVE,
            filename_prefix="Discord Bot/Anima"
        )
    except Exception as e:
        logger.error(f"Error preparing Anima workflow: {e}")
        await send_error_fallback(interaction, f"Failed to prepare Anima workflow: {e}")
        return

    generation_id = f"anima_{int(time.time())}_{actual_seed}"
    status_msg = status_msg_ref if status_msg_ref else [None]
    last_update_time = [0.0]
    last_presence_time = [0.0]

    total_steps = steps + 15  # Stage 1 + Stage 2 refiner
    init_bar = create_progress_bar(0, total_steps)
    init_embed = discord.Embed(
        title="🌸 Generating with Anima (2-Stage)...",
        description=(
            f"**Prompt:** {cleaned_prompt}\n"
            f"**Progress:** {init_bar}\n"
            f"**Resolution:** `{base_w}x{base_h}` ➔ `{hires_w}x{hires_h}` ({aspect_ratio})\n"
            f"**Architecture:** Anima DiT + Qwen3 0.6B + 2x NomosUni\n"
            f"**Steps:** {steps} + 15 | **CFG:** {cfg} | **Seed:** `{actual_seed}`"
        ),
        color=discord.Color.from_rgb(255, 182, 193)
    )
    init_embed.set_footer(text="⏳ Initializing Anima diffusion & Qwen text encoder...")
    cancel_view = CancelGenerationView(generation_id)

    try:
        if status_msg[0] is None:
            status_msg[0] = await send_followup_fallback(interaction, embed=init_embed, view=cancel_view)
        else:
            await status_msg[0].edit(embed=init_embed, view=cancel_view)
    except Exception:
        pass

    async def on_anima_progress(val, max_val):
        percent = min(100, int((val / max_val) * 100)) if max_val > 0 else 0
        presence_str = f"🌸 Anima: {percent}% (Step {val}/{max_val})"
        now = time.time()
        if now - last_presence_time[0] >= 15.0 or val >= max_val:
            last_presence_time[0] = now
            asyncio.create_task(update_bot_presence(presence_str))

        if now - last_update_time[0] >= 1.5 or val >= max_val:
            last_update_time[0] = now
            bar = create_progress_bar(val, max_val)
            progress_embed = discord.Embed(
                title="🌸 Generating with Anima (2-Stage)...",
                description=(
                    f"**Prompt:** {cleaned_prompt}\n"
                    f"**Progress:** {bar} ({percent}%)\n"
                    f"**Resolution:** `{base_w}x{base_h}` ➔ `{hires_w}x{hires_h}` ({aspect_ratio})\n"
                    f"**Architecture:** Anima DiT + Qwen3 0.6B + 2x NomosUni\n"
                    f"**Steps:** {steps} + 15 | **CFG:** {cfg} | **Seed:** `{actual_seed}`"
                ),
                color=discord.Color.from_rgb(255, 182, 193)
            )
            stage_hint = "Stage 1: Base Generation" if val <= steps else "Stage 2: Hi-Res Refinement"
            progress_embed.set_footer(text=f"⏳ Anima • {stage_hint} (Step {val}/{max_val})")
            try:
                if status_msg[0] is not None:
                    await status_msg[0].edit(embed=progress_embed, view=cancel_view)
            except Exception:
                pass

    try:
        t_start = time.time()
        outputs = await active_client.generate(
            workflow,
            generation_id=generation_id,
            progress_callback=on_anima_progress,
            user_id=interaction.user.id if getattr(interaction, "user", None) else None,
            channel_id=getattr(interaction, "channel_id", None),
            message_id=getattr(status_msg[0], "id", None) if status_msg[0] else None,
            command_type="anima",
            description=f"Anima ({cleaned_prompt[:25]}…)"
        )
        elapsed_time = time.time() - t_start
        t_breakdown = active_client.get_execution_timing()

        image_bytes = None
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
            image_bytes = outputs[0]
        elif isinstance(outputs, dict):
            for node_id, node_output in outputs.items():
                if isinstance(node_output, dict) and "images" in node_output:
                    for img_info in node_output["images"]:
                        filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder", "")
                        folder_type = img_info.get("type", "output")
                        if filename:
                            image_bytes = await active_client.get_view_image(filename, subfolder, folder_type)
                            if image_bytes:
                                break
                if image_bytes:
                    break

        if not image_bytes:
            logger.error("[/anima] ComfyUI produced no image output.")
            await send_error_fallback(interaction, "ComfyUI completed but returned no output image.")
            return

        file = discord.File(io.BytesIO(image_bytes), filename=f"{generation_id}.png")

        # Save to database cache
        db.save_generation(
            generation_id,
            {
                "prompt": cleaned_prompt,
                "negative_prompt": negative_prompt or ANIMA_DEFAULT_NEGATIVE,
                "aspect_ratio": aspect_ratio,
                "seed": actual_seed,
                "steps": steps,
                "cfg": cfg,
                "denoise_upscale": denoise_upscale,
                "width": hires_w,
                "height": hires_h,
                "model_name": active_unet,
                "engine": "anima",
                "timestamp": time.time(),
                "user_id": interaction.user.id if getattr(interaction, "user", None) else 0
            }
        )

        complete_embed = discord.Embed(
            title="🌸 Anima Illustration",
            description=f"**Prompt:** {cleaned_prompt}",
            color=discord.Color.from_rgb(255, 182, 193)
        )
        complete_embed.add_field(
            name="📐 Specs",
            value=f"`{base_w}x{base_h}` ➔ `{hires_w}x{hires_h}`\nRatio: `{aspect_ratio}`",
            inline=True
        )
        complete_embed.add_field(
            name="🌸 Architecture",
            value=f"Anima 2B DiT\nEncoder: `Qwen3 0.6B`",
            inline=True
        )
        t_str = f"{elapsed_time:.1f}s"
        sample_sec = t_breakdown.get("sampling_duration", 0.0) or t_breakdown.get("sample", 0.0)
        init_sec = t_breakdown.get("init_duration", 0.0) or t_breakdown.get("init", 0.0)
        if sample_sec > 0:
            t_str += f" (Init {init_sec:.1f}s | Gen {sample_sec:.1f}s)"
        complete_embed.add_field(
            name="⏱️ Render",
            value=f"{t_str}\nSeed: `{actual_seed}`",
            inline=True
        )
        complete_embed.set_image(url=f"attachment://{generation_id}.png")
        complete_embed.set_footer(
            text=f"Requested by {interaction.user.display_name} • Anima 2-Stage Hi-Res",
            icon_url=interaction.user.display_avatar.url if interaction.user.display_avatar else None
        )

        view = AnimaButtons(generation_id=generation_id)

        try:
            if status_msg[0] is not None:
                await status_msg[0].edit(content=None, embed=complete_embed, attachments=[file], view=view)
            else:
                await send_followup_fallback(interaction, embed=complete_embed, file=file, view=view)
        except Exception:
            await send_followup_fallback(interaction, embed=complete_embed, file=file, view=view)

    except StasisInterruptException:
        logger.info("[/anima] Execution cancelled by user.")
    except Exception as e:
        logger.error(f"[/anima] Generation error: {e}", exc_info=True)
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "cuda" in err_msg or "vram" in err_msg:
            logger.warning("[/anima] CUDA Out-of-Memory detected! Purging ComfyUI VRAM via /free...")
            try:
                await active_client.free_memory(unload_models=True, free_memory=True)
            except Exception:
                pass
            await send_error_fallback(interaction, "CUDA Out-of-Memory during Anima execution. VRAM cache has been cleared.")
        else:
            await send_error_fallback(interaction, f"Anima generation error: {e}")


async def handle_anima_reroll(interaction: discord.Interaction, generation_id: str):
    """Handles 🔄 Re-roll button for /anima generations."""
    gen = db.get_generation(generation_id)
    if not gen:
        await send_error_fallback(interaction, "Could not find original generation data to re-roll.")
        return

    await safe_defer(interaction, thinking=True)
    await execute_anima(
        interaction=interaction,
        prompt=gen.get("prompt", ""),
        aspect_ratio=gen.get("aspect_ratio", "4:3"),
        seed=None,
        steps=gen.get("steps", 35),
        cfg=gen.get("cfg", 5.0),
        denoise_upscale=gen.get("denoise_upscale", 0.3),
        negative_prompt=gen.get("negative_prompt"),
        model_name=gen.get("model_name"),
    )


async def handle_anima_remix(interaction: discord.Interaction, generation_id: str):
    """Handles ✏️ Remix button for /anima generations."""
    gen = db.get_generation(generation_id)
    if not gen:
        await send_error_fallback(interaction, "Could not find original generation data to remix.")
        return

    modal = RemixModal(
        original_prompt=gen.get("prompt", ""),
        generation_id=generation_id
    )
    await interaction.response.send_modal(modal)

"""
Krea 2 Service for Shallot-CUI Bot.
Handles Bertflow generation execution, interactive button callbacks,
and Krea 2 Blend Studio vision analysis and updates.
"""

import os
import io
import time
import random
import logging
import asyncio
import discord
from discord import app_commands
from PIL import Image

from config import COMFYUI_ADDRESS
from image_utils import QUADRANT_CACHE_DIR, detect_closest_krea_aspect_ratio, create_thumbnail_bytes
from comfy_client import ComfyClient, StasisInterruptException
import db
from parsers import (
    resolve_bertflow_dimensions,
    get_bertflow_unet_model,
    prepare_bertflow_workflow,
    expand_dynamic_prompt,
    fuse_krea2_blend_prompt,
    parse_gamble_prompt,
)
from characters import get_character_display_badge
from celebrities import get_celebrity_display_badge
from views import (
    CancelGenerationView,
    BertflowButtons,
    GambleButtons,
    build_gamble_embed,
    RemixModal,
    build_blend_krea_embed,
    BlendKreaButtons,
)
from core_helpers import (
    safe_defer,
    send_followup_fallback,
    send_error_fallback,
    edit_original_fallback,
    create_progress_bar,
    update_bot_presence,
    get_active_architecture,
    set_active_architecture,
)
from services.vision_service import run_vision_interrogate

logger = logging.getLogger("DiscordBot.KreaService")

_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

BERTFLOW_MODEL_CHOICES = [
    app_commands.Choice(name="Muse v3.5 Extended (Stable Yogi - Recommended)", value="museByStableYogi_v35Int8Extended.safetensors"),
    app_commands.Choice(name="Pornmaster v2 (Krea 2 FP8)", value="pornmasterKrea2_v1FP8.safetensors"),
]


async def execute_bertflow(
    interaction: discord.Interaction,
    prompt: str,
    aspect_ratio: str = "1:1",
    seed: int = None,
    steps: int = 8,
    model_name: str = None,
    wetness_strength: float = -2.0,
    status_msg_ref: list = None,
    init_image_name: str = None,
    comp_strength: str = "off",
    character: str = None,
    celebrity: str = None,
    client: ComfyClient = None,
    gamble_info: dict = None,
):
    """Executes Bert's photorealistic Krea 2 workflow with optional direct compositional reference, character LoRA, and favorite celebrity prompt injection."""
    active_client = client or _comfy_client
    target_arch = "KREA2"
    curr_arch = get_active_architecture()
    if curr_arch is not None and curr_arch != target_arch:
        logger.info(f"Switching architecture from {curr_arch} to {target_arch}. Purging ComfyUI VRAM via /free...")
        await active_client.free_memory()
    set_active_architecture(target_arch)

    cleaned_prompt, width, height = resolve_bertflow_dimensions(prompt, aspect_ratio)

    cleaned_prompt, is_gamble, poetic_result = parse_gamble_prompt(cleaned_prompt, engine="krea2")
    if is_gamble and poetic_result and not gamble_info:
        gamble_info = {
            "seed_word": poetic_result.seed_word,
            "mood": poetic_result.mood,
            "stanza": poetic_result.stanza,
            "engine": "krea2",
            "metaphor": poetic_result.metaphor,
            "setting": poetic_result.setting,
            "color_palette": poetic_result.color_palette,
            "aspect_ratio": aspect_ratio or "1:1",
        }

    actual_seed = seed if seed is not None else random.randint(1, 1125899906842624)
    # Expand dynamic wildcards {a|b|c} using actual_seed
    cleaned_prompt = expand_dynamic_prompt(cleaned_prompt, random.Random(actual_seed))
    active_unet = get_bertflow_unet_model(model_name)

    comp_info = f" | Comp: {comp_strength} ({init_image_name})" if init_image_name and comp_strength != "off" else ""
    char_info = f" | Character: {character}" if character and str(character).lower() not in ["none", "nochar", "off"] else ""
    celeb_info = f" | Celebrity: {celebrity}" if celebrity and str(celebrity).lower() not in ["none", "noceleb", "off"] else ""
    logger.info(f"[/bertflow] Prompt: '{cleaned_prompt}' | Res: {width}x{height} | Steps: {steps} | Model: {active_unet} | Wetness: {wetness_strength}{comp_info}{char_info}{celeb_info} | Seed: {actual_seed}")

    try:
        workflow = prepare_bertflow_workflow(
            prompt=cleaned_prompt,
            width=width,
            height=height,
            seed=actual_seed,
            steps=steps,
            unet_model=active_unet,
            wetness_strength=wetness_strength,
            init_image=init_image_name,
            comp_strength=comp_strength,
            character=character,
            celebrity=celebrity
        )
    except Exception as e:
        logger.error(f"Error preparing Bertflow workflow: {e}")
        await send_error_fallback(interaction, f"Failed to prepare Bertflow workflow: {e}")
        return

    if gamble_info:
        generation_id = f"gamble_krea_{int(time.time())}_{actual_seed}"
    else:
        generation_id = f"bert_{int(time.time())}_{actual_seed}"
    status_msg = status_msg_ref if status_msg_ref else [None]
    last_update_time = [0.0]

    init_bar = create_progress_bar(0, steps)
    comp_line = f" | **Comp:** `{comp_strength.title()}`" if init_image_name and comp_strength != "off" else ""
    char_line = f" | **Character:** `{get_character_display_badge(character, architecture='krea2')}`" if character and str(character).lower() not in ["none", "nochar", "off"] else ""
    celeb_line = f" | **Celebrity:** `{get_celebrity_display_badge(celebrity)}`" if celebrity and str(celebrity).lower() not in ["none", "noceleb", "off"] else ""
    init_embed = discord.Embed(
        title="📸 Generating with Bertflow...",
        description=(
            f"**Prompt:** {cleaned_prompt}\n"
            f"**Progress:** {init_bar}\n"
            f"**Resolution:** {width}x{height} ({aspect_ratio or '1:1'})\n"
            f"**Engine:** Krea 2 Turbo ({active_unet.split('.')[0]})\n"
            f"**Steps:** {steps}{comp_line}{char_line}{celeb_line} | **Seed:** `{actual_seed}`"
        ),
        color=discord.Color.from_rgb(235, 140, 52)
    )
    init_embed.set_footer(text="⏳ Initializing Krea 2 & rgthree LoRA stack...")
    cancel_view = CancelGenerationView(generation_id)

    try:
        if status_msg[0] is None:
            status_msg[0] = await send_followup_fallback(interaction, embed=init_embed, view=cancel_view)
        else:
            await status_msg[0].edit(embed=init_embed, view=cancel_view)
    except Exception:
        pass

    async def on_bertflow_progress(val, max_val):
        percent = min(100, int((val / max_val) * 100)) if max_val > 0 else 0
        presence_str = f"📸 Bertflow: {percent}% (Step {val}/{max_val})"
        asyncio.create_task(update_bot_presence(presence_str))

        now = time.time()
        if now - last_update_time[0] >= 1.5 or val >= max_val:
            last_update_time[0] = now
            bar = create_progress_bar(val, max_val)
            progress_embed = discord.Embed(
                title="📸 Generating with Bertflow...",
                description=(
                    f"**Prompt:** {cleaned_prompt}\n"
                    f"**Progress:** {bar} ({percent}%)\n"
                    f"**Resolution:** {width}x{height} ({aspect_ratio or '1:1'})\n"
                    f"**Engine:** Krea 2 Turbo ({active_unet.split('.')[0]})\n"
                    f"**Steps:** {steps}{comp_line}{char_line}{celeb_line} | **Seed:** `{actual_seed}`"
                ),
                color=discord.Color.from_rgb(235, 140, 52)
            )
            progress_embed.set_footer(text=f"⏳ Krea 2 Turbo • Step {val}/{max_val}")
            try:
                if status_msg[0] is not None:
                    await status_msg[0].edit(embed=progress_embed, view=cancel_view)
            except Exception:
                pass

    try:
        t_start = time.time()
        krea_desc = f"Krea 2 Blend ({cleaned_prompt[:25]}…)" if init_image_name else f"Bertflow ({cleaned_prompt[:25]}…)"
        outputs = await active_client.generate(
            workflow,
            generation_id=generation_id,
            progress_callback=on_bertflow_progress,
            user_id=interaction.user.id if getattr(interaction, "user", None) else None,
            channel_id=getattr(interaction, "channel_id", None),
            message_id=getattr(status_msg[0], "id", None) if status_msg[0] else None,
            command_type="blend-krea" if init_image_name else "bertflow",
            description=krea_desc
        )
        elapsed_time = time.time() - t_start
        t_breakdown = active_client.get_execution_timing()

        image_bytes = None
        output_filename = None
        if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], (bytes, bytearray)):
            image_bytes = outputs[0]
        elif isinstance(outputs, dict):
            for node_id, node_output in outputs.items():
                if isinstance(node_output, dict) and "images" in node_output:
                    for img_info in node_output["images"]:
                        output_filename = img_info.get("filename")
                        subfolder = img_info.get("subfolder", "")
                        img_type = img_info.get("type", "output")
                        image_bytes = await active_client.get_image(output_filename, subfolder, img_type)
                        if image_bytes:
                            break
                if image_bytes:
                    break

        if not image_bytes:
            await send_error_fallback(interaction, "Generation succeeded on ComfyUI but failed to retrieve image bytes.")
            return

        bert_record = {
            "is_bertflow": True,
            "engine": "krea2",
            "prompt": cleaned_prompt,
            "original_prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "width": width,
            "height": height,
            "steps": steps,
            "seed": actual_seed,
            "unet_model": active_unet,
            "wetness": wetness_strength,
            "character": character,
            "celebrity": celebrity,
            "user_id": interaction.user.id if getattr(interaction, "user", None) else None,
            "gamble_info": gamble_info
        }
        db.save_generation(generation_id, bert_record)
        try:
            from services.generation_service import active_generations
            active_generations[generation_id] = bert_record
        except Exception:
            pass

        try:
            os.makedirs(QUADRANT_CACHE_DIR, exist_ok=True)
            for fname in [f"{generation_id}.png", f"{generation_id}_1.png"]:
                path = os.path.join(QUADRANT_CACHE_DIR, fname)
                with open(path, "wb") as f:
                    f.write(image_bytes)
        except Exception as err:
            logger.warning(f"Failed to cache Bertflow image: {err}")

        file = discord.File(io.BytesIO(image_bytes), filename=f"{generation_id}.png")

        if gamble_info:
            complete_embed = build_gamble_embed(
                gamble_info=gamble_info,
                seed=actual_seed,
                width=width,
                height=height,
                model_name=active_unet,
                user_name=interaction.user.display_name if getattr(interaction, "user", None) else "User",
                user_id=interaction.user.id if getattr(interaction, "user", None) else 0,
                timing_data={"elapsed_time": elapsed_time},
                engine="krea2"
            )
            complete_embed.set_image(url=f"attachment://{generation_id}.png")
            view = GambleButtons(
                generation_id=generation_id,
                engine="krea2",
                mood=gamble_info.get("mood", "wild")
            )
        else:
            complete_embed = discord.Embed(
                title="📸 Bertflow Realism",
                description=f"**Prompt:** {cleaned_prompt}",
                color=discord.Color.from_rgb(235, 140, 52)
            )
            complete_embed.add_field(name="📐 Specs", value=f"`{width}x{height}`\n`{aspect_ratio or '1:1'}`", inline=True)
            complete_embed.add_field(name="⚡ Engine", value=f"Krea 2 Turbo\n`{active_unet.split('.')[0]}`", inline=True)
            if character and str(character).lower() not in ["none", "nochar", "off"]:
                char_badge = get_character_display_badge(character, architecture="krea2")
                complete_embed.add_field(name="🎭 Character", value=char_badge, inline=True)
            if celebrity and str(celebrity).lower() not in ["none", "noceleb", "off"]:
                complete_embed.add_field(name="🌟 Celebrity", value=get_celebrity_display_badge(celebrity), inline=True)
            t_str = f"{elapsed_time:.1f}s"
            sample_sec = t_breakdown.get("sampling_duration", 0.0) or t_breakdown.get("sample", 0.0)
            init_sec = t_breakdown.get("init_duration", 0.0) or t_breakdown.get("init", 0.0)
            if sample_sec > 0:
                t_str += f" (Init {init_sec:.1f}s | Gen {sample_sec:.1f}s)"
            complete_embed.add_field(name="⏱️ Render", value=f"{t_str}\nSeed: `{actual_seed}`", inline=True)
            complete_embed.set_image(url=f"attachment://{generation_id}.png")
            complete_embed.set_footer(text=f"Requested by {interaction.user.display_name} • Krea 2 Flow-Matching", icon_url=interaction.user.display_avatar.url if interaction.user.display_avatar else None)

            view = BertflowButtons(
                generation_id=generation_id,
                character=character
            )

        try:
            if status_msg[0] is not None:
                await status_msg[0].edit(content=None, embed=complete_embed, attachments=[file], view=view)
            else:
                await send_followup_fallback(interaction, embed=complete_embed, file=file, view=view)
        except Exception:
            await send_followup_fallback(interaction, embed=complete_embed, file=file, view=view)

    except StasisInterruptException:
        logger.info(f"[/bertflow] Execution cancelled by user.")
    except Exception as e:
        logger.error(f"[/bertflow] Generation error: {e}", exc_info=True)
        err_msg = str(e).lower()
        if "out of memory" in err_msg or "cuda" in err_msg or "vram" in err_msg:
            logger.warning("[/bertflow] CUDA Out-of-Memory detected! Purging ComfyUI VRAM via /free...")
            try:
                await active_client.free_memory(unload_models=True, free_memory=True)
            except Exception:
                pass
            await send_error_fallback(
                interaction,
                "⚠️ **GPU Out-of-Memory (8GB VRAM limit reached)**\n"
                "The bot automatically purged cached models and released VRAM.\n"
                "• Try using `1:1` or `16:9` standard resolution.\n"
                "• Use `/free` if you experience any residual stutter."
            )
        else:
            await send_error_fallback(interaction, f"An error occurred during Bertflow generation: {e}")
    finally:
        await update_bot_presence(None)


async def handle_bertflow_reroll(interaction: discord.Interaction, generation_id: str, client: ComfyClient = None):
    """Re-rolls a Bertflow generation with a fresh random seed."""
    await safe_defer(interaction, thinking=True)
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    new_seed = random.randint(1, 1125899906842624)
    await execute_bertflow(
        interaction=interaction,
        prompt=gen_data.get("original_prompt") or gen_data.get("prompt"),
        aspect_ratio=gen_data.get("aspect_ratio", "1:1"),
        seed=new_seed,
        steps=gen_data.get("steps", 8),
        model_name=gen_data.get("unet_model"),
        character=gen_data.get("character"),
        celebrity=gen_data.get("celebrity"),
        client=client
    )


async def handle_bertflow_remix(interaction: discord.Interaction, generation_id: str, client: ComfyClient = None):
    """Opens a Remix modal for tweaking Bertflow prompt and seed."""
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.response.send_message("Generation session data expired.", ephemeral=True)
        return

    orig_p = gen_data.get("original_prompt") or gen_data.get("prompt") or ""
    orig_seed = gen_data.get("seed")

    async def remix_callback(inter: discord.Interaction, g_id: str, new_prompt: str, new_seed: int):
        await safe_defer(inter, thinking=True)
        await execute_bertflow(
            interaction=inter,
            prompt=new_prompt,
            aspect_ratio=gen_data.get("aspect_ratio", "1:1"),
            seed=new_seed if new_seed is not None else random.randint(1, 1125899906842624),
            steps=gen_data.get("steps", 8),
            model_name=gen_data.get("unet_model"),
            character=gen_data.get("character"),
            celebrity=gen_data.get("celebrity"),
            client=client
        )

    modal = RemixModal(generation_id, initial_prompt=orig_p, initial_seed=orig_seed, on_submit_callback=remix_callback)
    try:
        if not interaction.response.is_done():
            await interaction.response.send_modal(modal)
    except discord.HTTPException as e:
        if e.code != 40060:
            logger.warning(f"Failed to send Bertflow Remix modal: {e}")


async def handle_bertflow_toggle_char(interaction: discord.Interaction, generation_id: str, client: ComfyClient = None):
    """Toggles character LoRA on/off for the given Bertflow generation."""
    await safe_defer(interaction, thinking=True)
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find generation session data. It may have expired.", ephemeral=True)
        return

    curr_char = gen_data.get("character")
    has_char = curr_char and str(curr_char).lower() not in ["none", "nochar", "off", "false"]
    if has_char:
        new_char = None
        gen_data["last_character"] = curr_char
        db.save_generation(generation_id, gen_data)
    else:
        cand = gen_data.get("last_character") or "ogarla.85"
        new_char = "ogarla.85" if "valerie" in str(cand).lower() else cand

    await execute_bertflow(
        interaction=interaction,
        prompt=gen_data.get("original_prompt") or gen_data.get("prompt"),
        aspect_ratio=gen_data.get("aspect_ratio", "1:1"),
        seed=gen_data.get("seed"),
        steps=gen_data.get("steps", 8),
        model_name=gen_data.get("unet_model"),
        character=new_char,
        celebrity=gen_data.get("celebrity"),
        client=client
    )


async def handle_bertflow_upscale(interaction: discord.Interaction, generation_id: str):
    """Upscales a Bertflow image (1.5x) using high-fidelity Lanczos sampling."""
    await safe_defer(interaction, thinking=True)
    gen_data = db.get_generation(generation_id)
    cache_path = os.path.join(QUADRANT_CACHE_DIR, f"{generation_id}.png")

    image_bytes = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                image_bytes = f.read()
        except Exception:
            pass

    if not image_bytes:
        await interaction.followup.send("Could not retrieve cached image for upscaling. It may have expired.", ephemeral=True)
        return

    def _resize_1_5x(data: bytes):
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w_orig, h_orig = img.size
        w_new, h_new = int(w_orig * 1.5), int(h_orig * 1.5)
        upscaled = img.resize((w_new, h_new), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        upscaled.save(out, format="PNG")
        return out.getvalue(), w_orig, h_orig, w_new, h_new

    try:
        upscaled_bytes, w, h, new_w, new_h = await asyncio.to_thread(_resize_1_5x, image_bytes)

        upscale_embed = discord.Embed(
            title="🔍 Bertflow Upscale (1.5x)",
            description=f"**Prompt:** {gen_data.get('prompt', '') if gen_data else 'Bertflow Generation'}",
            color=discord.Color.from_rgb(235, 140, 52)
        )
        upscale_embed.add_field(name="📐 Resolution", value=f"`{w}x{h}` ➔ `{new_w}x{new_h}`", inline=True)
        if gen_data and gen_data.get("character") and str(gen_data.get("character")).lower() not in ["none", "nochar", "off"]:
            upscale_embed.add_field(name="🎭 Character", value="🌿 Ogarla (Krea 2)", inline=True)
        upscale_embed.set_footer(text=f"Requested by {interaction.user.display_name} • Krea 2 Photorealism")
        upscale_embed.set_image(url=f"attachment://upscale_{generation_id}.png")

        file = discord.File(io.BytesIO(upscaled_bytes), filename=f"upscale_{generation_id}.png")
        await send_followup_fallback(interaction, embed=upscale_embed, file=file)
    except Exception as e:
        logger.error(f"Error upscaling Bertflow generation: {e}", exc_info=True)
        await send_error_fallback(interaction, f"Failed to upscale Bertflow image: {e}")


async def handle_update_blend_krea_view(
    interaction: discord.Interaction,
    generation_id: str,
    new_ar: str = None,
    new_steps: int = None,
    new_model: str = None,
    new_wetness: float = None,
    new_comp: str = None,
    new_char: str = None,
    new_celeb: str = None
):
    """Updates interactive buttons and embed for a /blend-krea session."""
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.response.send_message("⚠️ Blend session data expired.", ephemeral=True)
        return

    if new_ar:
        gen_data["ar"] = new_ar
    if new_steps is not None:
        gen_data["steps"] = int(new_steps)
    if new_model:
        gen_data["model_choice"] = new_model
    if new_wetness is not None:
        gen_data["wetness"] = float(new_wetness)
    if new_comp:
        gen_data["composition"] = new_comp
    if new_char:
        gen_data["char_choice"] = new_char
    if new_celeb:
        gen_data["celeb_choice"] = new_celeb

    db.save_generation(generation_id, gen_data)
    try:
        from services.generation_service import active_generations
        active_generations[generation_id] = gen_data
    except Exception:
        pass

    embed = build_blend_krea_embed(gen_data, author_str=gen_data.get("author_str", "User"), image_url=gen_data.get("image_url"))
    view = BlendKreaButtons(
        generation_id=generation_id,
        ar=gen_data.get("ar", "16:9"),
        model_choice=gen_data.get("model_choice", "muse"),
        wetness=gen_data.get("wetness", -2.0),
        composition=gen_data.get("composition", "off"),
        character=gen_data.get("char_choice", "none"),
        celebrity=gen_data.get("celeb_choice", "none"),
        steps=int(gen_data.get("steps", 8))
    )
    try:
        await interaction.response.edit_message(embed=embed, view=view)
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"Ignored update error: {e}")


async def handle_submit_edit_blend_krea_prompt(interaction: discord.Interaction, generation_id: str, new_prompt: str):
    """Handles modal submission for updating the prompt in a /blend-krea session."""
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.response.send_message("⚠️ Blend session data expired.", ephemeral=True)
        return

    gen_data["fused_prompt"] = new_prompt
    gen_data["user_prompt"] = new_prompt
    db.save_generation(generation_id, gen_data)

    embed = build_blend_krea_embed(gen_data, author_str=gen_data.get("author_str", "User"), image_url=gen_data.get("image_url"))
    view = BlendKreaButtons(
        generation_id=generation_id,
        ar=gen_data.get("ar", "16:9"),
        model_choice=gen_data.get("model_choice", "muse"),
        wetness=gen_data.get("wetness", -2.0),
        composition=gen_data.get("composition", "off"),
        character=gen_data.get("char_choice", "none"),
        celebrity=gen_data.get("celeb_choice", "none"),
        steps=int(gen_data.get("steps", 8))
    )
    await interaction.response.edit_message(embed=embed, view=view)


async def handle_generate_blend_krea(interaction: discord.Interaction, generation_id: str, client: ComfyClient = None):
    """Executes Bertflow generation for a /blend-krea session."""
    await safe_defer(interaction)
    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find blend session data. It may have expired.", ephemeral=True)
        return

    fused_prompt = gen_data.get("fused_prompt") or gen_data.get("krea2_prompt")
    ar = gen_data.get("ar", "16:9")
    steps = int(gen_data.get("steps", 8))
    model_choice = gen_data.get("model_choice", "muse")
    wetness = float(gen_data.get("wetness", -2.0))
    comp = gen_data.get("composition", "off")
    uploaded_image_name = gen_data.get("uploaded_image_name")
    character = gen_data.get("char_choice")
    celebrity = gen_data.get("celeb_choice")

    unet_name = "museByStableYogi_v35Int8Extended.safetensors" if "muse" in model_choice.lower() else "pornmasterKrea2_v1FP8.safetensors"
    await execute_bertflow(
        interaction,
        prompt=fused_prompt,
        aspect_ratio=ar,
        steps=steps,
        model_name=unet_name,
        wetness_strength=wetness,
        init_image_name=uploaded_image_name if comp != "off" else None,
        comp_strength=comp,
        character=character,
        celebrity=celebrity,
        client=client
    )


async def execute_blend_krea_core(
    interaction: discord.Interaction,
    image_bytes: bytes,
    filename: str,
    image_url: str,
    prompt: str = None,
    steps: int = 8,
    aspect_ratio: str = None,
    model: str = "muse",
    wetness: float = -2.0,
    composition: str = "off",
    character: str = "none",
    celebrity: str = "none",
    vision_engine: str = "qwen2.5-vl",
    client: ComfyClient = None,
):
    """Core logic to analyze an image with Qwen2.5-VL / JoyCaption / Florence-2 and initialize the Krea 2 Blend Studio dashboard."""
    try:
        safe_filename = filename or f"blend_krea_{random.randint(100000, 999999)}.png"
        resolved_engine = vision_engine or "qwen2.5-vl"
        logger.info(f"Analyzing Krea 2 blend image {safe_filename} using {resolved_engine} vision engine...")
        vision_res = await run_vision_interrogate(
            image_bytes=image_bytes,
            filename=safe_filename,
            engine=resolved_engine,
            target_arch="krea2",
            client=client
        )
        if not vision_res:
            await edit_original_fallback(interaction, content=f"❌ Failed to analyze image with {resolved_engine} vision engine.")
            return

        uploaded_name = vision_res["uploaded_name"]
        raw_krea2_prompt = vision_res.get("krea2_prompt") or ""
        detailed_caption = vision_res.get("detailed_caption") or raw_krea2_prompt

        resolved_ar = aspect_ratio
        if not resolved_ar or str(resolved_ar).lower() == "auto":
            try:
                with Image.open(io.BytesIO(image_bytes)) as pil_img:
                    resolved_ar = detect_closest_krea_aspect_ratio(pil_img.width, pil_img.height)
                    logger.info(f"Auto-detected Krea 2 aspect ratio {resolved_ar} from source image size ({pil_img.width}x{pil_img.height})")
            except Exception as e:
                logger.debug(f"Could not auto-detect AR: {e}")
                resolved_ar = "16:9"

        fused_prompt = fuse_krea2_blend_prompt(raw_krea2_prompt, prompt)

        # Prepare fast, lightweight thumbnail attachment for Discord embed
        thumb_bytes = await asyncio.to_thread(create_thumbnail_bytes, image_bytes)
        thumb_filename = "source_thumb.jpg"
        thumb_file = discord.File(io.BytesIO(thumb_bytes), filename=thumb_filename)
        effective_image_url = f"attachment://{thumb_filename}"

        engine_used_name = vision_res.get("engine_used", resolved_engine)
        generation_id = str(random.randint(100000, 999999))
        gen_data = {
            "caption": detailed_caption,
            "detailed_caption": detailed_caption,
            "krea2_prompt": raw_krea2_prompt,
            "user_prompt": prompt or "",
            "fused_prompt": fused_prompt,
            "uploaded_image_name": uploaded_name,
            "image_url": effective_image_url,
            "source_image_url": image_url,
            "ar": resolved_ar,
            "steps": int(steps or 8),
            "model_choice": model or "muse",
            "wetness": float(wetness if wetness is not None else -2.0),
            "composition": composition or "off",
            "char_choice": character or "none",
            "celeb_choice": celebrity or "none",
            "vision_engine": engine_used_name,
            "author_str": interaction.user.name
        }
        db.save_generation(generation_id, gen_data)

        embed = build_blend_krea_embed(gen_data, author_str=interaction.user.name, image_url=effective_image_url)
        view = BlendKreaButtons(
            generation_id=generation_id,
            ar=resolved_ar,
            model_choice=gen_data["model_choice"],
            wetness=gen_data["wetness"],
            composition=gen_data["composition"],
            character=gen_data["char_choice"],
            celebrity=gen_data["celeb_choice"],
            steps=gen_data["steps"]
        )
        await edit_original_fallback(interaction, content=None, embed=embed, view=view, attachments=[thumb_file])

    except Exception as e:
        logger.error(f"Error executing blend-krea workflow: {e}")
        await edit_original_fallback(interaction, content=f"❌ An error occurred during Krea 2 blend: {e}")

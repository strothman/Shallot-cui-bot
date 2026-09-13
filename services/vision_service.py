"""
Vision Service for Shallot-CUI Bot.
Handles image interrogation via Florence-2 / JoyCaption / Qwen2.5-VL,
multi-architecture prompt synthesis (SDXL, Krea 2, Flux), and SDXL blend studio initialization.
"""

import os
import io
import json
import random
import logging
import asyncio
import aiohttp
import discord
from PIL import Image

from config import COMFYUI_ADDRESS
from comfy_client import ComfyClient
import db
from parsers import (
    format_sdxl_prompt,
    format_krea2_prompt,
    format_flux_prompt,
    build_scapes_prompt,
    clean_midjourney_flags,
)
from image_utils import detect_closest_aspect_ratio, create_thumbnail_bytes
from views import build_blend_embed, BlendButtons, DescribeButtons
from core_helpers import (
    safe_defer,
    edit_original_fallback,
    send_error_fallback,
)
import re

logger = logging.getLogger("DiscordBot.VisionService")

# Local client instance for vision operations
_comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

user_vision_preferences: dict = {}


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



async def run_vision_interrogate(
    image_bytes: bytes,
    filename: str = None,
    engine: str = "auto",
    target_arch: str = "all",
    client: ComfyClient = None
) -> dict:
    """
    Unified vision interrogation engine. Supports JoyCaption, Qwen2.5-VL, and Florence-2
    with automatic graceful fallback, multi-target prompt formatting (SDXL, Krea 2, Flux),
    and proactive VRAM cleanup for 8GB GPUs.
    """
    if not image_bytes:
        return None

    comfy = client or _comfy_client
    safe_filename = filename or f"vision_interrogate_{random.randint(100000, 999999)}.png"
    upload_result = await comfy.upload_image(image_bytes, safe_filename)
    uploaded_name = upload_result.get("name")
    if not uploaded_name:
        logger.error("Failed to upload image to ComfyUI for vision interrogation.")
        return None

    # Resolve target engine
    norm_engine = (engine or "auto").lower()
    if norm_engine == "auto":
        if target_arch == "krea2":
            norm_engine = "qwen2.5-vl"
        elif target_arch in ["sdxl", "flux"]:
            norm_engine = "joycaption"
        else:
            norm_engine = "joycaption"

    engine_order = []
    if "joy" in norm_engine:
        fallback_florence = "workflows/DESCRIBE_blend.json" if target_arch == "sdxl" else "workflows/DESCRIBE_cuibot.json"
        engine_order = [("joycaption", "workflows/DESCRIBE_joycaption.json"), ("florence2", fallback_florence)]
    elif "qwen" in norm_engine:
        engine_order = [("qwen2.5-vl", "workflows/DESCRIBE_qwen_vl.json"), ("florence2", "workflows/DESCRIBE_cuibot.json")]
    else:
        wf = "workflows/DESCRIBE_blend.json" if target_arch == "sdxl" else "workflows/DESCRIBE_cuibot.json"
        engine_order = [("florence2", wf)]

    results = None
    engine_used = None

    for eng_name, wf_path in engine_order:
        try:
            if not os.path.exists(wf_path):
                logger.warning(f"Workflow file {wf_path} not found. Skipping {eng_name}.")
                continue

            with open(wf_path, "r", encoding="utf-8") as f:
                workflow = json.load(f)

            if "1" in workflow and "inputs" in workflow["1"]:
                workflow["1"]["inputs"]["image"] = uploaded_name

            logger.info(f"Executing {eng_name} vision workflow ({wf_path}) for {uploaded_name}...")
            results = await comfy.generate(workflow, timeout=14400)
            if results and isinstance(results, dict):
                engine_used = eng_name
                break
        except Exception as e:
            logger.warning(f"Engine {eng_name} failed ({e}). Attempting next fallback in pipeline...")
            results = None

    if not results:
        # Ultimate fallback directly via Florence-2 in-code 3-node workflow if files were missing or failed
        try:
            logger.info("Executing built-in Florence-2 fallback...")
            fallback_wf = {
                "1": {"inputs": {"image": uploaded_name}, "class_type": "LoadImage"},
                "2": {"inputs": {"model": "MiaoshouAI/Florence-2-large-PromptGen-v2.0", "precision": "fp16", "convert_to_safetensors": True}, "class_type": "DownloadAndLoadFlorence2Model"},
                "3": {"inputs": {"text_input": "", "task": "detailed_caption", "fill_mask": True, "keep_model_loaded": False, "max_new_tokens": 250, "num_beams": 3, "do_sample": False, "output_mask_select": "", "seed": random.randint(100000, 999999), "image": ["1", 0], "florence2_model": ["2", 0]}, "class_type": "Florence2Run"},
                "4": {"inputs": {"text": ["3", 2]}, "class_type": "ShowText|pysssss"}
            }
            results = await comfy.generate(fallback_wf, timeout=14400)
            engine_used = "florence2"
        except Exception as e:
            logger.error(f"All vision interrogation workflows and fallbacks failed: {e}")
            return None

    # Free vision model weights from GPU memory immediately so diffusion has full VRAM
    try:
        await comfy.free_memory(unload_models=True)
        logger.info(f"{engine_used} vision model VRAM successfully purged.")
    except Exception as e:
        logger.debug(f"Could not purge VRAM after vision model: {e}")

    # Extract text outputs across all possible node structures
    texts = []
    if isinstance(results, dict):
        for nid in ["4", "3", "9", "10", "11", "19", "20", "21"]:
            if nid in results:
                ndata = results[nid]
                if isinstance(ndata, dict):
                    for k in ["text", "string", "caption", "output"]:
                        if k in ndata and ndata[k]:
                            val = ndata[k]
                            val_str = val[0] if isinstance(val, list) else str(val)
                            if val_str and val_str.strip():
                                texts.append(val_str.strip())
                elif isinstance(ndata, list) and ndata:
                    val_str = str(ndata[0]).strip()
                    if val_str:
                        texts.append(val_str)

    raw_text = texts[0] if texts else "A detailed scene"
    detailed_text = texts[1] if len(texts) > 1 else raw_text

    # Synthesize tailored prompts for each architecture
    sdxl_prompt = format_sdxl_prompt(raw_text)
    sdxl_detailed_prompt = format_sdxl_prompt(detailed_text)
    krea2_prompt = format_krea2_prompt(detailed_text if "qwen" in str(engine_used) else raw_text) if target_arch in ["all", "krea2"] else ""
    flux_prompt = format_flux_prompt(detailed_text)

    display_engine = {
        "joycaption": "JoyCaption (SDXL & Flux)",
        "qwen2.5-vl": "Qwen2.5-VL (Krea 2)",
        "florence2": "Florence-2 (SDXL)" if target_arch == "sdxl" else "Florence-2 (Legacy)"
    }.get(engine_used, str(engine_used).title())

    res = {
        "caption": sdxl_prompt,
        "detailed_caption": flux_prompt,
        "sdxl_prompt": sdxl_prompt,
        "sdxl_detailed_prompt": sdxl_detailed_prompt,
        "flux_prompt": flux_prompt,
        "raw_text": raw_text,
        "engine_used": display_engine,
        "uploaded_name": uploaded_name
    }
    if target_arch in ["all", "krea2"]:
        res["krea2_prompt"] = krea2_prompt
    return res


async def run_florence_interrogate(image_url: str, client: ComfyClient = None) -> str:
    """Downloads an image from URL and uses the vision pipeline to generate an SDXL prompt description."""
    if not image_url:
        return None
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as resp:
                if resp.status != 200:
                    return None
                image_bytes = await resp.read()

        res = await run_vision_interrogate(image_bytes=image_bytes, engine="joycaption", target_arch="sdxl", client=client)
        return res.get("sdxl_prompt") if res else None
    except Exception as e:
        logger.error(f"Error running vision interrogate for adopted post: {e}")
        return None


async def execute_blend_core(
    interaction: discord.Interaction, 
    image_bytes: bytes, 
    filename: str, 
    image_url: str, 
    prompt: str = None,
    style: str = None,
    secondary_style: str = None,
    client: ComfyClient = None
):
    """Core execution logic for Florence-2 image blending and remixing in SDXL."""
    if style:
        scapes_info = build_scapes_prompt(
            user_prompt=prompt or "",
            style=style,
            secondary_style=secondary_style,
            mode=None,
            subject_type="scenery"
        )
        prompt = scapes_info["final_prompt"]

    try:
        safe_filename = filename or f"blend_sdxl_{random.randint(100000, 999999)}.png"
        logger.info(f"Analyzing SDXL blend image {safe_filename} using dedicated Florence-2 vision engine...")
        vision_res = await run_vision_interrogate(
            image_bytes=image_bytes,
            filename=safe_filename,
            engine="florence2",
            target_arch="sdxl",
            client=client
        )
        if not vision_res:
            await edit_original_fallback(interaction, content="❌ Failed to analyze image with Florence-2 vision engine.")
            return

        uploaded_name = vision_res["uploaded_name"]
        raw_caption = vision_res.get("sdxl_prompt") or ""
        raw_detailed_caption = vision_res.get("sdxl_detailed_prompt") or vision_res.get("caption") or raw_caption
        caption = raw_caption
        detailed_caption = raw_detailed_caption

        # Auto-detect native aspect ratio from uploaded image dimensions
        detected_ar = "16:9"
        try:
            with Image.open(io.BytesIO(image_bytes)) as pil_img:
                detected_ar = detect_closest_aspect_ratio(pil_img.width, pil_img.height)
                logger.info(f"Auto-detected aspect ratio {detected_ar} from image size ({pil_img.width}x{pil_img.height})")
        except Exception as e:
            logger.debug(f"Could not inspect image dimensions for auto AR: {e}")

        # Truncate descriptions to fit within Discord's 1024-character limit for embed fields
        if len(caption) > 1024:
            caption = caption[:1021] + "..."
        if len(detailed_caption) > 1024:
            detailed_caption = detailed_caption[:1021] + "..."

        # Prepare fast, lightweight thumbnail attachment for Discord embed
        thumb_bytes = await asyncio.to_thread(create_thumbnail_bytes, image_bytes)
        thumb_filename = "source_thumb.jpg"
        thumb_file = discord.File(io.BytesIO(thumb_bytes), filename=thumb_filename)
        effective_image_url = f"attachment://{thumb_filename}"

        # Store in SQLite generations cache for interactive button clicks
        generation_id = str(random.randint(100000, 999999))
        gen_data = {
            "caption": raw_caption,
            "detailed_caption": raw_detailed_caption,
            "extra_details": prompt or "",
            "uploaded_image_name": uploaded_name,
            "image_url": effective_image_url,
            "source_image_url": image_url,
            "user_prompt": prompt or "",
            "ar": detected_ar,
            "sr": True,
            "oga": False,
            "char_choice": "none",
            "model_choice": "wai",
            "comp_strength": "style",
            "sref_rand": "nosref",
            "author_str": interaction.user.name
        }
        db.save_generation(generation_id, gen_data)

        # Build streamlined embed response with 3-column dashboard
        embed = build_blend_embed(gen_data, author_str=interaction.user.name, image_url=effective_image_url)
        user_favs = db.get_favorite_styles(interaction.user.id) if (interaction and interaction.user) else []
        view = BlendButtons(
            generation_id=generation_id,
            ar=detected_ar,
            sr=True,
            oga=False,
            model_choice="wai",
            comp_strength="style",
            sref_rand="nosref",
            char_choice="none",
            user_favorites=user_favs
        )
        await edit_original_fallback(interaction, content=None, embed=embed, view=view, attachments=[thumb_file])

    except Exception as e:
        logger.error(f"Error executing blend workflow: {e}")
        await edit_original_fallback(interaction, content=f"❌ An error occurred: {e}")


async def execute_blend_message(interaction: discord.Interaction, message: discord.Message, client: ComfyClient = None):
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
            prompt=initial_prompt,
            client=client
        )
    except Exception as e:
        logger.error(f"Error in blend context menu: {e}")
        await edit_original_fallback(interaction, content=f"❌ Failed to process blend for message: {e}")


async def execute_describe_core(interaction: discord.Interaction, image: discord.Attachment, model: str = None, client: ComfyClient = None):
    """Executes multi-architecture image interrogation using JoyCaption, Qwen2.5-VL, or Florence-2."""
    selected_engine = model or user_vision_preferences.get(interaction.user.id, "joycaption")
    user_vision_preferences[interaction.user.id] = selected_engine

    friendly_name = {
        "joycaption": "JoyCaption",
        "qwen2.5-vl": "Qwen2.5-VL",
        "florence2": "Florence-2"
    }.get(selected_engine, "AI Vision")

    await interaction.response.send_message(f"Analyzing image with {friendly_name}...", ephemeral=False)
    
    if not image.content_type or not image.content_type.startswith("image/"):
        await edit_original_fallback(interaction, content="❌ Please upload a valid image file (PNG/JPG).")
        return
        
    try:
        image_bytes = await image.read()
        
        logger.info(f"Running vision interrogate with {selected_engine} for {image.filename}...")
        vision_res = await run_vision_interrogate(
            image_bytes=image_bytes,
            filename=image.filename,
            engine=selected_engine,
            target_arch="all",
            client=client
        )
        
        if not vision_res:
            await edit_original_fallback(interaction, content="❌ Failed to analyze image with vision model.")
            return

        sdxl_prompt = vision_res.get("sdxl_prompt") or ""
        krea2_prompt = vision_res.get("krea2_prompt") or ""
        flux_prompt = vision_res.get("flux_prompt") or ""
        engine_display = vision_res.get("engine_used", friendly_name)

        disp_krea2 = (krea2_prompt[:1021] + "...") if len(krea2_prompt) > 1024 else krea2_prompt
        disp_flux = (flux_prompt[:1021] + "...") if len(flux_prompt) > 1024 else flux_prompt
        disp_sdxl = (sdxl_prompt[:1021] + "...") if len(sdxl_prompt) > 1024 else sdxl_prompt

        generation_id = str(random.randint(100000, 999999))
        gen_data = {
            "caption": sdxl_prompt,
            "detailed_caption": flux_prompt,
            "krea2_prompt": krea2_prompt,
            "sdxl_prompt": sdxl_prompt,
            "flux_prompt": flux_prompt,
            "engine": engine_display
        }
        db.save_generation(generation_id, gen_data)

        embed = discord.Embed(
            title="Image Description & Multi-Architecture Analysis",
            color=discord.Color.blue()
        )
        embed.set_thumbnail(url=image.url)
        embed.add_field(name="📸 Krea 2 Photorealism", value=disp_krea2 or "No prompt generated", inline=False)
        embed.add_field(name="⚡ Flux Detailed Prose", value=disp_flux or "No prompt generated", inline=False)
        embed.add_field(name="🎨 SDXL Tags", value=disp_sdxl or "No prompt generated", inline=False)
        embed.set_footer(text=f"Analyzed using {engine_display} • Requested by {interaction.user.name}")
        
        view = DescribeButtons(generation_id, ar="16:9")
        await edit_original_fallback(interaction, content=None, embed=embed, view=view)

    except Exception as e:
        logger.error(f"Error executing describe command: {e}")
        await edit_original_fallback(interaction, content=f"❌ An error occurred while describing the image: {e}")


async def handle_generate_described(interaction: discord.Interaction, generation_id: str, desc_type: str, ar: str = "16:9", use_sr = True, use_oga: bool = False, model_choice: str = "hyphoria"):
    """Generates an image grid using stored caption or detailed description from /describe with chosen AR, LoRA, and model settings."""
    await safe_defer(interaction)

    gen_data = db.get_generation(generation_id)
    if not gen_data:
        await interaction.followup.send("Could not find description session data. It may have expired.", ephemeral=True)
        return

    if desc_type == "krea2":
        krea2_prompt = gen_data.get("krea2_prompt") or gen_data.get("detailed_caption") or gen_data.get("caption")
        if not krea2_prompt:
            await interaction.followup.send("No Krea 2 prompt found in session.", ephemeral=True)
            return
        char_val = "ogarla.85" if use_oga else None
        import bot as bot_module
        bertflow_func = getattr(bot_module, "execute_bertflow", None)
        if not bertflow_func:
            from services.krea_service import execute_bertflow as bertflow_func
        await bertflow_func(interaction, prompt=krea2_prompt, aspect_ratio=ar, character=char_val)
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

    import bot as bot_module
    imagine_func = getattr(bot_module, "execute_imagine", None)
    if not imagine_func and hasattr(interaction, "client"):
        imagine_func = getattr(interaction.client, "execute_imagine", None)
    if imagine_func:
        await imagine_func(interaction, prompt=full_prompt, checkpoint=selected_model)
    else:
        await interaction.followup.send("❌ Image generation handler unavailable.", ephemeral=True)


async def handle_update_describe_view(interaction: discord.Interaction, generation_id: str, new_ar: str, new_sr = True, new_oga: bool = False, new_model: str = "hyphoria"):
    """Updates the interactive buttons on the /describe result embed when AR, SR, Ogarla, or Model toggle is clicked."""
    view = DescribeButtons(generation_id, ar=new_ar, sr=new_sr, oga=new_oga, model_choice=new_model)
    try:
        await interaction.response.edit_message(view=view)
    except (discord.NotFound, discord.HTTPException) as e:
        logger.debug(f"Ignored expected interaction update error: {e}")


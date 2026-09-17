"""
Shallot-CUI Bot — Main Application Entry Point & Lifecycle Coordinator
"""

import os
import sys
import socket
import logging
import asyncio
import time
from typing import Optional, Dict, Any, List

import discord
from discord import app_commands
from discord.ext import commands
from dotenv import load_dotenv

from comfy_client import ComfyClient, StasisInterruptException
from config import (
    BOT_OWNER_ID,
    COMFYUI_ADDRESS,
    COMFYUI_CHECKPOINT,
    COMFYUI_BATCH_PATH,
    DEFAULT_NEGATIVE_PROMPT,
    IMAGE_SAVE_PREFIX,
    VRAM_CAUTION_THRESHOLD_PERCENT,
    VRAM_MIN_FREE_GB,
    PipelineDefaults,
    get_checkpoint_display_name,
    is_authorized_admin,
    SDXL_CHECKPOINT_CHOICES,
    SDXL_ENHANCEMENT_CHOICES,
    CHECKPOINT_CONFIGS,
)

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
import db
import model_architecture
from characters import get_character, mask_character_in_prompt, inject_trained_trigger_in_prompt, CHARACTERS
from celebrities import (
    CELEBRITY_CHOICES_KREA2,
    get_celebrity_autocomplete_choices,
    get_celebrity_display_badge,
    get_celebrity,
)
from core_helpers import (
    set_active_bot,
    get_active_bot,
    safe_defer,
    download_image,
    send_followup_fallback,
    send_error_fallback,
    edit_original_fallback,
    edit_message_fallback,
    is_interaction_expired,
    _update_button_state,
    check_gpu_vram_caution,
    terminate_existing_comfyui,
    update_bot_presence,
)
from services.system_service import (
    SETTINGS_FILE,
    settings,
    load_settings,
    save_settings,
    get_setting,
    set_setting,
    fetch_comfyui_queue,
    fetch_comfyui_system_stats,
    build_queue_embed,
    build_models_embed,
    purge_vram_core,
    create_progress_bar,
    update_console_title,
    get_effective_queue_counts,
    format_presence_status_text,
    touch_activity as _sys_touch_activity,
    sync_presence_now,
    on_engine_queue_change,
    get_last_generation_activity,
    is_idle_purged,
    set_idle_purged,
    IDLE_PURGE_TIMEOUT,
)

# Load environment variables
load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Configure Logging
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logging.getLogger("discord.client").setLevel(logging.ERROR)
logging.getLogger("discord.gateway").setLevel(logging.ERROR)
logging.getLogger("discord.ext.commands").setLevel(logging.ERROR)
logger = logging.getLogger("DiscordBot")

# Initialize ComfyUI client
comfy_client = ComfyClient(server_address=COMFYUI_ADDRESS)

# Setup Bot
intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents, max_messages=100)
bot.comfy_client = comfy_client
bot.db = db
set_active_bot(bot)

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
bot.active_generations = active_generations
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

from cogs.system_cog import SystemCog
from cogs.upscale_cog import UpscaleCog
from cogs.imagine_cog import ImagineCog
from services.interaction_dispatcher import dispatch_interaction

_vision_cog = VisionCog(bot)
_krea_cog = KreaCog(bot)
_system_cog = SystemCog(bot)
_upscale_cog = UpscaleCog(bot)
_imagine_cog = ImagineCog(bot)

ALL_COGS = [_vision_cog, _krea_cog, _system_cog, _upscale_cog, _imagine_cog]

async def setup_hook():
    """Asynchronously registers all modular cogs and persistent dynamic UI items on startup."""
    for cog in ALL_COGS:
        if cog.qualified_name not in bot.cogs:
            await bot.add_cog(cog)
            
    # Register persistent dynamic items for UI buttons
    from views import (
        CancelGenDynamicButton,
        IsolateDynamicButton,
        VariationDynamicButton,
        RerollDynamicButton,
        RemixDynamicButton,
        OutpaintDynamicButton,
    )
    bot.add_dynamic_items(
        CancelGenDynamicButton,
        IsolateDynamicButton,
        VariationDynamicButton,
        RerollDynamicButton,
        RemixDynamicButton,
        OutpaintDynamicButton,
    )

bot.setup_hook = setup_hook

# Synchronously ensure cogs are registered so test runners inspecting bot.tree find them immediately
try:
    loop = asyncio.get_running_loop()
    for cog in ALL_COGS:
        loop.create_task(bot.add_cog(cog))
except RuntimeError:
    for cog in ALL_COGS:
        asyncio.run(bot.add_cog(cog))

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
purge_vram_command = _system_cog.free_vram
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

update_presence = _system_cog.update_presence_task
periodic_scratch_maintenance = _system_cog.periodic_scratch_maintenance_task
idle_memory_watchdog = _system_cog.idle_memory_watchdog_task

_last_generation_activity = get_last_generation_activity()
_idle_purged = is_idle_purged()
_last_presence_sync_time = 0.0

def touch_activity():
    global _last_generation_activity, _idle_purged
    _sys_touch_activity()
    _last_generation_activity = get_last_generation_activity()
    _idle_purged = is_idle_purged()

bot.touch_activity = touch_activity

def load_generations():
    db.init_db()

def cleanup_orphaned_quadrants():
    db.cleanup_orphaned_quadrants()


@bot.event
async def on_ready():
    if not hasattr(bot, '_initial_ready_done'):
        bot._initial_ready_done = True
        load_generations()
        load_settings()
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

        asyncio.create_task(asyncio.to_thread(db.cleanup_orphaned_quadrants, 48.0))
        
        try:
            if await comfy_client.is_online():
                logger.info(f"🟢 ComfyUI server is ONLINE at {COMFYUI_ADDRESS}.")
                try:
                    dep_result = await comfy_client.check_dependencies()
                    if dep_result["ok"]:
                        logger.info("✅ All model dependencies are satisfied — no missing files detected.")
                    else:
                        missing = dep_result["missing"]
                        total_missing = sum(len(v) for v in missing.values())
                        logger.warning(f"⚠️  MISSING DEPENDENCIES: {total_missing} model file(s) not found in ComfyUI.")
                except Exception as dep_err:
                    logger.debug(f"Dependency check skipped: {dep_err}")
            else:
                logger.info(f"ℹ️ ComfyUI server is currently OFFLINE at {COMFYUI_ADDRESS}. Use /cui-start in Discord to launch it.")
        except Exception:
            logger.info(f"ℹ️ ComfyUI server is currently OFFLINE at {COMFYUI_ADDRESS}. Use /cui-start in Discord to launch it.")

        try:
            synced = await bot.tree.sync()
            logger.info(f"Synced {len(synced)} command(s)")
        except Exception as e:
            logger.error(f"Error syncing tree: {e}")

        try:
            if bot.user and bot.user.name != "Shallot-CUI Bot":
                await bot.user.edit(username="Shallot-CUI Bot")
                logger.info("Updated bot username to 'Shallot-CUI Bot'")
        except Exception as name_err:
            logger.debug(f"Username auto-update skipped: {name_err}")
    else:
        logger.info("Discord reconnected (on_ready fired again). Re-establishing ComfyUI WebSocket.")
        try:
            await comfy_client.start()
        except Exception as e:
            logger.warning(f"Failed to restart ComfyUI client on reconnect: {e}")

    touch_activity()
    logger.info(f"Bot connected as {bot.user}")

    # Ensure background tasks are running in SystemCog
    _system_cog.start_tasks()


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


def is_dynamic_component(custom_id: str) -> bool:
    """
    Checks if a custom_id is matched by any registered discord.ui.DynamicItem.
    Dynamic items are scheduled and handled natively by discord.py's ViewStore.
    """
    if not custom_id:
        return False
    try:
        view_store = getattr(bot._connection, "_view_store", None)
        if view_store and hasattr(view_store, "_dynamic_items"):
            for pattern in view_store._dynamic_items.keys():
                if pattern.fullmatch(custom_id):
                    return True
    except Exception:
        pass
    from views.dynamic_items import (
        CancelGenDynamicButton,
        IsolateDynamicButton,
        VariationDynamicButton,
        RerollDynamicButton,
        RemixDynamicButton,
        OutpaintDynamicButton,
    )
    for cls in (
        CancelGenDynamicButton,
        IsolateDynamicButton,
        VariationDynamicButton,
        RerollDynamicButton,
        RemixDynamicButton,
        OutpaintDynamicButton,
    ):
        template = getattr(cls, "__discord_ui_compiled_template__", None)
        if template and template.fullmatch(custom_id):
            return True
    return False


@bot.event
async def on_interaction(interaction: discord.Interaction):
    touch_activity()
    if interaction.type == discord.InteractionType.component:
        custom_id = interaction.data.get("custom_id", "") if interaction.data else ""
        if custom_id and is_dynamic_component(custom_id):
            return
        await dispatch_interaction(interaction)


async def on_close():
    await comfy_client.stop()


_instance_lock_socket = None

def acquire_instance_lock(port: int = 48123, silent: bool = False) -> bool:
    """
    Ensures only a single instance of the bot process can run on the machine at a time.
    Binds a localhost TCP socket on a dedicated lock port.
    If another instance is already active, binding fails and startup is aborted.
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

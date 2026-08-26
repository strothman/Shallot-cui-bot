"""
Core Helper Functions, Delivery Fallbacks, and System Utilities for Shallot-CUI Bot.
"""

import os
import sys
import io
import asyncio
import logging
import subprocess
import aiohttp
import discord
from config import (
    COMFYUI_ADDRESS,
    COMFYUI_BATCH_PATH,
    VRAM_CAUTION_THRESHOLD_PERCENT,
    VRAM_MIN_FREE_GB,
)

logger = logging.getLogger("DiscordBot")

# Process holder class to manage global comfy_process across modules cleanly
class ComfyProcessHolder:
    def __init__(self):
        self.process = None

comfy_process_holder = ComfyProcessHolder()

# Reference to the active bot instance (injected during bot startup)
_active_bot = None

def set_active_bot(bot_instance):
    global _active_bot
    _active_bot = bot_instance

def get_active_bot():
    return _active_bot


async def safe_defer(interaction: discord.Interaction, thinking: bool = False, ephemeral: bool = False):
    """Safely defers an interaction response without raising 404 Unknown Interaction errors if token expired."""
    if not interaction.response.is_done():
        try:
            await interaction.response.defer(thinking=thinking, ephemeral=ephemeral)
        except (discord.NotFound, discord.HTTPException) as e:
            logger.debug(f"Interaction defer skipped or expired: {e}")


async def download_image(url: str) -> bytes:
    """Downloads image bytes from a remote URL using aiohttp."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                raise Exception(f"HTTP {resp.status} fetching image from {url}")


async def send_followup_fallback(interaction: discord.Interaction, content=None, embed=None, file=None, files=None, view=None, ephemeral=False):
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

    try:
        return await interaction.followup.send(**kwargs, ephemeral=ephemeral)
    except (discord.HTTPException, discord.NotFound) as hex_err:
        if getattr(hex_err, 'code', None) in [50027, 10062, 10015] or getattr(hex_err, 'status', None) in [404, 400] or isinstance(hex_err, discord.NotFound):
            logger.info(f"Interaction token expired ({getattr(hex_err, 'code', '404')}). Falling back to channel.send.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    bot = get_active_bot()
                    if bot:
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
            return await channel.send(**kwargs)
        else:
            raise hex_err


async def send_error_fallback(interaction: discord.Interaction, message: str):
    """Sends an error message using interaction, with fallback to channel.send if expired."""
    if len(message) > 1980:
        message = message[:1977] + "..."
    try:
        await interaction.followup.send(message, ephemeral=True)
    except (discord.HTTPException, discord.NotFound) as hex_err:
        logger.info(f"Interaction token expired ({getattr(hex_err, 'code', '404')}) during error report. Falling back to channel.send.")
        try:
            channel = interaction.channel
            if not channel and interaction.channel_id:
                bot = get_active_bot()
                if bot:
                    channel = await bot.fetch_channel(interaction.channel_id)
            if channel:
                tag = f"{interaction.user.mention} " if (interaction and interaction.user) else ""
                await channel.send(f"{tag}❌ {message.replace('❌ ', '')}")
        except Exception as e:
            logger.debug(f"Failed channel.send fallback in send_error_fallback: {e}")


async def edit_original_fallback(interaction: discord.Interaction, content=None, embed=None, view=None):
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

    try:
        await interaction.edit_original_response(**edit_kwargs)
    except (discord.HTTPException, discord.NotFound) as hex_err:
        if getattr(hex_err, 'code', None) in [50027, 10062, 10015] or getattr(hex_err, 'status', None) in [404, 400] or isinstance(hex_err, discord.NotFound):
            logger.info(f"Interaction token expired ({getattr(hex_err, 'code', '404')}) during edit. Falling back to channel.send.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    bot = get_active_bot()
                    if bot:
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
        else:
            raise hex_err


async def edit_message_fallback(interaction: discord.Interaction, message_id: int, content=None, embed=None, file=None, view=None):
    """Edits a status message by ID. Falls back to channel.send if interaction token expired/fails."""
    chan_id = interaction.channel_id
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
            bot = get_active_bot()
            channel = interaction.channel or (await bot.fetch_channel(chan_id) if bot else None)
            if channel:
                message = await channel.fetch_message(message_id)
                if message:
                    if file:
                        file.fp.seek(0)
                        edit_kwargs["attachments"] = [file]
                    await message.edit(**edit_kwargs)
                    return
        except Exception as e:
            logger.info(f"Could not edit message via Bot API ({e}). Falling back to interaction/channel send.")

    try:
        if file:
            file.fp.seek(0)
            edit_kwargs["attachments"] = [file]
        await interaction.followup.edit_message(message_id, **edit_kwargs)
    except (discord.HTTPException, discord.NotFound) as hex_err:
        if getattr(hex_err, 'code', None) in [50027, 10062, 10015] or getattr(hex_err, 'status', None) in [404, 400] or isinstance(hex_err, discord.NotFound):
            logger.info(f"Interaction token expired during message edit. Sending new message to channel.")
            channel = interaction.channel
            if not channel and interaction.channel_id:
                try:
                    bot = get_active_bot()
                    if bot:
                        channel = await bot.fetch_channel(interaction.channel_id)
                except Exception:
                    pass
            if channel:
                if file:
                    file.fp.seek(0)
                    send_kwargs["file"] = file
                await channel.send(**send_kwargs)
        else:
            raise hex_err


async def _update_button_state(interaction: discord.Interaction, custom_id: str, style: discord.ButtonStyle, disabled: bool = True):
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
    bot = get_active_bot()
    if not bot or not bot.is_ready():
        return
    try:
        if status_text:
            activity = discord.Activity(type=discord.ActivityType.custom, name="Custom Status", state=status_text)
            await bot.change_presence(activity=activity)
        else:
            activity = discord.Activity(type=discord.ActivityType.custom, name="Custom Status", state="Processing 0 jobs")
            await bot.change_presence(activity=activity)
    except Exception as e:
        logger.debug(f"Failed to update bot presence: {e}")


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


def terminate_existing_comfyui() -> bool:
    """Terminates any active ComfyUI processes (tracked PID or listening on port 8188)."""
    killed = False
    if comfy_process_holder.process and comfy_process_holder.process.returncode is None:
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(comfy_process_holder.process.pid)], capture_output=True)
            else:
                comfy_process_holder.process.terminate()
            comfy_process_holder.process = None
            killed = True
        except Exception as e:
            logger.error(f"Error terminating tracked comfy_process: {e}")

    if os.name == 'nt':
        try:
            port = COMFYUI_ADDRESS.split(":")[-1] if ":" in COMFYUI_ADDRESS else "8188"
            out = subprocess.run(
                f'netstat -aon | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            lines = out.stdout.strip().splitlines()
            pids_to_kill = set()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pid = parts[-1]
                    if pid != "0":
                        pids_to_kill.add(pid)

            for pid in pids_to_kill:
                subprocess.run(["taskkill", "/F", "/T", "/PID", pid], capture_output=True)
                killed = True
        except Exception as e:
            logger.error(f"Error finding/killing process on port 8188: {e}")
    return killed

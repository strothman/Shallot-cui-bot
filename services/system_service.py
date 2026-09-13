"""
System & Administration Service for Shallot-CUI Bot.
Provides decoupled business logic for:
- ComfyUI server process lifecycle (start, stop, status)
- GPU VRAM memory management and purging
- Real-time queue and system telemetry metrics
- Model architecture registry and auto-scanning
- Bot configuration and personalization settings (variation mode, settings.json)
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from typing import Optional, Tuple, Dict, Any

import aiohttp
import discord

from config import COMFYUI_ADDRESS, COMFYUI_BATCH_PATH, VRAM_MIN_FREE_GB
from core_helpers import check_gpu_vram_caution
import db
import model_architecture

logger = logging.getLogger("DiscordBot.SystemService")

# Settings persistence
SETTINGS_FILE = "settings.json"
settings: Dict[str, Any] = {"variation_mode": "high"}


def load_settings() -> Dict[str, Any]:
    """Loads configuration settings from settings.json."""
    global settings
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
            logger.info(f"Loaded {len(settings)} configuration setting(s) from {SETTINGS_FILE}.")
        except Exception as e:
            logger.error(f"Error loading settings file: {e}")
            settings = {"variation_mode": "high"}
    else:
        settings = {"variation_mode": "high"}
    return settings


def save_settings() -> bool:
    """Saves current configuration settings to settings.json."""
    global settings
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving settings file: {e}")
        return False


def get_setting(key: str, default: Any = None) -> Any:
    """Retrieves a configuration setting value."""
    return settings.get(key, default)


def set_setting(key: str, value: Any) -> bool:
    """Sets a configuration setting and persists to disk."""
    settings[key] = value
    return save_settings()


# Load settings on module import
load_settings()


# =========================================================================
# ComfyUI Server Process Lifecycle
# =========================================================================

def terminate_existing_comfyui(tracked_process=None) -> bool:
    """
    Terminates any active ComfyUI processes (tracked PID or listening on port 8188).
    Returns True if any process was terminated.
    """
    killed = False
    if tracked_process and tracked_process.returncode is None:
        try:
            if os.name == 'nt':
                subprocess.run(["taskkill", "/F", "/T", "/PID", str(tracked_process.pid)], capture_output=True)
            else:
                tracked_process.terminate()
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


async def fetch_comfyui_queue(address: str = None) -> Optional[dict]:
    """Fetch current queue status from ComfyUI REST API."""
    addr = address or COMFYUI_ADDRESS
    try:
        url = f"http://{addr}/queue"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


async def fetch_comfyui_system_stats(address: str = None) -> Optional[dict]:
    """Fetch system stats from ComfyUI REST API."""
    addr = address or COMFYUI_ADDRESS
    try:
        url = f"http://{addr}/system_stats"
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None


# =========================================================================
# UI Formatting & Embed Builders
# =========================================================================

def progress_bar(percent: float, length: int = 10) -> str:
    """Creates a text-based progress bar."""
    filled = max(0, min(length, int(length * percent / 100)))
    bar = "█" * filled + "░" * (length - filled)
    return f"[{bar}]"


def extract_short_prompt(prompt_dict: dict) -> str:
    """Extract a short prompt preview from a ComfyUI workflow dict."""
    if not prompt_dict:
        return "Unknown"
    for node_id, node in prompt_dict.items():
        if isinstance(node, dict) and node.get("class_type") == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            if text and text not in ["positive prompt placeholder", "negative prompt placeholder"]:
                if len(text) > 50:
                    text = text[:47] + "…"
                return text
    return "No text prompt"


def build_queue_embed(queue: Optional[dict], stats: Optional[dict]) -> discord.Embed:
    """Builds a rich embed detailing current ComfyUI queue and GPU usage."""
    if queue is None:
        return discord.Embed(
            title="⚫ ComfyUI Status",
            description="Could not connect to ComfyUI server.",
            color=discord.Color.dark_grey()
        )

    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])

    if len(running) > 0:
        color = discord.Color.gold()
        status_emoji = "🟡"
    elif len(pending) > 0:
        color = discord.Color.orange()
        status_emoji = "🟠"
    else:
        color = discord.Color.green()
        status_emoji = "🟢"

    embed = discord.Embed(
        title=f"{status_emoji} ComfyUI Queue Status",
        color=color
    )

    # System / GPU info
    if stats:
        devices = stats.get("devices", [])
        for dev in devices:
            name = dev.get("name", "Unknown GPU").split(" : ")[0]
            vram_total = dev.get("vram_total", 0)
            vram_free = dev.get("vram_free", 0)
            if vram_total > 0:
                vram_used = vram_total - vram_free
                pct = (vram_used / vram_total) * 100
                vram_bar = progress_bar(pct)
                embed.add_field(
                    name="🖥️ GPU",
                    value=f"`{name}`\n{vram_bar} {vram_used / (1024**3):.1f} / {vram_total / (1024**3):.1f} GB ({pct:.0f}%)",
                    inline=False
                )

    # Engine-Aware Priority Queue Status
    try:
        from services.engine_queue import get_engine_queue
        eq_status = get_engine_queue().get_status()
        if eq_status.get("is_running"):
            engine_text = f"**Current Engine:** `{eq_status['current_engine_name']}`"
            if eq_status.get("active_job"):
                aj = eq_status["active_job"]
                engine_text += f"\n⚡ **Active Task:** [{aj['engine'].upper()}] {aj['description']} ({aj['running_seconds']}s)"
            if eq_status.get("pending_by_engine"):
                breakdown = ", ".join(f"`{k.upper()}`: {v}" for k, v in eq_status["pending_by_engine"].items())
                engine_text += f"\n⏳ **Queued by Engine:** {breakdown}"
            if eq_status.get("stats", {}).get("switches_prevented", 0) > 0:
                engine_text += f"\n🚀 **VRAM Swaps Prevented:** {eq_status['stats']['switches_prevented']}"
            embed.add_field(name="🧠 Engine-Aware Scheduler", value=engine_text, inline=False)
    except Exception as eq_ui_err:
        logger.debug(f"Engine-Aware Queue UI status error: {eq_ui_err}")

    # Active jobs
    if running:
        active_lines = []
        for job in running:
            prompt_id = job[1][:8] if len(job) > 1 else "?"
            prompt_json = job[2] if len(job) > 2 else {}
            prompt_text = extract_short_prompt(prompt_json)
            active_lines.append(f"⚡ `{prompt_id}…` — {prompt_text}")
        embed.add_field(
            name=f"🔥 Active ({len(running)})",
            value="\n".join(active_lines[:5]),
            inline=False
        )
    else:
        embed.add_field(name="Active", value="None", inline=True)

    # Pending jobs
    if pending:
        pending_lines = []
        for idx, job in enumerate(pending[:5]):
            prompt_id = job[1][:8] if len(job) > 1 else "?"
            prompt_json = job[2] if len(job) > 2 else {}
            prompt_text = extract_short_prompt(prompt_json)
            pending_lines.append(f"#{idx+1} `{prompt_id}…` — {prompt_text}")
        footer_extra = f"\n… and {len(pending) - 5} more" if len(pending) > 5 else ""
        embed.add_field(
            name=f"📋 Pending ({len(pending)})",
            value="\n".join(pending_lines) + footer_extra,
            inline=False
        )
    else:
        embed.add_field(name="Pending", value="None", inline=True)

    embed.set_footer(text=f"ComfyUI @ {COMFYUI_ADDRESS}")
    return embed


def build_models_embed(architecture: str = "all", model_type: str = "all") -> Optional[discord.Embed]:
    """Builds the registered models and LoRAs directory embed."""
    arch_filter = None if architecture == "all" else architecture
    type_filter = None if model_type == "all" else model_type
    
    entries = db.get_models_by_architecture(base_architecture=arch_filter, model_type=type_filter)
    if not entries:
        return None

    embed = discord.Embed(
        title="📦 Registered Model & LoRA Architectures",
        description="Architecture registry ensures compatible checkpoint and LoRA pairing without tensor mismatch errors.",
        color=discord.Color.blurple()
    )

    grouped: Dict[str, list] = {}
    for entry in entries:
        arch = entry.get("base_architecture", "unknown").upper()
        grouped.setdefault(arch, []).append(entry)

    for arch, items in grouped.items():
        lines = []
        for it in items[:12]:
            badge = model_architecture.get_architecture_badge(it.get("base_architecture", ""))
            mtype = "Checkpoint" if it.get("model_type") == "checkpoint" else "LoRA"
            subtype = f" ({it.get('sub_type')})" if it.get("sub_type") and it.get("sub_type") != "standard" else ""
            lines.append(f"• **{it.get('display_name')}** `{badge}` — *{mtype}{subtype}*")
        
        if len(items) > 12:
            lines.append(f"*...and {len(items) - 12} more*")

        embed.add_field(name=f"🏛️ {arch} Models ({len(items)})", value="\n".join(lines), inline=False)

    return embed


async def purge_vram_core(client=None) -> bool:
    """Sends command to ComfyUI client to unload models and free PyTorch CUDA cache."""
    if client is None:
        from comfy_client import comfy_client as default_client
        client = default_client
    return await client.free_memory(unload_models=True, free_memory=True)

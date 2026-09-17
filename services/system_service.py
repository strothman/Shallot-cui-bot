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
import time
import json
import logging
import asyncio
import subprocess
import gc
from typing import Optional, Tuple, Dict, Any, List

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


async def fetch_comfyui_queue(address: str = None, session: Optional[aiohttp.ClientSession] = None) -> Optional[dict]:
    """Fetch current queue status from ComfyUI REST API."""
    addr = address or COMFYUI_ADDRESS
    try:
        url = f"http://{addr}/queue"
        if session and not session.closed:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
        else:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                    if resp.status == 200:
                        return await resp.json()
    except Exception:
        pass
    return None


async def fetch_comfyui_system_stats(address: str = None, session: Optional[aiohttp.ClientSession] = None) -> Optional[dict]:
    """Fetch system stats from ComfyUI REST API."""
    addr = address or COMFYUI_ADDRESS
    try:
        url = f"http://{addr}/system_stats"
        if session and not session.closed:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
                if resp.status == 200:
                    return await resp.json()
        else:
            async with aiohttp.ClientSession() as temp_session:
                async with temp_session.get(url, timeout=aiohttp.ClientTimeout(total=3)) as resp:
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

    eq_status = {}
    # Engine-Aware Priority Queue Status
    try:
        from services.engine_queue import get_engine_queue
        eq = get_engine_queue()
        if eq and eq.get_status().get("is_running"):
            eq_status = eq.get_status()
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
    elif eq_status.get("active_job"):
        aj = eq_status["active_job"]
        embed.add_field(
            name="🔥 Active (1)",
            value=f"⚡ `[{aj['engine'].upper()}] {aj['job_id'][:8]}` — {aj['description']} ({aj['running_seconds']}s)",
            inline=False
        )
    else:
        embed.add_field(name="Active", value="None", inline=True)

    # Pending jobs (combined ComfyUI native + EngineAwareQueue pending)
    eq_pending_jobs = eq_status.get("pending_jobs", [])
    total_pending = len(pending) + len(eq_pending_jobs)
    if total_pending > 0:
        pending_lines = []
        item_idx = 1
        # ComfyUI native pending jobs
        for job in pending:
            prompt_id = job[1][:8] if len(job) > 1 else "?"
            prompt_json = job[2] if len(job) > 2 else {}
            prompt_text = extract_short_prompt(prompt_json)
            pending_lines.append(f"#{item_idx} `[COMFY] {prompt_id}…` — {prompt_text}")
            item_idx += 1
            if len(pending_lines) >= 5:
                break
        # EngineAwareQueue pending jobs
        if len(pending_lines) < 5:
            for j in eq_pending_jobs:
                job_id = j.get("job_id", "")[:8]
                engine = j.get("engine", "JOB").upper()
                desc = j.get("description", "Generation")
                wait_sec = j.get("waiting_seconds", 0)
                pending_lines.append(f"#{item_idx} `[{engine}] {job_id}` — {desc} ({wait_sec}s)")
                item_idx += 1
                if len(pending_lines) >= 5:
                    break

        footer_extra = f"\n… and {total_pending - 5} more" if total_pending > 5 else ""
        embed.add_field(
            name=f"📋 Pending ({total_pending})",
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


def trim_process_working_set(pid: Optional[int] = None) -> bool:
    """
    Trims the working set of the specified process (or current process if pid is None)
    by paging out unreferenced physical RAM pages to standby pool using the Windows API EmptyWorkingSet.
    Safely no-ops on non-Windows platforms.
    """
    if os.name != 'nt':
        return False
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        psapi.EmptyWorkingSet.argtypes = [ctypes.c_void_p]
        psapi.EmptyWorkingSet.restype = ctypes.c_int

        if pid is None or pid == os.getpid():
            h_process = kernel32.GetCurrentProcess()
            return bool(psapi.EmptyWorkingSet(h_process))

        # PROCESS_QUERY_INFORMATION (0x0400) | PROCESS_SET_QUOTA (0x0100)
        flags = 0x0400 | 0x0100
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int

        h_process = kernel32.OpenProcess(flags, False, int(pid))
        if not h_process:
            return False
        try:
            return bool(psapi.EmptyWorkingSet(h_process))
        finally:
            kernel32.CloseHandle(h_process)
    except Exception as e:
        logger.debug(f"Failed to trim working set for PID {pid}: {e}")
        return False


def get_comfyui_pids() -> List[int]:
    """Finds all PIDs of processes listening on the ComfyUI port (e.g. 8188)."""
    pids = []
    if os.name == 'nt':
        try:
            port = COMFYUI_ADDRESS.split(":")[-1] if ":" in COMFYUI_ADDRESS else "8188"
            out = subprocess.run(
                f'netstat -aon | findstr :{port}',
                shell=True, capture_output=True, text=True
            )
            lines = out.stdout.strip().splitlines()
            for line in lines:
                parts = line.split()
                if len(parts) >= 5 and "LISTENING" in line:
                    pid = int(parts[-1])
                    if pid > 0 and pid not in pids:
                        pids.append(pid)
        except Exception as e:
            logger.debug(f"Error finding ComfyUI PIDs: {e}")
    return pids


def _get_default_client():
    """Dynamically resolves a ComfyClient instance."""
    try:
        from comfy_client import comfy_client as default_client
        if default_client is not None:
            return default_client
    except Exception:
        pass
    try:
        from services.generation_service import _comfy_client
        if _comfy_client is not None:
            return _comfy_client
    except Exception:
        pass
    try:
        from comfy_client import ComfyClient
        from config import COMFYUI_ADDRESS
        return ComfyClient(server_address=COMFYUI_ADDRESS)
    except Exception:
        return None


async def purge_vram_core(client=None, trim_working_set: bool = True) -> bool:
    """Sends command to ComfyUI client to unload models, free PyTorch CUDA cache, and trim system RAM working set."""
    if client is None:
        client = _get_default_client()
    
    success = False
    try:
        if client and hasattr(client, "free_memory"):
            success = await client.free_memory(unload_models=True, free_memory=True)
    except Exception as e:
        logger.warning(f"Error calling comfy_client.free_memory: {e}")


    await asyncio.to_thread(gc.collect)

    if trim_working_set and os.name == 'nt':
        try:
            # 1. Trim bot's own working set
            await asyncio.to_thread(trim_process_working_set, os.getpid())

            # 2. Trim ComfyUI processes listening on port 8188
            comfy_pids = await asyncio.to_thread(get_comfyui_pids)
            for c_pid in comfy_pids:
                await asyncio.to_thread(trim_process_working_set, c_pid)
        except Exception as trim_err:
            logger.debug(f"Working set trim warning: {trim_err}")

    return success


# =========================================================================
# System Activity, Console Title, and Bot Presence Synchronization
# =========================================================================

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


def get_effective_queue_counts(comfy_queue: Optional[dict] = None) -> tuple[int, int]:
    """
    Calculates combined active (running) and pending jobs across ComfyUI and EngineAwareQueue.
    ComfyUI processes 1 job at a time while EngineAwareQueue buffers pending jobs
    to prevent VRAM thrashing on 8GB GPUs.
    """
    running = 0
    pending = 0
    if comfy_queue:
        running = len(comfy_queue.get("queue_running", []))
        pending = len(comfy_queue.get("queue_pending", []))

    try:
        from services.engine_queue import get_engine_queue
        eq = get_engine_queue()
        if eq and eq.get_status().get("is_running", False):
            eq_status = eq.get_status()
            if eq_status.get("active_job"):
                running = max(running, 1)
            pending += eq_status.get("pending_count", 0)
    except Exception as e:
        logger.debug(f"Could not read EngineAwareQueue status: {e}")

    return running, pending


def format_presence_status_text(running: int, pending: int) -> tuple[str, discord.ActivityType]:
    """Generates the presence string and Discord activity type from queue counts."""
    if running > 0:
        status_text = f"Processing {running} job{'s' if running != 1 else ''}"
        if pending > 0:
            status_text += f" | {pending} queued"
        return status_text, discord.ActivityType.playing
    elif pending > 0:
        status_text = f"{pending} queued job{'s' if pending != 1 else ''}"
        return status_text, discord.ActivityType.watching
    else:
        return "Ready ✓ | /imagine", discord.ActivityType.watching


_last_presence_sync_time: float = 0.0
_presence_sync_task: Optional[asyncio.Task] = None

_last_generation_activity: float = time.time()
_idle_purged: bool = False
IDLE_PURGE_TIMEOUT: float = float(os.getenv("IDLE_PURGE_TIMEOUT_SECONDS", "1800"))  # 30 minutes


def touch_activity():
    """Updates the last activity timestamp and resets the idle purge state."""
    global _last_generation_activity, _idle_purged
    _last_generation_activity = time.time()
    _idle_purged = False


def get_last_generation_activity() -> float:
    return _last_generation_activity


def is_idle_purged() -> bool:
    return _idle_purged


def set_idle_purged(val: bool):
    global _idle_purged
    _idle_purged = val


async def sync_presence_now(bot=None, client=None):
    """Immediately synchronizes bot presence with current ComfyUI and EngineAwareQueue state."""
    global _last_presence_sync_time
    _last_presence_sync_time = time.time()
    if bot is None:
        from core_helpers import get_active_bot
        bot = get_active_bot()
    if client is None:
        client = _get_default_client()

    if not bot:
        return

    try:
        session = getattr(client, "session", None)
        queue = await fetch_comfyui_queue(session=session)
        if queue is None:
            activity = discord.Activity(
                type=discord.ActivityType.watching,
                name="ComfyUI (offline)"
            )
            await bot.change_presence(status=discord.Status.dnd, activity=activity)
            update_console_title("ComfyUI (offline)")
            return

        running, pending = get_effective_queue_counts(queue)
        status_text, activity_type = format_presence_status_text(running, pending)

        activity = discord.Activity(
            type=activity_type,
            name=status_text
        )
        await bot.change_presence(status=discord.Status.online, activity=activity)
        update_console_title(status_text)
    except Exception as e:
        logger.debug(f"Presence update failed: {e}")


def on_engine_queue_change(bot=None, client=None):
    """Throttled callback triggered on EngineAwareQueue enqueue/start/cancel/completion."""
    global _last_presence_sync_time, _presence_sync_task
    touch_activity()
    if bot is None:
        from core_helpers import get_active_bot
        bot = get_active_bot()
    if not bot or not bot.is_ready():
        return
    now = time.time()
    # Respect Discord rate limits (minimum 3.5s interval between gateway presence updates)
    if now - _last_presence_sync_time >= 3.5:
        if _presence_sync_task and not _presence_sync_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            _presence_sync_task = loop.create_task(sync_presence_now(bot, client))
        except RuntimeError:
            pass
    else:
        if _presence_sync_task is None or _presence_sync_task.done():
            delay = max(0.5, 3.5 - (now - _last_presence_sync_time))
            async def _delayed_sync():
                await asyncio.sleep(delay)
                await sync_presence_now(bot, client)
            try:
                loop = asyncio.get_running_loop()
                _presence_sync_task = loop.create_task(_delayed_sync())
            except RuntimeError:
                pass


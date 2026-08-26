"""
Shallot-CUI Bot — Real-Time Live WebSocket Terminal Monitor & HUD

Connects to ComfyUI WebSocket and REST APIs to provide a sub-second,
flicker-free live terminal HUD displaying:
- Live GPU VRAM allocation meter
- Step-by-step KSampler / Diffusion progress bar (e.g. [████████░░] 80% Step 22/28)
- Active executing node identification
- Real-time queue depth & pending prompt previews
"""

import os
import sys
import json
import time
import uuid
import asyncio
from datetime import datetime
import aiohttp

# Load environment if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

COMFYUI_ADDRESS = os.getenv("COMFYUI_ADDRESS", "127.0.0.1:8188")
HTTP_URL = f"http://{COMFYUI_ADDRESS}"
WS_URL = f"ws://{COMFYUI_ADDRESS}/ws"

# ANSI Escape Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def render_ascii_bar(value: int, max_val: int, length: int = 24) -> str:
    """Renders a filled Unicode block progress bar."""
    if max_val <= 0 or value <= 0:
        return "░" * length
    fraction = min(1.0, max(0.0, value / max_val))
    filled = int(round(fraction * length))
    return ("█" * filled) + ("░" * (length - filled))

def format_bytes(bytes_num):
    if not bytes_num:
        return "0.00 GB"
    gb = bytes_num / (1024 ** 3)
    return f"{gb:.2f} GB"

def extract_prompt_preview(prompt_dict):
    if not isinstance(prompt_dict, dict):
        return "Custom Generation"
    texts = []
    for _, node in prompt_dict.items():
        if isinstance(node, dict) and node.get("class_type") in ("CLIPTextEncode", "TextEncode", "ShowText|pysssss"):
            txt = node.get("inputs", {}).get("text", "")
            if txt and txt not in ["positive prompt placeholder", "negative prompt placeholder"] and len(txt) > 2:
                if len(txt) > 36:
                    txt = txt[:33] + "..."
                texts.append(txt)
    return " | ".join(texts) if texts else "Image / Video Pipeline"

try:
    import db
except ImportError:
    db = None

class ComfyLiveMonitor:
    def __init__(self):
        self.client_id = f"monitor_{uuid.uuid4().hex[:8]}"
        self.is_online = False
        self.gpu_info = []
        self.queue_running = []
        self.queue_pending = []
        self.current_step = 0
        self.max_steps = 0
        self.current_node_id = None
        self.current_node_type = "Idle"
        self.current_node_title = ""
        self.active_prompt_text = ""
        self.last_draw_time = 0
        self.session = None

    async def fetch_system_stats(self):
        """Polls GPU VRAM, queue status, and shared live telemetry every second."""
        while True:
            try:
                if self.session and not self.session.closed:
                    # 1. System Stats (VRAM)
                    async with self.session.get(f"{HTTP_URL}/system_stats", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            self.gpu_info = data.get("devices", [])
                            self.is_online = True
                        else:
                            self.is_online = False

                    # 2. Queue Status
                    async with self.session.get(f"{HTTP_URL}/queue", timeout=aiohttp.ClientTimeout(total=2)) as resp:
                        if resp.status == 200:
                            q_data = await resp.json()
                            self.queue_running = q_data.get("queue_running", [])
                            self.queue_pending = q_data.get("queue_pending", [])

                # 3. Inter-process Live Telemetry from SQLite
                if db:
                    live_stat = db.get_live_status()
                    if live_stat and self.queue_running:
                        if live_stat.get("step") is not None and live_stat.get("max_steps") is not None:
                            self.current_step = live_stat["step"]
                            self.max_steps = live_stat["max_steps"]
                        if live_stat.get("node_id") and not self.current_node_id:
                            self.current_node_id = live_stat["node_id"]
                    elif not self.queue_running:
                        self.current_step = 0
                        self.max_steps = 0

                # Resolve active executing node title from running job graph
                if self.queue_running and len(self.queue_running[0]) > 2:
                    prompt_graph = self.queue_running[0][2]
                    if not self.active_prompt_text:
                        self.active_prompt_text = extract_prompt_preview(prompt_graph)
                    if self.current_node_id and str(self.current_node_id) in prompt_graph:
                        node_dict = prompt_graph[str(self.current_node_id)]
                        self.current_node_title = node_dict.get("_meta", {}).get("title") or node_dict.get("class_type", f"Node #{self.current_node_id}")
            except Exception:
                self.is_online = False

            self.draw_ui()
            await asyncio.sleep(0.8)

    def draw_ui(self):
        """Draws the terminal HUD."""
        # Use clear screen + cursor home for smooth updates
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = []

        lines.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
        lines.append(f"{BOLD}{CYAN}║{WHITE}   🧅 Shallot-CUI Bot  •  ComfyUI Live Terminal HUD  -  {now_str}  {CYAN}║{RESET}")
        lines.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}")

        # Server Status Line
        status_badge = f"{BOLD}{GREEN}● ONLINE{RESET}" if self.is_online else f"{BOLD}{RED}● OFFLINE (Reconnecting...){RESET}"
        lines.append(f" Server: {status_badge}  {DIM}Endpoint:{RESET} {HTTP_URL}")

        # GPU VRAM Visual Meters
        if self.gpu_info:
            for idx, dev in enumerate(self.gpu_info):
                name = dev.get("name", f"GPU {idx}")
                vram_free = dev.get("vram_free", 0)
                vram_total = dev.get("vram_total", 0)
                if vram_total > 0:
                    vram_used = vram_total - vram_free
                    pct = (vram_used / vram_total) * 100
                    color = GREEN if pct < 70 else (YELLOW if pct < 85 else RED)
                    vram_bar = render_ascii_bar(vram_used, vram_total, length=18)
                    lines.append(f" {BOLD}{WHITE}{name}:{RESET} {color}[{vram_bar}] {pct:.1f}%{RESET} ({format_bytes(vram_used)} / {format_bytes(vram_total)})")
        else:
            lines.append(f" {DIM}GPU VRAM: Waiting for server telemetry...{RESET}")

        lines.append(f"{BOLD}{CYAN}────────────────────────────────────────────────────────────────────{RESET}")

        # Active Job Progress Section
        lines.append(f"{BOLD}{MAGENTA}▶ ACTIVE GENERATION PROGRESS:{RESET}")
        if self.queue_running and self.max_steps > 0:
            pct = min(100, int((self.current_step / self.max_steps) * 100))
            prog_bar = render_ascii_bar(self.current_step, self.max_steps, length=28)
            node_label = self.current_node_title if self.current_node_title else (f"Node #{self.current_node_id}" if self.current_node_id else "Sampling")
            lines.append(f"  {BOLD}{GREEN}[{prog_bar}] {pct}%{RESET}  •  Step {BOLD}{self.current_step}/{self.max_steps}{RESET}  •  {CYAN}{node_label}{RESET}")
            if self.active_prompt_text:
                lines.append(f"  {DIM}Prompt:{RESET} {YELLOW}{self.active_prompt_text}{RESET}")
        elif self.queue_running:
            stage_desc = f"Running: {self.current_node_title}" if self.current_node_title else "Initializing / Model Loading..."
            lines.append(f"  {BOLD}{YELLOW}[ {stage_desc} ]{RESET} Active Jobs: {len(self.queue_running)}")
            if self.active_prompt_text:
                lines.append(f"  {DIM}Prompt:{RESET} {YELLOW}{self.active_prompt_text}{RESET}")
        else:
            lines.append(f"  {DIM}[ Idle • Ready for new generations from Discord ]{RESET}")

        lines.append(f"{BOLD}{CYAN}────────────────────────────────────────────────────────────────────{RESET}")

        # Queue Depth
        active_count = len(self.queue_running)
        pending_count = len(self.queue_pending)
        lines.append(f"{BOLD}{BLUE}📋 QUEUE STATUS:{RESET}  {BOLD}Active:{RESET} {GREEN}{active_count}{RESET}  |  {BOLD}Pending in line:{RESET} {YELLOW}{pending_count}{RESET}")

        if self.queue_pending:
            lines.append(f"  {BOLD}Upcoming Jobs:{RESET}")
            for idx, job in enumerate(self.queue_pending[:4]):
                p_num = job[0]
                p_json = job[2] if len(job) > 2 else {}
                snippet = extract_prompt_preview(p_json)
                lines.append(f"   {DIM}#{idx+1} [Job {p_num}]:{RESET} {snippet}")
            if len(self.queue_pending) > 4:
                lines.append(f"   {DIM}... and {len(self.queue_pending) - 4} more queued{RESET}")

        lines.append(f"\n{DIM}Live WebSocket stream active. Press Ctrl+C to minimize/exit.{RESET}\n")

        # Output to console with clear screen
        if os.name == 'nt':
            os.system('cls')
        else:
            print("\033[H\033[J", end="")
        print("\n".join(lines))

    async def listen_websocket(self):
        """Maintains persistent WebSocket stream with ComfyUI."""
        while True:
            try:
                async with self.session.ws_connect(f"{WS_URL}?clientId={self.client_id}") as ws:
                    self.is_online = True
                    self.draw_ui()

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            mtype = data.get("type")
                            mdata = data.get("data", {})

                            if mtype == "progress":
                                self.current_step = mdata.get("value", 0)
                                self.max_steps = mdata.get("max", 0)
                                self.current_node_id = mdata.get("node")
                                self.draw_ui()

                            elif mtype == "executing":
                                node_id = mdata.get("node")
                                if node_id is None:
                                    # Prompt finished
                                    self.current_step = 0
                                    self.max_steps = 0
                                    self.current_node_id = None
                                    self.current_node_title = ""
                                    self.active_prompt_text = ""
                                else:
                                    self.current_node_id = node_id
                                self.draw_ui()

                            elif mtype == "status":
                                status_obj = mdata.get("status", {})
                                exec_info = status_obj.get("exec_info", {})
                                q_remaining = exec_info.get("queue_remaining", 0)
                                self.draw_ui()

                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                            break
            except Exception:
                self.is_online = False
                self.current_step = 0
                self.max_steps = 0
                self.draw_ui()
                await asyncio.sleep(2.0)

    async def run(self):
        # Enable Windows ANSI escape sequence color support
        if os.name == 'nt':
            os.system('')

        async with aiohttp.ClientSession() as session:
            self.session = session
            await asyncio.gather(
                self.fetch_system_stats(),
                self.listen_websocket()
            )

def main():
    monitor = ComfyLiveMonitor()
    try:
        asyncio.run(monitor.run())
    except KeyboardInterrupt:
        print(f"\n{GREEN}Monitor closed.{RESET}")
        sys.exit(0)

if __name__ == "__main__":
    main()

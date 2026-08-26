import os
import sys
import json
import time
import urllib.request
from datetime import datetime

# Load environment if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

COMFYUI_ADDRESS = os.getenv("COMFYUI_ADDRESS", "127.0.0.1:8188")
API_URL = f"http://{COMFYUI_ADDRESS}"

# ANSI Escape Colors
RESET = "\033[0m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
CYAN = "\033[36m"
WHITE = "\033[37m"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def fetch_json(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'ComfyMonitor'})
        with urllib.request.urlopen(req, timeout=1.0) as response:
            return json.loads(response.read().decode('utf-8'))
    except Exception:
        return None

def extract_prompt_text(prompt_dict):
    if not prompt_dict:
        return "Unknown"
    
    texts = []
    # Try to extract the positive prompt text
    for node_id, node in prompt_dict.items():
        if node.get("class_type") == "CLIPTextEncode":
            text = node.get("inputs", {}).get("text", "")
            # Skip placeholders and negative prompts (which are usually much longer or have negative tags)
            if text and text not in ["positive prompt placeholder", "negative prompt placeholder"]:
                # Limit length for display
                if len(text) > 40:
                    text = text[:37] + "..."
                texts.append(text)
    
    return " | ".join(texts) if texts else "No text prompt"

def format_bytes(bytes_num):
    if not bytes_num:
        return "N/A"
    gb = bytes_num / (1024 ** 3)
    return f"{gb:.2f} GB"

def draw_dashboard():
    clear_screen()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"{BOLD}{CYAN}=================================================={RESET}")
    print(f"{BOLD}{WHITE}   ComfyUI CLI Monitor  -  {now}{RESET}")
    print(f"{BOLD}{CYAN}=================================================={RESET}\n")
    
    # 1. Fetch System Stats
    stats = fetch_json(f"{API_URL}/system_stats")
    if stats is None:
        print(f"Status: {BOLD}{RED}OFFLINE{RESET} (Could not connect to {API_URL})")
        print("\nPress Ctrl+C to exit. Retrying in 2 seconds...")
        return
        
    print(f"Status: {BOLD}{GREEN}ONLINE{RESET} ({COMFYUI_ADDRESS})")
    
    # Draw GPU / Memory info
    devices = stats.get("devices", [])
    for idx, dev in enumerate(devices):
        name = dev.get("name", "Unknown GPU")
        vram_free = dev.get("vram_free")
        vram_total = dev.get("vram_total")
        
        vram_str = ""
        if vram_free and vram_total:
            used = vram_total - vram_free
            percent = (used / vram_total) * 100
            vram_str = f"VRAM: {format_bytes(used)} / {format_bytes(vram_total)} ({percent:.1f}%)"
        
        print(f"Device [{idx}]: {BOLD}{WHITE}{name}{RESET}")
        if vram_str:
            print(f"  {vram_str}")
            
    print(f"\n{BOLD}{BLUE}--- QUEUE STATUS ---{RESET}")
    
    # 2. Fetch Queue Stats
    queue = fetch_json(f"{API_URL}/queue")
    if not queue:
        print("Failed to fetch queue info.")
        return
        
    running = queue.get("queue_running", [])
    pending = queue.get("queue_pending", [])
    
    # Running Jobs
    print(f"Active Jobs: {BOLD}{GREEN}{len(running)}{RESET}")
    for job in running:
        prompt_number = job[0]
        prompt_id = job[1]
        prompt_json = job[2]
        prompt_text = extract_prompt_text(prompt_json)
        print(f"  [{prompt_number}] {BOLD}{GREEN}RUNNING{RESET} - ID: {prompt_id[:8]}... - '{prompt_text}'")
        
    # Pending Jobs
    print(f"Pending Jobs: {BOLD}{YELLOW}{len(pending)}{RESET}")
    for idx, job in enumerate(pending[:5]):  # Show top 5
        prompt_number = job[0]
        prompt_id = job[1]
        prompt_json = job[2]
        prompt_text = extract_prompt_text(prompt_json)
        print(f"  [{prompt_number}] PENDING #{idx+1} - ID: {prompt_id[:8]}... - '{prompt_text}'")
        
    if len(pending) > 5:
        print(f"  ... and {len(pending) - 5} more pending jobs.")
        
    print(f"\n{BOLD}{CYAN}--------------------------------------------------{RESET}")
    print("Auto-refreshing every 20 seconds. Press Ctrl+C to exit.")

def main():
    # Enable virtual terminal (ANSI color support) on Windows CMD/PowerShell
    if os.name == 'nt':
        os.system('')
        
    try:
        while True:
            draw_dashboard()
            time.sleep(20.0)
    except KeyboardInterrupt:
        print("\nExiting monitor.")
        sys.exit(0)

if __name__ == "__main__":
    main()

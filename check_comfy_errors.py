r"""
ComfyUI Server Error Analyzer & Diagnostics Tool
Shallot-CUI Bot Companion

Scans C:\ComfyUI\ComfyUI\user\comfyui.log and analyzes:
- Critical Runtime Errors & Crashes (CUDA OOM, PyTorch tensor mismatches)
- Custom Node Import Failures & Missing Dependencies
- Missing Model Checkpoints, LoRAs, and Motion Models
- Harmless Notices (e.g. extra_pnginfo, deprecation notices)
- Hardware & VRAM health status

Provides actionable remediation steps and generates a clean copy-pasteable
report formatted for Gemini / AI assistants.
"""

import os
import sys
import re
import io
from datetime import datetime

# Set default console output encoding to UTF-8
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

LOG_PATH_CANDIDATES = [
    r"C:\ComfyUI\ComfyUI\user\comfyui.log",
    r"C:\ComfyUI\user\comfyui.log",
    r"C:\ComfyUI\ComfyUI\comfyui.log",
    r"C:\ComfyUI\comfyui.log",
]

# ANSI Colors
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

def find_log_file():
    for p in LOG_PATH_CANDIDATES:
        if os.path.isfile(p):
            return p
    return None

KNOWN_RULES = [
    {
        "pattern": r"\[AnimateDiffEvo\]\s*-\s*ERROR\s*-\s*No motion models found",
        "category": "Missing Model",
        "severity": "WARNING",
        "title": "AnimateDiff: No Motion Models Found",
        "explanation": "ComfyUI AnimateDiff Evolved custom node is loaded, but no AnimateDiff motion model files (.safetensors / .ckpt) were found in the models folder.",
        "action": "If you plan to use AnimateDiff, download motion models (e.g., v3_sd15_mm.ckpt) into `C:\\ComfyUI\\ComfyUI\\models\\animatediff_models\\`. If you only use Wan 2.2 / LTX for video, this warning can be safely ignored."
    },
    {
        "pattern": r"extra_pnginfo\[\d+\] is not a dict or missing 'workflow' key",
        "category": "Harmless Notice",
        "severity": "INFO",
        "title": "API Prompt Notice (extra_pnginfo)",
        "explanation": "ComfyUI notices that the generation request was submitted via API without attaching a visual WebUI node graph.",
        "action": "Safe to ignore. Generation workflows and output image saves complete normally."
    },
    {
        "pattern": r"The pynvml package is deprecated",
        "category": "Deprecation Notice",
        "severity": "INFO",
        "title": "PyNVML Deprecation Notice",
        "explanation": "PyTorch / Torchvision uses legacy pynvml wrapper.",
        "action": "Safe to ignore. Does not affect GPU memory or performance."
    },
    {
        "pattern": r"No OpenGL_accelerate module loaded",
        "category": "Optional Dependency",
        "severity": "INFO",
        "title": "OpenGL Acceleration Disabled",
        "explanation": "Optional PyOpenGL acceleration helper is not installed in the embedded Python environment.",
        "action": "Safe to ignore. ComfyUI uses CUDA directly for all image and video operations."
    },
    {
        "pattern": r"CUDA out of memory",
        "category": "VRAM Exhaustion",
        "severity": "ERROR",
        "title": "CUDA Out Of Memory (OOM)",
        "explanation": "The GPU ran out of dedicated VRAM while sampling a large model or high resolution.",
        "action": "1. Ensure `--lowvram` or `--reserve-vram 2` is enabled in `run_nvidia_gpu.bat`.\n2. Avoid running external video transcoders (Tdarr, Handbrake) simultaneously.\n3. Use 2-stage latent upscaling (`/sdxl`) instead of high native resolutions."
    },
    {
        "pattern": r"ImportError:\s*No module named ['\"]([^'\"]+)['\"]",
        "category": "Missing Python Package",
        "severity": "ERROR",
        "title": "Custom Node Missing Dependency",
        "explanation": "A custom node failed to load because a required Python package is missing.",
        "action": "Open terminal in `C:\\ComfyUI` and run `.\\python_embeded\\python.exe -m pip install <package_name>`."
    },
    {
        "pattern": r"Some nodes \((\d+)\) could not be loaded",
        "category": "Custom Node Partial Load",
        "severity": "WARNING",
        "title": "Custom Nodes Partial Load Failure",
        "explanation": "Certain custom node classes could not be loaded due to optional dependencies.",
        "action": "If your workflows don't use those specific nodes, it can be safely ignored. Otherwise, update the custom node via ComfyUI Manager."
    }
]

def analyze_log(log_path: str, max_lines: int = 1500):
    if not os.path.exists(log_path):
        return None

    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()

    lines = all_lines[-max_lines:] if len(all_lines) > max_lines else all_lines
    full_text = "".join(lines)

    # 1. Extract Hardware / Platform Info
    hw_info = {}
    m = re.search(r"\*\*\s*Python version:\s*([^\r\n]+)", full_text)
    if m: hw_info["python"] = m.group(1).strip()
    m = re.search(r"\*\*\s*ComfyUI version:\s*([^\r\n]+)", full_text)
    if m: hw_info["comfyui_version"] = m.group(1).strip()
    m = re.search(r"Device:\s*cuda:\d+\s+([^\r\n:]+)", full_text)
    if m: hw_info["gpu"] = m.group(1).strip()
    m = re.search(r"Total VRAM\s*(\d+)\s*MB", full_text)
    if m: hw_info["vram_mb"] = m.group(1).strip()
    m = re.search(r"CPU:\s*([^\r\n]+)", full_text)
    if m: hw_info["cpu"] = m.group(1).strip()
    m = re.search(r"NVIDIA Driver:\s*([^\r\n]+)", full_text)
    if m: hw_info["driver"] = m.group(1).strip()

    # 2. Match Rules
    findings = []
    matched_patterns = set()

    for rule in KNOWN_RULES:
        matches = list(re.finditer(rule["pattern"], full_text, re.IGNORECASE))
        if matches:
            matched_patterns.add(rule["pattern"])
            findings.append({
                "rule": rule,
                "count": len(matches),
                "last_match": matches[-1].group(0)
            })

    # 3. Catch generic tracebacks / unhandled errors
    tracebacks = []
    tb_pattern = r"(Traceback \(most recent call last\):[\s\S]*?(?:Error|Exception): [^\r\n]+)"
    for tb_match in re.finditer(tb_pattern, full_text):
        tb_text = tb_match.group(1)
        # Skip if already categorized
        if not any(pat in tb_text for pat in matched_patterns):
            tracebacks.append(tb_text.strip())

    return {
        "log_path": log_path,
        "total_log_lines": len(all_lines),
        "analyzed_lines": len(lines),
        "hw_info": hw_info,
        "findings": findings,
        "unhandled_tracebacks": tracebacks
    }

def print_report(analysis):
    if not analysis:
        print(f"{BOLD}{RED}❌ Could not find ComfyUI log file.{RESET}")
        print("Searched locations:")
        for p in LOG_PATH_CANDIDATES:
            print(f"  - {p}")
        return

    hw = analysis["hw_info"]
    findings = analysis["findings"]
    tbs = analysis["unhandled_tracebacks"]

    print(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║{WHITE}   🧅 Shallot-CUI  •  ComfyUI Server Error & Diagnostics Report    {CYAN}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════╝{RESET}\n")

    print(f"{BOLD}{WHITE}📁 Log File:{RESET} {analysis['log_path']} ({analysis['total_log_lines']} total lines)")
    
    # System Info
    print(f"\n{BOLD}{BLUE}🖥️ System & Hardware Overview:{RESET}")
    if hw.get("gpu"):
        vram_gb = round(int(hw.get("vram_mb", 0)) / 1024, 1)
        print(f"  • {BOLD}GPU:{RESET} {GREEN}{hw['gpu']}{RESET} ({vram_gb} GB VRAM) | Driver: {hw.get('driver', 'N/A')}")
    if hw.get("cpu"):
        print(f"  • {BOLD}CPU:{RESET} {hw['cpu']}")
    if hw.get("python"):
        print(f"  • {BOLD}Python:{RESET} {hw['python']}")
    if hw.get("comfyui_version"):
        print(f"  • {BOLD}ComfyUI Version:{RESET} {hw['comfyui_version']}")

    # Categorized Findings
    print(f"\n{BOLD}{CYAN}────────────────────────────────────────────────────────────────────{RESET}")
    print(f"{BOLD}{MAGENTA}🔍 Log Analysis & Detected Issues:{RESET}")

    if not findings and not tbs:
        print(f"  {BOLD}{GREEN}✅ Clean Health Status:{RESET} No errors or critical warnings found in recent logs!")
    else:
        for idx, item in enumerate(findings, 1):
            rule = item["rule"]
            sev = rule["severity"]
            if sev == "ERROR":
                badge = f"{BOLD}{RED}[CRITICAL ERROR]{RESET}"
            elif sev == "WARNING":
                badge = f"{BOLD}{YELLOW}[WARNING]{RESET}"
            else:
                badge = f"{BOLD}{CYAN}[INFO / BENIGN]{RESET}"

            print(f"\n {idx}. {badge} {BOLD}{WHITE}{rule['title']}{RESET} (Occurrences: {item['count']})")
            print(f"    {DIM}Category:{RESET} {rule['category']}")
            print(f"    {BOLD}Explanation:{RESET} {rule['explanation']}")
            print(f"    {BOLD}{GREEN}Action / Fix:{RESET} {rule['action']}")

    # Unhandled Tracebacks
    if tbs:
        print(f"\n{BOLD}{RED}⚠️ Unhandled Python Tracebacks ({len(tbs)}):{RESET}")
        for idx, tb in enumerate(tbs[-3:], 1):
            print(f"\n--- Traceback #{idx} ---")
            print(f"{RED}{tb[:600]}{RESET}")
            if len(tb) > 600:
                print(f"{DIM}... (truncated){RESET}")

    # Copy-Paste Export Block for Gemini / AI
    print(f"\n{BOLD}{CYAN}════════════════════════════════════════════════════════════════════{RESET}")
    print(f"{BOLD}{YELLOW}📋 Copy-Paste Summary for Gemini / Antigravity Assistant:{RESET}")
    print(f"{DIM}Select and copy the block below if you need assistant debugging:{RESET}\n")

    markdown_export = [
        "```markdown",
        f"### ComfyUI Diagnostics Report ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
        f"- GPU: {hw.get('gpu', 'Unknown')} ({hw.get('vram_mb', 'N/A')} MB VRAM, Driver {hw.get('driver', 'N/A')})",
        f"- CPU: {hw.get('cpu', 'Unknown')}",
        f"- Python: {hw.get('python', 'Unknown')}",
        "",
        "#### Detected Log Findings:"
    ]
    if findings:
        for f in findings:
            r = f["rule"]
            markdown_export.append(f"- [{r['severity']}] **{r['title']}**: {r['explanation']}")
    else:
        markdown_export.append("- No critical log errors detected.")

    if tbs:
        markdown_export.append("\n#### Recent Unhandled Tracebacks:")
        for tb in tbs[-2:]:
            markdown_export.append(f"```python\n{tb[:500]}\n```")

    markdown_export.append("```")
    print("\n".join(markdown_export))

if __name__ == "__main__":
    log_file = find_log_file()
    if log_file:
        res = analyze_log(log_file)
        print_report(res)
    else:
        print_report(None)

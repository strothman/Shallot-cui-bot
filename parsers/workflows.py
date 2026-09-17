"""
ComfyUI Workflow Templates, AST Manipulations, LoRA Injectors, and Bertflow Pipeline for Shallot-CUI Bot.
"""

import os
import re
import json
import copy
import random
import logging

from model_architecture import (
    Architecture, 
    SubType, 
    detect_model_architecture, 
    resolve_lora_for_architecture, 
    validate_architecture_compatibility,
    get_architecture_badge
)

logger = logging.getLogger("DiscordBot.Parsers.Workflows")

_WORKFLOW_CACHE: dict[str, dict] = {}


def load_workflow_template(path: str) -> dict:
    """Loads and caches workflow JSON templates in memory, returning a deepcopy to prevent repeated disk I/O."""
    if path not in _WORKFLOW_CACHE:
        with open(path, "r", encoding="utf-8") as f:
            _WORKFLOW_CACHE[path] = json.load(f)
    return copy.deepcopy(_WORKFLOW_CACHE[path])


RE_LORA_TAG = re.compile(r'<lora:([^>:]+)(?::([^>]+))?>')
RE_SEMI_REALISM = re.compile(r'[-—–]{1,2}(?:semi-realism|sr)(?![a-zA-Z])(?:\s+|\.)?([0-9\.]+)?', re.IGNORECASE)
RE_OGARLA = re.compile(r'[-—–]{1,2}(?:ogarla|oga)(?![a-zA-Z])(?:\s+|\.)?([0-9\.]+)?', re.IGNORECASE)


def parse_loras(prompt: str, is_flux: bool = False, target_arch: str = None):
    """
    Parses <lora:name:weight> tags, and also supports shorthand --sr.XX or --srXX
    mapping to Semi-realism_illustrious, and --ogarla shorthand.
    Returns (cleaned_prompt, list of (lora_name, weight_float)).
    """
    loras = []
    effective_arch = Architecture.FLUX if is_flux else (target_arch or Architecture.SDXL)
    
    # 1. Parse --sr or --semi-realism shorthand (e.g., --sr.85, --sr85, --sr 0.85, --semi-realism .75, --semi-realism)
    sr_match = re.search(r'[-—–]{1,2}(?:semi-realism|sr)(?![a-zA-Z])(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    sr_parsed = False
    if sr_match and (sr_match.group(1) or sr_match.group(0).startswith('-') or sr_match.group(0).startswith('—') or sr_match.group(0).startswith('–')):
        try:
            val_str = sr_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.70
            
            if not is_flux:
                loras.append(("Semi-realism_illustrious.safetensors", weight))
                sr_parsed = True
                if "semi-realism" not in prompt.lower():
                    prompt = f"semi-realism, {prompt}".strip()
            prompt = re.sub(r'[-—–]{1,2}(?:semi-realism|sr)(?![a-zA-Z])(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()
        except Exception as e:
            logger.error(f"Error parsing --sr/--semi-realism shorthand: {e}")

    # 2. Parse --oga/--ogarla shorthand (e.g., --oga.70, --ogarla.90, --ogarla)
    oga_match = re.search(r'[-—–]{1,2}(?:ogarla|oga)(?![a-zA-Z])(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    oga_parsed = False
    if oga_match and (oga_match.group(1) or oga_match.group(0).startswith('-') or oga_match.group(0).startswith('—') or oga_match.group(0).startswith('–')):
        try:
            val_str = oga_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85
            
            lora_name = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, weight))
            oga_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:ogarla|oga)(?![a-zA-Z])(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Harmonize trigger keywords to match model training dataset
            if is_flux:
                if "ogarlaflux" not in prompt.lower():
                    if "ogarla" in prompt.lower():
                        prompt = re.sub(r'\bogarla\b', 'ogarlaflux', prompt, flags=re.IGNORECASE)
                    else:
                        prompt = f"ogarlaflux, {prompt}".strip()
            else:
                if "ogarla" not in prompt.lower():
                    prompt = f"ogarla, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --oga/--ogarla shorthand: {e}")

    # 2.2 Keyword Fallback: Detect 'ogarla' / 'ogarlaflux' in prompt text even if user didn't write '--'
    if not oga_parsed:
        if is_flux and re.search(r'\b(?:ogarlaflux|ogarla)\b', prompt, flags=re.IGNORECASE):
            loras.append(("ogarlaflux_epoch_5.safetensors", 0.85))
            oga_parsed = True
            if "ogarlaflux" not in prompt.lower():
                prompt = re.sub(r'\bogarla\b', 'ogarlaflux', prompt, flags=re.IGNORECASE)
        elif not is_flux and re.search(r'\bogarla\b', prompt, flags=re.IGNORECASE):
            loras.append(("ogarla_epoch_5.safetensors", 0.85))
            oga_parsed = True

    # 2.3 Parse --valerie/--val shorthand (e.g., --valerie.75, --val.85, --valerie)
    val_match = re.search(r'[-—–]{1,2}(?:valerie|val)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    val_parsed = False
    if val_match and (val_match.group(1) or val_match.group(0).startswith('-') or val_match.group(0).startswith('—') or val_match.group(0).startswith('–')):
        try:
            val_str = val_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, weight))
            val_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:valerie|val)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Silently inject trained trigger 'jen'
            if re.search(r'\b(?:valerie|val)\b', prompt, flags=re.IGNORECASE):
                prompt = re.sub(r'\b(?:valerie|val)\b', 'jen', prompt, flags=re.IGNORECASE)
            elif "jen" not in prompt.lower():
                prompt = f"jen, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --valerie shorthand: {e}")

    # 2.4 Keyword Fallback: Detect 'valerie' / 'jen' in prompt text even if user didn't write '--'
    if not val_parsed:
        if re.search(r'\bvalerie\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            val_parsed = True
            # Silently substitute pseudonym with trained trigger for ComfyUI
            prompt = re.sub(r'\bvalerie\b', 'jen', prompt, flags=re.IGNORECASE)
        elif re.search(r'\bjen\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("jen_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            val_parsed = True

    # 2.5 Parse --sully/--sul shorthand (e.g., --sully.80, --sul.85, --sully)
    sul_match = re.search(r'[-—–]{1,2}(?:sully|sul)(?:\s+|\.)?([0-9\.]+)?', prompt, flags=re.IGNORECASE)
    sul_parsed = False
    if sul_match and (sul_match.group(1) or sul_match.group(0).startswith('-') or sul_match.group(0).startswith('—') or sul_match.group(0).startswith('–')):
        try:
            val_str = sul_match.group(1)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, weight))
            sul_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:sully|sul)(?:\s+|\.)?[0-9\.]*', '', prompt, flags=re.IGNORECASE).strip()

            # Silently inject trained trigger 'susa' and target traits
            susa_traits = "black hair, thin rim glasses"
            if re.search(r'\b(?:sully|sul)\b', prompt, flags=re.IGNORECASE):
                prompt = re.sub(r'\b(?:sully|sul)\b', f'susa, {susa_traits}', prompt, flags=re.IGNORECASE)
            elif "susa" not in prompt.lower():
                prompt = f"susa, {susa_traits}, {prompt}".strip()
            elif "thin rim glasses" not in prompt.lower():
                prompt = f"{prompt}, {susa_traits}".strip()
        except Exception as e:
            logger.error(f"Error parsing --sully shorthand: {e}")

    # 2.6 Keyword Fallback: Detect 'sully' / 'susa' in prompt text even if user didn't write '--'
    if not sul_parsed:
        susa_traits = "black hair, thin rim glasses"
        if re.search(r'\bsully\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            sul_parsed = True
            prompt = re.sub(r'\bsully\b', f'susa, {susa_traits}', prompt, flags=re.IGNORECASE)
        elif re.search(r'\bsusa\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("susa_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            sul_parsed = True
            if "thin rim glasses" not in prompt.lower():
                prompt = f"{prompt}, {susa_traits}".strip()

    # 2.7 Parse --mageill / --mag shorthand with optional epoch (e.g. --mageill, --mag, --mageill3, --mag6, --mageill-e4, --mageill5.75)
    mag_match = re.search(r'[-—–]{1,2}(?:mageill|mag)(?:[-_]?e?([3-6]))?(?:(?:\s+|\.)([0-9\.]+))?', prompt, flags=re.IGNORECASE)
    mag_parsed = False
    if mag_match and (mag_match.group(1) or mag_match.group(2) or mag_match.group(0).startswith('-') or mag_match.group(0).startswith('—') or mag_match.group(0).startswith('–')):
        try:
            epoch_str = mag_match.group(1) if mag_match.group(1) else "5"
            val_str = mag_match.group(2)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_file = f"mageill_epoch_{epoch_str}.safetensors"
            lora_name = resolve_lora_for_architecture(lora_file, effective_arch)
            loras.append((lora_name, weight))
            mag_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:mageill|mag)(?:[-_]?e?[3-6])?(?:(?:\s+|\.)[0-9\.]+)?', '', prompt, flags=re.IGNORECASE).strip()

            if "mageill" not in prompt.lower():
                prompt = f"mageill, {prompt}".strip()
        except Exception as e:
            logger.error(f"Error parsing --mageill shorthand: {e}")

    # 2.8 Keyword Fallback: Detect 'mageill' in prompt text even if user didn't write '--'
    if not mag_parsed:
        if re.search(r'\bmageill\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("mageill_epoch_5.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            mag_parsed = True

    # 2.9 Parse --cheri / --che shorthand with optional epoch (e.g. --cheri, --che, --cheri4, --che6, --cheri-e4, --cheri.80, --cheri4.75)
    che_match = re.search(r'[-—–]{1,2}(?:cheri|che)(?:[-_]?e?([46]))?(?:(?:\s+|\.)([0-9\.]+))?', prompt, flags=re.IGNORECASE)
    che_parsed = False
    if che_match and (che_match.group(1) or che_match.group(2) or che_match.group(0).startswith('-') or che_match.group(0).startswith('—') or che_match.group(0).startswith('–')):
        try:
            epoch_str = che_match.group(1) if che_match.group(1) else "6"
            val_str = che_match.group(2)
            if val_str:
                if val_str.startswith('.'):
                    weight = float(val_str)
                elif val_str.isdigit():
                    w_val = float(val_str)
                    weight = w_val / 100.0 if w_val > 1.0 else w_val
                else:
                    weight = float(val_str)
            else:
                weight = 0.85

            lora_file = f"cheri_epoch_{epoch_str}.safetensors"
            lora_name = resolve_lora_for_architecture(lora_file, effective_arch)
            loras.append((lora_name, weight))
            che_parsed = True
            prompt = re.sub(r'[-—–]{1,2}(?:cheri|che)(?:[-_]?e?[46])?(?:(?:\s+|\.)[0-9\.]+)?', '', prompt, flags=re.IGNORECASE).strip()

            # Ensure trigger 'cheri' and required trait 'blonde hair'
            cheri_traits = "blonde hair"
            if re.search(r'\b(?:cheri|che)\b', prompt, flags=re.IGNORECASE):
                if "blonde hair" not in prompt.lower():
                    prompt = re.sub(r'\b(?:cheri|che)\b', f'cheri, {cheri_traits}', prompt, flags=re.IGNORECASE)
            elif "cheri" not in prompt.lower():
                prompt = f"cheri, {cheri_traits}, {prompt}".strip()
            elif "blonde hair" not in prompt.lower():
                prompt = f"{prompt}, {cheri_traits}".strip()
        except Exception as e:
            logger.error(f"Error parsing --cheri shorthand: {e}")

    # 2.10 Keyword Fallback: Detect 'cheri' in prompt text even if user didn't write '--'
    if not che_parsed:
        if re.search(r'\bcheri\b', prompt, flags=re.IGNORECASE):
            lora_name = resolve_lora_for_architecture("cheri_epoch_6.safetensors", effective_arch)
            loras.append((lora_name, 0.85))
            che_parsed = True
            if "blonde hair" not in prompt.lower():
                prompt = f"{prompt}, blonde hair".strip()

    # 2.11 Keyword Fallback: Detect 'semi-realism' in prompt text even if user didn't write '--'
    if not sr_parsed and not is_flux:
        if re.search(r'\b(?:semi[- ]realism|semirealism)\b', prompt, flags=re.IGNORECASE):
            loras.append(("Semi-realism_illustrious.safetensors", 0.70))
            sr_parsed = True

    # 3. Parse standard <lora:name:weight> tags with architecture resolution
    pattern = r'<lora:([^>:]+)(?::([^>]+))?>'
    matches = re.findall(pattern, prompt)
    for name, weight in matches:
        try:
            w = float(weight) if weight else 1.0
        except ValueError:
            w = 1.0
        clean_lora_name = name.strip()
        resolved_name = resolve_lora_for_architecture(clean_lora_name, effective_arch)
        loras.append((resolved_name, w))
        
    cleaned_prompt = re.sub(pattern, '', prompt).strip()
    return cleaned_prompt, loras

def validate_workflow_loras(
    workflow: dict, 
    checkpoint_name: str, 
    loras: list
) -> tuple[bool, list[str]]:
    """
    Validates that all parsed LoRAs and active workflow LoRAs match the architecture
    of the selected checkpoint model.
    Returns: (is_valid: bool, messages: list[str])
    """
    messages = []
    is_valid = True
    
    for lora_item in (loras or []):
        lora_name = lora_item[0] if isinstance(lora_item, (list, tuple)) else str(lora_item)
        compat, msg, suggested = validate_architecture_compatibility(checkpoint_name, lora_name)
        if not compat and not suggested:
            is_valid = False
            messages.append(f"❌ {msg}")
        elif msg:
            messages.append(f"ℹ️ {msg}")

    return is_valid, messages


def apply_loras_to_workflow(workflow, loras):
    """
    Configures pre-wired static LoraLoader nodes (Node 75 Semi-realism, Node 76 Ogarla)
    and dynamically chains any extra custom LoRAs into the workflow.
    Ensures LoRA nodes are ALWAYS present in the workflow graph, setting strengths to 0.0
    when not prompted/disabled.
    """
    is_flux = ("12" in workflow and "1" in workflow and "4" not in workflow) or ("76" in workflow and workflow["76"].get("class_type") == "LoraLoaderModelOnly")
    target_arch = Architecture.FLUX if is_flux else Architecture.SDXL
    
    from services.workflow_adapter import find_node
    node_75_match = find_node(workflow, class_type="LoraLoader", title_contains="Semi-Realism", fallback_id="75")
    node_76_match = find_node(workflow, class_type=["LoraLoader", "LoraLoaderModelOnly"], title_contains="Ogarla", fallback_id="76")

    # Reset pre-wired static LoRA nodes to 0.0 (disabled) by default
    if node_75_match and node_75_match[1].get("class_type") == "LoraLoader":
        node_75_match[1]["inputs"]["strength_model"] = 0.0
        node_75_match[1]["inputs"]["strength_clip"] = 0.0
        node_75_match[1]["inputs"]["lora_name"] = "Semi-realism_illustrious.safetensors"
        
    if node_76_match:
        if node_76_match[1].get("class_type") == "LoraLoaderModelOnly":
            node_76_match[1]["inputs"]["strength_model"] = 0.0
            node_76_match[1]["inputs"]["lora_name"] = "ogarlaflux_epoch_5.safetensors"
        elif node_76_match[1].get("class_type") == "LoraLoader":
            node_76_match[1]["inputs"]["strength_model"] = 0.0
            node_76_match[1]["inputs"]["strength_clip"] = 0.0
            node_76_match[1]["inputs"]["lora_name"] = "ogarla_epoch_5.safetensors"

    extra_loras = []
    
    if loras:
        for lora_name, weight in loras:
            resolved_lora = resolve_lora_for_architecture(lora_name, target_arch)
            lname_clean = resolved_lora.lower()
            if "semi-realism" in lname_clean and node_75_match:
                node_75_match[1]["inputs"]["strength_model"] = weight
                node_75_match[1]["inputs"]["strength_clip"] = weight
                node_75_match[1]["inputs"]["lora_name"] = "Semi-realism_illustrious.safetensors"
            elif "ogarla" in lname_clean and node_76_match:
                target_file = "ogarlaflux_epoch_5.safetensors" if is_flux else "ogarla_epoch_5.safetensors"
                node_76_match[1]["inputs"]["strength_model"] = weight
                if "strength_clip" in node_76_match[1]["inputs"]:
                    node_76_match[1]["inputs"]["strength_clip"] = weight
                node_76_match[1]["inputs"]["lora_name"] = target_file
            else:
                candidate_fname = resolved_lora if resolved_lora.endswith((".safetensors", ".ckpt")) else f"{resolved_lora}.safetensors"
                try:
                    from characters import get_character
                    char_prof = get_character(resolved_lora)
                    if char_prof:
                        if is_flux and not char_prof.lora_flux:
                            logger.info(f"Skipping character '{char_prof.display_name}' on Flux (no Flux LoRA available).")
                            continue
                        elif not is_flux and not char_prof.lora_sdxl:
                            logger.info(f"Skipping character '{char_prof.display_name}' on SDXL (no SDXL LoRA available).")
                            continue
                except Exception:
                    pass

                _, lora_arch, _ = detect_model_architecture(candidate_fname)
                if lora_arch != Architecture.UNKNOWN and lora_arch != target_arch:
                    logger.warning(
                        f"apply_loras_to_workflow: Skipping incompatible LoRA '{candidate_fname}' ({lora_arch.upper()}) "
                        f"for workflow architecture {target_arch.upper()}."
                    )
                    continue
                extra_loras.append((candidate_fname, weight))

    # 2. If workflow has no pre-wired Node 75/76 or there are extra custom LoRAs, dynamically chain them
    model_loader_match = find_node(workflow, class_type="UnetLoaderGGUF", fallback_id="1") if is_flux else find_node(workflow, class_type="CheckpointLoaderSimple", fallback_id="4")
    clip_loader_match = find_node(workflow, class_type="DualCLIPLoader", fallback_id="12") if is_flux else find_node(workflow, class_type="CheckpointLoaderSimple", fallback_id="4")

    base_model_id = model_loader_match[0] if model_loader_match else ("1" if is_flux else "4")
    base_clip_id = clip_loader_match[0] if clip_loader_match else ("12" if is_flux else "4")

    current_model_source = [node_76_match[0], 0] if node_76_match else [base_model_id, 0]
    current_clip_source = [node_76_match[0], 1] if (node_76_match and not is_flux) else ([base_clip_id, 0] if is_flux else [base_clip_id, 1])

    start_node_id = 100
    for idx, (lora_name, weight) in enumerate(extra_loras):
        node_id = str(start_node_id + idx)
        if not lora_name.endswith(".safetensors") and not lora_name.endswith(".ckpt"):
            lora_name += ".safetensors"
            
        strength_clip = weight
        
        if is_flux:
            workflow[node_id] = {
                "inputs": {
                    "model": current_model_source,
                    "lora_name": lora_name,
                    "strength_model": weight
                },
                "class_type": "LoraLoaderModelOnly"
            }
            current_model_source = [node_id, 0]
        else:
            workflow[node_id] = {
                "inputs": {
                    "model": current_model_source,
                    "clip": current_clip_source,
                    "lora_name": lora_name,
                    "strength_model": weight,
                    "strength_clip": strength_clip
                },
                "class_type": "LoraLoader"
            }
            current_model_source = [node_id, 0]
            current_clip_source = [node_id, 1]

    # 3. Connect IPAdapter if present
    ipadapter_out_node_id = None
    for node_id, node in workflow.items():
        if node.get("class_type") == "IPAdapterUnifiedLoader":
            node["inputs"]["model"] = current_model_source
        if node.get("class_type") in ["IPAdapter", "IPAdapterAdvanced"]:
            ipadapter_out_node_id = node_id

    model_after_ipadapter = [ipadapter_out_node_id, 0] if ipadapter_out_node_id else current_model_source

    # 4. Connect FreeU if present
    freeu_node_id = None
    for node_id, node in workflow.items():
        if node.get("class_type") in ["FreeU", "FreeU_V2"]:
            node["inputs"]["model"] = model_after_ipadapter
            freeu_node_id = node_id
            break

    final_sampler_model_source = [freeu_node_id, 0] if freeu_node_id else model_after_ipadapter

    # 5. Re-route KSamplers and CLIPTextEncodes
    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSampler (Efficient)"]:
            node["inputs"]["model"] = final_sampler_model_source
        elif node.get("class_type") == "CLIPTextEncode" and not is_flux:
            node["inputs"]["clip"] = current_clip_source
            
    return workflow


def apply_face_detailer_to_workflow(
    workflow: dict,
    seed: int = 123456789,
    cfg: float = 4.0,
    sampler_name: str = "dpmpp_2m",
    scheduler: str = "karras",
    steps: int = 20,
    denoise: float = 0.40,
    guide_size: int = 512,
    max_size: int = 768,
    detector_model: str = "bbox/face_yolov8m.pt"
) -> dict:
    """
    Injects ComfyUI-Impact-Pack UltralyticsDetectorProvider and FaceDetailer nodes
    into an SDXL workflow, routing decoded images through face enhancement before display.
    Optimized for 8GB VRAM cards with guide_size=512 and moderate steps/denoise.
    """
    vae_decode_node_id = None
    save_or_preview_node_id = None
    for node_id, node in workflow.items():
        ctype = node.get("class_type")
        if ctype == "VAEDecode":
            vae_decode_node_id = node_id
        elif ctype in ["PreviewImage", "SaveImage"]:
            save_or_preview_node_id = node_id

    if not vae_decode_node_id:
        return workflow

    model_source = None
    clip_source = None
    vae_source = None

    for node_id, node in workflow.items():
        if node.get("class_type") == "CheckpointLoaderSimple":
            vae_source = [node_id, 2]
            if not model_source:
                model_source = [node_id, 0]
            if not clip_source:
                clip_source = [node_id, 1]

    for node_id, node in workflow.items():
        if node.get("class_type") in ["KSampler", "KSampler (Efficient)"]:
            model_source = node["inputs"].get("model", model_source)
            break

    if "6" in workflow and "clip" in workflow["6"].get("inputs", {}):
        clip_source = workflow["6"]["inputs"]["clip"]

    detector_node_id = "80"
    workflow[detector_node_id] = {
        "inputs": {
            "model_name": detector_model
        },
        "class_type": "UltralyticsDetectorProvider",
        "_meta": {
            "title": "Ultralytics Detector Provider (Face)"
        }
    }

    detailer_node_id = "85"
    workflow[detailer_node_id] = {
        "inputs": {
            "image": [vae_decode_node_id, 0],
            "model": model_source,
            "clip": clip_source,
            "vae": vae_source,
            "guide_size": guide_size,
            "guide_size_for": True,
            "max_size": max_size,
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": sampler_name,
            "scheduler": scheduler,
            "denoise": denoise,
            "feather": 5,
            "noise_mask": True,
            "force_inpaint": True,
            "bbox_threshold": 0.5,
            "bbox_dilation": 10,
            "bbox_crop_factor": 3.0,
            "sam_detection_hint": "center-1",
            "sam_dilation": 0,
            "sam_threshold": 0.93,
            "sam_bbox_expansion": 0,
            "sam_mask_hint_threshold": 0.7,
            "sam_mask_hint_use_negative": "False",
            "drop_size": 10,
            "bbox_detector": [detector_node_id, 0],
            "wildcard": "",
            "cycle": 1,
            "positive": ["6", 0] if "6" in workflow else None,
            "negative": ["7", 0] if "7" in workflow else None
        },
        "class_type": "FaceDetailer",
        "_meta": {
            "title": "Face Detailer (Impact Pack - 8GB Safe)"
        }
    }

    if save_or_preview_node_id and save_or_preview_node_id in workflow:
        workflow[save_or_preview_node_id]["inputs"]["images"] = [detailer_node_id, 0]

    return workflow


def apply_ipadapter_to_workflow(workflow: dict, image_name: str, weight: float = 0.20, preset: str = None, node_prefix: str = "cref"):
    """
    Dynamically injects an IP-Adapter node chain into a workflow dict for a reference image (e.g. --cref or --sref).
    """
    is_flux = ("12" in workflow and "1" in workflow and "4" not in workflow) or ("76" in workflow and workflow["76"].get("class_type") == "LoraLoaderModelOnly")
    if is_flux:
        return workflow

    ksampler_node_id = None
    if "3" in workflow and workflow["3"].get("class_type") in ["KSampler", "KSampler (Efficient)", "KSamplerAdvanced"]:
        ksampler_node_id = "3"
    else:
        for nid, node in workflow.items():
            if node.get("class_type") in ["KSampler", "KSampler (Efficient)", "KSamplerAdvanced"]:
                ksampler_node_id = nid
                break

    if not ksampler_node_id or "inputs" not in workflow[ksampler_node_id]:
        return workflow

    current_model = workflow[ksampler_node_id]["inputs"].get("model")
    if not current_model:
        if "76" in workflow:
            current_model = ["76", 0]
        elif "4" in workflow:
            current_model = ["4", 0]
        else:
            for nid, node in workflow.items():
                if node.get("class_type") in ["CheckpointLoaderSimple", "UNETLoader"]:
                    current_model = [nid, 0]
                    break

    if not current_model:
        return workflow

    loader_id = f"{node_prefix}_loader_20"
    load_img_id = f"{node_prefix}_img_21"
    ip_node_id = f"{node_prefix}_ip_23"

    actual_preset = preset or "PLUS (high strength)"

    workflow[loader_id] = {
        "inputs": {
            "model": current_model,
            "preset": actual_preset
        },
        "class_type": "IPAdapterUnifiedLoader",
        "_meta": {"title": f"IPAdapter Loader ({node_prefix.upper()})"}
    }

    workflow[load_img_id] = {
        "inputs": {
            "image": image_name,
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": f"Load Image ({node_prefix.upper()})"}
    }

    workflow[ip_node_id] = {
        "inputs": {
            "model": [loader_id, 0],
            "ipadapter": [loader_id, 1],
            "image": [load_img_id, 0],
            "weight": weight,
            "weight_type": "ease in-out" if node_prefix == "cref" else "linear",
            "combine_embeds": "concat",
            "start_at": 0.0,
            "end_at": 1.0,
            "embeds_scaling": "K+V"
        },
        "class_type": "IPAdapterAdvanced",
        "_meta": {"title": f"IPAdapter ({node_prefix.upper()})"}
    }

    workflow[ksampler_node_id]["inputs"]["model"] = [ip_node_id, 0]
    return workflow


def get_bertflow_unet_model(preferred_model: str = None) -> str:
    """
    Finds the available Krea 2 UNET model in ComfyUI models/unet or models/diffusion_models directory.
    Prefers museByStableYogi_v35Int8Extended.safetensors, then pornmasterKrea2_v1FP8.safetensors.
    """
    search_dirs = [
        r"C:\ComfyUI\ComfyUI\models\unet",
        r"C:\ComfyUI\ComfyUI\models\diffusion_models",
    ]
    if preferred_model:
        for d in search_dirs:
            if os.path.exists(os.path.join(d, preferred_model)):
                return preferred_model

    muse_candidates = [
        "museByStableYogi_v35Int8Extended.safetensors",
        "museByStableYogi_v30TurboInt8.safetensors",
        "museByStableYogi_v10TurboFP8.safetensors",
    ]
    for m in muse_candidates:
        for d in search_dirs:
            if os.path.exists(os.path.join(d, m)):
                return m

    pm_candidates = [
        "pornmasterKrea2_v1FP8.safetensors",
    ]
    for pm in pm_candidates:
        for d in search_dirs:
            if os.path.exists(os.path.join(d, pm)):
                return pm

    for d in search_dirs:
        if os.path.isdir(d):
            for fname in os.listdir(d):
                fl = fname.lower()
                if ("krea2" in fl or "muse" in fl) and (fl.endswith(".safetensors") or fl.endswith(".gguf")):
                    return fname

    return "museByStableYogi_v35Int8Extended.safetensors"


def prepare_bertflow_workflow(
    prompt: str,
    width: int = 1224,
    height: int = 1224,
    seed: int = None,
    steps: int = 8,
    unet_model: str = None,
    wetness_strength: float = -2.0,
    init_image: str = None,
    comp_strength: str = "off",
    filename_prefix: str = None,
    character: str = None,
    character_strength: float = None,
    celebrity: str = None,
    outpaint_pad: tuple[int, int, int, int] = None
) -> dict:
    """
    Loads workflows/bertflow.json and populates prompt, seed, dimensions, steps, model, wetness strength,
    optional character LoRA, and optional favorite celebrity prompt injection.
    Optionally injects direct compositional reference (img2img VAE latent) when init_image and comp_strength are specified.
    """
    wf = load_workflow_template("workflows/bertflow.json")

    final_seed = seed if seed is not None else random.randint(1, 1125899906842624)
    model_name = unet_model or get_bertflow_unet_model()

    # Parse wetness / skin finish flags from prompt if present
    if prompt:
        wet_val_match = re.search(r'--(?:wetness|wet)\s*([+-]?\d+(?:\.\d+)?)\b', prompt, re.IGNORECASE)
        if wet_val_match:
            try:
                wetness_strength = float(wet_val_match.group(1))
            except (ValueError, TypeError):
                pass
            prompt = re.sub(r'--(?:wetness|wet)\s*[+-]?\d+(?:\.\d+)?\b', '', prompt, flags=re.IGNORECASE).strip()
        elif re.search(r'--dry\b', prompt, re.IGNORECASE):
            wetness_strength = -3.0
            prompt = re.sub(r'--dry\b', '', prompt, flags=re.IGNORECASE).strip()
        elif re.search(r'--matte\b', prompt, re.IGNORECASE):
            wetness_strength = -2.5
            prompt = re.sub(r'--matte\b', '', prompt, flags=re.IGNORECASE).strip()
        elif re.search(r'--dewy\b', prompt, re.IGNORECASE):
            wetness_strength = -0.5
            prompt = re.sub(r'--dewy\b', '', prompt, flags=re.IGNORECASE).strip()

    # Parse character flag from prompt if present (e.g. --ogarla.85, --oga)
    active_char = character
    char_weight = character_strength
    if prompt:
        oga_match = re.search(r'--(ogarla|oga)(?:\.(\d+))?\b', prompt, re.IGNORECASE)
        if oga_match:
            active_char = "ogarla"
            if char_weight is None and oga_match.group(2):
                char_weight = float(oga_match.group(2)) / 100.0 if len(oga_match.group(2)) == 2 else float(f"0.{oga_match.group(2)}")
            prompt = re.sub(r'--(ogarla|oga)(?:\.\d+)?\b', '', prompt, flags=re.IGNORECASE).strip()

    # Parse celebrity flag from prompt if present (e.g. --audrey, --zendaya, --celeb margot)
    active_celeb = celebrity
    if prompt:
        try:
            from celebrities import FAVORITE_CELEBRITIES
            celeb_match = re.search(r'--celeb(?:rity)?\s+([A-Za-z0-9_\-]+)\b', prompt, re.IGNORECASE)
            if celeb_match:
                active_celeb = celeb_match.group(1)
                prompt = re.sub(r'--celeb(?:rity)?\s+[A-Za-z0-9_\-]+\b', '', prompt, flags=re.IGNORECASE).strip()
            else:
                for cid, cprof in FAVORITE_CELEBRITIES.items():
                    flag_names = [cid] + list(cprof.shorthands)
                    pattern = rf'--(?:{"|".join(re.escape(s) for s in flag_names)})\b'
                    if re.search(pattern, prompt, re.IGNORECASE):
                        active_celeb = cid
                        prompt = re.sub(pattern, '', prompt, flags=re.IGNORECASE).strip()
                        break
        except Exception:
            pass

    # Determine character LoRA file, weight, and trigger word
    char_lora_file = None
    char_trigger = None
    if active_char and str(active_char).lower() not in ["none", "nochar", "off", "false"]:
        c_str = str(active_char).lower()
        if "ogarla" in c_str or "oga" in c_str:
            char_lora_file = "Krea2\\ogarla_krea2.safetensors"
            char_trigger = "ogarla"
            if char_weight is None:
                dot_match = re.search(r'\.(\d+)', c_str)
                if dot_match:
                    val = dot_match.group(1)
                    char_weight = float(val) / 100.0 if len(val) == 2 else float(f"0.{val}")
                else:
                    char_weight = 0.70

    # Inject trigger word into prompt if needed
    cleaned_prompt = prompt or ""
    if char_trigger:
        if not re.search(rf'\b{re.escape(char_trigger)}\b', cleaned_prompt, re.IGNORECASE):
            cleaned_prompt = f"{char_trigger}, {cleaned_prompt}".strip()
        cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).lstrip(":,.- ").strip()

    # Inject favorite celebrity preset into prompt if specified
    if active_celeb and str(active_celeb).lower() not in ["none", "noceleb", "off", "false"]:
        try:
            from celebrities import inject_celebrity_in_prompt
            cleaned_prompt = inject_celebrity_in_prompt(cleaned_prompt, active_celeb)
            cleaned_prompt = re.sub(r"\s+", " ", cleaned_prompt).lstrip(":,.- ").strip()
        except Exception:
            pass

    # Anatomical artifact protection for Krea 2 Flow-Matching
    male_anatomy_pattern = r'\b(penis|cock|shaft|dick|erection|phallus)\b'
    if re.search(male_anatomy_pattern, cleaned_prompt, re.IGNORECASE):
        clean_anatomy_anchor = "smooth natural shaft, clean coronal sulcus, realistic anatomy, circumcised, smooth skin"
        if not any(k in cleaned_prompt.lower() for k in ["smooth natural shaft", "clean coronal sulcus"]):
            cleaned_prompt = f"{cleaned_prompt}, {clean_anatomy_anchor}"

    from services.workflow_adapter import (
        set_workflow_prompt,
        set_workflow_seed,
        set_workflow_dimensions,
        set_workflow_sampler_params,
        set_workflow_checkpoint,
        set_workflow_filename_prefix,
        find_node
    )

    set_workflow_prompt(wf, positive=cleaned_prompt)
    set_workflow_seed(wf, final_seed)
    set_workflow_dimensions(wf, width=width, height=height)
    set_workflow_sampler_params(wf, steps=steps)
    if "599" in wf and "inputs" in wf["599"]:
        wf["599"]["inputs"]["end_at_step"] = steps
        if "649" in wf:
            wf["599"]["inputs"]["noise_seed"] = ["649", 0]

    set_workflow_checkpoint(wf, unet_name=model_name)

    # rgthree / LoRA Stack configuration (Node 822)
    lora_stack_node = find_node(wf, class_type=["lorastack", "powerloraloader", "loraloader"], title_contains="lora", fallback_id="822")
    if lora_stack_node and "inputs" in lora_stack_node[1]:
        inputs = lora_stack_node[1]["inputs"]
        if "lora_1" in inputs:
            inputs["lora_1"]["strength"] = float(wetness_strength)
            inputs["lora_1"]["on"] = (wetness_strength != 0.0)
        if char_lora_file:
            inputs["lora_2"] = {
                "on": True,
                "lora": char_lora_file,
                "strength": float(char_weight if char_weight is not None else 0.85)
            }
        elif "lora_2" in inputs:
            inputs["lora_2"]["on"] = False

    try:
        from image_utils import get_dated_save_prefix
        chosen_prefix = filename_prefix or f"{get_dated_save_prefix('bertflow')}bertflow_seed{final_seed}"
    except Exception:
        chosen_prefix = filename_prefix or f"Discord Bot/bertflow/bertflow_seed{final_seed}"

    set_workflow_filename_prefix(wf, chosen_prefix)

    # 1. Native Directional Outpainting with ImagePadForOutpaint + VAEEncodeForInpaint
    if init_image and outpaint_pad:
        left, top, right, bottom = outpaint_pad
        wf["900"] = {
            "inputs": {
                "image": init_image,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Original Image for Outpaint"}
        }
        wf["901"] = {
            "inputs": {
                "image": ["900", 0],
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "feathering": 72
            },
            "class_type": "ImagePadForOutpaint",
            "_meta": {"title": "Pad Image for Outpainting"}
        }
        wf["902"] = {
            "inputs": {
                "pixels": ["901", 0],
                "vae": ["757", 0],
                "mask": ["901", 1],
                "grow_mask_by": 6
            },
            "class_type": "VAEEncodeForInpaint",
            "_meta": {"title": "VAE Encode (for Inpainting)"}
        }
        if "599" in wf and "inputs" in wf["599"]:
            wf["599"]["inputs"]["latent_image"] = ["902", 0]
            wf["599"]["inputs"]["start_at_step"] = 0
            wf["599"]["inputs"]["end_at_step"] = steps
            wf["599"]["inputs"]["add_noise"] = "enable"
            wf["599"]["inputs"]["return_with_leftover_noise"] = "disable"

    # 2. Optional direct compositional reference (img2img / VAE latent injection)
    elif init_image and comp_strength and str(comp_strength).lower() not in ["off", "none", "style", "false"]:
        wf["900"] = {
            "inputs": {
                "image": init_image,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "Load Init Image for Composition"}
        }
        wf["901"] = {
            "inputs": {
                "image": ["900", 0],
                "upscale_method": "bilinear",
                "width": width,
                "height": height,
                "crop": "center"
            },
            "class_type": "ImageScale",
            "_meta": {"title": "Scale Init Image"}
        }
        wf["902"] = {
            "inputs": {
                "pixels": ["901", 0],
                "vae": ["757", 0]
            },
            "class_type": "VAEEncode",
            "_meta": {"title": "VAE Encode Init Latent"}
        }

        c_str = str(comp_strength).lower()
        if "strong" in c_str or "50" in c_str:
            start_step = max(1, min(steps - 1, round(steps * 0.50)))
        elif "subtle" in c_str or "85" in c_str:
            start_step = max(1, min(steps - 1, round(steps * 0.15)))
        else:  # default medium (~70% denoise)
            start_step = max(1, min(steps - 1, round(steps * 0.30)))

        if "599" in wf and "inputs" in wf["599"]:
            wf["599"]["inputs"]["latent_image"] = ["902", 0]
            wf["599"]["inputs"]["start_at_step"] = start_step
            wf["599"]["inputs"]["add_noise"] = "enable"
            wf["599"]["inputs"]["return_with_leftover_noise"] = "disable"

    return wf

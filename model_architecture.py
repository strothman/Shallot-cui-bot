"""
Model & LoRA Architecture Detection, Classification, and Compatibility Guard.
Provides fast safetensors header inspection, architecture resolution, and cross-model validation.
"""

import os
import json
import struct
import logging
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("DiscordBot.ModelArch")

# =========================================================================
# Architecture & Sub-Type Taxonomy
# =========================================================================

class Architecture:
    SDXL = "sdxl"
    FLUX = "flux"
    SD15 = "sd15"
    SD35 = "sd35"
    WAN = "wan"
    LTX = "ltx"
    HUNYUAN = "hunyuan"
    UNKNOWN = "unknown"

    ALL = [SDXL, FLUX, SD15, SD35, WAN, LTX, HUNYUAN, UNKNOWN]

class SubType:
    STANDARD = "standard"
    ILLUSTRIOUS = "illustrious"
    PONY = "pony"
    DANBOORU = "danbooru"
    REALISTIC = "realistic"
    GGUF = "gguf"
    HIGH_NOISE = "high_noise"
    LOW_NOISE = "low_noise"
    GENERAL = "general"

class ModelType:
    CHECKPOINT = "checkpoint"
    LORA = "lora"
    VAE = "vae"
    CLIP = "clip"
    UNET = "unet"
    UNKNOWN = "unknown"

# Display badges for UI / Discord embeds
ARCH_BADGES = {
    Architecture.SDXL: "🎨 [SDXL]",
    Architecture.FLUX: "⚡ [FLUX]",
    Architecture.SD15: "🖌️ [SD1.5]",
    Architecture.SD35: "🌟 [SD3.5]",
    Architecture.WAN: "🎬 [WAN]",
    Architecture.LTX: "🎥 [LTX]",
    Architecture.HUNYUAN: "🐉 [HUNYUAN]",
    Architecture.UNKNOWN: "❓ [MODEL]"
}

# =========================================================================
# Safetensors Header Inspection (Zero-Weight Loading)
# =========================================================================

def read_safetensors_header(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Reads only the JSON header of a .safetensors file without loading model weights.
    Returns the decoded header dict containing tensor metadata and '__metadata__'.
    """
    if not os.path.isfile(filepath):
        return None
    try:
        with open(filepath, "rb") as f:
            header_bytes = f.read(8)
            if len(header_bytes) < 8:
                return None
            header_len = struct.unpack("<Q", header_bytes)[0]
            # Safety sanity check: header shouldn't exceed 100MB
            if header_len <= 0 or header_len > 100 * 1024 * 1024:
                return None
            json_bytes = f.read(header_len)
            header_str = json_bytes.decode("utf-8", errors="ignore")
            return json.loads(header_str)
    except Exception as e:
        logger.debug(f"Could not read safetensors header from {filepath}: {e}")
        return None

# =========================================================================
# Architecture & SubType Detection Engine
# =========================================================================

def detect_model_architecture(filename_or_path: str, file_path: Optional[str] = None) -> Tuple[str, str, str]:
    """
    Detects (model_type, base_architecture, sub_type) from file metadata or filename heuristics.
    
    Returns:
        (model_type, base_architecture, sub_type)
    """
    target_path = file_path or filename_or_path
    clean_name = os.path.basename(filename_or_path).lower()
    
    # 1. First, check if file exists and inspect safetensors header
    if os.path.isfile(target_path) and clean_name.endswith(".safetensors"):
        header = read_safetensors_header(target_path)
        if header:
            meta = header.get("__metadata__", {})
            ss_base = str(meta.get("ss_base_model_version", "")).lower()
            modelspec_arch = str(meta.get("modelspec.architecture", "")).lower()
            
            # Check modelspec / kohya metadata
            if "flux" in ss_base or "flux" in modelspec_arch:
                return (ModelType.LORA if is_lora_keys(header) else ModelType.CHECKPOINT, Architecture.FLUX, SubType.STANDARD)
            if "sdxl" in ss_base or "sdxl" in modelspec_arch or "stable-diffusion-xl" in ss_base:
                subtype = detect_sdxl_subtype(clean_name, meta)
                return (ModelType.LORA if is_lora_keys(header) else ModelType.CHECKPOINT, Architecture.SDXL, subtype)
            if "sd_v1" in ss_base or "sd1" in ss_base or "v1-5" in ss_base:
                return (ModelType.LORA if is_lora_keys(header) else ModelType.CHECKPOINT, Architecture.SD15, SubType.STANDARD)
            if "sd3" in ss_base:
                return (ModelType.LORA if is_lora_keys(header) else ModelType.CHECKPOINT, Architecture.SD35, SubType.STANDARD)

            # Check key structure
            keys = list(header.keys())
            if any("double_blocks" in k for k in keys):
                return (ModelType.LORA if is_lora_keys(header) else ModelType.UNET, Architecture.FLUX, SubType.STANDARD)
            if any("lora_unet_down_blocks" in k or "lora_te1" in k or "lora_te2" in k for k in keys):
                subtype = detect_sdxl_subtype(clean_name, meta)
                return (ModelType.LORA, Architecture.SDXL, subtype)
            if any("wan" in k for k in keys):
                return (ModelType.LORA if is_lora_keys(header) else ModelType.CHECKPOINT, Architecture.WAN, SubType.STANDARD)

    # 2. Heuristic Pattern Detection based on Filename & Dialects
    model_type = ModelType.LORA if ("lora" in clean_name or "epoch" in clean_name or "sr" in clean_name or "semi-realism" in clean_name) else ModelType.CHECKPOINT

    # Video Models (Wan, LTX, Hunyuan)
    if "wan" in clean_name or "wan21" in clean_name or "wan22" in clean_name or "i2v" in clean_name:
        subtype = SubType.HIGH_NOISE if "high" in clean_name else (SubType.LOW_NOISE if "low" in clean_name else SubType.STANDARD)
        return (model_type, Architecture.WAN, subtype)
    if "ltx" in clean_name:
        return (model_type, Architecture.LTX, SubType.STANDARD)
    if "hunyuan" in clean_name:
        return (model_type, Architecture.HUNYUAN, SubType.STANDARD)

    # Flux Models
    if "flux" in clean_name or "ogarlaflux" in clean_name:
        subtype = SubType.GGUF if clean_name.endswith(".gguf") else SubType.STANDARD
        return (model_type, Architecture.FLUX, subtype)

    # SD 3.5 Models
    if "sd3" in clean_name or "sd35" in clean_name:
        return (model_type, Architecture.SD35, SubType.STANDARD)

    # SD 1.5 Models
    if "sd15" in clean_name or "v1-5" in clean_name or "v15" in clean_name or "realisticvision" in clean_name:
        return (model_type, Architecture.SD15, SubType.STANDARD)

    # SDXL Checkpoints and LoRAs (Illustrious, Pony, Realistic, Standard)
    if any(k in clean_name for k in ["illustrious", "wai", "hyphoria", "semi-realism", "novafurry"]):
        return (model_type, Architecture.SDXL, SubType.ILLUSTRIOUS)
    if "pony" in clean_name:
        return (model_type, Architecture.SDXL, SubType.PONY)
    if any(k in clean_name for k in ["realvis", "juggernaut", "copax", "ultrarealistic", "biglust", "lustify", "xl"]):
        return (model_type, Architecture.SDXL, SubType.REALISTIC)

    # Default fallback to SDXL for repository context if safetensors
    if clean_name.endswith(".safetensors"):
        return (model_type, Architecture.SDXL, SubType.STANDARD)

    return (ModelType.CHECKPOINT, Architecture.UNKNOWN, SubType.GENERAL)

def is_lora_keys(header: Dict[str, Any]) -> bool:
    """Detects whether keys contain standard LoRA weight keys."""
    keys = list(header.keys())
    return any(k.startswith("lora_") or "lora_up" in k or "lora_down" in k or "lora_A" in k for k in keys)

def detect_sdxl_subtype(clean_name: str, meta: Dict[str, Any]) -> str:
    """Helper to detect SDXL flavor."""
    text = (clean_name + " " + json.dumps(meta)).lower()
    if any(k in text for k in ["illustrious", "wai", "hyphoria", "danbooru", "semi-realism"]):
        return SubType.ILLUSTRIOUS
    if "pony" in text:
        return SubType.PONY
    if any(k in text for k in ["realvis", "juggernaut", "photoreal", "realistic"]):
        return SubType.REALISTIC
    return SubType.STANDARD

# =========================================================================
# LoRA Variant Auto-Router & Compatibility Validator
# =========================================================================

# Known LoRA cross-architecture family registry
LORA_FAMILY_VARIANTS = {
    "ogarla": {
        Architecture.SDXL: "ogarla_epoch_5.safetensors",
        f"{Architecture.SDXL}:{SubType.PONY}": "ogarlapony_epoch_6.safetensors",
        Architecture.FLUX: "ogarlaflux_epoch_5.safetensors",
    },
    "semi-realism": {
        Architecture.SDXL: "Semi-realism_illustrious.safetensors",
        f"{Architecture.SDXL}:{SubType.ILLUSTRIOUS}": "Semi-realism_illustrious.safetensors",
    }
}

def resolve_lora_for_architecture(lora_name: str, target_arch: str, target_subtype: str = SubType.STANDARD) -> str:
    """
    Given a LoRA name and the target engine architecture, finds the best matching variant.
    Returns the resolved filename, or the original filename if no variant exists.
    """
    clean = os.path.basename(lora_name).lower()
    
    # Check Ogarla family
    if "ogarla" in clean:
        if target_arch == Architecture.FLUX:
            return "ogarlaflux_epoch_5.safetensors"
        elif target_arch == Architecture.SDXL:
            if target_subtype == SubType.PONY or "pony" in str(target_subtype).lower():
                return "ogarlapony_epoch_6.safetensors"
            return "ogarla_epoch_5.safetensors"

    # Check Semi-realism family
    if "semi-realism" in clean or "sr" == clean.split(".")[0]:
        if target_arch == Architecture.SDXL:
            return "Semi-realism_illustrious.safetensors"
            
    return lora_name

def validate_architecture_compatibility(
    checkpoint_name: str, 
    lora_name: str
) -> Tuple[bool, str, Optional[str]]:
    """
    Validates if a checkpoint and a LoRA share compatible base architectures.
    
    Returns:
        (is_compatible: bool, message: str, suggested_variant: Optional[str])
    """
    _, ckpt_arch, ckpt_subtype = detect_model_architecture(checkpoint_name)
    _, lora_arch, lora_subtype = detect_model_architecture(lora_name)

    # 1. Direct Base Architecture Match
    if ckpt_arch == lora_arch and ckpt_arch != Architecture.UNKNOWN:
        # Check sub-type soft warnings
        if ckpt_arch == Architecture.SDXL:
            if ckpt_subtype == SubType.PONY and lora_subtype == SubType.ILLUSTRIOUS:
                return (True, "Compatible base SDXL, but mixing Pony checkpoint with Illustrious LoRA may affect tags.", None)
            if ckpt_subtype == SubType.REALISTIC and lora_subtype == SubType.PONY:
                return (True, "Compatible base SDXL, but Pony LoRA with realistic checkpoint may require explicit prompt tuning.", None)
        return (True, f"Perfect match: Both model and LoRA use {ckpt_arch.upper()}.", None)

    # 2. Incompatible Base Architecture - Check if an auto-routed replacement exists
    suggested = resolve_lora_for_architecture(lora_name, ckpt_arch, ckpt_subtype)
    if suggested != lora_name:
        _, sug_arch, _ = detect_model_architecture(suggested)
        if sug_arch == ckpt_arch:
            return (
                False, 
                f"LoRA '{lora_name}' is for {lora_arch.upper()}, but checkpoint is {ckpt_arch.upper()}. "
                f"Auto-routed to compatible variant '{suggested}'.",
                suggested
            )

    # 3. Incompatible and No Replacement
    return (
        False, 
        f"Incompatible architecture: Checkpoint '{checkpoint_name}' is {ckpt_arch.upper()} but LoRA '{lora_name}' is {lora_arch.upper()}.",
        None
    )

def get_architecture_badge(arch: str) -> str:
    """Returns formatted badge for a given architecture."""
    return ARCH_BADGES.get(arch.lower(), "📦 [MODEL]")


def scan_and_register_comfyui_models(comfyui_root: str = r"C:\ComfyUI\ComfyUI") -> dict:
    """
    Scans local ComfyUI model directories, extracts architecture metadata from Safetensors headers,
    and automatically registers discovered models into SQLite model_registry.
    """
    import db

    models_dir = os.path.join(comfyui_root, "models")
    if not os.path.exists(models_dir):
        return {"success": False, "error": f"Directory not found: {models_dir}", "scanned": 0, "registered": 0}

    scan_targets = [
        ("checkpoints", ModelType.CHECKPOINT),
        ("loras", ModelType.LORA),
        ("diffusion_models", ModelType.UNET),
        ("unet", ModelType.UNET),
    ]

    scanned_count = 0
    registered_count = 0
    summary = {
        "success": True,
        "checkpoints": 0,
        "loras": 0,
        "unets": 0,
        "total_registered": 0
    }

    for sub_dir, default_mtype in scan_targets:
        target_path = os.path.join(models_dir, sub_dir)
        if not os.path.exists(target_path):
            continue

        for root, _, files in os.walk(target_path):
            for file in files:
                if not file.lower().endswith((".safetensors", ".ckpt", ".pt", ".bin")):
                    continue

                scanned_count += 1
                full_path = os.path.join(root, file)
                # Relative path from target directory for nested models
                rel_filename = os.path.relpath(full_path, target_path).replace("\\", "/")

                mtype, arch, subtype = detect_model_architecture(rel_filename, full_path)
                mtype = mtype if mtype != ModelType.UNKNOWN else default_mtype

                # Friendly display name
                clean_name = os.path.splitext(os.path.basename(file))[0].replace("_", " ")

                success = db.upsert_model_registry(
                    filename=rel_filename,
                    model_type=mtype,
                    base_architecture=arch,
                    sub_type=subtype,
                    display_name=clean_name,
                    metadata={"local_path": full_path, "auto_scanned": True}
                )

                if success:
                    registered_count += 1
                    if mtype == ModelType.CHECKPOINT:
                        summary["checkpoints"] += 1
                    elif mtype == ModelType.LORA:
                        summary["loras"] += 1
                    else:
                        summary["unets"] += 1

    summary["scanned"] = scanned_count
    summary["total_registered"] = registered_count
    logger.info(f"Model scan complete: Scanned {scanned_count} files, registered {registered_count} entries into SQLite.")
    return summary


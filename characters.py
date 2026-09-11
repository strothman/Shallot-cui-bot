"""
Character Registry & Masking System for Shallot-cui-bot.
Provides centralized character configuration, LoRA mappings, prompt shorthands,
and silent trigger word translation for privacy masking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re
import discord
from discord import app_commands

@dataclass
class CharacterProfile:
    id: str
    display_name: str
    trained_trigger: str
    lora_sdxl: Optional[str]
    lora_flux: Optional[str] = None
    lora_krea2: Optional[str] = None
    default_weight: float = 0.85
    shorthands: List[str] = field(default_factory=list)
    description: str = ""
    base_prompt_traits: Optional[str] = None
    is_private: bool = False

# Registered Characters
CHARACTERS: Dict[str, CharacterProfile] = {
    "ogarla": CharacterProfile(
        id="ogarla",
        display_name="Ogarla",
        trained_trigger="ogarla",
        lora_sdxl="ogarla_epoch_5.safetensors",
        lora_flux="ogarlaflux_epoch_5.safetensors",
        lora_krea2="Krea2\\ogarla_krea2.safetensors",
        default_weight=0.70,
        shorthands=["ogarla", "oga"],
        description="Original Ogarla character LoRA (SDXL, Flux & Krea 2)",
        is_private=False
    ),
    "valerie": CharacterProfile(
        id="valerie",
        display_name="Valerie",
        trained_trigger="jen",
        lora_sdxl="jen_epoch_5.safetensors",
        lora_flux=None,
        lora_krea2=None,
        default_weight=0.90,
        shorthands=["valerie", "val"],
        description="Valerie character LoRA (SDXL)",
        base_prompt_traits="brown hair, dark brown eyes, realistic skin texture",
        is_private=True
    ),
    "sully": CharacterProfile(
        id="sully",
        display_name="Sully",
        trained_trigger="susa",
        lora_sdxl="susa_epoch_6.safetensors",
        lora_flux=None,
        default_weight=0.85,
        shorthands=["sully", "sul"],
        description="Sully character LoRA (SDXL)",
        base_prompt_traits="black hair, thin rim glasses",
        is_private=True
    ),
    "mageill": CharacterProfile(
        id="mageill",
        display_name="Mageill",
        trained_trigger="mageill",
        lora_sdxl="mageill_epoch_5.safetensors",
        default_weight=0.85,
        shorthands=["mageill", "mag", "mageill5", "mag5", "mageill_e5", "mag_e5"],
        description="Original Mageill character LoRA (SDXL) - Epoch 5 (Default)",
        is_private=False
    ),
    "mageill_e3": CharacterProfile(
        id="mageill_e3",
        display_name="Mageill (Epoch 3)",
        trained_trigger="mageill",
        lora_sdxl="mageill_epoch_3.safetensors",
        default_weight=0.85,
        shorthands=["mageill3", "mag3", "mageill_e3", "mag_e3"],
        description="Mageill character LoRA (SDXL) - Epoch 3",
        is_private=False
    ),
    "mageill_e4": CharacterProfile(
        id="mageill_e4",
        display_name="Mageill (Epoch 4)",
        trained_trigger="mageill",
        lora_sdxl="mageill_epoch_4.safetensors",
        default_weight=0.85,
        shorthands=["mageill4", "mag4", "mageill_e4", "mag_e4"],
        description="Mageill character LoRA (SDXL) - Epoch 4",
        is_private=False
    ),
    "mageill_e6": CharacterProfile(
        id="mageill_e6",
        display_name="Mageill (Epoch 6)",
        trained_trigger="mageill",
        lora_sdxl="mageill_epoch_6.safetensors",
        default_weight=0.85,
        shorthands=["mageill6", "mag6", "mageill_e6", "mag_e6"],
        description="Mageill character LoRA (SDXL) - Epoch 6",
        is_private=False
    ),
    "cheri": CharacterProfile(
        id="cheri",
        display_name="Cheri",
        trained_trigger="cheri",
        lora_sdxl="cheri_epoch_6.safetensors",
        default_weight=0.85,
        shorthands=["cheri", "che", "cheri6", "che6", "cheri_e6"],
        description="Original Cheri character LoRA (SDXL) - Epoch 6 (Default)",
        base_prompt_traits="blonde hair",
        is_private=False
    ),
    "cheri_e4": CharacterProfile(
        id="cheri_e4",
        display_name="Cheri (Epoch 4)",
        trained_trigger="cheri",
        lora_sdxl="cheri_epoch_4.safetensors",
        default_weight=0.85,
        shorthands=["cheri4", "che4", "cheri_e4"],
        description="Cheri character LoRA (SDXL) - Epoch 4",
        base_prompt_traits="blonde hair",
        is_private=False
    ),
}

def get_character(key: str) -> Optional[CharacterProfile]:
    """Finds a character by ID or alias shorthand."""
    if not key:
        return None
    k = key.lower().strip()
    if k in CHARACTERS:
        return CHARACTERS[k]
    for char in CHARACTERS.values():
        if k == char.display_name.lower() or k in [s.lower() for s in char.shorthands]:
            return char
    return None

def get_all_characters() -> List[CharacterProfile]:
    """Returns all registered characters."""
    return list(CHARACTERS.values())

def mask_character_in_prompt(prompt: str, character_id: Optional[str] = None) -> str:
    """
    Replaces trained trigger words with user-facing display names for privacy.
    e.g. 'jen, coffee shop' -> 'valerie, coffee shop'
    e.g. 'susa, reading book' -> 'sully, reading book'
    """
    if character_id:
        char = get_character(character_id)
        targets = [char] if char else []
    else:
        targets = list(CHARACTERS.values())

    for char in targets:
        if not char or not char.is_private:
            continue
        pattern = rf"\b{re.escape(char.trained_trigger)}\b"
        prompt = re.sub(pattern, char.id, prompt, flags=re.IGNORECASE)
    return prompt

def inject_trained_trigger_in_prompt(prompt: str, character_id: str) -> str:
    """
    Silently substitutes the character alias with the real trained trigger word and base traits for ComfyUI.
    e.g. 'sully, sitting on a bench' -> 'susa, black hair, thin rim glasses, sitting on a bench'
    """
    char = get_character(character_id)
    if not char:
        return prompt

    trigger_phrase = char.trained_trigger
    if char.base_prompt_traits:
        trigger_phrase = f"{char.trained_trigger}, {char.base_prompt_traits}"

    # If character trained trigger is already present
    if re.search(rf"\b{re.escape(char.trained_trigger)}\b", prompt, flags=re.IGNORECASE):
        if char.base_prompt_traits and not re.search(rf"\b{re.escape(char.base_prompt_traits)}\b", prompt, flags=re.IGNORECASE):
            return f"{prompt}, {char.base_prompt_traits}"
        return prompt

    # Replace character alias/name with trigger_phrase if present
    for alias in [char.id, char.display_name] + char.shorthands:
        pattern = rf"\b{re.escape(alias)}\b"
        if re.search(pattern, prompt, flags=re.IGNORECASE):
            return re.sub(pattern, trigger_phrase, prompt, flags=re.IGNORECASE)

    # Otherwise prepend the trigger phrase
    return f"{trigger_phrase}, {prompt}".strip()


def scan_krea2_loras(lora_dir: Optional[str] = None) -> List[str]:
    """
    Scans the Krea 2 LoRA directory (defaulting to C:\\ComfyUI\\ComfyUI\\models\\loras\\Krea2)
    for available .safetensors files. Automatically discovers new LoRAs and returns their relative paths.
    """
    import os
    target_dirs = []
    if lora_dir:
        target_dirs.append(lora_dir)
    default_dir = r"C:\ComfyUI\ComfyUI\models\loras\Krea2"
    if default_dir not in target_dirs:
        target_dirs.append(default_dir)

    discovered = []
    for d in target_dirs:
        if os.path.exists(d) and os.path.isdir(d):
            for fname in os.listdir(d):
                if fname.lower().endswith(".safetensors"):
                    rel_path = os.path.join("Krea2", fname)
                    discovered.append(rel_path)
                    # Dynamic auto-registration if not already registered
                    stem = os.path.splitext(fname)[0].lower()
                    char_key = stem.replace("_krea2", "").replace("-krea2", "").replace("krea2_", "")
                    if char_key not in CHARACTERS and not any(w in char_key for w in ["wetness", "de-oiler", "deoiler", "skin", "matte", "gloss", "lighting"]):
                        CHARACTERS[char_key] = CharacterProfile(
                            id=char_key,
                            display_name=char_key.capitalize(),
                            trained_trigger=char_key,
                            lora_sdxl=None,
                            lora_flux=None,
                            lora_krea2=rel_path,
                            default_weight=0.85,
                            shorthands=[char_key],
                            description=f"Auto-discovered Krea 2 LoRA ({fname})",
                            is_private=False
                        )
    return discovered


CHARACTER_EMOJIS: Dict[str, str] = {
    "ogarla": "🌿",
    "valerie": "✨",
    "sully": "👓",
    "cheri": "🌸",
    "cheri_e4": "🌸",
    "mageill": "🔮",
    "mageill_e3": "🔮",
    "mageill_e4": "🔮",
    "mageill_e6": "🔮",
}

def get_character_display_badge(
    key: Optional[str] = None,
    architecture: Optional[str] = None,
    prompt: Optional[str] = None
) -> str:
    """
    Returns a unified, beautifully-formatted user-facing character badge for Discord embeds and views.
    Handles character IDs, weight suffixes, aliases, architecture context, and fallback prompt inspection.
    """
    if not key or str(key).lower() in ["none", "nochar", "off", "false"]:
        if prompt:
            p_lower = str(prompt).lower()
            if "--valerie" in p_lower or "--val" in p_lower or "valerie" in p_lower:
                key = "valerie"
            elif "--sully" in p_lower or "--sul" in p_lower or "sully" in p_lower:
                key = "sully"
            elif "--cheri4" in p_lower:
                key = "cheri_e4"
            elif "--cheri" in p_lower or "cheri" in p_lower:
                key = "cheri"
            elif "--mageill6" in p_lower:
                key = "mageill_e6"
            elif "--mageill4" in p_lower:
                key = "mageill_e4"
            elif "--mageill3" in p_lower:
                key = "mageill_e3"
            elif "--mageill" in p_lower or "mageill" in p_lower:
                key = "mageill"
            elif "--ogarla" in p_lower or "--oga" in p_lower or "ogarla" in p_lower:
                key = "ogarla"
            else:
                return "None"
        else:
            return "None"

    k = str(key).lower().strip()
    is_krea2 = architecture and ("krea" in str(architecture).lower())

    if is_krea2:
        if k in ["ogarla.85", "ogarla", "oga"]:
            return "🌿 Ogarla (.85 - Default)"
        elif k in ["ogarla.70", "ogarla_light"]:
            return "🌿 Ogarla (.70 - Light)"
        elif k in ["valerie.90", "valerie", "val"]:
            return "✨ Valerie (.90 - Default)"
        elif k in ["valerie.70", "valerie_light"]:
            return "✨ Valerie (.70 - Light)"

    # SDXL / General Presets
    sdxl_map = {
        "ogarla": "🌿 Ogarla (--ogarla.70)",
        "ogarla.70": "🌿 Ogarla (--ogarla.70)",
        "ogarla.85": "🌿 Ogarla (--ogarla.85)",
        "valerie": "👩 Valerie (--valerie.85)",
        "valerie.85": "👩 Valerie (--valerie.85)",
        "valerie.70": "👩 Valerie (--valerie.70)",
        "valerie.90": "✨ Valerie (.90 - Default)",
        "sully": "👓 Sully (--sully.85)",
        "sully.85": "👓 Sully (--sully.85)",
        "sully.70": "👓 Sully (--sully.70)",
        "cheri": "🌸 Cheri (Epoch 6)",
        "cheri.85": "🌸 Cheri E6 (--cheri.85)",
        "cheri_e4": "🌸 Cheri (Epoch 4)",
        "cheri4": "🌸 Cheri (Epoch 4)",
        "mageill": "🔮 Mageill (Epoch 5)",
        "mageill.85": "🔮 Mageill E5 (--mageill.85)",
        "mageill_e6": "🔮 Mageill (Epoch 6)",
        "mageill6": "🔮 Mageill (Epoch 6)",
        "mageill_e4": "🔮 Mageill (Epoch 4)",
        "mageill4": "🔮 Mageill (Epoch 4)",
        "mageill_e3": "🔮 Mageill (Epoch 3)",
        "mageill3": "🔮 Mageill (Epoch 3)",
    }
    if k in sdxl_map:
        return sdxl_map[k]

    # Dynamic fallback lookup in CHARACTERS
    char = get_character(k)
    if char:
        emoji = CHARACTER_EMOJIS.get(char.id, "🎭")
        return f"{emoji} {char.display_name}"

    return str(key)


def get_character_autocomplete_choices(
    current: str = "",
    architecture: Any = None
) -> List[app_commands.Choice[str]]:
    """
    Dynamically generates and filters Discord slash command choices based on registered characters
    and newly-scanned LoRAs for the specified architecture.
    """
    arch_str = str(architecture).lower() if architecture else "sdxl"
    choices: List[app_commands.Choice[str]] = []

    if "krea" in arch_str:
        try:
            scan_krea2_loras()
        except Exception:
            pass

        # Filter characters that have Krea 2 LoRAs
        for char_id, char in CHARACTERS.items():
            if not char.lora_krea2:
                continue
            emoji = CHARACTER_EMOJIS.get(char_id, "🎭")
            def_wt = 0.85 if char_id == "ogarla" else (char.default_weight or 0.85)
            def_wt_str = f"{int(round(def_wt * 100)):02d}"
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} Krea 2 (. {def_wt_str} - Default)".replace("(. ", "(."),
                value=f"{char_id}.{def_wt_str}"
            ))
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} Krea 2 (.70 - Light)",
                value=f"{char_id}.70"
            ))

    elif "flux" in arch_str:
        for char_id, char in CHARACTERS.items():
            if not char.lora_flux:
                continue
            emoji = CHARACTER_EMOJIS.get(char_id, "🌿")
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} Flux (.85 - Default)",
                value=f"{char_id}.85"
            ))
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} Flux (.70 - Light)",
                value=f"{char_id}.70"
            ))

    else:
        # SDXL (Default)
        # Order prioritized: Mageill, Ogarla, Valerie, Sully, Cheri
        priority_order = ["mageill", "ogarla", "valerie", "sully", "cheri"]
        seen = set()
        ordered_chars = []
        for pid in priority_order:
            if pid in CHARACTERS and CHARACTERS[pid].lora_sdxl:
                ordered_chars.append(CHARACTERS[pid])
                seen.add(pid)
        for cid, char in CHARACTERS.items():
            if cid not in seen and char.lora_sdxl:
                ordered_chars.append(char)

        for char in ordered_chars:
            emoji = CHARACTER_EMOJIS.get(char.id, "🎭")
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} (.85 - Default)",
                value=f"{char.id}.85"
            ))
            choices.append(app_commands.Choice(
                name=f"{emoji} {char.display_name} (.70 - Light)",
                value=f"{char.id}.70"
            ))

    # Apply search filtering if user typed characters
    if current:
        cur_clean = current.lower().strip().lstrip("-")
        filtered = [
            c for c in choices
            if cur_clean in c.name.lower() or cur_clean in c.value.lower()
        ]
        if filtered:
            return filtered[:25]

    return choices[:25]


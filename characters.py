"""
Character Registry & Masking System for Shallot-cui-bot.
Provides centralized character configuration, LoRA mappings, prompt shorthands,
and silent trigger word translation for privacy masking.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import re

@dataclass
class CharacterProfile:
    id: str
    display_name: str
    trained_trigger: str
    lora_sdxl: Optional[str]
    lora_flux: Optional[str] = None
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
        default_weight=0.85,
        shorthands=["ogarla", "oga"],
        description="Original Ogarla character LoRA (SDXL & Flux)",
        is_private=False
    ),
    "valerie": CharacterProfile(
        id="valerie",
        display_name="Valerie",
        trained_trigger="jen",
        lora_sdxl="jen_epoch_5.safetensors",
        lora_flux=None,
        default_weight=0.85,
        shorthands=["valerie", "val"],
        description="Valerie character LoRA (SDXL)",
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

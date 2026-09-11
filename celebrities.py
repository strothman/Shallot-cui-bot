"""
Celebrity Registry & Prompt Injection System for Krea 2 Turbo.
Provides favorite celebrity definitions, display formatting, and prompt injection
leveraging Krea 2's native inherent facial recognition without needing LoRAs.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass, field
import re
from discord import app_commands


@dataclass
class CelebrityProfile:
    id: str
    display_name: str
    prompt_name: str
    emoji: str = "🌟"
    description: str = ""
    shorthands: List[str] = field(default_factory=list)


FAVORITE_CELEBRITIES: Dict[str, CelebrityProfile] = {
    "audrey_hepburn": CelebrityProfile(
        id="audrey_hepburn",
        display_name="Audrey Hepburn",
        prompt_name="Audrey Hepburn",
        emoji="👑",
        description="Classic icon (Breakfast at Tiffany's, Roman Holiday)",
        shorthands=["audrey", "hepburn"]
    ),
    "grace_kelly": CelebrityProfile(
        id="grace_kelly",
        display_name="Grace Kelly",
        prompt_name="Grace Kelly",
        emoji="💎",
        description="Classic elegance (Rear Window, To Catch a Thief)",
        shorthands=["grace", "kelly"]
    ),
    "nicole_kidman": CelebrityProfile(
        id="nicole_kidman",
        display_name="Nicole Kidman",
        prompt_name="Nicole Kidman",
        emoji="✨",
        description="Regal Australian actress (Moulin Rouge, The Others)",
        shorthands=["nicole", "kidman"]
    ),
    "margot_robbie": CelebrityProfile(
        id="margot_robbie",
        display_name="Margot Robbie",
        prompt_name="Margot Robbie",
        emoji="💖",
        description="A-list Australian actress (Barbie, Once Upon a Time)",
        shorthands=["margot", "robbie"]
    ),
    "sandra_bullock": CelebrityProfile(
        id="sandra_bullock",
        display_name="Sandra Bullock",
        prompt_name="Sandra Bullock",
        emoji="🎬",
        description="Acclaimed actress (Gravity, Speed, Miss Congeniality)",
        shorthands=["sandra", "bullock"]
    ),
    "emma_stone": CelebrityProfile(
        id="emma_stone",
        display_name="Emma Stone",
        prompt_name="Emma Stone",
        emoji="🌟",
        description="Expressive Oscar winner (La La Land, Poor Things)",
        shorthands=["stone", "emmastone"]
    ),
    "anya_taylor_joy": CelebrityProfile(
        id="anya_taylor_joy",
        display_name="Anya Taylor-Joy",
        prompt_name="Anya Taylor-Joy",
        emoji="👁️",
        description="Distinctive wide-set eyes (The Queen's Gambit, Furiosa)",
        shorthands=["anya", "taylorjoy", "taylor-joy"]
    ),
    "zendaya": CelebrityProfile(
        id="zendaya",
        display_name="Zendaya",
        prompt_name="Zendaya",
        emoji="🌺",
        description="Modern fashion & screen icon (Dune, Euphoria)",
        shorthands=["zendaya"]
    ),
    "cameron_diaz": CelebrityProfile(
        id="cameron_diaz",
        display_name="Cameron Diaz",
        prompt_name="Cameron Diaz",
        emoji="☀️",
        description="Iconic 90s/2000s star (The Mask, Charlie's Angels)",
        shorthands=["cameron", "diaz"]
    ),
    "saoirse_ronan": CelebrityProfile(
        id="saoirse_ronan",
        display_name="Saoirse Ronan",
        prompt_name="Saoirse Ronan",
        emoji="🎭",
        description="Acclaimed Irish actress (Little Women, Lady Bird, Atonement)",
        shorthands=["saoirse", "ronan"]
    ),
    "emma_watson": CelebrityProfile(
        id="emma_watson",
        display_name="Emma Watson",
        prompt_name="Emma Watson",
        emoji="📖",
        description="English actress (Harry Potter, Beauty and the Beast)",
        shorthands=["watson", "emmawatson"]
    ),
    "michelle_pfeiffer": CelebrityProfile(
        id="michelle_pfeiffer",
        display_name="Michelle Pfeiffer",
        prompt_name="Michelle Pfeiffer",
        emoji="🐾",
        description="Legendary actress (Scarface, Batman Returns, Stardust)",
        shorthands=["pfeiffer", "michelle"]
    ),
    "gal_gadot": CelebrityProfile(
        id="gal_gadot",
        display_name="Gal Gadot",
        prompt_name="Gal Gadot",
        emoji="⚡",
        description="Action star (Wonder Woman, Red Notice)",
        shorthands=["gal", "gadot"]
    ),
    "taylor_swift": CelebrityProfile(
        id="taylor_swift",
        display_name="Taylor Swift",
        prompt_name="Taylor Swift",
        emoji="🎸",
        description="Global music superstar (Midnights, 1989, Eras)",
        shorthands=["taylor", "swift"]
    ),
    "ariana_grande": CelebrityProfile(
        id="ariana_grande",
        display_name="Ariana Grande",
        prompt_name="Ariana Grande",
        emoji="🎀",
        description="Pop superstar (Wicked, Thank U Next, Sweetener)",
        shorthands=["ariana", "grande"]
    ),
    "keira_knightley": CelebrityProfile(
        id="keira_knightley",
        display_name="Keira Knightley",
        prompt_name="Keira Knightley",
        emoji="🗡️",
        description="British period drama icon (Pirates of the Caribbean, Pride & Prejudice)",
        shorthands=["keira", "knightley"]
    ),
}


def get_celebrity(key: Optional[str]) -> Optional[CelebrityProfile]:
    """Finds a celebrity profile by id, display name, or alias shorthand."""
    if not key or str(key).lower() in ["none", "noceleb", "off", "false"]:
        return None
    raw = str(key).lower().strip()
    normalized = raw.replace("-", "_").replace(" ", "_")
    if normalized in FAVORITE_CELEBRITIES:
        return FAVORITE_CELEBRITIES[normalized]
    for celeb in FAVORITE_CELEBRITIES.values():
        if (
            raw == celeb.display_name.lower()
            or raw == celeb.id.lower()
            or raw in [s.lower() for s in celeb.shorthands]
        ):
            return celeb
    return None


def get_celebrity_display_badge(key: Optional[str] = None) -> str:
    """Returns a unified user-facing celebrity badge for Discord embeds and views."""
    if not key or str(key).lower() in ["none", "noceleb", "off", "false"]:
        return "None"
    celeb = get_celebrity(key)
    if celeb:
        return f"{celeb.emoji} {celeb.display_name}"
    return str(key)


def inject_celebrity_in_prompt(prompt: str, celebrity_key: str) -> str:
    """
    Injects celebrity name into the prompt if not already present.
    Prepends to the prompt so Krea 2 focuses identity locking on the subject.
    """
    celeb = get_celebrity(celebrity_key)
    if not celeb:
        return prompt

    clean_p = prompt.strip() if prompt else ""
    # If already present, do not duplicate
    if re.search(rf"\b{re.escape(celeb.prompt_name)}\b", clean_p, re.IGNORECASE):
        return clean_p

    if not clean_p:
        return celeb.prompt_name
    return f"{celeb.prompt_name}, {clean_p}"


def get_celebrity_autocomplete_choices(current: str = "") -> List[app_commands.Choice[str]]:
    """Generates autocomplete choices for slash commands."""
    choices = [
        app_commands.Choice(
            name=f"{celeb.emoji} {celeb.display_name}",
            value=cid
        )
        for cid, celeb in FAVORITE_CELEBRITIES.items()
    ]
    if current:
        cur = current.lower().strip()
        filtered = [c for c in choices if cur in c.name.lower() or cur in c.value.lower()]
        return filtered[:25]
    return choices[:25]


CELEBRITY_CHOICES_KREA2: List[app_commands.Choice[str]] = [
    app_commands.Choice(name=f"{c.emoji} {c.display_name}", value=c.id)
    for c in FAVORITE_CELEBRITIES.values()
]

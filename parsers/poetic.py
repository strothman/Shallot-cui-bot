"""
Poetic Synthesis Engine and Dynamic Allegory Composer for Shallot-CUI Bot.
Elevates the 'Poetic Gamble' concept into an intelligent, multi-layered visual synthesizer
specifically tailored for SDXL Fine Art and Krea 2 Photorealism (Bertflow).
"""

import random
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

# =========================================================================
# Mood Archetypes & Creative Data Banks
# =========================================================================

class PoeticMood:
    WILD = "wild"
    ETHEREAL = "ethereal"
    GOTHIC = "gothic"
    NOIR = "noir"
    MYTHIC = "mythic"
    SURREAL = "surreal"

    ALL = [WILD, ETHEREAL, GOTHIC, NOIR, MYTHIC, SURREAL]

MOOD_DISPLAY_NAMES = {
    PoeticMood.WILD: "🎲 Wild Mystery",
    PoeticMood.ETHEREAL: "🌌 Ethereal & Dreamlike",
    PoeticMood.GOTHIC: "🕯️ Gothic Melancholy",
    PoeticMood.NOIR: "⚡ Neon Noir & Solitude",
    PoeticMood.MYTHIC: "🌿 Mythic & Ancient",
    PoeticMood.SURREAL: "🌀 Surreal Impossible",
}

# Mystery seed words if the user leaves seed empty
WILDCARD_SEEDS = [
    "solitude", "ember", "whisper", "memory", "tide", "echo",
    "gossamer", "hollow", "aurora", "reverie", "threshold", "silence",
    "fragility", "ascension", "abyss", "mirage", "sanctuary", "oblivion",
    "luminescence", "quicksilver", "twilight", "yearning", "labyrinth", "ephemeral"
]

# Physical, grounding metaphors organized by mood
METAPHORS_BY_MOOD: Dict[str, List[str]] = {
    PoeticMood.ETHEREAL: [
        "a glowing brass lantern burning underwater with steady golden luminescence",
        "weightless water droplets suspended in mid-air catching twilight rays",
        "golden moth wings delicately woven from illuminated parchment",
        "a silver key resting upon clear submerged riverbed stones",
        "shimmering glass ribbons drifting weightlessly through misty air",
        "a floating teacup reflecting an intricate constellation of stars",
        "a delicate blue flame dancing inside a porcelain bowl",
    ],
    PoeticMood.GOTHIC: [
        "roots of ancient black ivy growing through cracked marble pillars",
        "a tarnished antique mirror reflecting drifting candle smoke and shadows",
        "wilted dark crimson roses draped over a dust-covered velvet armchair",
        "a broken ornate crown resting in overgrown tall grass",
        "black candle wax dripping over weathered stone engravings",
        "a birdcage woven from rusted iron vines holding an amber crystal",
        "an abandoned violin resting on a damp stone staircase",
    ],
    PoeticMood.NOIR: [
        "cigarette smoke curling through a single shaft of amber light in rain",
        "neon reflections fracturing across puddle-soaked asphalt",
        "a solitary silhouette standing at the edge of an empty wet pier",
        "a cracked payphone receiver dangling under flickering tungsten lamps",
        "rain cascading down a steamy window overlooking dark city rooftops",
        "a rotary clock frozen at midnight beside a half-filled glass",
    ],
    PoeticMood.MYTHIC: [
        "ancient golden wheat growing through the fractured stone of a forgotten throne",
        "a monolithic weeping willow with leaves glowing like pale sunlight",
        "a celestial halo carved from weathered sandstone resting against ancient moss",
        "a crystal spring welling up through carved obsidian runes",
        "a stone statue half-reclaimed by luminous flowering brambles",
        "a sacred silver horn resting on a sunlit mountain altar",
    ],
    PoeticMood.SURREAL: [
        "a grand spiral staircase ascending endlessly into an overcast sky",
        "architectural archways folding seamlessly into ocean waves",
        "an antique pocket watch overflowing with wild lavender and blooming moss",
        "a quiet ballroom where gravity reverses and chandeliers float like jellyfish",
        "a weathered wooden doorway standing alone in an endless sea of mist",
        "trees whose leaves are shimmering antique mirrors reflecting twilight",
    ]
}

SETTINGS_BY_MOOD: Dict[str, List[str]] = {
    PoeticMood.ETHEREAL: [
        "an ethereal flooded conservatory with morning mist pouring through glass skylights",
        "a serene moonlit lake pier surrounded by pale rising vapor",
        "a luminous cavern illuminated by floating bioluminescent particles",
        "an endless shallow reflecting pool under a dawn twilight sky",
    ],
    PoeticMood.GOTHIC: [
        "a decaying cathedral sanctuary with shattered stained-glass windows",
        "a neglected candlelit library with towering gothic bookshelves and dust motes",
        "a foggy courtyard of an abandoned baroque mansion surrounded by iron gates",
        "a subterranean stone chapel lit only by clusters of dripping wax tapers",
    ],
    PoeticMood.NOIR: [
        "a rain-slicked alleyway bathed in deep amber and indigo neon glow",
        "a desolate 1950s roadside diner window at 3 AM during a coastal downpour",
        "an empty elevated train platform shrouded in industrial fog and distant city lights",
        "a shadowy art deco apartment interior overlooking rain-drenched streets",
    ],
    PoeticMood.MYTHIC: [
        "a sun-dappled ancient olive grove surrounding weathered classical ruins",
        "a high mountain sanctuary above an ocean of clouds in golden dawn light",
        "an overgrown sacred spring framed by cyclopean moss-covered stones",
        "a serene forgotten shrine hidden within a primeval cedar forest",
    ],
    PoeticMood.SURREAL: [
        "an impossible dreamscape of floating architectural terraces among storm clouds",
        "a vast quiet desert where crystalline monoliths reflect two moons",
        "a grand marble hall where the floor dissolves into calm turquoise water",
        "a twilight meadow where gravity bends and paper lanterns float like planets",
    ]
}

LIGHTING_AND_ATMOSPHERE: Dict[str, List[str]] = {
    PoeticMood.ETHEREAL: [
        "blue hour twilight mist with soft glowing amber reflections",
        "ethereal diffused morning light filtering through humid haze",
        "pale celestial moonlight illuminating gentle floating dust motes",
    ],
    PoeticMood.GOTHIC: [
        "dramatic chiaroscuro lighting, deep velvet shadows, and warm candlelit rim",
        "overcast twilight with stormy indigo clouds and a single shaft of gold",
        "moody nocturnal gloom pierced by warm amber flame and dark fog",
    ],
    PoeticMood.NOIR: [
        "harsh cinematic rim lighting, high-contrast shadows, and wet neon specular sheen",
        "dense foggy haze pierced by warm sodium-vapor streetlamps and headlights",
        "smoky violet twilight with rain streaks catching cold blue backlight",
    ],
    PoeticMood.MYTHIC: [
        "volumetric golden hour sunbeams slicing through ancient forest canopy",
        "radiant dawn glow with warm dust particles and heavenly atmospheric depth",
        "serene alpine daylight with crystal-clear horizon and luminous haze",
    ],
    PoeticMood.SURREAL: [
        "dreamlike dual-source lighting with warm ochre and cool cyan highlights",
        "eerie eclipse twilight with a glowing coronal corona in a dark sky",
        "hyper-real twilight illumination casting gentle impossible shadows",
    ]
}

COLOR_HARMONIES: Dict[str, List[str]] = {
    PoeticMood.ETHEREAL: [
        "deep teal, soft seafoam, and warm burnished gold",
        "bruised rose, smoky lavender, and pale ivory",
        "opalescent silver, pale cyan, and warm champagne highlights",
    ],
    PoeticMood.GOTHIC: [
        "black-green, deep crimson velvet, and warm antique bronze",
        "smoky violet, charcoal shadow, and candle amber",
        "midnight navy, rust orange, and weathered gold leaf",
    ],
    PoeticMood.NOIR: [
        "monochromatic charcoal, slate, and vibrant amber neon accents",
        "indigo shadow, cold cyan reflections, and tungsten yellow",
        "pitch black, gunmetal gray, and neon crimson rim light",
    ],
    PoeticMood.MYTHIC: [
        "warm terracotta, olive green, and radiant golden leaf",
        "white marble, cerulean blue, and honey-gold sunlight",
        "earthy sienna, dark pine, and polished copper",
    ],
    PoeticMood.SURREAL: [
        "iridescent magenta, deep cobalt, and glowing mint green",
        "burnt ochre, twilight lavender, and bioluminescent turquoise",
        "monochrome slate contrasted against brilliant liquid gold",
    ]
}

EMOTIONAL_LENSES = [
    "haunting reverie", "quiet obsession", "tender yearning", "sacred awe",
    "melancholy nostalgia", "serene detachment", "solitary reflection", "dreamlike wonder"
]


@dataclass
class PoeticPromptResult:
    prompt: str
    stanza: str
    seed_word: str
    mood: str
    engine: str
    metaphor: str
    setting: str
    color_palette: str


def _resolve_mood(mood: Optional[str], rng: random.Random) -> str:
    if not mood or mood.lower() == PoeticMood.WILD or mood.lower() not in PoeticMood.ALL:
        concrete_moods = [PoeticMood.ETHEREAL, PoeticMood.GOTHIC, PoeticMood.NOIR, PoeticMood.MYTHIC, PoeticMood.SURREAL]
        return rng.choice(concrete_moods)
    return mood.lower()


def synthesize_poetic_prompt(
    seed: Optional[str] = None,
    mood: Optional[str] = "wild",
    engine: str = "sdxl",
    rng_seed: Optional[int] = None
) -> PoeticPromptResult:
    """
    Synthesizes a cohesive, non-literal poetic visual prompt for SDXL or Krea 2.
    
    Avoids meta-prompt tokens like 'Interpret poetically' or 'ONE WORD SEED',
    assembling instead a fluid, grammar-correct narrative scene with rich physical metaphors.
    """
    if rng_seed is None:
        rng_seed = random.randint(1, 999999999)
    rng = random.Random(rng_seed)

    # Clean and resolve seed word
    clean_seed = (seed or "").strip()
    # Strip any accidental flag tokens if user passed raw prompt
    clean_seed = re.sub(r'[-\u2014\u2013]{1,2}(?:gamble|poetic|ar|sref)\b.*', '', clean_seed, flags=re.IGNORECASE).strip()
    if not clean_seed:
        clean_seed = rng.choice(WILDCARD_SEEDS)

    chosen_mood = _resolve_mood(mood, rng)

    # Draw curated components aligned to the chosen mood
    lens = rng.choice(EMOTIONAL_LENSES)
    metaphor = rng.choice(METAPHORS_BY_MOOD.get(chosen_mood, METAPHORS_BY_MOOD[PoeticMood.ETHEREAL]))
    setting = rng.choice(SETTINGS_BY_MOOD.get(chosen_mood, SETTINGS_BY_MOOD[PoeticMood.ETHEREAL]))
    lighting = rng.choice(LIGHTING_AND_ATMOSPHERE.get(chosen_mood, LIGHTING_AND_ATMOSPHERE[PoeticMood.ETHEREAL]))
    palette = rng.choice(COLOR_HARMONIES.get(chosen_mood, COLOR_HARMONIES[PoeticMood.ETHEREAL]))

    # Craft 2-line lyrical stanza for Discord Embed
    stanza_line1 = f"A {lens} of *{clean_seed}* — {metaphor}."
    stanza_line2 = f"Set within {setting}, bathed in {lighting}."
    stanza = f"{stanza_line1}\n{stanza_line2}"

    # Build diffusion prompt tailored to the target architecture
    is_krea = (engine.lower() in ["krea", "krea2", "bertflow"])
    
    if is_krea:
        # Krea 2 / Bertflow thrives on tactile photographic realism and candid lighting
        diffusion_prompt = (
            f"A poetic visual allegory of {clean_seed}: {metaphor}, located within {setting}. "
            f"{lighting}, harmonious {palette} color grading. "
            f"35mm film photograph, highly detailed tactile textures, shallow depth of field, award-winning atmospheric composition, no text, no watermark"
        )
    else:
        # SDXL Fine Art thrives on rich painterly illustration, volumetric atmosphere, and deep textures
        diffusion_prompt = (
            f"masterpiece, best quality, fine art poetic illustration of {clean_seed}. "
            f"A visual allegory: {metaphor}, situated within {setting}. "
            f"{lighting}, color palette of {palette}. "
            f"Rich painterly textures, chiaroscuro depth, atmospheric mood, cinematic visual storytelling, no text, no watermark"
        )

    return PoeticPromptResult(
        prompt=diffusion_prompt,
        stanza=stanza,
        seed_word=clean_seed,
        mood=chosen_mood,
        engine="krea2" if is_krea else "sdxl",
        metaphor=metaphor,
        setting=setting,
        color_palette=palette,
    )

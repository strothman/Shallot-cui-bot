"""
Krea 2 (Bertflow) Blend Studio Controls and View Components for Shallot-CUI Bot.
"""

import logging
import discord
from characters import get_character_display_badge
from celebrities import FAVORITE_CELEBRITIES, get_celebrity_display_badge

logger = logging.getLogger("DiscordBot.Views.BlendKrea")


def build_blend_krea_embed(gen_data: dict, author_str: str = "User", image_url: str = None) -> discord.Embed:
    """Builds a streamlined, photorealism-focused embed for /blend-krea sessions."""
    vision_prompt = gen_data.get("krea2_prompt") or gen_data.get("detailed_caption", "No description")
    user_prompt = gen_data.get("user_prompt", "").strip()
    fused_prompt = gen_data.get("fused_prompt") or vision_prompt
    ar = gen_data.get("ar", "16:9")
    steps = int(gen_data.get("steps", 8))
    model_choice = gen_data.get("model_choice", "muse")
    wetness = float(gen_data.get("wetness", -2.0))
    comp = gen_data.get("composition", "off")
    char_choice = gen_data.get("char_choice", "none")
    celeb_choice = gen_data.get("celeb_choice", "none")

    model_display = "Muse v3.5 Extended" if "muse" in model_choice.lower() else "Pornmaster v2 (FP8)"
    if wetness <= -1.0:
        skin_display = "Matte Pores (-2.0)"
    elif wetness < 0.5:
        skin_display = "Natural Baseline (0.0)"
    elif wetness <= 1.5:
        skin_display = "Glossy / Dewy (+1.0)"
    else:
        skin_display = f"{wetness:+.1f}"

    if comp == "medium":
        comp_display = "Medium (Balanced Silhouette & Pose - 70% Denoise)"
    elif comp == "strong":
        comp_display = "Strong (Strict Silhouette Lock - 50% Denoise)"
    elif comp == "subtle":
        comp_display = "Subtle (Loose Pose & Atmosphere - 85% Denoise)"
    else:
        comp_display = "Off (Semantic Vision Only)"

    char_display = get_character_display_badge(char_choice, architecture="krea2")
    celeb_display = get_celebrity_display_badge(celeb_choice)

    embed = discord.Embed(
        title="📸 Krea 2 Blend Studio",
        description="Blend and remix your uploaded image using Florence-2 AI vision and Bert's Krea 2 Turbo photorealism pipeline.",
        color=discord.Color.from_rgb(220, 90, 40)
    )
    thumb_url = image_url or gen_data.get("image_url")
    if thumb_url:
        embed.set_thumbnail(url=thumb_url)

    disp_prompt = fused_prompt[:1020] + "..." if len(fused_prompt) > 1024 else fused_prompt
    embed.add_field(name="📜 Generation Prompt", value=disp_prompt, inline=False)

    settings_lines = [
        f"📐 **Ratio:** `{ar}` • 🤖 **Engine:** `{model_display}` • ⚡ **Steps:** `{steps}` • 💧 **Skin:** `{skin_display}`",
        f"🖼️ **Direct Comp:** `{comp_display}`",
        f"🎭 **Character:** `{char_display}` • 🌟 **Celebrity:** `{celeb_display}`"
    ]

    embed.add_field(
        name="⚙️ Pipeline Settings",
        value="\n".join(settings_lines),
        inline=False
    )
    embed.set_footer(text=f"Requested by {author_str} • Krea 2 Turbo Flow-Matching")
    return embed


class BlendKreaButtons(discord.ui.View):
    def __init__(self, generation_id: str, ar: str = "16:9", model_choice: str = "muse", wetness: float = -2.0, composition: str = "off", character: str = "none", celebrity: str = "none", steps: int = 8):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.ar = ar
        self.model_choice = model_choice
        self.wetness = wetness
        self.composition = composition or "off"
        self.character = character or "none"
        self.celebrity = celebrity or "none"
        self.steps = int(steps or 8)

        # Row 0: Aspect Ratios (21:9, 16:9, 1:1, 3:4, 9:16)
        ar_options = [("21:9", "21:9"), ("16:9", "16:9"), ("1:1", "1:1"), ("3:4", "3:4"), ("9:16", "9:16")]
        for label, val in ar_options:
            is_selected = (self.ar == val)
            style = discord.ButtonStyle.primary if is_selected else discord.ButtonStyle.secondary
            self.add_item(discord.ui.Button(
                label=f"📐 {label}",
                style=style,
                custom_id=f"set_blend_krea_ar:{self.generation_id}:{val}",
                row=0
            ))

        # Row 1: Direct Composition Dropdown Selector
        comp_options = [
            discord.SelectOption(
                label="Off (Semantic Vision Only)",
                value="off",
                emoji="🚫",
                description="Full creative freedom without pose lock (Default)",
                default=(self.composition in [None, "off", "none", "style"])
            ),
            discord.SelectOption(
                label="Subtle (Loose Pose & Atmosphere)",
                value="subtle",
                emoji="🖼️",
                description="Light structural guide (85% denoise)",
                default=(self.composition in ["subtle", "85"])
            ),
            discord.SelectOption(
                label="Medium (Balanced Silhouette & Pose)",
                value="medium",
                emoji="🖼️",
                description="Preserves subject outline & pose (70% denoise)",
                default=(self.composition in ["medium", "70"])
            ),
            discord.SelectOption(
                label="Strong (Strict Silhouette & Pose Lock)",
                value="strong",
                emoji="🔒",
                description="Locks silhouette & composition tightly (50% denoise)",
                default=(self.composition in ["strong", "50"])
            ),
        ]
        self.add_item(discord.ui.Select(
            placeholder="🖼️ Select Composition / Pose Retention...",
            options=comp_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_krea_comp:{self.generation_id}",
            row=1
        ))

        # Row 2: Character LoRA Dropdown Selector
        char_options = [
            discord.SelectOption(
                label="None (No Character LoRA)",
                value="none",
                emoji="🚫",
                description="Generate without character presets",
                default=(self.character in [None, "none", "nochar"])
            ),
            discord.SelectOption(
                label="Ogarla (Krea 2 Light - 0.70)",
                value="ogarla.70",
                emoji="🌿",
                description="Subtle Krea 2 Character LoRA (Recommended Default)",
                default=(self.character in ["ogarla", "ogarla.70", "oga", "ogarla_light"])
            ),
            discord.SelectOption(
                label="Ogarla (Krea 2 - 0.85)",
                value="ogarla.85",
                emoji="🌿",
                description="Trained Krea 2 Character LoRA (Heavy)",
                default=(self.character == "ogarla.85")
            ),
        ]
        self.add_item(discord.ui.Select(
            placeholder="🎭 Select Character LoRA (Ogarla)...",
            options=char_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_krea_char:{self.generation_id}",
            row=2
        ))

        # Row 3: Favorite Celebrity Dropdown Selector
        celeb_options = [
            discord.SelectOption(
                label="None (No Celebrity Preset)",
                value="none",
                emoji="🚫",
                description="Generate without celebrity injection",
                default=(self.celebrity in [None, "none", "noceleb", "off"])
            )
        ]
        for cid, cprof in FAVORITE_CELEBRITIES.items():
            is_def = (
                str(self.celebrity).lower() == cid.lower()
                or str(self.celebrity).lower() == cprof.display_name.lower()
                or str(self.celebrity).lower() in [s.lower() for s in cprof.shorthands]
            )
            celeb_options.append(discord.SelectOption(
                label=cprof.display_name,
                value=cid,
                emoji=cprof.emoji,
                description=cprof.description[:100],
                default=is_def
            ))

        self.add_item(discord.ui.Select(
            placeholder="🌟 Select Celebrity Preset (Audrey Hepburn, Zendaya, etc.)...",
            options=celeb_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_krea_celeb:{self.generation_id}",
            row=3
        ))

        # Row 4: Action & Engine Controls
        is_muse = ("muse" in self.model_choice.lower())
        model_label = "🤖 Engine: Muse" if is_muse else "🤖 Engine: Pornmaster"
        self.add_item(discord.ui.Button(
            label=model_label,
            style=discord.ButtonStyle.primary,
            custom_id=f"toggle_blend_krea_model:{self.generation_id}",
            row=4
        ))

        self.add_item(discord.ui.Button(
            label=f"⚡ Steps: {self.steps}",
            style=discord.ButtonStyle.secondary if self.steps == 8 else discord.ButtonStyle.primary,
            custom_id=f"toggle_blend_krea_steps:{self.generation_id}",
            row=4
        ))

        if self.wetness <= -1.0:
            wet_label = "💧 Skin: Matte"
            wet_style = discord.ButtonStyle.primary
        elif self.wetness < 0.5:
            wet_label = "💧 Skin: Natural"
            wet_style = discord.ButtonStyle.secondary
        else:
            wet_label = "💧 Skin: Glossy"
            wet_style = discord.ButtonStyle.secondary

        self.add_item(discord.ui.Button(
            label=wet_label,
            style=wet_style,
            custom_id=f"toggle_blend_krea_wetness:{self.generation_id}",
            row=4
        ))

        self.add_item(discord.ui.Button(
            label="✏️ Edit Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"edit_blend_krea_prompt:{self.generation_id}",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Generate Krea 2",
            style=discord.ButtonStyle.danger,
            custom_id=f"gen_blend_krea:{self.generation_id}",
            row=4
        ))

import logging
import re
import discord
from core_helpers import send_error_fallback
from characters import get_character_display_badge
from celebrities import FAVORITE_CELEBRITIES, get_celebrity_display_badge, get_celebrity

logger = logging.getLogger("DiscordBot")

class CustomSrefModal(discord.ui.Modal, title="Change Style Reference (--sref)"):
    sref_input = discord.ui.TextInput(
        label="Enter --sref code, style name, or image URL",
        placeholder="e.g. 772382, --sref 492104, or https://...",
        required=True,
        max_length=200
    )

    def __init__(self, generation_id: str, index: int, on_submit_callback=None):
        super().__init__()
        self.generation_id = generation_id
        self.index = index
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = self.sref_input.value.strip()
            if val and self.on_submit_callback:
                await self.on_submit_callback(interaction, self.generation_id, self.index, val)
        except Exception as e:
            logger.error(f"Error in CustomSrefModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to apply style reference: {e}")
class CancelGenerationView(discord.ui.View):
    """Temporary view attached to generation status messages allowing users to cancel execution."""
    def __init__(self, generation_id: str):
        super().__init__(timeout=600)
        self.generation_id = generation_id

        self.add_item(discord.ui.Button(
            label="🛑 Cancel",
            style=discord.ButtonStyle.danger,
            custom_id=f"cancel_gen:{self.generation_id}"
        ))


class RemixModal(discord.ui.Modal, title="✏️ Remix / Tweak Prompt"):
    prompt_input = discord.ui.TextInput(
        label="Edit Prompt",
        style=discord.TextStyle.paragraph,
        required=True,
        max_length=2000
    )
    seed_input = discord.ui.TextInput(
        label="Seed (leave blank for random)",
        required=False,
        max_length=20
    )

    def __init__(self, generation_id: str, initial_prompt: str = "", initial_seed: int = None, on_submit_callback=None):
        super().__init__()
        self.generation_id = generation_id
        self.on_submit_callback = on_submit_callback
        self.prompt_input.default = initial_prompt
        if initial_seed is not None:
            self.seed_input.default = str(initial_seed)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            p = self.prompt_input.value.strip()
            s = self.seed_input.value.strip()
            seed_val = int(s) if s.isdigit() else None
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, self.generation_id, p, seed_val)
        except Exception as e:
            logger.error(f"Error in RemixModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to submit remix: {e}")



class SavedSrefSelectView(discord.ui.View):
    def __init__(self, generation_id: str, index: int, favorites: list[dict], select_callback=None):
        super().__init__(timeout=60)
        self.generation_id = generation_id
        self.index = index
        self.callback_fn = select_callback

        options = []
        for fav in favorites[:25]:
            code = fav.get("style_code")
            name = fav.get("style_name", f"Style {code}")
            options.append(discord.SelectOption(
                label=name[:100],
                value=str(code),
                description=f"--sref {code}"
            ))

        select = discord.ui.Select(
            placeholder="⭐ Choose a saved style from /my_prompts...",
            min_values=1,
            max_values=1,
            options=options
        )
        select.callback = self.on_select
        self.add_item(select)

    async def on_select(self, interaction: discord.Interaction):
        selected_code = interaction.data["values"][0]
        if self.callback_fn:
            await self.callback_fn(interaction, self.generation_id, self.index, selected_code)


class IsolatedImageButtons(discord.ui.View):
    def __init__(self, generation_id: str, index: int, has_sref: bool = False, is_blend: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.index = index
        self.is_blend = is_blend

        # Row 0: Upscale & Variation Options (4 buttons max)
        self.add_item(discord.ui.Button(
            label="⚡ Detailed Upscale (1.25x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:1.25",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Creative Upscale (1.5x)",
            style=discord.ButtonStyle.success,
            custom_id=f"upscale_run:{self.generation_id}:{self.index}:1.5",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Subtle)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_subtle:{self.generation_id}:{self.index}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Strong)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_strong:{self.generation_id}:{self.index}",
            row=0
        ))

        # Row 1: Actions, Favorites, Remix & Studio Loop (up to 5 buttons max)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=1
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.primary,
            custom_id=f"remix:{self.generation_id}",
            row=1
        ))
        if is_blend:
            self.add_item(discord.ui.Button(
                label="🎛️ Adjust Blend",
                style=discord.ButtonStyle.secondary,
                custom_id=f"reblend:{self.generation_id}",
                row=1
            ))

        # Row 2: Change Style Reference (--sref) (only if has_sref=True, 3 buttons max)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="🎨 Custom --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_custom:{self.generation_id}:{self.index}",
                row=2
            ))
            self.add_item(discord.ui.Button(
                label="🎲 Random --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_random:{self.generation_id}:{self.index}",
                row=2
            ))
            self.add_item(discord.ui.Button(
                label="⭐ Saved --sref",
                style=discord.ButtonStyle.secondary,
                custom_id=f"sref_change_saved:{self.generation_id}:{self.index}",
                row=2
            ))


class UpscaleButtons(discord.ui.View):
    def __init__(self, generation_id: str, index: int, upscale_scale: str = "1.25", has_sref: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.index = index
        self.upscale_scale = upscale_scale

        # Row 0: Re-roll Grid & Redo Upscale
        self.add_item(discord.ui.Button(
            label="🔄 Start Over (New Grid)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"reroll:{self.generation_id}",
            row=0
        ))

        self.add_item(discord.ui.Button(
            label="⚡ Vary Upscale Details",
            style=discord.ButtonStyle.secondary,
            custom_id=f"upscale_redo:{self.generation_id}:{self.index}:{self.upscale_scale}",
            row=0
        ))

        # Row 1: Variations (Subtle vs Strong)
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Subtle)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_subtle:{self.generation_id}:{self.index}",
            row=1
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Vary (Strong)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"vary_strong:{self.generation_id}:{self.index}",
            row=1
        ))

        # Row 2: Favorite Buttons (Style & Prompt)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=2
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=2
        ))

        # Row 3: Change Style Reference (--sref)
        self.add_item(discord.ui.Button(
            label="🎨 Custom --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_custom:{self.generation_id}:{self.index}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="🎲 Random --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_random:{self.generation_id}:{self.index}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="⭐ Saved --sref",
            style=discord.ButtonStyle.secondary,
            custom_id=f"sref_change_saved:{self.generation_id}:{self.index}",
            row=3
        ))

class GridButtons(discord.ui.View):
    def __init__(self, generation_id, has_sref=False, is_blend=False):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.is_blend = is_blend

        # Row 0: Upscale buttons U1-U4
        for i in range(1, 5):
            self.add_item(discord.ui.Button(
                label=f"U{i}", 
                style=discord.ButtonStyle.secondary, 
                custom_id=f"upscale:{self.generation_id}:{i}",
                row=0
            ))

        # Row 1: Variation buttons V1-V4 + Re-roll
        for i in range(1, 5):
            self.add_item(discord.ui.Button(
                label=f"V{i}",
                style=discord.ButtonStyle.secondary,
                custom_id=f"variation:{self.generation_id}:{i}",
                row=1
            ))
        self.add_item(discord.ui.Button(
            label="🔄",
            style=discord.ButtonStyle.secondary,
            custom_id=f"reroll:{self.generation_id}",
            row=1
        ))

        # Row 2: Actions & Favorites (up to 5 buttons, adhering to Discord limit)
        if has_sref:
            self.add_item(discord.ui.Button(
                label="⭐ Favorite Style",
                style=discord.ButtonStyle.success,
                custom_id=f"fav_style:{self.generation_id}",
                row=2
            ))
        self.add_item(discord.ui.Button(
            label="⭐ Favorite Prompt",
            style=discord.ButtonStyle.success,
            custom_id=f"fav_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"copy_prompt:{self.generation_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.primary,
            custom_id=f"remix:{self.generation_id}",
            row=2
        ))
        if is_blend:
            self.add_item(discord.ui.Button(
                label="🎛️ Adjust Blend",
                style=discord.ButtonStyle.secondary,
                custom_id=f"reblend:{self.generation_id}",
                row=2
            ))

class DescribeButtons(discord.ui.View):
    def __init__(self, generation_id: str, ar: str = "16:9", sr = True, oga: bool = False, model_choice: str = "hyphoria"):
        super().__init__(timeout=None) # Persistent buttons
        self.generation_id = generation_id
        self.ar = ar
        self.sr = sr
        self.oga = oga
        self.model_choice = model_choice

        # Resolve sr mode
        if isinstance(sr, str):
            sr_mode = sr
        elif sr is True:
            sr_mode = 'sr90' if model_choice == 'hyphoria' else 'sr75'
        else:
            sr_mode = 'nosr'
        sr_tag = sr_mode
        oga_tag = 'oga' if self.oga else 'nooga'

        # Row 0: Aspect Ratio Selection (21:9, 16:9, 10:7, 3:5, 9:16)
        ar_options = [("21:9", "21:9"), ("16:9", "16:9"), ("10:7", "10:7"), ("3:5", "3:5"), ("9:16", "9:16")]
        for label, val in ar_options:
            is_selected = (self.ar == val)
            style = discord.ButtonStyle.primary if is_selected else discord.ButtonStyle.secondary
            self.add_item(discord.ui.Button(
                label=f"📐 {label}",
                style=style,
                custom_id=f"set_desc_ar:{self.generation_id}:{val}:{sr_tag}:{oga_tag}:{self.model_choice}",
                row=0
            ))

        # Row 1: LoRA Toggles (Semi-Realism & Ogarla)
        if sr_mode == "nosr":
            sr_label = "✨ Semi-Realism (--sr): OFF"
            sr_style = discord.ButtonStyle.secondary
            toggle_sr_val = "sr60"
        elif sr_mode in ["sr60", "sr.60"]:
            sr_label = "✨ Semi-Realism (--sr.60): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr70"
        elif sr_mode in ["sr70", "sr.70"]:
            sr_label = "✨ Semi-Realism (--sr.70): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr80"
        elif sr_mode in ["sr80", "sr.80"]:
            sr_label = "✨ Semi-Realism (--sr.80): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "sr90"
        elif sr_mode in ["sr90", "sr.90", "sr"]:
            sr_label = "✨ Semi-Realism (--sr.90): ON"
            sr_style = discord.ButtonStyle.primary
            toggle_sr_val = "nosr"
        else:
            sr_label = "✨ Semi-Realism (--sr): OFF"
            sr_style = discord.ButtonStyle.secondary
            toggle_sr_val = "sr60"

        self.add_item(discord.ui.Button(
            label=sr_label,
            style=sr_style,
            custom_id=f"toggle_desc_sr:{self.generation_id}:{self.ar}:{toggle_sr_val}:{oga_tag}:{self.model_choice}",
            row=1
        ))

        oga_style = discord.ButtonStyle.primary if self.oga else discord.ButtonStyle.secondary
        oga_label = "🌿 Ogarla (--ogarla.70): ON" if self.oga else "🌿 Ogarla (--ogarla.70): OFF"
        toggle_oga_val = 'nooga' if self.oga else 'oga'
        self.add_item(discord.ui.Button(
            label=oga_label,
            style=oga_style,
            custom_id=f"toggle_desc_oga:{self.generation_id}:{self.ar}:{sr_tag}:{toggle_oga_val}:{self.model_choice}",
            row=1
        ))

        # Row 2: Model Toggle (Hyphoria NAI)
        is_hyphoria = (self.model_choice == "hyphoria")
        hyp_style = discord.ButtonStyle.primary if is_hyphoria else discord.ButtonStyle.secondary
        hyp_label = "🤖 Model: Hyphoria NAI" if is_hyphoria else "🤖 Model: Default Checkpoint"
        next_model = "default" if is_hyphoria else "hyphoria"
        self.add_item(discord.ui.Button(
            label=hyp_label,
            style=hyp_style,
            custom_id=f"toggle_desc_model:{self.generation_id}:{self.ar}:{sr_tag}:{oga_tag}:{next_model}",
            row=2
        ))

        # Row 3: Generation Targets (Caption, Detailed, Krea 2)
        self.add_item(discord.ui.Button(
            label="🎨 Generate Caption",
            style=discord.ButtonStyle.primary,
            custom_id=f"gen_desc:{self.generation_id}:caption:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="🎨 Generate Detailed",
            style=discord.ButtonStyle.success,
            custom_id=f"gen_desc:{self.generation_id}:detailed:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Generate Krea 2",
            style=discord.ButtonStyle.danger,
            custom_id=f"gen_desc:{self.generation_id}:krea2:{self.ar}:{sr_tag}:{oga_tag}:{self.model_choice}",
            row=3
        ))

def build_blend_embed(gen_data: dict, author_str: str = "User", image_url: str = None, is_edited: bool = False) -> discord.Embed:
    """Builds a streamlined, professional embed for /blend sessions."""
    caption = gen_data.get("caption", "No caption")
    detailed = gen_data.get("detailed_caption", "No detailed description")
    extra = gen_data.get("extra_details", "").strip()
    
    ar = gen_data.get("ar", "16:9")
    sr = gen_data.get("sr", True)
    oga = gen_data.get("oga", False)
    model_choice = gen_data.get("model_choice", "wai")
    comp_strength = gen_data.get("comp_strength", "style")
    sref_rand = gen_data.get("sref_rand", "nosref")

    model_names = {
        "wai": "Wai Illustrious SDXL v1.70",
        "illustrious_realism": "Illustrious Realism v1.0",
        "realvis": "RealVisXL V4.0",
        "juggernaut": "Juggernaut XL",
        "copax": "Copax Timeless XL",
        "ultra": "Ultra Realistic XL v2.5",
        "hyphoria": "Hyphoria NAI",
        "nova": "Nova Furry",
        "muse": "Muse v3.5 Extended (Krea 2)",
        "pornmaster": "Pornmaster v2 (Krea 2 FP8)",
        "default": "Wai Illustrious SDXL v1.70"
    }
    model_display = model_names.get(model_choice, model_choice)

    comp_names = {
        "style": "🎨 Style Only (0.20)",
        "low": "🖼️ Low Comp (0.35)",
        "med": "🖼️ Med Comp (0.60)",
        "high": "🖼️ High Comp (0.85)"
    }
    comp_display = comp_names.get(comp_strength, comp_strength)

    char_choice = gen_data.get("char_choice")
    if not char_choice:
        char_choice = "ogarla" if oga else "none"

    char_display = get_character_display_badge(char_choice, architecture="sdxl")

    if sr is False or sr == "nosr":
        sr_display = "OFF"
    elif isinstance(sr, str) and sr.startswith("sr"):
        val = sr[2:]
        sr_display = f"--sr.{val}" if val.isdigit() and len(val) == 2 else f"--{sr}"
    elif sr is True:
        sr_display = "ON (--sr.75)"
    else:
        sr_display = str(sr)

    preset_style_names = {
        "nosref": "OFF",
        "sref": "1 Random Style",
        "sref1": "1 Random Style",
        "sref5": "5 Styles Batch",
        "sref10": "10 Styles Batch",
        "sref15": "15 Styles Batch",
        "preset_junji_ito": "🖋️ Junji Ito",
        "preset_martine_johanna": "🎨 Martine Johanna",
        "preset_dark_fantasy_landscape": "🏰 Dark Fantasy",
        "preset_cyberpunk_cityscape": "🌆 Cyberpunk",
        "preset_ethereal_portrait": "✨ Ethereal Portrait",
    }
    if sref_rand in preset_style_names:
        sref_display = preset_style_names[sref_rand]
    elif isinstance(sref_rand, str) and sref_rand.startswith("saved_"):
        sref_display = f"⭐ Saved (--sref {sref_rand[6:]})"
    elif not sref_rand:
        sref_display = "OFF"
    else:
        sref_display = str(sref_rand)

    title_text = "🎨 Image Blend Studio (Customized)" if is_edited else "🎨 Image Blend Studio"
    embed = discord.Embed(
        title=title_text,
        description="*Select your aspect ratio, model, and reference composition below, then click **Blend**.*",
        color=discord.Color.from_rgb(0, 168, 252)
    )
    
    embed.add_field(
        name="📋 Short Caption / Tags",
        value=f"> {caption[:900]}",
        inline=False
    )
    
    embed.add_field(
        name="🔍 Detailed Vision Description",
        value=f"> {detailed[:1000]}",
        inline=False
    )

    if extra:
        embed.add_field(
            name="✨ Custom Extra Details",
            value=f"> `{extra}`",
            inline=False
        )

    # 3-Column Studio Dashboard Fields
    embed.add_field(
        name="📐 Canvas & Framing",
        value=f"**Ratio:** `{ar}`\n**Comp:** `{comp_display}`",
        inline=True
    )
    embed.add_field(
        name="🤖 Checkpoint",
        value=f"**Model:** `{model_display}`\n**Realism:** `{sr_display}`",
        inline=True
    )
    embed.add_field(
        name="🎭 Aesthetics",
        value=f"**Char:** `{char_display}`\n**Style:** `{sref_display}`",
        inline=True
    )

    if image_url:
        embed.set_thumbnail(url=image_url)
    embed.set_footer(text=f"Florence-2 Vision AI • Requested by {author_str}")
    return embed


def build_blend_complete_embed(
    display_prompt: str,
    selected_model: str,
    seed: int,
    width: int,
    height: int,
    comp_strength: str = "style",
    cfg: float = 4.0,
    sref_info: dict = None,
    cref_image_name: str = None,
    cref_weight: float = 0.85,
    is_magic: bool = False,
    char_choice: str = None,
    sr_choice = None,
    user_name: str = "User",
    image_url: str = None,
    elapsed_time: float = None,
    expanded_prompts: list = None
) -> discord.Embed:
    """Builds a polished, 3-column inline studio dashboard embed for completed blend generations."""
    model_friendly_names = {
        "waiIllustriousSDXL_v170.safetensors": "Wai Illustrious SDXL v1.70",
        "illustriousRealismBy_v10VAE.safetensors": "Illustrious Realism V1",
        "RealVisXL_V4.0.safetensors": "RealVisXL V4.0",
        "juggernautXL_ragnarok.safetensors": "Juggernaut XL Ragnarok",
        "CopaxTimeLessXL.safetensors": "Copax Timeless XL",
        "ultraRealisticByStable_v25.safetensors": "UltraRealistic V2.5",
        "hyphoriaIlluNAI_v001.safetensors": "Hyphoria NAI v0.01",
        "novaFurryXL_ilV180A.safetensors": "Nova Furry XL v1.8",
        "wai": "Wai Illustrious SDXL v1.70",
        "illustrious_realism": "Illustrious Realism V1",
        "realvis": "RealVisXL V4.0",
        "juggernaut": "Juggernaut XL Ragnarok",
        "copax": "Copax Timeless XL",
        "ultra": "UltraRealistic V2.5",
        "hyphoria": "Hyphoria NAI v0.01",
        "nova": "Nova Furry XL v1.8",
    }
    model_display = model_friendly_names.get(selected_model, selected_model.replace(".safetensors", ""))

    comp_map = {
        "style": "🎨 Style Only (0.20)",
        "low": "🖼️ Light Comp (0.35)",
        "med": "🖼️ Med Comp (0.60)",
        "high": "🖼️ High Comp (0.85)",
    }
    comp_display = comp_map.get(comp_strength, comp_strength)

    # Resolve character badge
    char_display = get_character_display_badge(char_choice, architecture="sdxl", prompt=display_prompt)

    # Resolve semi-realism badge
    if not sr_choice or sr_choice in ["nosr", False]:
        m = re.search(r'--sr\.?(\d+)', display_prompt)
        if m:
            sr_display = f"--sr.{m.group(1)}"
        elif "semi-realism" in display_prompt.lower():
            sr_display = "Enabled"
        else:
            sr_display = "OFF"
    else:
        if isinstance(sr_choice, str) and sr_choice.startswith("sr"):
            sr_display = f"--sr.{sr_choice[2:]}"
        else:
            sr_display = "Enabled"

    # Resolve style / sref badge
    if sref_info and isinstance(sref_info, dict) and "code" in sref_info:
        s_name = sref_info.get("name", "")
        style_display = f"`--sref {sref_info['code']}`" + (f" *({s_name})*" if s_name else "")
    else:
        m = re.search(r'--sref\s+(\d+)', display_prompt)
        if m:
            style_display = f"`--sref {m.group(1)}`"
        elif "--sref random" in display_prompt.lower():
            style_display = "🎲 Random Style"
        else:
            style_display = "OFF"

    # Clean word-boundary prompt truncation
    clean_p = display_prompt.strip()
    if len(clean_p) > 300:
        truncated = clean_p[:297].rsplit(" ", 1)[0] + "..."
    else:
        truncated = clean_p

    embed = discord.Embed(
        title="✨ Image Blend Complete",
        description=f"**Steering Prompt:**\n> {truncated}",
        color=discord.Color.from_rgb(138, 43, 226) # Studio Violet #8A2BE2
    )

    if expanded_prompts and "{" in display_prompt and "}" in display_prompt:
        try:
            from parsers import clean_quadrant_prompts
            cleaned_eps = clean_quadrant_prompts(expanded_prompts, display_prompt)
            q_lines = []
            for idx, clean_ep in enumerate(cleaned_eps):
                if len(clean_ep) > 100:
                    clean_ep = clean_ep[:97] + "..."
                q_lines.append(f"**Q{idx+1}:** {clean_ep}")
            if q_lines:
                embed.add_field(
                    name="🔀 Selected Quadrant Prompts",
                    value="\n".join(q_lines),
                    inline=False
                )
        except Exception:
            pass

    # 3-Column Studio Dashboard
    embed.add_field(
        name="📐 Canvas & Framing",
        value=f"**Size:** `{width}x{height}`\n**Comp:** `{comp_display}`",
        inline=True
    )
    embed.add_field(
        name="🤖 Checkpoint & Tech",
        value=f"**Model:** `{model_display}`\n**Seed:** `{seed}`\n**CFG:** `{cfg:.1f}`",
        inline=True
    )
    extra_meta = []
    if is_magic:
        extra_meta.append("✨ Magic")
    if cref_image_name:
        extra_meta.append(f"👤 Cref ({cref_weight:.2f})")
    extra_str = f"\n**Extra:** {', '.join(extra_meta)}" if extra_meta else ""

    embed.add_field(
        name="🎭 Aesthetics & Identity",
        value=f"**Char:** `{char_display}`\n**Realism:** `{sr_display}`\n**Style:** {style_display}{extra_str}",
        inline=True
    )

    if image_url:
        embed.set_thumbnail(url=image_url)

    timing_str = f" • Rendered in {elapsed_time:.1f}s" if elapsed_time else ""
    embed.set_footer(text=f"Requested by {user_name}{timing_str} • Seed: {seed}")
    return embed


def build_blended_image_embed(
    index: int,
    display_prompt: str,
    checkpoint: str,
    target_seed: int,
    width: int,
    height: int,
    comp_strength: str = "style",
    cfg: float = 4.0,
    sref_info: dict = None,
    char_choice: str = None,
    sr_choice = None,
    user_name: str = "User",
    user_id: int = None,
    image_url: str = None,
    is_blend: bool = True
) -> discord.Embed:
    """Builds a polished 3-column inline studio dashboard embed for an isolated/blended single image."""
    model_friendly_names = {
        "waiIllustriousSDXL_v170.safetensors": "Wai Illustrious SDXL v1.70",
        "illustriousRealismBy_v10VAE.safetensors": "Illustrious Realism V1",
        "RealVisXL_V4.0.safetensors": "RealVisXL V4.0",
        "juggernautXL_ragnarok.safetensors": "Juggernaut XL Ragnarok",
        "CopaxTimeLessXL.safetensors": "Copax Timeless XL",
        "ultraRealisticByStable_v25.safetensors": "UltraRealistic V2.5",
        "hyphoriaIlluNAI_v001.safetensors": "Hyphoria NAI v0.01",
        "novaFurryXL_ilV180A.safetensors": "Nova Furry XL v1.8",
        "wai": "Wai Illustrious SDXL v1.70",
        "illustrious_realism": "Illustrious Realism V1",
        "realvis": "RealVisXL V4.0",
        "juggernaut": "Juggernaut XL Ragnarok",
        "copax": "Copax Timeless XL",
        "ultra": "UltraRealistic V2.5",
        "hyphoria": "Hyphoria NAI v0.01",
        "nova": "Nova Furry XL v1.8",
    }
    model_display = model_friendly_names.get(checkpoint, str(checkpoint).replace(".safetensors", ""))

    comp_map = {
        "style": "🎨 Style Only (0.20)",
        "low": "🖼️ Light Comp (0.35)",
        "med": "🖼️ Med Comp (0.60)",
        "high": "🖼️ High Comp (0.85)",
    }
    comp_display = comp_map.get(comp_strength, comp_strength)

    # Resolve character badge
    char_display = get_character_display_badge(char_choice, architecture="sdxl", prompt=display_prompt)

    # Resolve semi-realism badge
    if not sr_choice or sr_choice in ["nosr", False]:
        m = re.search(r'--sr\.?(\d+)', display_prompt)
        if m:
            sr_display = f"--sr.{m.group(1)}"
        elif "semi-realism" in display_prompt.lower():
            sr_display = "Enabled"
        else:
            sr_display = "OFF"
    else:
        if isinstance(sr_choice, str) and sr_choice.startswith("sr"):
            sr_display = f"--sr.{sr_choice[2:]}"
        else:
            sr_display = "Enabled"

    # Resolve style / sref badge
    if sref_info and isinstance(sref_info, dict) and "code" in sref_info:
        s_name = sref_info.get("name", "")
        style_display = f"`--sref {sref_info['code']}`" + (f" *({s_name})*" if s_name else "")
    else:
        m = re.search(r'--sref\s+(\d+)', display_prompt)
        if m:
            style_display = f"`--sref {m.group(1)}`"
        elif "--sref random" in display_prompt.lower():
            style_display = "🎲 Random Style"
        else:
            style_display = "OFF"

    clean_p = display_prompt.strip()
    if len(clean_p) > 300:
        truncated = clean_p[:297].rsplit(" ", 1)[0] + "..."
    else:
        truncated = clean_p

    title_txt = f"✨ Blended Image {index}" if is_blend else f"🖼️ Isolated Image {index}"
    embed = discord.Embed(
        title=title_txt,
        description=f"**Prompt:**\n> {truncated}",
        color=discord.Color.from_rgb(138, 43, 226) if is_blend else discord.Color.blurple()
    )

    # 3-Column Studio Dashboard
    embed.add_field(
        name="📐 Canvas & Framing",
        value=f"**Size:** `{width}x{height}`\n**Comp:** `{comp_display}`" if is_blend else f"**Size:** `{width}x{height}`",
        inline=True
    )
    embed.add_field(
        name="🤖 Checkpoint & Tech",
        value=f"**Model:** `{model_display}`\n**Seed:** `{target_seed}`",
        inline=True
    )
    embed.add_field(
        name="🎭 Aesthetics & Identity",
        value=f"**Char:** `{char_display}`\n**Realism:** `{sr_display}`\n**Style:** {style_display}",
        inline=True
    )

    if image_url:
        embed.set_thumbnail(url=image_url)

    id_str = f" (ID: {user_id})" if user_id else ""
    embed.set_footer(text=f"Requested by {user_name}{id_str} • Seed: {target_seed}")
    return embed


class BlendButtons(discord.ui.View):
    def __init__(
        self,
        generation_id: str,
        ar: str = "16:9",
        sr = True,
        oga: bool = False,
        model_choice: str = "wai",
        comp_strength: str = "style",
        sref_rand = "nosref",
        char_choice: str = None,
        tab: str = "canvas",
        user_favorites: list = None
    ):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.ar = ar
        self.sr = sr
        self.model_choice = model_choice
        self.comp_strength = comp_strength
        self.sref_rand = sref_rand
        self.tab = tab if tab in ["canvas", "style"] else "canvas"
        self.user_favorites = user_favorites or []

        # Resolve char_choice & oga
        if char_choice is not None:
            self.char_choice = char_choice
            self.oga = (char_choice == "ogarla")
        else:
            self.oga = oga
            self.char_choice = "ogarla" if oga else "none"

        # Resolve sr string
        if self.sr in ["nosr", False]:
            cur_sr = "nosr"
        elif self.sr in ["sr60", "sr.60"]:
            cur_sr = "sr60"
        elif self.sr in ["sr70", "sr.70"]:
            cur_sr = "sr70"
        elif self.sr in ["sr75", "sr.75"]:
            cur_sr = "sr75"
        elif self.sr in ["sr80", "sr.80"]:
            cur_sr = "sr80"
        elif self.sr in ["sr90", "sr.90", "sr", True]:
            cur_sr = "sr90"
        else:
            cur_sr = str(self.sr)

        # Resolve style string
        if isinstance(self.sref_rand, str):
            cur_style = self.sref_rand
        elif self.sref_rand is True:
            cur_style = "sref"
        else:
            cur_style = "nosref"

        # Build dynamic peek summaries for tab buttons
        char_short = {
            "none": "No Char",
            "ogarla": "Ogarla",
            "valerie": "Valerie",
            "sully": "Sully",
            "cheri": "Cheri",
            "cheri_e4": "Cheri E4",
            "mageill": "Mageill",
            "mageill_e6": "Mageill E6",
            "mageill_e4": "Mageill E4",
            "mageill_e3": "Mageill E3",
        }.get(self.char_choice, self.char_choice)
        sr_short = "No SR" if cur_sr == "nosr" else f"--{cur_sr}"
        style_tab_peek = f"🎭 Characters & Styles [{char_short} • {sr_short}] ➡️"

        model_short = {
            "wai": "Wai",
            "illustrious_realism": "Illu Real",
            "realvis": "RealVis",
            "juggernaut": "Juggernaut",
            "copax": "Copax",
            "ultra": "Ultra",
            "hyphoria": "Hyphoria",
            "nova": "Nova",
            "muse": "Muse v3.5",
            "pornmaster": "Pornmaster v2",
        }.get(self.model_choice, self.model_choice)
        canvas_tab_peek = f"📐 Canvas & Model [{self.ar} • {model_short}] ⬅️"

        if self.tab == "canvas":
            # Row 0: Aspect Ratio Selection Dropdown
            ar_options = [
                discord.SelectOption(label="16:9 Landscape (1344x768)", value="16:9", emoji="📐", default=(self.ar == "16:9")),
                discord.SelectOption(label="21:9 Ultra-Wide (1536x640)", value="21:9", emoji="📐", default=(self.ar == "21:9")),
                discord.SelectOption(label="10:7 Standard / Tablet (1216x832)", value="10:7", emoji="📐", default=(self.ar == "10:7")),
                discord.SelectOption(label="1:1 Square (1024x1024)", value="1:1", emoji="📐", default=(self.ar == "1:1")),
                discord.SelectOption(label="3:5 Portrait (832x1216)", value="3:5", emoji="📐", default=(self.ar == "3:5")),
                discord.SelectOption(label="9:16 Tall Portrait (768x1344)", value="9:16", emoji="📐", default=(self.ar == "9:16")),
            ]
            ar_select = discord.ui.Select(
                placeholder="📐 Select Aspect Ratio...",
                options=ar_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_ar:{self.generation_id}",
                row=0
            )
            self.add_item(ar_select)

            # Row 1: Model Checkpoint Dropdown
            model_options = [
                discord.SelectOption(label="Wai Illustrious SDXL v1.70 (Anime / Illustration)", value="wai", emoji="🌸", default=(self.model_choice in ["wai", "default"])),
                discord.SelectOption(label="Illustrious Realism v1.0 (Anime Realism)", value="illustrious_realism", emoji="🎨", default=(self.model_choice == "illustrious_realism")),
                discord.SelectOption(label="RealVisXL V4.0 (Photorealistic)", value="realvis", emoji="📸", default=(self.model_choice == "realvis")),
                discord.SelectOption(label="Juggernaut XL (Balanced Realism)", value="juggernaut", emoji="⚔️", default=(self.model_choice == "juggernaut")),
                discord.SelectOption(label="Copax Timeless XL (Cinematic)", value="copax", emoji="🎬", default=(self.model_choice == "copax")),
                discord.SelectOption(label="Ultra Realistic XL v2.5 (Fine Details)", value="ultra", emoji="✨", default=(self.model_choice == "ultra")),
                discord.SelectOption(label="Hyphoria NAI (Illustrious NAI)", value="hyphoria", emoji="🤖", default=(self.model_choice == "hyphoria")),
                discord.SelectOption(label="Nova Furry (Stylized)", value="nova", emoji="🦊", default=(self.model_choice == "nova")),
                discord.SelectOption(label="Muse v3.5 Extended (Krea 2 Turbo Photorealism)", value="muse", emoji="⚡", description="Stable Yogi flow-matching checkpoint", default=(self.model_choice == "muse")),
                discord.SelectOption(label="Pornmaster v2 (Krea 2 FP8 Photorealism)", value="pornmaster", emoji="⚡", description="Krea 2 Turbo FP8 checkpoint", default=(self.model_choice == "pornmaster")),
            ]
            model_select = discord.ui.Select(
                placeholder="🤖 Select Model Checkpoint...",
                options=model_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_model:{self.generation_id}",
                row=1
            )
            self.add_item(model_select)

            # Row 2: Reference & Composition Strength Dropdown
            comp_options = [
                discord.SelectOption(label="Style Reference Only (0.20 weight)", value="style", emoji="🎨", description="Transfers style/colors without copying layout", default=(self.comp_strength == "style")),
                discord.SelectOption(label="Light Composition (0.35 weight)", value="low", emoji="🖼️", description="Soft pose/layout reference", default=(self.comp_strength == "low")),
                discord.SelectOption(label="Medium Composition (0.60 weight)", value="med", emoji="🖼️", description="Balanced character identity & layout", default=(self.comp_strength == "med")),
                discord.SelectOption(label="Strong Composition (0.85 weight)", value="high", emoji="🖼️", description="Strict pose, framing & structural locking", default=(self.comp_strength == "high")),
            ]
            comp_select = discord.ui.Select(
                placeholder="🖼️ Select Reference & Composition Strength...",
                options=comp_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_comp:{self.generation_id}",
                row=2
            )
            self.add_item(comp_select)

            # Row 3: Tab Switch Button to Characters & Styles (with peek badge)
            self.add_item(discord.ui.Button(
                label=style_tab_peek[:80],
                style=discord.ButtonStyle.primary,
                custom_id=f"switch_blend_tab:{self.generation_id}:style",
                row=3
            ))

        else:
            # tab == "style"
            # Row 0: Character LoRA Dropdown
            char_options = [
                discord.SelectOption(label="None (No Character LoRA)", value="none", emoji="🚫", description="Generate without character presets", default=(self.char_choice == "none")),
                discord.SelectOption(label="Ogarla", value="ogarla", emoji="🌿", description="Original fantasy character LoRA (--ogarla.70)", default=(self.char_choice == "ogarla")),
                discord.SelectOption(label="Valerie", value="valerie", emoji="👩", description="Consistent Valerie character preset (--valerie.85)", default=(self.char_choice == "valerie")),
                discord.SelectOption(label="Sully", value="sully", emoji="👓", description="Black hair & thin-rim glasses (--sully.85)", default=(self.char_choice == "sully")),
                discord.SelectOption(label="Cheri (Epoch 6 - Default)", value="cheri", emoji="🌸", description="Blonde hair signature preset (--cheri.85)", default=(self.char_choice == "cheri")),
                discord.SelectOption(label="Cheri (Epoch 4)", value="cheri_e4", emoji="🌸", description="Cheri Epoch 4 model (--cheri4.85)", default=(self.char_choice == "cheri_e4")),
                discord.SelectOption(label="Mageill (Epoch 5 - Default)", value="mageill", emoji="🔮", description="Original Mageill character preset (--mageill.85)", default=(self.char_choice == "mageill")),
                discord.SelectOption(label="Mageill (Epoch 6)", value="mageill_e6", emoji="🔮", description="Mageill Epoch 6 model (--mageill6.85)", default=(self.char_choice == "mageill_e6")),
                discord.SelectOption(label="Mageill (Epoch 4)", value="mageill_e4", emoji="🔮", description="Mageill Epoch 4 model (--mageill4.85)", default=(self.char_choice == "mageill_e4")),
                discord.SelectOption(label="Mageill (Epoch 3)", value="mageill_e3", emoji="🔮", description="Mageill Epoch 3 model (--mageill3.85)", default=(self.char_choice == "mageill_e3")),
            ]
            char_select = discord.ui.Select(
                placeholder="🎭 Select Character LoRA Preset...",
                options=char_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_char:{self.generation_id}",
                row=0
            )
            self.add_item(char_select)

            # Row 1: Semi-Realism Strength Dropdown
            sr_options = [
                discord.SelectOption(label="Semi-Realism: OFF", value="nosr", emoji="🚫", description="Default anime/model stylization without LoRA", default=(cur_sr == "nosr")),
                discord.SelectOption(label="Light Semi-Realism (--sr.60)", value="sr60", emoji="✨", description="Subtle skin texture and soft shading", default=(cur_sr == "sr60")),
                discord.SelectOption(label="Balanced Semi-Realism (--sr.70)", value="sr70", emoji="✨", description="Balanced anime/realism blend", default=(cur_sr == "sr70")),
                discord.SelectOption(label="Medium-High Semi-Realism (--sr.75)", value="sr75", emoji="✨", description="Enhanced depth and realistic lighting", default=(cur_sr == "sr75")),
                discord.SelectOption(label="High Semi-Realism (--sr.80)", value="sr80", emoji="✨", description="Strong realism pass with micro-details", default=(cur_sr == "sr80")),
                discord.SelectOption(label="Maximum Realism (--sr.90)", value="sr90", emoji="✨", description="Intense photorealistic rendering pass", default=(cur_sr == "sr90")),
            ]
            sr_select = discord.ui.Select(
                placeholder="✨ Select Semi-Realism Strength...",
                options=sr_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_sr:{self.generation_id}",
                row=1
            )
            self.add_item(sr_select)

            # Row 2: Style & Sref Mode Dropdown
            style_options = [
                discord.SelectOption(label="Style Sref: OFF", value="nosref", emoji="🚫", description="No style reference applied", default=(cur_style == "nosref")),
                discord.SelectOption(label="1 Random Style (--sref random)", value="sref", emoji="🎲", description="Inject 1 randomized style code", default=(cur_style in ["sref", "sref1"])),
                discord.SelectOption(label="5 Styles Batch (--sref batch:5)", value="sref5", emoji="🎲", description="Queues 5 variants with random & saved styles", default=(cur_style == "sref5")),
                discord.SelectOption(label="10 Styles Batch (--sref batch:10)", value="sref10", emoji="🎲", description="Queues 10 variants with random & saved styles", default=(cur_style == "sref10")),
                discord.SelectOption(label="15 Styles Batch (--sref batch:15)", value="sref15", emoji="🎲", description="Queues 15 variants with random & saved styles", default=(cur_style == "sref15")),
                discord.SelectOption(label="Junji Ito (Horror Manga Ink)", value="preset_junji_ito", emoji="🖋️", description="Clinical cross-hatching, high-contrast ink, spirals", default=(cur_style == "preset_junji_ito")),
                discord.SelectOption(label="Martine Johanna (Pastel Surreal)", value="preset_martine_johanna", emoji="🎨", description="Prismatic pastel strokes, dreamy gaze", default=(cur_style == "preset_martine_johanna")),
                discord.SelectOption(label="Dark Fantasy Landscape", value="preset_dark_fantasy_landscape", emoji="🏰", description="Moody gothic spires, volumetric fog", default=(cur_style == "preset_dark_fantasy_landscape")),
                discord.SelectOption(label="Cyberpunk Cityscape", value="preset_cyberpunk_cityscape", emoji="🌆", description="Neon monoliths, rain-slicked futuristic streets", default=(cur_style == "preset_cyberpunk_cityscape")),
                discord.SelectOption(label="Ethereal Fine Art Portrait", value="preset_ethereal_portrait", emoji="✨", description="Delicate soft focus, gentle pastel tones", default=(cur_style == "preset_ethereal_portrait")),
            ]
            # Dynamic user favorite styles from /styles (up to 4)
            if self.user_favorites:
                for fav in self.user_favorites[:4]:
                    code = str(fav.get("style_code") or fav.get("code") or "")
                    name = fav.get("style_name") or fav.get("name") or f"Style {code}"
                    if code:
                        val = f"saved_{code}"
                        style_options.append(
                            discord.SelectOption(
                                label=f"⭐ {name[:60]}",
                                value=val,
                                description=f"--sref {code}",
                                default=(cur_style == val)
                            )
                        )

            style_select = discord.ui.Select(
                placeholder="🎨 Select Style & Sref Mode...",
                options=style_options,
                min_values=1,
                max_values=1,
                custom_id=f"set_blend_style:{self.generation_id}",
                row=2
            )
            self.add_item(style_select)

            # Row 3: Tab Switch Button to Canvas & Model (with peek badge)
            self.add_item(discord.ui.Button(
                label=canvas_tab_peek[:80],
                style=discord.ButtonStyle.secondary,
                custom_id=f"switch_blend_tab:{self.generation_id}:canvas",
                row=3
            ))

        # Row 4: Action Launchers (always accessible from both tabs!)
        self.add_item(discord.ui.Button(
            label="✏️ Edit Prompts",
            style=discord.ButtonStyle.secondary,
            custom_id=f"edit_blend_prompt:{self.generation_id}",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="🏷️ Blend Tags",
            style=discord.ButtonStyle.success,
            custom_id=f"blend_desc:{self.generation_id}:caption",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="✨ Blend Scene",
            style=discord.ButtonStyle.primary,
            custom_id=f"blend_desc:{self.generation_id}:detailed",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="⚡ Blend in Krea 2",
            style=discord.ButtonStyle.danger,
            custom_id=f"blend_desc:{self.generation_id}:krea2",
            row=4
        ))


class EditBlendPromptModal(discord.ui.Modal, title="✏️ Edit & Refine Blend Prompts"):
    def __init__(self, generation_id: str, current_caption: str, current_detailed: str, current_extra: str = "", on_submit_callback=None):
        super().__init__()
        self.generation_id = generation_id
        self.on_submit_callback = on_submit_callback

        self.caption_input = discord.ui.TextInput(
            label="Caption (Short / Tags)",
            style=discord.TextStyle.paragraph,
            default=current_caption[:400] if current_caption else "",
            max_length=1000,
            required=False,
            placeholder="e.g. 1girl, solo, palshallot, adventurer outfit, looking at viewer"
        )
        self.add_item(self.caption_input)

        self.detailed_input = discord.ui.TextInput(
            label="Detailed Vision Description",
            style=discord.TextStyle.paragraph,
            default=current_detailed[:1500] if current_detailed else "",
            max_length=2000,
            required=False,
            placeholder="Edit or expand the scene description generated by Florence-2"
        )
        self.add_item(self.detailed_input)

        self.extra_input = discord.ui.TextInput(
            label="✨ Extra Details / Add-ons (Optional)",
            style=discord.TextStyle.short,
            default=current_extra[:200] if current_extra else "",
            max_length=400,
            required=False,
            placeholder="e.g. glowing magic aura, sunset rim light, cinematic lens flare"
        )
        self.add_item(self.extra_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            if self.on_submit_callback:
                await self.on_submit_callback(
                    interaction,
                    self.generation_id,
                    self.caption_input.value.strip(),
                    self.detailed_input.value.strip(),
                    self.extra_input.value.strip()
                )
        except Exception as e:
            logger.error(f"Error in EditBlendPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to update blend prompt: {e}")


def build_blend_krea_embed(gen_data: dict, author_str: str = "User", image_url: str = None) -> discord.Embed:
    """Builds a streamlined, photorealism-focused embed for /blend-krea sessions."""
    vision_prompt = gen_data.get("krea2_prompt") or gen_data.get("detailed_caption", "No description")
    user_prompt = gen_data.get("user_prompt", "").strip()
    fused_prompt = gen_data.get("fused_prompt") or vision_prompt
    ar = gen_data.get("ar", "16:9")
    model_choice = gen_data.get("model_choice", "muse")
    wetness = float(gen_data.get("wetness", -2.0))
    comp = gen_data.get("composition", "off")
    char_choice = gen_data.get("char_choice", "none")
    celeb_choice = gen_data.get("celeb_choice", "none")

    model_display = "Muse v3.5 Extended" if "muse" in model_choice.lower() else "Pornmaster v2 (FP8)"
    if wetness == -2.0:
        skin_display = "Matte Pores (-2.0)"
    elif wetness == 0.0:
        skin_display = "Natural Baseline (0.0)"
    elif wetness == 1.0:
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
    if image_url:
        embed.set_thumbnail(url=image_url)

    disp_prompt = fused_prompt[:1020] + "..." if len(fused_prompt) > 1024 else fused_prompt
    embed.add_field(name="📜 Generation Prompt", value=disp_prompt, inline=False)

    settings_lines = [
        f"📐 **Ratio:** `{ar}` • 🤖 **Engine:** `{model_display}` • 💧 **Skin:** `{skin_display}`",
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


class EditBlendKreaModal(discord.ui.Modal):
    def __init__(self, generation_id: str, current_prompt: str = "", on_submit_callback=None):
        super().__init__(title="✏️ Edit Generation Prompt")
        self.generation_id = generation_id
        self.on_submit_callback = on_submit_callback

        self.prompt_input = discord.ui.TextInput(
            label="Generation Prompt",
            style=discord.TextStyle.paragraph,
            default=current_prompt[:2000] if current_prompt else "",
            placeholder="Edit, add prefixes, or rewrite the generation prompt for Krea 2...",
            required=True,
            max_length=2000
        )
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            val = self.prompt_input.value.strip()
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, self.generation_id, val)
        except Exception as e:
            logger.error(f"Error in EditBlendKreaModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to update Krea 2 prompt: {e}")


class BlendKreaButtons(discord.ui.View):
    def __init__(self, generation_id: str, ar: str = "16:9", model_choice: str = "muse", wetness: float = -2.0, composition: str = "off", character: str = "none", celebrity: str = "none"):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.ar = ar
        self.model_choice = model_choice
        self.wetness = wetness
        self.composition = composition or "off"
        self.character = character or "none"
        self.celebrity = celebrity or "none"

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
            discord.SelectOption(
                label="Valerie (Krea 2 - 0.90)",
                value="valerie.90",
                emoji="✨",
                description="Valerie Krea 2 Character LoRA (Brunette, Brown Eyes)",
                default=(self.character in ["valerie", "valerie.90", "val"])
            ),
            discord.SelectOption(
                label="Valerie (Krea 2 Light - 0.70)",
                value="valerie.70",
                emoji="✨",
                description="Subtle Valerie Krea 2 Character LoRA",
                default=(self.character in ["valerie.70", "valerie_light"])
            ),
        ]
        self.add_item(discord.ui.Select(
            placeholder="🎭 Select Character LoRA (Ogarla / Valerie)...",
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

        if self.wetness == -2.0:
            wet_label = "💧 Skin: Matte"
            wet_style = discord.ButtonStyle.primary
        elif self.wetness == 0.0:
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


class StasisControlsView(discord.ui.View):
    def __init__(self, generation_id: str, user_id: int):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.user_id = user_id

        self.add_item(discord.ui.Button(
            label="⏸️ Pause / Stasis",
            style=discord.ButtonStyle.secondary,
            custom_id=f"stasis_pause:{self.generation_id}:{self.user_id}"
        ))

class StasisPausedView(discord.ui.View):
    def __init__(self, generation_id: str, user_id: int):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.user_id = user_id

        self.add_item(discord.ui.Button(
            label="▶️ Resume",
            style=discord.ButtonStyle.primary,
            custom_id=f"stasis_resume:{self.generation_id}:{self.user_id}"
        ))


class StudyImagineModal(discord.ui.Modal, title="🎨 Imagine with Extracted Prompt"):
    prompt_input = discord.ui.TextInput(
        label="Prompt",
        style=discord.TextStyle.paragraph,
        placeholder="Enter or edit your prompt...",
        required=True,
        max_length=4000
    )
    flags_input = discord.ui.TextInput(
        label="Optional Flags (--ar, --sr, --sref, etc.)",
        style=discord.TextStyle.short,
        placeholder="e.g. --ar 16:9 --sr.60 --sref 772382",
        required=False,
        max_length=500
    )

    def __init__(self, initial_prompt: str, on_submit_callback=None):
        super().__init__()
        self.prompt_input.default = initial_prompt[:4000]
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            full_prompt = self.prompt_input.value.strip()
            flags = self.flags_input.value.strip()
            if flags:
                full_prompt = f"{full_prompt} {flags}"
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, full_prompt)
        except Exception as e:
            logger.error(f"Error in StudyImagineModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to queue generation: {e}")



class StudyButtons(discord.ui.View):
    def __init__(self, prompt: str, imagine_callback=None):
        super().__init__(timeout=None)
        self.prompt = prompt
        self.imagine_callback = imagine_callback

    @discord.ui.button(label="🎨 Imagine", style=discord.ButtonStyle.primary, custom_id="study_imagine_btn")
    async def imagine_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        modal = StudyImagineModal(initial_prompt=self.prompt, on_submit_callback=self.imagine_callback)
        await interaction.response.send_modal(modal)

    @discord.ui.button(label="📋 Copy /imagine", style=discord.ButtonStyle.secondary, custom_id="study_copy_btn")
    async def copy_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        cmd_text = f"/imagine prompt: {self.prompt}"
        if len(cmd_text) > 1950:
            cmd_text = cmd_text[:1950] + "..."
        await interaction.response.send_message(
            content=f"Copy this into your chat bar to toggle settings:\n```{cmd_text}```",
            ephemeral=True
        )

    @discord.ui.button(label="⭐ Save Prompt", style=discord.ButtonStyle.success, custom_id="study_fav_btn")
    async def fav_button(self, interaction: discord.Interaction, button: discord.ui.Button):
        import db
        short_name = self.prompt[:30].strip() + ("..." if len(self.prompt) > 30 else "")
        db.add_favorite_prompt(interaction.user.id, short_name, self.prompt)
        await interaction.response.send_message(f"⭐ Saved prompt to your favorites (`/my_prompts`)!", ephemeral=True)


class EditStyleModal(discord.ui.Modal, title="✏️ Edit Style Reference"):
    def __init__(self, user_id: int, style: dict, on_save_callback=None):
        super().__init__()
        self.user_id = user_id
        self.style_code = style["style_code"]
        self.on_save_callback = on_save_callback

        self.name_input = discord.ui.TextInput(
            label="Style Name",
            placeholder="e.g. Gothic Watercolor Wash",
            default=style.get("style_name", "")[:100],
            required=True,
            max_length=100
        )
        self.prompt_input = discord.ui.TextInput(
            label="Style Prompt",
            style=discord.TextStyle.paragraph,
            placeholder="Enter style prompt description...",
            default=style.get("style_prompt", "")[:2000],
            required=True,
            max_length=2000
        )
        self.add_item(self.name_input)
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            import db
            new_name = self.name_input.value.strip()
            new_prompt = self.prompt_input.value.strip()
            
            updated = db.update_favorite_style(self.user_id, self.style_code, new_name, new_prompt)
            if updated:
                if self.on_save_callback:
                    await self.on_save_callback(interaction, self.style_code, new_name, new_prompt)
                else:
                    await interaction.response.send_message(f"✅ Updated style `{new_name}` (Code: `{self.style_code}`).", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Failed to update style.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in EditStyleModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit style: {e}")



class StylePaginationView(discord.ui.View):
    def __init__(self, user_id: int, favorites: list[dict], per_page: int = 8):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.favorites = favorites
        self.per_page = per_page
        self.current_page = 0
        self.selected_code = None

        self._update_components()

    @property
    def total_pages(self) -> int:
        import math
        return max(1, math.ceil(len(self.favorites) / self.per_page))

    def _update_components(self):
        self.clear_items()
        
        if not self.favorites:
            return

        # Ensure page is within bounds
        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]

        # Check if selected_code is in page_items, else pick first
        code_in_page = any(f["style_code"] == self.selected_code for f in page_items)
        if not code_in_page and page_items:
            self.selected_code = page_items[0]["style_code"]

        # 1. Select menu for current page styles
        options = []
        for fav in page_items:
            code_str = str(fav["style_code"])
            is_default = (self.selected_code == fav["style_code"])
            options.append(discord.SelectOption(
                label=f"{fav['style_name']}"[:100],
                value=code_str,
                description=f"Code: {fav['style_code']}",
                default=is_default
            ))

        select = discord.ui.Select(
            placeholder="Select a style to edit or delete...",
            min_values=1,
            max_values=1,
            options=options,
            row=0
        )
        select.callback = self.on_select_style
        self.add_item(select)

        # 2. Navigation buttons
        prev_btn = discord.ui.Button(
            label="◀ Prev",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page == 0),
            row=1
        )
        prev_btn.callback = self.on_prev_page
        self.add_item(prev_btn)

        next_btn = discord.ui.Button(
            label="Next ▶",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page >= self.total_pages - 1),
            row=1
        )
        next_btn.callback = self.on_next_page
        self.add_item(next_btn)

        # 3. Action buttons (Edit, Delete)
        edit_btn = discord.ui.Button(
            label="✏️ Edit Style",
            style=discord.ButtonStyle.primary,
            row=1
        )
        edit_btn.callback = self.on_edit_style
        self.add_item(edit_btn)

        delete_btn = discord.ui.Button(
            label="🗑️ Delete Style",
            style=discord.ButtonStyle.danger,
            row=1
        )
        delete_btn.callback = self.on_delete_style
        self.add_item(delete_btn)

    def build_embed(self) -> discord.Embed:
        if not self.favorites:
            return discord.Embed(
                title="⭐ Your Favorite Styles",
                description="You have no saved style references yet. Click `⭐ Favorite Style` on any completed image grid to save styles!"
            )

        embed = discord.Embed(
            title="⭐ Your Favorite Styles",
            description="Select a style name or code using the `favorite_style` parameter when running `/imagine`."
        )

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        for fav in page_items:
            s_name = fav['style_name']
            if len(s_name) > 180:
                s_name = s_name[:177] + "..."
            s_prompt = fav['style_prompt']
            if len(s_prompt) > 950:
                s_prompt = s_prompt[:950] + "..."

            field_name = f"✨ {s_name} (Code: `{fav['style_code']}`)"
            if len(field_name) > 250:
                field_name = field_name[:245] + "...)"

            embed.add_field(
                name=field_name,
                value=f"**Prompt:** *{s_prompt}*",
                inline=False
            )

        embed.set_footer(text=f"Page {self.current_page + 1} of {self.total_pages} ({len(self.favorites)} total saved styles)")
        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This menu is for the command caller only.", ephemeral=True)
            return False
        return True

    async def on_select_style(self, interaction: discord.Interaction):
        selected_str = interaction.data["values"][0]
        self.selected_code = int(selected_str)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_prev_page(self, interaction: discord.Interaction):
        if self.current_page > 0:
            self.current_page -= 1
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_next_page(self, interaction: discord.Interaction):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_edit_style(self, interaction: discord.Interaction):
        selected = next((f for f in self.favorites if f["style_code"] == self.selected_code), None)
        if not selected:
            await interaction.response.send_message("Please select a style first.", ephemeral=True)
            return

        modal = EditStyleModal(self.user_id, selected, on_save_callback=self._handle_style_edited)
        await interaction.response.send_modal(modal)

    async def _handle_style_edited(self, interaction: discord.Interaction, style_code: int, new_name: str, new_prompt: str):
        import db
        self.favorites = db.get_favorite_styles(self.user_id)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_delete_style(self, interaction: discord.Interaction):
        import db
        if not self.selected_code:
            await interaction.response.send_message("Please select a style first.", ephemeral=True)
            return

        db.remove_favorite_style(self.user_id, self.selected_code)
        self.favorites = db.get_favorite_styles(self.user_id)
        
        if not self.favorites:
            self.clear_items()
            await interaction.response.edit_message(embed=self.build_embed(), view=None)
            return

        page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        if page_items:
            self.selected_code = page_items[0]["style_code"]
        else:
            self.current_page = max(0, self.total_pages - 1)
            page_items = self.favorites[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_code = page_items[0]["style_code"]

        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)


class EditPromptModal(discord.ui.Modal, title="✏️ Edit Saved Prompt"):
    def __init__(self, user_id: int, prompt_data: dict, on_save_callback=None):
        super().__init__()
        self.user_id = user_id
        self.prompt_id = prompt_data["id"]
        self.on_save_callback = on_save_callback

        self.name_input = discord.ui.TextInput(
            label="Prompt Label / Nickname",
            placeholder="e.g. Semi-realism Girl",
            default=prompt_data.get("prompt_name", "")[:100],
            required=True,
            max_length=100
        )
        self.prompt_input = discord.ui.TextInput(
            label="Full Prompt Text",
            style=discord.TextStyle.paragraph,
            placeholder="Enter prompt text...",
            default=prompt_data.get("prompt_text", "")[:4000],
            required=True,
            max_length=4000
        )
        self.add_item(self.name_input)
        self.add_item(self.prompt_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            import db
            new_name = self.name_input.value.strip()
            new_text = self.prompt_input.value.strip()
            
            updated = db.update_favorite_prompt(self.user_id, self.prompt_id, new_name, new_text)
            if updated:
                if self.on_save_callback:
                    await self.on_save_callback(interaction, self.prompt_id, new_name, new_text)
                else:
                    await interaction.response.send_message(f"✅ Updated prompt **\"{new_name}\"** (ID: `{self.prompt_id}`).", ephemeral=True)
            else:
                await interaction.response.send_message("❌ Failed to update prompt.", ephemeral=True)
        except Exception as e:
            logger.error(f"Error in EditPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit prompt: {e}")



class PromptPaginationView(discord.ui.View):
    def __init__(self, user_id: int, prompts: list[dict], per_page: int = 5, imagine_callback=None, bertflow_callback=None):
        super().__init__(timeout=180)
        self.user_id = user_id
        self.prompts = prompts
        self.per_page = per_page
        self.imagine_callback = imagine_callback
        self.bertflow_callback = bertflow_callback
        self.current_page = 0
        self.selected_id = None

        self._update_components()

    @property
    def total_pages(self) -> int:
        import math
        return max(1, math.ceil(len(self.prompts) / self.per_page))

    def _update_components(self):
        self.clear_items()
        
        if not self.prompts:
            return

        if self.current_page >= self.total_pages:
            self.current_page = max(0, self.total_pages - 1)

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]

        id_in_page = any(p["id"] == self.selected_id for p in page_items)
        if not id_in_page and page_items:
            self.selected_id = page_items[0]["id"]

        # 1. Select Menu
        options = []
        for p in page_items:
            id_str = str(p["id"])
            is_default = (self.selected_id == p["id"])
            options.append(discord.SelectOption(
                label=f"{p['prompt_name']}"[:100],
                value=id_str,
                description=f"ID: {p['id']}",
                default=is_default
            ))

        select = discord.ui.Select(
            placeholder="Select a prompt to copy, imagine, edit, or delete...",
            min_values=1,
            max_values=1,
            options=options,
            row=0
        )
        select.callback = self.on_select_prompt
        self.add_item(select)

        # 2. Navigation buttons
        prev_btn = discord.ui.Button(
            label="◀ Prev",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page == 0),
            row=1
        )
        prev_btn.callback = self.on_prev_page
        self.add_item(prev_btn)

        next_btn = discord.ui.Button(
            label="Next ▶",
            style=discord.ButtonStyle.secondary,
            disabled=(self.current_page >= self.total_pages - 1),
            row=1
        )
        next_btn.callback = self.on_next_page
        self.add_item(next_btn)

        # 3. Action buttons
        copy_btn = discord.ui.Button(
            label="📋 Copy Full",
            style=discord.ButtonStyle.secondary,
            row=1
        )
        copy_btn.callback = self.on_copy_prompt
        self.add_item(copy_btn)

        imagine_btn = discord.ui.Button(
            label="🎨 Imagine",
            style=discord.ButtonStyle.primary,
            row=1
        )
        imagine_btn.callback = self.on_imagine_prompt
        self.add_item(imagine_btn)

        if self.bertflow_callback:
            bertflow_btn = discord.ui.Button(
                label="⚡ Bertflow",
                style=discord.ButtonStyle.success,
                row=1
            )
            bertflow_btn.callback = self.on_bertflow_prompt
            self.add_item(bertflow_btn)

        edit_btn = discord.ui.Button(
            label="✏️ Edit",
            style=discord.ButtonStyle.primary,
            row=2
        )
        edit_btn.callback = self.on_edit_prompt
        self.add_item(edit_btn)

        delete_btn = discord.ui.Button(
            label="🗑️ Delete",
            style=discord.ButtonStyle.danger,
            row=2
        )
        delete_btn.callback = self.on_delete_prompt
        self.add_item(delete_btn)

    def build_embed(self) -> discord.Embed:
        if not self.prompts:
            return discord.Embed(
                title="⭐ Your Favorite Prompts",
                description="You have no saved favorite prompts yet! Click **⭐ Favorite Prompt** on any completed image grid to save prompts.",
                color=discord.Color.gold()
            )

        embed = discord.Embed(
            title="⭐ Your Favorite Prompts",
            description=f"You have **{len(self.prompts)}** saved prompt(s). Select a prompt to copy full text, generate, edit, or delete:",
            color=discord.Color.gold()
        )

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        for p in page_items:
            p_name = p['prompt_name']
            if len(p_name) > 180:
                p_name = p_name[:177] + "..."

            p_text = p['prompt_text']
            if len(p_text) > 950:
                display_text = p_text[:950] + "..."
            else:
                display_text = p_text

            field_name = f"📌 {p_name} (ID: {p['id']})"
            if len(field_name) > 250:
                field_name = field_name[:245] + "...)"

            embed.add_field(
                name=field_name,
                value=f"```\n{display_text}\n```",
                inline=False
            )

        embed.set_footer(text=f"Page {self.current_page + 1} of {self.total_pages} ({len(self.prompts)} total saved prompts)")
        return embed

    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.user_id:
            await interaction.response.send_message("This menu is for the command caller only.", ephemeral=True)
            return False
        return True

    async def on_select_prompt(self, interaction: discord.Interaction):
        selected_str = interaction.data["values"][0]
        self.selected_id = int(selected_str)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_prev_page(self, interaction: discord.Interaction):
        if self.current_page > 0:
            self.current_page -= 1
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_next_page(self, interaction: discord.Interaction):
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]
            self._update_components()
            await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_copy_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        full_text = selected["prompt_text"]
        header = f"📋 **Full Prompt for \"{selected['prompt_name']}\" (ID: {selected['id']}):**\n"
        
        # If full text fits in a single message codeblock
        if len(header) + len(full_text) + 10 <= 1980:
            content = f"{header}```\n{full_text}\n```"
            await interaction.response.send_message(content=content, ephemeral=True)
        else:
            await interaction.response.send_message(content=f"{header}*(Prompt split across messages due to length)*:", ephemeral=True)
            # Chunk long prompts into 1900 character blocks
            chunk_size = 1900
            for i in range(0, len(full_text), chunk_size):
                chunk = full_text[i:i+chunk_size]
                await interaction.followup.send(content=f"```\n{chunk}\n```", ephemeral=True)

    async def on_imagine_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = StudyImagineModal(initial_prompt=selected["prompt_text"], on_submit_callback=self.imagine_callback)
        await interaction.response.send_modal(modal)

    async def on_bertflow_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = StudyImagineModal(initial_prompt=selected["prompt_text"], on_submit_callback=self.bertflow_callback)
        await interaction.response.send_modal(modal)

    async def on_edit_prompt(self, interaction: discord.Interaction):
        selected = next((p for p in self.prompts if p["id"] == self.selected_id), None)
        if not selected:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        modal = EditPromptModal(self.user_id, selected, on_save_callback=self._handle_prompt_edited)
        await interaction.response.send_modal(modal)

    async def _handle_prompt_edited(self, interaction: discord.Interaction, prompt_id: int, new_name: str, new_text: str):
        import db
        self.prompts = db.get_favorite_prompts(self.user_id)
        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)

    async def on_delete_prompt(self, interaction: discord.Interaction):
        import db
        if not self.selected_id:
            await interaction.response.send_message("Please select a prompt first.", ephemeral=True)
            return

        db.remove_favorite_prompt(self.user_id, self.selected_id)
        self.prompts = db.get_favorite_prompts(self.user_id)
        
        if not self.prompts:
            self.clear_items()
            await interaction.response.edit_message(embed=self.build_embed(), view=None)
            return

        page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
        if page_items:
            self.selected_id = page_items[0]["id"]
        else:
            self.current_page = max(0, self.total_pages - 1)
            page_items = self.prompts[self.current_page * self.per_page : (self.current_page + 1) * self.per_page]
            if page_items:
                self.selected_id = page_items[0]["id"]

        self._update_components()
        await interaction.response.edit_message(embed=self.build_embed(), view=self)


class EditAdoptPromptModal(discord.ui.Modal, title="✏️ Edit Adopted Prompt"):
    prompt_input = discord.ui.TextInput(
        label="Prompt Text",
        style=discord.TextStyle.paragraph,
        placeholder="Edit your prompt...",
        required=True,
        max_length=4000
    )

    def __init__(self, adopt_id: str, current_prompt: str, on_submit_callback=None):
        super().__init__()
        self.adopt_id = adopt_id
        self.prompt_input.default = current_prompt[:4000]
        self.on_submit_callback = on_submit_callback

    async def on_submit(self, interaction: discord.Interaction):
        try:
            new_prompt = self.prompt_input.value.strip()
            if self.on_submit_callback:
                await self.on_submit_callback(interaction, self.adopt_id, new_prompt)
        except Exception as e:
            logger.error(f"Error in EditAdoptPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to edit adopted prompt: {e}")



class AdoptButtons(discord.ui.View):
    def __init__(self, adopt_id: str, ogarla_on: bool = False, cref_weight: float = 0.20, semi_realism_weight: float = 0.0, random_sref_on: bool = False):
        super().__init__(timeout=None) # Persistent buttons
        self.adopt_id = adopt_id
        self.ogarla_on = ogarla_on
        self.cref_weight = cref_weight
        self.semi_realism_weight = float(semi_realism_weight) if isinstance(semi_realism_weight, (int, float, str)) else 0.0
        self.random_sref_on = random_sref_on

        # Row 0: Prompt Utilities
        self.add_item(discord.ui.Button(
            label="✏️ Edit Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_edit_prompt:{adopt_id}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="📋 Copy Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_copy:{adopt_id}",
            row=0
        ))
        self.add_item(discord.ui.Button(
            label="⭐ Save Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_save:{adopt_id}",
            row=0
        ))

        # Row 1: Character, Style & Reference Weight Controls
        oga_label = "👩 Ogarla Main: ON" if ogarla_on else "👩 Ogarla Main: OFF"
        oga_style = discord.ButtonStyle.primary if ogarla_on else discord.ButtonStyle.secondary
        self.add_item(discord.ui.Button(
            label=oga_label,
            style=oga_style,
            custom_id=f"adopt_toggle_oga:{adopt_id}",
            row=1
        ))

        if self.semi_realism_weight > 0.0:
            sr_label = f"🌟 Semi-Realism: {self.semi_realism_weight:.2f}"
            sr_style = discord.ButtonStyle.primary
        else:
            sr_label = "🌟 Semi-Realism: OFF"
            sr_style = discord.ButtonStyle.secondary

        self.add_item(discord.ui.Button(
            label=sr_label,
            style=sr_style,
            custom_id=f"adopt_toggle_sr:{adopt_id}",
            row=1
        ))

        rnd_label = "🎲 Random Style: ON" if random_sref_on else "🎲 Random Style: OFF"
        rnd_style = discord.ButtonStyle.primary if random_sref_on else discord.ButtonStyle.secondary
        self.add_item(discord.ui.Button(
            label=rnd_label,
            style=rnd_style,
            custom_id=f"adopt_toggle_sref:{adopt_id}",
            row=1
        ))

        cw_label = f"🎚️ Ref Weight: {cref_weight:.2f}"
        self.add_item(discord.ui.Button(
            label=cw_label,
            style=discord.ButtonStyle.secondary,
            custom_id=f"adopt_cycle_cw:{adopt_id}",
            row=1
        ))

        # Row 2: Action Launchers (Green buttons at bottom)
        self.add_item(discord.ui.Button(
            label="🎨 Imagine Grid",
            style=discord.ButtonStyle.success,
            custom_id=f"adopt_imagine:{adopt_id}",
            row=2
        ))
        self.add_item(discord.ui.Button(
            label="✨ Flux HD",
            style=discord.ButtonStyle.success,
            custom_id=f"adopt_flux:{adopt_id}",
            row=2
        ))



class VideoPromptModal(discord.ui.Modal, title="🎬 Animate Image to Video"):
    """Modal allowing the user to configure motion prompt and Wan 2.2 settings before queuing video generation."""
    def __init__(self, default_prompt: str = "", default_duration: str = "10", default_smoothness: str = "smooth", on_submit_callback=None):
        super().__init__()
        self.on_submit_callback = on_submit_callback

        self.prompt_input = discord.ui.TextInput(
            label="Motion Prompt",
            style=discord.TextStyle.paragraph,
            placeholder="Describe desired motion or use flags (e.g. hair flowing, --zoom-in, --cinematic)",
            default=default_prompt[:800] if default_prompt else "",
            max_length=1000,
            required=True
        )
        self.add_item(self.prompt_input)

        self.duration_input = discord.ui.TextInput(
            label="Duration in Seconds (10 or 5)",
            style=discord.TextStyle.short,
            placeholder="10 (default) or 5",
            default=str(default_duration) if default_duration else "10",
            max_length=2,
            required=False
        )
        self.add_item(self.duration_input)

        self.smoothness_input = discord.ui.TextInput(
            label="Smoothness Mode (smooth / fast)",
            style=discord.TextStyle.short,
            placeholder="smooth (32 FPS) or fast (16 FPS)",
            default=str(default_smoothness) if default_smoothness else "smooth",
            max_length=10,
            required=False
        )
        self.add_item(self.smoothness_input)

        self.seed_input = discord.ui.TextInput(
            label="Seed (Optional)",
            style=discord.TextStyle.short,
            placeholder="Leave empty for random seed",
            default="",
            max_length=20,
            required=False
        )
        self.add_item(self.seed_input)

    async def on_submit(self, interaction: discord.Interaction):
        try:
            if self.on_submit_callback:
                await self.on_submit_callback(
                    interaction,
                    self.prompt_input.value.strip(),
                    self.duration_input.value.strip(),
                    self.smoothness_input.value.strip(),
                    self.seed_input.value.strip()
                )
        except Exception as e:
            logger.error(f"Error in VideoPromptModal submit: {e}")
            await send_error_fallback(interaction, f"Failed to queue video animation: {e}")


def build_video_complete_embed(
    prompt: str,
    duration_sec: float,
    total_output_frames: int,
    out_fps: int,
    orig_w: int,
    orig_h: int,
    width: int,
    height: int,
    video_seed: int,
    elapsed_time: float,
    init_sec: float = 0.0,
    sample_sec: float = 0.0,
    post_sec: float = 0.0,
    motion_badges: list = None,
    smoothness: str = "smooth",
    user_name: str = "User",
    user_id: int = 0
) -> discord.Embed:
    """Builds a polished, 3-column inline studio dashboard embed for completed Wan 2.2 video animations."""
    badges_str = " • ".join(motion_badges) if motion_badges else "🎬 Natural Motion"
    mode_str = "Smooth (32 FPS • RIFE 2x)" if smoothness == "smooth" else "Fast (16 FPS • Native)"

    embed = discord.Embed(
        title="🎬 Wan 2.2 Studio Video Generation Complete",
        color=discord.Color.from_rgb(88, 101, 242)
    )

    # Column 1: Motion & Camera
    col1_val = (
        f"**Prompt:** {prompt[:180]}{'...' if len(prompt) > 180 else ''}\n"
        f"**Cues:** `{badges_str}`\n"
        f"**Framing:** `{orig_w}x{orig_h}` → `{width}x{height}`"
    )
    embed.add_field(name="🎬 Motion & Camera", value=col1_val, inline=True)

    # Column 2: Video Specs
    col2_val = (
        f"**Duration:** `{duration_sec:.1f}s`\n"
        f"**Framerate:** `{out_fps} FPS`\n"
        f"**Frames:** `{total_output_frames} frames`\n"
        f"**Mode:** `{mode_str}`"
    )
    embed.add_field(name="⏱️ Video Specs", value=col2_val, inline=True)

    # Column 3: Engine & Render
    col3_val = (
        f"**Model:** `Wan 2.2 14B GGUF`\n"
        f"**Sampling:** `6 Steps (Shift 8.0)`\n"
        f"**Render Time:** `{elapsed_time:.1f}s`\n"
        f"**Seed:** `{video_seed}`"
    )
    embed.add_field(name="⚡ Engine & Render", value=col3_val, inline=True)

    user_info = f"Requested by {user_name}" if user_name else "Requested"
    id_info = f" (ID: {user_id})" if user_id else ""
    embed.set_footer(text=f"{user_info}{id_info} • Sample: {sample_sec:.1f}s | Total: {elapsed_time:.1f}s")

    return embed


class VideoActionView(discord.ui.View):
    """Interactive action view attached to completed /video generations."""
    def __init__(
        self,
        generation_id: str,
        on_reroll_cb=None,
        on_remix_cb=None,
        on_toggle_fps_cb=None,
        smoothness: str = "smooth"
    ):
        super().__init__(timeout=1800)
        self.generation_id = generation_id
        self.on_reroll_cb = on_reroll_cb
        self.on_remix_cb = on_remix_cb
        self.on_toggle_fps_cb = on_toggle_fps_cb
        self.smoothness = smoothness

        # [ 🔄 Re-roll ]
        self.reroll_btn = discord.ui.Button(
            label="🔄 Re-roll",
            style=discord.ButtonStyle.primary,
            custom_id=f"video_reroll:{generation_id}"
        )
        self.reroll_btn.callback = self._on_reroll
        self.add_item(self.reroll_btn)

        # [ ✏️ Remix Motion ]
        self.remix_btn = discord.ui.Button(
            label="✏️ Remix Motion",
            style=discord.ButtonStyle.secondary,
            custom_id=f"video_remix:{generation_id}"
        )
        self.remix_btn.callback = self._on_remix
        self.add_item(self.remix_btn)

        # [ ⚡ Switch FPS Mode ]
        toggle_label = "⚡ Switch to Fast (16 FPS)" if smoothness == "smooth" else "🎬 Switch to Smooth (32 FPS)"
        self.toggle_btn = discord.ui.Button(
            label=toggle_label,
            style=discord.ButtonStyle.secondary,
            custom_id=f"video_toggle_fps:{generation_id}"
        )
        self.toggle_btn.callback = self._on_toggle_fps
        self.add_item(self.toggle_btn)

    async def _on_reroll(self, interaction: discord.Interaction):
        try:
            if self.on_reroll_cb:
                await self.on_reroll_cb(interaction, self.generation_id)
        except Exception as e:
            logger.error(f"Error in VideoActionView reroll: {e}")
            await send_error_fallback(interaction, f"Failed to re-roll video: {e}")

    async def _on_remix(self, interaction: discord.Interaction):
        try:
            if self.on_remix_cb:
                await self.on_remix_cb(interaction, self.generation_id)
        except Exception as e:
            logger.error(f"Error in VideoActionView remix: {e}")
            await send_error_fallback(interaction, f"Failed to open remix modal: {e}")

    async def _on_toggle_fps(self, interaction: discord.Interaction):
        try:
            if self.on_toggle_fps_cb:
                await self.on_toggle_fps_cb(interaction, self.generation_id)
        except Exception as e:
            logger.error(f"Error in VideoActionView toggle fps: {e}")
            await send_error_fallback(interaction, f"Failed to toggle FPS mode: {e}")


class BertflowButtons(discord.ui.View):
    """Buttons attached to /bertflow photorealism generations.
    Actions are handled persistently via custom_id in bot.py on_interaction to prevent duplicate triggers.
    """
    def __init__(
        self,
        generation_id: str,
        on_reroll_cb=None,
        on_remix_cb=None,
        character: str = None,
        on_toggle_char_cb=None,
        on_upscale_cb=None
    ):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.on_reroll_cb = on_reroll_cb
        self.on_remix_cb = on_remix_cb
        self.character = character
        self.on_toggle_char_cb = on_toggle_char_cb
        self.on_upscale_cb = on_upscale_cb

        self.reroll_btn = discord.ui.Button(
            label="🔄 Re-roll",
            style=discord.ButtonStyle.primary,
            custom_id=f"bertflow_reroll:{generation_id}"
        )
        self.add_item(self.reroll_btn)

        self.remix_btn = discord.ui.Button(
            label="✏️ Remix",
            style=discord.ButtonStyle.secondary,
            custom_id=f"bertflow_remix:{generation_id}"
        )
        self.add_item(self.remix_btn)

        has_char = character and str(character).lower() not in ["none", "nochar", "off", "false"]
        if not has_char:
            char_label = "🌿 Ogarla: OFF"
        elif "valerie" in str(character).lower() or "val" in str(character).lower():
            char_label = "✨ Valerie: ON"
        elif "ogarla" in str(character).lower() or "oga" in str(character).lower():
            char_label = "🌿 Ogarla: ON"
        else:
            char_label = f"🎭 {character}: ON"
        char_style = discord.ButtonStyle.success if has_char else discord.ButtonStyle.secondary
        self.toggle_char_btn = discord.ui.Button(
            label=char_label,
            style=char_style,
            custom_id=f"bertflow_toggle_char:{generation_id}"
        )
        self.add_item(self.toggle_char_btn)

        self.upscale_btn = discord.ui.Button(
            label="🔍 Upscale (1.5x)",
            style=discord.ButtonStyle.secondary,
            custom_id=f"bertflow_upscale:{generation_id}"
        )
        self.add_item(self.upscale_btn)



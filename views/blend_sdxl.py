"""
SDXL Blend Studio Controls, Embed Builders, and View Components for Shallot-CUI Bot.
"""

import logging
import re
import discord
from core_helpers import send_error_fallback
from characters import get_character_display_badge
from config import get_checkpoint_display_name

logger = logging.getLogger("DiscordBot.Views.BlendSDXL")


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
        "sref": "🎲 --sref random",
        "sref1": "🎲 --sref random",
        "sref5": "5 Styles Batch",
        "sref10": "10 Styles Batch",
        "sref15": "15 Styles Batch",
        "preset_junji_ito": "🖋️ Junji Ito",
        "preset_martine_johanna": "🎨 Martine Johanna",
        "preset_dark_fantasy_landscape": "🏰 Dark Fantasy",
        "preset_cyberpunk_cityscape": "🌆 Cyberpunk",
        "preset_ethereal_portrait": "✨ Ethereal Portrait",
    }
    if sref_rand in ["sref", "sref1", True] or (isinstance(sref_rand, str) and sref_rand.lower() in ["true", "on"]):
        sref_display = "🎲 --sref random"
    elif sref_rand in preset_style_names:
        sref_display = preset_style_names[sref_rand]
    elif isinstance(sref_rand, str) and sref_rand.startswith("saved_"):
        sref_display = f"⭐ Saved (--sref {sref_rand[6:]})"
    elif not sref_rand or str(sref_rand).lower() in ["false", "none", "off", "nosref"]:
        sref_display = "OFF"
    else:
        sref_display = str(sref_rand)

    title_text = "🎨 SDXL Blend Studio (Customized)" if is_edited else "🎨 SDXL Blend Studio"
    embed = discord.Embed(
        title=title_text,
        description="*Select your aspect ratio, SDXL checkpoint, and reference composition below, then click **Blend**.*",
        color=discord.Color.from_rgb(0, 168, 252)
    )
    
    embed.add_field(
        name="🏷️ SDXL Subject Tags",
        value=f"> {caption[:900]}",
        inline=False
    )
    
    embed.add_field(
        name="✨ SDXL Scene & Composition",
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

    thumb_url = image_url or gen_data.get("image_url")
    if thumb_url:
        embed.set_thumbnail(url=thumb_url)
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
    model_display = get_checkpoint_display_name(selected_model)

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
    model_display = get_checkpoint_display_name(checkpoint)

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
        tab: str = None,
        user_favorites: list = None
    ):
        super().__init__(timeout=None)
        self.generation_id = generation_id
        self.ar = ar
        self.sr = sr
        self.model_choice = model_choice
        self.comp_strength = comp_strength
        self.sref_rand = sref_rand
        self.tab = tab
        self.user_favorites = user_favorites or []

        # Resolve char_choice & oga
        if char_choice is not None:
            self.char_choice = char_choice
            self.oga = (char_choice == "ogarla")
        else:
            self.oga = oga
            self.char_choice = "ogarla" if oga else "none"

        # Resolve sr state & clean string for default matching
        if self.sr in ["nosr", False, None]:
            sr_val = "nosr"
        elif isinstance(self.sr, str) and self.sr.startswith("sr"):
            sr_val = self.sr
        elif self.sr is True:
            sr_val = "sr75"
        else:
            sr_val = str(self.sr)

        # Resolve --sref random toggle state
        is_sref_on = self.sref_rand in ["sref", "sref1", True] or (isinstance(self.sref_rand, str) and self.sref_rand.lower() in ["true", "1", "on"])
        sref_label = "🎲 --sref random: ON" if is_sref_on else "🎲 --sref random: OFF"
        sref_style = discord.ButtonStyle.primary if is_sref_on else discord.ButtonStyle.secondary

        # Row 0: Model Checkpoint Dropdown (SDXL checkpoints only)
        model_options = [
            discord.SelectOption(label="Wai Illustrious SDXL v1.70 (Anime / Illustration)", value="wai", emoji="🌸", default=(self.model_choice in ["wai", "default"])),
            discord.SelectOption(label="Illustrious Realism v1.0 (Anime Realism)", value="illustrious_realism", emoji="🎨", default=(self.model_choice == "illustrious_realism")),
            discord.SelectOption(label="RealVisXL V4.0 (Photorealistic)", value="realvis", emoji="📸", default=(self.model_choice == "realvis")),
            discord.SelectOption(label="Juggernaut XL (Balanced Realism)", value="juggernaut", emoji="⚔️", default=(self.model_choice == "juggernaut")),
            discord.SelectOption(label="Copax Timeless XL (Cinematic)", value="copax", emoji="🎬", default=(self.model_choice == "copax")),
            discord.SelectOption(label="Ultra Realistic XL v2.5 (Fine Details)", value="ultra", emoji="✨", default=(self.model_choice == "ultra")),
            discord.SelectOption(label="Hyphoria NAI (Illustrious NAI)", value="hyphoria", emoji="🤖", default=(self.model_choice == "hyphoria")),
            discord.SelectOption(label="Nova Furry (Stylized)", value="nova", emoji="🦊", default=(self.model_choice == "nova")),
        ]
        model_select = discord.ui.Select(
            placeholder="🤖 Select Model Checkpoint...",
            options=model_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_model:{self.generation_id}",
            row=0
        )
        self.add_item(model_select)

        # Row 1: Character LoRA Dropdown
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
            row=1
        )
        self.add_item(char_select)

        # Row 2: Aspect Ratio Dropdown
        ar_options = [
            discord.SelectOption(label="1:1 Square (1024x1024)", value="1:1", emoji="📐", description="Square avatar / profile framing", default=(self.ar == "1:1")),
            discord.SelectOption(label="16:9 Widescreen (1344x768)", value="16:9", emoji="📐", description="Cinematic landscape & wallpaper", default=(self.ar == "16:9")),
            discord.SelectOption(label="9:16 Portrait / Story (768x1344)", value="9:16", emoji="📐", description="Full vertical phone / reel format", default=(self.ar == "9:16")),
            discord.SelectOption(label="4:3 Standard (1152x864)", value="4:3", emoji="📐", description="Classic standard landscape", default=(self.ar == "4:3")),
            discord.SelectOption(label="3:4 Standard Tall (864x1152)", value="3:4", emoji="📐", description="Classic portrait framing", default=(self.ar == "3:4")),
            discord.SelectOption(label="21:9 Ultra-Wide (1536x640)", value="21:9", emoji="📐", description="Panoramic cinematic banner", default=(self.ar == "21:9")),
            discord.SelectOption(label="3:5 Mobile Portrait (768x1280)", value="3:5", emoji="📐", description="Tall mobile screen format", default=(self.ar == "3:5")),
            discord.SelectOption(label="10:7 Classic Photo (1280x896)", value="10:7", emoji="📐", description="Classic print photograph ratio", default=(self.ar == "10:7")),
        ]
        ar_select = discord.ui.Select(
            placeholder="📐 Select Aspect Ratio...",
            options=ar_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_ar:{self.generation_id}",
            row=2
        )
        self.add_item(ar_select)

        # Row 3: Semi-Realism Strength Dropdown
        sr_options = [
            discord.SelectOption(label="OFF (Disabled)", value="nosr", emoji="🚫", description="No semi-realism LoRA applied", default=(sr_val == "nosr")),
            discord.SelectOption(label="Subtle (--sr.60)", value="sr60", emoji="✨", description="Gentle hint of realism (weight 0.60)", default=(sr_val == "sr60")),
            discord.SelectOption(label="Medium (--sr.70)", value="sr70", emoji="✨", description="Balanced anime realism (weight 0.70)", default=(sr_val == "sr70")),
            discord.SelectOption(label="Default (--sr.75)", value="sr75", emoji="✨", description="Signature blend realism (weight 0.75)", default=(sr_val in ["sr75", "sr"])),
            discord.SelectOption(label="Strong (--sr.80)", value="sr80", emoji="✨", description="Enhanced photographic textures (weight 0.80)", default=(sr_val == "sr80")),
            discord.SelectOption(label="High (--sr.90)", value="sr90", emoji="✨", description="Maximum photographic realism (weight 0.90)", default=(sr_val == "sr90")),
        ]
        sr_select = discord.ui.Select(
            placeholder="✨ Select Semi-Realism Strength...",
            options=sr_options,
            min_values=1,
            max_values=1,
            custom_id=f"set_blend_sr:{self.generation_id}",
            row=3
        )
        self.add_item(sr_select)

        # Row 4: Action & Toggle Buttons
        comp_button_labels = {
            "style": "🎨 Comp: Style (.20)",
            "low": "🖼️ Comp: Low (.35)",
            "med": "🖼️ Comp: Med (.60)",
            "high": "🖼️ High Comp (.85)"
        }
        comp_btn_label = comp_button_labels.get(self.comp_strength, "🖼️ Comp: Low (.35)")

        self.add_item(discord.ui.Button(
            label="🎨 Blend Image",
            style=discord.ButtonStyle.primary,
            custom_id=f"blend_desc:{self.generation_id}:blend",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label="✏️ Edit Prompt",
            style=discord.ButtonStyle.secondary,
            custom_id=f"edit_blend_prompt:{self.generation_id}",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label=comp_btn_label,
            style=discord.ButtonStyle.secondary,
            custom_id=f"cycle_blend_comp:{self.generation_id}",
            row=4
        ))
        self.add_item(discord.ui.Button(
            label=sref_label,
            style=sref_style,
            custom_id=f"toggle_blend_sref:{self.generation_id}",
            row=4
        ))

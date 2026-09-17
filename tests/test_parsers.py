"""
Automated Test Suite Module
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import unittest
import copy
import random
import logging
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch
from PIL import Image
import io

# Disable log output during test execution
logging.disable(logging.CRITICAL)

from parsers import (
    parse_aspect_ratio,
    parse_loras,
    apply_loras_to_workflow,
    parse_seed,
    parse_stylize,
    parse_sref,
    parse_cref,
    apply_ipadapter_to_workflow,
    expand_dynamic_prompt,
    parse_magic_prompt,
    apply_magic_enhancement,
    calculate_wan_dimensions,
    extract_positive_prompt,
)
from image_utils import (
    calculate_outpaint_padding,
    composite_outpaint_seamless,
    composite_outpaint_seamless_async,
    create_outpaint_edge_bleed_canvas,
    create_outpaint_edge_bleed_canvas_async,
    save_quadrant_images,
    get_quadrant_bytes,
)
from bot import build_blend_workflow
from error_handler import (
    error_handler,
    ErrorCategory,
    ErrorSeverity,
    AutoFixAction,
    AutoFixResult,
)
import model_architecture
import characters
import db


class TestParsers(unittest.TestCase):
    def test_module1_aspect_ratios(self):
        """Test aspect ratio parsing and resolution calculations."""
        # 16:9 for SD1.5 (base area 262144 -> ~680x384 rounded to 64: 640x384 or similar)
        p1, w1, h1 = parse_aspect_ratio("a prompt --ar 16:9", model_name="v1-5")
        self.assertEqual(p1, "a prompt")
        self.assertAlmostEqual(w1 / h1, 16 / 9, delta=0.3)

        # 9:16 for SDXL (base area 1048576 -> 768x1344)
        p2, w2, h2 = parse_aspect_ratio("a prompt --ar 9:16", model_name="xl")
        self.assertEqual(p2, "a prompt")
        self.assertTrue(h2 > w2)

        # 21:9 for SDXL
        p3, w3, h3 = parse_aspect_ratio("a prompt --ar 21:9", model_name="xl")
        self.assertTrue(w3 > h3)

        # 3:5 for SDXL
        p4, w4, h4 = parse_aspect_ratio("a prompt --ar 3:5", model_name="xl")
        self.assertEqual(p4, "a prompt")
        self.assertTrue(h4 > w4)
        self.assertAlmostEqual(w4 / h4, 3 / 5, delta=0.2)

        # 10:7 for SDXL
        p5, w5, h5 = parse_aspect_ratio("a prompt --ar 10:7", model_name="xl")
        self.assertEqual(p5, "a prompt")
        self.assertTrue(w5 > h5)
        self.assertAlmostEqual(w5 / h5, 10 / 7, delta=0.2)

        # Floating point / Taskbar fit 16:9.3 and 1920:1032 for SDXL
        p6, w6, h6 = parse_aspect_ratio("a prompt --ar 16:9.3", model_name="xl")
        self.assertEqual(p6, "a prompt")
        self.assertTrue(w6 > h6)

        p7, w7, h7 = parse_aspect_ratio("a prompt --ar 1920:1032", model_name="xl")
        self.assertEqual(p7, "a prompt")
        self.assertAlmostEqual(w7 / h7, 1920 / 1032, delta=0.1)

        p8, w8, h8 = parse_aspect_ratio("a prompt --ar 1.86:1", model_name="xl")
        self.assertEqual(p8, "a prompt")
        self.assertAlmostEqual(w8 / h8, 1.86, delta=0.1)

        # Test multiple aspect ratio flags in same prompt (last one must override earlier ones)
        p9, w9, h9 = parse_aspect_ratio("wide vista --ar 16:9 extra text --ar 3:5", model_name="xl")
        self.assertEqual(p9, "wide vista extra text")
        self.assertTrue(h9 > w9)
        self.assertAlmostEqual(w9 / h9, 3 / 5, delta=0.2)

        # Test build_scapes_prompt with mode=None
        from parsers import build_scapes_prompt
        sinfo = build_scapes_prompt(user_prompt="test", style="junji_ito", mode=None)
        self.assertNotIn("--ar", sinfo["final_prompt"])

        # Test format_image_filename helper
        from image_utils import format_image_filename
        fn1 = format_image_filename("grid", 123456, "jpg")
        self.assertTrue(fn1.startswith("grid_"))
        self.assertTrue(fn1.endswith("_seed123456.jpg"))

        fn2 = format_image_filename("isolated_1", 999, "png")
        self.assertTrue(fn2.startswith("isolated_1_"))
        self.assertTrue(fn2.endswith("_seed999.png"))

    def test_module2_prompt_parsers(self):
        """Test prompt flag parsing: loras, seed, stylize, sref, magic, wildcards."""
        # LoRA shorthand
        p_lora, loras = parse_loras("warrior --sr.85 <lora:my_lora:0.7>")
        self.assertIn(("Semi-realism_illustrious.safetensors", 0.85), loras)
        self.assertIn(("my_lora", 0.7), loras)

        # Seed parsing
        p_seed, seed = parse_seed("cyberpunk motorcycle --seed 123456")
        self.assertEqual(p_seed, "cyberpunk motorcycle")
        self.assertEqual(seed, 123456)

        # Stylize / Raw
        p_style, cfg, quality = parse_stylize("apple on table --s 900")
        self.assertTrue(cfg > 10.0)
        self.assertTrue(quality)

        p_raw, cfg_raw, quality_raw = parse_stylize("apple on table --raw")
        self.assertFalse(quality_raw)
        self.assertEqual(cfg_raw, 3.0)

        # Sref parsing
        p_sref, url, weight, info = parse_sref("retro city --sw 0.85 --sref http://example.com/img.jpg")
        self.assertEqual(url, "http://example.com/img.jpg")
        self.assertEqual(weight, 0.85)

        p_rnd, url_rnd, weight_rnd, info_rnd = parse_sref("retro city --sref random")
        self.assertIsNotNone(info_rnd)
        self.assertIn("code", info_rnd)
        self.assertIn("name", info_rnd)

        # Test copy-pasted sref autocomplete labels
        p_pasted, _, _, info_pasted = parse_sref("glory. --sr.60 --ar 10:7 --sref 🎲 Random (--sref random)")
        self.assertIsNotNone(info_pasted)
        self.assertIn("code", info_pasted)
        self.assertEqual(p_pasted, "glory. --sr.60 --ar 10:7, " + info_pasted["prompt"])

        p_pasted2, _, _, info_pasted2 = parse_sref("glory. --sref Cyberpunk Neon (837192)")
        self.assertIsNotNone(info_pasted2)
        self.assertEqual(info_pasted2["code"], 837192)

        # Cref parsing
        p_cref, c_url, c_weight = parse_cref("cyberpunk warrior --cw 0.85 --cref http://example.com/face.png")
        self.assertEqual(p_cref, "cyberpunk warrior")
        self.assertEqual(c_url, "http://example.com/face.png")
        self.assertEqual(c_weight, 0.85)

        # Wildcard expansion
        rng = random.Random(42)
        expanded = expand_dynamic_prompt("a photo of a {dragon|tiger|wolf}", rng)
        self.assertIn(expanded, ["a photo of a dragon", "a photo of a tiger", "a photo of a wolf"])

        # Magic prompt flag
        p_magic, is_magic = parse_magic_prompt("dragon sitting on chest --magic")
        self.assertTrue(is_magic)
        enhanced = apply_magic_enhancement(p_magic, 100)
        self.assertTrue(len(enhanced) > len(p_magic))

        # Test clean_quadrant_prompts wildcard choice extraction
        from parsers import clean_quadrant_prompts
        raw_p = "Semi-realism, ogarla, {low angle|high angle}, 2girls, brown hair, {red Crop top|white swimsuit}, {oral sex|handjob}"
        exp_prompts = [
            "Semi-realism, ogarla, low angle, 2girls, brown hair, red Crop top, oral sex",
            "Semi-realism, ogarla, low angle, 2girls, brown hair, white swimsuit, oral sex",
            "Semi-realism, ogarla, high angle, 2girls, brown hair, red Crop top, handjob",
            "Semi-realism, ogarla, high angle, 2girls, brown hair, white swimsuit, handjob",
        ]
        cleaned = clean_quadrant_prompts(exp_prompts, raw_p)
        self.assertEqual(cleaned[0], "low angle, red Crop top, oral sex")
        self.assertEqual(cleaned[1], "low angle, white swimsuit, oral sex")
        self.assertEqual(cleaned[2], "high angle, red Crop top, handjob")
        self.assertEqual(cleaned[3], "high angle, white swimsuit, handjob")

    def test_module4_outpaint_padding(self):
        """Test padding and canvas expansion calculations."""
        # Create dummy 512x512 PNG image bytes
        img = Image.new("RGB", (512, 512), color="red")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        dummy_bytes = buf.getvalue()

        left, top, right, bottom, res_bytes, out_w, out_h = calculate_outpaint_padding(dummy_bytes, "16:9")
        self.assertTrue(right > 0 or left > 0)
        self.assertAlmostEqual(out_w / out_h, 16 / 9, delta=0.2)

        left_35, top_35, right_35, bottom_35, _, w35, h35 = calculate_outpaint_padding(dummy_bytes, "3:5")
        self.assertAlmostEqual(w35 / h35, 3 / 5, delta=0.2)

        left_107, top_107, right_107, bottom_107, _, w107, h107 = calculate_outpaint_padding(dummy_bytes, "10:7")
        self.assertAlmostEqual(w107 / h107, 10 / 7, delta=0.2)

        # Test Zoom Out 1.5x
        left_z, top_z, right_z, bottom_z, _, zw, zh = calculate_outpaint_padding(dummy_bytes, "1.5x")
        self.assertTrue(left_z > 0 and top_z > 0)

        # Test Directional Pan (Up, Down, Left, Right)
        l_up, t_up, r_up, b_up, _, w_up, h_up = calculate_outpaint_padding(dummy_bytes, "up")
        self.assertEqual(t_up, 384)
        self.assertEqual(b_up, 0)
        self.assertEqual(l_up, 0)
        self.assertEqual(r_up, 0)
        self.assertEqual(h_up, 1024 + 384)
        self.assertEqual(w_up, 1024)

        l_down, t_down, r_down, b_down, _, w_down, h_down = calculate_outpaint_padding(dummy_bytes, "down")
        self.assertEqual(t_down, 0)
        self.assertEqual(b_down, 384)
        self.assertEqual(l_down, 0)
        self.assertEqual(r_down, 0)
        self.assertEqual(h_down, 1024 + 384)
        self.assertEqual(w_down, 1024)

        l_left, t_left, r_left, b_left, _, w_left, h_left = calculate_outpaint_padding(dummy_bytes, "left")
        self.assertEqual(t_left, 0)
        self.assertEqual(b_left, 0)
        self.assertEqual(l_left, 384)
        self.assertEqual(r_left, 0)
        self.assertEqual(w_left, 1024 + 384)
        self.assertEqual(h_left, 1024)

        l_right, t_right, r_right, b_right, _, w_right, h_right = calculate_outpaint_padding(dummy_bytes, "right")
        self.assertEqual(t_right, 0)
        self.assertEqual(b_right, 0)
        self.assertEqual(l_right, 0)
        self.assertEqual(r_right, 384)
        self.assertEqual(w_right, 1024 + 384)
        self.assertEqual(h_right, 1024)

        # Test oversized input image normalization (preventing multi-megapixel explosion)
        oversized = Image.new("RGB", (1624, 940), color="blue")
        buf_over = io.BytesIO()
        oversized.save(buf_over, format="PNG")
        l_over, t_over, r_over, b_over, pad_bytes, w_over, h_over = calculate_outpaint_padding(buf_over.getvalue(), "1.5x")
        self.assertTrue(w_over <= 1664)
        self.assertTrue(h_over <= 1024)
        self.assertTrue(w_over * h_over <= 1_600_000)

    def test_module4_composite_outpaint_seamless(self):
        """Test seamless alpha-feathered outpaint compositing to eliminate seam lines."""
        # Create a red original image (100x100)
        orig = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf_o = io.BytesIO()
        orig.save(buf_o, format="PNG")

        # Create a green generated canvas (160x160)
        gen = Image.new("RGB", (160, 160), color=(0, 255, 0))
        buf_g = io.BytesIO()
        gen.save(buf_g, format="PNG")

        # Composite with 30px padding on all sides
        res_bytes = composite_outpaint_seamless(
            original_img_bytes=buf_o.getvalue(),
            generated_img_bytes=buf_g.getvalue(),
            left=30,
            top=30,
            right=30,
            bottom=30,
            feather_radius=10
        )
        self.assertTrue(len(res_bytes) > 0)
        res_img = Image.open(io.BytesIO(res_bytes))
        self.assertEqual(res_img.size, (160, 160))

        # Deep center pixel should be pristine original red
        center_color = res_img.getpixel((80, 80))
        self.assertEqual(center_color, (255, 0, 0))

        # Outer padding pixel should be pure generated green
        outer_color = res_img.getpixel((5, 5))
        self.assertEqual(outer_color, (0, 255, 0))

        # Boundary pixel should be a smooth blend
        blend_color = res_img.getpixel((33, 80))
        self.assertTrue(0 < blend_color[0] < 255)
        self.assertTrue(0 < blend_color[1] < 255)

        # Async variant test
        import asyncio
        async_res = asyncio.run(composite_outpaint_seamless_async(
            original_img_bytes=buf_o.getvalue(),
            generated_img_bytes=buf_g.getvalue(),
            left=30,
            top=30,
            right=30,
            bottom=30,
            feather_radius=10
        ))
        self.assertEqual(len(async_res), len(res_bytes))

    def test_module4_outpaint_edge_bleed_and_prompt_adaptation(self):
        """Test directional edge-bleed canvas generation and prompt adaptation for outpainting."""
        from services.grid_actions_service import adapt_prompt_for_outpaint

        # 1. Test prompt adaptation
        raw_p = "A portrait featuring three characters against a red background. The central figure is a skeleton, close-up of face."
        adapted_zoom = adapt_prompt_for_outpaint(raw_p, "1.5x")
        self.assertNotIn("three characters", adapted_zoom.lower())
        self.assertIn("characters", adapted_zoom.lower())
        self.assertNotIn("close-up", adapted_zoom.lower())
        self.assertIn("wider angle view", adapted_zoom.lower())

        adapted_down = adapt_prompt_for_outpaint(raw_p, "down")
        self.assertIn("lower body", adapted_down.lower())

        # 2. Test edge-bleed canvas generation
        orig = Image.new("RGB", (100, 100), color=(180, 20, 20))
        buf = io.BytesIO()
        orig.save(buf, format="PNG")

        bleed_bytes = create_outpaint_edge_bleed_canvas(
            buf.getvalue(), left=20, top=20, right=20, bottom=20, blur_radius=8, feather_radius=15
        )
        self.assertTrue(len(bleed_bytes) > 0)
        bleed_img = Image.open(io.BytesIO(bleed_bytes))
        self.assertEqual(bleed_img.size, (140, 140))
        self.assertEqual(bleed_img.mode, "RGBA")

        # Margin pixel should carry the edge-bled color with Alpha = 0 (outpaint mask)
        margin_px = bleed_img.getpixel((5, 5))
        self.assertEqual(margin_px[0], 180)
        self.assertEqual(margin_px[3], 0)

        # Center pixel should have Alpha = 255 (keep mask)
        center_px = bleed_img.getpixel((70, 70))
        self.assertEqual(center_px[3], 255)

        # 3. Test async variant
        import asyncio
        async_bleed = asyncio.run(create_outpaint_edge_bleed_canvas_async(
            buf.getvalue(), left=10, top=10, right=10, bottom=10
        ))
        self.assertTrue(len(async_bleed) > 0)



    def test_module14_wan_video_dimensions(self):
        """Test Wan 2.2 video dimension math for 8GB VRAM cards across different source aspect ratios."""
        # 16:9 widescreen source image (1920x1080) -> expect ~832x480 (multiples of 16)
        w_169, h_169 = calculate_wan_dimensions(1920, 1080, target_area=399360)
        self.assertEqual(w_169 % 16, 0)
        self.assertEqual(h_169 % 16, 0)
        self.assertAlmostEqual(w_169 / h_169, 16 / 9, delta=0.2)
        self.assertTrue(w_169 * h_169 <= 420000) # Stay under VRAM limit

        # 9:16 portrait source image (1080x1920) -> expect ~480x832
        w_916, h_916 = calculate_wan_dimensions(1080, 1920, target_area=399360)
        self.assertEqual(w_916 % 16, 0)
        self.assertEqual(h_916 % 16, 0)
        self.assertTrue(h_916 > w_916)
        self.assertAlmostEqual(w_916 / h_916, 9 / 16, delta=0.2)

        # 1:1 square source image (1024x1024) -> expect equal width and height rounded to 16px grid (624x624)
        w_11, h_11 = calculate_wan_dimensions(1024, 1024, target_area=399360)
        self.assertEqual(w_11, h_11)
        self.assertEqual(w_11 % 16, 0)

        # 21:9 cinematic ultra-wide source image (2560x1080)
        w_219, h_219 = calculate_wan_dimensions(2560, 1080, target_area=399360)
        self.assertEqual(w_219 % 16, 0)
        self.assertEqual(h_219 % 16, 0)
        self.assertTrue(w_219 > h_219)

    def test_module18_sref_change_isolated(self):
        """Test IsolatedImageButtons --sref change buttons and sref prompt replacement."""
        from views import IsolatedImageButtons, CustomSrefModal, SavedSrefSelectView
        import re

        view = IsolatedImageButtons("123456", 1, has_sref=True)
        custom_ids = [item.custom_id for item in view.children]
        self.assertIn("sref_change_custom:123456:1", custom_ids)
        self.assertIn("sref_change_random:123456:1", custom_ids)
        self.assertIn("sref_change_saved:123456:1", custom_ids)

        # Test prompt sref replacement
        raw_p = "dog in lavender field --sr.60 --ar 10:7 --sref 772382"
        cleaned_p = re.sub(r'[-\u2014\u2013]{1,2}sref\s+[^\s]+(?:\s*\([^)]*\))?', '', raw_p, flags=re.IGNORECASE).strip()
        self.assertEqual(cleaned_p, "dog in lavender field --sr.60 --ar 10:7")

        new_full_p = f"{cleaned_p} --sref 847291"
        self.assertEqual(new_full_p, "dog in lavender field --sr.60 --ar 10:7 --sref 847291")

        # Verify parse_seed returns 2 elements (prompt, seed)
        from parsers import parse_seed
        parsed_p, seed_val = parse_seed(new_full_p)
        self.assertEqual(parsed_p, "dog in lavender field --sr.60 --ar 10:7 --sref 847291")
        self.assertIsNone(seed_val)

    def test_module19_study_prompt_extraction(self):
        """Test PNG positive prompt extraction for /study command."""
        from PIL.PngImagePlugin import PngInfo

        # 1. A1111 parameters
        img1 = Image.new("RGB", (50, 50))
        meta1 = PngInfo()
        meta1.add_text("parameters", "a beautiful fantasy castle\nNegative prompt: blurry\nSteps: 20")
        buf1 = io.BytesIO()
        img1.save(buf1, format="PNG", pnginfo=meta1)
        self.assertEqual(extract_positive_prompt(buf1.getvalue()), "a beautiful fantasy castle")

        # 2. ComfyUI API prompt JSON
        img2 = Image.new("RGB", (50, 50))
        meta2 = PngInfo()
        meta2.add_text("prompt", json.dumps({
            "1": {"class_type": "KSampler", "inputs": {"positive": ["2", 0]}},
            "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "epic space battle"}}
        }))
        buf2 = io.BytesIO()
        img2.save(buf2, format="PNG", pnginfo=meta2)
        self.assertEqual(extract_positive_prompt(buf2.getvalue()), "epic space battle")

        # 3. Image with no metadata
        img3 = Image.new("RGB", (50, 50))
        buf3 = io.BytesIO()
        img3.save(buf3, format="PNG")
        self.assertEqual(extract_positive_prompt(buf3.getvalue()), "NOT FOUND")

    def test_module22_smart_art_director_engine(self):
        """Test Smart Art Director engine keyword classification and sref pairing."""
        from parsers import parse_smart_prompt, apply_smart_magic_and_sref

        # 1. Parse --smart flag
        cleaned, is_smart = parse_smart_prompt("ogarla in a neon cyberpunk alleyway --smart")
        self.assertEqual(cleaned, "ogarla in a neon cyberpunk alleyway")
        self.assertTrue(is_smart)

        # 2. Cyberpunk classification for SDXL
        enhanced, sref = apply_smart_magic_and_sref("cyberpunk robot city", is_flux=False)
        self.assertIn("futuristic neon reflections", enhanced)
        self.assertEqual(sref, "113408")

        # 3. Fantasy classification for Flux
        enhanced_flux, sref_flux = apply_smart_magic_and_sref("dragon guarding castle", is_flux=True)
        self.assertIn("intricate ornate detail", enhanced_flux)
        self.assertIsNone(sref_flux)

    def test_module27_lightning_removal_and_standard_lora_flow(self):
        """Test that lightning shorthand is cleanly ignored and standard FreeU flow is preserved."""
        cleaned, loras = parse_loras("cyberpunk warrior --lightning", is_flux=False)
        # Should not inject any lightning LoRA
        self.assertEqual(len(loras), 0)
        self.assertEqual(cleaned, "cyberpunk warrior --lightning")

        # FreeU (Node 20) should remain connected in the standard pipeline
        with open("workflows/sdxl_powerhouse_2stage.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        configured = apply_loras_to_workflow(wf, loras)
        self.assertEqual(configured["3"]["inputs"]["model"], ["20", 0])

    def test_module28_consolidated_enhancements(self):
        """Test that consolidated enhancement choices correctly decode and configure workflow flags."""
        from bot import SDXL_ENHANCEMENT_CHOICES
        self.assertTrue(len(SDXL_ENHANCEMENT_CHOICES) > 0)

        # Verify values
        sdxl_vals = [c.value for c in SDXL_ENHANCEMENT_CHOICES]
        self.assertNotIn("lightning", sdxl_vals)
        self.assertNotIn("all", sdxl_vals)
        self.assertIn("ultimate", sdxl_vals)
        self.assertIn("smart", sdxl_vals)
        self.assertIn("magic", sdxl_vals)
        self.assertIn("smart+magic", sdxl_vals)
        self.assertIn("no_freeu", sdxl_vals)
        self.assertIn("powerhouse", sdxl_vals)

        # Discord requires choice names <= 100 chars
        for c in SDXL_ENHANCEMENT_CHOICES:
            self.assertLessEqual(len(c.name), 100, f"Choice name exceeds 100 chars: {c.name}")

        # Test prompt flag parsers for powerhouse and freeu
        from parsers import parse_powerhouse_prompt, parse_freeu_prompt
        p1, is_ph = parse_powerhouse_prompt("cyberpunk city street --powerhouse")
        self.assertTrue(is_ph)
        self.assertEqual(p1, "cyberpunk city street")

        p2, is_ph2 = parse_powerhouse_prompt("cyberpunk city street --ph")
        self.assertTrue(is_ph2)
        self.assertEqual(p2, "cyberpunk city street")

        p3, is_ph3 = parse_powerhouse_prompt("cyberpunk city street --refine")
        self.assertTrue(is_ph3)
        self.assertEqual(p3, "cyberpunk city street")

        p4, is_raw = parse_freeu_prompt("cyberpunk city street --raw")
        self.assertTrue(is_raw)
        self.assertEqual(p4, "cyberpunk city street")

        p5, is_nofreeu = parse_freeu_prompt("cyberpunk city street --nofreeu")
        self.assertTrue(is_nofreeu)
        self.assertEqual(p5, "cyberpunk city street")

        p6, is_nofreeu2 = parse_freeu_prompt("cyberpunk city street --disable-freeu")
        self.assertTrue(is_nofreeu2)
        self.assertEqual(p6, "cyberpunk city street")

    def test_module29_reroll_lora_preservation(self):
        """Test that re-rolled generations properly preserve and wire Ogarla & Semi-Realism LoRAs."""
        from parsers import parse_loras, apply_loras_to_workflow
        test_prompt = "Semi-realism, masterpiece, best quality, absurddres. ogarla, awe. 2girls."
        cleaned, loras = parse_loras(test_prompt, is_flux=False)
        self.assertTrue(any(l[0] == "ogarla_epoch_5.safetensors" for l in loras))
        self.assertTrue(any(l[0] == "Semi-realism_illustrious.safetensors" for l in loras))

        with open("workflows/txt2img_lowres.json", "r", encoding="utf-8") as f:
            wf = json.load(f)
        applied = apply_loras_to_workflow(wf, loras)
        self.assertEqual(applied["75"]["inputs"]["strength_model"], 0.70)
        self.assertEqual(applied["76"]["inputs"]["strength_model"], 0.85)

    def test_module35_copy_prompt_large_text(self):
        """Test handle_copy_prompt safely handles both short and long (>2000 chars) prompt strings."""
        import db
        from unittest.mock import AsyncMock, MagicMock
        from bot import handle_copy_prompt
        import asyncio

        gen_id = "test_large_prompt_123"
        long_prompt = "a " * 1500  # 3000 characters
        db.save_generation(gen_id, {"prompt": long_prompt, "status": "completed"})

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = False
        mock_interaction.response.send_message = AsyncMock()

        asyncio.run(handle_copy_prompt(mock_interaction, gen_id))

        self.assertTrue(mock_interaction.response.send_message.called)
        call_kwargs = mock_interaction.response.send_message.call_args.kwargs
        # Verify message content length is <= 2000 chars and file attachment is provided
    def test_valerie_lora_parsing(self):
        """Test parsing --valerie and bare 'valerie' keywords in prompt with silent trigger substitution."""
        from parsers import parse_loras

        # 1. Test shorthand flag --valerie.80
        cleaned, loras = parse_loras("portrait of a woman --valerie.80", is_flux=False)
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "jen_epoch_5.safetensors")
        self.assertAlmostEqual(loras[0][1], 0.80)
        self.assertIn("jen", cleaned)
        self.assertNotIn("--valerie", cleaned)

        # 2. Test keyword fallback
        cleaned2, loras2 = parse_loras("valerie walking in cyberpunk rain", is_flux=False)
        self.assertEqual(len(loras2), 1)
        self.assertEqual(loras2[0][0], "jen_epoch_5.safetensors")
        self.assertEqual(cleaned2, "jen walking in cyberpunk rain")

    def test_sully_lora_parsing(self):
        """Test parsing --sully and bare 'sully' keywords in prompt with silent trigger & traits substitution."""
        from parsers import parse_loras

        # 1. Test shorthand flag --sully.80
        cleaned, loras = parse_loras("portrait of a student in library --sully.80", is_flux=False)
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "susa_epoch_6.safetensors")
        self.assertAlmostEqual(loras[0][1], 0.80)
        self.assertIn("susa", cleaned)
        self.assertIn("black hair, thin rim glasses", cleaned)
        self.assertNotIn("--sully", cleaned)

        # 2. Test keyword fallback
        cleaned2, loras2 = parse_loras("sully reading a book", is_flux=False)
        self.assertEqual(len(loras2), 1)
        self.assertEqual(loras2[0][0], "susa_epoch_6.safetensors")
        self.assertIn("susa", cleaned2)
        self.assertIn("black hair, thin rim glasses", cleaned2)

    def test_mageill_lora_parsing(self):
        """Test parsing --mageill with multiple epochs (3, 4, 5, 6) and custom weights."""
        from parsers import parse_loras

        # 1. Default epoch (Epoch 5) with --mageill
        cleaned, loras = parse_loras("portrait of an enchantress --mageill", is_flux=False)
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "mageill_epoch_5.safetensors")
        self.assertAlmostEqual(loras[0][1], 0.85)
        self.assertIn("mageill", cleaned)
        self.assertNotIn("--mageill", cleaned)

        # 2. Shorthand --mag (also defaults to Epoch 5)
        cleaned_mag, loras_mag = parse_loras("portrait of an enchantress --mag", is_flux=False)
        self.assertEqual(len(loras_mag), 1)
        self.assertEqual(loras_mag[0][0], "mageill_epoch_5.safetensors")

        # 3. Explicit Epoch 3: --mageill3
        c3, l3 = parse_loras("fantasy scene --mageill3", is_flux=False)
        self.assertEqual(len(l3), 1)
        self.assertEqual(l3[0][0], "mageill_epoch_3.safetensors")
        self.assertAlmostEqual(l3[0][1], 0.85)

        # 4. Explicit Epoch 4 with weight: --mag4.80
        c4, l4 = parse_loras("fantasy scene --mag4.80", is_flux=False)
        self.assertEqual(len(l4), 1)
        self.assertEqual(l4[0][0], "mageill_epoch_4.safetensors")
        self.assertAlmostEqual(l4[0][1], 0.80)

        # 5. Explicit Epoch 6 with hyphen: --mageill-e6.70
        c6, l6 = parse_loras("fantasy scene --mageill-e6.70", is_flux=False)
        self.assertEqual(len(l6), 1)
        self.assertEqual(l6[0][0], "mageill_epoch_6.safetensors")
        self.assertAlmostEqual(l6[0][1], 0.70)

        # 6. Keyword fallback: bare 'mageill'
        ck, lk = parse_loras("mageill floating above glowing crystals", is_flux=False)
        self.assertEqual(len(lk), 1)
        self.assertEqual(lk[0][0], "mageill_epoch_5.safetensors")
        self.assertAlmostEqual(lk[0][1], 0.85)

    def test_cheri_lora_parsing(self):
        """Test parsing --cheri with multiple epochs (4, 6), custom weights, and blonde hair injection."""
        from parsers import parse_loras

        # 1. Default epoch (Epoch 6) with --cheri
        cleaned, loras = parse_loras("portrait of a woman in garden --cheri", is_flux=False)
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "cheri_epoch_6.safetensors")
        self.assertAlmostEqual(loras[0][1], 0.85)
        self.assertIn("cheri", cleaned)
        self.assertIn("blonde hair", cleaned)
        self.assertNotIn("--cheri", cleaned)

        # 2. Shorthand --che
        c_che, l_che = parse_loras("portrait of a woman --che", is_flux=False)
        self.assertEqual(len(l_che), 1)
        self.assertEqual(l_che[0][0], "cheri_epoch_6.safetensors")
        self.assertIn("blonde hair", c_che)

        # 3. Explicit Epoch 4: --cheri4
        c4, l4 = parse_loras("fashion model --cheri4", is_flux=False)
        self.assertEqual(len(l4), 1)
        self.assertEqual(l4[0][0], "cheri_epoch_4.safetensors")
        self.assertAlmostEqual(l4[0][1], 0.85)
        self.assertIn("blonde hair", c4)

        # 4. Explicit Epoch 4 with custom weight: --che4.80
        c4w, l4w = parse_loras("fashion model --che4.80", is_flux=False)
        self.assertEqual(len(l4w), 1)
        self.assertEqual(l4w[0][0], "cheri_epoch_4.safetensors")
        self.assertAlmostEqual(l4w[0][1], 0.80)

        # 5. Explicit Epoch 6 with custom weight: --cheri6.70
        c6w, l6w = parse_loras("fashion model --cheri6.70", is_flux=False)
        self.assertEqual(len(l6w), 1)
        self.assertEqual(l6w[0][0], "cheri_epoch_6.safetensors")
        self.assertAlmostEqual(l6w[0][1], 0.70)

        # 6. Keyword fallback: bare 'cheri'
        ck, lk = parse_loras("cheri enjoying ice cream at the beach", is_flux=False)
        self.assertEqual(len(lk), 1)
        self.assertEqual(lk[0][0], "cheri_epoch_6.safetensors")
        self.assertAlmostEqual(lk[0][1], 0.85)
        self.assertIn("blonde hair", ck)

    def test_module42_video_motion_flags(self):
        """Test parse_video_motion_flags parsing camera directives, badges, and prompt augmentation."""
        from parsers import parse_video_motion_flags

        # 1. Test zoom flags
        clean, badges, aug = parse_video_motion_flags("a majestic dragon --zoom-in")
        self.assertEqual(clean, "a majestic dragon")
        self.assertIn("🎥 Zoom In", badges)
        self.assertIn("slow cinematic camera zoom in", aug)

        # 2. Test multi-flag combinations: pan, orbit, cinematic
        clean2, badges2, aug2 = parse_video_motion_flags("cyberpunk sports car drifting in neon rain --pan-right --orbit --cinematic")
        self.assertEqual(clean2, "cyberpunk sports car drifting in neon rain")
        self.assertIn("🎥 Pan Right", badges2)
        self.assertIn("🎥 Orbit", badges2)
        self.assertIn("✨ Cinematic", badges2)
        self.assertIn("smooth cinematic camera pan to the right", aug2)
        self.assertIn("orbital camera movement", aug2)

        # 3. Test subtle and tilt flags
        clean3, badges3, aug3 = parse_video_motion_flags("close up portrait of a warrior --subtle --tilt-up")
        self.assertEqual(clean3, "close up portrait of a warrior")
        self.assertIn("🍃 Subtle", badges3)
        self.assertIn("🎥 Tilt Up", badges3)

        # 3b. Test realtime / normal-speed flag
        clean_rt, badges_rt, aug_rt = parse_video_motion_flags("a dancer performing a choreography --realtime")
        self.assertEqual(clean_rt, "a dancer performing a choreography")
        self.assertIn("⏱️ Real-Time", badges_rt)
        self.assertIn("real-time motion, natural speed playback", aug_rt)

        # 4. Test prompt with no motion flags
        clean4, badges4, aug4 = parse_video_motion_flags("a cozy cabin in snowy woods")
        self.assertEqual(clean4, "a cozy cabin in snowy woods")
        self.assertEqual(badges4, [])
        self.assertEqual(aug4, "a cozy cabin in snowy woods")

        # 5. Test empty or None input
        clean5, badges5, aug5 = parse_video_motion_flags(None)
        self.assertEqual(clean5, "")
        self.assertEqual(badges5, [])
        self.assertEqual(aug5, "")

    def test_module70_vision_ai_multi_target_formatters(self):
        """Test JoyCaption, Qwen2.5-VL, and multi-architecture prompt formatters & workflow integrity."""
        from parsers import format_sdxl_prompt, format_flux_prompt, format_krea2_prompt
        import bot

        # 1. Test format_sdxl_prompt
        raw_sdxl = "The photo depicts a beautiful girl wearing a leather jacket. There is neon lighting in the background, 8k resolution, masterpiece."
        sdxl = format_sdxl_prompt(raw_sdxl)
        self.assertNotIn("the photo depicts", sdxl.lower())
        self.assertNotIn("there is", sdxl.lower())
        self.assertIn("leather jacket", sdxl.lower())
        self.assertIn(",", sdxl)

        # 2. Test format_flux_prompt
        raw_flux = "This is an image of an adventurer in a sunlit forest. Masterpiece, best quality, ultra high res."
        flux = format_flux_prompt(raw_flux)
        self.assertNotIn("this is an image of", flux.lower())
        self.assertNotIn("masterpiece", flux.lower())
        self.assertNotIn("best quality", flux.lower())
        self.assertTrue(flux.startswith("An adventurer") or flux.startswith("Adventurer"))

        # 3. Test format_krea2_prompt
        raw_krea = "The image features a serene mountain lake at dusk with misty reflections."
        krea = format_krea2_prompt(raw_krea)
        self.assertNotIn("the image features", krea.lower())
        self.assertIn("mountain lake", krea.lower())

        # 4. Test workflow JSON integrity
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        joy_path = os.path.join(project_root, "workflows", "DESCRIBE_joycaption.json")
        qwen_path = os.path.join(project_root, "workflows", "DESCRIBE_qwen_vl.json")
        self.assertTrue(os.path.exists(joy_path), "DESCRIBE_joycaption.json must exist")
        self.assertTrue(os.path.exists(qwen_path), "DESCRIBE_qwen_vl.json must exist")

        with open(joy_path, "r", encoding="utf-8") as f:
            joy_wf = json.load(f)
        self.assertIn("1", joy_wf)
        self.assertIn("2", joy_wf)

        with open(qwen_path, "r", encoding="utf-8") as f:
            qwen_wf = json.load(f)
        self.assertIn("1", qwen_wf)
        self.assertIn("2", qwen_wf)

        # 5. Test bot vision interrogate availability
        self.assertTrue(hasattr(bot, "run_vision_interrogate"))
        self.assertTrue(hasattr(bot, "user_vision_preferences"))

    def test_module70b_describe_overall_to_general_sanitization(self):
        """Test that the word 'overall' cannot be used in /describe and is replaced with 'general' (preserving case and leaving 'overalls' intact)."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        from parsers import sanitize_describe_text, format_sdxl_prompt, format_flux_prompt, format_krea2_prompt
        from services.vision_service import execute_describe_core
        import db

        # 1. Direct unit test of sanitize_describe_text
        self.assertEqual(sanitize_describe_text("The overall mood is serene."), "The general mood is serene.")
        self.assertEqual(sanitize_describe_text("Overall, the lighting is warm."), "General, the lighting is warm.")
        self.assertEqual(sanitize_describe_text("OVERALL ATMOSPHERE"), "GENERAL ATMOSPHERE")
        self.assertEqual(sanitize_describe_text("She is wearing blue denim overalls in a barn."), "She is wearing blue denim overalls in a barn.")
        self.assertEqual(sanitize_describe_text("an overall aesthetic with overalls."), "a general aesthetic with overalls.")
        self.assertEqual(sanitize_describe_text(""), "")
        self.assertEqual(sanitize_describe_text(None), "")

        # 2. Test formatters with 'overall'
        raw_joy = "The image captures a woman with an overall graceful posture, soft lighting, 8k resolution."
        sdxl_out = format_sdxl_prompt(raw_joy)
        self.assertNotIn("overall", sdxl_out.lower())
        self.assertIn("general", sdxl_out.lower())

        raw_flux = "This is an image of a tranquil landscape with an overall misty ambiance, masterpiece."
        flux_out = format_flux_prompt(raw_flux)
        self.assertNotIn("overall", flux_out.lower())
        self.assertIn("general", flux_out.lower())

        raw_krea = "The photo features an astronaut on Mars with an overall cinematic composition."
        krea_out = format_krea2_prompt(raw_krea)
        self.assertNotIn("overall", krea_out.lower())
        self.assertIn("general", krea_out.lower())

        # 3. Test execute_describe_core end-to-end sanitization
        mock_interaction = MagicMock()
        mock_interaction.user.id = 77777
        mock_interaction.user.name = "DescribeUser"
        mock_interaction.response.send_message = AsyncMock()

        mock_image = MagicMock()
        mock_image.content_type = "image/png"
        mock_image.filename = "test_overall.png"
        mock_image.url = "https://example.com/test_overall.png"
        mock_image.read = AsyncMock(return_value=b"fake_image_bytes")

        mock_vision_res = {
            "caption": "an overall sunny day, 1girl, smiling, overalls",
            "sdxl_prompt": "an overall sunny day, 1girl, smiling, overalls",
            "flux_prompt": "An overall sunny day with a girl smiling in overalls",
            "krea2_prompt": "An overall sunny day with a girl smiling in overalls",
            "engine_used": "JoyCaption"
        }

        with patch("services.vision_service.run_vision_interrogate", new=AsyncMock(return_value=mock_vision_res)), \
             patch("services.vision_service.edit_original_fallback", new=AsyncMock()) as mock_edit:
            asyncio.run(execute_describe_core(mock_interaction, mock_image, model="joycaption"))

            mock_edit.assert_called_once()
            call_kwargs = mock_edit.call_args[1]
            embed = call_kwargs["embed"]
            desc_val = embed.fields[0].value
            self.assertNotIn("overall ", desc_val.lower())
            self.assertIn("general", desc_val.lower())
            # 'overalls' (the clothing) should be preserved
            self.assertIn("overalls", desc_val.lower())

    def test_deduplicate_intro_quality_tags(self):
        """Test deduplicate_intro_quality_tags removes duplicate intro lines."""
        from parsers import deduplicate_intro_quality_tags

        # Redundant duplicate from blend-sdxl screenshot
        prompt1 = "masterpiece, best quality, absurdres. Semi-realism, masterpiece, best quality. A digital painting of a young woman with pale skin"
        res1 = deduplicate_intro_quality_tags(prompt1)
        self.assertNotIn("Semi-realism, masterpiece, best quality", res1)
        self.assertEqual(res1.count("masterpiece, best quality"), 1)
        self.assertTrue(res1.startswith("masterpiece, best quality, absurdres. Semi-realism,"))

        # Reversed order
        prompt2 = "Semi-realism, masterpiece, best quality. masterpiece, best quality, absurdres. A warrior in battle"
        res2 = deduplicate_intro_quality_tags(prompt2)
        self.assertEqual(res2.count("masterpiece, best quality"), 1)

        # Clean prompt unchanged
        prompt3 = "masterpiece, best quality, absurdres. A peaceful village"
        res3 = deduplicate_intro_quality_tags(prompt3)
        self.assertEqual(res3, prompt3)

    def test_blend_prompt_no_redundant_intro(self):
        """Test /blend-sdxl generation produces non-redundant prompt without duplicate quality intros."""
        import bot
        from unittest.mock import MagicMock, AsyncMock, patch
        import asyncio

        gen_data = {
            "caption": "A digital painting of a young woman kneeling on a red carpet",
            "detailed_caption": "A digital painting of a young woman kneeling on a red carpet with torn outfit",
            "uploaded_image_name": "blend_test_img.png",
            "ar": "16:9",
            "sr": True,
            "char_choice": "none",
            "model_choice": "hyphoria",
            "comp_strength": "style",
            "sref_rand": "nosref"
        }
        bot.db.save_generation("gen_blend_dedup_test", gen_data)
        bot.active_generations["gen_blend_dedup_test"] = gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_blend_dedup_test", desc_type="blend",
                ar="16:9", use_sr=True, char_choice="none", use_sref_rand="nosref"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            # Verify no redundant 'masterpiece, best quality' in prompt passed to execution
            self.assertNotIn("Semi-realism, masterpiece, best quality", called_prompt)
            self.assertIn("Semi-realism,", called_prompt)

    def test_blend_character_lora_trigger_substitution(self):
        """Test /blend-sdxl with character presets substitutes triggers cleanly without leaving aliases."""
        from parsers import parse_loras

        # Test Valerie trigger substitution
        val_prompt = "Semi-realism, valerie, kneeling on a red carpet --sr.90 --valerie.85 --ar 16:9"
        cleaned_val, loras_val = parse_loras(val_prompt)
        self.assertIn("jen", cleaned_val)
        self.assertNotIn("valerie", cleaned_val)
        self.assertIn(("jen_epoch_5.safetensors", 0.85), loras_val)

        # Test Sully trigger substitution
        sul_prompt = "Semi-realism, sully, reading a scroll in library --sr.90 --sully.85 --ar 16:9"
        cleaned_sul, loras_sul = parse_loras(sul_prompt)
        self.assertIn("susa", cleaned_sul)
        self.assertIn("black hair, thin rim glasses", cleaned_sul)
        self.assertNotIn("sully", cleaned_sul)
        self.assertIn(("susa_epoch_6.safetensors", 0.85), loras_sul)

        # Test Cheri trigger substitution
        che_prompt = "Semi-realism, cheri, walking in the garden --sr.90 --cheri.85 --ar 16:9"
        cleaned_che, loras_che = parse_loras(che_prompt)
        self.assertIn("cheri", cleaned_che)
        self.assertIn("blonde hair", cleaned_che)
        self.assertIn(("cheri_epoch_6.safetensors", 0.85), loras_che)



if __name__ == '__main__':
    unittest.main()

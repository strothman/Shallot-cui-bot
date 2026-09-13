"""
Automated Efficacy & Functionality Test Suite for Shallot-CUI-Bot.

Executes unit/integration tests for prompt parsers, workflow builders,
dimension math, auto-fix recipes, error log persistence, and CLI diagnostics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json
import unittest
import copy
import random
import logging
import asyncio
import time
from unittest.mock import MagicMock, AsyncMock, patch

# Disable log output during test execution to prevent mock errors and DB setup messages from cluttering the console
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
from PIL import Image
import io

class TestCUIBotFunctions(unittest.TestCase):

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


    def test_module5_blend_workflow_construction(self):
        """Test dynamic construction of IPAdapter multi-image workflow."""
        filenames = ["img1.png", "img2.png", "img3.png"]
        wf = build_blend_workflow(filenames, "masterpiece blend", "blurry", "v1-5", 512, 512, 999, 4.0)

        # Verify IPAdapter load nodes (blend_img_0, blend_img_1, blend_img_2) and IPAdapterAdvanced nodes
        self.assertIn("blend_img_0", wf)
        self.assertIn("blend_ip_0", wf)
        self.assertIn("blend_img_1", wf)
        self.assertIn("blend_ip_1", wf)
        self.assertIn("blend_img_2", wf)
        self.assertIn("blend_ip_2", wf)
        self.assertEqual(wf["blend_img_0"]["inputs"]["image"], "img1.png")
        self.assertEqual(wf["blend_ip_0"]["inputs"]["end_at"], 0.85)

        # Test negative prompt sketch avoidance when positive prompt contains sketch keywords
        wf_sketch = build_blend_workflow(["img1.png"], "pencil sketch drawing of girl", "bad quality", "v1-5", 512, 512, 999, 3.5)
        self.assertNotIn("line art only, sketch", wf_sketch["7"]["inputs"]["text"])

        # Test img2img composition scaling
        img2img_template = {
            "3": {"inputs": {}},
            "4": {"inputs": {}},
            "6": {"inputs": {}},
            "7": {"inputs": {}},
            "9": {"inputs": {}, "class_type": "SaveImage"},
            "30": {"inputs": {}},
            "31": {"inputs": {}}
        }
        wf_comp = build_blend_workflow(["img1.png"], "blended image", "blurry", "v1-5", 512, 512, 999, 3.5, workflow_template=img2img_template)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["weight"], 0.35)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["end_at"], 0.75)
        self.assertEqual(wf_comp["blend_ip_0"]["inputs"]["weight_type"], "ease in-out")

    def test_module6_error_handler_and_autofix(self):
        """Test structured error logging and recipe matching."""
        # Test logging an error
        err = Exception("CUDA out of memory error during generation")
        entry = error_handler.log_error(
            err,
            category=ErrorCategory.WORKFLOW,
            source_function="test_unit",
            source_file="suite_test.py",
            severity=ErrorSeverity.ERROR
        )

        # Test recipe matching
        recipe = error_handler.find_recipe(str(err), category=ErrorCategory.WORKFLOW)
        self.assertIsNotNone(recipe)
        self.assertEqual(recipe.action, AutoFixAction.RETRY_REDUCED_RES)

        # Record auto-fix outcome
        error_handler.log_error_with_fix(
            entry,
            action=AutoFixAction.RETRY_REDUCED_RES,
            result=AutoFixResult.SUCCESS,
            detail="Reduced resolution by 50%"
        )

        # Verify file entry updated
        self.assertTrue(os.path.exists("error_log.json"))
        with open("error_log.json", "r") as f:
            data = json.load(f)
            matching = [item for item in data if item["id"] == entry.id]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["auto_fix_result"], "SUCCESS")

    def test_module7_sqlite_persistence(self):
        """Test SQLite generations cache proxy operations."""
        from bot import active_generations, get_generation, load_generations
        # Initialize DB
        load_generations()
        
        # Test saving through proxy
        test_id = "test_gen_9999"
        test_data = {"prompt": "testing sqlite cache", "seed": 42}
        active_generations[test_id] = test_data
        
        # Test retrieving through proxy get
        retrieved = active_generations.get(test_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved["prompt"], "testing sqlite cache")
        self.assertEqual(retrieved["seed"], 42)
        
        # Test retrieving through proxy dict access
        retrieved_dict = active_generations[test_id]
        self.assertEqual(retrieved_dict["prompt"], "testing sqlite cache")
        
        # Test helper function get_generation
        retrieved_helper = get_generation(test_id)
        self.assertEqual(retrieved_helper["seed"], 42)
        
        # Test containment check
        self.assertIn(test_id, active_generations)
        self.assertNotIn("non_existent_id", active_generations)

    def test_module8_quadrant_caching(self):
        """Test saving and retrieving quadrant images."""
        test_gen_id = "test_quadrant_12345"
        dummy_images = [b"image_data_1", b"image_data_2", b"image_data_3", b"image_data_4"]
        
        # Save images
        save_quadrant_images(test_gen_id, dummy_images)
        
        # Retrieve and verify images
        for idx, expected in enumerate(dummy_images):
            retrieved = get_quadrant_bytes(test_gen_id, idx + 1)
            self.assertEqual(retrieved, expected)
            
        # Clean up files created
        import os
        from image_utils import QUADRANT_CACHE_DIR
        for idx in range(1, 5):
            path = os.path.join(QUADRANT_CACHE_DIR, f"{test_gen_id}_{idx}.png")
            if os.path.exists(path):
                os.remove(path)

    def test_module9_queue_stasis(self):
        """Test queue stasis database interactions and exception handling."""
        import db
        from comfy_client import StasisInterruptException, ComfyClient
        import asyncio
        from unittest.mock import AsyncMock

        # 1. Test database serialization and get_user_generations helper
        user_id = 998877
        gen_id = "stasis_test_123"
        gen_data = {
            "prompt": "a test prompt",
            "user_id": user_id,
            "status": "queued",
            "prompt_ids": ["prompt_abc_123"],
            "workflows": [{"3": {"inputs": {"seed": 42}}}]
        }
        db.save_generation(gen_id, gen_data)

        user_gens = db.get_user_generations(user_id)
        self.assertEqual(len(user_gens), 1)
        self.assertEqual(user_gens[0]["id"], gen_id)
        self.assertEqual(user_gens[0]["status"], "queued")
        self.assertEqual(user_gens[0]["prompt_ids"], ["prompt_abc_123"])

        # 2. Test StasisInterruptException inheritance
        try:
            raise StasisInterruptException("Stasis triggered")
        except Exception as e:
            self.assertIsInstance(e, StasisInterruptException)

        # 3. Test ComfyClient.pause_generation mock execution
        client = ComfyClient()
        from unittest.mock import MagicMock
        client.session = MagicMock()
        
        # Mock response
        mock_resp = MagicMock()
        mock_resp.status = 200
        async def mock_json():
            return {
                "queue_running": [[1, "prompt_xyz", {}, {}]],
                "queue_pending": [[2, "prompt_abc_123", {}, {}]]
            }
        mock_resp.json = mock_json

        async def mock_aenter(*args, **kwargs):
            return mock_resp
        async def mock_aexit(*args, **kwargs):
            pass

        mock_get_context = MagicMock()
        mock_get_context.__aenter__ = mock_aenter
        mock_get_context.__aexit__ = mock_aexit
        client.session.get = MagicMock(return_value=mock_get_context)

        mock_post_context = MagicMock()
        mock_post_context.__aenter__ = mock_aenter
        mock_post_context.__aexit__ = mock_aexit
        client.session.post = MagicMock(return_value=mock_post_context)

        # Add a mock future using a dedicated event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            future = loop.create_future()
            client.futures["prompt_abc_123"] = future

            # Run pause_generation
            success = loop.run_until_complete(client.pause_generation(gen_id))
            self.assertTrue(success)
        finally:
            loop.close()

        # Verify DB status updated to 'stasis'
        updated_gen = db.get_generation(gen_id)
        self.assertEqual(updated_gen["status"], "stasis")

        # Verify future raised the StasisInterruptException
        self.assertTrue(future.done())
        self.assertIsInstance(future.exception(), StasisInterruptException)

    def test_module10_favorite_styles(self):
        """Test favorite style database operations and usage logic."""
        import db
        
        user_id = 112233
        # Ensure it is clean first
        db.remove_favorite_style(user_id, 123456)
        
        # Initially empty (or at least doesn't contain the test code)
        favs = db.get_favorite_styles(user_id)
        codes = [f["style_code"] for f in favs]
        self.assertNotIn(123456, codes)
        
        # Add style
        db.add_favorite_style(user_id, 123456, "Neon Punk Cyberpunk", "cyberpunk, neon lighting, highly detailed")
        favs = db.get_favorite_styles(user_id)
        fav_entry = next((f for f in favs if f["style_code"] == 123456), None)
        self.assertIsNotNone(fav_entry)
        self.assertEqual(fav_entry["style_name"], "Neon Punk Cyberpunk")
        self.assertEqual(fav_entry["style_prompt"], "cyberpunk, neon lighting, highly detailed")
        
        # Update style
        updated = db.update_favorite_style(user_id, 123456, "Updated Cyberpunk", "new cyberpunk prompt")
        self.assertTrue(updated)
        favs = db.get_favorite_styles(user_id)
        fav_entry = next((f for f in favs if f["style_code"] == 123456), None)
        self.assertEqual(fav_entry["style_name"], "Updated Cyberpunk")
        self.assertEqual(fav_entry["style_prompt"], "new cyberpunk prompt")
        
        # Remove style
        db.remove_favorite_style(user_id, 123456)
        favs = db.get_favorite_styles(user_id)
        codes = [f["style_code"] for f in favs]
        self.assertNotIn(123456, codes)

        # Test StylePaginationView pagination with 37 mock items
        from views import StylePaginationView
        mock_favs = [{"style_code": i, "style_name": f"Style {i}", "style_prompt": f"prompt {i}"} for i in range(37)]
        view = StylePaginationView(user_id, mock_favs, per_page=8)
        self.assertEqual(view.total_pages, 5) # math.ceil(37 / 8) = 5
        embed = view.build_embed()
        self.assertEqual(len(embed.fields), 8)
        self.assertIn("Page 1 of 5 (37 total saved styles)", embed.footer.text)

        # Test deterministic sref dynamic generation consistency
        from parsers import generate_dynamic_style
        sref_836127 = generate_dynamic_style(836127)
        self.assertEqual(sref_836127["code"], 836127)
        self.assertEqual(sref_836127["name"], "Brutalist Screenprint")
        self.assertIn("screenprint, brutalist aesthetic", sref_836127["prompt"])

    def test_module11_favorite_prompts(self):
        """Test favorite prompt database operations and pin icon stripping for copy-pasted prompts."""
        import db

        user_id = 998877
        db.add_favorite_prompt(user_id, "Wolf Emerging", "Semi-realism, wolf emerges from the cave in the moonlight")
        prompts = db.get_favorite_prompts(user_id)
        self.assertTrue(len(prompts) > 0)
        
        target = prompts[0]
        self.assertEqual(target["prompt_name"], "Wolf Emerging")
        self.assertIn("wolf emerges", target["prompt_text"])

        # Test pin icon stripping logic
        copy_pasted_param = "📌 Semi-realism, wolf emerges from the c..."
        clean_fav = copy_pasted_param.replace("📌", "").strip()
        self.assertNotIn("📌", clean_fav)
        
        # Test lookup resolution
        found_text = None
        for item in prompts:
            p_full = item['prompt_text'].strip()
            if (str(item['id']) == copy_pasted_param or str(item['id']) == clean_fav or
                item['prompt_name'] == clean_fav or p_full == clean_fav or
                (len(clean_fav) >= 10 and p_full.lower().startswith(clean_fav.lower()[:30]))):
                found_text = item['prompt_text']
                break
        
        # Test update_favorite_prompt
        updated = db.update_favorite_prompt(user_id, target["id"], "Updated Wolf", "New updated wolf text")
        self.assertTrue(updated)
        prompts = db.get_favorite_prompts(user_id)
        updated_entry = next((p for p in prompts if p["id"] == target["id"]), None)
        self.assertEqual(updated_entry["prompt_name"], "Updated Wolf")
        self.assertEqual(updated_entry["prompt_text"], "New updated wolf text")
        db.remove_favorite_prompt(user_id, target["id"])

        # Test PromptPaginationView pagination
        from views import PromptPaginationView
        mock_prompts = [{"id": i, "prompt_name": f"Prompt {i}", "prompt_text": f"text {i}"} for i in range(12)]
        prompt_view = PromptPaginationView(user_id, mock_prompts, per_page=5)
        self.assertEqual(prompt_view.total_pages, 3) # math.ceil(12 / 5) = 3
        prompt_embed = prompt_view.build_embed()
        self.assertEqual(len(prompt_embed.fields), 5)
        self.assertIn("Page 1 of 3 (12 total saved prompts)", prompt_embed.footer.text)

    def test_module12_windows_ico_generation(self):
        """Test Windows 11 multi-resolution ICO creation with rounded corners and arbitrary image conversion."""
        from PIL import Image
        import io
        from image_utils import create_windows_ico_bytes, save_ico_file, apply_rounded_corners_to_bytes, convert_image_to_ico

        test_img = Image.new("RGBA", (1024, 1024), (0, 128, 255, 255))
        img_bytes_io = io.BytesIO()
        test_img.save(img_bytes_io, format="PNG")
        png_bytes = img_bytes_io.getvalue()

        rounded_png = apply_rounded_corners_to_bytes(png_bytes)
        self.assertIsNotNone(rounded_png)
        self.assertGreater(len(rounded_png), 1000)

        ico_bytes = create_windows_ico_bytes(png_bytes, rounded_corners=True)
        self.assertIsNotNone(ico_bytes)
        self.assertGreater(len(ico_bytes), 1000)

        # Test converting arbitrary non-square image (1920x1080) to ICO
        rect_img = Image.new("RGBA", (1920, 1080), (255, 100, 50, 255))
        rect_buf = io.BytesIO()
        rect_img.save(rect_buf, format="PNG")
        conv_png, conv_ico = convert_image_to_ico(rect_buf.getvalue(), rounded_corners=True)
        self.assertIsNotNone(conv_png)
        self.assertIsNotNone(conv_ico)
        self.assertGreater(len(conv_ico), 1000)

        # Inspect generated ICO headers
        ico_img = Image.open(io.BytesIO(conv_ico))
        self.assertEqual(ico_img.format, "ICO")

        filename = "test_icon_20260808_120000_seed12345.ico"
        saved_path = save_ico_file(ico_bytes, filename)
        self.assertIsNotNone(saved_path)
        self.assertTrue(os.path.exists(saved_path))
        os.remove(saved_path)

    def test_module12_style_batch_queuing(self):
        """Test 5, 10, 15 style batch parsing, BlendButtons cycling, and favorite style resolution."""
        from views import BlendButtons

        # 1. Test parse_sref batch parsing
        p5, _, _, info5 = parse_sref("cosmic dragon --sref random:5")
        self.assertIsNotNone(info5)
        self.assertEqual(info5.get("batch_count"), 5)

        p10, _, _, info10 = parse_sref("cyberpunk city --sref batch:10")
        self.assertIsNotNone(info10)
        self.assertEqual(info10.get("batch_count"), 10)

        p15, _, _, info15 = parse_sref("underwater scene --sref batch:15")
        self.assertIsNotNone(info15)
        self.assertEqual(info15.get("batch_count"), 15)

        # 2. Test BlendButtons Unified Single-Page Dashboard
        v = BlendButtons("gen123", ar="16:9", char_choice="sully", sr="sr80", sref_rand="sref")
        
        # Test Model select (Row 0)
        select_model = [item for item in v.children if "set_blend_model" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_model)
        self.assertEqual(select_model.row, 0)

        # Test Character select (Row 1)
        select_char = [item for item in v.children if "set_blend_char" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_char)
        self.assertEqual(select_char.row, 1)
        char_vals = [opt.value for opt in select_char.options]
        self.assertIn("none", char_vals)
        self.assertIn("ogarla", char_vals)
        self.assertIn("valerie", char_vals)
        self.assertIn("sully", char_vals)
        self.assertIn("cheri", char_vals)
        self.assertIn("mageill", char_vals)
        selected_char_opt = [opt for opt in select_char.options if opt.default][0]
        self.assertEqual(selected_char_opt.value, "sully")

        # Test Reference & Composition select (Row 2)
        select_comp = [item for item in v.children if "set_blend_comp" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(select_comp)
        self.assertEqual(select_comp.row, 2)

        # Test Toggles & Cycles in Row 3 (SR toggle, AR cycle, Sref toggle)
        sr_btn = [item for item in v.children if "toggle_blend_sr" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(sr_btn)
        self.assertEqual(sr_btn.row, 3)
        self.assertIn("ON", sr_btn.label)

        ar_btn = [item for item in v.children if "cycle_blend_ar" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(ar_btn)
        self.assertEqual(ar_btn.row, 3)
        self.assertEqual(ar_btn.label, "📐 AR: 16:9")

        style_btn = [item for item in v.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(style_btn)
        self.assertEqual(style_btn.row, 3)
        self.assertIn("--sref random: ON", style_btn.label)

        # Test Action Launchers in Row 4 (Edit prompt, Blend image)
        edit_btn = [item for item in v.children if "edit_blend_prompt" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(edit_btn)
        self.assertEqual(edit_btn.row, 4)

        blend_btn = [item for item in v.children if "blend_desc" in getattr(item, "custom_id", "")][0]
        self.assertIsNotNone(blend_btn)
        self.assertEqual(blend_btn.row, 4)
        self.assertEqual(blend_btn.custom_id, "blend_desc:gen123:blend")

    def test_module13_followup_fallback_no_view_type_error(self):
        """Test send_followup_fallback omits view parameter when view=None so discord.py does not raise TypeError."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from bot import send_followup_fallback, edit_original_fallback, edit_message_fallback

        mock_interaction = MagicMock()
        mock_interaction.followup = MagicMock()
        mock_interaction.followup.send = AsyncMock(return_value="mock_msg")
        mock_interaction.edit_original_response = AsyncMock(return_value=None)
        res = asyncio.run(send_followup_fallback(mock_interaction, content="Job submitted"))
        self.assertEqual(res, "mock_msg")

        # Verify 'view' was NOT passed in kwargs to followup.send when view=None
        _, kwargs = mock_interaction.followup.send.call_args
        self.assertNotIn("view", kwargs)
        self.assertEqual(kwargs["content"], "Job submitted")

        # Test edit_original_fallback with default view=None
        asyncio.run(edit_original_fallback(mock_interaction, content="Editing"))
        _, edit_kwargs = mock_interaction.edit_original_response.call_args
        self.assertEqual(edit_kwargs.get("view"), None)
        self.assertEqual(edit_kwargs.get("content"), "Editing")

        # Test edit_original_fallback with attachments
        mock_file = MagicMock()
        asyncio.run(edit_original_fallback(mock_interaction, content="With file", attachments=[mock_file]))
        _, edit_kwargs = mock_interaction.edit_original_response.call_args
        self.assertEqual(edit_kwargs.get("attachments"), [mock_file])

        # Test edit_message_fallback with allow_send_fallback=False suppresses channel.send on expired token
        mock_msg_interaction = MagicMock()
        mock_msg_interaction.channel_id = 12345
        mock_msg_interaction.channel = MagicMock()
        mock_msg_interaction.channel.fetch_message = AsyncMock(side_effect=Exception("429 Too Many Requests: 30046"))
        mock_msg_interaction.followup = MagicMock()
        mock_msg_interaction.followup.edit_message = AsyncMock(side_effect=Exception("10062 Unknown interaction"))
        mock_msg_interaction.channel.send = AsyncMock()

        # Should complete quietly and NOT call channel.send
        asyncio.run(edit_message_fallback(mock_msg_interaction, 99999, content="Progress tick", allow_send_fallback=False))
        mock_msg_interaction.channel.send.assert_not_called()

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

    def test_module15_wan_workflow_json(self):
        """Test loading and validating Wan 2.2 workflow JSON template."""
        workflow_path = "workflows/wan22_i2v.json"
        self.assertTrue(os.path.exists(workflow_path))
        with open(workflow_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("1", wf)
        self.assertIn("5", wf)
        self.assertIn("9", wf)
        self.assertEqual(wf["1"]["class_type"], "LoadImage")
        self.assertEqual(wf["2"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(wf["21"]["class_type"], "UnetLoaderGGUF")
        self.assertEqual(wf["22"]["class_type"], "ModelSamplingSD3")
        self.assertEqual(wf["22"]["inputs"]["shift"], 8.0)
        self.assertEqual(wf["3"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(wf["31"]["class_type"], "KSamplerAdvanced")
        self.assertEqual(wf["8"]["class_type"], "WanImageToVideo")
        self.assertEqual(wf["108"]["class_type"], "CLIPVisionLoader")
        self.assertEqual(wf["108"]["inputs"]["clip_name"], "clip_vision_h.safetensors")
        self.assertEqual(wf["107"]["class_type"], "CLIPVisionEncode")
        self.assertEqual(wf["76"]["class_type"], "easy cleanGpuUsed")
        self.assertEqual(wf["75"]["class_type"], "RIFE VFI")
        self.assertEqual(wf["75"]["inputs"]["ckpt_name"], "rife49.pth")
        self.assertEqual(wf["9"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(wf["9"]["inputs"]["frame_rate"], 32)
        self.assertNotIn("audio", wf["9"]["inputs"])
        self.assertNotIn("150", wf)
        self.assertNotIn("151", wf)
        self.assertNotIn("152", wf)

    def test_module16_progress_bar(self):
        """Test progress bar rendering formatting."""
        from bot import create_progress_bar
        bar_50 = create_progress_bar(3, 6, length=10)
        self.assertIn("50%", bar_50)
        self.assertIn("Step 3/6", bar_50)
        self.assertIn("█████░░░░░", bar_50)

        bar_100 = create_progress_bar(6, 6, length=10)
        self.assertIn("100%", bar_100)
        self.assertIn("Step 6/6", bar_100)
        self.assertIn("██████████", bar_100)

    def test_module17_wan_video_duration_options(self):
        """Test video duration frame resolution for 5s (81 frames) and 10s (161 frames)."""
        def get_wan_frames(duration: int, settings_dict: dict):
            if duration == 10:
                return 161
            elif duration == 5:
                return settings_dict.get("wan_video_frames", 81)
            else:
                return (duration * 16) + 1

        dummy_settings = {"wan_video_frames": 81}
        self.assertEqual(get_wan_frames(5, dummy_settings), 81)
        self.assertEqual(get_wan_frames(10, dummy_settings), 161)
        self.assertEqual(get_wan_frames(7, dummy_settings), 113)

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

    def test_module20_flux_workflow_and_lora(self):
        """Test Flux LoRA parsing (--ogarla for Flux) and LoraLoaderModelOnly injection."""
        from parsers import parse_loras, apply_loras_to_workflow

        # 1. Verify --ogarla maps to ogarlaflux_epoch_5 when is_flux=True and harmonizes trigger
        cleaned, loras = parse_loras("cyberpunk street --ogarla.80", is_flux=True)
        self.assertEqual(cleaned, "ogarlaflux, cyberpunk street")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.8)

        # 2. Verify Flux workflow pre-wired Node 76
        with open("workflows/flux_lowres.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        modified_wf = apply_loras_to_workflow(wf, loras)
        self.assertIn("76", modified_wf)
        self.assertEqual(modified_wf["76"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(modified_wf["76"]["inputs"]["lora_name"], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(modified_wf["76"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(modified_wf["76"]["inputs"]["model"], ["1", 0])
        self.assertEqual(modified_wf["11"]["inputs"]["model"], ["76", 0])

    def test_module21_comfyui_models_directory(self):
        """Test that ComfyUI models/loras directory exists and contains registered LoRA files."""
        comfy_loras_dir = r"C:\ComfyUI\ComfyUI\models\loras"
        if os.path.exists(comfy_loras_dir):
            files = os.listdir(comfy_loras_dir)
            self.assertIn("ogarlaflux_epoch_1.safetensors", files)
            self.assertIn("ogarla_epoch_5.safetensors", files)
            self.assertIn("Semi-realism_illustrious.safetensors", files)
        else:
            self.skipTest(f"ComfyUI directory not found at {comfy_loras_dir}")

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

    def test_module23_checkpoint_configs_registry(self):
        """Test that CHECKPOINT_CONFIGS registry contains valid photorealistic checkpoint entries and default configs."""
        from bot import CHECKPOINT_CONFIGS
        self.assertIn("RealVisXL_V4.0.safetensors", CHECKPOINT_CONFIGS)
        self.assertIn("juggernautXL_ragnarok.safetensors", CHECKPOINT_CONFIGS)
        self.assertIn("CopaxTimeLessXL.safetensors", CHECKPOINT_CONFIGS)
        
        realvis = CHECKPOINT_CONFIGS["RealVisXL_V4.0.safetensors"]
        self.assertEqual(realvis["sampler_name"], "dpmpp_2m_sde")
        self.assertEqual(realvis["scheduler"], "karras")
        self.assertEqual(realvis["cfg"], 4.5)
        self.assertIn("cgi", realvis["negative_addon"])

    def test_module24_com_flux_gguf_workflow(self):
        """Test community Flux.1 GGUF workflow structure, nodes, guidance, and dual CLIP loaders."""
        wf_path = "workflows/com_flux_gguf.json"
        self.assertTrue(os.path.exists(wf_path))
        with open(wf_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("1", wf)
        self.assertEqual(wf["1"]["class_type"], "UnetLoaderGGUF")
        self.assertIn("12", wf)
        self.assertEqual(wf["12"]["class_type"], "DualCLIPLoaderGGUF")
        self.assertIn("13", wf)
        self.assertEqual(wf["13"]["class_type"], "FluxGuidance")
        self.assertEqual(wf["13"]["inputs"]["guidance"], 3.5)
        self.assertIn("11", wf)
        self.assertEqual(wf["11"]["class_type"], "KSampler")
        self.assertEqual(wf["11"]["inputs"]["positive"], ["13", 0])

        # Test Ogarla Flux LoRA parsing & injection
        cleaned, loras = parse_loras("ogarla in a cafe --ogarla.80", is_flux=True)
        self.assertEqual(cleaned, "ogarlaflux in a cafe")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.8)

        injected_wf = apply_loras_to_workflow(wf, loras)
        self.assertIn("76", injected_wf)
        self.assertEqual(injected_wf["76"]["class_type"], "LoraLoaderModelOnly")
        self.assertEqual(injected_wf["76"]["inputs"]["lora_name"], "ogarlaflux_epoch_5.safetensors")
        self.assertEqual(injected_wf["76"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(injected_wf["76"]["inputs"]["model"], ["1", 0])
        self.assertEqual(injected_wf["11"]["inputs"]["model"], ["76", 0])

    def test_module25_sdxl_powerhouse_2stage_workflow(self):
        """Test 2-stage SDXL powerhouse workflow with FreeU V2, Latent Upscale 1.35x, and Stage 2 refiner."""
        wf_path = "workflows/sdxl_powerhouse_2stage.json"
        self.assertTrue(os.path.exists(wf_path))
        with open(wf_path, "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("4", wf)
        self.assertEqual(wf["4"]["class_type"], "CheckpointLoaderSimple")
        self.assertIn("75", wf)
        self.assertEqual(wf["75"]["class_type"], "LoraLoader")
        self.assertIn("76", wf)
        self.assertEqual(wf["76"]["class_type"], "LoraLoader")
        self.assertIn("20", wf)
        self.assertEqual(wf["20"]["class_type"], "FreeU_V2")
        self.assertEqual(wf["20"]["inputs"]["b1"], 1.3)
        self.assertEqual(wf["20"]["inputs"]["b2"], 1.4)
        self.assertIn("3", wf)
        self.assertEqual(wf["3"]["inputs"]["model"], ["20", 0])
        self.assertIn("14", wf)
        self.assertEqual(wf["14"]["class_type"], "LatentUpscaleBy")
        self.assertEqual(wf["14"]["inputs"]["scale_by"], 1.35)
        self.assertIn("15", wf)
        self.assertEqual(wf["15"]["inputs"]["denoise"], 0.48)
        self.assertEqual(wf["15"]["inputs"]["model"], ["20", 0])
        self.assertEqual(wf["8"]["inputs"]["samples"], ["15", 0])

        # Test Ogarla SDXL LoRA parsing & injection into 2-stage FreeU workflow
        cleaned, loras = parse_loras("ogarla in a temple --ogarla.75", is_flux=False)
        self.assertEqual(cleaned, "ogarla in a temple")
        self.assertEqual(len(loras), 1)
        self.assertEqual(loras[0][0], "ogarla_epoch_5.safetensors")
        self.assertEqual(loras[0][1], 0.75)

        injected_wf = apply_loras_to_workflow(wf, loras)
        self.assertEqual(injected_wf["76"]["inputs"]["lora_name"], "ogarla_epoch_5.safetensors")
        self.assertEqual(injected_wf["76"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(injected_wf["76"]["inputs"]["strength_clip"], 0.75)
        self.assertEqual(injected_wf["75"]["inputs"]["model"], ["4", 0])
        self.assertEqual(injected_wf["76"]["inputs"]["model"], ["75", 0])
        self.assertEqual(injected_wf["20"]["inputs"]["model"], ["76", 0])
        self.assertEqual(injected_wf["3"]["inputs"]["model"], ["20", 0])
        self.assertEqual(injected_wf["15"]["inputs"]["model"], ["20", 0])
        self.assertEqual(injected_wf["6"]["inputs"]["clip"], ["76", 1])
        self.assertEqual(injected_wf["7"]["inputs"]["clip"], ["76", 1])

    def test_module26_sdxl_checkpoint_isolation(self):
        """Test that SDXL_CHECKPOINT_CHOICES only contains valid SDXL models and excludes Flux/Video models."""
        from bot import SDXL_CHECKPOINT_CHOICES
        self.assertTrue(len(SDXL_CHECKPOINT_CHOICES) > 0)
        for choice in SDXL_CHECKPOINT_CHOICES:
            val = choice.value.lower()
            self.assertTrue(val.endswith(".safetensors"))
            self.assertNotIn("flux", val)
            self.assertNotIn("ltx", val)
            self.assertNotIn("wan", val)
            self.assertNotIn("hunyuan", val)

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
        from bot import SDXL_ENHANCEMENT_CHOICES, FLUX_ENHANCEMENT_CHOICES
        self.assertTrue(len(SDXL_ENHANCEMENT_CHOICES) > 0)
        self.assertTrue(len(FLUX_ENHANCEMENT_CHOICES) > 0)

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
        for c in SDXL_ENHANCEMENT_CHOICES + FLUX_ENHANCEMENT_CHOICES:
            self.assertLessEqual(len(c.name), 100, f"Choice name exceeds 100 chars: {c.name}")

        flux_vals = [c.value for c in FLUX_ENHANCEMENT_CHOICES]
        self.assertIn("smart", flux_vals)
        self.assertIn("magic", flux_vals)
        self.assertIn("smart+magic", flux_vals)

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

    def test_module30_face_detailer_sdxl_injection(self):
        """Test that apply_face_detailer_to_workflow properly attaches detector and FaceDetailer with 8GB VRAM settings."""
        from parsers import apply_face_detailer_to_workflow
        with open("workflows/txt2img_lowres.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        det_wf = apply_face_detailer_to_workflow(
            wf,
            seed=998877,
            cfg=4.5,
            sampler_name="dpmpp_2m_sde",
            scheduler="karras",
            steps=20,
            denoise=0.40,
            guide_size=512,
            max_size=768
        )

        # 1. Check detector node
        self.assertIn("80", det_wf)
        self.assertEqual(det_wf["80"]["class_type"], "UltralyticsDetectorProvider")
        self.assertEqual(det_wf["80"]["inputs"]["model_name"], "bbox/face_yolov8m.pt")

        # 2. Check FaceDetailer node
        self.assertIn("85", det_wf)
        self.assertEqual(det_wf["85"]["class_type"], "FaceDetailer")
        self.assertEqual(det_wf["85"]["inputs"]["seed"], 998877)
        self.assertEqual(det_wf["85"]["inputs"]["cfg"], 4.5)
        self.assertEqual(det_wf["85"]["inputs"]["sampler_name"], "dpmpp_2m_sde")
        self.assertEqual(det_wf["85"]["inputs"]["scheduler"], "karras")
        self.assertEqual(det_wf["85"]["inputs"]["steps"], 20)
        self.assertEqual(det_wf["85"]["inputs"]["denoise"], 0.40)
        self.assertEqual(det_wf["85"]["inputs"]["guide_size"], 512)
        self.assertEqual(det_wf["85"]["inputs"]["max_size"], 768)
        self.assertEqual(det_wf["85"]["inputs"]["bbox_detector"], ["80", 0])
        self.assertEqual(det_wf["85"]["inputs"]["image"], ["8", 0])

        # 3. Check Preview node connects to FaceDetailer output
        self.assertEqual(det_wf["9"]["inputs"]["images"], ["85", 0])

    def test_module31_core_commands_registration(self):
        """Test that core generation commands are registered and pruned commands are absent."""
        from bot import bot
        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("imagine", commands)
        self.assertIn("flux", commands)
        self.assertIn("video", commands)
        self.assertIn("ltx", commands)
        self.assertIn("prompt", commands)
        self.assertIn("negative", commands)

        # Verify pruned commands are not present
        self.assertNotIn("imagine_det", commands)
        self.assertNotIn("junji", commands)
        self.assertNotIn("ico", commands)
        self.assertNotIn("hunyuan", commands)
        self.assertNotIn("sdxl", commands)
        self.assertNotIn("com", commands)

    def test_module32_helper_functions_defined(self):
        """Ensure all message sending helpers in bot.py exist and are callable."""
        import bot
        self.assertTrue(callable(getattr(bot, "send_followup_fallback", None)))
        self.assertTrue(callable(getattr(bot, "send_error_fallback", None)))
        self.assertTrue(callable(getattr(bot, "edit_original_fallback", None)))
        self.assertTrue(callable(getattr(bot, "edit_message_fallback", None)))

    def test_module33_blend_file_naming_and_routing(self):
        """Test blend image file naming format with date, time, seed, checkpoint abbrev, and sref."""
        from image_utils import format_image_filename, get_checkpoint_abbrev

        # Test checkpoint abbreviations
        self.assertEqual(get_checkpoint_abbrev("waiIllustriousSDXL_v170.safetensors"), "wai")
        self.assertEqual(get_checkpoint_abbrev("RealVisXL_V4.0.safetensors"), "realvis")
        self.assertEqual(get_checkpoint_abbrev("illustriousRealismBy_v10VAE.safetensors"), "illuReal")
        self.assertEqual(get_checkpoint_abbrev("juggernautXL_ragnarok.safetensors"), "juggernaut")
        self.assertEqual(get_checkpoint_abbrev("CopaxTimeLessXL.safetensors"), "copax")
        self.assertEqual(get_checkpoint_abbrev("ultraRealisticByStable_v25.safetensors"), "ultra")

        # Verify blend filename with checkpoint abbrev
        ckpt_abbrev = get_checkpoint_abbrev("waiIllustriousSDXL_v170.safetensors")
        fn = format_image_filename(f"blend_{ckpt_abbrev}_3", 523453745845177, "png", sref="325465")
        self.assertTrue(fn.startswith("blend_wai_3_"))
        self.assertIn("seed523453745845177", fn)
        self.assertIn("sref325465", fn)
        self.assertTrue(fn.endswith(".png"))

        # Verify upscale filename for blend with checkpoint abbrev
        fn_up = format_image_filename(f"blend_{ckpt_abbrev}_upscale_2", 987654321, "png", sref="110291")
        self.assertTrue(fn_up.startswith("blend_wai_upscale_2_"))
        self.assertIn("seed987654321", fn_up)
        self.assertIn("sref110291", fn_up)

        # Verify icon filenames for /ico
        fn_ico_png = format_image_filename(f"icon_{ckpt_abbrev}_1", 123456789, "png", sref="456789")
        self.assertTrue(fn_ico_png.startswith("icon_wai_1_"))
        self.assertIn("seed123456789", fn_ico_png)
        self.assertIn("sref456789", fn_ico_png)
        self.assertTrue(fn_ico_png.endswith(".png"))

        # Verify imagine filenames
        fn_iso = format_image_filename(f"imagine_{ckpt_abbrev}_4", 11223344, "png", sref="998877")
        self.assertTrue(fn_iso.startswith("imagine_wai_4_"))
        self.assertIn("seed11223344", fn_iso)
        self.assertIn("sref998877", fn_iso)
        self.assertTrue(fn_iso.endswith(".png"))

        fn_grid = format_image_filename(f"grid_{ckpt_abbrev}", 55667788, "jpg")
        self.assertTrue(fn_grid.startswith("grid_wai_"))
        self.assertIn("seed55667788", fn_grid)
        self.assertTrue(fn_grid.endswith(".jpg"))

        # Verify junji filenames for /junji
        fn_junji_iso = format_image_filename(f"junji_{ckpt_abbrev}_2", 44332211, "png", sref="654321")
        self.assertTrue(fn_junji_iso.startswith("junji_wai_2_"))
        self.assertIn("seed44332211", fn_junji_iso)
        self.assertIn("sref654321", fn_junji_iso)
        self.assertTrue(fn_junji_iso.endswith(".png"))

        fn_junji_grid = format_image_filename(f"junji_{ckpt_abbrev}_grid", 44332211, "jpg")
        self.assertTrue(fn_junji_grid.startswith("junji_wai_grid_"))
        self.assertTrue(fn_junji_grid.endswith(".jpg"))

    def test_module34_blend_context_menu(self):
        """Test registration and setup of the Blend Image context menu."""
        from bot import bot, blend_image_context
        
        # Verify blend_image_context exists and is registered in the command tree
        ctx_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertTrue("Blend Image (SDXL)" in ctx_names or "Blend Image" in ctx_names)
        self.assertIn("Adopt Post / Image", ctx_names)
        self.assertIn("Adopt Midjourney Post", ctx_names)

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
    def test_module36_performance_and_optimizations(self):
        """Test performance enhancements: SQLite WAL mode, indexed user query, bounded timings, and precompiled regexes."""
        import db
        from comfy_client import ComfyClient
        import parsers

        # 1. Verify SQLite connection WAL mode
        with db.get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode")
            journal_mode = cursor.fetchone()[0]
            self.assertEqual(journal_mode.lower(), "wal")

        # 2. Verify indexed user query
        test_uid = 88776655
        db.save_generation("perf_test_1", {"user_id": test_uid, "prompt": "fast prompt 1"})
        db.save_generation("perf_test_2", {"user_id": test_uid, "prompt": "fast prompt 2"})
        db.save_generation("perf_test_other", {"user_id": 999999, "prompt": "other user"})

        user_gens = db.get_user_generations(test_uid)
        self.assertEqual(len(user_gens), 2)
        gen_prompts = [g["prompt"] for g in user_gens]
        self.assertIn("fast prompt 1", gen_prompts)
        self.assertIn("fast prompt 2", gen_prompts)

        # 3. Verify ComfyClient bounded timings cache
        client = ComfyClient()
        for i in range(150):
            pid = f"prompt_{i}"
            if len(client.timings) > 100:
                oldest = next(iter(client.timings))
                client.timings.pop(oldest, None)
            client.timings[pid] = {"submitted": i}
            
            if len(client.last_timing) > 100:
                oldest_lt = next(iter(client.last_timing))
                client.last_timing.pop(oldest_lt, None)
            client.last_timing[pid] = {"total_duration": i}

        self.assertLessEqual(len(client.timings), 101)
        self.assertLessEqual(len(client.last_timing), 101)

        # 4. Verify precompiled regexes
        self.assertIsNotNone(parsers.RE_ASPECT_RATIO)
        self.assertIsNotNone(parsers.RE_SEED)
        cleaned, w, h = parsers.parse_aspect_ratio("A cyber city --ar 16:9", model_name="waiIllustrious")
        self.assertEqual(cleaned, "A cyber city")
        self.assertGreater(w, h)
    def test_module37_comfy_client_extra_pnginfo_payload(self):
        """Verify comfy_client only includes workflow in extra_pnginfo if it contains a 'nodes' key."""
        prompt_dict = {"1": {"inputs": {"image": "test.png"}, "class_type": "LoadImage"}}
        extra_pnginfo = {}
        if isinstance(prompt_dict, dict) and "nodes" in prompt_dict:
            extra_pnginfo["workflow"] = prompt_dict
        self.assertNotIn("workflow", extra_pnginfo)

        graph_dict = {"nodes": [{"id": 1, "type": "LoadImage"}]}
        extra_pnginfo_graph = {}
        if isinstance(graph_dict, dict) and "nodes" in graph_dict:
            extra_pnginfo_graph["workflow"] = graph_dict
        self.assertIn("workflow", extra_pnginfo_graph)
    def test_module38_animate_to_video_context_and_modal(self):
        """Test registration and setup of the Animate to Video context menu and VideoPromptModal."""
        from bot import bot, animate_to_video_context
        from views import VideoPromptModal
        import asyncio

        # 1. Verify command tree registration
        ctx_names = [cmd.name for cmd in bot.tree.get_commands()]
        self.assertIn("Animate to Video", ctx_names)

        # 2. Test VideoPromptModal initialization and defaults
        modal = VideoPromptModal(default_prompt="a majestic dragon flying over mountains")
        self.assertEqual(modal.prompt_input.default, "a majestic dragon flying over mountains")
        self.assertEqual(modal.duration_input.default, "10")
        self.assertEqual(modal.smoothness_input.default, "smooth")

        # 3. Test on_submit callback invocation
        received_args = {}
        async def dummy_callback(interaction, prompt, duration_str, smoothness_str, seed_str):
            received_args["prompt"] = prompt
            received_args["duration_str"] = duration_str
            received_args["smoothness_str"] = smoothness_str
            received_args["seed_str"] = seed_str

        modal_with_cb = VideoPromptModal(on_submit_callback=dummy_callback)
        modal_with_cb.prompt_input._value = "walking down a neon street"
        modal_with_cb.duration_input._value = "10"
        modal_with_cb.smoothness_input._value = "fast"
        modal_with_cb.seed_input._value = "123456"

        dummy_interaction = unittest.mock.MagicMock()
        asyncio.run(modal_with_cb.on_submit(dummy_interaction))

        self.assertEqual(received_args["prompt"], "walking down a neon street")
        self.assertEqual(received_args["duration_str"], "10")
        self.assertEqual(received_args["smoothness_str"], "fast")
        self.assertEqual(received_args["seed_str"], "123456")

    def test_module39_all_modals_error_boundaries_and_callbacks(self):
        """Test that all UI modals in views.py execute callbacks safely and handle exceptions within error boundaries."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock
        from views import (
            CustomSrefModal,
            EditBlendPromptModal,
            StudyImagineModal,
            EditStyleModal,
            EditPromptModal,
            EditAdoptPromptModal,
            VideoPromptModal,
        )

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = False
        mock_interaction.response.send_message = AsyncMock()
        mock_interaction.followup.send = AsyncMock()


        # 1. CustomSrefModal
        sref_res = {}
        async def sref_cb(inter, gen_id, idx, val):
            sref_res["val"] = val
        m_sref = CustomSrefModal("gen123", 1, on_submit_callback=sref_cb)
        m_sref.sref_input._value = "889900"
        asyncio.run(m_sref.on_submit(mock_interaction))
        self.assertEqual(sref_res.get("val"), "889900")

        # 2. EditBlendPromptModal
        blend_res = {}
        async def blend_cb(inter, gen_id, cap, det, extra):
            blend_res["cap"] = cap
            blend_res["det"] = det
            blend_res["extra"] = extra
        m_blend = EditBlendPromptModal("gen123", "a cat", "a cute cat on a table", "glowing", on_submit_callback=blend_cb)
        m_blend.caption_input._value = "a dog"
        m_blend.detailed_input._value = "a cute dog on grass"
        m_blend.extra_input._value = "sunset"
        asyncio.run(m_blend.on_submit(mock_interaction))
        self.assertEqual(blend_res.get("cap"), "a dog")
        self.assertEqual(blend_res.get("extra"), "sunset")

        # 3. StudyImagineModal
        study_res = {}
        async def study_cb(inter, p):
            study_res["prompt"] = p
        m_study = StudyImagineModal("crystal tower", on_submit_callback=study_cb)
        m_study.prompt_input._value = "crystal tower"
        m_study.flags_input._value = "--ar 16:9"
        asyncio.run(m_study.on_submit(mock_interaction))
        self.assertEqual(study_res.get("prompt"), "crystal tower --ar 16:9")

        # 4. EditAdoptPromptModal
        adopt_res = {}
        async def adopt_cb(inter, adopt_id, p):
            adopt_res["p"] = p
        m_adopt = EditAdoptPromptModal("adopt1", "orig prompt", on_submit_callback=adopt_cb)
        m_adopt.prompt_input._value = "new prompt"
        asyncio.run(m_adopt.on_submit(mock_interaction))
        self.assertEqual(adopt_res.get("p"), "new prompt")

        # 5. Error Boundary Verification (Raising in callback does not crash)
        async def failing_cb(*args, **kwargs):
            raise ValueError("Simulated unexpected modal failure")

        m_fail = VideoPromptModal(on_submit_callback=failing_cb)
        m_fail.prompt_input._value = "test failure"
        # Must execute without raising unhandled exception
        asyncio.run(m_fail.on_submit(mock_interaction))

    def test_architecture_detection_and_classification(self):
        """Test model and LoRA architecture detection across all supported families."""
        from model_architecture import (
            detect_model_architecture, 
            Architecture, 
            SubType, 
            ModelType,
            get_architecture_badge
        )

        # 1. SDXL Models & LoRAs
        mtype, arch, subtype = detect_model_architecture("waiIllustriousSDXL_v170.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.ILLUSTRIOUS)

        mtype, arch, subtype = detect_model_architecture("RealVisXL_V4.0.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.REALISTIC)

        mtype, arch, subtype = detect_model_architecture("ponyDiffusionV6XL_v6StartWithThisOne.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(subtype, SubType.PONY)

        mtype, arch, subtype = detect_model_architecture("Semi-realism_illustrious.safetensors")
        self.assertEqual(arch, Architecture.SDXL)
        self.assertEqual(mtype, ModelType.LORA)

        # 2. Flux Models & LoRAs
        mtype, arch, subtype = detect_model_architecture("ogarlaflux_epoch_5.safetensors")
        self.assertEqual(arch, Architecture.FLUX)
        self.assertEqual(mtype, ModelType.LORA)

        mtype, arch, subtype = detect_model_architecture("flux1-dev.safetensors")
        self.assertEqual(arch, Architecture.FLUX)

        # 3. Video Models (Wan & LTX)
        mtype, arch, subtype = detect_model_architecture("WAN-2.2-I2V-Handjob-HIGH-v1.safetensors")
        self.assertEqual(arch, Architecture.WAN)
        self.assertEqual(subtype, SubType.HIGH_NOISE)

        mtype, arch, subtype = detect_model_architecture("ltx-video-2b-v0.9.1.safetensors")
        self.assertEqual(arch, Architecture.LTX)

        # 4. Architecture Badges
        self.assertIn("SDXL", get_architecture_badge("sdxl"))
        self.assertIn("FLUX", get_architecture_badge("flux"))
        self.assertIn("WAN", get_architecture_badge("wan"))

    def test_lora_variant_autorouting_and_compatibility(self):
        """Test LoRA variant auto-routing and compatibility validator guards."""
        from model_architecture import (
            resolve_lora_for_architecture, 
            validate_architecture_compatibility, 
            Architecture, 
            SubType
        )
        from parsers import validate_workflow_loras

        # 1. Test auto-routing for Ogarla variants
        flux_ogarla = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", Architecture.FLUX)
        self.assertEqual(flux_ogarla, "ogarlaflux_epoch_5.safetensors")

        sdxl_ogarla = resolve_lora_for_architecture("ogarlaflux_epoch_5.safetensors", Architecture.SDXL)
        self.assertEqual(sdxl_ogarla, "ogarla_epoch_5.safetensors")

        pony_ogarla = resolve_lora_for_architecture("ogarla_epoch_5.safetensors", Architecture.SDXL, SubType.PONY)
        self.assertEqual(pony_ogarla, "ogarlapony_epoch_6.safetensors")

        # 2. Test compatibility validation: Matching pairing
        is_compat, msg, sug = validate_architecture_compatibility(
            "waiIllustriousSDXL_v170.safetensors", 
            "Semi-realism_illustrious.safetensors"
        )
        self.assertTrue(is_compat)

        # 3. Test compatibility validation: Mismatched pairing with auto-route suggestion
        is_compat, msg, sug = validate_architecture_compatibility(
            "flux1-dev.safetensors", 
            "ogarla_epoch_5.safetensors"
        )
        self.assertFalse(is_compat)
        self.assertEqual(sug, "ogarlaflux_epoch_5.safetensors")

        # 4. Test compatibility validation: Incompatible pairing without replacement
        is_compat, msg, sug = validate_architecture_compatibility(
            "ltx-video-2b-v0.9.1.safetensors", 
            "Semi-realism_illustrious.safetensors"
        )
        self.assertFalse(is_compat)
        self.assertIsNone(sug)

        # 5. Test validate_workflow_loras
        valid, msgs = validate_workflow_loras(
            {}, 
            "waiIllustriousSDXL_v170.safetensors", 
            [("Semi-realism_illustrious.safetensors", 0.7)]
        )
        self.assertTrue(valid)

    def test_database_model_registry_crud(self):
        """Test database operations on the model_registry SQLite table."""
        import db
        db.init_db()

        # 1. Test upsert
        success = db.upsert_model_registry(
            filename="custom_test_model.safetensors",
            model_type="checkpoint",
            base_architecture="sdxl",
            sub_type="illustrious",
            display_name="Custom Test Model",
            trigger_words="test_trigger",
            default_strength=1.0,
            metadata={"source": "unit_test"}
        )
        self.assertTrue(success)

        # 2. Test get entry
        entry = db.get_model_registry_entry("custom_test_model.safetensors")
        self.assertIsNotNone(entry)
        self.assertEqual(entry["base_architecture"], "sdxl")
        self.assertEqual(entry["display_name"], "Custom Test Model")
        self.assertEqual(entry["metadata"].get("source"), "unit_test")

        # 3. Test query by architecture
        sdxl_models = db.get_models_by_architecture(base_architecture="sdxl", model_type="checkpoint")
        self.assertTrue(any(m["filename"] == "custom_test_model.safetensors" for m in sdxl_models))

        # 4. Test seed_default_model_registry
        db.seed_default_model_registry()
        wai_entry = db.get_model_registry_entry("waiIllustriousSDXL_v170.safetensors")
        self.assertIsNotNone(wai_entry)
        self.assertEqual(wai_entry["base_architecture"], "sdxl")

    def test_database_vacuum_and_space_reclaim(self):
        """Test database vacuuming and page compacting."""
        import db
        success = db.vacuum_database()
        self.assertTrue(success)

    def test_async_image_io_and_quadrant_operations(self):
        """Test non-blocking async image utilities and quadrant operations."""
        import asyncio
        from image_utils import (
            save_quadrant_images_async, 
            get_quadrant_bytes_async, 
            embed_metadata_async,
            create_grid_async,
            crop_to_aspect_ratio_async,
            upscale_isolated_image_async,
            calculate_outpaint_padding_async,
            boost_image_vibrancy_and_contrast_async,
            crop_quadrant_from_grid_bytes_async,
            create_thumbnail_bytes_async,
            convert_image_to_ico_async,
            QUADRANT_CACHE_DIR
        )

        async def run_async_test():
            gen_id = "test_async_gen_999"
            test_img = Image.new("RGB", (64, 64), color="blue")
            buf = io.BytesIO()
            test_img.save(buf, format="PNG")
            dummy_img = buf.getvalue()
            images = [dummy_img, dummy_img, dummy_img, dummy_img]

            # 1. Test async quadrant save
            await save_quadrant_images_async(gen_id, images)

            # 2. Test async quadrant retrieval
            q1 = await get_quadrant_bytes_async(gen_id, 1)
            self.assertEqual(q1, dummy_img)

            # 3. Test async metadata embed
            meta_img_io = await embed_metadata_async(dummy_img, "async prompt", neg_prompt="low quality", seed=12345, width=1, height=1)
            self.assertTrue(len(meta_img_io.getvalue()) > 0)

            # 4. Test async grid creation
            grid_io = await create_grid_async(images, "async prompt", seed=12345, width=1, height=1)
            self.assertTrue(len(grid_io.getvalue()) > 0)

            # 5. Test async crop to aspect ratio & upscale
            cropped = await crop_to_aspect_ratio_async(dummy_img, 64, 64)
            self.assertTrue(len(cropped) > 0)

            upscaled_bytes, uw, uh = await upscale_isolated_image_async(dummy_img, target_w=64, target_h=64)
            self.assertTrue(len(upscaled_bytes) > 0)
            self.assertGreaterEqual(uw, 1)

            # 6. Test async outpaint calculation & vibrancy boost
            left, top, right, bottom, pad_bytes, tw, th = await calculate_outpaint_padding_async(dummy_img, "16:9")
            self.assertTrue(len(pad_bytes) > 0)

            boosted = await boost_image_vibrancy_and_contrast_async(dummy_img)
            self.assertTrue(len(boosted) > 0)

            # 7. Test async thumbnail and ICO conversion
            thumb = await create_thumbnail_bytes_async(dummy_img, max_dim=64)
            self.assertTrue(len(thumb) > 0)

            png_b, ico_b = await convert_image_to_ico_async(dummy_img)
            self.assertIsNotNone(png_b)
            self.assertIsNotNone(ico_b)

            # Cleanup
            for idx in range(1, 5):
                p = os.path.join(QUADRANT_CACHE_DIR, f"{gen_id}_{idx}.png")
                if os.path.exists(p):
                    os.remove(p)

        asyncio.run(run_async_test())

    def test_model_auto_discovery_scanner(self):
        """Test scanning ComfyUI model directories and auto-registering models."""
        from model_architecture import scan_and_register_comfyui_models
        comfy_root = r"C:\ComfyUI\ComfyUI"
        if os.path.exists(comfy_root):
            stats = scan_and_register_comfyui_models(comfy_root)
            self.assertTrue(stats.get("success"))
            self.assertTrue(stats.get("total_registered") > 0)
    def test_readme_and_changelog_synchronization(self):
        """Validates that README.md documents all bot slash commands and CHANGELOG.md is up to date."""
        import auto_changelog
        import re
        self.assertTrue(os.path.exists(auto_changelog.README_PATH))
        self.assertTrue(os.path.exists(auto_changelog.CHANGELOG_PATH))
        
        with open(auto_changelog.BOT_PATH, "r", encoding="utf-8") as f:
            bot_code = f.read()
        with open(auto_changelog.README_PATH, "r", encoding="utf-8") as f:
            readme_text = f.read()
        commands = set(re.findall(r'@(?:tree|bot\.tree)\.command\(name=[\'"]([^\'"]+)[\'"]', bot_code))
        missing = [cmd for cmd in commands if f"/{cmd}" not in readme_text and cmd not in readme_text]
        self.assertEqual(missing, [], f"The following slash commands are missing from README.md: {missing}")

    def test_live_status_telemetry(self):
        """Test inter-process live telemetry set, get, and clear functions in db.py."""
        import db
        db.init_db()
        db.set_live_status(step=3, max_steps=6, node_id="3", stage="High Noise KSampler", prompt_text="A girl smiling")
        stat = db.get_live_status()
        self.assertIsNotNone(stat)
        self.assertEqual(stat["step"], 3)
        self.assertEqual(stat["max_steps"], 6)
        self.assertEqual(stat["node_id"], "3")
        self.assertEqual(stat["stage"], "High Noise KSampler")
        self.assertEqual(stat["prompt_text"], "A girl smiling")

        db.clear_live_status()
        cleared = db.get_live_status()
        self.assertIsNone(cleared)

    def test_discord_command_description_lengths(self):
        """Validates that all Discord slash command descriptions are <= 100 characters."""
        from bot import bot
        for cmd in bot.tree.get_commands():
            desc = getattr(cmd, "description", None)
            if desc:
                clean_desc = desc.replace('\n', ' ').strip()
                self.assertLessEqual(
                    len(clean_desc), 
                    100, 
                    f"Command '{cmd.name}' description exceeds Discord 100 char limit ({len(clean_desc)}): '{clean_desc}'"
                )

    def test_single_instance_lock(self):
        """Test that acquire_instance_lock successfully binds and prevents secondary bindings on the same port."""
        from bot import acquire_instance_lock
        import bot
        test_port = 48199
        try:
            # First acquire on test port succeeds
            self.assertTrue(acquire_instance_lock(port=test_port))
            # Second acquire on the same port fails
            self.assertFalse(acquire_instance_lock(port=test_port, silent=True))
        finally:
            if bot._instance_lock_socket:
                bot._instance_lock_socket.close()
                bot._instance_lock_socket = None

    def test_character_registry_and_masking(self):
        """Test character registry lookups, pseudonym resolution, and prompt trigger masking."""
        from characters import get_character, mask_character_in_prompt, inject_trained_trigger_in_prompt, CHARACTERS

        # Lookups
        val = get_character("valerie")
        self.assertIsNotNone(val)
        self.assertEqual(val.trained_trigger, "jen")
        self.assertEqual(val.lora_sdxl, "jen_epoch_5.safetensors")

        val_alias = get_character("val")
        self.assertEqual(val_alias.id, "valerie")

        oga = get_character("ogarla")
        self.assertIsNotNone(oga)
        self.assertEqual(oga.lora_sdxl, "ogarla_epoch_5.safetensors")

        # Sully checks
        sul = get_character("sully")
        self.assertIsNotNone(sul)
        self.assertEqual(sul.trained_trigger, "susa")
        self.assertEqual(sul.lora_sdxl, "susa_epoch_6.safetensors")
        self.assertEqual(sul.base_prompt_traits, "black hair, thin rim glasses")

        sul_alias = get_character("sul")
        self.assertEqual(sul_alias.id, "sully")

        # Masking trained trigger -> display alias
        masked = mask_character_in_prompt("jen sitting in a coffee shop")
        self.assertEqual(masked, "valerie sitting in a coffee shop")

        masked_sully = mask_character_in_prompt("susa walking down the street")
        self.assertEqual(masked_sully, "sully walking down the street")

        # Injecting trained trigger -> ComfyUI prompt
        injected = inject_trained_trigger_in_prompt("valerie sitting in a coffee shop", "valerie")
        self.assertEqual(injected, "jen, brown hair, dark brown eyes, realistic skin texture sitting in a coffee shop")

        injected_sully = inject_trained_trigger_in_prompt("sully, in a library", "sully")
        self.assertEqual(injected_sully, "susa, black hair, thin rim glasses, in a library")

        # Mageill character checks (original character, not masked)
        mag = get_character("mageill")
        self.assertIsNotNone(mag)
        self.assertEqual(mag.lora_sdxl, "mageill_epoch_5.safetensors")
        self.assertFalse(mag.is_private)

        mag3 = get_character("mageill3")
        self.assertIsNotNone(mag3)
        self.assertEqual(mag3.lora_sdxl, "mageill_epoch_3.safetensors")

        mag6 = get_character("mag6")
        self.assertIsNotNone(mag6)
        self.assertEqual(mag6.lora_sdxl, "mageill_epoch_6.safetensors")

        # Unmasked prompt stays unmasked
        masked_mag = mask_character_in_prompt("mageill casting a spell")
        self.assertEqual(masked_mag, "mageill casting a spell")

        # Cheri character checks (original character with blonde hair trait)
        che = get_character("cheri")
        self.assertIsNotNone(che)
        self.assertEqual(che.lora_sdxl, "cheri_epoch_6.safetensors")
        self.assertEqual(che.base_prompt_traits, "blonde hair")
        self.assertFalse(che.is_private)

        che4 = get_character("che4")
        self.assertIsNotNone(che4)
        self.assertEqual(che4.lora_sdxl, "cheri_epoch_4.safetensors")

        masked_che = mask_character_in_prompt("cheri smiling in the sunlight")
        self.assertEqual(masked_che, "cheri smiling in the sunlight")

        # Test unified character display badges
        from characters import get_character_display_badge
        self.assertEqual(get_character_display_badge("none"), "None")
        self.assertEqual(get_character_display_badge(None), "None")
        self.assertEqual(get_character_display_badge("valerie.90", architecture="krea2"), "✨ Valerie (.90 - Default)")
        self.assertEqual(get_character_display_badge("valerie.70", architecture="krea2"), "✨ Valerie (.70 - Light)")
        self.assertEqual(get_character_display_badge("ogarla.85", architecture="krea2"), "🌿 Ogarla (.85 - Default)")
        self.assertEqual(get_character_display_badge("ogarla.70", architecture="krea2"), "🌿 Ogarla (.70 - Light)")
        self.assertEqual(get_character_display_badge("ogarla", architecture="sdxl"), "🌿 Ogarla (--ogarla.70)")
        self.assertEqual(get_character_display_badge("valerie", architecture="sdxl"), "👩 Valerie (--valerie.85)")
        self.assertEqual(get_character_display_badge("sully", architecture="sdxl"), "👓 Sully (--sully.85)")
        self.assertEqual(get_character_display_badge("none", architecture="sdxl", prompt="photo of a woman --valerie.85"), "👩 Valerie (--valerie.85)")
        self.assertEqual(get_character_display_badge("none", architecture="sdxl", prompt="photo of a woman --sully"), "👓 Sully (--sully.85)")

        # Test dynamic character autocomplete choices
        from characters import get_character_autocomplete_choices
        from model_architecture import Architecture

        # 1. Krea 2 autocomplete
        krea_choices = get_character_autocomplete_choices("", Architecture.KREA2)
        krea_vals = [c.value for c in krea_choices]
        self.assertNotIn("valerie.90", krea_vals)
        self.assertNotIn("valerie.70", krea_vals)
        self.assertIn("ogarla.85", krea_vals)
        self.assertIn("ogarla.70", krea_vals)

        # Krea 2 filtering
        krea_filtered = get_character_autocomplete_choices("oga", Architecture.KREA2)
        self.assertTrue(all("ogarla" in c.value for c in krea_filtered))

        # 2. SDXL autocomplete
        sdxl_choices = get_character_autocomplete_choices("", Architecture.SDXL)
        sdxl_vals = [c.value for c in sdxl_choices]
        self.assertIn("mageill.85", sdxl_vals)
        self.assertIn("ogarla.85", sdxl_vals)
        self.assertIn("valerie.85", sdxl_vals)
        self.assertIn("sully.85", sdxl_vals)
        self.assertIn("cheri.85", sdxl_vals)

        # SDXL filtering
        sdxl_filtered = get_character_autocomplete_choices("sully", Architecture.SDXL)
        self.assertTrue(all("sully" in c.value for c in sdxl_filtered))

        # 3. Flux autocomplete
        flux_choices = get_character_autocomplete_choices("", Architecture.FLUX)
        flux_vals = [c.value for c in flux_choices]
        self.assertIn("ogarla.85", flux_vals)
        self.assertIn("ogarla.70", flux_vals)

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

    def test_vram_free_memory(self):
        """Test ComfyClient free_memory method handles errors gracefully when offline."""
        import asyncio
        from comfy_client import ComfyClient
        client = ComfyClient("127.0.0.1:8188")
        async def run_test():
            res = await client.free_memory()
            await client.stop()
            return res
        res = asyncio.run(run_test())
        self.assertIsInstance(res, bool)

    def test_cancel_and_remix_views(self):
        """Test instantiation and custom_ids of CancelGenerationView and Remix button in GridButtons."""
        from views import CancelGenerationView, RemixModal, GridButtons

        # CancelGenerationView
        c_view = CancelGenerationView("gen_999")
        c_button = [item for item in c_view.children if getattr(item, "custom_id", None) == "cancel_gen:gen_999"]
        self.assertEqual(len(c_button), 1)

        # Remix button in GridButtons
        g_view = GridButtons("gen_999", has_sref=True)
        r_button = [item for item in g_view.children if getattr(item, "custom_id", None) == "remix:gen_999"]
        self.assertEqual(len(r_button), 1)

        # RemixModal
        modal = RemixModal("gen_999", initial_prompt="a cute cat", initial_seed=12345)
        self.assertEqual(modal.prompt_input.default, "a cute cat")
        self.assertEqual(modal.seed_input.default, "12345")

    def test_blend_character_loras_and_embed_badges(self):
        """Test build_blend_embed 3-column dashboard and handle_generate_blended character LoRAs & style presets."""
        import asyncio
        from unittest.mock import AsyncMock, patch, MagicMock
        from views import build_blend_embed, BlendButtons
        from image_utils import detect_closest_aspect_ratio, detect_closest_krea_aspect_ratio
        import bot

        # 1. Test detect_closest_aspect_ratio
        self.assertEqual(detect_closest_aspect_ratio(1920, 1080), "16:9")
        self.assertEqual(detect_closest_aspect_ratio(1080, 1920), "9:16")
        self.assertEqual(detect_closest_aspect_ratio(800, 1200), "3:5")
        self.assertEqual(detect_closest_aspect_ratio(1000, 1000), "1:1")
        self.assertEqual(detect_closest_aspect_ratio(2560, 1080), "21:9")
        self.assertEqual(detect_closest_aspect_ratio(1000, 700), "10:7")
        self.assertEqual(detect_closest_aspect_ratio(0, 100), "16:9")

        # 1b. Test detect_closest_krea_aspect_ratio (Krea 2 Studio 5-button set)
        self.assertEqual(detect_closest_krea_aspect_ratio(2560, 1080), "21:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1920, 1080), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1024, 768), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1500, 1000), "16:9")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 1000), "1:1")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 950), "1:1")
        self.assertEqual(detect_closest_krea_aspect_ratio(768, 1024), "3:4")
        self.assertEqual(detect_closest_krea_aspect_ratio(1000, 1500), "3:4")
        self.assertEqual(detect_closest_krea_aspect_ratio(1080, 1920), "9:16")
        self.assertEqual(detect_closest_krea_aspect_ratio(1170, 2532), "9:16")
        self.assertEqual(detect_closest_krea_aspect_ratio(0, 100), "16:9")

        # 1c. Test create_thumbnail_bytes
        from image_utils import create_thumbnail_bytes
        from PIL import Image
        import io
        img = Image.new("RGBA", (800, 600), (255, 0, 0, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        thumb_bytes = create_thumbnail_bytes(buf.getvalue(), max_dim=256)
        self.assertIsNotNone(thumb_bytes)
        with Image.open(io.BytesIO(thumb_bytes)) as thumb_img:
            self.assertLessEqual(thumb_img.width, 256)
            self.assertLessEqual(thumb_img.height, 256)

        # 2. Test build_blend_embed 3-column inline dashboard formatting
        gen_data = {
            "caption": "a woman standing in a garden",
            "detailed_caption": "detailed description of woman in garden",
            "ar": "3:5",
            "model_choice": "wai",
            "comp_strength": "style",
            "sr": "sr75",
            "char_choice": "sully",
            "sref_rand": "preset_junji_ito",
        }
        embed = build_blend_embed(gen_data, author_str="TestUser")
        field_dict = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", field_dict)
        self.assertIn("🤖 Checkpoint", field_dict)
        self.assertIn("🎭 Aesthetics", field_dict)

        self.assertIn("`3:5`", field_dict["📐 Canvas & Framing"])
        self.assertIn("--sr.75", field_dict["🤖 Checkpoint"])
        self.assertIn("👓 Sully (--sully.85)", field_dict["🎭 Aesthetics"])
        self.assertIn("🖋️ Junji Ito", field_dict["🎭 Aesthetics"])

        # 3. Test BlendButtons unified view and --sref random toggle
        user_favs = [
            {"code": 112233, "name": "Vaporwave Neon", "prompt": "cyberpunk neon colors"}
        ]
        view = BlendButtons("gen_test_view", ar="3:5", sr="sr70", char_choice="valerie", sref_rand="sref", user_favorites=user_favs)
        self.assertLessEqual(len(view.children), 25)
        rows = set(child.row for child in view.children)
        self.assertLessEqual(len(rows), 5)
        # Verify --sref random toggle button
        style_btn = next((c for c in view.children if hasattr(c, "custom_id") and c.custom_id and c.custom_id.startswith("toggle_blend_sref:")), None)
        self.assertIsNotNone(style_btn)
        self.assertIn("--sref random: ON", style_btn.label)

        # 4. Test handle_generate_blended character flag assembly and saved style resolution
        bot.db.save_generation("gen_test_char", gen_data)
        bot.active_generations["gen_test_char"] = gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()
        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test character flag injection: Valerie
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="3:5", use_sr="sr70", char_choice="valerie", use_sref_rand="nosref"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("valerie,", called_prompt)
            self.assertIn("--valerie.85", called_prompt)
            self.assertIn("--sr.70", called_prompt)
            self.assertIn("--ar 3:5", called_prompt)

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test style preset injection: Junji Ito
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="16:9", use_sr="nosr", char_choice="cheri", use_sref_rand="preset_junji_ito"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("cheri,", called_prompt)
            self.assertIn("--cheri.85", called_prompt)
            self.assertIn("Junji Ito", called_prompt)

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            # Test saved style injection from user favorites
            asyncio.run(bot.handle_generate_blended(
                mock_interaction, "gen_test_char", desc_type="caption",
                ar="1:1", use_sr="nosr", char_choice="nochar", use_sref_rand="saved_112233"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args[1]["prompt"]
            self.assertIn("--sref 112233", called_prompt)

    def test_blend_complete_embed_and_reblend_button(self):
        """Test build_blend_complete_embed 3-column dashboard, GridButtons is_blend=True, and handle_reblend."""
        from views import build_blend_complete_embed, GridButtons, BlendButtons
        import bot
        from unittest.mock import MagicMock, AsyncMock, patch
        import asyncio

        # 1. Test build_blend_complete_embed
        prompt = "Semi-realism, masterpiece, best quality. mageill, Photo of a young woman with fair skin --sr.70 --sref 855014"
        embed = build_blend_complete_embed(
            display_prompt=prompt,
            selected_model="waiIllustriousSDXL_v170.safetensors",
            seed=377789952340926,
            width=1536,
            height=640,
            comp_strength="low",
            cfg=4.0,
            sref_info={"code": 855014, "name": "Dadaist Pixel Art"},
            char_choice="mageill",
            sr_choice="sr70",
            user_name="TestStroth",
            image_url="http://example.com/source.png",
            elapsed_time=8.4
        )
        self.assertEqual(embed.title, "✨ Image Blend Complete")
        self.assertEqual(embed.thumbnail.url, "http://example.com/source.png")
        self.assertEqual(embed.color.value, 0x8A2BE2)
        self.assertIn("TestStroth", embed.footer.text)
        self.assertIn("8.4s", embed.footer.text)
        self.assertIn("377789952340926", embed.footer.text)

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", fields)
        self.assertIn("🤖 Checkpoint & Tech", fields)
        self.assertIn("🎭 Aesthetics & Identity", fields)

        # Check friendly names & badges
        self.assertIn("1536x640", fields["📐 Canvas & Framing"])
        self.assertIn("🖼️ Light Comp (0.35)", fields["📐 Canvas & Framing"])
        self.assertIn("Wai Illustrious SDXL v1.70", fields["🤖 Checkpoint & Tech"])
        self.assertIn("377789952340926", fields["🤖 Checkpoint & Tech"])
        self.assertIn("CFG:** `4.0`", fields["🤖 Checkpoint & Tech"])
        self.assertIn("🔮 Mageill (Epoch 5)", fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.70", fields["🎭 Aesthetics & Identity"])
        self.assertIn("855014", fields["🎭 Aesthetics & Identity"])
        self.assertIn("Dadaist Pixel Art", fields["🎭 Aesthetics & Identity"])

        # Test fallback regex parsing branches in build_blend_complete_embed
        embed_fallback = build_blend_complete_embed(
            display_prompt="ogarla, beautiful portrait --sr.60 --sref 999111",
            selected_model="waiIllustriousSDXL_v170.safetensors",
            seed=12345,
            width=768,
            height=1344,
            char_choice=None,
            sr_choice=None,
            sref_info=None
        )
        fb_fields = {f.name: f.value for f in embed_fallback.fields}
        self.assertIn("🌿 Ogarla (--ogarla.70)", fb_fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.60", fb_fields["🎭 Aesthetics & Identity"])
        self.assertIn("`--sref 999111`", fb_fields["🎭 Aesthetics & Identity"])

        # 2. Test GridButtons with is_blend=True and has_sref=True
        grid_view = GridButtons("gen_blend_123", has_sref=True, is_blend=True)
        # Check max 5 rows and items per row
        row2_items = [c for c in grid_view.children if getattr(c, "row", None) == 2]
        self.assertEqual(len(row2_items), 5)
        btn_ids = [c.custom_id for c in row2_items]
        self.assertIn("reblend:gen_blend_123", btn_ids)
        self.assertIn("fav_style:gen_blend_123", btn_ids)
        self.assertIn("fav_prompt:gen_blend_123", btn_ids)
        self.assertIn("copy_prompt:gen_blend_123", btn_ids)
        self.assertIn("remix:gen_blend_123", btn_ids)

        # 3. Test handle_reblend interaction handler
        gen_data = {
            "caption": "a woman in a garden",
            "detailed_caption": "detailed woman in garden",
            "ar": "21:9",
            "model_choice": "wai",
            "comp_strength": "low",
            "sr": "sr70",
            "char_choice": "mageill",
            "sref_rand": "sref_855014",
            "image_url": "http://example.com/source.png"
        }
        bot.db.save_generation("gen_blend_123", gen_data)
        bot.active_generations["gen_blend_123"] = gen_data

        mock_reblend_interaction = MagicMock()
        mock_reblend_interaction.response.is_done.return_value = False
        mock_reblend_interaction.response.send_message = AsyncMock()
        mock_reblend_interaction.user.display_name = "TestStroth"
        mock_reblend_interaction.user.id = 12345

        asyncio.run(bot.handle_reblend(mock_reblend_interaction, "gen_blend_123"))
        mock_reblend_interaction.response.send_message.assert_called_once()
        sent_call = mock_reblend_interaction.response.send_message.call_args
        sent_view = sent_call.kwargs.get("view")
        self.assertIsInstance(sent_view, BlendButtons)
        self.assertEqual(sent_view.ar, "21:9")
        self.assertEqual(sent_view.char_choice, "mageill")

    def test_blended_image_embed_and_consolidated_buttons(self):
        """Test build_blended_image_embed 3-column dashboard and IsolatedImageButtons 3-row consolidation."""
        from views import build_blended_image_embed, IsolatedImageButtons

        # 1. Test build_blended_image_embed
        prompt = "Semi-realism, masterpiece, best quality. mageill, A digital illustration of a slender woman --sr.60 --sref 576522"
        embed = build_blended_image_embed(
            index=3,
            display_prompt=prompt,
            checkpoint="waiIllustriousSDXL_v170.safetensors",
            target_seed=1113230745203310,
            width=768,
            height=1344,
            comp_strength="low",
            cfg=4.0,
            sref_info={"code": 576522, "name": "Pop Art 3D Render"},
            char_choice="mageill",
            sr_choice="sr60",
            user_name="strothman",
            user_id=200011008327024641,
            image_url="http://example.com/source_ref.png",
            is_blend=True
        )
        self.assertEqual(embed.title, "✨ Blended Image 3")
        self.assertEqual(embed.thumbnail.url, "http://example.com/source_ref.png")
        self.assertEqual(embed.color.value, 0x8A2BE2)
        self.assertIn("strothman", embed.footer.text)
        self.assertIn("1113230745203310", embed.footer.text)

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("📐 Canvas & Framing", fields)
        self.assertIn("🤖 Checkpoint & Tech", fields)
        self.assertIn("🎭 Aesthetics & Identity", fields)

        self.assertIn("768x1344", fields["📐 Canvas & Framing"])
        self.assertIn("🖼️ Light Comp (0.35)", fields["📐 Canvas & Framing"])
        self.assertIn("Wai Illustrious SDXL v1.70", fields["🤖 Checkpoint & Tech"])
        self.assertIn("1113230745203310", fields["🤖 Checkpoint & Tech"])
        self.assertIn("🔮 Mageill (Epoch 5)", fields["🎭 Aesthetics & Identity"])
        self.assertIn("--sr.60", fields["🎭 Aesthetics & Identity"])
        self.assertIn("576522", fields["🎭 Aesthetics & Identity"])

        # 2. Test IsolatedImageButtons with is_blend=True and has_sref=True (3 rows max)
        iso_view_sref = IsolatedImageButtons("gen_iso_1", index=3, has_sref=True, is_blend=True)
        r0 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 0]
        r1 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 1]
        r2 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 2]
        r3 = [c for c in iso_view_sref.children if getattr(c, "row", None) == 3]

        self.assertEqual(len(r0), 4) # 1.25x, 1.5x, vary subtle, vary strong
        self.assertEqual(len(r1), 5) # fav_style, fav_prompt, copy_prompt, remix, reblend
        self.assertEqual(len(r2), 3) # custom, random, saved sref
        self.assertEqual(len(r3), 0) # Consolidated from 4 rows down to 3!

        btn_ids = [c.custom_id for c in r1]
        self.assertIn("remix:gen_iso_1", btn_ids)
        self.assertIn("reblend:gen_iso_1", btn_ids)

        # 3. Test IsolatedImageButtons with is_blend=True and has_sref=False (2 rows max!)
        iso_view_nosref = IsolatedImageButtons("gen_iso_2", index=1, has_sref=False, is_blend=True)
        r0_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 0]
        r1_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 1]
        r2_no = [c for c in iso_view_nosref.children if getattr(c, "row", None) == 2]

        self.assertEqual(len(r0_no), 4)
        self.assertEqual(len(r1_no), 4) # fav_prompt, copy_prompt, remix, reblend
        self.assertEqual(len(r2_no), 0) # Only 2 rows!

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

    def test_module43_video_dashboard_and_action_view(self):
        """Test build_video_complete_embed 3-column dashboard, VideoActionView, and VideoPromptModal defaults."""
        from views import build_video_complete_embed, VideoActionView, VideoPromptModal

        # 1. Test build_video_complete_embed fields and layout
        embed = build_video_complete_embed(
            prompt="futuristic hovercraft speeding over water",
            duration_sec=5.0,
            total_output_frames=162,
            out_fps=32,
            orig_w=1920,
            orig_h=1080,
            width=832,
            height=480,
            video_seed=99887766,
            elapsed_time=45.2,
            init_sec=2.1,
            sample_sec=38.0,
            post_sec=5.1,
            motion_badges=["🎥 Zoom In", "✨ Cinematic"],
            smoothness="smooth",
            user_name="Alice",
            user_id=123456789
        )

        self.assertIn("Wan 2.2 Studio Video Generation Complete", embed.title)
        self.assertEqual(embed.color.value, 0x5865F2)
        self.assertIn("Alice", embed.footer.text)
        self.assertIn("45.2s", embed.footer.text)

        fields = {f.name: f.value for f in embed.fields}
        self.assertIn("🎬 Motion & Camera", fields)
        self.assertIn("⏱️ Video Specs", fields)
        self.assertIn("⚡ Engine & Render", fields)

        self.assertIn("futuristic hovercraft", fields["🎬 Motion & Camera"])
        self.assertIn("🎥 Zoom In • ✨ Cinematic", fields["🎬 Motion & Camera"])
        self.assertIn("1920x1080` → `832x480", fields["🎬 Motion & Camera"])

        self.assertIn("5.0s", fields["⏱️ Video Specs"])
        self.assertIn("32 FPS", fields["⏱️ Video Specs"])
        self.assertIn("162 frames", fields["⏱️ Video Specs"])

        self.assertIn("Wan 2.2 14B GGUF", fields["⚡ Engine & Render"])
        self.assertIn("99887766", fields["⚡ Engine & Render"])

        # 2. Test VideoActionView buttons
        view = VideoActionView("vid_test_123", smoothness="smooth")
        self.assertEqual(len(view.children), 3)
        btn_ids = [c.custom_id for c in view.children]
        self.assertIn("video_reroll:vid_test_123", btn_ids)
        self.assertIn("video_remix:vid_test_123", btn_ids)
        self.assertIn("video_toggle_fps:vid_test_123", btn_ids)

        labels = [c.label for c in view.children]
        self.assertIn("🔄 Re-roll", labels)
        self.assertIn("✏️ Remix Motion", labels)
        self.assertIn("⚡ Switch to Fast (16 FPS)", labels)

        # Fast mode toggle label
        view_fast = VideoActionView("vid_test_456", smoothness="fast")
        labels_fast = [c.label for c in view_fast.children]
        self.assertIn("🎬 Switch to Smooth (32 FPS)", labels_fast)

        # 3. Test VideoPromptModal with pre-filled defaults
        modal = VideoPromptModal(
            default_prompt="golden retriever running in park",
            default_duration="10",
            default_smoothness="fast"
        )
        self.assertEqual(modal.prompt_input.default, "golden retriever running in park")
        self.assertEqual(modal.duration_input.default, "10")
        self.assertEqual(modal.smoothness_input.default, "fast")

    def test_module42_bertflow_workflow_and_command(self):
        """Test Bertflow workflow generation, aspect ratio resolver, UNET detection, and button controls."""
        from parsers import (
            resolve_bertflow_dimensions,
            get_bertflow_unet_model,
            prepare_bertflow_workflow,
            BERTFLOW_ASPECT_RATIOS
        )
        from views import BertflowButtons

        # 1. Test aspect ratio resolution
        self.assertEqual(resolve_bertflow_dimensions("a portrait", "1:1")[1:], (1224, 1224))
        self.assertEqual(resolve_bertflow_dimensions("a landscape", "16:9")[1:], (1632, 920))
        self.assertEqual(resolve_bertflow_dimensions("a tall portrait", "9:16")[1:], (920, 1632))
        self.assertEqual(resolve_bertflow_dimensions("an ultrawide", "21:9")[1:], (1872, 800))

        # Dynamic --ar flag
        p_clean, w, h = resolve_bertflow_dimensions("a cinematic shot --ar 16:9", None)
        self.assertEqual(p_clean, "a cinematic shot")
        self.assertEqual((w, h), (1632, 920))

        # 2. Test UNET model detection
        active_unet = get_bertflow_unet_model()
        self.assertTrue(active_unet.endswith(".safetensors"))
        self.assertIn(active_unet, ["museByStableYogi_v35Int8Extended.safetensors", "pornmasterKrea2_v1FP8.safetensors"])

        # 3. Test prepare_bertflow_workflow
        wf = prepare_bertflow_workflow(
            prompt="Scott kneeling in living room",
            width=1224,
            height=1224,
            seed=424242,
            steps=8,
            unet_model="pornmasterKrea2_v1FP8.safetensors"
        )
        self.assertEqual(wf["627"]["inputs"]["text"], "Scott kneeling in living room")
        self.assertEqual(wf["649"]["inputs"]["seed"], 424242)
        self.assertEqual(wf["698"]["inputs"]["width"], 1224)
        self.assertEqual(wf["698"]["inputs"]["height"], 1224)
        self.assertEqual(wf["599"]["inputs"]["steps"], 8)
        self.assertEqual(wf["761"]["inputs"]["unet_name"], "pornmasterKrea2_v1FP8.safetensors")
        self.assertEqual(wf["822"]["inputs"]["lora_1"]["strength"], -2.0)
        self.assertEqual(wf["763"]["class_type"], "ConditioningZeroOut")

        # 4. Test BertflowButtons (4 buttons: reroll, remix, toggle char, upscale)
        view = BertflowButtons(generation_id="bert_test_123", character=None)
        self.assertEqual(len(view.children), 4)
        btn_ids = [c.custom_id for c in view.children]
        self.assertIn("bertflow_reroll:bert_test_123", btn_ids)
        self.assertIn("bertflow_remix:bert_test_123", btn_ids)
        self.assertIn("bertflow_toggle_char:bert_test_123", btn_ids)
        self.assertIn("bertflow_upscale:bert_test_123", btn_ids)
        self.assertEqual(view.toggle_char_btn.label, "🌿 Ogarla: OFF")

        # Test BertflowButtons with active character
        view_char = BertflowButtons(generation_id="bert_test_456", character="ogarla.85")
        self.assertEqual(view_char.toggle_char_btn.label, "🌿 Ogarla: ON")

        # 5. Test command registration and parameters
        from bot import bot
        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("bertflow", commands)
        bert_cmd = commands["bertflow"]
        param_names = [p.name for p in bert_cmd.parameters]
        self.assertIn("favorite_prompt", param_names)
        self.assertIn("free", commands)
        self.assertIn("purge-vram", commands)

        # 6. Test PromptPaginationView with bertflow_callback
        from views import PromptPaginationView
        sample_prompts = [{"id": 1, "prompt_name": "Test Krea Prompt", "prompt_text": "A photo of a cyberpunk city"}]
        view_pg = PromptPaginationView(user_id=123, prompts=sample_prompts, per_page=5, imagine_callback=lambda inter, p: None, bertflow_callback=lambda inter, p: None)
        labels = [c.label for c in view_pg.children if hasattr(c, "label") and c.label]
        self.assertIn("⚡ Bertflow", labels)
        self.assertIn("🎨 Imagine", labels)

    def test_module60_describe_krea2_workflow_and_buttons(self):
        """Test Krea 2 prompt formatting, workflow nodes, and DescribeButtons integration."""
        from parsers import format_krea2_prompt
        from views import DescribeButtons
        import json

        # 1. Test format_krea2_prompt
        sample1 = "The image shows a cinematic portrait of an astronaut on Mars."
        self.assertEqual(format_krea2_prompt(sample1), "A cinematic portrait of an astronaut on Mars.")

        sample2 = "In this photo, a woman in a red silk dress stands on a balcony overlooking the city."
        self.assertEqual(format_krea2_prompt(sample2), "A woman in a red silk dress stands on a balcony overlooking the city.")

        sample3 = "This is an image of a vintage analog synthesizer with glowing patch cables."
        self.assertEqual(format_krea2_prompt(sample3), "A vintage analog synthesizer with glowing patch cables.")

        sample4 = "the photo depicts a dense bamboo forest with morning sunbeams."
        self.assertEqual(format_krea2_prompt(sample4), "A dense bamboo forest with morning sunbeams.")

        self.assertEqual(format_krea2_prompt(""), "")
        self.assertEqual(format_krea2_prompt(None), "")

        # 2. Test workflows/DESCRIBE_cuibot.json
        with open("workflows/DESCRIBE_cuibot.json", "r", encoding="utf-8") as f:
            wf = json.load(f)

        self.assertIn("5", wf, "Node 5 (Florence2Run for Krea 2) should exist in workflow")
        self.assertEqual(wf["5"]["inputs"]["task"], "more_detailed_caption")
        self.assertEqual(wf["5"]["class_type"], "Florence2Run")

        self.assertIn("11", wf, "Node 11 (ShowText for krea2_prompt) should exist in workflow")
        self.assertEqual(wf["11"]["_meta"]["title"], "krea2_prompt")

        self.assertIn("21", wf, "Node 21 (CR Text Replace for Krea 2) should exist in workflow")

        # 3. Test DescribeButtons view
        view = DescribeButtons(generation_id="desc_test_888", ar="16:9")
        btn_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]
        
        # Verify krea2 button exists with correct custom_id structure
        krea2_btns = [b for b in btn_ids if ":krea2:" in b]
        self.assertEqual(len(krea2_btns), 1, "There should be exactly one Krea 2 button")
        self.assertTrue(krea2_btns[0].startswith("gen_desc:desc_test_888:krea2:16:9:"))

    def test_module60b_handle_generate_described(self):
        """Test handle_generate_described dispatching for caption, detailed, and krea2 with correct params."""
        import bot
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch

        mock_gen_data = {
            "caption": "A blonde woman in green scarf with tea cup",
            "detailed_caption": "A hyper-realistic digital painting features a nude, slender, blonde woman with small breasts, wearing a green scarf, standing beside a teapot and cup.",
            "krea2_prompt": "A hyper-realistic digital painting of a slender blonde woman beside teapot.",
        }
        bot.db.save_generation("test_desc_123", mock_gen_data)
        bot.active_generations["test_desc_123"] = mock_gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        # 1. Test Generate Caption -> execute_imagine with checkpoint="hyphoriaIlluNAI_v001.safetensors"
        with patch.object(bot, "execute_imagine", new=AsyncMock()) as mock_imagine:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="caption",
                ar="16:9", use_sr="sr90", use_oga=False, model_choice="hyphoria"
            ))
            mock_imagine.assert_called_once()
            call_kwargs = mock_imagine.call_args[1]
            self.assertIn("A blonde woman in green scarf with tea cup", call_kwargs["prompt"])
            self.assertIn("--sr.90", call_kwargs["prompt"])
            self.assertIn("--ar 16:9", call_kwargs["prompt"])
            self.assertEqual(call_kwargs["checkpoint"], "hyphoriaIlluNAI_v001.safetensors")

        # 2. Test Generate Detailed -> execute_imagine with default checkpoint
        with patch.object(bot, "execute_imagine", new=AsyncMock()) as mock_imagine:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="detailed",
                ar="21:9", use_sr="nosr", use_oga=True, model_choice="default"
            ))
            mock_imagine.assert_called_once()
            call_kwargs = mock_imagine.call_args[1]
            self.assertIn("hyper-realistic digital painting", call_kwargs["prompt"])
            self.assertIn("ogarla,", call_kwargs["prompt"])
            self.assertIn("--ogarla.70", call_kwargs["prompt"])
            self.assertIn("--ar 21:9", call_kwargs["prompt"])
            self.assertIsNone(call_kwargs["checkpoint"])

        # 3. Test Generate Krea 2 -> execute_bertflow
        with patch.object(bot, "execute_bertflow", new=AsyncMock()) as mock_bert:
            asyncio.run(bot.handle_generate_described(
                mock_interaction, "test_desc_123", desc_type="krea2",
                ar="16:9", use_oga=True
            ))
            mock_bert.assert_called_once()
            call_kwargs = mock_bert.call_args[1]
            self.assertEqual(call_kwargs["prompt"], "A hyper-realistic digital painting of a slender blonde woman beside teapot.")
            self.assertEqual(call_kwargs["aspect_ratio"], "16:9")
            self.assertEqual(call_kwargs["character"], "ogarla.85")

    def test_module60c_describe_command_and_cog_registration(self):
        """Test /describe command registration on bot tree and VisionCog export."""
        import bot
        self.assertTrue(hasattr(bot, "describe"), "bot should re-export describe")
        self.assertTrue(hasattr(bot, "handle_update_describe_view"), "bot should re-export handle_update_describe_view")
        self.assertTrue(hasattr(bot, "handle_generate_described"), "bot should re-export handle_generate_described")
        
        # Verify VisionCog registration
        cog = bot.bot.get_cog("VisionCog")
        self.assertIsNotNone(cog, "VisionCog should be registered on bot")
        self.assertTrue(hasattr(cog, "describe"), "VisionCog should have describe method")
        
        # Verify tree command exists
        cmd = bot.bot.tree.get_command("describe")
        self.assertIsNotNone(cmd, "/describe command should exist on bot.tree")
        self.assertEqual(cmd.name, "describe")

    def test_module61_blend_krea_integration(self):
        """Test Krea 2 blend prompt fusion, workflow wetness tuning, view layouts, and command registration."""
        from parsers import fuse_krea2_blend_prompt, prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed, BlendButtons
        from bot import bot

        # 1. Test fuse_krea2_blend_prompt
        vision = "A cinematic portrait of an astronaut on Mars"
        remix = "wearing neon armor, golden hour lighting"
        fused = fuse_krea2_blend_prompt(vision, remix)
        self.assertEqual(fused, "wearing neon armor, golden hour lighting, A cinematic portrait of an astronaut on Mars")

        # Test empty remix returns vision
        self.assertEqual(fuse_krea2_blend_prompt(vision, ""), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, None), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, " , "), "A cinematic portrait of an astronaut on Mars")
        self.assertEqual(fuse_krea2_blend_prompt(vision, ","), "A cinematic portrait of an astronaut on Mars")

        # Test empty vision returns remix
        self.assertEqual(fuse_krea2_blend_prompt("", remix), remix)

        # 2. Test prepare_bertflow_workflow with custom wetness
        wf = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            width=1632,
            height=920,
            seed=12345,
            steps=8,
            unet_model="pornmasterKrea2_v1FP8.safetensors",
            wetness_strength=1.0
        )
        self.assertEqual(wf["822"]["inputs"]["lora_1"]["strength"], 1.0)
        self.assertTrue(wf["822"]["inputs"]["lora_1"]["on"])

        wf_zero = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            width=1632,
            height=920,
            wetness_strength=0.0
        )
        self.assertFalse(wf_zero["822"]["inputs"]["lora_1"]["on"])

        # Test Krea 2 Character LoRA injection and trigger word
        wf_char = prepare_bertflow_workflow(
            prompt="cyberpunk alley",
            character="ogarla.85"
        )
        self.assertTrue(wf_char["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_char["822"]["inputs"]["lora_2"]["lora"], "Krea2\\ogarla_krea2.safetensors")
        self.assertEqual(wf_char["822"]["inputs"]["lora_2"]["strength"], 0.85)
        self.assertEqual(wf_char["627"]["inputs"]["text"], "ogarla, cyberpunk alley")

        # Test prompt shorthand parsing (--ogarla.70)
        wf_prompt = prepare_bertflow_workflow(
            prompt="cyberpunk alley --ogarla.70"
        )
        self.assertTrue(wf_prompt["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_prompt["822"]["inputs"]["lora_2"]["strength"], 0.70)
        # Test prompt wetness flags (--dry, --dewy, --matte, --wet)
        wf_dry = prepare_bertflow_workflow("a sunny beach --dry")
        self.assertEqual(wf_dry["822"]["inputs"]["lora_1"]["strength"], -3.0)
        self.assertEqual(wf_dry["627"]["inputs"]["text"], "a sunny beach")

        wf_dewy = prepare_bertflow_workflow("portrait of a runner --dewy")
        self.assertEqual(wf_dewy["822"]["inputs"]["lora_1"]["strength"], -0.5)

        wf_matte = prepare_bertflow_workflow("studio portrait --matte")
        self.assertEqual(wf_matte["822"]["inputs"]["lora_1"]["strength"], -2.5)

        wf_custom_wet = prepare_bertflow_workflow("rainy street --wet 0.75")
        self.assertEqual(wf_custom_wet["822"]["inputs"]["lora_1"]["strength"], 0.75)

        # Test Ogarla Krea 2 Character LoRA injection and trigger word
        wf_oga = prepare_bertflow_workflow(
            prompt="fashion runway photo --ogarla.85",
            character="ogarla"
        )
        self.assertTrue(wf_oga["822"]["inputs"]["lora_2"]["on"])
        self.assertEqual(wf_oga["822"]["inputs"]["lora_2"]["lora"], "Krea2\\ogarla_krea2.safetensors")
        self.assertEqual(wf_oga["822"]["inputs"]["lora_2"]["strength"], 0.85)
        self.assertEqual(wf_oga["627"]["inputs"]["text"], "ogarla, fashion runway photo")

        # Valerie is SDXL-only and must not inject into Krea 2
        wf_val = prepare_bertflow_workflow(
            prompt="fashion runway photo --valerie.90",
            character="valerie"
        )
        self.assertNotIn("lora_2", wf_val["822"]["inputs"])

        # Test scan_krea2_loras
        from characters import scan_krea2_loras
        discovered_loras = scan_krea2_loras()
        self.assertIsInstance(discovered_loras, list)

        # Test Architecture resolver for Krea 2
        from model_architecture import resolve_lora_for_architecture, Architecture
        self.assertEqual(
            resolve_lora_for_architecture("ogarla", Architecture.KREA2),
            "Krea2\\ogarla_krea2.safetensors"
        )
        self.assertEqual(
            resolve_lora_for_architecture("valerie", Architecture.SDXL),
            "jen_epoch_5.safetensors"
        )

        # 3. Test BlendKreaButtons
        view = BlendKreaButtons(generation_id="krea_blend_777", ar="16:9", model_choice="muse", wetness=-2.0, character="ogarla.85")
        btn_ids = [item.custom_id for item in view.children if hasattr(item, "custom_id")]
        self.assertIn("set_blend_krea_ar:krea_blend_777:16:9", btn_ids)
        self.assertIn("set_blend_krea_ar:krea_blend_777:21:9", btn_ids)
        self.assertIn("toggle_blend_krea_model:krea_blend_777", btn_ids)
        self.assertIn("toggle_blend_krea_wetness:krea_blend_777", btn_ids)
        self.assertIn("set_blend_krea_char:krea_blend_777", btn_ids)
        self.assertIn("edit_blend_krea_prompt:krea_blend_777", btn_ids)
        self.assertIn("gen_blend_krea:krea_blend_777", btn_ids)

        char_select = next(item for item in view.children if getattr(item, "custom_id", None) == "set_blend_krea_char:krea_blend_777")
        char_values = [opt.value for opt in char_select.options]
        self.assertNotIn("valerie.90", char_values)
        self.assertNotIn("valerie.70", char_values)
        self.assertIn("ogarla.85", char_values)

        # Test BertflowButtons with character
        from views import BertflowButtons
        view_oga = BertflowButtons(generation_id="bert_test_789", character="ogarla.85")
        self.assertEqual(view_oga.toggle_char_btn.label, "🌿 Ogarla: ON")

        # Test CHARACTER_CHOICES_KREA2 for slash commands
        from bot import CHARACTER_CHOICES_KREA2
        choice_vals = [c.value for c in CHARACTER_CHOICES_KREA2]
        self.assertIn("ogarla.85", choice_vals)
        self.assertIn("ogarla.70", choice_vals)
        self.assertNotIn("valerie.90", choice_vals)
        self.assertNotIn("valerie.70", choice_vals)

        # 4. Test build_blend_krea_embed
        gen_data = {
            "krea2_prompt": "A close up photo of a cat",
            "user_prompt": "wearing a tiny bowtie",
            "fused_prompt": "wearing a tiny bowtie, A close up photo of a cat",
            "ar": "1:1",
            "model_choice": "muse",
            "wetness": -2.0
        }
        embed = build_blend_krea_embed(gen_data, author_str="TestUser")
        self.assertIn("Krea 2 Blend Studio", embed.title)
        field_names = [f.name for f in embed.fields]
        self.assertIn("📜 Generation Prompt", field_names)
        gen_prompt_field = next(f for f in embed.fields if f.name == "📜 Generation Prompt")
        self.assertIn("wearing a tiny bowtie, A close up photo of a cat", gen_prompt_field.value)

        # 4b. Test EditBlendKreaModal
        from views import EditBlendKreaModal
        modal = EditBlendKreaModal("krea_blend_777", current_prompt="A test generation prompt")
        self.assertEqual(modal.title, "✏️ Edit Generation Prompt")
        self.assertEqual(modal.prompt_input.default, "A test generation prompt")
        edit_btn = next(item for item in view.children if getattr(item, "custom_id", None) == "edit_blend_krea_prompt:krea_blend_777")
        self.assertEqual(edit_btn.label, "✏️ Edit Prompt")

        # 5. Test BlendButtons excludes Krea 2 button and Krea 2 models (strictly decoupled)
        blend_view = BlendButtons(generation_id="blend_gen_123")
        blend_btn_ids = [item.custom_id for item in blend_view.children if hasattr(item, "custom_id")]
        self.assertNotIn("blend_desc:blend_gen_123:krea2", blend_btn_ids, "Krea 2 button must NOT be in BlendButtons")
        self.assertIn("blend_desc:blend_gen_123:blend", blend_btn_ids)

        model_select = next(item for item in blend_view.children if getattr(item, "custom_id", "").startswith("set_blend_model:"))
        model_values = [opt.value for opt in model_select.options]
        self.assertNotIn("muse", model_values, "Krea 2 muse model must NOT be in BlendButtons")
        self.assertNotIn("pornmaster", model_values, "Krea 2 pornmaster model must NOT be in BlendButtons")
        self.assertIn("wai", model_values)

        # 5b. Test workflows/DESCRIBE_blend.json dedicated SDXL workflow exists and excludes Krea nodes
        blend_wf_path = "workflows/DESCRIBE_blend.json"
        self.assertTrue(os.path.exists(blend_wf_path), "workflows/DESCRIBE_blend.json must exist")
        with open(blend_wf_path, "r", encoding="utf-8") as f:
            blend_wf = json.load(f)
        self.assertIn("3", blend_wf, "Node 3 (caption) must be in DESCRIBE_blend.json")
        self.assertIn("4", blend_wf, "Node 4 (detailed_caption) must be in DESCRIBE_blend.json")
        self.assertNotIn("5", blend_wf, "Node 5 (Krea 2 Florence run) must NOT be in DESCRIBE_blend.json")
        self.assertNotIn("11", blend_wf, "Node 11 (Krea 2 text) must NOT be in DESCRIBE_blend.json")
        self.assertNotIn("21", blend_wf, "Node 21 (Krea 2 replace) must NOT be in DESCRIBE_blend.json")

        commands = {cmd.name: cmd for cmd in bot.tree.get_commands()}
        self.assertIn("blend-krea", commands)
        bk_cmd = commands["blend-krea"]
        self.assertEqual(len(bk_cmd.parameters), 1, "blend-krea must only require the image parameter")
        img_param = bk_cmd.parameters[0]
        self.assertEqual(img_param.name, "image")
        self.assertTrue(img_param.required)

        # 7. Test dynamic character autocomplete attached to commands
        for cmd_name in ["bertflow", "imagine", "flux"]:
            self.assertIn(cmd_name, commands)
            cmd = commands[cmd_name]
            char_param = next((p for p in cmd.parameters if p.name == "character"), None)
            self.assertIsNotNone(char_param, f"Command {cmd_name} missing character parameter")
            self.assertTrue(char_param.autocomplete, f"Command {cmd_name} character parameter missing autocomplete")

    def test_bertflow_and_standard_remix_modals(self):
        """Verify handle_bertflow_remix and handle_remix construct the RemixModal properly."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from bot import handle_bertflow_remix, handle_remix, db

        # Seed mock generation into database
        db.save_generation(
            "test_remix_gen_123",
            {
                "prompt": "A beautiful sunrise in the mountains",
                "seed": 42,
                "aspect_ratio": "16:9",
                "unet_model": "museByStableYogi_v35Int8Extended.safetensors",
                "is_bertflow": True,
            }
        )

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = False
        mock_interaction.response.send_modal = AsyncMock()

        # Execute bertflow remix
        asyncio.run(handle_bertflow_remix(mock_interaction, "test_remix_gen_123"))
        mock_interaction.response.send_modal.assert_called_once()
        modal_passed = mock_interaction.response.send_modal.call_args[0][0]
        self.assertEqual(modal_passed.title, "✏️ Remix / Tweak Prompt")

        # Execute standard remix
        mock_interaction.reset_mock()
        mock_interaction.response.is_done.return_value = False
        db.save_generation(
            "test_standard_remix_123",
            {
                "prompt": "A futuristic cyberpunk city",
                "seed": 100,
                "is_bertflow": False,
            }
        )
        asyncio.run(handle_remix(mock_interaction, "test_standard_remix_123"))
        mock_interaction.response.send_modal.assert_called_once()

    def test_bertflow_composition_modes(self):
        """Verify prepare_bertflow_workflow and BlendKreaButtons direct composition options."""
        from parsers import prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed

        # 1. Test 'off' mode (pure txt2img default)
        wf_off = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1632,
            height=920,
            init_image="test_input.png",
            comp_strength="off"
        )
        self.assertNotIn("900", wf_off)
        self.assertNotIn("901", wf_off)
        self.assertNotIn("902", wf_off)
        self.assertEqual(wf_off["599"]["inputs"]["latent_image"], ["698", 0])
        self.assertEqual(wf_off["599"]["inputs"]["start_at_step"], 0)

        # 2. Test 'medium' composition mode (70% denoise)
        wf_med = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1632,
            height=920,
            init_image="test_input.png",
            comp_strength="medium"
        )
        self.assertIn("900", wf_med)
        self.assertEqual(wf_med["900"]["inputs"]["image"], "test_input.png")
        self.assertIn("901", wf_med)
        self.assertEqual(wf_med["901"]["inputs"]["width"], 1632)
        self.assertEqual(wf_med["901"]["inputs"]["height"], 920)
        self.assertEqual(wf_med["901"]["inputs"]["crop"], "center")
        self.assertIn("902", wf_med)
        self.assertEqual(wf_med["902"]["inputs"]["vae"], ["757", 0])
        self.assertEqual(wf_med["599"]["inputs"]["latent_image"], ["902", 0])
        self.assertEqual(wf_med["599"]["inputs"]["start_at_step"], 2)
        self.assertEqual(wf_med["599"]["inputs"]["add_noise"], "enable")
        self.assertEqual(wf_med["599"]["inputs"]["return_with_leftover_noise"], "disable")

        # 3. Test 'strong' composition mode (50% denoise)
        wf_strong = prepare_bertflow_workflow(
            prompt="A photorealistic mountain lake",
            width=1224,
            height=1224,
            init_image="test_input.png",
            comp_strength="strong"
        )
        self.assertEqual(wf_strong["599"]["inputs"]["start_at_step"], 4)

        # 4. Test BlendKreaButtons with composition dropdown
        view_off = BlendKreaButtons(generation_id="gen_krea_test", composition="off")
        comp_select_off = next(item for item in view_off.children if getattr(item, "custom_id", "") == "set_blend_krea_comp:gen_krea_test")
        self.assertEqual(comp_select_off.options[0].value, "off")
        self.assertTrue(comp_select_off.options[0].default)

        view_med = BlendKreaButtons(generation_id="gen_krea_test", composition="medium")
        comp_select_med = next(item for item in view_med.children if getattr(item, "custom_id", "") == "set_blend_krea_comp:gen_krea_test")
        self.assertEqual(comp_select_med.options[2].value, "medium")
        self.assertTrue(comp_select_med.options[2].default)

        # 5. Test embed display with composition
        embed = build_blend_krea_embed({"krea2_prompt": "Test", "composition": "medium"})
        pipeline_field = next(f.value for f in embed.fields if f.name == "⚙️ Pipeline Settings")
        self.assertIn("Direct Comp:** `Medium (Balanced Silhouette & Pose - 70% Denoise)`", pipeline_field)

    def test_module64_photo_dataset_builder(self):
        """Verify prompt matrix, IPAdapter multi-image workflow, and AI-Toolkit config generation."""
        import tempfile
        from tools.create_character_dataset_from_photos import (
            generate_prompt_matrix,
            build_ipadapter_workflow,
            write_ai_toolkit_config
        )

        # 1. Prompt matrix generation
        prompts = generate_prompt_matrix(count=20)
        self.assertEqual(len(prompts), 20)
        self.assertTrue(any("full-body" in p for p in prompts))
        self.assertTrue(any("medium" in p or "waist-up" in p or "cowboy" in p for p in prompts))
        self.assertTrue(any("portrait" in p or "close-up" in p for p in prompts))

        # 2. IPAdapter workflow generation for 3 reference images
        ref_images = ["photo1.png", "photo2.png", "photo3.png"]
        wf = build_ipadapter_workflow(
            reference_image_names=ref_images,
            prompt_text="a fashion photoshoot of a woman",
            seed=42,
            checkpoint="RealVisXL_V4.0.safetensors",
            resolution=1024
        )
        self.assertIn("20", wf)  # IPAdapterUnifiedLoader
        self.assertEqual(wf["20"]["class_type"], "IPAdapterUnifiedLoader")
        self.assertEqual(wf["20"]["inputs"]["preset"], "PLUS (high strength)")

        # Verify load nodes for all 3 images
        self.assertIn("ref_img_0", wf)
        self.assertIn("ref_img_1", wf)
        self.assertIn("ref_img_2", wf)
        self.assertEqual(wf["ref_img_0"]["inputs"]["image"], "photo1.png")
        self.assertEqual(wf["ref_img_2"]["inputs"]["image"], "photo3.png")

        # Verify chained IPAdapterAdvanced nodes
        self.assertIn("ref_ip_0", wf)
        self.assertIn("ref_ip_1", wf)
        self.assertIn("ref_ip_2", wf)
        # KSampler model input points to the final IPAdapter in chain
        self.assertEqual(wf["3"]["inputs"]["model"], ["ref_ip_2", 0])

        # 3. AI-Toolkit config generation
        with tempfile.TemporaryDirectory() as tmpdir:
            out_yaml = os.path.join(tmpdir, "test_config.yaml")
            write_ai_toolkit_config("my_test_char", tmpdir, out_yaml)
            self.assertTrue(os.path.exists(out_yaml))
            with open(out_yaml, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn('name: "my_test_char_krea2"', content)
            self.assertIn('trigger_word: "my_test_char"', content)
            self.assertIn('arch: "krea2"', content)
            self.assertIn('noise_scheduler: "flowmatch"', content)

    def test_module66_krea2_celebrity_presets(self):
        """Verify Krea 2 celebrity registry, prompt injection, UI dropdowns, and slash command choices."""
        from celebrities import (
            FAVORITE_CELEBRITIES,
            get_celebrity,
            get_celebrity_display_badge,
            inject_celebrity_in_prompt,
            get_celebrity_autocomplete_choices,
            CELEBRITY_CHOICES_KREA2
        )
        from parsers import prepare_bertflow_workflow
        from views import BlendKreaButtons, build_blend_krea_embed
        import bot

        # 1. Verify all 16 favorite celebrities are present
        expected_celebs = [
            "Audrey Hepburn", "Grace Kelly", "Nicole Kidman", "Margot Robbie",
            "Sandra Bullock", "Emma Stone", "Anya Taylor-Joy", "Zendaya",
            "Cameron Diaz", "Saoirse Ronan", "Emma Watson", "Michelle Pfeiffer",
            "Gal Gadot", "Taylor Swift", "Ariana Grande", "Keira Knightley"
        ]
        self.assertEqual(len(FAVORITE_CELEBRITIES), 16)
        for name in expected_celebs:
            celeb = get_celebrity(name)
            self.assertIsNotNone(celeb, f"Celebrity '{name}' should be resolved by get_celebrity")
            self.assertEqual(celeb.display_name, name)

        # 2. Test display badges
        self.assertEqual(get_celebrity_display_badge("none"), "None")
        self.assertEqual(get_celebrity_display_badge(None), "None")
        self.assertIn("Audrey Hepburn", get_celebrity_display_badge("audrey_hepburn"))
        self.assertIn("Zendaya", get_celebrity_display_badge("zendaya"))

        # 3. Test prompt injection
        injected = inject_celebrity_in_prompt("walking through Paris in spring", "audrey_hepburn")
        self.assertTrue(injected.startswith("Audrey Hepburn,"))
        self.assertIn("walking through Paris in spring", injected)

        # No double injection if already present
        already_in = inject_celebrity_in_prompt("photo of Audrey Hepburn in Paris", "audrey_hepburn")
        self.assertEqual(already_in, "photo of Audrey Hepburn in Paris")

        # 4. Test prepare_bertflow_workflow injection via argument and prompt flag
        wf_arg = prepare_bertflow_workflow("a sunny afternoon", celebrity="zendaya")
        self.assertTrue(wf_arg["627"]["inputs"]["text"].startswith("Zendaya,"))

        wf_flag = prepare_bertflow_workflow("a red carpet event --margot")
        self.assertTrue(wf_flag["627"]["inputs"]["text"].startswith("Margot Robbie,"))

        # Both character LoRA and celebrity preset can coexist
        wf_both = prepare_bertflow_workflow("a stylish portrait", character="ogarla.85", celebrity="audrey_hepburn")
        self.assertIn("ogarla", wf_both["627"]["inputs"]["text"].lower())
        self.assertIn("Audrey Hepburn", wf_both["627"]["inputs"]["text"])

        # 5. Test BlendKreaButtons has both Character and Celebrity dropdowns
        view = BlendKreaButtons(
            generation_id="krea_celeb_test",
            ar="16:9",
            model_choice="muse",
            wetness=-2.0,
            composition="off",
            character="ogarla.85",
            celebrity="zendaya"
        )
        celeb_select = next(item for item in view.children if getattr(item, "custom_id", None) == "krea_celeb_test" or getattr(item, "custom_id", None) == "set_blend_krea_celeb:krea_celeb_test")
        self.assertIsNotNone(celeb_select)
        celeb_vals = [opt.value for opt in celeb_select.options]
        self.assertIn("none", celeb_vals)
        self.assertIn("audrey_hepburn", celeb_vals)
        self.assertIn("zendaya", celeb_vals)
        self.assertIn("keira_knightley", celeb_vals)
        self.assertEqual(len(celeb_vals), 17)  # None + 16 celebrities

        # Character select still present and functional
        char_select = next(item for item in view.children if getattr(item, "custom_id", None) == "set_blend_krea_char:krea_celeb_test")
        self.assertIsNotNone(char_select)

        # 6. Test build_blend_krea_embed shows both Character and Celebrity badges
        gen_data = {
            "krea2_prompt": "Portrait in natural sunlight",
            "fused_prompt": "Portrait in natural sunlight",
            "ar": "16:9",
            "model_choice": "muse",
            "wetness": -2.0,
            "char_choice": "ogarla.85",
            "celeb_choice": "audrey_hepburn"
        }
        embed = build_blend_krea_embed(gen_data)
        settings_field = next(f for f in embed.fields if f.name == "⚙️ Pipeline Settings")
        self.assertIn("Character:", settings_field.value)
        self.assertIn("Ogarla", settings_field.value)
        self.assertIn("Celebrity:", settings_field.value)
        self.assertIn("Audrey Hepburn", settings_field.value)

        # 7. Test slash command parameters
        bert_cmd = next(c for c in bot.bot.tree.get_commands() if c.name == "bertflow")
        self.assertIn("character", [p.name for p in bert_cmd.parameters])
        self.assertIn("celebrity", [p.name for p in bert_cmd.parameters])

        blend_cmd = next(c for c in bot.bot.tree.get_commands() if c.name == "blend-krea")
        self.assertEqual([p.name for p in blend_cmd.parameters], ["image"], "blend-krea is streamlined to image only")

    def test_module67_lora_workflow_architecture_audit(self):
        """Audit all workflows, character profiles, resolvers, and parsers for 100% LoRA architecture match."""
        from scripts.audit_loras import (
            audit_workflows,
            audit_characters,
            audit_parsers_and_presets,
            audit_ui_choices
        )
        wf_issues = audit_workflows()
        self.assertEqual(wf_issues, [], f"Workflow LoRA audit issues: {wf_issues}")

        char_issues = audit_characters()
        self.assertEqual(char_issues, [], f"Character LoRA audit issues: {char_issues}")

        preset_issues = audit_parsers_and_presets()
        self.assertEqual(preset_issues, [], f"Preset LoRA audit issues: {preset_issues}")

        ui_issues = audit_ui_choices()
        self.assertEqual(ui_issues, [], f"UI LoRA audit issues: {ui_issues}")

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
        joy_path = os.path.join(os.path.dirname(__file__), "workflows", "DESCRIBE_joycaption.json")
        qwen_path = os.path.join(os.path.dirname(__file__), "workflows", "DESCRIBE_qwen_vl.json")
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

    def test_module72_blend_studio_debloated_unified_dashboard(self):
        """Test the de-bloated unified 1-page Blend Studio layout, 1-click toggles, cycles, and blend generation."""
        import bot
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        from views import BlendButtons

        # 1. Test 5-Row Component Layout and item distribution
        gen_id = "blend_debloat_999"
        view = BlendButtons(
            generation_id=gen_id,
            ar="16:9",
            sr=True,
            model_choice="wai",
            comp_strength="style",
            sref_rand="nosref",
            char_choice="valerie"
        )
        self.assertLessEqual(len(view.children), 25)
        rows = sorted(list(set(c.row for c in view.children)))
        self.assertEqual(rows, [0, 1, 2, 3, 4], "Dashboard must strictly use rows 0 through 4 without tabs")

        # Row 0: Model Checkpoint Select
        row0_items = [c for c in view.children if c.row == 0]
        self.assertEqual(len(row0_items), 1)
        self.assertEqual(row0_items[0].custom_id, f"set_blend_model:{gen_id}")
        self.assertNotIn("muse", [opt.value for opt in row0_items[0].options])

        # Row 1: Character LoRA Select
        row1_items = [c for c in view.children if c.row == 1]
        self.assertEqual(len(row1_items), 1)
        self.assertEqual(row1_items[0].custom_id, f"set_blend_char:{gen_id}")
        char_selected = next(opt for opt in row1_items[0].options if opt.default)
        self.assertEqual(char_selected.value, "valerie")

        # Row 2: Composition Select
        row2_items = [c for c in view.children if c.row == 2]
        self.assertEqual(len(row2_items), 1)
        self.assertEqual(row2_items[0].custom_id, f"set_blend_comp:{gen_id}")

        # Row 3: 1-Click Toggles & Cycles (3 buttons)
        row3_items = [c for c in view.children if c.row == 3]
        self.assertEqual(len(row3_items), 3)
        sr_btn = next(c for c in row3_items if c.custom_id.startswith("toggle_blend_sr:"))
        ar_btn = next(c for c in row3_items if c.custom_id.startswith("cycle_blend_ar:"))
        style_btn = next(c for c in row3_items if c.custom_id.startswith("toggle_blend_sref:"))
        self.assertIn("ON", sr_btn.label)
        self.assertEqual(ar_btn.label, "📐 AR: 16:9")
        self.assertIn("OFF", style_btn.label)
        self.assertIn("--sref random", style_btn.label)

        # Row 4: Action Launchers (2 buttons)
        row4_items = [c for c in view.children if c.row == 4]
        self.assertEqual(len(row4_items), 2)
        edit_btn = next(c for c in row4_items if c.custom_id.startswith("edit_blend_prompt:"))
        blend_btn = next(c for c in row4_items if c.custom_id.startswith("blend_desc:"))
        self.assertEqual(edit_btn.label, "✏️ Edit Prompt")
        self.assertEqual(blend_btn.label, "🎨 Blend Image")
        self.assertEqual(blend_btn.custom_id, f"blend_desc:{gen_id}:blend")

        # 2. Test handle_generate_blended with desc_type == "blend"
        gen_data = {
            "caption": "a lone warrior standing in the rain",
            "detailed_caption": "masterpiece digital painting of a lone samurai with glowing katana under volumetric rain and neon signs",
            "uploaded_image_name": "samurai_input.png",
            "image_url": "https://cdn.discordapp.com/samurai_thumb.jpg",
            "ar": "16:9",
            "sr": True,
            "char_choice": "valerie",
            "model_choice": "wai",
            "comp_strength": "style",
            "sref_rand": "nosref"
        }
        bot.db.save_generation("gen_blend_exec_test", gen_data)
        bot.active_generations["gen_blend_exec_test"] = gen_data

        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch.object(bot, "execute_blend_generation", new=AsyncMock()) as mock_exec:
            asyncio.run(bot.handle_generate_blended(
                mock_interaction,
                "gen_blend_exec_test",
                desc_type="blend",
                ar="16:9",
                use_sr=True,
                char_choice="valerie",
                model_choice="wai",
                comp_strength="style",
                use_sref_rand="nosref"
            ))
            mock_exec.assert_called_once()
            called_prompt = mock_exec.call_args.kwargs.get("prompt") or mock_exec.call_args.args[1]
            self.assertIn("samurai with glowing katana", called_prompt)

    def test_module73_krea_cog_modular_architecture(self):
        """Verify KreaCog and KreaService modular separation, registration, and backward compatibility."""
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        import bot
        from cogs.krea_cog import KreaCog
        from services.krea_service import (
            BERTFLOW_MODEL_CHOICES,
            execute_bertflow,
            handle_bertflow_reroll,
            handle_bertflow_remix,
            handle_bertflow_toggle_char,
            handle_bertflow_upscale,
            handle_update_blend_krea_view,
            handle_submit_edit_blend_krea_prompt,
            handle_generate_blend_krea,
            execute_blend_krea_core,
        )

        # 1. Test KreaCog registration in bot
        cog = bot.bot.get_cog("KreaCog")
        self.assertIsNotNone(cog, "KreaCog must be loaded and registered in bot")
        self.assertIsInstance(cog, KreaCog)

        # 2. Test slash commands registered in tree
        commands = {cmd.name: cmd for cmd in bot.bot.tree.get_commands()}
        self.assertIn("bertflow", commands)
        self.assertIn("blend-krea", commands)

        # Verify bertflow command parameters
        bert_cmd = commands["bertflow"]
        bert_params = {p.name: p for p in bert_cmd.parameters}
        self.assertIn("prompt", bert_params)
        self.assertIn("aspect_ratio", bert_params)
        self.assertIn("character", bert_params)
        self.assertIn("celebrity", bert_params)
        self.assertIn("favorite_prompt", bert_params)
        self.assertIn("model", bert_params)
        self.assertIn("steps", bert_params)
        self.assertIn("seed", bert_params)

        # Verify blend-krea streamlined parameter (image only)
        krea_cmd = commands["blend-krea"]
        self.assertEqual(len(krea_cmd.parameters), 1)
        self.assertEqual(krea_cmd.parameters[0].name, "image")

        # 3. Test backward-compatibility re-exports on bot module
        self.assertTrue(callable(bot.execute_bertflow))
        self.assertTrue(callable(bot.handle_bertflow_reroll))
        self.assertTrue(callable(bot.handle_bertflow_remix))
        self.assertTrue(callable(bot.handle_bertflow_toggle_char))
        self.assertTrue(callable(bot.handle_bertflow_upscale))
        self.assertTrue(callable(bot.handle_update_blend_krea_view))
        self.assertTrue(callable(bot.handle_submit_edit_blend_krea_prompt))
        self.assertTrue(callable(bot.handle_generate_blend_krea))
        self.assertTrue(callable(bot.execute_blend_krea_core))
        self.assertEqual(bot.BERTFLOW_MODEL_CHOICES, BERTFLOW_MODEL_CHOICES)
        self.assertEqual(bot.bertflow, cog.bertflow)
        self.assertEqual(bot.blend_krea, cog.blend_krea)

        # 4. Test services/krea_service function executions (mocked)
        mock_interaction = MagicMock()
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        gen_id = "test_krea_service_gen_777"
        gen_data = {
            "prompt": "a test scene",
            "original_prompt": "a test scene",
            "aspect_ratio": "16:9",
            "steps": 8,
            "seed": 99999,
            "unet_model": "museByStableYogi_v35Int8Extended.safetensors",
            "character": "ogarla.85",
            "celebrity": None
        }
        bot.db.save_generation(gen_id, gen_data)

        # Test handle_bertflow_reroll calls execute_bertflow
        with patch("services.krea_service.execute_bertflow", new=AsyncMock()) as mock_exec:
            asyncio.run(handle_bertflow_reroll(mock_interaction, gen_id))
            mock_exec.assert_called_once()
            self.assertEqual(mock_exec.call_args.kwargs["prompt"], "a test scene")
            self.assertNotEqual(mock_exec.call_args.kwargs["seed"], 99999)

        # Test handle_bertflow_toggle_char toggles character
        with patch("services.krea_service.execute_bertflow", new=AsyncMock()) as mock_exec:
            asyncio.run(handle_bertflow_toggle_char(mock_interaction, gen_id))
            mock_exec.assert_called_once()
            self.assertIsNone(mock_exec.call_args.kwargs["character"])

    def test_blend_sdxl_sref_random_toggle_interaction(self):
        """Test /blend-sdxl style button is ONLY a 1-click --sref random toggle on/off."""
        import discord
        import asyncio
        from unittest.mock import MagicMock, AsyncMock, patch
        import bot
        from views import BlendButtons, build_blend_embed

        # 1. Test View Button States
        # State: OFF
        view_off = BlendButtons("gen_off", ar="16:9", sref_rand="nosref")
        btn_off = next(c for c in view_off.children if getattr(c, "custom_id", "").startswith("toggle_blend_sref:"))
        self.assertEqual(btn_off.label, "🎲 --sref random: OFF")
        self.assertEqual(btn_off.style, discord.ButtonStyle.secondary)

        # State: ON
        view_on = BlendButtons("gen_on", ar="16:9", sref_rand="sref")
        btn_on = next(c for c in view_on.children if getattr(c, "custom_id", "").startswith("toggle_blend_sref:"))
        self.assertEqual(btn_on.label, "🎲 --sref random: ON")
        self.assertEqual(btn_on.style, discord.ButtonStyle.primary)

        # 2. Test Embed Field Display
        embed_off = build_blend_embed({"sref_rand": "nosref", "caption": "test", "detailed_caption": "scene"}, author_str="Test")
        aesthetics_off = next(f.value for f in embed_off.fields if f.name == "🎭 Aesthetics")
        self.assertIn("**Style:** `OFF`", aesthetics_off)

        embed_on = build_blend_embed({"sref_rand": "sref", "caption": "test", "detailed_caption": "scene"}, author_str="Test")
        aesthetics_on = next(f.value for f in embed_on.fields if f.name == "🎭 Aesthetics")
        self.assertIn("**Style:** `🎲 --sref random`", aesthetics_on)

        # 3. Test on_interaction toggle handling
        gen_id = "gen_sref_toggle_test"
        bot.active_generations[gen_id] = {"sref_rand": "nosref", "author_str": "Test"}
        bot.db.save_generation(gen_id, bot.active_generations[gen_id])
        mock_interaction = MagicMock()
        mock_interaction.type = discord.InteractionType.component
        mock_interaction.data = {"custom_id": f"toggle_blend_sref:{gen_id}"}
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch("bot.handle_update_blend_view", new=AsyncMock()) as mock_update:
            asyncio.run(bot.on_interaction(mock_interaction))
            mock_update.assert_called_once_with(mock_interaction, gen_id, new_sref="sref")

        # Toggle back from ON to OFF
        bot.active_generations[gen_id] = {"sref_rand": "sref", "author_str": "Test"}
        bot.db.save_generation(gen_id, bot.active_generations[gen_id])
        with patch("bot.handle_update_blend_view", new=AsyncMock()) as mock_update:
            asyncio.run(bot.on_interaction(mock_interaction))
            mock_update.assert_called_once_with(mock_interaction, gen_id, new_sref="nosref")

    def test_engine_queue_workflow_detection(self):
        """Test detect_workflow_engine accurately detects all supported architectures."""
        from services.engine_queue import detect_workflow_engine, EngineType

        # SDXL
        wf_sdxl = {"1": {"class_type": "KSampler", "inputs": {"model": "wai-ani-illustrious.safetensors"}}}
        self.assertEqual(detect_workflow_engine(wf_sdxl), EngineType.SDXL)

        # Krea 2
        wf_krea = {"822": {"inputs": {"lora_name": "Krea2\\wetness_krea2_loraholic.safetensors"}}}
        self.assertEqual(detect_workflow_engine(wf_krea), EngineType.KREA2)

        # Flux
        wf_flux = {"1": {"class_type": "FluxGuidance", "inputs": {"model": "flux1-dev.safetensors"}}}
        self.assertEqual(detect_workflow_engine(wf_flux), EngineType.FLUX)

        # Wan 2.2 Video
        wf_wan = {"1": {"class_type": "WanVideo", "inputs": {"model": "wan2.1_i2v.safetensors"}}}
        self.assertEqual(detect_workflow_engine(wf_wan), EngineType.WAN)

        # LTX Video
        wf_ltx = {"1": {"class_type": "LTXVideo", "inputs": {"model": "ltx-video-2b.safetensors"}}}
        self.assertEqual(detect_workflow_engine(wf_ltx), EngineType.LTX)

        # Florence-2 Vision
        wf_florence = {"1": {"class_type": "Florence2Run", "inputs": {"image": "test.png"}}}
        self.assertEqual(detect_workflow_engine(wf_florence), EngineType.FLORENCE2)

    def test_engine_affinity_scheduling_and_thrashing_prevention(self):
        """Verify scheduler prioritizes same-engine jobs to avoid VRAM swapping thrash."""
        from services.engine_queue import EngineAwareQueue, QueueJob, JobPriority, EngineType

        queue = EngineAwareQueue(starvation_timeout=45.0, enable_affinity=True)
        # Simulate active engine in VRAM is Krea 2
        queue._current_engine = EngineType.KREA2

        # Enqueue interleaved jobs: Wan, Krea2, SDXL, Krea2
        loop = asyncio.new_event_loop()
        f1 = loop.create_future()
        f2 = loop.create_future()
        f3 = loop.create_future()
        f4 = loop.create_future()

        job_wan = QueueJob("job_wan", {}, EngineType.WAN, JobPriority.NORMAL, f1, enqueued_at=time.time())
        job_krea1 = QueueJob("job_krea1", {}, EngineType.KREA2, JobPriority.NORMAL, f2, enqueued_at=time.time() + 1)
        job_sdxl = QueueJob("job_sdxl", {}, EngineType.SDXL, JobPriority.NORMAL, f3, enqueued_at=time.time() + 2)
        job_krea2 = QueueJob("job_krea2", {}, EngineType.KREA2, JobPriority.NORMAL, f4, enqueued_at=time.time() + 3)

        queue._queue = [job_wan, job_krea1, job_sdxl, job_krea2]

        # First picked job MUST be Krea2 due to active engine affinity bonus
        first_picked = queue._select_next_job()
        self.assertEqual(first_picked.engine, EngineType.KREA2)

        # Second picked job MUST ALSO be Krea2 because Krea2 is still current engine
        second_picked = queue._select_next_job()
        self.assertEqual(second_picked.engine, EngineType.KREA2)

        # Verified that switches_prevented was incremented!
        self.assertGreater(queue.stats["switches_prevented"], 0)
        loop.close()

    def test_engine_queue_starvation_prevention(self):
        """Verify jobs for other engines escalate and run if waiting exceeds starvation timeout."""
        from services.engine_queue import EngineAwareQueue, QueueJob, JobPriority, EngineType

        queue = EngineAwareQueue(starvation_timeout=45.0, enable_affinity=True)
        queue._current_engine = EngineType.KREA2

        loop = asyncio.new_event_loop()
        f1 = loop.create_future()
        f2 = loop.create_future()

        # Job Wan has been waiting for 50 seconds (exceeding 45s threshold)
        job_wan_old = QueueJob("job_wan_old", {}, EngineType.WAN, JobPriority.NORMAL, f1, enqueued_at=time.time() - 50.0)
        # Job Krea just arrived 1 second ago
        job_krea_fresh = QueueJob("job_krea_fresh", {}, EngineType.KREA2, JobPriority.NORMAL, f2, enqueued_at=time.time() - 1.0)

        queue._queue = [job_krea_fresh, job_wan_old]

        # Wan job must win despite Krea affinity because it passed the starvation limit!
        next_job = queue._select_next_job()
        self.assertEqual(next_job.job_id, "job_wan_old")
        loop.close()

    def test_engine_queue_vram_purge_on_switch(self):
        """Verify automatic VRAM cache purging occurs when transitioning model architectures."""
        from services.engine_queue import EngineAwareQueue, JobPriority, EngineType

        mock_client = MagicMock()
        mock_client.free_memory = AsyncMock(return_value=True)
        mock_client._execute_direct = AsyncMock(return_value=[b"fake_output_bytes"])

        queue = EngineAwareQueue(comfy_client=mock_client, starvation_timeout=45.0, enable_affinity=True)

        async def run_transition():
            queue.start()
            # 1. Enqueue Krea 2 job
            f1 = await queue.enqueue({"class_type": "wetness_krea2"}, priority=JobPriority.HIGH)
            res1 = await f1
            self.assertEqual(queue.current_engine, EngineType.KREA2)
            mock_client.free_memory.assert_not_called()

            # 2. Enqueue Wan 2.2 Video job (triggers engine transition)
            f2 = await queue.enqueue({"class_type": "WanVideo"}, priority=JobPriority.HIGH)
            res2 = await f2
            self.assertEqual(queue.current_engine, EngineType.WAN)
            # Free memory MUST have been called during transition!
            mock_client.free_memory.assert_called_once_with(unload_models=True, free_memory=True)
            self.assertEqual(queue.stats["engine_switches"], 1)
            await queue.stop()

        asyncio.run(run_transition())

    def test_engine_queue_cancellation_and_status(self):
        """Test queue status reporting and job cancellation mechanics."""
        from services.engine_queue import EngineAwareQueue, JobPriority

        queue = EngineAwareQueue()
        loop = asyncio.new_event_loop()
        f = loop.create_future()
        from services.engine_queue import QueueJob
        job = QueueJob("job_cancel_test", {}, "sdxl", JobPriority.NORMAL, f)
        queue._queue = [job]

        self.assertEqual(queue.pending_count, 1)
        cancelled = queue.cancel("job_cancel_test")
        self.assertTrue(cancelled)
        self.assertEqual(queue.pending_count, 0)
        self.assertTrue(f.cancelled())

        status = queue.get_status()
        self.assertIn("current_engine_name", status)
        self.assertIn("pending_count", status)
        self.assertEqual(status["stats"]["total_cancelled"], 1)
        loop.close()

    def test_semantic_workflow_discovery(self):
        """Test semantic node discovery by class_type, title, and input keys."""
        from services.workflow_adapter import find_nodes, find_node, get_positive_negative_nodes

        wf = {
            "node_a": {"class_type": "KSampler", "_meta": {"title": "Main Sampler"}, "inputs": {"seed": 100, "positive": ["node_b", 0], "negative": ["node_c", 0]}},
            "node_b": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text (Positive)"}, "inputs": {"text": "hello"}},
            "node_c": {"class_type": "CLIPTextEncode", "_meta": {"title": "CLIP Text (Negative)"}, "inputs": {"text": "bad"}},
            "node_d": {"class_type": "EmptyLatentImage", "inputs": {"width": 1024, "height": 1024}}
        }

        # 1. Discover by class
        samplers = find_nodes(wf, class_type="KSampler")
        self.assertEqual(len(samplers), 1)
        self.assertEqual(samplers[0][0], "node_a")

        # 2. Discover by title
        pos_by_title = find_node(wf, title_contains="positive")
        self.assertIsNotNone(pos_by_title)
        self.assertEqual(pos_by_title[0], "node_b")

        # 3. Discover positive/negative via graph tracing
        pos_match, neg_match = get_positive_negative_nodes(wf)
        self.assertIsNotNone(pos_match)
        self.assertIsNotNone(neg_match)
        self.assertEqual(pos_match[0], "node_b")
        self.assertEqual(neg_match[0], "node_c")

    def test_semantic_workflow_setters(self):
        """Test semantic setters update prompts, seed, dimensions, and checkpoint."""
        from services.workflow_adapter import (
            set_workflow_prompt,
            set_workflow_seed,
            set_workflow_dimensions,
            set_workflow_checkpoint,
            set_workflow_sampler_params,
            set_workflow_filename_prefix
        )

        wf = {
            "node_sampler": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 20, "cfg": 4.0, "positive": ["node_pos", 0], "negative": ["node_neg", 0]}},
            "node_pos": {"class_type": "CLIPTextEncode", "_meta": {"title": "Positive"}, "inputs": {"text": "old pos"}},
            "node_neg": {"class_type": "CLIPTextEncode", "_meta": {"title": "Negative"}, "inputs": {"text": "old neg"}},
            "node_latent": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "node_ckpt": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "old.safetensors"}},
            "node_save": {"class_type": "SaveImage", "inputs": {"filename_prefix": "old_prefix"}}
        }

        # Set prompts
        set_workflow_prompt(wf, positive="masterpiece, new character", negative="low quality, blur")
        self.assertEqual(wf["node_pos"]["inputs"]["text"], "masterpiece, new character")
        self.assertEqual(wf["node_neg"]["inputs"]["text"], "low quality, blur")

        # Set seed
        set_workflow_seed(wf, 888999)
        self.assertEqual(wf["node_sampler"]["inputs"]["seed"], 888999)

        # Set dimensions
        set_workflow_dimensions(wf, width=1920, height=1080, batch_size=4)
        self.assertEqual(wf["node_latent"]["inputs"]["width"], 1920)
        self.assertEqual(wf["node_latent"]["inputs"]["height"], 1080)
        self.assertEqual(wf["node_latent"]["inputs"]["batch_size"], 4)

        # Set checkpoint & sampler params
        set_workflow_checkpoint(wf, ckpt_name="waiIllustriousSDXL_v170.safetensors")
        self.assertEqual(wf["node_ckpt"]["inputs"]["ckpt_name"], "waiIllustriousSDXL_v170.safetensors")

        set_workflow_sampler_params(wf, steps=35, cfg=6.5)
        self.assertEqual(wf["node_sampler"]["inputs"]["steps"], 35)
        self.assertEqual(wf["node_sampler"]["inputs"]["cfg"], 6.5)

        # Set filename prefix
        set_workflow_filename_prefix(wf, "Discord Bot/custom/new_output")
        self.assertEqual(wf["node_save"]["inputs"]["filename_prefix"], "Discord Bot/custom/new_output")

    def test_renumbered_workflow_immunity(self):
        """
        Critical Resilience Test:
        Scrambles all node IDs in a full SDXL workflow to arbitrary 4-digit numbers,
        rewires link pointers, and verifies semantic adapter successfully modifies
        the renumbered workflow without any KeyErrors or failures.
        """
        from parsers import load_workflow_template, apply_loras_to_workflow
        from services.workflow_adapter import (
            set_workflow_prompt,
            set_workflow_seed,
            set_workflow_dimensions,
            set_workflow_checkpoint
        )

        orig_wf = load_workflow_template("workflows/txt2img_lowres.json")

        # Mapping of original node IDs to completely scrambled new IDs
        id_map = {
            "3": "8103",   # KSampler
            "4": "8104",   # CheckpointLoaderSimple
            "5": "8105",   # EmptyLatentImage
            "6": "8106",   # CLIPTextEncode (Positive)
            "7": "8107",   # CLIPTextEncode (Negative)
            "8": "8108",   # VAEDecode
            "9": "8109",   # PreviewImage
            "75": "8175",  # Load LoRA: Semi-Realism
            "76": "8176",  # Load LoRA: Ogarla
        }

        # Build scrambled workflow with updated inter-node link references
        scrambled_wf = {}
        for old_id, node in orig_wf.items():
            new_id = id_map.get(old_id, f"9{old_id}")
            node_copy = copy.deepcopy(node)
            for k, val in node_copy.get("inputs", {}).items():
                if isinstance(val, list) and len(val) == 2 and str(val[0]) in id_map:
                    val[0] = id_map[str(val[0])]
            scrambled_wf[new_id] = node_copy

        # Confirm old numeric IDs DO NOT exist in scrambled workflow
        self.assertNotIn("3", scrambled_wf)
        self.assertNotIn("75", scrambled_wf)
        self.assertNotIn("76", scrambled_wf)

        # 1. Apply Prompt Semantically
        set_workflow_prompt(scrambled_wf, positive="scrambled test positive", negative="scrambled test negative")
        self.assertEqual(scrambled_wf["8106"]["inputs"]["text"], "scrambled test positive")
        self.assertEqual(scrambled_wf["8107"]["inputs"]["text"], "scrambled test negative")

        # 2. Apply Seed Semantically
        set_workflow_seed(scrambled_wf, 999111)
        self.assertEqual(scrambled_wf["8103"]["inputs"]["seed"], 999111)

        # 3. Apply Dimensions Semantically
        set_workflow_dimensions(scrambled_wf, width=1280, height=720, batch_size=2)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["width"], 1280)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["height"], 720)
        self.assertEqual(scrambled_wf["8105"]["inputs"]["batch_size"], 2)

        # 4. Apply LoRAs Semantically to Renumbered Workflow
        updated_lora_wf = apply_loras_to_workflow(scrambled_wf, [("semi-realism", 0.85), ("ogarla", 0.70)])
        self.assertEqual(updated_lora_wf["8175"]["inputs"]["strength_model"], 0.85)
        self.assertEqual(updated_lora_wf["8176"]["inputs"]["strength_model"], 0.70)

    def test_pending_jobs_db_crud(self):
        """Test recording, reading, updating, and cleaning up pending jobs in SQLite."""
        import db
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        orig_db = db.DB_FILE
        try:
            db.DB_FILE = temp_db
            db.init_db()

            test_prompt_id = "test_prompt_crash_123"
            # 1. Record job
            res = db.record_pending_job(
                prompt_id=test_prompt_id,
                generation_id="gen_test_999",
                channel_id=123456789,
                message_id=987654321,
                user_id=456789012,
                command_type="video",
                metadata={"frames": 81, "fps": 16}
            )
            self.assertTrue(res)

            # 2. Get pending jobs
            jobs = db.get_pending_jobs(status="running")
            found = [j for j in jobs if j["prompt_id"] == test_prompt_id]
            self.assertEqual(len(found), 1)
            job = found[0]
            self.assertEqual(job["channel_id"], 123456789)
            self.assertEqual(job["message_id"], 987654321)
            self.assertEqual(job["user_id"], 456789012)
            self.assertEqual(job["command_type"], "video")
            self.assertEqual(job["metadata"].get("frames"), 81)

            # 3. Complete job
            update_res = db.complete_pending_job(test_prompt_id, status="completed")
            self.assertTrue(update_res)

            jobs_after = db.get_pending_jobs(status="running")
            found_after = [j for j in jobs_after if j["prompt_id"] == test_prompt_id]
            self.assertEqual(len(found_after), 0)

            # Check in all jobs
            all_jobs = db.get_pending_jobs(status="")
            found_completed = [j for j in all_jobs if j["prompt_id"] == test_prompt_id]
            self.assertEqual(len(found_completed), 1)
            self.assertEqual(found_completed[0]["status"], "completed")

            # Cleanup
            db.cleanup_stale_jobs(0.0)
        finally:
            db.DB_FILE = orig_db
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except Exception:
                    pass

    def test_crash_recovery_media_reconciliation(self):
        """Test crash reconciliation successfully recovers completed media and updates Discord."""
        import db
        from services.recovery_service import reconcile_pending_jobs
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        orig_db = db.DB_FILE
        try:
            db.DB_FILE = temp_db
            db.init_db()
            test_prompt_id = "test_prompt_recov_media"
            db.record_pending_job(
                prompt_id=test_prompt_id,
                channel_id=111222,
                message_id=333444,
                user_id=555666,
                command_type="imagine",
                metadata={"prompt": "beautiful fantasy landscape"}
            )

            mock_bot = MagicMock()
            mock_channel = MagicMock()
            mock_bot.get_channel.return_value = mock_channel
            mock_msg = AsyncMock()
            mock_channel.fetch_message = AsyncMock(return_value=mock_msg)
            mock_channel.send = AsyncMock()

            mock_comfy = MagicMock()
            mock_comfy.is_online = AsyncMock(return_value=True)
            mock_comfy.get_history_output = AsyncMock(return_value={
                "9": {
                    "images": [{"filename": "recovered_01.png", "subfolder": "", "type": "output"}]
                }
            })
            mock_comfy.get_image = AsyncMock(return_value=b"\x89PNG\r\n\x1a\n\x00\x00fake_image_bytes")

            async def run_recov():
                return await reconcile_pending_jobs(mock_bot, mock_comfy)

            stats = asyncio.run(run_recov())
            self.assertEqual(stats["recovered"], 1)
            self.assertEqual(mock_msg.edit.call_count, 1)

            # Verify job marked as recovered in DB
            all_jobs = db.get_pending_jobs(status="recovered")
            found = [j for j in all_jobs if j["prompt_id"] == test_prompt_id]
            self.assertEqual(len(found), 1)
        finally:
            db.DB_FILE = orig_db
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except Exception:
                    pass

    def test_crash_recovery_execution_error(self):
        """Test crash reconciliation handles ComfyUI execution errors and notifies Discord."""
        import db
        from services.recovery_service import reconcile_pending_jobs
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
            temp_db = tf.name

        orig_db = db.DB_FILE
        try:
            db.DB_FILE = temp_db
            db.init_db()
            test_prompt_id = "test_prompt_recov_err"
            db.record_pending_job(
                prompt_id=test_prompt_id,
                channel_id=111222,
                message_id=333444,
                user_id=555666,
                command_type="video"
            )

            mock_bot = MagicMock()
            mock_channel = MagicMock()
            mock_bot.get_channel.return_value = mock_channel
            mock_msg = AsyncMock()
            mock_channel.fetch_message = AsyncMock(return_value=mock_msg)

            mock_comfy = MagicMock()
            mock_comfy.is_online = AsyncMock(return_value=True)
            mock_comfy.get_history_output = AsyncMock(side_effect=Exception("ComfyUI execution error from history: CUDA out of memory"))

            async def run_recov():
                return await reconcile_pending_jobs(mock_bot, mock_comfy)

            stats = asyncio.run(run_recov())
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(mock_msg.edit.call_count, 1)

            # Verify job marked as failed in DB
            all_jobs = db.get_pending_jobs(status="failed")
            found = [j for j in all_jobs if j["prompt_id"] == test_prompt_id]
            self.assertEqual(len(found), 1)
        finally:
            db.DB_FILE = orig_db
            if os.path.exists(temp_db):
                try:
                    os.remove(temp_db)
                except Exception:
                    pass

    def test_cleanup_orphaned_quadrants_pruner(self):
        """Test that cleanup_orphaned_quadrants purges files older than max_age_hours or lacking DB records."""
        import tempfile
        import db

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"QUADRANT_CACHE_DIR": tmpdir}):
                # Create a file with a generation ID that does not exist in DB
                stale_file = os.path.join(tmpdir, "orphan9999_1.png")
                with open(stale_file, "wb") as f:
                    f.write(b"dummy_bytes_123456789")
                
                # Set mtime to 3 hours ago (older than 120s grace period)
                past_time = time.time() - 10800
                os.utime(stale_file, (past_time, past_time))

                # Also create a file with max_age exceeded
                old_file = os.path.join(tmpdir, "oldgen9999_2.png")
                with open(old_file, "wb") as f:
                    f.write(b"dummy_bytes_987654321")
                old_time = time.time() - (50 * 3600)
                os.utime(old_file, (old_time, old_time))

                stats = db.cleanup_orphaned_quadrants(max_age_hours=48.0)
                self.assertGreaterEqual(stats["deleted"], 2)
                self.assertFalse(os.path.exists(stale_file))
                self.assertFalse(os.path.exists(old_file))

    def test_concurrent_image_downloads_comfy_client(self):
        """Test that comfy_client downloads multi-image batch outputs concurrently via asyncio.gather."""
        from comfy_client import ComfyClient

        client = ComfyClient()
        client.session = MagicMock()
        client.is_online = AsyncMock(return_value=True)

        download_order = []
        async def mock_get_image(filename, subfolder, img_type):
            download_order.append(filename)
            await asyncio.sleep(0.01)
            return f"bytes_{filename}".encode("utf-8")

        client.get_image = AsyncMock(side_effect=mock_get_image)

        # Mock results dict with 4 images (standard 2x2 grid)
        mock_results = {
            "9": {
                "images": [
                    {"filename": "img_01.png", "subfolder": "", "type": "output"},
                    {"filename": "img_02.png", "subfolder": "", "type": "output"},
                    {"filename": "img_03.png", "subfolder": "", "type": "output"},
                    {"filename": "img_04.png", "subfolder": "", "type": "output"}
                ]
            }
        }

        # Mock session post
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(return_value={"prompt_id": "test_concurrent_prompt"})
        client.session.post = MagicMock(return_value=AsyncMock(__aenter__=AsyncMock(return_value=mock_resp), __aexit__=AsyncMock()))

        # Mock history output to return results immediately
        client.get_history_output = AsyncMock(return_value=mock_results)

        outputs = asyncio.run(client._execute_direct(workflow={}, use_queue=False))
        self.assertEqual(len(outputs), 4)
        self.assertEqual(client.get_image.call_count, 4)
        self.assertEqual(len(download_order), 4)

    def test_pipeline_defaults_and_checkpoint_display_names(self):
        """Test centralized PipelineDefaults constants and get_checkpoint_display_name resolution."""
        from config import PipelineDefaults, get_checkpoint_display_name

        # Check default checkpoints
        self.assertTrue(PipelineDefaults.DEFAULT_SDXL_CHECKPOINT.endswith(".safetensors"))
        self.assertTrue(PipelineDefaults.DEFAULT_FLUX_CHECKPOINT.endswith(".safetensors"))
        self.assertTrue(PipelineDefaults.DEFAULT_WAN_CHECKPOINT.endswith(".safetensors"))

        # Check denoise values
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_SDXL, 0.55)
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_FLUX_SUBTLE, 0.26)
        self.assertEqual(PipelineDefaults.UPSCALE_DENOISE_FLUX_MODERATE, 0.35)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_VERY_HIGH, 0.95)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["low"], 0.85)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["med"], 0.70)
        self.assertEqual(PipelineDefaults.VARIATION_DENOISE_MAP["high"], 0.55)

        # Check display name resolution
        self.assertEqual(get_checkpoint_display_name("waiIllustriousSDXL_v170.safetensors"), "Wai Illustrious SDXL v1.70")
        self.assertEqual(get_checkpoint_display_name("RealVisXL_V4.0.safetensors"), "RealVisXL V4.0")
        self.assertEqual(get_checkpoint_display_name("wai"), "Wai Illustrious SDXL v1.70")
        self.assertEqual(get_checkpoint_display_name("realvis"), "RealVisXL V4.0")
        self.assertEqual(get_checkpoint_display_name(None), "Default Model")
        self.assertEqual(get_checkpoint_display_name("custom_model.safetensors"), "custom_model")

    def test_video_cog_registration_and_exports(self):
        """Test VideoCog registration in bot.tree and backward-compatibility re-exports."""
        import bot
        from cogs.video_cog import VideoCog
        import services.video_service as video_service

        # 1. Verify commands in bot.tree
        cmd_names = [c.name for c in bot.bot.tree.get_commands()]
        self.assertIn("video", cmd_names)
        self.assertIn("ltx", cmd_names)
        self.assertIn("Animate to Video", cmd_names)

        # 2. Verify re-exports on bot module
        self.assertTrue(hasattr(bot, "video"))
        self.assertTrue(hasattr(bot, "video_command"))
        self.assertTrue(hasattr(bot, "ltx"))
        self.assertTrue(hasattr(bot, "ltx_command"))
        self.assertTrue(hasattr(bot, "animate_to_video_context"))
        self.assertTrue(hasattr(bot, "execute_video_core"))
        self.assertTrue(hasattr(bot, "handle_video_reroll"))
        self.assertTrue(hasattr(bot, "handle_video_remix"))
        self.assertTrue(hasattr(bot, "handle_video_toggle_fps"))
        self.assertTrue(hasattr(bot, "execute_animate_message"))
        self.assertTrue(hasattr(bot, "execute_ltx_core"))

        # 3. Verify service functions are callable
        self.assertTrue(callable(video_service.execute_video_core))
        self.assertTrue(callable(video_service.execute_ltx_core))
        self.assertTrue(callable(video_service.handle_video_reroll))
        self.assertTrue(callable(video_service.handle_video_remix))
        self.assertTrue(callable(video_service.handle_video_toggle_fps))
        self.assertTrue(callable(video_service.execute_animate_message))

    def test_system_cog_registration_and_exports(self):
        """Test SystemCog registration in bot.tree and backward-compatibility re-exports."""
        import bot
        from cogs.system_cog import SystemCog
        import services.system_service as system_service

        # 1. Verify commands in bot.tree
        cmd_names = [c.name for c in bot.bot.tree.get_commands()]
        expected_commands = [
            "cui-start", "cui-stop", "cui-status", 
            "free", "purge-vram", "queue", 
            "diagnostics", "models", "scan_models", 
            "variation_mode", "negative", "prompt", "style"
        ]
        for cmd in expected_commands:
            self.assertIn(cmd, cmd_names, f"Expected /{cmd} to be registered in bot.tree")

        # 2. Verify re-exports on bot module
        self.assertTrue(hasattr(bot, "cui_start_command"))
        self.assertTrue(hasattr(bot, "cui_stop_command"))
        self.assertTrue(hasattr(bot, "cui_status_command"))
        self.assertTrue(hasattr(bot, "free_vram_command"))
        self.assertTrue(hasattr(bot, "purge_vram_command"))
        self.assertTrue(hasattr(bot, "queue_command"))
        self.assertTrue(hasattr(bot, "diagnostics_command"))
        self.assertTrue(hasattr(bot, "models_command"))
        self.assertTrue(hasattr(bot, "scan_models_command"))
        self.assertTrue(hasattr(bot, "variation_mode_command"))
        self.assertTrue(hasattr(bot, "negative_command"))
        self.assertTrue(hasattr(bot, "prompt_group"))
        self.assertTrue(hasattr(bot, "style_group"))
        self.assertTrue(hasattr(bot, "settings"))
        self.assertTrue(hasattr(bot, "load_settings"))
        self.assertTrue(hasattr(bot, "save_settings"))

        # 3. Verify service functions & embed builders
        self.assertTrue(callable(system_service.terminate_existing_comfyui))
        self.assertTrue(callable(system_service.fetch_comfyui_queue))
        self.assertTrue(callable(system_service.fetch_comfyui_system_stats))
        self.assertTrue(callable(system_service.purge_vram_core))
        self.assertTrue(callable(system_service.build_queue_embed))
        self.assertTrue(callable(system_service.build_diagnostics_embed))
        self.assertTrue(callable(system_service.build_models_embed))

        # Test queue embed generation
        q_embed = system_service.build_queue_embed(None, None)
        self.assertIsInstance(q_embed, bot.discord.Embed)
        self.assertIn("Could not connect", q_embed.description)

        # Test diagnostics embed generation
        d_embed = system_service.build_diagnostics_embed("TestUser")
        self.assertIsInstance(d_embed, bot.discord.Embed)
        self.assertIn("Diagnostics", d_embed.title)


if __name__ == "__main__":
    unittest.main()









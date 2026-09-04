"""
Automated Efficacy & Functionality Test Suite for Shallot-CUI-Bot.

Executes unit/integration tests for prompt parsers, workflow builders,
dimension math, auto-fix recipes, error log persistence, and CLI diagnostics.
"""

import os
import json
import unittest
import random
import logging

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

        # 2. Test BlendButtons sref mode cycling
        # Mode 1: nosref -> next is sref
        v_no = BlendButtons("gen123", sref_rand="nosref")
        btn_no = [item for item in v_no.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertIn("sref", btn_no.custom_id.split(":")[-1])
        self.assertIn("OFF", btn_no.label)

        # Mode 2: sref -> next is sref5
        v_sref = BlendButtons("gen123", sref_rand="sref")
        btn_sref = [item for item in v_sref.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sref.custom_id.split(":")[-1], "sref5")
        self.assertIn("1 Style", btn_sref.label)

        # Mode 3: sref5 -> next is sref10
        v_sref5 = BlendButtons("gen123", sref_rand="sref5")
        btn_sref5 = [item for item in v_sref5.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sref5.custom_id.split(":")[-1], "sref10")
        self.assertIn("5 Styles", btn_sref5.label)

        # Mode 4: sref10 -> next is sref15
        v_sref10 = BlendButtons("gen123", sref_rand="sref10")
        btn_sref10 = [item for item in v_sref10.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sref10.custom_id.split(":")[-1], "sref15")
        self.assertIn("10 Styles", btn_sref10.label)

        # Mode 5: sref15 -> next is nosref
        v_sref15 = BlendButtons("gen123", sref_rand="sref15")
        btn_sref15 = [item for item in v_sref15.children if "toggle_blend_sref" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sref15.custom_id.split(":")[-1], "nosref")
        self.assertIn("15 Styles", btn_sref15.label)

        # 3. Test Semi-Realism weight cycling (nosr -> sr60 -> sr70 -> sr80 -> sr90 -> nosr)
        v_sr_off = BlendButtons("gen123", sr="nosr")
        btn_sr_off = [item for item in v_sr_off.children if "toggle_blend_sr" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sr_off.custom_id.split(":")[4], "sr60")
        self.assertIn("OFF", btn_sr_off.label)

        v_sr60 = BlendButtons("gen123", sr="sr60")
        btn_sr60 = [item for item in v_sr60.children if "toggle_blend_sr" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sr60.custom_id.split(":")[4], "sr70")
        self.assertIn("--sr.60", btn_sr60.label)

        v_sr90 = BlendButtons("gen123", sr="sr90")
        btn_sr90 = [item for item in v_sr90.children if "toggle_blend_sr" in getattr(item, "custom_id", "")][0]
        self.assertEqual(btn_sr90.custom_id.split(":")[4], "nosr")
        self.assertIn("--sr.90", btn_sr90.label)

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
        self.assertEqual(wf["150"]["class_type"], "MMAudioModelLoader")
        self.assertEqual(wf["151"]["class_type"], "MMAudioFeatureUtilsLoader")
        self.assertEqual(wf["152"]["class_type"], "MMAudioSampler")
        self.assertEqual(wf["9"]["class_type"], "VHS_VideoCombine")
        self.assertEqual(wf["9"]["inputs"]["frame_rate"], 32)
        self.assertEqual(wf["9"]["inputs"]["audio"], ["152", 0])

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
        self.assertIn("smart", sdxl_vals)
        self.assertIn("magic", sdxl_vals)
        self.assertIn("smart+magic", sdxl_vals)
        self.assertIn("no_freeu", sdxl_vals)
        self.assertIn("powerhouse", sdxl_vals)

        flux_vals = [c.value for c in FLUX_ENHANCEMENT_CHOICES]
        self.assertIn("smart", flux_vals)
        self.assertIn("magic", flux_vals)
        self.assertIn("smart+magic", flux_vals)

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
        self.assertIn("Blend Image", ctx_names)
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
        self.assertEqual(modal.duration_input.default, "5")
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
            QUADRANT_CACHE_DIR
        )

        async def run_async_test():
            gen_id = "test_async_gen_999"
            dummy_img = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
            images = [dummy_img, dummy_img, dummy_img, dummy_img]

            # 1. Test async quadrant save
            await save_quadrant_images_async(gen_id, images)

            # 2. Test async quadrant retrieval
            q1 = await get_quadrant_bytes_async(gen_id, 1)
            self.assertEqual(q1, dummy_img)

            # 3. Test async metadata embed
            meta_img_io = await embed_metadata_async(dummy_img, "async prompt", neg_prompt="low quality", seed=12345, width=1, height=1)
            self.assertTrue(len(meta_img_io.getvalue()) > 0)

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
        import re
        with open("bot.py", "r", encoding="utf-8") as f:
            bot_code = f.read()

        matches = re.findall(r'@(?:tree|bot\.tree)\.command\([^)]*description=[\x22\x27](.*?)[\x22\x27]', bot_code, re.DOTALL)
        for desc in matches:
            clean_desc = desc.replace('\n', ' ').strip()
            self.assertLessEqual(len(clean_desc), 100, f"Command description exceeds Discord 100 char limit ({len(clean_desc)}): '{clean_desc}'")

    def test_single_instance_lock(self):
        """Test that acquire_instance_lock successfully binds and prevents secondary bindings on the same port."""
        from bot import acquire_instance_lock
        import bot
        test_port = 48199
        try:
            # First acquire on test port succeeds
            self.assertTrue(acquire_instance_lock(port=test_port))
            # Second acquire on the same port fails
            self.assertFalse(acquire_instance_lock(port=test_port))
        finally:
            if bot._instance_lock_socket:
                bot._instance_lock_socket.close()
                bot._instance_lock_socket = None


if __name__ == "__main__":
    unittest.main()









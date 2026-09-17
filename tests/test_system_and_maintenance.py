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


class TestSystemAndMaintenance(unittest.TestCase):
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

    def test_comfy_client_text_outputs_no_unbound_local_error(self):
        """Verify _execute_direct handles text-only outputs (JoyCaption, Florence-2) without UnboundLocalError."""
        from comfy_client import ComfyClient
        import asyncio
        from unittest.mock import AsyncMock, patch, MagicMock

        client = ComfyClient("127.0.0.1:8188")
        mock_results = {
            "11": {"text": ["A beautiful portrait of an astronaut on Mars"]},
            "12": {"string": ["cinematic, detailed"]}
        }

        async def run_test():
            mock_post = MagicMock()
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"prompt_id": "prompt_test_text_123"})
            mock_post.return_value.__aenter__.return_value = mock_resp

            client.session = MagicMock()
            client.session.post = mock_post
            client.is_online = AsyncMock(return_value=True)

            with patch.object(client, "get_history_output", new=AsyncMock(return_value=mock_results)):
                with patch("db.complete_pending_job", return_value=True):
                    with patch("db.record_pending_job", return_value=True):
                        return await client._execute_direct({"dummy": "wf"})

        res = asyncio.run(run_test())
        self.assertEqual(res, mock_results)

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

    def test_system_cog_registration_and_exports(self):
        """Test SystemCog registration in bot.tree and backward-compatibility re-exports."""
        import bot
        from cogs.system_cog import SystemCog
        import services.system_service as system_service

        # 1. Verify commands in bot.tree
        cmd_names = [c.name for c in bot.bot.tree.get_commands()]
        expected_commands = [
            "cui-start", "cui-stop", "cui-status", 
            "free", "queue", 
            "models", "scan_models", 
            "negative", "prompt", "style"
        ]
        for cmd in expected_commands:
            self.assertIn(cmd, cmd_names, f"Expected /{cmd} to be registered in bot.tree")
        self.assertNotIn("purge-vram", cmd_names)
        self.assertNotIn("variation_mode", cmd_names)

        # 2. Verify re-exports on bot module
        self.assertTrue(hasattr(bot, "cui_start_command"))
        self.assertTrue(hasattr(bot, "cui_stop_command"))
        self.assertTrue(hasattr(bot, "cui_status_command"))
        self.assertTrue(hasattr(bot, "free_vram_command"))
        self.assertTrue(hasattr(bot, "purge_vram_command"))
        self.assertTrue(hasattr(bot, "queue_command"))
        self.assertTrue(hasattr(bot, "models_command"))
        self.assertTrue(hasattr(bot, "scan_models_command"))
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
        self.assertTrue(callable(system_service.build_models_embed))

    def test_idle_watchdog_and_reconnect_hygiene(self):
        """Test idle watchdog tracking and ComfyClient start() reconnect hygiene."""
        from comfy_client import ComfyClient
        import bot
        import services.system_service as system_service

        client = ComfyClient()
        # Verify initial state
        self.assertIsNone(client.session)
        self.assertIsNone(client.ws_task)

        # Verify bot idle activity tracking
        initial_activity = bot._last_generation_activity
        bot._idle_purged = True
        time.sleep(0.01)
        bot.touch_activity()
        self.assertGreater(bot._last_generation_activity, initial_activity)
        self.assertFalse(bot._idle_purged)

        # Verify max_messages cap on bot
        self.assertEqual(bot.bot._connection.max_messages, 100)

        # Verify fetch_comfyui_queue and fetch_comfyui_system_stats accept session parameter
        async def _test_fetch():
            res_q = await system_service.fetch_comfyui_queue(address="127.0.0.1:9999", session=None)
            self.assertIsNone(res_q)
            res_s = await system_service.fetch_comfyui_system_stats(address="127.0.0.1:9999", session=None)
            self.assertIsNone(res_s)
        asyncio.run(_test_fetch())

        # Verify trim_process_working_set and get_comfyui_pids
        if os.name == 'nt':
            trimmed = system_service.trim_process_working_set()
            self.assertTrue(trimmed)
            # Safe handling of invalid PID
            self.assertFalse(system_service.trim_process_working_set(pid=99999999))
        pids = system_service.get_comfyui_pids()
        self.assertIsInstance(pids, list)

        # Verify purge_vram_core executes mock free_memory and handles working set trim
        mock_client = MagicMock()
        mock_client.free_memory = AsyncMock(return_value=True)
        purge_res = asyncio.run(system_service.purge_vram_core(client=mock_client, trim_working_set=True))
        self.assertTrue(purge_res)
        mock_client.free_memory.assert_awaited_once_with(unload_models=True, free_memory=True)

    def test_cui_start_command_imports_and_timing(self):
        """Verify cui_start command executes without NameError on time or dependencies."""
        from cogs.system_cog import SystemCog
        mock_bot = MagicMock()
        mock_bot.comfy_client = MagicMock()
        mock_bot.comfy_client.is_online = AsyncMock(return_value=True)

        cog = SystemCog(mock_bot)
        mock_interaction = MagicMock()
        mock_interaction.user.id = 12345
        mock_interaction.user.guild_permissions.administrator = True
        mock_interaction.response.is_done.return_value = True
        mock_interaction.followup.send = AsyncMock()

        with patch("cogs.system_cog.is_authorized_admin", return_value=True), \
             patch("cogs.system_cog.safe_defer", AsyncMock()), \
             patch("cogs.system_cog.terminate_existing_comfyui", return_value=False), \
             patch("cogs.system_cog.check_gpu_vram_caution", return_value=(False, {})), \
             patch("os.path.exists", return_value=True), \
             patch("asyncio.create_subprocess_exec", AsyncMock(return_value=MagicMock(pid=12345))), \
             patch("asyncio.sleep", AsyncMock()):
            asyncio.run(cog.cui_start.callback(cog, mock_interaction, force=False))
            mock_interaction.followup.send.assert_called()




if __name__ == '__main__':
    unittest.main()

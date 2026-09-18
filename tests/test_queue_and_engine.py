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


class TestQueueAndEngine(unittest.TestCase):
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

    def test_queue_presence_and_multi_job_tracking(self):
        """
        Verifies that when multiple jobs are queued in EngineAwareQueue:
        1. get_effective_queue_counts combines ComfyUI and EngineAwareQueue jobs.
        2. format_presence_status_text correctly reflects queued jobs (e.g. 'Processing 1 job | 3 queued').
        3. EngineAwareQueue change listeners fire on enqueue and cancel.
        4. build_queue_embed includes pending jobs from EngineAwareQueue.
        """
        import bot
        from bot import get_effective_queue_counts, format_presence_status_text
        from services.engine_queue import EngineAwareQueue, JobPriority
        import services.system_service as system_service

        # 1. Test format_presence_status_text
        txt, act = format_presence_status_text(1, 0)
        self.assertEqual(txt, "Processing 1 job")
        self.assertEqual(act, bot.discord.ActivityType.playing)

        txt, act = format_presence_status_text(1, 3)
        self.assertEqual(txt, "Processing 1 job | 3 queued")
        self.assertEqual(act, bot.discord.ActivityType.playing)

        txt, act = format_presence_status_text(2, 4)
        self.assertEqual(txt, "Processing 2 jobs | 4 queued")
        self.assertEqual(act, bot.discord.ActivityType.playing)

        txt, act = format_presence_status_text(0, 2)
        self.assertEqual(txt, "2 queued jobs")
        self.assertEqual(act, bot.discord.ActivityType.watching)

        txt, act = format_presence_status_text(0, 0)
        self.assertEqual(txt, "Ready ✓ | /imagine")
        self.assertEqual(act, bot.discord.ActivityType.watching)

        # 2. Test get_effective_queue_counts with ComfyUI + EngineAwareQueue
        comfy_queue = {
            "queue_running": [["prompt_1", "client_1", {}]],
            "queue_pending": []
        }
        with patch("services.engine_queue.get_engine_queue") as mock_eq:
            mock_queue_inst = MagicMock()
            mock_queue_inst.get_status.return_value = {
                "is_running": True,
                "active_job": None,
                "pending_count": 3
            }
            mock_eq.return_value = mock_queue_inst
            running, pending = get_effective_queue_counts(comfy_queue)
            self.assertEqual(running, 1)
            self.assertEqual(pending, 3)
            status_str, _ = format_presence_status_text(running, pending)
            self.assertEqual(status_str, "Processing 1 job | 3 queued")

        # 3. Test EngineAwareQueue change listener registration and callback
        queue = EngineAwareQueue()
        fired = [0]
        def on_change():
            fired[0] += 1
        queue.add_change_listener(on_change)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(queue.enqueue(
                workflow={},
                engine="krea2",
                priority=JobPriority.NORMAL,
                description="Krea 2 Blend Test"
            ))
            self.assertGreaterEqual(fired[0], 1)

            queue.cancel(queue._queue[0].job_id)
            self.assertGreaterEqual(fired[0], 2)
        finally:
            loop.close()

        # 4. Test build_queue_embed includes EngineAwareQueue pending jobs
        with patch("services.engine_queue.get_engine_queue") as mock_eq:
            mock_queue_inst = MagicMock()
            mock_queue_inst.get_status.return_value = {
                "is_running": True,
                "current_engine": "krea2",
                "current_engine_name": "Krea 2",
                "active_job": {
                    "job_id": "job_active_1",
                    "engine": "krea2",
                    "engine_name": "Krea 2",
                    "description": "Active Krea Generation",
                    "user_id": 12345,
                    "priority": "NORMAL",
                    "running_seconds": 4.5
                },
                "pending_count": 2,
                "pending_by_engine": {"krea2": 2},
                "pending_jobs": [
                    {
                        "job_id": "job_pend_1",
                        "engine": "krea2",
                        "engine_name": "Krea 2",
                        "description": "Blend Portrait #1",
                        "user_id": 12345,
                        "priority": "NORMAL",
                        "waiting_seconds": 10.2
                    },
                    {
                        "job_id": "job_pend_2",
                        "engine": "krea2",
                        "engine_name": "Krea 2",
                        "description": "Blend Portrait #2",
                        "user_id": 12345,
                        "priority": "NORMAL",
                        "waiting_seconds": 5.1
                    }
                ],
                "stats": {"switches_prevented": 1}
            }
            mock_eq.return_value = mock_queue_inst

            embed = system_service.build_queue_embed(queue={"queue_running": [], "queue_pending": []}, stats={})
            field_names = [f.name for f in embed.fields]
            field_values = {f.name: f.value for f in embed.fields}

            self.assertIn("📋 Pending (2)", field_names)
            self.assertIn("Blend Portrait #1", field_values["📋 Pending (2)"])
            self.assertIn("Blend Portrait #2", field_values["📋 Pending (2)"])

    def test_engine_queue_cancel_by_generation(self):
        """Verify cancel_by_generation cancels all matching pending and active jobs."""
        from services.engine_queue import EngineAwareQueue, JobPriority, QueueJob

        queue = EngineAwareQueue()
        loop = asyncio.new_event_loop()
        try:
            f1 = loop.create_future()
            f2 = loop.create_future()
            f3 = loop.create_future()
            f_active = loop.create_future()

            job1 = QueueJob("job_1", {}, "sdxl", JobPriority.NORMAL, f1, generation_id="gen_cancel_target")
            job2 = QueueJob("job_2", {}, "sdxl", JobPriority.NORMAL, f2, generation_id="gen_cancel_target")
            job3 = QueueJob("job_3", {}, "krea2", JobPriority.NORMAL, f3, generation_id="gen_other")
            job_act = QueueJob("job_act", {}, "sdxl", JobPriority.NORMAL, f_active, generation_id="gen_cancel_target")

            queue._queue = [job1, job2, job3]
            queue._active_job = job_act

            # Cancel matching generation_id
            cancelled_count = queue.cancel_by_generation("gen_cancel_target")
            self.assertEqual(cancelled_count, 3)
            self.assertTrue(f1.cancelled())
            self.assertTrue(f2.cancelled())
            self.assertTrue(f_active.cancelled())
            self.assertFalse(f3.cancelled())

            # Only job3 should remain in queue
            self.assertEqual(len(queue._queue), 1)
            self.assertEqual(queue._queue[0].job_id, "job_3")
            self.assertEqual(queue.stats["total_cancelled"], 3)
        finally:
            loop.close()

    def test_comfy_client_pause_generation_cancels_queue(self):
        """Verify comfy_client.pause_generation succeeds even if prompt_ids is empty by cancelling queued jobs."""
        import db
        from comfy_client import ComfyClient
        from services.engine_queue import get_engine_queue, JobPriority, QueueJob

        gen_id = "test_gen_queue_only_cancel"
        db.save_generation(gen_id, {
            "status": "pending",
            "prompt_ids": []
        })

        eq = get_engine_queue()
        loop = asyncio.new_event_loop()
        try:
            f = loop.create_future()
            job = QueueJob("job_q1", {}, "sdxl", JobPriority.NORMAL, f, generation_id=gen_id)
            eq._queue.append(job)

            client = ComfyClient()
            success = loop.run_until_complete(client.pause_generation(gen_id))
            self.assertTrue(success)
            self.assertTrue(f.cancelled())
            self.assertEqual(db.get_generation(gen_id)["status"], "stasis")
        finally:
            loop.close()


if __name__ == '__main__':
    unittest.main()


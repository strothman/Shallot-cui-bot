import asyncio
import uuid
import json
import logging
import aiohttp
from error_handler import error_handler, ErrorCategory, ErrorSeverity
import db

logger = logging.getLogger("ComfyClient")

class StasisInterruptException(Exception):
    """Exception raised when a generation is paused and put into stasis."""
    pass

class ComfyClient:
    def __init__(self, server_address="127.0.0.1:8188"):
        self.server_address = server_address
        self.client_id = str(uuid.uuid4())
        self.session = None
        self.ws = None
        self.loop = None
        self.ws_task = None
        self.futures = {} # prompt_id -> asyncio.Future
        self.results = {} # prompt_id -> node_id -> output
        self.progress_callbacks = {} # prompt_id -> callback
        self.timings = {} # prompt_id -> timestamps dict
        self.last_timing = {} # prompt_id -> calculated breakdown
        self.running = False

    async def is_online(self, retries=2, timeout=6.0) -> bool:
        """Check if the ComfyUI server is online and responding, with retries for high-load spikes."""
        url = f"http://{self.server_address}/system_stats"
        for _ in range(retries):
            try:
                if self.session and not self.session.closed:
                    async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        if resp.status == 200:
                            return True
                else:
                    async with aiohttp.ClientSession() as temp_session:
                        async with temp_session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                            if resp.status == 200:
                                return True
            except Exception:
                await asyncio.sleep(0.5)
        return False

    async def get_system_stats(self):
        """Fetch system stats (GPU, VRAM) from ComfyUI."""
        url = f"http://{self.server_address}/system_stats"
        try:
            if self.session and not self.session.closed:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                    if resp.status == 200:
                        return await resp.json()
            else:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                        if resp.status == 200:
                            return await resp.json()
        except Exception:
            return None
        return None

    async def get_queue(self):
        """Fetch queue status (running, pending) from ComfyUI."""
        url = f"http://{self.server_address}/queue"
        try:
            if self.session and not self.session.closed:
                async with self.session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                    if resp.status == 200:
                        return await resp.json()
            else:
                async with aiohttp.ClientSession() as temp_session:
                    async with temp_session.get(url, timeout=aiohttp.ClientTimeout(total=3.0)) as resp:
                        if resp.status == 200:
                            return await resp.json()
        except Exception:
            return None
        return None

    async def start(self):
        """Start the aiohttp session and websocket listener."""
        self.session = aiohttp.ClientSession()
        self.running = True
        self.loop = asyncio.get_running_loop()
        self.ws_task = asyncio.create_task(self._ws_listener())

    async def stop(self):
        """Stop session and websocket listener."""
        self.running = False
        if self.ws:
            await self.ws.close()
        if self.ws_task:
            self.ws_task.cancel()
        if self.session:
            await self.session.close()

    async def _ws_listener(self):
        """Listen to ComfyUI websocket events and resolve futures."""
        while self.running:
            try:
                ws_url = f"ws://{self.server_address}/ws?clientId={self.client_id}"
                async with self.session.ws_connect(ws_url) as ws:
                    self.ws = ws
                    logger.info("Connected to ComfyUI WebSocket.")
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            data = json.loads(msg.data)
                            await self._handle_ws_msg(data)
            except (aiohttp.ClientConnectorError, aiohttp.ClientError, OSError) as conn_err:
                # ComfyUI server is offline or restarting — silently wait and retry without spamming error logs
                logger.debug(f"ComfyUI WebSocket not connected ({conn_err}). Retrying in 5s...")
                self.ws = None
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                error_handler.log_error(
                    e,
                    category=ErrorCategory.WORKFLOW,
                    source_function="_ws_listener",
                    source_file="comfy_client.py",
                    severity=ErrorSeverity.WARNING,
                    context={"server_address": self.server_address}
                )
                logger.warning(f"WebSocket error: {e}. Reconnecting in 5s...")
                await asyncio.sleep(5)

    async def _handle_ws_msg(self, data):
        """Handle incoming WebSocket messages from ComfyUI."""
        msg_type = data.get("type")
        msg_data = data.get("data", {})
        prompt_id = msg_data.get("prompt_id")

        if not prompt_id or prompt_id not in self.futures:
            return

        future = self.futures[prompt_id]

        if msg_type == "progress":
            val = msg_data.get("value")
            max_val = msg_data.get("max")
            if prompt_id in self.timings:
                now = asyncio.get_event_loop().time()
                if self.timings[prompt_id]["first_step"] is None:
                    self.timings[prompt_id]["first_step"] = now
                self.timings[prompt_id]["last_step"] = now
            if prompt_id in self.progress_callbacks and val is not None and max_val is not None:
                cb = self.progress_callbacks[prompt_id]
                try:
                    if asyncio.iscoroutinefunction(cb):
                        asyncio.create_task(cb(val, max_val))
                    else:
                        cb(val, max_val)
                except Exception as cb_err:
                    logger.debug(f"Progress callback error: {cb_err}")

        elif msg_type == "executing" and msg_data.get("node") is None:
            # Execution finished (this event sends node=None when finished)
            now = asyncio.get_event_loop().time()
            if prompt_id in self.timings:
                t = self.timings.pop(prompt_id)
                t_sub = t.get("submitted", now)
                t_first = t.get("first_step") or now
                t_last = t.get("last_step") or t_first
                if len(self.last_timing) > 100:
                    oldest = next(iter(self.last_timing))
                    self.last_timing.pop(oldest, None)
                self.last_timing[prompt_id] = {
                    "init_duration": round(max(0.0, t_first - t_sub), 2),
                    "sampling_duration": round(max(0.0, t_last - t_first), 2),
                    "post_duration": round(max(0.0, now - t_last), 2),
                    "total_duration": round(max(0.0, now - t_sub), 2)
                }
            if not future.done():
                future.set_result(self.results.pop(prompt_id, {}))
                self.futures.pop(prompt_id, None)

        elif msg_type == "executed":
            node_id = msg_data.get("node")
            node_output = msg_data.get("output", {})
            if prompt_id in self.results:
                self.results[prompt_id][node_id] = node_output

        elif msg_type == "execution_error":
            node_id = msg_data.get("node_id")
            node_type = msg_data.get("node_type")
            err_msg = msg_data.get("exception_message", "Unknown error")
            err_obj = Exception(f"ComfyUI execution error on node {node_id} ({node_type}): {err_msg}")
            error_handler.log_error(
                err_obj,
                category=ErrorCategory.WORKFLOW,
                source_function="_handle_ws_msg",
                source_file="comfy_client.py",
                severity=ErrorSeverity.ERROR,
                context={"prompt_id": prompt_id, "node_id": node_id, "node_type": node_type, "exception_message": err_msg}
            )
            logger.error(f"ComfyUI execution error on node {node_id} ({node_type}): {err_msg}")
            self.results.pop(prompt_id, None)
            if not future.done():
                future.set_exception(err_obj)
                self.futures.pop(prompt_id, None)

        elif msg_type == "execution_interrupted":
            err_obj = Exception("ComfyUI execution was interrupted.")
            error_handler.log_error(
                err_obj,
                category=ErrorCategory.WORKFLOW,
                source_function="_handle_ws_msg",
                source_file="comfy_client.py",
                severity=ErrorSeverity.WARNING,
                context={"prompt_id": prompt_id}
            )
            logger.warning("ComfyUI execution was interrupted.")
            self.results.pop(prompt_id, None)
            if not future.done():
                future.set_exception(err_obj)
                self.futures.pop(prompt_id, None)

    async def get_image(self, filename, subfolder, img_type):
        """Download raw image bytes from ComfyUI."""
        url = f"http://{self.server_address}/view"
        params = {"filename": filename, "subfolder": subfolder, "type": img_type}
        async with self.session.get(url, params=params) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                err = Exception(f"Failed to get image: HTTP {resp.status}")
                error_handler.log_error(
                    err,
                    category=ErrorCategory.IMAGE_IO,
                    source_function="get_image",
                    source_file="comfy_client.py",
                    severity=ErrorSeverity.ERROR,
                    context={"filename": filename, "subfolder": subfolder, "img_type": img_type, "status": resp.status}
                )
                raise err

    async def get_history_output(self, prompt_id):
        """Query ComfyUI /history/{prompt_id} to check if execution completed or errored."""
        if not self.session:
            return None
        url = f"http://{self.server_address}/history/{prompt_id}"
        try:
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if prompt_id in data:
                        entry = data[prompt_id]
                        status = entry.get("status", {})
                        if status.get("status_str") == "error":
                            messages = status.get("messages", [])
                            err_msg = str(messages) if messages else "Execution error"
                            raise Exception(f"ComfyUI execution error from history: {err_msg}")
                        return entry.get("outputs", {})
        except Exception as e:
            if "ComfyUI execution error" in str(e):
                raise e
        return None

    async def generate(self, workflow, timeout=14400, retries=1, generation_id=None, progress_callback=None):
        """Submit prompt to ComfyUI and await the generated images with automatic retry for transient errors."""
        if not await self.is_online():
            raise Exception("ComfyUI server is currently offline. Please start it using `/cui-start` first.")

        attempt = 0
        last_exception = None

        while attempt <= retries:
            attempt += 1
            if not self.session:
                await self.start()

            url = f"http://{self.server_address}/prompt"
            extra_pnginfo = {}
            if isinstance(workflow, dict) and "nodes" in workflow:
                extra_pnginfo["workflow"] = workflow

            payload = {
                "prompt": workflow,
                "client_id": self.client_id,
                "extra_data": {
                    "extra_pnginfo": extra_pnginfo
                }
            }

            try:
                async with self.session.post(url, json=payload) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        raise Exception(f"Failed to queue prompt: HTTP {resp.status} - {text}")
                    
                    res = await resp.json()
                    prompt_id = res["prompt_id"]
                    if len(self.timings) > 100:
                        oldest_t = next(iter(self.timings))
                        self.timings.pop(oldest_t, None)
                    self.timings[prompt_id] = {
                        "submitted": asyncio.get_event_loop().time(),
                        "first_step": None,
                        "last_step": None,
                        "finished": None
                    }

                    if progress_callback:
                        self.progress_callbacks[prompt_id] = progress_callback

                    if generation_id:
                        gen_data = db.get_generation(generation_id)
                        if gen_data:
                            if "prompt_ids" not in gen_data:
                                gen_data["prompt_ids"] = []
                            if prompt_id not in gen_data["prompt_ids"]:
                                gen_data["prompt_ids"].append(prompt_id)
                            db.save_generation(generation_id, gen_data)

                future = self.loop.create_future()
                self.futures[prompt_id] = future
                self.results[prompt_id] = {}

                # Check if history already finished (e.g. cached response)
                initial_history = await self.get_history_output(prompt_id)
                if initial_history is not None and not future.done():
                    future.set_result(initial_history)

                try:
                    try:
                        start_time = asyncio.get_event_loop().time()
                        while not future.done():
                            elapsed = asyncio.get_event_loop().time() - start_time
                            if elapsed >= timeout:
                                final_history = await self.get_history_output(prompt_id)
                                if final_history is not None and not future.done():
                                    future.set_result(final_history)
                                    break
                                raise asyncio.TimeoutError()

                            remaining = min(2.0, max(0.1, timeout - elapsed))
                            try:
                                results_dict = await asyncio.wait_for(asyncio.shield(future), timeout=remaining)
                                break
                            except asyncio.TimeoutError:
                                # 2s polling window expired; check history endpoint as fallback
                                poll_history = await self.get_history_output(prompt_id)
                                if poll_history is not None and not future.done():
                                    future.set_result(poll_history)
                                    break

                        if future.exception():
                            raise future.exception()

                        results_dict = future.result()
                        
                        # Check if there are any images, gifs, or videos in the outputs
                        output_bytes_list = []
                        has_outputs = False
                        
                        for node_id, output in results_dict.items():
                            if output:
                                for key in ["images", "gifs", "videos"]:
                                    if key in output:
                                        has_outputs = True
                                        for item in output[key]:
                                            file_bytes = await self.get_image(item["filename"], item.get("subfolder", ""), item.get("type", "output"))
                                            output_bytes_list.append(file_bytes)
                        
                        if has_outputs:
                            return output_bytes_list
                        else:
                            return results_dict

                    finally:
                        self.futures.pop(prompt_id, None)
                        self.results.pop(prompt_id, None)
                        self.progress_callbacks.pop(prompt_id, None)

                except StasisInterruptException as e:
                    raise e
                except asyncio.TimeoutError:
                    err = Exception("Generation timed out.")
                    error_handler.log_error(
                        err,
                        category=ErrorCategory.WORKFLOW,
                        source_function="generate",
                        source_file="comfy_client.py",
                        severity=ErrorSeverity.ERROR,
                        context={"prompt_id": prompt_id, "timeout": timeout, "attempt": attempt}
                    )
                    last_exception = err
                except Exception as e:
                    last_exception = e

            except StasisInterruptException as e:
                raise e
            except Exception as e:
                last_exception = e
                error_handler.log_error(
                    e,
                    category=ErrorCategory.WORKFLOW,
                    source_function="generate",
                    source_file="comfy_client.py",
                    severity=ErrorSeverity.WARNING if attempt <= retries else ErrorSeverity.ERROR,
                    context={"server_address": self.server_address, "attempt": attempt}
                )

            if attempt <= retries:
                logger.warning(f"Generation attempt {attempt} failed ({last_exception}). Retrying in 2 seconds...")
                await asyncio.sleep(2)

        raise last_exception

    async def pause_generation(self, generation_id):
        """Pause a generation by cancelling its pending prompts in ComfyUI and raising StasisInterruptException on its futures."""
        gen_data = db.get_generation(generation_id)
        if not gen_data:
            return False
        
        prompt_ids = gen_data.get("prompt_ids", [])
        if not prompt_ids:
            return False
        
        # 1. Fetch ComfyUI Queue
        queue_data = None
        try:
            if not self.session:
                await self.start()
            url = f"http://{self.server_address}/queue"
            async with self.session.get(url) as resp:
                if resp.status == 200:
                    queue_data = await resp.json()
        except Exception as e:
            logger.error(f"Failed to fetch queue for pause: {e}")
        
        running_ids = []
        if queue_data:
            running_ids = [job[1] for job in queue_data.get("queue_running", [])]
        
        # 2. Cancel/delete them in ComfyUI
        for p_id in prompt_ids:
            if p_id in running_ids:
                try:
                    url = f"http://{self.server_address}/interrupt"
                    async with self.session.post(url) as resp:
                        pass
                except Exception as e:
                    logger.error(f"Failed to interrupt running prompt {p_id}: {e}")
            else:
                try:
                    url = f"http://{self.server_address}/queue"
                    payload = {"delete": [p_id]}
                    async with self.session.post(url, json=payload) as resp:
                        pass
                except Exception as e:
                    logger.error(f"Failed to delete pending prompt {p_id}: {e}")
        
        # 3. Raise StasisInterruptException on the futures
        for p_id in prompt_ids:
            future = self.futures.get(p_id)
            if future and not future.done():
                future.set_exception(StasisInterruptException("Generation paused and put in stasis."))
                self.futures.pop(p_id, None)
                self.results.pop(p_id, None)
        
        # 4. Update the DB status
        gen_data["status"] = "stasis"
        db.save_generation(generation_id, gen_data)
        return True


    async def upload_image(self, image_bytes, filename):
        """Upload an image to ComfyUI server."""
        if not self.session:
            await self.start()
        url = f"http://{self.server_address}/upload/image"
        data = aiohttp.FormData()
        data.add_field('image', image_bytes, filename=filename, content_type='image/png')
        async with self.session.post(url, data=data) as resp:
            if resp.status == 200:
                res = await resp.json()
                return res  # returns dictionary with "name", "subfolder", "type"
            else:
                text = await resp.text()
                err = Exception(f"Failed to upload image: HTTP {resp.status} - {text}")
                error_handler.log_error(
                    err,
                    category=ErrorCategory.IMAGE_IO,
                    source_function="upload_image",
                    source_file="comfy_client.py",
                    severity=ErrorSeverity.ERROR,
                    context={"filename": filename, "status": resp.status}
                )
                raise err

    def get_execution_timing(self, prompt_id: str = None) -> dict:
        """Returns the phase-by-phase execution time breakdown for a prompt_id or the most recent run."""
        if prompt_id and prompt_id in self.last_timing:
            return self.last_timing[prompt_id]
        if self.last_timing:
            return list(self.last_timing.values())[-1]
        return {"init_duration": 0.0, "sampling_duration": 0.0, "post_duration": 0.0, "total_duration": 0.0}

    async def check_dependencies(self, workflows_dir="workflows", settings_path="settings.json"):
        """
        Scan workflow JSONs, settings.json, and hardcoded bot references to build a manifest
        of every model file the bot expects (checkpoints, LoRAs, VAEs, UNETs, CLIP models,
        upscale models, RIFE checkpoints). Then query ComfyUI /object_info to see what's
        actually installed and report any missing dependencies.

        Returns a dict:
            {
                "available": {category: [list of installed filenames]},
                "required":  {category: {filename: [sources...]}},
                "missing":   {category: {filename: [sources...]}},
                "ok":        bool  (True if nothing is missing)
            }
        """
        import os
        import glob

        # ── 1. Build the "required" manifest from all sources ──────────────────

        # Maps: category -> { filename -> set of sources }
        required = {
            "checkpoints": {},
            "loras": {},
            "vae": {},
            "unets": {},
            "clip": {},
            "upscale_models": {},
            "rife": {},
        }

        def _add(category, filename, source):
            if not filename or filename in ("", "none", "None"):
                return
            required[category].setdefault(filename, set()).add(source)

        # ── 1a. Scan workflow JSON files ───────────────────────────────────────
        # Maps ComfyUI class_type -> (input_key, our category)
        NODE_MODEL_KEYS = {
            "CheckpointLoaderSimple":  [("ckpt_name", "checkpoints")],
            "LoraLoader":             [("lora_name", "loras")],
            "LoraLoaderModelOnly":    [("lora_name", "loras")],
            "VAELoader":              [("vae_name", "vae")],
            "UNETLoader":             [("unet_name", "unets")],
            "UnetLoaderGGUF":         [("unet_name", "unets")],
            "CLIPLoader":             [("clip_name", "clip")],
            "CLIPVisionLoader":       [("clip_name", "clip")],
            "DualCLIPLoader":         [("clip_name1", "clip"), ("clip_name2", "clip")],
            "DualCLIPLoaderGGUF":     [("clip_name1", "clip"), ("clip_name2", "clip")],
            "UpscaleModelLoader":     [("model_name", "upscale_models")],
            "RIFE_VFI":               [("ckpt_name", "rife")],
            "RIFE VFI":               [("ckpt_name", "rife")],
            "UltralyticsDetectorProvider": [("model_name", "bbox_detectors")],
        }

        if os.path.isdir(workflows_dir):
            for wf_path in glob.glob(os.path.join(workflows_dir, "*.json")):
                wf_name = os.path.basename(wf_path)
                try:
                    with open(wf_path, "r", encoding="utf-8") as f:
                        wf = json.load(f)
                    for node_id, node in wf.items():
                        class_type = node.get("class_type", "")
                        if class_type in NODE_MODEL_KEYS:
                            inputs = node.get("inputs", {})
                            for input_key, category in NODE_MODEL_KEYS[class_type]:
                                val = inputs.get(input_key)
                                if isinstance(val, str) and val:
                                    _add(category, val, f"workflow:{wf_name}")
                except Exception:
                    pass

        # ── 1b. Scan settings.json ─────────────────────────────────────────────
        if os.path.isfile(settings_path):
            try:
                with open(settings_path, "r", encoding="utf-8") as f:
                    settings = json.load(f)
                for key, category in [
                    ("wan_high_gguf", "unets"),
                    ("wan_low_gguf", "unets"),
                    ("wan_clip", "clip"),
                    ("wan_clip_vision", "clip"),
                    ("wan_vae", "vae"),
                    ("rife_ckpt", "rife"),
                ]:
                    val = settings.get(key)
                    if val:
                        _add(category, val, "settings.json")
            except Exception:
                pass

        # ── 1c. Hardcoded bot references ───────────────────────────────────────
        # SDXL checkpoint choices from bot.py
        HARDCODED_CHECKPOINTS = [
            "waiIllustriousSDXL_v170.safetensors",
            "RealVisXL_V4.0.safetensors",
            "juggernautXL_ragnarok.safetensors",
            "CopaxTimeLessXL.safetensors",
            "ultraRealisticByStable_v25.safetensors",
            "hyphoriaRealIllu_v09.safetensors",
            "hyphoriaIlluNAI_v001.safetensors",
            "illustriousRealismBy_v10VAE.safetensors",
            "bigLust_v16.safetensors",
            "lustifySDXLNSFWSFW_v10.safetensors",
            "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
            "RealVisXL_V5.0_Lightning_fp16.safetensors",
            "novaFurryXL_ilV180A.safetensors",
        ]
        for ckpt in HARDCODED_CHECKPOINTS:
            _add("checkpoints", ckpt, "bot.py:SDXL_CHECKPOINT_CHOICES")

        # Default LoRAs used by the bot
        HARDCODED_LORAS = [
            "Semi-realism_illustrious.safetensors",
            "ogarla_epoch_5.safetensors",
            "ogarlaflux_epoch_5.safetensors",
        ]
        for lora in HARDCODED_LORAS:
            _add("loras", lora, "bot.py/parsers.py:hardcoded")

        # Upscaler
        _add("upscale_models", "4x_foolhardy_Remacri.pth", "bot.py:upscale_command")

        # ── 2. Query ComfyUI /object_info to see what's installed ──────────────
        available = {
            "checkpoints": [],
            "loras": [],
            "vae": [],
            "unets": [],
            "clip": [],
            "upscale_models": [],
            "rife": [],
            "bbox_detectors": [],
        }

        try:
            url = f"http://{self.server_address}/object_info"
            session = self.session
            needs_close = False
            if not session or session.closed:
                session = aiohttp.ClientSession()
                needs_close = True
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10.0)) as resp:
                    if resp.status == 200:
                        data = await resp.json()

                        def _extract(node_type, input_key):
                            try:
                                node_def = data.get(node_type)
                                if not node_def:
                                    return []
                                raw = node_def["input"]["required"][input_key]
                                if isinstance(raw, list):
                                    if len(raw) > 0 and isinstance(raw[0], (list, tuple)):
                                        return list(raw[0])
                                    for item in raw:
                                        if isinstance(item, dict) and "options" in item:
                                            return list(item["options"])
                                        if isinstance(item, (list, tuple)):
                                            return list(item)
                                return []
                            except (KeyError, IndexError, TypeError):
                                return []

                        available["checkpoints"] = _extract("CheckpointLoaderSimple", "ckpt_name")
                        available["vae"] = _extract("VAELoader", "vae_name")
                        available["upscale_models"] = _extract("UpscaleModelLoader", "model_name")
                        available["bbox_detectors"] = _extract("UltralyticsDetectorProvider", "model_name")

                        # LoRAs: check both LoraLoader and LoraLoaderModelOnly
                        lora_set = set()
                        lora_set.update(_extract("LoraLoader", "lora_name"))
                        lora_set.update(_extract("LoraLoaderModelOnly", "lora_name"))
                        available["loras"] = list(lora_set)

                        # UNETs: check UNETLoader and UnetLoaderGGUF
                        unet_set = set()
                        unet_set.update(_extract("UNETLoader", "unet_name"))
                        unet_set.update(_extract("UnetLoaderGGUF", "unet_name"))
                        available["unets"] = list(unet_set)

                        # CLIP: CLIPLoader, CLIPVisionLoader, DualCLIPLoader, DualCLIPLoaderGGUF
                        clip_set = set()
                        clip_set.update(_extract("CLIPLoader", "clip_name"))
                        clip_set.update(_extract("CLIPVisionLoader", "clip_name"))
                        for dual_key in ["clip_name1", "clip_name2"]:
                            clip_set.update(_extract("DualCLIPLoader", dual_key))
                            clip_set.update(_extract("DualCLIPLoaderGGUF", dual_key))
                        available["clip"] = list(clip_set)

                        # RIFE (supports both RIFE VFI and RIFE_VFI class names)
                        rife_list = _extract("RIFE VFI", "ckpt_name") or _extract("RIFE_VFI", "ckpt_name")
                        available["rife"] = rife_list
            finally:
                if needs_close:
                    await session.close()
        except Exception as e:
            logger.warning(f"Could not query ComfyUI /object_info for dependency check: {e}")

        # ── 3. Diff: find what's required but not available ────────────────────
        missing = {}
        for category, models in required.items():
            avail_set = set(available.get(category, []))
            for filename, sources in models.items():
                if filename not in avail_set:
                    missing.setdefault(category, {})[filename] = list(sources)

        # Convert source sets to lists for the result
        required_out = {}
        for category, models in required.items():
            if models:
                required_out[category] = {fn: list(srcs) for fn, srcs in models.items()}

        return {
            "available": available,
            "required": required_out,
            "missing": missing,
            "ok": len(missing) == 0,
        }

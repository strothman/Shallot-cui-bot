"""
Engine-Aware Priority Queue & VRAM Thrashing Prevention Service.
Intelligently groups generation jobs by model architecture (SDXL, Flux, Krea 2, Wan 2.2, LTX, Florence-2),
prioritizes fast interactive requests ahead of long renders, prevents starvation with dynamic age escalation,
and automatically purges VRAM during necessary architecture transitions.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Any, Optional, List, Callable, Union

logger = logging.getLogger("DiscordBot.EngineQueue")


class EngineType:
    SDXL = "sdxl"
    FLUX = "flux"
    KREA2 = "krea2"
    WAN = "wan"
    LTX = "ltx"
    HUNYUAN = "hunyuan"
    FLORENCE2 = "florence2"
    OTHER = "other"

    ALL = [SDXL, FLUX, KREA2, WAN, LTX, HUNYUAN, FLORENCE2, OTHER]


ENGINE_DISPLAY_NAMES = {
    EngineType.SDXL: "🎨 SDXL",
    EngineType.FLUX: "⚡ Flux.1",
    EngineType.KREA2: "📸 Krea 2 Turbo",
    EngineType.WAN: "🎬 Wan 2.2 Video",
    EngineType.LTX: "🎥 LTX Video",
    EngineType.HUNYUAN: "🐉 Hunyuan Video",
    EngineType.FLORENCE2: "🔍 Florence-2 Vision",
    EngineType.OTHER: "📦 Standard",
}


class JobPriority(IntEnum):
    LOW = 1       # Multi-minute videos, batch jobs, synthetic dataset generation
    NORMAL = 2    # Standard /imagine, /flux, /blend-krea generations
    HIGH = 3      # Interactive button actions (remix, reroll, quick upscales, vision interrogation)


def detect_workflow_engine(workflow: Dict[str, Any]) -> str:
    """
    Fast, zero-overhead heuristic detector that classifies a ComfyUI workflow
    into its target engine architecture without loading any model weights.
    """
    if not isinstance(workflow, dict):
        return EngineType.OTHER

    # Stringify workflow text once for broad pattern matching if needed
    full_text = ""
    try:
        full_text = str(workflow).lower()
    except Exception:
        pass

    # 1. Florence-2 Vision Analysis
    if "florence2" in full_text or "florence-2" in full_text:
        return EngineType.FLORENCE2

    # 2. Video Models (Wan 2.2, LTX, Hunyuan)
    if "wanvideo" in full_text or "wan2" in full_text or "wan_2" in full_text or "wan21" in full_text:
        return EngineType.WAN
    if "ltxvideo" in full_text or "ltx" in full_text or "ltxv" in full_text:
        return EngineType.LTX
    if "hunyuan" in full_text:
        return EngineType.HUNYUAN

    # 3. Krea 2 / Bertflow
    if any(k in full_text for k in ["krea2", "krea_2", "wetness_krea2", "muse", "pornmaster", "bertflow"]):
        return EngineType.KREA2

    # 4. Flux.1
    if any(k in full_text for k in ["fluxguidance", "flux1-dev", "flux1-schnell", "flux1", "ogarlaflux"]):
        return EngineType.FLUX

    # 5. SDXL / Default
    if any(k in full_text for k in ["sdxl", "illustrious", "pony", "hyphoria", "wai-ani", "wai-real"]):
        return EngineType.SDXL

    # If KSampler or standard generation nodes are found, default to SDXL
    if "ksampler" in full_text or "vaedecode" in full_text:
        return EngineType.SDXL

    return EngineType.OTHER


@dataclass
class QueueJob:
    job_id: str
    workflow: Dict[str, Any]
    engine: str
    priority: JobPriority
    future: asyncio.Future
    enqueued_at: float = field(default_factory=time.time)
    progress_callback: Optional[Callable] = None
    generation_id: Optional[str] = None
    user_id: Optional[int] = None
    channel_id: Optional[int] = None
    message_id: Optional[int] = None
    command_type: str = "imagine"
    metadata: Optional[Dict[str, Any]] = None
    description: str = "Generation"
    timeout: float = 14400
    retries: int = 1
    cancelled: bool = False

    def effective_score(self, current_engine: Optional[str], starvation_seconds: float = 45.0) -> float:
        """
        Calculates dynamic job scheduling score:
        - Higher score = scheduled earlier.
        - Bonus points for matching current active engine in VRAM (affinity).
        - Escalates as job waits in queue to prevent starvation.
        """
        age = time.time() - self.enqueued_at
        score = float(self.priority) * 10.0

        # Model affinity bonus: +15 points if engine is already loaded in VRAM
        if current_engine and self.engine == current_engine:
            score += 15.0

        # Anti-starvation escalation: Add 1 point for every 3 seconds waited
        # After starvation_seconds, age weight jumps sharply to guarantee next pick
        if age >= starvation_seconds:
            score += 100.0 + (age - starvation_seconds)
        else:
            score += (age / 3.0)

        return score


class EngineAwareQueue:
    """
    Intelligent Priority Queue that manages ComfyUI generation jobs with
    model-affinity scheduling, VRAM purge safety, and fair anti-starvation.
    """

    def __init__(
        self,
        comfy_client=None,
        starvation_timeout: float = 45.0,
        enable_affinity: bool = True
    ):
        self.comfy_client = comfy_client
        self.starvation_timeout = starvation_timeout
        self.enable_affinity = enable_affinity

        self._queue: List[QueueJob] = []
        self._active_job: Optional[QueueJob] = None
        self._current_engine: Optional[str] = None
        self._worker_task: Optional[asyncio.Task] = None
        self._queue_lock = asyncio.Lock()
        self._new_job_event = asyncio.Event()
        self._running = False

        # Statistics & telemetry
        self.stats = {
            "total_enqueued": 0,
            "total_completed": 0,
            "total_cancelled": 0,
            "switches_prevented": 0,
            "engine_switches": 0,
        }
        self._change_listeners: List[Callable] = []

    def add_change_listener(self, callback: Callable):
        """Registers a callback invoked when queue state changes (enqueue, start, cancel, complete)."""
        if callback not in self._change_listeners:
            self._change_listeners.append(callback)

    def remove_change_listener(self, callback: Callable):
        """Unregisters a change listener callback."""
        if callback in self._change_listeners:
            self._change_listeners.remove(callback)

    def _notify_change(self):
        """Invokes registered callbacks safely without blocking the event loop."""
        for cb in list(self._change_listeners):
            try:
                res = cb()
                if asyncio.iscoroutine(res):
                    try:
                        loop = asyncio.get_running_loop()
                        loop.create_task(res)
                    except RuntimeError:
                        pass
            except Exception as e:
                logger.debug(f"Queue change callback error: {e}")

    @property
    def current_engine(self) -> Optional[str]:
        return self._current_engine

    @property
    def active_job(self) -> Optional[QueueJob]:
        return self._active_job

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Starts the background queue dispatcher worker."""
        if self._running:
            return
        self._running = True
        active_loop = loop or asyncio.get_event_loop()
        self._worker_task = active_loop.create_task(self._worker_loop())
        logger.info("EngineAwareQueue worker started.")

    async def stop(self):
        """Gracefully cancels the worker loop and cancels pending jobs."""
        self._running = False
        self._new_job_event.set()
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
            self._worker_task = None

        async with self._queue_lock:
            for job in self._queue:
                if not job.future.done():
                    job.future.cancel()
            self._queue.clear()
        logger.info("EngineAwareQueue worker stopped.")

    async def enqueue(
        self,
        workflow: Dict[str, Any],
        engine: Optional[str] = None,
        priority: JobPriority = JobPriority.NORMAL,
        progress_callback: Optional[Callable] = None,
        generation_id: Optional[str] = None,
        user_id: Optional[int] = None,
        channel_id: Optional[int] = None,
        message_id: Optional[int] = None,
        command_type: str = "imagine",
        metadata: Optional[Dict[str, Any]] = None,
        description: str = "Generation",
        timeout: float = 14400,
        retries: int = 1
    ) -> asyncio.Future:
        """
        Enqueues a workflow job and returns an asyncio.Future that resolves with
        the outputs generated by ComfyUI.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        detected_engine = engine or detect_workflow_engine(workflow)
        import uuid
        job_id = f"job_{uuid.uuid4().hex[:8]}"

        job = QueueJob(
            job_id=job_id,
            workflow=workflow,
            engine=detected_engine,
            priority=priority,
            future=future,
            enqueued_at=time.time(),
            progress_callback=progress_callback,
            generation_id=generation_id,
            user_id=user_id,
            channel_id=channel_id,
            message_id=message_id,
            command_type=command_type,
            metadata=metadata,
            description=description,
            timeout=timeout,
            retries=retries
        )

        async with self._queue_lock:
            self._queue.append(job)
            self.stats["total_enqueued"] += 1

        logger.info(
            f"Enqueued job {job_id} [{detected_engine.upper()}] (priority={priority.name}, queue_depth={len(self._queue)})"
        )
        self._new_job_event.set()
        self._notify_change()
        return future

    def cancel(self, job_id: str) -> bool:
        """Cancels a pending or active job by its job_id."""
        for job in self._queue:
            if job.job_id == job_id:
                job.cancelled = True
                if not job.future.done():
                    job.future.cancel()
                self._queue.remove(job)
                self.stats["total_cancelled"] += 1
                logger.info(f"Cancelled pending job {job_id}.")
                self._notify_change()
                return True

        if self._active_job and self._active_job.job_id == job_id:
            self._active_job.cancelled = True
            if not self._active_job.future.done():
                self._active_job.future.cancel()
            self.stats["total_cancelled"] += 1
            logger.info(f"Cancelled active job {job_id}.")
            self._notify_change()
            return True

        return False

    def _select_next_job(self) -> Optional[QueueJob]:
        """
        Selects the best job to run next from pending queue according to:
        - Affinity bonus matching current_engine (if enable_affinity)
        - Base priority level
        - Anti-starvation age weighting
        """
        if not self._queue:
            return None

        if not self.enable_affinity or not self._current_engine:
            # Simple priority + age sort (FIFO within same priority)
            self._queue.sort(key=lambda j: (j.priority, -j.enqueued_at), reverse=True)
            return self._queue.pop(0)

        # Score every pending job
        best_job = None
        best_score = -float("inf")
        best_idx = 0

        for idx, job in enumerate(self._queue):
            if job.cancelled:
                continue
            score = job.effective_score(self._current_engine, starvation_seconds=self.starvation_timeout)
            if score > best_score:
                best_score = score
                best_job = job
                best_idx = idx

        if best_job:
            self._queue.pop(best_idx)
            # Track if an engine switch was prevented
            if self._current_engine and best_job.engine == self._current_engine and len(self._queue) > 0:
                # If there were jobs with different engines in the queue, we successfully avoided a switch
                if any(j.engine != self._current_engine for j in self._queue):
                    self.stats["switches_prevented"] += 1
            return best_job

        return None

    async def _worker_loop(self):
        """Main dispatcher loop."""
        while self._running:
            job: Optional[QueueJob] = None

            async with self._queue_lock:
                job = self._select_next_job()

            if not job:
                self._new_job_event.clear()
                try:
                    await self._new_job_event.wait()
                except asyncio.CancelledError:
                    break
                continue

            if job.cancelled:
                continue

            self._active_job = job
            self._notify_change()

            try:
                # 1. Check for Engine Switch and execute VRAM purge if changing models
                if self._current_engine is not None and self._current_engine != job.engine:
                    logger.info(
                        f"Engine transition detected: [{self._current_engine.upper()}] -> [{job.engine.upper()}]. Purging VRAM cache."
                    )
                    self.stats["engine_switches"] += 1
                    if self.comfy_client and hasattr(self.comfy_client, "free_memory"):
                        try:
                            await self.comfy_client.free_memory(unload_models=True, free_memory=True)
                        except Exception as purge_err:
                            logger.warning(f"VRAM purge warning during engine transition: {purge_err}")

                self._current_engine = job.engine

                # 2. Execute via direct ComfyClient call
                if self.comfy_client:
                    # Invoke raw direct execution on ComfyClient
                    exec_func = getattr(self.comfy_client, "_execute_direct", None) or self.comfy_client.generate
                    outputs = await exec_func(
                        workflow=job.workflow,
                        timeout=job.timeout,
                        retries=job.retries,
                        generation_id=job.generation_id,
                        progress_callback=job.progress_callback,
                        channel_id=job.channel_id,
                        message_id=job.message_id,
                        user_id=job.user_id,
                        command_type=job.command_type,
                        metadata=job.metadata,
                        use_queue=False  # Avoid recursive re-queueing
                    )
                    if not job.future.done():
                        job.future.set_result(outputs)
                else:
                    # Mock output for standalone testing
                    if not job.future.done():
                        job.future.set_result({"status": "mock_completed", "engine": job.engine})

                self.stats["total_completed"] += 1

            except asyncio.CancelledError:
                if not job.future.done():
                    job.future.cancel()
                logger.info(f"Job {job.job_id} cancelled during execution.")
            except Exception as e:
                logger.error(f"Job {job.job_id} execution failed: {e}", exc_info=True)
                if not job.future.done():
                    job.future.set_exception(e)
            finally:
                self._active_job = None
                self._notify_change()

    def get_status(self) -> Dict[str, Any]:
        """Returns structured metrics for Discord status and /queue command."""
        pending_by_engine: Dict[str, int] = {}
        for j in self._queue:
            pending_by_engine[j.engine] = pending_by_engine.get(j.engine, 0) + 1

        active_info = None
        if self._active_job:
            active_info = {
                "job_id": self._active_job.job_id,
                "engine": self._active_job.engine,
                "engine_name": ENGINE_DISPLAY_NAMES.get(self._active_job.engine, self._active_job.engine),
                "description": self._active_job.description,
                "user_id": self._active_job.user_id,
                "priority": self._active_job.priority.name,
                "running_seconds": round(time.time() - self._active_job.enqueued_at, 1),
            }

        pending_jobs = [
            {
                "job_id": j.job_id,
                "engine": j.engine,
                "engine_name": ENGINE_DISPLAY_NAMES.get(j.engine, j.engine.upper()),
                "description": j.description,
                "user_id": j.user_id,
                "priority": j.priority.name,
                "waiting_seconds": round(time.time() - j.enqueued_at, 1),
            }
            for j in self._queue
        ]

        return {
            "is_running": self._running,
            "current_engine": self._current_engine,
            "current_engine_name": ENGINE_DISPLAY_NAMES.get(self._current_engine, "None") if self._current_engine else "Idle",
            "active_job": active_info,
            "pending_count": len(self._queue),
            "pending_by_engine": pending_by_engine,
            "pending_jobs": pending_jobs,
            "stats": dict(self.stats)
        }


# Global queue singleton
_global_engine_queue: Optional[EngineAwareQueue] = None


def get_engine_queue() -> EngineAwareQueue:
    """Returns or lazily creates the global EngineAwareQueue singleton."""
    global _global_engine_queue
    if _global_engine_queue is None:
        _global_engine_queue = EngineAwareQueue()
    return _global_engine_queue


def set_engine_queue_client(comfy_client) -> EngineAwareQueue:
    """Binds ComfyClient to the global EngineAwareQueue."""
    queue = get_engine_queue()
    queue.comfy_client = comfy_client
    return queue

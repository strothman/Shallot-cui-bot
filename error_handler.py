"""
Centralized Error Handling & Auto-Fix System for CUI-Server-Bot.

Provides:
- Structured JSON error journal (error_log.json) with rolling history
- Error categorization (WORKFLOW, IMAGE_IO, DISCORD, PARSING, CACHE)
- Auto-fix registry: known error patterns → automatic remediation
- Fix attempt tracking (what was tried, did it work?)
"""

import os
import json
import time
import traceback
import logging
import re
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable, Any
from datetime import datetime, timezone

logger = logging.getLogger("ErrorHandler")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ErrorCategory(str, Enum):
    WORKFLOW = "WORKFLOW"
    IMAGE_IO = "IMAGE_IO"
    DISCORD = "DISCORD"
    PARSING = "PARSING"
    CACHE = "CACHE"
    UNKNOWN = "UNKNOWN"


class ErrorSeverity(str, Enum):
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AutoFixAction(str, Enum):
    RETRY_REDUCED_RES = "RETRY_REDUCED_RES"
    FLUSH_AND_RETRY = "FLUSH_AND_RETRY"
    RECONNECT_AND_RETRY = "RECONNECT_AND_RETRY"
    RETRY_WITHOUT_LORA = "RETRY_WITHOUT_LORA"
    RETRY_SIMPLE = "RETRY_SIMPLE"
    LOG_AND_NOTIFY = "LOG_AND_NOTIFY"
    NONE = "NONE"


class AutoFixResult(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    SKIPPED = "SKIPPED"


# ---------------------------------------------------------------------------
# Error Entry
# ---------------------------------------------------------------------------

@dataclass
class ErrorEntry:
    """A single structured error log entry."""
    timestamp: str
    category: str
    severity: str
    source_function: str
    source_file: str
    error_type: str
    error_message: str
    context: dict = field(default_factory=dict)
    auto_fix_attempted: bool = False
    auto_fix_action: str = AutoFixAction.NONE.value
    auto_fix_result: str = AutoFixResult.NOT_ATTEMPTED.value
    auto_fix_detail: str = ""
    stack_trace: str = ""
    id: str = ""

    def __post_init__(self):
        if not self.id:
            ts_compact = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.id = f"err_{ts_compact}_{id(self) % 10000:04d}"

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Auto-Fix Recipe
# ---------------------------------------------------------------------------

@dataclass
class AutoFixRecipe:
    """Defines a pattern-matched auto-fix rule."""
    name: str
    category: ErrorCategory
    pattern: str  # regex pattern to match against error_message
    action: AutoFixAction
    description: str
    max_retries: int = 1
    _compiled: re.Pattern = field(default=None, repr=False, init=False)

    def __post_init__(self):
        self._compiled = re.compile(self.pattern, re.IGNORECASE)

    def matches(self, error_message: str) -> bool:
        return bool(self._compiled.search(error_message))


# ---------------------------------------------------------------------------
# Default Auto-Fix Recipes (Workflow Execution Category)
# ---------------------------------------------------------------------------

DEFAULT_RECIPES = [
    AutoFixRecipe(
        name="timeout_retry",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(timed?\s*out|timeout|generation timed out)",
        action=AutoFixAction.RETRY_REDUCED_RES,
        description="Retry at 75% resolution after timeout",
        max_retries=1,
    ),
    AutoFixRecipe(
        name="comfyui_500_flush",
        category=ErrorCategory.WORKFLOW,
        pattern=r"failed to queue prompt.*(?:HTTP 5\d{2}|500|502|503)",
        action=AutoFixAction.FLUSH_AND_RETRY,
        description="Flush ComfyUI queue and retry after HTTP 5xx",
        max_retries=1,
    ),
    AutoFixRecipe(
        name="websocket_reconnect",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(websocket|ws|connection\s*(reset|closed|refused|error))",
        action=AutoFixAction.RECONNECT_AND_RETRY,
        description="Force WebSocket reconnect and retry once",
        max_retries=1,
    ),
    AutoFixRecipe(
        name="vram_oom",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(cuda out of memory|out of memory|vram|oom|allocat)",
        action=AutoFixAction.RETRY_REDUCED_RES,
        description="Retry at 50% resolution after VRAM OOM",
        max_retries=1,
    ),
    AutoFixRecipe(
        name="checkpoint_not_found",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(checkpoint|ckpt|model).*not\s*found",
        action=AutoFixAction.LOG_AND_NOTIFY,
        description="Log with suggested fix; checkpoint file not found on disk",
    ),
    AutoFixRecipe(
        name="lora_not_found",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(lora|loraloader).*not\s*found",
        action=AutoFixAction.RETRY_WITHOUT_LORA,
        description="Strip the failing LoRA, retry, and warn user",
        max_retries=1,
    ),
    AutoFixRecipe(
        name="execution_error",
        category=ErrorCategory.WORKFLOW,
        pattern=r"comfyui execution error",
        action=AutoFixAction.LOG_AND_NOTIFY,
        description="ComfyUI node execution error; log details for IDE review",
    ),
    AutoFixRecipe(
        name="execution_interrupted",
        category=ErrorCategory.WORKFLOW,
        pattern=r"(execution.*interrupted|interrupted)",
        action=AutoFixAction.RETRY_SIMPLE,
        description="Retry once after ComfyUI execution was interrupted",
        max_retries=1,
    ),
]


# ---------------------------------------------------------------------------
# Error Handler (Singleton-like)
# ---------------------------------------------------------------------------

class ErrorHandler:
    """
    Central error handler: logs structured errors, matches auto-fix recipes,
    and tracks fix attempts.
    """

    def __init__(
        self,
        log_file: str = "error_log.json",
        max_entries: int = 500,
        recipes: list[AutoFixRecipe] | None = None,
    ):
        self.log_file = log_file
        self.max_entries = max_entries
        self.recipes = recipes or list(DEFAULT_RECIPES)
        self._entries: list[dict] = []
        self._retry_tracker: dict[str, int] = {}  # "recipe_name:context_key" -> attempt count
        self._load()

    # -- Persistence --

    def _load(self):
        """Load existing error log from disk."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, "r", encoding="utf-8") as f:
                    self._entries = json.load(f)
                logger.info(f"Loaded {len(self._entries)} error log entries from {self.log_file}")
            except Exception as e:
                logger.warning(f"Could not load error log: {e}")
                self._entries = []
        else:
            self._entries = []

    def _save(self):
        """Persist error log to disk with rolling limit."""
        # Trim to max_entries (keep most recent)
        if len(self._entries) > self.max_entries:
            self._entries = self._entries[-self.max_entries:]
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                json.dump(self._entries, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save error log: {e}")

    # -- Core Logging --

    def log_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        source_function: str,
        source_file: str = "bot.py",
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        context: dict | None = None,
    ) -> ErrorEntry:
        """
        Log a structured error entry. Returns the ErrorEntry for further use.
        """
        entry = ErrorEntry(
            timestamp=datetime.now(timezone.utc).isoformat(),
            category=category.value,
            severity=severity.value,
            source_function=source_function,
            source_file=source_file,
            error_type=type(exception).__name__,
            error_message=str(exception),
            context=context or {},
            stack_trace=traceback.format_exc(),
        )

        self._entries.append(entry.to_dict())
        self._save()

        # Also log through standard Python logger
        logger.error(
            f"[{entry.category}] {entry.source_function}: "
            f"{entry.error_type} - {entry.error_message}"
        )

        return entry

    def log_error_with_fix(
        self,
        entry: ErrorEntry,
        action: AutoFixAction,
        result: AutoFixResult,
        detail: str = "",
    ):
        """Update an existing error entry with auto-fix outcome."""
        entry.auto_fix_attempted = True
        entry.auto_fix_action = action.value
        entry.auto_fix_result = result.value
        entry.auto_fix_detail = detail

        # Update the last entry in the log (the one we just added)
        for i in range(len(self._entries) - 1, -1, -1):
            if self._entries[i].get("id") == entry.id:
                self._entries[i] = entry.to_dict()
                break
        self._save()

    # -- Auto-Fix Matching --

    def find_recipe(self, error_message: str, category: ErrorCategory | None = None) -> AutoFixRecipe | None:
        """Find the first matching auto-fix recipe for an error message."""
        for recipe in self.recipes:
            if category and recipe.category != category:
                continue
            if recipe.matches(error_message):
                return recipe
        return None

    def can_retry(self, recipe: AutoFixRecipe, context_key: str) -> bool:
        """Check if we haven't exceeded max retries for this recipe+context."""
        tracker_key = f"{recipe.name}:{context_key}"
        attempts = self._retry_tracker.get(tracker_key, 0)
        return attempts < recipe.max_retries

    def record_retry(self, recipe: AutoFixRecipe, context_key: str):
        """Record a retry attempt for tracking purposes."""
        tracker_key = f"{recipe.name}:{context_key}"
        self._retry_tracker[tracker_key] = self._retry_tracker.get(tracker_key, 0) + 1

    def clear_retry_tracker(self, context_key: str = None):
        """Clear retry tracking. If context_key given, only clear that key."""
        if context_key:
            keys_to_clear = [k for k in self._retry_tracker if k.endswith(f":{context_key}")]
            for k in keys_to_clear:
                del self._retry_tracker[k]
        else:
            self._retry_tracker.clear()

    # -- Query / Stats --

    def get_recent_errors(self, count: int = 20, category: str | None = None) -> list[dict]:
        """Get the most recent error entries, optionally filtered by category."""
        filtered = self._entries
        if category:
            filtered = [e for e in filtered if e.get("category") == category]
        return filtered[-count:]

    def get_stats(self, hours: int = 24) -> dict:
        """Get error statistics for the last N hours."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        recent = []
        for entry in self._entries:
            try:
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts >= cutoff:
                    recent.append(entry)
            except (KeyError, ValueError):
                continue

        stats = {
            "total": len(recent),
            "by_category": {},
            "by_severity": {},
            "auto_fixed": 0,
            "needs_attention": 0,
        }

        for entry in recent:
            cat = entry.get("category", "UNKNOWN")
            sev = entry.get("severity", "UNKNOWN")
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1

            if entry.get("auto_fix_result") == AutoFixResult.SUCCESS.value:
                stats["auto_fixed"] += 1
            elif entry.get("auto_fix_result") in (
                AutoFixResult.FAILED.value,
                AutoFixResult.NOT_ATTEMPTED.value,
            ):
                if entry.get("severity") in (ErrorSeverity.ERROR.value, ErrorSeverity.CRITICAL.value):
                    stats["needs_attention"] += 1

        return stats

    def get_needs_attention(self, hours: int = 24) -> list[dict]:
        """Get errors that were NOT auto-fixed and need human attention."""
        cutoff = datetime.now(timezone.utc).timestamp() - (hours * 3600)
        results = []
        for entry in self._entries:
            try:
                ts = datetime.fromisoformat(entry["timestamp"]).timestamp()
                if ts < cutoff:
                    continue
            except (KeyError, ValueError):
                continue

            fix_result = entry.get("auto_fix_result", AutoFixResult.NOT_ATTEMPTED.value)
            if fix_result in (AutoFixResult.FAILED.value, AutoFixResult.NOT_ATTEMPTED.value):
                if entry.get("severity") in (ErrorSeverity.ERROR.value, ErrorSeverity.CRITICAL.value):
                    results.append(entry)

        return results


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

error_handler = ErrorHandler()

import sqlite3
import json
import os
import logging

logger = logging.getLogger("DiscordBot.DB")
DB_FILE = "cache.db"

_save_counter = 0

def get_db_connection():
    """Returns a SQLite connection configured with WAL mode, normal synchrony, and optimized cache."""
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA cache_size=-64000;")  # 64MB cache
    except Exception as e:
        logger.debug(f"Error setting SQLite PRAGMAs: {e}")
    return conn

def init_db():
    """Initialize SQLite database schema and indexes."""
    try:
        with get_db_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generations (
                    id TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS favorite_styles (
                    user_id INTEGER,
                    style_code INTEGER,
                    style_name TEXT,
                    style_prompt TEXT,
                    timestamp REAL DEFAULT (julianday('now')),
                    PRIMARY KEY (user_id, style_code)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS favorite_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    prompt_name TEXT,
                    prompt_text TEXT,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS negative_prompts (
                    user_id INTEGER PRIMARY KEY,
                    negative_text TEXT,
                    timestamp REAL DEFAULT (julianday('now'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS generation_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT,
                    duration_seconds REAL,
                    init_seconds REAL DEFAULT 0.0,
                    sampling_seconds REAL DEFAULT 0.0,
                    post_seconds REAL DEFAULT 0.0,
                    model_name TEXT,
                    steps INTEGER,
                    resolution TEXT,
                    status TEXT,
                    error_message TEXT,
                    user_id INTEGER,
                    metadata TEXT,
                    timestamp REAL DEFAULT (julianday('now')),
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_registry (
                    filename TEXT PRIMARY KEY,
                    model_type TEXT,
                    base_architecture TEXT,
                    sub_type TEXT,
                    display_name TEXT,
                    trigger_words TEXT DEFAULT '',
                    default_strength REAL DEFAULT 1.0,
                    metadata TEXT DEFAULT '{}',
                    timestamp REAL DEFAULT (julianday('now')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS live_status (
                    id INTEGER PRIMARY KEY,
                    step INTEGER DEFAULT 0,
                    max_steps INTEGER DEFAULT 0,
                    node_id TEXT,
                    stage TEXT,
                    prompt_id TEXT,
                    prompt_text TEXT,
                    updated_at REAL DEFAULT (julianday('now'))
                )
            """)
            try:
                conn.execute("ALTER TABLE generation_metrics ADD COLUMN init_seconds REAL DEFAULT 0.0")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE generation_metrics ADD COLUMN sampling_seconds REAL DEFAULT 0.0")
            except Exception:
                pass
            try:
                conn.execute("ALTER TABLE generation_metrics ADD COLUMN post_seconds REAL DEFAULT 0.0")
            except Exception:
                pass

            # Performance indexes
            try:
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_timestamp ON generations(timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_generations_user_id ON generations(json_extract(data, '$.user_id'))")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_favorite_styles_user ON favorite_styles(user_id, timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_favorite_prompts_user ON favorite_prompts(user_id, timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_command_status ON generation_metrics(command, status, timestamp DESC)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_metrics_user_id ON generation_metrics(user_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_model_registry_arch ON model_registry(base_architecture, model_type)")
            except Exception as e:
                logger.debug(f"Index creation note: {e}")

            conn.commit()
        seed_default_model_registry()
        logger.info("SQLite database initialized and model registry seeded successfully.")
    except Exception as e:
        logger.error(f"Error initializing SQLite database: {e}")

def get_generation(generation_id: str) -> dict:
    """Fetch generation data dictionary by ID from SQLite."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT data FROM generations WHERE id = ?", (generation_id,))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
    except Exception as e:
        logger.error(f"Error fetching generation {generation_id} from SQLite: {e}")
    return None

def save_generation(generation_id: str, data: dict):
    """Insert or replace generation data and periodically prune older entries."""
    global _save_counter
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO generations (id, data) VALUES (?, ?)",
                (generation_id, json.dumps(data))
            )
            conn.commit()
        
        # Prune older entries periodically rather than on every single write
        _save_counter += 1
        if _save_counter % 25 == 0:
            prune_cache()
    except Exception as e:
        logger.error(f"Error saving generation {generation_id} to SQLite: {e}")

def prune_cache(limit=2000):
    """Keep the last N generations and remove files for deleted entries."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Find IDs that are older and exceed the limit
            cursor.execute(
                "SELECT id FROM generations WHERE id NOT IN (SELECT id FROM generations ORDER BY timestamp DESC LIMIT ?)",
                (limit,)
            )
            ids_to_delete = [row[0] for row in cursor.fetchall()]
            
            if ids_to_delete:
                # Delete database records
                cursor.execute(
                    "DELETE FROM generations WHERE id NOT IN (SELECT id FROM generations ORDER BY timestamp DESC LIMIT ?)",
                    (limit,)
                )
                conn.commit()
                
                # Delete quadrant cache files for pruned entries
                quadrant_dir = os.getenv("QUADRANT_CACHE_DIR", r"C:\ComfyUI\ComfyUI\output\Discord Bot\scratch")
                if os.path.exists(quadrant_dir):
                    for k in ids_to_delete:
                        for idx in range(1, 5):
                            path = os.path.join(quadrant_dir, f"{k}_{idx}.png")
                            if os.path.exists(path):
                                try:
                                    os.remove(path)
                                except Exception:
                                    pass
                logger.info(f"Pruned {len(ids_to_delete)} generations from database and cleared associated files.")
                vacuum_database()
    except Exception as e:
        logger.error(f"Error pruning SQLite cache: {e}")

def vacuum_database() -> bool:
    """Executes SQLite VACUUM to reclaim unused disk space and optimize pages."""
    try:
        with get_db_connection() as conn:
            conn.execute("VACUUM")
        logger.info("SQLite database successfully vacuumed and compacted.")
        return True
    except Exception as e:
        logger.error(f"Error vacuuming SQLite database: {e}")
        return False

def cleanup_orphaned_quadrants():
    """Removes any quadrant cache files that are no longer in the database."""
    try:
        quadrant_dir = os.getenv("QUADRANT_CACHE_DIR", r"C:\ComfyUI\ComfyUI\output\Discord Bot\scratch")
        if not os.path.exists(quadrant_dir):
            return
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM generations")
            valid_prefixes = set(row[0] for row in cursor.fetchall())
            
        cleaned_count = 0
        for filename in os.listdir(quadrant_dir):
            parts = filename.split("_")
            if parts:
                gen_id = parts[0]
                if gen_id not in valid_prefixes:
                    path = os.path.join(quadrant_dir, filename)
                    try:
                        os.remove(path)
                        cleaned_count += 1
                    except Exception:
                        pass
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} orphaned cached quadrant images.")
    except Exception as e:
        logger.error(f"Error during quadrant cleanup sweep: {e}")

def get_user_generations(user_id: int) -> list[dict]:
    """Fetch all generations matching the user_id efficiently using indexed JSON extraction."""
    res = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Fast indexed JSON filter
            cursor.execute(
                "SELECT id, data FROM generations WHERE json_extract(data, '$.user_id') = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[1])
                    data["id"] = row[0]
                    res.append(data)
                except Exception:
                    pass
    except Exception as e:
        logger.error(f"Error fetching user generations: {e}")
    return res

def add_favorite_style(user_id: int, style_code: int, style_name: str, style_prompt: str):
    """Save a style code and its details to the user's favorites."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO favorite_styles (user_id, style_code, style_name, style_prompt) VALUES (?, ?, ?, ?)",
                (user_id, style_code, style_name, style_prompt)
            )
            conn.commit()
        logger.info(f"Added favorite style {style_code} for user {user_id}")
    except Exception as e:
        logger.error(f"Error adding favorite style {style_code} for user {user_id}: {e}")

def remove_favorite_style(user_id: int, style_code: int):
    """Remove a style code from the user's favorites."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM favorite_styles WHERE user_id = ? AND style_code = ?",
                (user_id, style_code)
            )
            conn.commit()
        logger.info(f"Removed favorite style {style_code} for user {user_id}")
    except Exception as e:
        logger.error(f"Error removing favorite style {style_code} for user {user_id}: {e}")

def update_favorite_style(user_id: int, style_code: int, new_name: str, new_prompt: str) -> bool:
    """Update a favorite style's name and prompt."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE favorite_styles SET style_name = ?, style_prompt = ? WHERE user_id = ? AND style_code = ?",
                (new_name, new_prompt, user_id, style_code)
            )
            conn.commit()
            updated = cursor.rowcount > 0
        logger.info(f"Updated favorite style {style_code} for user {user_id} (success={updated})")
        return updated
    except Exception as e:
        logger.error(f"Error updating favorite style {style_code} for user {user_id}: {e}")
        return False

def get_favorite_styles(user_id: int) -> list[dict]:
    """Fetch all favorite styles for the user, ordered by timestamp descending."""
    res = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT style_code, style_name, style_prompt FROM favorite_styles WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            for row in cursor.fetchall():
                res.append({
                    "style_code": row[0],
                    "style_name": row[1],
                    "style_prompt": row[2]
                })
    except Exception as e:
        logger.error(f"Error fetching favorite styles for user {user_id}: {e}")
    return res

def add_favorite_prompt(user_id: int, prompt_name: str, prompt_text: str):
    """Save a prompt and its alias/name to the user's favorite prompts."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT INTO favorite_prompts (user_id, prompt_name, prompt_text) VALUES (?, ?, ?)",
                (user_id, prompt_name, prompt_text)
            )
            conn.commit()
        logger.info(f"Added favorite prompt '{prompt_name}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error adding favorite prompt for user {user_id}: {e}")

def remove_favorite_prompt(user_id: int, prompt_id: int):
    """Remove a favorite prompt by ID for a user."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "DELETE FROM favorite_prompts WHERE id = ? AND user_id = ?",
                (prompt_id, user_id)
            )
            conn.commit()
        logger.info(f"Removed favorite prompt ID {prompt_id} for user {user_id}")
    except Exception as e:
        logger.error(f"Error removing favorite prompt ID {prompt_id} for user {user_id}: {e}")

def update_favorite_prompt(user_id: int, prompt_id: int, new_name: str, new_text: str) -> bool:
    """Update a favorite prompt's name and text by ID."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE favorite_prompts SET prompt_name = ?, prompt_text = ? WHERE id = ? AND user_id = ?",
                (new_name, new_text, prompt_id, user_id)
            )
            conn.commit()
            updated = cursor.rowcount > 0
        logger.info(f"Updated favorite prompt ID {prompt_id} for user {user_id} (success={updated})")
        return updated
    except Exception as e:
        logger.error(f"Error updating favorite prompt ID {prompt_id} for user {user_id}: {e}")
        return False

def get_favorite_prompts(user_id: int) -> list[dict]:
    """Fetch all favorite prompts for a user, ordered by timestamp descending."""
    res = []
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, prompt_name, prompt_text FROM favorite_prompts WHERE user_id = ? ORDER BY timestamp DESC",
                (user_id,)
            )
            for row in cursor.fetchall():
                res.append({
                    "id": row[0],
                    "prompt_name": row[1],
                    "prompt_text": row[2]
                })
    except Exception as e:
        logger.error(f"Error fetching favorite prompts for user {user_id}: {e}")
    return res

DEFAULT_NEGATIVE_PROMPT = "blurry, deformed, poorly drawn, bad quality, distorted, extra limbs, bad anatomy, text, watermarks"

def get_negative_prompt(user_id: int) -> str:
    """Fetch the custom negative prompt for a user (or global fallback user_id=0), defaulting to DEFAULT_NEGATIVE_PROMPT."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT negative_text FROM negative_prompts WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            # Try global fallback user_id = 0
            cursor.execute("SELECT negative_text FROM negative_prompts WHERE user_id = 0")
            row0 = cursor.fetchone()
            if row0 and row0[0]:
                return row0[0]
    except Exception as e:
        logger.error(f"Error fetching negative prompt for user {user_id}: {e}")
    return DEFAULT_NEGATIVE_PROMPT

def set_negative_prompt(user_id: int, negative_text: str):
    """Save custom negative prompt for a user (or global user_id=0)."""
    try:
        with get_db_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO negative_prompts (user_id, negative_text) VALUES (?, ?)",
                (user_id, negative_text.strip())
            )
            conn.commit()
        logger.info(f"Updated negative prompt for user {user_id}")
    except Exception as e:
        logger.error(f"Error setting negative prompt for user {user_id}: {e}")

def reset_negative_prompt(user_id: int):
    """Reset custom negative prompt for a user back to default."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM negative_prompts WHERE user_id = ?", (user_id,))
            conn.commit()
        logger.info(f"Reset negative prompt for user {user_id}")
    except Exception as e:
        logger.error(f"Error resetting negative prompt for user {user_id}: {e}")

def record_generation_metric(
    command: str,
    duration_seconds: float,
    model_name: str = "",
    steps: int = 0,
    resolution: str = "",
    status: str = "success",
    error_message: str = "",
    user_id: int = None,
    init_seconds: float = 0.0,
    sampling_seconds: float = 0.0,
    post_seconds: float = 0.0,
    metadata: dict = None
):
    """Records performance and diagnostic metrics for each generation run."""
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO generation_metrics 
                (command, duration_seconds, init_seconds, sampling_seconds, post_seconds, model_name, steps, resolution, status, error_message, user_id, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                command,
                round(duration_seconds, 2),
                round(init_seconds, 2),
                round(sampling_seconds, 2),
                round(post_seconds, 2),
                os.path.basename(model_name) if model_name else "",
                steps,
                resolution,
                status,
                error_message,
                user_id,
                json.dumps(metadata or {})
            ))
            conn.commit()
    except Exception as e:
        logger.error(f"Error logging generation metric: {e}")

def get_recent_metrics(limit: int = 20, command: str = None) -> list:
    """Fetches recent generation telemetry records for troubleshooting."""
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if command:
                cursor.execute("""
                    SELECT * FROM generation_metrics 
                    WHERE command = ? 
                    ORDER BY id DESC LIMIT ?
                """, (command, limit))
            else:
                cursor.execute("""
                    SELECT * FROM generation_metrics 
                    ORDER BY id DESC LIMIT ?
                """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
    except Exception as e:
        logger.error(f"Error fetching generation metrics: {e}")
        return []

def get_performance_summary() -> dict:
    """Calculates average render times, phase breakdowns, and failure rates per command/model."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT 
                    command,
                    COUNT(*) as total_runs,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successes,
                    SUM(CASE WHEN status != 'success' THEN 1 ELSE 0 END) as failures,
                    AVG(CASE WHEN status = 'success' THEN duration_seconds ELSE NULL END) as avg_duration,
                    AVG(CASE WHEN status = 'success' THEN init_seconds ELSE NULL END) as avg_init,
                    AVG(CASE WHEN status = 'success' THEN sampling_seconds ELSE NULL END) as avg_sampling,
                    AVG(CASE WHEN status = 'success' THEN post_seconds ELSE NULL END) as avg_post,
                    MIN(CASE WHEN status = 'success' THEN duration_seconds ELSE NULL END) as min_duration,
                    MAX(CASE WHEN status = 'success' THEN duration_seconds ELSE NULL END) as max_duration
                FROM generation_metrics
                GROUP BY command
            """)
            rows = cursor.fetchall()
            summary = {}
            for row in rows:
                cmd, total, succ, fail, avg_dur, avg_init, avg_samp, avg_post, min_dur, max_dur = row
                summary[cmd] = {
                    "total_runs": total,
                    "successes": succ or 0,
                    "failures": fail or 0,
                    "avg_duration": round(avg_dur, 2) if avg_dur else 0,
                    "avg_init": round(avg_init, 2) if avg_init else 0,
                    "avg_sampling": round(avg_samp, 2) if avg_samp else 0,
                    "avg_post": round(avg_post, 2) if avg_post else 0,
                    "min_duration": round(min_dur, 2) if min_dur else 0,
                    "max_duration": round(max_dur, 2) if max_dur else 0
                }
            return summary
    except Exception as e:
        logger.error(f"Error calculating performance summary: {e}")
        return {}

# =========================================================================
# Model & LoRA Architecture Registry Operations
# =========================================================================

def upsert_model_registry(
    filename: str,
    model_type: str,
    base_architecture: str,
    sub_type: str = "standard",
    display_name: str = "",
    trigger_words: str = "",
    default_strength: float = 1.0,
    metadata: dict = None
) -> bool:
    """Upsert a model or LoRA architecture definition into SQLite."""
    try:
        meta_json = json.dumps(metadata or {})
        clean_fn = os.path.basename(filename)
        with get_db_connection() as conn:
            conn.execute("""
                INSERT INTO model_registry (
                    filename, model_type, base_architecture, sub_type, 
                    display_name, trigger_words, default_strength, metadata, 
                    timestamp, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, julianday('now'), datetime('now', 'localtime'))
                ON CONFLICT(filename) DO UPDATE SET
                    model_type = excluded.model_type,
                    base_architecture = excluded.base_architecture,
                    sub_type = excluded.sub_type,
                    display_name = excluded.display_name,
                    trigger_words = excluded.trigger_words,
                    default_strength = excluded.default_strength,
                    metadata = excluded.metadata,
                    timestamp = julianday('now'),
                    updated_at = datetime('now', 'localtime')
            """, (clean_fn, model_type, base_architecture, sub_type, display_name or clean_fn, trigger_words, default_strength, meta_json))
            conn.commit()
            return True
    except Exception as e:
        logger.error(f"Error upserting model registry for {filename}: {e}")
        return False

def get_model_registry_entry(filename: str) -> dict:
    """Retrieves metadata and architecture for a specific model or LoRA."""
    try:
        clean_fn = os.path.basename(filename)
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM model_registry WHERE filename = ?", (clean_fn,))
            row = cursor.fetchone()
            if row:
                d = dict(row)
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                return d
    except Exception as e:
        logger.error(f"Error retrieving model registry entry for {filename}: {e}")
    return None

def get_models_by_architecture(base_architecture: str = None, model_type: str = None) -> list:
    """Queries registered models filtered by architecture and/or model type."""
    try:
        with get_db_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if base_architecture and model_type:
                cursor.execute("""
                    SELECT * FROM model_registry 
                    WHERE base_architecture = ? AND model_type = ?
                    ORDER BY display_name ASC
                """, (base_architecture, model_type))
            elif base_architecture:
                cursor.execute("""
                    SELECT * FROM model_registry 
                    WHERE base_architecture = ?
                    ORDER BY display_name ASC
                """, (base_architecture,))
            elif model_type:
                cursor.execute("""
                    SELECT * FROM model_registry 
                    WHERE model_type = ?
                    ORDER BY display_name ASC
                """, (model_type,))
            else:
                cursor.execute("SELECT * FROM model_registry ORDER BY display_name ASC")
            
            results = []
            for row in cursor.fetchall():
                d = dict(row)
                d["metadata"] = json.loads(d.get("metadata") or "{}")
                results.append(d)
            return results
    except Exception as e:
        logger.error(f"Error querying models by architecture: {e}")
        return []

def list_all_registered_models() -> list:
    """Returns all registered checkpoints, LoRAs, and models."""
    return get_models_by_architecture()

def seed_default_model_registry():
    """Initializes the database registry with known models and LoRAs."""
    # 1. SDXL Checkpoints
    sdxl_ckpts = [
        ("waiIllustriousSDXL_v170.safetensors", "checkpoint", "sdxl", "illustrious", "Wai Illustrious SDXL v1.70"),
        ("RealVisXL_V4.0.safetensors", "checkpoint", "sdxl", "realistic", "RealVisXL V4.0"),
        ("juggernautXL_ragnarok.safetensors", "checkpoint", "sdxl", "realistic", "Juggernaut XL"),
        ("CopaxTimeLessXL.safetensors", "checkpoint", "sdxl", "realistic", "Copax Timeless XL"),
        ("ultraRealisticByStable_v25.safetensors", "checkpoint", "sdxl", "realistic", "Ultra Realistic XL v2.5"),
        ("hyphoriaRealIllu_v09.safetensors", "checkpoint", "sdxl", "illustrious", "Hyphoria Real Illu v0.9"),
        ("hyphoriaIlluNAI_v001.safetensors", "checkpoint", "sdxl", "illustrious", "Hyphoria NAI"),
        ("illustriousRealismBy_v10VAE.safetensors", "checkpoint", "sdxl", "illustrious", "Illustrious Realism v1.0"),
        ("bigLust_v16.safetensors", "checkpoint", "sdxl", "realistic", "Big Lust v1.6"),
        ("lustifySDXLNSFWSFW_v10.safetensors", "checkpoint", "sdxl", "realistic", "Lustify v1.0"),
        ("ponyDiffusionV6XL_v6StartWithThisOne.safetensors", "checkpoint", "sdxl", "pony", "Pony Diffusion V6 XL"),
        ("RealVisXL_V5.0_Lightning_fp16.safetensors", "checkpoint", "sdxl", "realistic", "RealVisXL V5.0 Lightning"),
        ("novaFurryXL_ilV180A.safetensors", "checkpoint", "sdxl", "illustrious", "Nova Furry XL"),
    ]
    for fn, mtype, arch, subtype, dname in sdxl_ckpts:
        upsert_model_registry(fn, mtype, arch, subtype, display_name=dname)

    # 2. Known LoRAs
    loras = [
        ("Semi-realism_illustrious.safetensors", "lora", "sdxl", "illustrious", "Semi-Realism Illustrious", "semi-realism", 0.70),
        ("ogarla_epoch_5.safetensors", "lora", "sdxl", "standard", "Ogarla (SDXL Main)", "ogarla", 0.70),
        ("ogarla_epoch_6.safetensors", "lora", "sdxl", "standard", "Ogarla v6 (SDXL)", "ogarla", 0.70),
        ("ogarlapony_epoch_6.safetensors", "lora", "sdxl", "pony", "Ogarla Pony (SDXL)", "ogarlapony, score_9, score_8_up", 0.75),
        ("ogarlaflux_epoch_1.safetensors", "lora", "flux", "standard", "Ogarla Flux v1", "ogarlaflux", 0.80),
        ("ogarlaflux_epoch_5.safetensors", "lora", "flux", "standard", "Ogarla Flux v5", "ogarlaflux", 0.80),
        ("pworship_high_noise.safetensors", "lora", "wan", "high_noise", "Pussy Worship (Wan High)", "", 0.70),
        ("pworship_low_noise.safetensors", "lora", "wan", "low_noise", "Pussy Worship (Wan Low)", "", 0.70),
        ("WAN-2.2-I2V-Handjob-HIGH-v1.safetensors", "lora", "wan", "high_noise", "Wan 2.2 Handjob High", "", 0.95),
        ("WAN-2.2-I2V-Handjob-LOW-v1.safetensors", "lora", "wan", "low_noise", "Wan 2.2 Handjob Low", "", 0.85),
    ]
    for fn, mtype, arch, subtype, dname, triggers, strength in loras:
        upsert_model_registry(fn, mtype, arch, subtype, display_name=dname, trigger_words=triggers, default_strength=strength)

    # 3. Video / Other Checkpoints
    video_models = [
        ("ltx-video-2b-v0.9.1.safetensors", "checkpoint", "ltx", "standard", "LTX-Video 2B"),
        ("dasiwaWAN22I2V14B_midnightflirtHigh-Q3_K_M.gguf", "unet", "wan", "high_noise", "Wan 2.2 High Noise (GGUF)"),
        ("dasiwaWAN22I2V14B_midnightflirtLow-Q3_K_M.gguf", "unet", "wan", "low_noise", "Wan 2.2 Low Noise (GGUF)"),
    ]
    for fn, mtype, arch, subtype, dname in video_models:
        upsert_model_registry(fn, mtype, arch, subtype, display_name=dname)


def set_live_status(step: int, max_steps: int, node_id: str = None, stage: str = None, prompt_id: str = None, prompt_text: str = None):
    """Store active generation step progress and pipeline stage for cross-process live telemetry."""
    try:
        with get_db_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO live_status (id, step, max_steps, node_id, stage, prompt_id, prompt_text, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, julianday('now'))
            """, (step, max_steps, str(node_id) if node_id is not None else None, stage, prompt_id, prompt_text))
            conn.commit()
    except Exception as e:
        logger.debug(f"Error setting live status: {e}")


def get_live_status() -> dict:
    """Retrieve active generation step progress and pipeline stage from SQLite."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT step, max_steps, node_id, stage, prompt_id, prompt_text, (julianday('now') - updated_at) * 86400 FROM live_status WHERE id = 1")
            row = cursor.fetchone()
            if row:
                step, max_steps, node_id, stage, prompt_id, prompt_text, age_sec = row
                if age_sec is not None and age_sec < 180:
                    return {
                        "step": step or 0,
                        "max_steps": max_steps or 0,
                        "node_id": node_id,
                        "stage": stage or "",
                        "prompt_id": prompt_id,
                        "prompt_text": prompt_text or "",
                        "age_sec": round(age_sec, 1)
                    }
    except Exception as e:
        logger.debug(f"Error fetching live status: {e}")
    return None


def clear_live_status():
    """Clear active generation live progress record."""
    try:
        with get_db_connection() as conn:
            conn.execute("DELETE FROM live_status WHERE id = 1")
            conn.commit()
    except Exception as e:
        logger.debug(f"Error clearing live status: {e}")


"""
Crash Recovery & Job Reconciliation Service.

Monitors SQLite-journaled pending jobs across bot or system restarts.
Queries ComfyUI's /history endpoint on startup, retrieves rendered images/videos,
and automatically edits or delivers completed outputs to Discord channels.
"""

import asyncio
import io
import logging
import os
from typing import Optional, List, Dict, Any

import discord
import db
from error_handler import error_handler, ErrorCategory, ErrorSeverity

logger = logging.getLogger("DiscordBot.RecoveryService")


async def reconcile_pending_jobs(bot: discord.Client, comfy_client: Any) -> Dict[str, int]:
    """
    Scans SQLite pending_jobs table for unfinished in-flight generations.
    Reconciles each against ComfyUI execution history and updates Discord messages.
    """
    stats = {
        "found": 0,
        "recovered": 0,
        "failed": 0,
        "interrupted": 0,
        "stale_cleaned": 0
    }

    # 1. Clean up jobs older than 24 hours
    try:
        stats["stale_cleaned"] = db.cleanup_stale_jobs(24.0)
        if stats["stale_cleaned"] > 0:
            logger.info(f"Purged {stats['stale_cleaned']} stale pending job(s) older than 24 hours.")
    except Exception as clean_err:
        logger.debug(f"Stale cleanup note: {clean_err}")

    # 2. Fetch all currently active / running jobs
    pending = db.get_pending_jobs(status="running")
    stats["found"] = len(pending)

    if not pending:
        logger.info("No interrupted jobs found in SQLite journal. Recovery clean.")
        return stats

    logger.info(f"🔄 Found {len(pending)} pending in-flight job(s). Starting crash recovery reconciliation...")

    # 3. Check if ComfyUI is online
    is_online = False
    for check_attempt in range(5):
        try:
            if await comfy_client.is_online():
                is_online = True
                break
        except Exception:
            pass
        await asyncio.sleep(2)

    if not is_online:
        logger.warning("ComfyUI is currently offline. Skipping immediate recovery; jobs remain journaled.")
        return stats

    # 4. Reconcile each job
    for job in pending:
        prompt_id = job.get("prompt_id")
        channel_id = job.get("channel_id")
        message_id = job.get("message_id")
        user_id = job.get("user_id")
        command_type = job.get("command_type", "imagine")
        meta = job.get("metadata", {})

        if not prompt_id:
            continue

        try:
            # Query ComfyUI history
            outputs = None
            try:
                outputs = await comfy_client.get_history_output(prompt_id)
            except Exception as hist_err:
                if "ComfyUI execution error" in str(hist_err):
                    # Execution failed inside ComfyUI
                    logger.warning(f"Pending job {prompt_id} failed during execution: {hist_err}")
                    db.complete_pending_job(prompt_id, status="failed")
                    stats["failed"] += 1
                    await _notify_discord_job_status(
                        bot, channel_id, message_id, user_id,
                        status="failed",
                        title=f"⚠️ {command_type.capitalize()} Execution Failed",
                        description=f"Job was interrupted or encountered an error in ComfyUI:\n`{hist_err}`"
                    )
                    continue
                else:
                    logger.debug(f"History query error for {prompt_id}: {hist_err}")

            if outputs is not None:
                # ComfyUI finished rendering this job!
                output_bytes_list = []
                for node_id, node_out in outputs.items():
                    if isinstance(node_out, dict):
                        for media_key in ["images", "videos", "gifs"]:
                            if media_key in node_out:
                                for item in node_out[media_key]:
                                    try:
                                        data = await comfy_client.get_image(
                                            filename=item.get("filename"),
                                            subfolder=item.get("subfolder", ""),
                                            img_type=item.get("type", "output")
                                        )
                                        if data:
                                            ext = os.path.splitext(item.get("filename", ".png"))[1] or ".png"
                                            output_bytes_list.append((data, ext))
                                    except Exception as img_err:
                                        logger.warning(f"Could not retrieve media file {item}: {img_err}")

                if output_bytes_list:
                    # Deliver recovered media to Discord
                    delivered = await _deliver_recovered_media(
                        bot=bot,
                        channel_id=channel_id,
                        message_id=message_id,
                        user_id=user_id,
                        command_type=command_type,
                        media_items=output_bytes_list,
                        metadata=meta
                    )
                    if delivered:
                        db.complete_pending_job(prompt_id, status="recovered")
                        stats["recovered"] += 1
                        logger.info(f"✅ Successfully recovered and delivered job {prompt_id} ({command_type}) to Discord.")
                    else:
                        db.complete_pending_job(prompt_id, status="recovered_offline")
                        stats["recovered"] += 1
                else:
                    # Outputs were empty or not media
                    db.complete_pending_job(prompt_id, status="completed")
                    stats["recovered"] += 1
            else:
                # Not in history — check if prompt is in ComfyUI queue
                in_queue = await _is_prompt_in_comfy_queue(comfy_client, prompt_id)
                if in_queue:
                    logger.info(f"Job {prompt_id} is still actively executing in ComfyUI queue. Leaving as running.")
                else:
                    # Not in history and not in queue — job was dropped when bot/system crashed
                    logger.info(f"Job {prompt_id} was dropped before completion. Marking as interrupted.")
                    db.complete_pending_job(prompt_id, status="interrupted")
                    stats["interrupted"] += 1
                    await _notify_discord_job_status(
                        bot, channel_id, message_id, user_id,
                        status="interrupted",
                        title=f"⚠️ {command_type.capitalize()} Interrupted by Restart",
                        description="The system or bot restarted while this generation was in progress. Please run your prompt again."
                    )

        except Exception as job_err:
            error_handler.log_error(
                job_err,
                category=ErrorCategory.WORKFLOW,
                source_function="reconcile_pending_jobs",
                source_file="recovery_service.py",
                severity=ErrorSeverity.WARNING,
                context={"prompt_id": prompt_id, "channel_id": channel_id, "message_id": message_id}
            )
            logger.warning(f"Error during job reconciliation for {prompt_id}: {job_err}")

    logger.info(
        f"🏁 Crash recovery finished: {stats['recovered']} recovered, {stats['failed']} failed, "
        f"{stats['interrupted']} interrupted."
    )
    return stats


async def _is_prompt_in_comfy_queue(comfy_client: Any, prompt_id: str) -> bool:
    """Checks whether prompt_id is present in ComfyUI's pending or running queue."""
    if not hasattr(comfy_client, "session") or not comfy_client.session:
        return False
    try:
        url = f"http://{comfy_client.server_address}/queue"
        async with comfy_client.session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                running = data.get("queue_running", [])
                pending = data.get("queue_pending", [])
                for item in running + pending:
                    # Item structure in ComfyUI: [order, prompt_id, prompt_dict, extra_data, ...]
                    if len(item) > 1 and item[1] == prompt_id:
                        return True
    except Exception:
        pass
    return False


async def _deliver_recovered_media(
    bot: discord.Client,
    channel_id: Optional[int],
    message_id: Optional[int],
    user_id: Optional[int],
    command_type: str,
    media_items: List[tuple],
    metadata: Dict[str, Any]
) -> bool:
    """Edits original status message or sends recovered attachments to the Discord channel."""
    if not channel_id:
        logger.debug("No channel_id associated with pending job; skipping Discord delivery.")
        return False

    channel = bot.get_channel(channel_id)
    if channel is None and hasattr(bot, "fetch_channel"):
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception:
            channel = None

    if channel is None:
        logger.warning(f"Could not resolve channel {channel_id} for recovery delivery.")
        return False

    files = []
    for idx, (raw_bytes, ext) in enumerate(media_items):
        filename = f"recovered_{command_type}_{idx + 1}{ext}"
        files.append(discord.File(fp=io.BytesIO(raw_bytes), filename=filename))

    embed = discord.Embed(
        title=f"🔄 Recovered {command_type.capitalize()} Output",
        description="Your generation completed while the bot was reconnecting. Output has been safely preserved!",
        color=discord.Color.green()
    )
    prompt_text = metadata.get("prompt")
    if prompt_text:
        embed.add_field(name="Prompt", value=prompt_text[:250], inline=False)
    if user_id:
        embed.add_field(name="Requested By", value=f"<@{user_id}>", inline=True)
    embed.set_footer(text="Shallot-CUI Bot • Crash Recovery & SQLite Journaling")

    # Attempt to edit original message if message_id exists
    message = None
    if message_id:
        try:
            message = await channel.fetch_message(message_id)
        except (discord.NotFound, discord.Forbidden):
            message = None
        except Exception as e:
            logger.debug(f"Could not fetch message {message_id}: {e}")

    content = f"<@{user_id}> 🔄 **Your generation was recovered:**" if user_id else "🔄 **Generation Recovered:**"

    if message:
        try:
            # First try updating message with attachments
            await message.edit(content=content, embed=embed, attachments=files)
            return True
        except Exception as edit_err:
            logger.debug(f"Message edit with attachments failed ({edit_err}); falling back to new post.")
            try:
                await message.edit(content="🔄 **Generation Recovered** (see below):", embed=None)
            except Exception:
                pass

    # Send as new message to channel
    try:
        await channel.send(content=content, embed=embed, files=files)
        return True
    except Exception as send_err:
        logger.error(f"Failed to deliver recovered media to channel {channel_id}: {send_err}")
        return False


async def _notify_discord_job_status(
    bot: discord.Client,
    channel_id: Optional[int],
    message_id: Optional[int],
    user_id: Optional[int],
    status: str,
    title: str,
    description: str
):
    """Updates original message or sends a status notification when a job failed or was interrupted."""
    if not channel_id:
        return

    channel = bot.get_channel(channel_id)
    if channel is None and hasattr(bot, "fetch_channel"):
        try:
            channel = await bot.fetch_channel(channel_id)
        except Exception:
            channel = None

    if not channel:
        return

    embed = discord.Embed(
        title=title,
        description=description,
        color=discord.Color.orange() if status == "interrupted" else discord.Color.red()
    )
    if user_id:
        embed.add_field(name="Requested By", value=f"<@{user_id}>", inline=True)
    embed.set_footer(text="Shallot-CUI Bot • Crash Recovery")

    message = None
    if message_id:
        try:
            message = await channel.fetch_message(message_id)
        except Exception:
            message = None

    if message:
        try:
            await message.edit(content="", embed=embed)
            return
        except Exception:
            pass

    try:
        mention = f"<@{user_id}> " if user_id else ""
        await channel.send(content=mention, embed=embed)
    except Exception:
        pass


def start_crash_recovery(bot: discord.Client, comfy_client: Any) -> asyncio.Task:
    """Spawns non-blocking background task to reconcile pending jobs without delaying bot startup."""
    loop = asyncio.get_event_loop()
    task = loop.create_task(reconcile_pending_jobs(bot, comfy_client))
    return task

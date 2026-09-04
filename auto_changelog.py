"""
Daily Changelog Synchronizer for Shallot-CUI Bot.

Inspects git history and repository modifications for the current day (or since
the last recorded changelog entry), summarizes changes under conventional categories
(Added, Changed, Fixed, Performance), and updates CHANGELOG.md idempotently.
"""

import os
import re
import datetime
import subprocess
from typing import List, Dict, Tuple

CHANGELOG_PATH = os.path.join(os.path.dirname(__file__), "CHANGELOG.md")

def get_git_commits_for_today(target_date_str: str = None) -> List[Tuple[str, str, str]]:
    """
    Retrieves git commits for the specified date (YYYY-MM-DD) or today.
    Returns list of (commit_hash, subject, author).
    """
    if not target_date_str:
        target_date_str = datetime.date.today().isoformat()

    since_date = f"{target_date_str} 00:00:00"
    until_date = f"{target_date_str} 23:59:59"

    cmd = [
        "git", "log",
        f"--since={since_date}",
        f"--until={until_date}",
        "--pretty=format:%h|%s|%an"
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True, cwd=os.path.dirname(__file__))
        lines = res.stdout.strip().splitlines()
        commits = []
        for line in lines:
            if not line.strip():
                continue
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append((parts[0], parts[1], parts[2]))
        return commits
    except Exception:
        return []

IGNORED_WORKING_TREE_PATTERNS = [
    "CHANGELOG.md",
    "cache.db",
    "error_log.json",
    "test_run.tmp",
    "__pycache__",
    ".tmp",
]

def get_working_tree_changes() -> List[str]:
    """Returns uncommitted / modified relevant source files in the working tree."""
    try:
        res = subprocess.run(["git", "status", "-s"], capture_output=True, text=True, check=True, cwd=os.path.dirname(__file__))
        modified = []
        for line in res.stdout.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            # Skip ignored paths
            if any(ign in line for ign in IGNORED_WORKING_TREE_PATTERNS):
                continue
            modified.append(line)
        return modified
    except Exception:
        return []

FRIENDLY_FILE_DESCRIPTIONS = {
    "characters.py": "Character system & presets (Valerie, Sully, Ogarla) with privacy protection",
    "bot.py": "Core slash commands, buttons, and Discord event handlers",
    "parsers.py": "Prompt modifier parser (aspect ratios, styles, wildcards, and character shortcuts)",
    "comfy_client.py": "ComfyUI communication, task queueing, and VRAM memory auto-purge",
    "views.py": "Interactive Discord buttons, Cancel controls, and Remix popups",
    "suite_test.py": "Automated verification test suite",
    "README.md": "Novice-friendly user guide and quickstart documentation",
    "PROJECT_STATE.md": "Project status, features, and architecture documentation",
    "image_utils.py": "Image processing (grid stitching, splitting, and scaling)",
    "db.py": "Database storage for favorite prompts, styles, and history",
    "monitor.py": "GPU telemetry and generation progress tracking",
    "config.py": "Bot settings, models, and safety thresholds",
}

def format_friendly_commit(subject: str, chash: str) -> str:
    """Formats a git commit message into a clean, novice-friendly sentence."""
    clean_subj = subject
    for prefix in ["feat:", "fix:", "refactor:", "perf:", "chore:", "docs:", "test:", "style:"]:
        if clean_subj.lower().startswith(prefix):
            clean_subj = clean_subj[len(prefix):].strip()
            break
    # Remove scope like feat(ui):
    clean_subj = re.sub(r'^[a-z_]+(?:\([^\)]+\))?:\s*', '', clean_subj, flags=re.IGNORECASE).strip()
    if clean_subj:
        clean_subj = clean_subj[0].upper() + clean_subj[1:]
    return f"**{clean_subj}**"

def categorize_changes(commits: List[Tuple[str, str, str]], working_tree: List[str]) -> Dict[str, List[str]]:
    """Groups changes into Added, Changed, Fixed, Performance, and Maintenance with beginner-friendly text."""
    categories = {
        "Added": [],
        "Changed": [],
        "Fixed": [],
        "Performance": [],
        "Maintenance": []
    }

    for chash, subject, _ in commits:
        subj_lower = subject.lower()
        friendly_entry = format_friendly_commit(subject, chash)

        if subj_lower.startswith("feat:") or " add " in subj_lower or " added " in subj_lower:
            categories["Added"].append(friendly_entry)
        elif subj_lower.startswith("fix:") or " fix " in subj_lower or " fixed " in subj_lower:
            categories["Fixed"].append(friendly_entry)
        elif subj_lower.startswith("perf:") or "optimize" in subj_lower or "speed" in subj_lower:
            categories["Performance"].append(friendly_entry)
        elif subj_lower.startswith("chore:") or subj_lower.startswith("docs:") or subj_lower.startswith("test:"):
            categories["Maintenance"].append(friendly_entry)
        else:
            categories["Changed"].append(friendly_entry)

    # If there are active working tree changes, present them as readable component updates
    if working_tree:
        described_files = set()
        for item in working_tree:
            # Extract just the filename from git status (e.g., 'M bot.py' -> 'bot.py')
            parts = item.split()
            fname = parts[-1] if parts else item
            base = os.path.basename(fname)
            desc = FRIENDLY_FILE_DESCRIPTIONS.get(base, f"Updated `{base}`")
            if desc not in described_files:
                categories["Maintenance"].append(f"Component polish: {desc}")
                described_files.add(desc)

    return {k: v for k, v in categories.items() if v}

def update_changelog(date_str: str = None) -> bool:
    """Updates CHANGELOG.md with today's changes if any exist."""
    if not date_str:
        date_str = datetime.date.today().isoformat()

    commits = get_git_commits_for_today(date_str)
    working_tree = get_working_tree_changes()

    if not commits and not working_tree:
        print(f"ℹ️ No modifications or commits found for {date_str}. Changelog is up to date.")
        return False

    categories = categorize_changes(commits, working_tree)
    if not categories:
        print(f"ℹ️ No categorized changes for {date_str}.")
        return False

    if not os.path.exists(CHANGELOG_PATH):
        content = "# Changelog\n\nAll notable changes to **Shallot-CUI Bot** will be documented in this file.\n\n---\n\n"
    else:
        with open(CHANGELOG_PATH, "r", encoding="utf-8") as f:
            content = f.read()

    # If today's section already has detailed human-written entries, don't overwrite them
    date_header_pattern = rf"## \[{re.escape(date_str)}\].*?(?=\n## \[|\Z)"
    existing_match = re.search(date_header_pattern, content, flags=re.DOTALL)
    if existing_match and "### Added" in existing_match.group(0) and "Working tree" not in existing_match.group(0):
        print(f"ℹ️ Changelog already has human-curated entry for {date_str}. Preserving.")
        verify_readme_synchronization()
        return True

    # Build today's entry markdown
    entry_lines = [f"## [{date_str}]", ""]
    for cat_name, items in categories.items():
        entry_lines.append(f"### {cat_name}")
        for item in items:
            entry_lines.append(f"* {item}")
        entry_lines.append("")
    entry_lines.append("---")
    entry_lines.append("")
    entry_text = "\n".join(entry_lines)

    if existing_match:
        new_content = re.sub(date_header_pattern, entry_text.strip(), content, flags=re.DOTALL)
    else:
        # Insert right after the header block
        match = re.search(r"(# Changelog.*?\n---\n\n)", content, flags=re.DOTALL)
        if match:
            header_end = match.end()
            new_content = content[:header_end] + entry_text + content[header_end:]
        else:
            new_content = entry_text + content

    with open(CHANGELOG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)

    print(f"[+] Successfully updated CHANGELOG.md for {date_str}.")
    verify_readme_synchronization()
    return True

README_PATH = os.path.join(os.path.dirname(__file__), "README.md")
BOT_PATH = os.path.join(os.path.dirname(__file__), "bot.py")

def verify_readme_synchronization():
    """Validates that all registered slash commands in bot.py exist in README.md."""
    if not os.path.exists(README_PATH) or not os.path.exists(BOT_PATH):
        return

    try:
        with open(BOT_PATH, "r", encoding="utf-8") as f:
            bot_code = f.read()
        with open(README_PATH, "r", encoding="utf-8") as f:
            readme_text = f.read()

        commands = set(re.findall(r'@(?:tree|bot\.tree)\.command\(name=[\'"]([^\'"]+)[\'"]', bot_code))
        missing_commands = [cmd for cmd in sorted(commands) if f"/{cmd}" not in readme_text and cmd not in readme_text]

        if missing_commands:
            print("\n⚠️ [DOCS ALERT] The following slash commands in bot.py are missing from README.md:")
            for cmd in missing_commands:
                print(f"   - /{cmd}")
            print("   👉 Please update README.md so the user guide stays synchronized!")
        else:
            print(f"✅ README.md is fully synchronized! (Covering all {len(commands)} slash commands)")
    except Exception as e:
        print(f"ℹ️ Could not verify README sync: {e}")

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    update_changelog()
    if len(sys.argv) > 1 and sys.argv[1] == "--check-readme":
        verify_readme_synchronization()


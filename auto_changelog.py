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

def categorize_changes(commits: List[Tuple[str, str, str]], working_tree: List[str]) -> Dict[str, List[str]]:
    """Groups changes into Added, Changed, Fixed, Performance, and Maintenance."""
    categories = {
        "Added": [],
        "Changed": [],
        "Fixed": [],
        "Performance": [],
        "Maintenance": []
    }

    for chash, subject, _ in commits:
        subj_lower = subject.lower()
        formatted_entry = f"**{subject}** (`{chash}`)"

        if subj_lower.startswith("feat:") or " add " in subj_lower or " added " in subj_lower:
            categories["Added"].append(formatted_entry)
        elif subj_lower.startswith("fix:") or " fix " in subj_lower or " fixed " in subj_lower:
            categories["Fixed"].append(formatted_entry)
        elif subj_lower.startswith("perf:") or "optimize" in subj_lower or "speed" in subj_lower:
            categories["Performance"].append(formatted_entry)
        elif subj_lower.startswith("chore:") or subj_lower.startswith("docs:") or subj_lower.startswith("test:"):
            categories["Maintenance"].append(formatted_entry)
        else:
            categories["Changed"].append(formatted_entry)

    # If there are active working tree changes not yet committed
    if working_tree:
        for item in working_tree:
            categories["Maintenance"].append(f"Working tree modification: `{item}`")

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

    # Check if entry for today already exists; if so, replace it cleanly
    date_header_pattern = rf"## \[{re.escape(date_str)}\].*?(?=\n## \[|\Z)"
    if re.search(date_header_pattern, content, flags=re.DOTALL):
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
    return True

if __name__ == "__main__":
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    update_changelog()


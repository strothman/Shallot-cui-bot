#!/usr/bin/env python3
"""
CLI Error Review & Diagnostic Tool for Shallot-CUI-Bot.

Reads error_log.json and prints formatted summaries, category breakdowns,
frequency statistics, and highlights unhandled/failed errors needing IDE attention.
"""

import sys
import argparse
from datetime import datetime
from error_handler import error_handler, AutoFixResult

def format_timestamp(iso_str: str) -> str:
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return iso_str

def main():
    parser = argparse.ArgumentParser(description="Review Shallot-CUI-Bot Error Logs")
    parser.add_argument("--hours", type=int, default=24, help="Filter errors within the last N hours (default: 24)")
    parser.add_argument("--category", type=str, default=None, help="Filter by specific error category")
    parser.add_argument("--recent", type=int, default=10, help="Show last N errors (default: 10)")
    parser.add_argument("--needs-attention", action="store_true", help="Only show errors requiring human/IDE intervention")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(" SHALLOT-CUI-BOT ERROR LOG & DIAGNOSTICS DASHBOARD")
    print("="*60)

    stats = error_handler.get_stats(hours=args.hours)
    print(f"\nSummary (Last {args.hours} Hours):")
    print(f"   * Total Logged Errors : {stats['total']}")
    print(f"   * Auto-Fixed           : {stats['auto_fixed']} [OK]")
    print(f"   * Needs Attention      : {stats['needs_attention']} [FAIL]")

    if stats["by_category"]:
        print("\nBreakdown by Category:")
        for cat, count in stats["by_category"].items():
            print(f"   * {cat:<12} : {count}")

    if args.needs_attention:
        entries = error_handler.get_needs_attention(hours=args.hours)
        print(f"\nUnresolved / Failed Errors Requiring Attention ({len(entries)}):")
    else:
        entries = error_handler.get_recent_errors(count=args.recent, category=args.category)
        print(f"\nRecent Error Entries (Showing last {len(entries)}):")

    if not entries:
        print("   (No matching errors found)")
    else:
        for idx, err in enumerate(entries, 1):
            ts = format_timestamp(err.get("timestamp", ""))
            cat = err.get("category", "UNKNOWN")
            sev = err.get("severity", "ERROR")
            func = err.get("source_function", "unknown")
            file_name = err.get("source_file", "bot.py")
            err_type = err.get("error_type", "Exception")
            msg = err.get("error_message", "")
            
            fix_attempted = err.get("auto_fix_attempted", False)
            fix_action = err.get("auto_fix_action", "NONE")
            fix_result = err.get("auto_fix_result", "NOT_ATTEMPTED")

            status_icon = "[FIXED]" if fix_result == AutoFixResult.SUCCESS.value else "[NEEDS FIX]"
            
            print(f"\n [{idx}] {ts} | {status_icon} | {cat} ({sev})")
            print(f"     Location : {func}() in {file_name}")
            print(f"     Error    : {err_type}: {msg}")
            
            if fix_attempted or fix_action != "NONE":
                print(f"     Auto-Fix : Action={fix_action} -> Result={fix_result}")
                if err.get("auto_fix_detail"):
                    print(f"     Detail   : {err.get('auto_fix_detail')}")

            ctx = err.get("context", {})
            if ctx:
                formatted_ctx = ", ".join([f"{k}={v}" for k, v in ctx.items() if v])
                print(f"     Context  : {formatted_ctx}")

    print("\n" + "="*60 + "\n")

if __name__ == "__main__":
    main()

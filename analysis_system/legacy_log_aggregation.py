"""
legacy_log_aggregation.py — alternate hierarchy-preprocessing step.

Unlike anomaly_workflow.py (which pulls the whole hierarchy tree plus a
hardcoded EXTRA_SOURCES list, then applies Pass 1/Pass 2 rule-based
filtering), this pulls ONLY the specific paths configured in .env's
LOG_FILE_PATHS — nothing else is fetched from the vault at all. What gets
pulled is what gets analyzed; there is no local post-pull filtering step
doing the real work.

LOG_FILE_PATHS is a comma-separated list where each entry is either:
  - an absolute server-side path, fetched as-is and the same for every
    hierarchy (e.g. /var/log/secure), or
  - a path relative to this hierarchy's own vault folder, resolved as
    <vault_root>/<hierarchy>/<path> (e.g. athinio/system/rationalclient.log).

Every file actually pulled gets concatenated into one
"===== <relative_path> =====" - delimited file. That aggregated file is what
gets fed into test.py's InitialAnalysisNode onward; everything downstream of
that (classification, dataset examples, search, explain, IOC vectors,
markdown report) is the exact same LangGraph pipeline as the current
workflow — only this preprocessing step is different.

Usage:
    python legacy_log_aggregation.py 5/101/1/4/1
    python legacy_log_aggregation.py 5/101/1/4/1 --vault-root /rationalVault/data
    python legacy_log_aggregation.py 5/101/1/4/1 --run-analysis
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows consoles sometimes default to a non-UTF-8 codepage; match the same
# reconfiguration anomaly_workflow.py does so this behaves the same way there.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

load_dotenv()

from mcp_client import fetch_directory_files, decode_file_content
from anomaly_workflow import DEFAULT_VAULT_ROOT, CACHE_DIRNAME

AGGREGATED_FILENAME = "logs_aggregated.txt"

# Fallback if LOG_FILE_PATHS isn't set in .env — the same files
# anomaly_workflow.py's EXTRA_SOURCES/Pass-2 checks already treat as logs,
# plus two hierarchy-relative ones confirmed present in the real vault tree.
DEFAULT_LOG_FILE_PATHS = [
    "/var/log/secure",
    "/var/log/osstatus.log",
    "/var/log/audit/audit.log",
    "/var/log/rkhunter/rkhunter.log",
    "/var/log/log_disable.log",
    "athinio/system/osstatus.log",
    "athinio/system/rationalclient.log",
]


def _load_log_file_paths() -> list[str]:
    raw = os.getenv("LOG_FILE_PATHS", "")
    paths = [p.strip() for p in raw.split(",") if p.strip()]
    return paths or DEFAULT_LOG_FILE_PATHS

# Output/scratch files that must never get folded back into their own input
# on a re-run, matching the "previously-generated output files are excluded
# from consolidation" behavior the old single-file consolidation had.
EXCLUDE_TOP_LEVEL_NAMES = {
    AGGREGATED_FILENAME,
    "anomaly_findings.txt",   # belongs to the other (rule-based) pipeline
    "vector_group_output.xml",
}
EXCLUDE_DIR_NAMES = {CACHE_DIRNAME}


def _read_as_text_or_placeholder(path: Path) -> str:
    """Non-UTF-8 (binary) files are included as a placeholder line rather
    than raw bytes, matching the old consolidation behavior."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        size = path.stat().st_size
        return f"[binary file, {size} bytes, not included]"


def _pull_one_configured_path(configured_path: str, vault_root: str, hierarchy_clean: str, hierarchy_dir: Path) -> None:
    """
    Fetch one LOG_FILE_PATHS entry and write it under hierarchy_dir, mirroring
    its own path shape locally. configured_path may name either a single file
    or a directory on the server — mcp_server.py's fetch_directory_files
    handles both, we just need to know which happened to place the result
    correctly (a single-file fetch returns one entry whose relative_path is
    just the bare filename, with no directory prefix to preserve).
    """
    is_absolute = configured_path.startswith("/")
    if is_absolute:
        server_path = configured_path
    else:
        server_path = f"{vault_root.rstrip('/')}/{hierarchy_clean}/{configured_path.strip('/')}"

    try:
        entries = fetch_directory_files(server_path)
    except Exception as exc:
        print(f"  ! Failed to fetch {server_path}: {exc}")
        return
    if not entries:
        print(f"  - Nothing returned for {server_path} (may not exist on this system)")
        return

    server_leaf = Path(server_path.rstrip("/")).name
    is_single_file_fetch = len(entries) == 1 and entries[0].get("relative_path") == server_leaf

    if is_absolute:
        local_base = (
            Path(server_path.rstrip("/")).parent.as_posix().lstrip("/")
            if is_single_file_fetch else server_path.strip("/")
        )
    else:
        if is_single_file_fetch:
            parent = Path(configured_path.strip("/")).parent.as_posix()
            local_base = "" if parent == "." else parent
        else:
            local_base = configured_path.strip("/")

    for entry in entries:
        rel = entry.get("relative_path")
        if not rel:
            continue
        target = (hierarchy_dir / local_base / rel) if local_base else (hierarchy_dir / rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(decode_file_content(entry))

    print(f"  + Pulled {server_path} -> {local_base or '.'}/ ({len(entries)} file(s))")


def pull_configured_log_files(vault_root: str, hierarchy_clean: str, hierarchy_dir: Path) -> None:
    """Pull ONLY the paths listed in LOG_FILE_PATHS (.env) — nothing else is
    fetched from the vault at all, unlike anomaly_workflow.py's whole-tree
    pull plus hardcoded EXTRA_SOURCES."""
    for configured_path in _load_log_file_paths():
        _pull_one_configured_path(configured_path, vault_root, hierarchy_clean, hierarchy_dir)


def consolidate_log_files(hierarchy_dir: Path) -> str:
    """
    Walk the pulled hierarchy and concatenate every file found into one
    heading-delimited string. No name/location-based filtering happens here
    anymore — pull_configured_log_files() already fetched exactly and only
    what LOG_FILE_PATHS names, so whatever landed under hierarchy_dir is by
    construction the intended set. (A prior version re-filtered here with a
    name heuristic, which actively dropped explicitly-configured files whose
    name didn't happen to contain "log" — e.g. /var/log/secure.)
    """
    sections = []
    for path in sorted(hierarchy_dir.rglob("*")):
        if not path.is_file():
            continue

        rel = path.relative_to(hierarchy_dir)
        parts = rel.parts
        if parts and parts[0] in EXCLUDE_DIR_NAMES:
            continue
        if len(parts) == 1 and (parts[0] in EXCLUDE_TOP_LEVEL_NAMES or parts[0].startswith("analysis_report_")):
            continue

        content = _read_as_text_or_placeholder(path)
        sections.append(f"===== {rel.as_posix()} =====\n{content}")

    if not sections:
        return "No log files found in this pull.\n"

    return "\n\n".join(sections) + "\n"


def run_legacy_log_workflow(hierarchy: str, vault_root: str, hierarchies_dir: Path) -> Path:
    """Pull ONLY the log paths configured in .env's LOG_FILE_PATHS (nothing
    else from the vault), aggregate them, and write that as the single file
    meant to feed test.py's pipeline. Returns the path to that aggregated file."""
    hierarchy_clean = hierarchy.strip("/\\")
    hierarchy_dir = hierarchies_dir / Path(hierarchy_clean)

    configured_paths = _load_log_file_paths()
    print("\n" + "=" * 70)
    print(f"PULLING CONFIGURED LOG FILES ONLY ({len(configured_paths)} path(s) from LOG_FILE_PATHS)")
    print("=" * 70)
    for p in configured_paths:
        print(f"  - {p}")
    pull_configured_log_files(vault_root, hierarchy_clean, hierarchy_dir)

    print("\n" + "=" * 70)
    print("CONSOLIDATING LOG FILES ONLY (legacy-style aggregation)")
    print("=" * 70)
    content = consolidate_log_files(hierarchy_dir)

    output_path = hierarchy_dir / AGGREGATED_FILENAME
    output_path.write_text(content, encoding="utf-8")

    line_count = content.count("\n")
    print(f"\n✓ Wrote aggregated log file to {output_path} ({line_count} lines)")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Legacy-style hierarchy preprocessing: pull ONLY the paths configured in .env's "
                    "LOG_FILE_PATHS (nothing else fetched from the vault), then aggregate them into one "
                    "heading-delimited file instead of running rule-based Pass 1/Pass 2 filtering."
    )
    parser.add_argument("hierarchy", help="e.g. 5/101/1/4/1")
    parser.add_argument("--vault-root", default=DEFAULT_VAULT_ROOT)
    parser.add_argument("--dest", default=str(Path(__file__).parent / "hierarchies"))
    parser.add_argument(
        "--run-analysis",
        action="store_true",
        help="Also run test.py's LangGraph pipeline against the aggregated file, same as the current workflow does for anomaly_findings.txt.",
    )
    args = parser.parse_args()

    hierarchies_dir = Path(args.dest)
    output_path = run_legacy_log_workflow(args.hierarchy, args.vault_root, hierarchies_dir)

    if args.run_analysis:
        from test import _run_workflow_for_logs_file
        _run_workflow_for_logs_file(output_path, hierarchies_dir)


if __name__ == "__main__":
    main()

from fastmcp import FastMCP
from pathlib import Path
from dotenv import load_dotenv
import base64
import os

load_dotenv()

mcp = FastMCP("Internal-AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The real vault data root is an absolute, filesystem-root-level path
# (/rationalVault/data), unrelated to wherever this script's repo checkout
# happens to live — it is NOT BASE_DIR/rationalVault/data. Using a
# BASE_DIR-relative path here previously caused upload_file to silently
# write reports to the wrong location once mcp_server.py moved into the
# hierarchy_system/ subdirectory (BASE_DIR shifted, this path didn't).
# Override via DATA_ROOT env var if a deployment's real path differs.
DATA_ROOT = Path(os.getenv("DATA_ROOT", "/rationalVault/data")).resolve()

def _read_file_entry(file_path: Path, relative_path: str) -> dict:
    raw = file_path.read_bytes()
    try:
        content = raw.decode("utf-8")
        encoding = "utf-8"
    except UnicodeDecodeError:
        content = base64.b64encode(raw).decode("ascii")
        encoding = "base64"

    return {
        "relative_path": relative_path,
        "content": content,
        "encoding": encoding,
    }


@mcp.tool()
def fetch_directory_files(root_path: str) -> list[dict]:
    """
    Recursively read every file under root_path on the server filesystem and
    return each one's path (relative to root_path) plus its content, so a
    caller can reconstruct the same directory tree locally. If root_path
    points directly at a single file rather than a directory, returns just
    that file, using its own filename as relative_path.
    """
    base = Path(root_path)
    if not base.is_absolute():
        base = Path(BASE_DIR) / base
    if not base.exists():
        return []

    if base.is_file():
        return [_read_file_entry(base, base.name)]

    if not base.is_dir():
        return []

    files = []
    for file_path in sorted(base.rglob("*")):
        if not file_path.is_file():
            continue
        files.append(_read_file_entry(file_path, file_path.relative_to(base).as_posix()))

    return files


@mcp.tool()
def upload_file(relative_path: str, content: str) -> str:
    """
    Write content to relative_path under rationalVault/data on this machine,
    preserving hierarchy structure (e.g. "5/101/1/4/1/analysis_report_....md")
    so results land alongside the source files they were generated from.
    """
    target = (DATA_ROOT / relative_path).resolve()

    if not str(target).startswith(str(DATA_ROOT)):
        return "Rejected: path escapes data directory"

    if not content or len(content.strip()) == 0:
        print("⚠️ Empty content received — skipping overwrite.")
        return "Empty content ignored"

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nFile saved at: {target}")

    return f"{relative_path} stored successfully"


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8002
    )
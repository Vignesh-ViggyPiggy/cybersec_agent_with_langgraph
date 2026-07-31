from fastmcp import FastMCP
from pathlib import Path
import base64
import os

mcp = FastMCP("Internal-AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@mcp.tool()
def fetch_directory_files(root_path: str) -> list[dict]:
    """
    Recursively read every file under root_path on the server filesystem and
    return each one's path (relative to root_path) plus its content, so a
    caller can reconstruct the same directory tree locally.
    """
    base = Path(root_path)
    if not base.is_absolute():
        base = Path(BASE_DIR) / base
    if not base.exists() or not base.is_dir():
        return []

    files = []
    for file_path in sorted(base.rglob("*")):
        if not file_path.is_file():
            continue

        raw = file_path.read_bytes()
        try:
            content = raw.decode("utf-8")
            encoding = "utf-8"
        except UnicodeDecodeError:
            content = base64.b64encode(raw).decode("ascii")
            encoding = "base64"

        files.append({
            "relative_path": file_path.relative_to(base).as_posix(),
            "content": content,
            "encoding": encoding,
        })

    return files


@mcp.tool()
def upload_file(relative_path: str, content: str) -> str:
    """
    Write content to relative_path under rationalVault/data on this machine,
    preserving hierarchy structure (e.g. "5/101/1/4/1/analysis_report_....md")
    so results land alongside the source files they were generated from.
    """
    data_root = (Path(BASE_DIR) / "rationalVault" / "data").resolve()
    target = (data_root / relative_path).resolve()

    if not str(target).startswith(str(data_root)):
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
        port=8000
    )
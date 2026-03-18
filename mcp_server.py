from fastmcp import FastMCP
import os

mcp = FastMCP("Internal-AI")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@mcp.tool()
def upload_file(filename: str, content: str) -> str:

    # removes paths sent by the client.
    filename = os.path.basename(filename)
    file_path = os.path.join(BASE_DIR, filename)

    if not content or len(content.strip()) == 0:
        print("⚠️ Empty content received — skipping overwrite.")
        return "Empty content ignored"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        os.fsync(f.fileno())

    print(f"\nFile saved at: {file_path}")

    return f"{filename} stored successfully in Internal AI"


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8000
    )
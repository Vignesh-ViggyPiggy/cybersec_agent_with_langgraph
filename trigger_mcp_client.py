# trigger_analysis.py — run manually on the vault machine
import argparse
import asyncio
import os
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()
ANALYSIS_SERVER_URL = os.getenv("ANALYSIS_SERVER_URL") or "http://<analysis-machine-ip>:8001/mcp"


async def _trigger_async(hierarchy, vault_root=None):
    client = Client(ANALYSIS_SERVER_URL)
    args = {"hierarchy": hierarchy}
    if vault_root:
        args["vault_root"] = vault_root
    async with client:
        return await client.call_tool("analyze_hierarchy", args)


def main():
    parser = argparse.ArgumentParser(description="Ask the analysis machine to run the workflow for one hierarchy.")
    parser.add_argument("hierarchy", help="e.g. 5/101/1/4/1")
    parser.add_argument("--vault-root", default=None)
    args = parser.parse_args()
    print(asyncio.run(_trigger_async(args.hierarchy, args.vault_root)))


if __name__ == "__main__":
    main()
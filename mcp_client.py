import asyncio
import os
from fastmcp import Client
from dotenv import load_dotenv

load_dotenv()

async def send_files_async(file, toolname):

    #server_url = os.getenv("MCP_SERVER_URL")

    client = Client("http://100.90.44.5:8000/mcp")

    # read file produced by log analyzer
    with open(file, "r", encoding="utf-8") as f:
        content = f.read()

    async with client:
        result = await client.call_tool(
            toolname,
            {
                "filename": file,
                "content": content
            }
        )
        return result


def send_files(file, toolname):
    return asyncio.run(send_files_async(file, toolname))

# send_files("new_ioc.xml", "upload_file")
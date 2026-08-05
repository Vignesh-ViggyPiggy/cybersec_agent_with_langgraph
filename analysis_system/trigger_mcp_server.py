# analysis_server.py
from fastmcp import FastMCP
from pathlib import Path

from test import _run_workflow_for_hierarchy, DEFAULT_VAULT_ROOT
from mcp_client import send_files

mcp = FastMCP("Analysis-Trigger")

HIERARCHIES_DIR = Path(__file__).parent / "hierarchies"


@mcp.tool()
def analyze_hierarchy(hierarchy: str, vault_root: str = DEFAULT_VAULT_ROOT) -> dict:
    """
    Run the cybersecurity analysis workflow for one hierarchy path
    (e.g. "5/101/1/4/1"). Pulls that hierarchy's files from the vault,
    consolidates them, runs the LangGraph workflow, then pushes the report
    and IOC XML back to the vault alongside the source files.
    """
    result = _run_workflow_for_hierarchy(hierarchy, vault_root, HIERARCHIES_DIR)

    hierarchy_clean = hierarchy.strip("/\\")
    report_path = result.get("report_path")
    xml_output_path = result.get("xml_output_path")

    for local_path in (report_path, xml_output_path):
        if not local_path:
            continue
        local_file = Path(local_path)
        send_files(
            str(local_file),
            "upload_file",
            relative_path=f"{hierarchy_clean}/{local_file.name}",
        )

    return {
        "hierarchy": hierarchy,
        "report_path": report_path,
        "xml_output_path": xml_output_path,
        "threat_level": result.get("explainer_output", {}).get("threat_level"),
    }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8001)
# Cybersecurity Log Analysis Agent with LangGraph

This project is a LangGraph-based cybersecurity analyst that processes Linux log data, classifies incidents using local datasets, enriches context with web search, and generates per-customer reports plus IOC XML outputs.

## What Changed in This Version

- Added customer hierarchy batch processing from `hierarchies/**/logs.txt`.
- Added dataset-driven incident type routing before threat-intel search.
- Outputs are now written beside each customer log file.
- Added optional MCP upload utilities (`mcp_client.py`, `mcp_server.py`).

## Current Workflow

The graph in `test.py` runs these nodes:

1. `InitialAnalysisNode`
2. `InitialSearchFromLogsToDatasetNode`
3. Conditional route:
    - `GettingExamplesUsingIncidentTypeNode` when an incident type matches dataset taxonomy
    - direct to `QuestionFormerNode` when no match
4. `QuestionFormerNode`
5. `ContextDeriverFromSearchQueriesUsingDDGNode`
6. `ExplainerOutputNode`
7. `IOCVectorGroupAdderNode`
8. `MarkdownReportGeneratorNode`

## Project Layout

```text
cybersec_agent_with_langgraph/
   test.py                      # main LangGraph workflow and batch runner
   datasets_files/              # merged incident datasets used for routing/examples
   hierarchies/                 # customer folders (e.g., 1/, 2/, ...)
      <customer>/.../logs.txt    # input log file per customer hierarchy
   mcp_client.py                # optional uploader client
   mcp_server.py                # FastMCP server for file upload endpoint
```

## Requirements

- Python 3.8+
- Ollama running with the model used in `test.py`
- Internet access for DuckDuckGo enrichment (optional but recommended)

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

Create `.env` (optional):

```env
REMOTE_SERVER_URL=http://localhost:8000
NUM_LINES=10
```

Notes:
- If remote collection is unavailable, workflow logic can still run against local `logs.txt` inputs.
- Incident taxonomy is loaded from JSON files in `datasets_files/`.

## Usage

Run the workflow:

```bash
python test.py
```

Runtime behavior:

1. Scans `hierarchies/` recursively for `logs.txt`.
2. Runs one workflow execution per discovered customer log file.
3. If no hierarchy logs exist, falls back to root-level `logs.txt`.

## Outputs

For each processed `logs.txt`, the workflow writes files in the same folder:

- `analysis_report_YYYY-MM-DD_HH-MM-SS.md`
- `vector_group_output.xml`

Example output location:

```text
hierarchies/1/<subtree>/analysis_report_2026-03-18_11-25-30.md
hierarchies/1/<subtree>/vector_group_output.xml
```

Git ignore behavior:
- `hierarchies/**/analysis_report_*.md` is ignored to avoid committing generated reports.

## Optional MCP Upload

- Start server:

```bash
python mcp_server.py
```

- Use `mcp_client.py` helpers to upload generated artifacts to the MCP endpoint.

## Troubleshooting

- `No customer hierarchies were found and logs.txt is missing`:
   Add at least one `logs.txt` under `hierarchies/` or create root `logs.txt`.
- Empty or weak analysis output:
   Verify Ollama model availability and internet access for search enrichment.
- Dataset routing seems off:
   Check `datasets_files/*.json` contains valid `incident_type` fields.

## Last Updated

March 18, 2026

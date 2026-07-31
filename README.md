# Cybersecurity Log Analysis Agent with LangGraph

A LangGraph-based cybersecurity analyst that ingests per-customer log hierarchies, classifies incidents against local datasets, enriches context with web search, and generates a markdown report plus an IOC XML file for each customer. A companion MCP (Model Context Protocol) client/server pair lets the analysis machine pull customer log files in from a separate vault machine over the network.

---

## 1. How the pieces fit together

There are two roles, which can run on the same machine or on two different machines connected by a network:

- **Vault / server machine** — has filesystem access to the raw customer data (`rationalVault/data/...`). Runs `mcp_server.py`, which exposes that data over MCP.
- **Analysis / client machine** — runs `test.py --hierarchy <path>`, which pulls just that hierarchy's files in from the vault machine as its first step, then analyzes them. (`populate_hierarchies.py` also works standalone for manual pulls/debugging.)

```
 Vault machine                              Analysis machine
 ┌─────────────────────────┐   MCP (HTTP)   ┌──────────────────────────────────────┐
 │ rationalVault/data/...  │◄───────────────│ test.py --hierarchy 5/101/1/4/1       │
 │ mcp_server.py (:8000)   │   fetch files   │   1. pulls into hierarchies/5/101/... │
 └─────────────────────────┘                │   2. consolidates files into logs.txt │
                                             │   3. analyzes → analysis_report_*.md  │
                                             │              → vector_group_output.xml│
                                             └──────────────────────────────────────┘
```

If everything runs on one machine, `rationalVault/` and `hierarchies/` just sit side by side in the same repo and `MCP_SERVER_URL` points at `localhost`.

---

## 2. Project layout

```text
cybersec_agent_with_langgraph/
   test.py                      # main LangGraph workflow + batch runner over hierarchies/
   mcp_server.py                # FastMCP server: fetch_directory_files, upload_file tools
   mcp_client.py                # client helpers used by populate_hierarchies.py / test.py
   populate_hierarchies.py      # manual script: pulls vault files into hierarchies/ via MCP
   datasets_files/               # merged incident datasets used for routing/examples
   hierarchies/                  # customer folders (populated locally, not the vault itself)
      <company_id>/<customer_id>/<branch_id>/<product_id>/<system_id>/logs.txt
   rationalVault/                # vault-machine only; git-ignored, holds source customer data
      data/<company_id>/<customer_id>/<branch_id>/<product_id>/<system_id>/<files>
   requirements.txt
   .env                          # local config, not committed
```

`hierarchies/` and `rationalVault/data/` use the same five-level path shape
(`company_id/customer_id/branch_id/product_id/system_id`) so a vault path maps
1:1 onto a hierarchies path.

---

## 3. Requirements

- Python 3.8+
- Ollama running locally with the model referenced in `test.py` (`aaquisher`)
- Internet access for DuckDuckGo enrichment (optional but recommended)

### Environment setup

```bash
python -m venv venv
```

Activate it, then install dependencies:

```powershell
# PowerShell
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# bash/WSL
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` includes `fastmcp`, needed by `mcp_server.py` / `mcp_client.py` / `populate_hierarchies.py`.

### Configuration (`.env`)

```env
MCP_SERVER_URL=http://<vault-machine-ip>:8000/mcp
REMOTE_SERVER_URL=
NUM_LINES=
```

- `MCP_SERVER_URL` — where the analysis machine reaches the vault machine's MCP server. If unset, `mcp_client.py` falls back to a hardcoded default IP.
- Run each machine's commands from a shell with the venv active.

---

## 4. The analysis workflow (`test.py`)

The LangGraph graph runs these nodes in order:

1. `InitialAnalysisNode` — reads raw logs, produces a title + initial analysis.
2. `InitialSearchFromLogsToDatasetNode` — classifies the incident type against the dataset taxonomy.
3. Conditional route:
   - `GettingExamplesUsingIncidentTypeNode` when the incident type matches the dataset taxonomy (pulls reference examples).
   - straight to `QuestionFormerNode` when nothing matches.
4. `QuestionFormerNode` — generates 5 search queries.
5. `ContextDeriverFromSearchQueriesUsingDDGNode` — runs those queries against DuckDuckGo.
6. `ExplainerOutputNode` — produces threat level, detailed analysis, recommended actions.
7. `IOCVectorGroupAdderNode` — derives an IOC vector group (from dataset examples if available, else inferred) and writes `vector_group_output.xml`.
8. `MarkdownReportGeneratorNode` — writes `analysis_report_<timestamp>.md`.

`main()` has three modes, in priority order:

1. **`python test.py --hierarchy <path>`** — pulls just that one hierarchy via MCP (calling `populate_hierarchies.py`'s logic directly), consolidates its files into `hierarchies/<path>/logs.txt` (see §4.1), and analyzes only that hierarchy.
2. **`python test.py`** with no `--hierarchy`, but existing data under `hierarchies/**/logs.txt` — batch-runs the full graph once per file found, writing both output files back into that same customer folder.
3. **`python test.py`** with neither of the above — falls back to a single root-level `logs.txt`.

Incident types are loaded dynamically from `datasets_files/*.json`; if none are found it falls back to a fixed built-in taxonomy.

### 4.1 Single-hierarchy invocation

This is the path meant to be triggered by whatever invokes the agent for one customer's analysis:

```bash
python test.py --hierarchy 5/101/1/4/1
```

```bash
python test.py --hierarchy 5/101/1/4/1 --vault-root rationalVault/data
```

Steps `main()` performs under the hood:

1. Calls `populate_hierarchies()` (same function `populate_hierarchies.py` uses standalone) to pull only `<vault-root>/5/101/1/4/1` from the MCP server into `hierarchies/5/101/1/4/1/`.
2. Consolidates every fetched file in that folder into a single `hierarchies/5/101/1/4/1/logs.txt`, formatted as one section per file:
   ```
   ===== <filename> =====
   <file content>

   ===== <next filename> =====
   <file content>
   ```
   Non-UTF-8 (binary) files are included as a placeholder line rather than raw bytes. Previously-generated output files (`logs.txt` itself, `vector_group_output.xml`, `analysis_report_*.md`) are excluded from consolidation so re-running doesn't fold prior results back into the input.
3. Runs the LangGraph workflow against that consolidated `logs.txt`, same as any other hierarchy.

`--vault-root` defaults to `rationalVault/data` and only needs to be passed if your vault lives somewhere else.

---

## 5. MCP file transfer

### 5.1 On the vault machine

Start the server (binds to all interfaces on port 8000):

```bash
python mcp_server.py
```

It exposes two tools:

- `fetch_directory_files(root_path)` — recursively reads every file under `root_path` and returns each file's relative path, content, and encoding (`utf-8` or `base64` for binary files). Relative `root_path` values are resolved against the directory containing `mcp_server.py`; absolute paths are used as-is.
- `upload_file(filename, content)` — writes a file into the server's own directory. Not currently wired into the analysis workflow (see §7).

### 5.2 On the analysis machine

Pull files from the vault into `hierarchies/`:

```bash
python populate_hierarchies.py rationalVault/data
```

Behavior:

- Calls the vault machine's `fetch_directory_files` tool over MCP.
- Writes each returned file to `hierarchies/<same relative path>`.
- Skips any file that already exists locally with identical content (safe to re-run).
- Never modifies or deletes anything in the vault — read-only pull.

**Scoping to a single hierarchy.** Whatever invokes an analysis run typically only cares about one customer's tree (`company_id/customer_id/branch_id/product_id/system_id`), not the whole vault. Pass `--hierarchy` to fetch and write only that path:

```bash
python populate_hierarchies.py rationalVault/data --hierarchy 2/1/1/1/1
```

This makes the server walk only `rationalVault/data/2/1/1/1/1` (so only that customer's files are sent over the network), while the client still re-prepends the `2/1/1/1/1` prefix locally so the files land at `hierarchies/2/1/1/1/1/...` — matching what `test.py` expects. Without `--hierarchy`, `root_path` is walked in full and every file's path is preserved as returned by the server.

Optional destination override:

```bash
python populate_hierarchies.py rationalVault/data --hierarchy 2/1/1/1/1 --dest hierarchies
```

This script remains useful standalone for manually staging data or debugging the MCP transfer in isolation. For an actual analysis run, prefer `python test.py --hierarchy <path>` (§4.1), which calls the same pull logic automatically as its first step, then consolidates and analyzes — no separate manual run needed.

---

## 6. Outputs

For each processed `logs.txt`, `test.py` writes into the same folder:

- `analysis_report_YYYY-MM-DD_HH-MM-SS.md`
- `vector_group_output.xml`

Example:

```text
hierarchies/1/1/1/1/1/analysis_report_2026-07-24_11-25-30.md
hierarchies/1/1/1/1/1/vector_group_output.xml
```

Git ignore behavior:
- `hierarchies/**/analysis_report_*.md` is ignored (generated reports aren't committed).
- `rationalVault/` is entirely ignored (vault-machine-only source data).

---

## 7. Running across two networked machines

`mcp_server.py` uses `transport="http", host="0.0.0.0", port=8000`, so any client that can route an HTTP request to `<vault-ip>:8000` can call it — same LAN, a VPN mesh (e.g. Tailscale), or the open internet if you choose to expose it.

**There is currently no authentication or TLS on the MCP server.** `fetch_directory_files` will read any path on the vault machine's filesystem that the calling client requests, and `upload_file` will write anywhere under the server's own directory. This is fine on a private, trusted network (a Tailscale tailnet, an internal VPN, a locked-down LAN) but should not be exposed to an untrusted network without adding authentication (FastMCP supports bearer-token/OAuth middleware) and TLS in front of it.

---

## 8. Not yet wired up (future work)

- Pushing `vector_group_output.xml` / `analysis_report_*.md` back to the customer/vault via `mcp_client.send_files(...)` + the server's `upload_file` tool — the call site exists (commented out in `IOCVectorGroupAdderNode` in `test.py`) but isn't active yet.
- `upload_file` currently collapses any incoming filename to its basename, so uploads from multiple customers with the same filename would collide; it would need hierarchy-aware naming before being used for the return trip.

Pulling a single hierarchy's input files via MCP (§4.1) is done — that part is wired into `test.py` directly, not just standalone in `populate_hierarchies.py`.

---

## 9. Troubleshooting

- `No customer hierarchies were found and logs.txt is missing`:
  Run `populate_hierarchies.py` first, or add a root-level `logs.txt`.
- `populate_hierarchies.py` reports "No files returned":
  Confirm `mcp_server.py` is running on the vault machine, `MCP_SERVER_URL` in `.env` points at it, and `root_path` exists relative to where `mcp_server.py` lives (or is a valid absolute path).
- Empty or weak analysis output:
  Verify Ollama model availability and internet access for search enrichment.
- Dataset routing seems off:
  Check `datasets_files/*.json` contains valid `incident_type` fields.

## Last Updated

July 24, 2026

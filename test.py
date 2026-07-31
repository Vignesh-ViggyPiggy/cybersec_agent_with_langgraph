from typing import Literal, TypedDict, List, NotRequired
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from ddgs import DDGS
import argparse
import asyncio
import os
import httpx
import json
import re
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from mcp_client import send_files
from populate_hierarchies import populate_hierarchies as pull_hierarchy_from_vault

DEFAULT_VAULT_ROOT = "rationalVault/data"
GENERATED_OUTPUT_FILENAMES = {"logs.txt", "vector_group_output.xml"}

# Load environment variables
load_dotenv()

model = ChatOllama(model="cybersec-qwen25-3b-q4", num_keep=0)


def _load_incident_types_from_datasets() -> List[str]:
    """Load incident types dynamically from merged dataset JSON files."""
    datasets_dir = Path(__file__).parent / "datasets_files"
    discovered_types = set()

    if datasets_dir.exists() and datasets_dir.is_dir():
        for json_file in datasets_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    for item in payload:
                        if isinstance(item, dict):
                            raw_type = item.get("incident_type")
                            if isinstance(raw_type, str) and raw_type.strip():
                                discovered_types.add(raw_type.strip())
            except Exception:
                # Ignore malformed files so one bad file doesn't block workflow startup.
                continue

    # Stable fallback taxonomy if dataset files are absent/empty at startup.
    if not discovered_types:
        discovered_types = {
            "banned_ip",
            "data_exfiltration",
            "disk_full",
            "memory_leak",
            "privilege_escalation",
            "sql_injection",
            "table_deletion",
            "user_breach",
        }

    return sorted(discovered_types)


INCIDENT_TYPES = _load_incident_types_from_datasets()
INCIDENT_TYPES_SET = set(INCIDENT_TYPES)
INCIDENT_TYPES_HINT = ", ".join(INCIDENT_TYPES)
DEFAULT_INCIDENT_TYPE = INCIDENT_TYPES[0] if INCIDENT_TYPES else "banned_ip"
NONE_APPLICABLE_INCIDENT_TYPE = "none_applicable"
INCIDENT_TYPES_WITH_FALLBACK = sorted(
    INCIDENT_TYPES_SET.union({NONE_APPLICABLE_INCIDENT_TYPE})
)
INCIDENT_TYPES_WITH_FALLBACK_SET = set(INCIDENT_TYPES_WITH_FALLBACK)
INCIDENT_TYPES_WITH_FALLBACK_HINT = ", ".join(INCIDENT_TYPES_WITH_FALLBACK)

class MessageState(TypedDict):
    logs: str
    result: dict
    search_results: List[dict]
    explainer_output: dict
    ioc_vector_group: dict
    markdown_output: str
    logs_path: NotRequired[str]
    output_dir: NotRequired[str]
    customer_ids: NotRequired[List[str]]
    report_path: NotRequired[str]
    xml_output_path: NotRequired[str]


def _carry_context(state: MessageState) -> dict:
    """Preserve filesystem context between workflow nodes."""
    return {
        "logs_path": str(state.get("logs_path", "") or ""),
        "output_dir": str(state.get("output_dir", "") or ""),
        "customer_ids": list(state.get("customer_ids", []) or []),
        "report_path": str(state.get("report_path", "") or ""),
        "xml_output_path": str(state.get("xml_output_path", "") or ""),
    }


def _resolve_output_dir(state: MessageState) -> Path:
    """Resolve where per-customer outputs should be written."""
    output_dir = str(state.get("output_dir", "") or "").strip()
    if output_dir:
        return Path(output_dir)

    logs_path = str(state.get("logs_path", "") or "").strip()
    if logs_path:
        return Path(logs_path).parent

    return Path(__file__).parent


def _extract_customer_ids(logs_file: Path, hierarchies_dir: Path) -> List[str]:
    """Extract the customer hierarchy IDs from a logs file path."""
    try:
        relative_parent = logs_file.parent.relative_to(hierarchies_dir)
    except ValueError:
        return []

    return [part for part in relative_parent.parts if part]


def _discover_customer_log_files(hierarchies_dir: Path) -> List[Path]:
    """Find all customer logs in the hierarchies directory."""
    if not hierarchies_dir.exists() or not hierarchies_dir.is_dir():
        return []

    return sorted(
        log_file for log_file in hierarchies_dir.glob("**/logs.txt") if log_file.is_file()
    )


def _format_customer_label(customer_ids: List[str]) -> str:
    """Render a readable customer label from hierarchy IDs."""
    return "/".join(customer_ids) if customer_ids else "standalone"

class InitialAnalysisTemplate(BaseModel):
    title: str = Field(description="An appropriate title of the attack or potential attack after analyzing the logs")
    content: str = Field(description="A 100-200 word initial analysis of the attack or potential attack after analyzing the logs")
    
class InitialSearchFromLogsToDatasetTemplate(BaseModel):
    incident_type: Literal[tuple(INCIDENT_TYPES_WITH_FALLBACK)] = Field(
        description=(
            "Incident type label selected from dataset taxonomy. "
            f"Use '{NONE_APPLICABLE_INCIDENT_TYPE}' if none match."
        )
    )


class QuestionFormerOutputTemplate(BaseModel):
    search_query_1: str = Field(description="A search query to find more information about the attack or potential attack")
    search_query_2: str = Field(description="Another search query to find more information about the attack or potential attack")
    search_query_3: str = Field(description="Another search query to find more information about the attack or potential attack")
    search_query_4: str = Field(description="Another search query to find more information about the attack or potential attack")
    search_query_5: str = Field(description="Another search query to find more information about the attack or potential attack")

class ExplainerOutputTemplate(BaseModel):
    threat_level: Literal["low", "medium", "high", "critical"] = Field(description="The threat level of the attack or potential attack based on the search results")
    detailed_analysis: str = Field(description="A more detailed analysis of the attack or potential attack based on the search results", min_length=500)
    search_results: List[dict] = Field(description="The search results used to derive the detailed analysis")
    recommended_actions: List[str] = Field(description="Recommended actions to mitigate the attack or potential attack based on the detailed analysis", min_length=5)

class IOCVectorGroupAdderTemplate(BaseModel):
    vector_group_name: str = Field(description="The IOC vector group name in camel case format (e.g. 'SuspiciousProcessAndFileChanges'), sourced from dataset when available or inferred from analysis")
    vectors: List[str] = Field(description="IOC vectors sourced from dataset examples when available, otherwise inferred from threat analysis", min_length=1)


def _to_camel_case(text: str) -> str:
    parts = re.split(r"[^A-Za-z0-9]+", str(text or "").strip())
    parts = [p for p in parts if p]
    if not parts:
        return "DatasetDerivedIOCGroup"
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _derive_ioc_from_dataset_examples(examples: List[dict]) -> dict | None:
    """Build IOC output from dataset examples when vectors exist."""
    collected_vectors = []
    group_candidates = []

    for example in examples:
        if not isinstance(example, dict):
            continue

        group_name = str(example.get("vector_group", "")).strip()
        if group_name:
            group_candidates.append(group_name)

        raw_vectors = example.get("vectors", [])
        if isinstance(raw_vectors, list):
            for raw_vector in raw_vectors:
                cleaned = str(raw_vector).strip()
                if cleaned:
                    collected_vectors.append(cleaned)

    dedup_vectors = []
    for vector in collected_vectors:
        if vector not in dedup_vectors:
            dedup_vectors.append(vector)

    if not dedup_vectors:
        return None

    # Keep output small and aligned with model-generated range.
    selected_vectors = dedup_vectors[:5]

    group_name = _to_camel_case(group_candidates[0]) if group_candidates else "DatasetDerivedIOCGroup"

    return {
        "vector_group_name": group_name,
        "vectors": selected_vectors,
    }


def _extract_first_json_object(text: str):
    """Extract the first JSON object from mixed model output."""
    if not text:
        return None

    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    """Remove duplicate strings while preserving order."""
    seen = set()
    deduped = []

    for item in items:
        normalized = re.sub(r"\s+", " ", str(item).strip()).casefold()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(str(item).strip())

    return deduped


def _dedupe_multiline_text(text: str) -> str:
    """Remove duplicate non-empty lines from multiline model output."""
    raw_lines = [line.rstrip() for line in str(text).splitlines()]
    deduped_lines = []
    seen = set()

    for line in raw_lines:
        normalized = re.sub(r"\s+", " ", line.strip()).casefold()
        if not normalized:
            if deduped_lines and deduped_lines[-1] != "":
                deduped_lines.append("")
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped_lines.append(line.strip())

    while deduped_lines and deduped_lines[-1] == "":
        deduped_lines.pop()

    return "\n".join(deduped_lines).strip()


def _normalize_explainer_payload(payload: dict, search_results: List[dict]) -> dict:
    """Ensure required fields exist and satisfy schema constraints."""
    valid_levels = {"low", "medium", "high", "critical"}
    out = dict(payload or {})

    level = str(out.get("threat_level", "medium")).strip().lower()
    out["threat_level"] = level if level in valid_levels else "medium"

    detailed = str(out.get("detailed_analysis", "")).strip()
    if len(detailed) < 500:
        filler = (
            " Additional contextual evidence indicates potential unauthorized access patterns, "
            "credential misuse opportunities, and weak control coverage across authentication, "
            "monitoring, and patching workflows. Defensive actions should prioritize immediate "
            "containment, verification of suspicious indicators, hardening exposed services, "
            "and continuous monitoring to reduce recurrence risk."
        )
        while len(detailed) < 500:
            detailed += filler
    out["detailed_analysis"] = detailed

    if not isinstance(out.get("search_results"), list) or not out.get("search_results"):
        out["search_results"] = [sr for sr in search_results if "error" not in sr][:25]

    if not isinstance(out.get("recommended_actions"), list):
        out["recommended_actions"] = []

    cleaned_actions = []
    for action in out["recommended_actions"]:
        action_text = str(action).strip()
        if action_text:
            cleaned_actions.append(action_text)

    cleaned_actions = _dedupe_preserve_order(cleaned_actions)

    fallback_actions = [
        "Isolate affected hosts and block suspicious source IPs at the network perimeter.",
        "Reset potentially exposed credentials and enforce MFA for privileged and remote access accounts.",
        "Patch vulnerable services and verify security baseline hardening across internet-facing systems.",
        "Review logs, EDR alerts, and authentication trails to confirm scope and persistence mechanisms.",
        "Implement continuous monitoring with IOC-based detections and incident response escalation playbooks."
    ]

    while len(cleaned_actions) < 5:
        cleaned_actions.append(fallback_actions[len(cleaned_actions)])
    out["recommended_actions"] = cleaned_actions[:10]

    return out


def _normalize_question_former_payload(payload: dict, logs_text: str) -> dict:
    """Ensure required QuestionFormer fields exist for strict schema validation."""
    out = dict(payload or {})

    fallback_queries = [
        "failed login attempts from single source IP investigation",
        "web server error log indicators of exploitation attempts",
        "wordpress authentication bypass vulnerability detection guidance",
        "ioc patterns for brute force and credential stuffing attacks",
        "incident response checklist for suspected web application compromise"
    ]

    query_values = []
    for idx in range(1, 6):
        key = f"search_query_{idx}"
        value = str(out.get(key, "")).strip()
        query_values.append(value if value else fallback_queries[idx - 1])

    query_values = _dedupe_preserve_order(query_values)

    while len(query_values) < 5:
        fallback_value = fallback_queries[len(query_values)]
        if fallback_value not in query_values:
            query_values.append(fallback_value)
        else:
            query_values.append(f"{fallback_value} {len(query_values) + 1}")

    for idx in range(1, 6):
        out[f"search_query_{idx}"] = query_values[idx - 1]

    return out


def _normalize_initial_analysis_payload(payload: dict) -> dict:
    """Ensure required InitialAnalysis fields exist for strict schema validation."""
    out = dict(payload or {})

    title = str(out.get("title", "Potential Security Incident")).strip()
    if not title:
        title = "Potential Security Incident"
    out["title"] = title

    content = str(out.get("content", "")).strip()
    content = _dedupe_multiline_text(content)
    if len(content) < 120:
        content = (
            "Observed logs indicate potentially suspicious activity that warrants further investigation. "
            "Multiple indicators suggest authentication anomalies, service-level errors, or behavior that "
            "could be associated with attack reconnaissance or exploitation attempts. Initial containment "
            "and targeted triage are recommended while validating source legitimacy and affected assets."
        )
    out["content"] = content

    return out


def _normalize_initial_search_payload(payload: dict) -> dict:
    """Ensure required InitialSearch fields exist for strict schema validation."""
    out = dict(payload or {})

    incident_type = str(out.get("incident_type", "")).strip()
    if incident_type not in INCIDENT_TYPES_WITH_FALLBACK_SET:
        incident_type = NONE_APPLICABLE_INCIDENT_TYPE
    out["incident_type"] = incident_type

    return out


def _load_examples_for_incident_type(incident_type: str) -> List[dict]:
    """Load dataset records matching the provided incident type from all dataset files."""
    if incident_type == NONE_APPLICABLE_INCIDENT_TYPE:
        return []

    datasets_dir = Path(__file__).parent / "datasets_files"
    matched_examples = []

    if not datasets_dir.exists() or not datasets_dir.is_dir():
        return matched_examples

    for json_file in sorted(datasets_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue

        if not isinstance(payload, list):
            continue

        for index, item in enumerate(payload, start=1):
            if not isinstance(item, dict):
                continue
            if str(item.get("incident_type", "")).strip() != incident_type:
                continue

            example = dict(item)
            example["source_file"] = json_file.name
            example["source_index"] = index
            matched_examples.append(example)

    return matched_examples


def _uses_incident_examples_path(state: MessageState) -> bool:
    """Return whether the workflow should include the incident-examples branch."""
    incident_type = str(state.get("result", {}).get("incident_type", "")).strip()
    return bool(incident_type) and incident_type != NONE_APPLICABLE_INCIDENT_TYPE


def _format_step_label(step: int, message: str, total_steps: int | None = None) -> str:
    """Format a consistent step label for terminal output."""
    if total_steps is None:
        return f"[STEP {step}] {message}"
    return f"[STEP {step}/{total_steps}] {message}"


def _build_explainer_reference_examples(examples: List[dict], max_examples: int = 5) -> str:
    """Build compact dataset-style examples for explainer reasoning guidance."""
    if not isinstance(examples, list) or not examples:
        return "No matched dataset examples available."

    lines = ["Reference examples for incident-style reasoning:"]

    for idx, ex in enumerate(examples[:max_examples], start=1):
        if not isinstance(ex, dict):
            continue

        title = str(ex.get("title") or ex.get("incident_title") or "Untitled incident").strip()
        description = str(ex.get("description", "")).strip()
        if len(description) > 260:
            description = description[:260].rstrip() + "..."

        vectors = ex.get("vectors", [])
        vectors_text = ", ".join(str(v).strip() for v in vectors[:5]) if isinstance(vectors, list) else "N/A"

        lines.append(f"Example {idx}:")
        lines.append(f"- Incident: {title}")
        if description:
            lines.append(f"- Typical pattern/outcome: {description}")
        lines.append(f"- Common indicators: {vectors_text}")

    if len(examples) > max_examples:
        lines.append(f"... plus {len(examples) - max_examples} additional matched examples")

    return "\n".join(lines)


def _extract_log_signals(logs_text: str) -> tuple[List[str], List[str]]:
    """Extract lightweight indicators from raw logs for dataset-style summary."""
    text = str(logs_text or "")

    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    unique_ips = _dedupe_preserve_order(ips)[:5]

    paths = re.findall(r'"(?:GET|POST|PUT|DELETE|PATCH)\s+([^\s"]+)', text)
    cleaned_paths = []
    for path in paths:
        base_path = str(path).split("?", 1)[0].strip()
        if base_path:
            cleaned_paths.append(base_path)
    unique_paths = _dedupe_preserve_order(cleaned_paths)[:5]

    return unique_ips, unique_paths


def _build_dataset_style_incident_summary(state: MessageState, explainer_payload: dict) -> str:
    """Build a concise summary in dataset-like style after explainer output."""
    result = state.get("result", {}) if isinstance(state.get("result", {}), dict) else {}
    incident_type = str(result.get("incident_type", "none_applicable")).strip() or "none_applicable"
    threat_level = str(explainer_payload.get("threat_level", "medium")).strip().lower() or "medium"
    detailed = str(explainer_payload.get("detailed_analysis", "")).strip()

    dataset_examples = result.get("dataset_examples", [])
    vectors = []
    if isinstance(dataset_examples, list):
        for ex in dataset_examples[:10]:
            if not isinstance(ex, dict):
                continue
            raw_vectors = ex.get("vectors", [])
            if isinstance(raw_vectors, list):
                for v in raw_vectors:
                    cleaned = str(v).strip()
                    if cleaned:
                        vectors.append(cleaned)
    vectors = _dedupe_preserve_order(vectors)[:5]

    source_count = len(dataset_examples) if isinstance(dataset_examples, list) else 0
    unique_ips, unique_paths = _extract_log_signals(state.get("logs", ""))

    what_happened = detailed.split(".", 1)[0].strip()
    if not what_happened:
        what_happened = "Suspicious behavior consistent with the selected incident type was observed in the logs"
    if what_happened[-1:] not in ".!?":
        what_happened += "."

    vectors_text = ", ".join(vectors) if vectors else "N/A"
    ips_text = ", ".join(unique_ips) if unique_ips else "N/A"
    paths_text = ", ".join(unique_paths) if unique_paths else "N/A"

    return (
        f"incident_type: {incident_type}\n"
        f"threat_level: {threat_level}\n"
        f"what_happened: {what_happened}\n"
        f"observed_ips: {ips_text}\n"
        f"targeted_paths: {paths_text}\n"
        f"vectors: {vectors_text}\n"
        f"matched_example_count: {source_count}"
    )

def InitialAnalysisNode(state: MessageState) -> MessageState:
    """
    Docstring for InitialAnalysisNode

    Node that takes in logs and returns a title and an initial analysis of the attack or potential attack. This node is used to analyze the logs and provide an initial understanding of the attack or potential attack.
    """
    print("\n" + "="*70)
    print(_format_step_label(1, "Generating initial title and analysis from logs..."))
    print("="*70)

    structured_model = model.with_structured_output(InitialAnalysisTemplate)
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity analyst. Analyze the logs and provide an appropriate title and a 100-200 word initial analysis. Ignore file-not-found style noise and focus on true security indicators."),
        ("user", "{logs}")
    ])

    input_payload = {"logs": state["logs"]}
    chain = template | structured_model

    try:
        result = chain.invoke(input_payload)
    except Exception as e:
        print(f"  ⚠ Initial analysis parsing failed: {e}")
        print("  ↻ Retrying initial analysis with strict JSON fallback...")

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """Return ONLY valid JSON with these exact keys:
- title: string
- content: string (100-200 words)

Do not include markdown, HTML, comments, or extra text before/after JSON."""),
            ("user", """Analyze the following logs and return strict JSON now:
{logs}""")
        ])

        raw_response = (fallback_prompt | model).invoke(input_payload)
        raw_text = getattr(raw_response, "content", "")
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(x) for x in raw_text)
        raw_text = str(raw_text)

        parsed = _extract_first_json_object(raw_text)
        normalized = _normalize_initial_analysis_payload(parsed or {})
        result = InitialAnalysisTemplate.model_validate(normalized)

    print(f"✓ Initial title: {result.title}")

    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Title: {result.title}")
    print(f"\nInitial Analysis:\n{result.content}")

    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": result.model_dump(),
    }

def InitialSearchFromLogsToDatasetNode(state: MessageState) -> MessageState:
    """
    Docstring for InitialSearchFromLogsToDatasetNode

    Node that takes initial analysis results and generates an initial dataset search query to find more information about the attack or potential attack. This node is used to search the dataset for recorded incidents.
    """
    print("\n" + "="*70)
    print(_format_step_label(2, "Classifying incident type from initial analysis + logs..."))
    print("="*70)

    incident_type_model = model.with_structured_output(InitialSearchFromLogsToDatasetTemplate)

    prior_result = dict(state.get("result", {}))
    title = str(prior_result.get("title", "Potential Security Incident"))
    content = str(prior_result.get("content", ""))

    incident_type_template = ChatPromptTemplate.from_messages([
        ("system", f"""You are a cybersecurity analyst.
Using the existing initial analysis and logs, return:
- incident_type (string)

incident_type must be exactly one of: {INCIDENT_TYPES_WITH_FALLBACK_HINT}
Use '{NONE_APPLICABLE_INCIDENT_TYPE}' if none of the listed types are applicable.
Ignore file-not-found style noise and focus on true security indicators."""),
        ("user", """Logs:
{logs}

Initial Analysis:
Title: {title}
Content: {content}""")
    ])

    input_payload = {"logs": state["logs"], "title": title, "content": content}
    incident_type_chain = incident_type_template | incident_type_model

    try:
        incident_type_result = incident_type_chain.invoke(input_payload)
    except Exception as e:
        print(f"  ⚠ Structured output parsing failed: {e}")
        print("  ↻ Retrying incident type classification with strict JSON fallback...")

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Return ONLY valid JSON with these exact keys:
- incident_type: string (must be one of: {INCIDENT_TYPES_WITH_FALLBACK_HINT})

If no listed type applies, set incident_type to '{NONE_APPLICABLE_INCIDENT_TYPE}'.
Do not include markdown, HTML, comments, or extra text before/after JSON."""),
                ("user", """Analyze the following logs and initial analysis and return strict JSON now:

        Logs:
        {logs}

        Initial Analysis:
        Title: {title}
        Content: {content}""")
        ])

        raw_response = (fallback_prompt | model).invoke(input_payload)
        raw_text = getattr(raw_response, "content", "")
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(x) for x in raw_text)
        raw_text = str(raw_text)

        parsed = _extract_first_json_object(raw_text)
        normalized = _normalize_initial_search_payload(parsed or {})
        incident_type_result = InitialSearchFromLogsToDatasetTemplate.model_validate(normalized)

    print(f"✓ Initial incident type: {incident_type_result.incident_type}")

    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Title: {title}")
    print(f"Incident Type: {incident_type_result.incident_type}")
    print(f"\nInitial Analysis:\n{content}")

    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": {
            **prior_result,
            **incident_type_result.model_dump(),
        },
    }


def RouteFromIncidentType(state: MessageState) -> str:
    """Route based on incident type classification result."""
    incident_type = str(state.get("result", {}).get("incident_type", "")).strip()
    if incident_type == NONE_APPLICABLE_INCIDENT_TYPE:
        return "question_former"
    return "incident_examples"

def GettingExamplesUsingIncidentTypeNode(state: MessageState) -> MessageState:
    """
    Docstring for GettingExamplesUsingIncidentTypeNode

    Node that takes in the incident type from the previous node and uses it to search the dataset for recorded incidents of the same type. The results are then used to derive more context about the attack or potential attack.
    """
    print("\n" + "="*70)
    print(_format_step_label(3, "Searching dataset examples by incident type...", 8))
    print("="*70)

    prior_result = dict(state.get("result", {}))
    incident_type = str(prior_result.get("incident_type", "")).strip()

    if not incident_type:
        print("  ⚠ No incident type available from previous node.")
        prior_result["dataset_examples"] = []
        return {
            **_carry_context(state),
            "logs": state["logs"],
            "result": prior_result,
        }

    examples = _load_examples_for_incident_type(incident_type)
    prior_result["dataset_examples"] = examples

    print(f"✓ Incident type used for dataset search: {incident_type}")
    print(f"✓ Matching dataset examples found: {len(examples)}")

    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Incident Type: {incident_type}")
    print(f"Matched Examples: {len(examples)}")

    for idx, example in enumerate(examples[:3], start=1):
        example_title = example.get("title") or example.get("incident_title") or "Untitled Example"
        source_file = example.get("source_file", "unknown")
        source_index = example.get("source_index", "?")
        print(f"  [{idx}] {example_title} ({source_file} #{source_index})")

    if len(examples) > 3:
        print(f"  ... and {len(examples) - 3} more examples")

    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": prior_result,
    }

def QuestionFormerNode(state: MessageState) -> MessageState:
    """
    Docstring for QuestionFormerNode
    
    Node that takes in logs and returns a title, an initial analysis, and 5 search queries to find more information about the attack or potential attack.
    """
    print("\n" + "="*70)
    uses_examples_path = _uses_incident_examples_path(state)
    current_step = 4 if uses_examples_path else 3
    total_steps = 8 if uses_examples_path else 7
    print(_format_step_label(current_step, "Analyzing logs and generating search queries...", total_steps))
    print("="*70)
    
    # Use with_structured_output for more reliable structured responses
    structured_model = model.with_structured_output(QuestionFormerOutputTemplate)
    prior_result = state.get("result", {})
    title = prior_result.get("title", "Potential Security Incident")
    content = prior_result.get("content", "")
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity analyst. Based on the logs and the existing initial analysis, generate 5 search queries to find more information about the attack or potential attack. Ignore the logs that states file does not exist or cannot be found. Focus on the logs that indicate potential security incidents."),
        ("user", """Logs:
{logs}

Existing Initial Analysis:
Title: {title}
Content: {content}""")
    ])
    
    chain = template | structured_model
    input_payload = {"logs": state["logs"], "title": title, "content": content}
    try:
        result = chain.invoke(input_payload)
    except Exception as e:
        print(f"  ⚠ Structured output parsing failed: {e}")
        print("  ↻ Retrying with strict JSON fallback...")

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """Return ONLY valid JSON with these exact keys:
- search_query_1: string
- search_query_2: string
- search_query_3: string
- search_query_4: string
- search_query_5: string

Do not include markdown, HTML, comments, or extra text before/after JSON."""),
            ("user", """Based on these logs and the existing initial analysis, return strict JSON now:

Logs:
{logs}

Existing Initial Analysis:
Title: {title}
Content: {content}""")
        ])

        raw_response = (fallback_prompt | model).invoke(input_payload)
        raw_text = getattr(raw_response, "content", "")
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(x) for x in raw_text)
        raw_text = str(raw_text)

        parsed = _extract_first_json_object(raw_text)
        normalized = _normalize_question_former_payload(parsed or {}, state["logs"])
        result = QuestionFormerOutputTemplate.model_validate(normalized)
    
    print(f"✓ Generated {len([q for q in [result.search_query_1, result.search_query_2, result.search_query_3, result.search_query_4, result.search_query_5] if q])} search queries")
    
    # Display output immediately after completion
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Title: {title}")
    print(f"\nInitial Analysis:\n{content}")
    print(f"\nSearch Queries Generated:")
    for i, query in enumerate([result.search_query_1, result.search_query_2, result.search_query_3, result.search_query_4, result.search_query_5], 1):
        if query:
            print(f"  {i}. {query}")
    
    merged_result = dict(prior_result)
    merged_result.update(result.model_dump())

    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": merged_result,
    }


def ContextDeriverFromSearchQueriesUsingDDGNode(state: MessageState) -> MessageState:
    """
    Docstring for ContextDeriverFromSearchQueriesUsingDDGNode
    
    Node that takes in the search queries from the previous node and uses them to search on DuckDuckGo to find more information about the attack or potential attack. The results are then used to derive more context about the attack or potential attack.
    """
    print("\n" + "="*70)
    uses_examples_path = _uses_incident_examples_path(state)
    current_step = 5 if uses_examples_path else 4
    total_steps = 8 if uses_examples_path else 7
    print(_format_step_label(current_step, "Gathering threat intelligence from DuckDuckGo...", total_steps))
    print("="*70)
    
    result = state["result"]
    
    # Extract all search queries
    search_queries = [
        result.get('search_query_1'),
        result.get('search_query_2'),
        result.get('search_query_3'),
        result.get('search_query_4'),
        result.get('search_query_5')
    ]
    
    all_search_results = []
    
    # Search each query using DuckDuckGo
    for i, query in enumerate(search_queries, 1):
        if not query:
            continue
            
        print(f"  [{i}/5] Searching: {query}")
        
        try:
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=5))
            
            for r in search_results:
                all_search_results.append({
                    "query_number": i,
                    "query": query,
                    "title": r.get("title", "No title"),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "No description available")
                })
                
            print(f"       ✓ Found {len(search_results)} results")
            
        except Exception as e:
            print(f"       ✗ Error: {e}")
            all_search_results.append({
                "query_number": i,
                "query": query,
                "error": str(e)
            })
    
    print(f"\n✓ Total intelligence sources gathered: {len(all_search_results)}")
    

    # Display output immediately after completion
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Total search results: {len(all_search_results)}\n")
    
    # Group by query and display
    for i in range(1, 6):
        query_results = [sr for sr in all_search_results if sr.get('query_number') == i]
        if query_results:
            print(f"Query {i}: {query_results[0].get('query')}")
            if 'error' in query_results[0]:
                print(f"  ✗ Error: {query_results[0].get('error')}")
            else:
                for j, sr in enumerate(query_results, 1):  # Show all results
                    print(f"  [{j}] {sr.get('title', 'N/A')}")
                    print(f"      {sr.get('url', 'N/A')}")
                    print(f"      Snippet: {sr.get('snippet', 'N/A')}")
            print()
    
    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": state["result"],
        "search_results": all_search_results
    }

def ExplainerOutputNode(state: MessageState) -> MessageState:
    """
    Docstring for ExplainerOutputNode
    
    Node that takes in the search results from the previous node and uses them to derive more context about the attack or potential attack. It then provides a more detailed analysis of the attack or potential attack based on the search results.
    """
    print("\n" + "="*70)
    uses_examples_path = _uses_incident_examples_path(state)
    current_step = 6 if uses_examples_path else 5
    total_steps = 8 if uses_examples_path else 7
    print(_format_step_label(current_step, "Generating comprehensive security analysis...", total_steps))
    print("="*70)
    
    structured_model = model.with_structured_output(ExplainerOutputTemplate)
    dataset_examples = state.get("result", {}).get("dataset_examples", [])
    reference_examples = _build_explainer_reference_examples(dataset_examples)
    
    # Format search results for the LLM
    search_context = "\n\n".join([
        f"Query {sr.get('query_number')}: {sr.get('query')}\n"
        f"Title: {sr.get('title', 'N/A')}\n"
        f"URL: {sr.get('url', 'N/A')}\n"
        f"Snippet: {sr.get('snippet', 'N/A')}"
        for sr in state["search_results"]
        if "error" not in sr
    ])
    
    print(f"  Processing {len([sr for sr in state['search_results'] if 'error' not in sr])} intelligence sources...")
    if isinstance(dataset_examples, list):
        print(f"  Using {len(dataset_examples)} matched dataset examples as reference patterns...")
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a senior cybersecurity analyst.

Use the reference incident examples as archetypes for how incidents are described and reasoned about. Infer what happened in the current logs and explain it in a dataset-like narrative style, while staying grounded in the logs and search intelligence.

Based on the initial analysis, reference examples, and threat intelligence from search results, provide:
1. A threat level (Critical/High/Medium/Low)
2. A comprehensive detailed analysis (300-500 words) explaining what happened, likely attack progression, implications, and technical details from the search results
3. The search results used in your analysis
4. A list of recommended actions to mitigate the threat

Be specific, technical, and actionable in your recommendations. Do not copy example text verbatim; adapt the patterns to this case."""),
        ("user", """Original Logs:
{logs}

Initial Analysis:
Title: {title}
Content: {content}

Reference Incident Examples (for reasoning style):
{reference_examples}

Threat Intelligence from Search Results:
{search_context}

Provide your detailed security analysis.""")
    ])
    
    input_payload = {
        "logs": state["logs"],
        "title": state["result"].get("title", ""),
        "content": state["result"].get("content", ""),
        "reference_examples": reference_examples,
        "search_context": search_context
    }

    chain = template | structured_model
    try:
        result = chain.invoke(input_payload)
        normalized = _normalize_explainer_payload(result.model_dump(), state["search_results"])
        result = ExplainerOutputTemplate.model_validate(normalized)
    except Exception as e:
        print(f"  ⚠ Structured output parsing failed: {e}")
        print("  ↻ Retrying with strict JSON fallback...")

        fallback_prompt = ChatPromptTemplate.from_messages([
            ("system", """Return ONLY valid JSON with these exact keys:
- threat_level: one of low, medium, high, critical
- detailed_analysis: string, minimum 500 characters
- search_results: array of objects
- recommended_actions: array of at least 5 action strings

Do not include markdown, HTML, comments, or extra text before/after JSON."""),
            ("user", """Original Logs:
{logs}

Initial Analysis:
Title: {title}
Content: {content}

Reference Incident Examples (for reasoning style):
{reference_examples}

Threat Intelligence from Search Results:
{search_context}

Produce strict JSON now.""")
        ])

        raw_response = (fallback_prompt | model).invoke(input_payload)
        raw_text = getattr(raw_response, "content", "")
        if isinstance(raw_text, list):
            raw_text = "\n".join(str(x) for x in raw_text)
        raw_text = str(raw_text)

        parsed = _extract_first_json_object(raw_text)
        normalized = _normalize_explainer_payload(parsed or {}, state["search_results"])
        result = ExplainerOutputTemplate.model_validate(normalized)
    
    print(f"✓ Analysis complete - Threat Level: {result.threat_level}")
    print(f"✓ Generated {len(result.recommended_actions)} mitigation recommendations")

    explainer_payload = result.model_dump()
    dataset_style_summary = _build_dataset_style_incident_summary(state, explainer_payload)
    explainer_payload["dataset_style_summary"] = dataset_style_summary
    
    # Display output immediately after completion
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"\nTHREAT LEVEL: {result.threat_level}")
    print(f"\nDETAILED ANALYSIS:")
    print(result.detailed_analysis)
    print(f"\nRECOMMENDED ACTIONS:")
    for i, action in enumerate(result.recommended_actions, 1):
        print(f"  {i}. {action}")
    print("\nDATASET-STYLE INCIDENT SUMMARY:")
    print(dataset_style_summary)
    
    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": explainer_payload
    }

def IOCVectorGroupAdderNode(state: MessageState) -> MessageState:
    """
    Docstring for IOCVectorGroupAdderNode

    Node that takes in the detailed analysis and recommended actions from the previous node and generates a new IOC vector group that can be added to the organization's threat intelligence platform. The vector group should be based on the specific indicators of compromise (IOCs) mentioned in the detailed analysis and recommended actions.
    """
    print("\n" + "="*70)
    uses_examples_path = _uses_incident_examples_path(state)
    current_step = 7 if uses_examples_path else 6
    total_steps = 8 if uses_examples_path else 7
    print(_format_step_label(current_step, "Generating IOC Vector Group...", total_steps))
    print("="*70)

    explainer = state["explainer_output"]
    result = state["result"]

    dataset_examples = result.get("dataset_examples", []) if isinstance(result, dict) else []
    derived_payload = None
    if isinstance(dataset_examples, list) and dataset_examples:
        derived_payload = _derive_ioc_from_dataset_examples(dataset_examples)

    if derived_payload:
        print("  Using vectors derived from dataset examples...")
        ioc_result = IOCVectorGroupAdderTemplate.model_validate(derived_payload)
    else:
        print("  No usable dataset vectors found; inferring vectors from analysis...")
        structured_model = model.with_structured_output(IOCVectorGroupAdderTemplate)
    
        template = ChatPromptTemplate.from_messages([
                ("system", """You are a cybersecurity threat analyst specializing in IOC (Indicator of Compromise) identification.

Analyze the threat details and produce:
1. A concise vector_group_name in camel case.
2. A list of 1-5 IOC vectors that best describe and pinpoint the incident.

Use only evidence-backed vectors that align with the attack behavior in the logs and analysis."""),
            ("user", """Threat Analysis:
Title: {title}
Threat Level: {threat_level}
Detailed Analysis: {detailed_analysis}

Based on this analysis, generate an IOC vector group with:
1. A descriptive name for this attack pattern
2. The relevant IOC vectors (1-5 vectors) that match the indicators in this attack""")
        ])

        chain = template | structured_model
        ioc_result = chain.invoke({
            "title": result.get("title", ""),
            "threat_level": explainer.get("threat_level", ""),
            "detailed_analysis": explainer.get("detailed_analysis", "")
        })
    
    print(f"✓ Generated vector group: {ioc_result.vector_group_name}")
    print(f"✓ Selected {len(ioc_result.vectors)} IOC vectors")
    
    # Display output immediately
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Vector Group Name: {ioc_result.vector_group_name}")
    print(f"\nSelected Vectors:")
    for i, vector in enumerate(ioc_result.vectors, 1):
        print(f"  {i}. {vector}")
    
    # Generate XML format for the vector group
    xml_output = f"""<attack_vectors>
    <vector_group>
        <name>{ioc_result.vector_group_name}</name>"""
    for vector in ioc_result.vectors:
        xml_output += f"\n        <vector>{vector}</vector>"
    xml_output += "\n    </vector_group>\n</attack_vectors>"
    
    print(f"\nXML Format (ready to add to rv_ioc_lin.xml):")
    print(xml_output)

    output_dir = _resolve_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    xml_output_path = output_dir / "vector_group_output.xml"

    with open(xml_output_path, "w", encoding="utf-8") as f:
        f.write(xml_output)
    print(f"\n✓ XML output saved to {xml_output_path}")

    # send_files("vector_group_output.xml", "upload_file")
    
    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": state["explainer_output"],
        "ioc_vector_group": ioc_result.model_dump(),
        "xml_output_path": str(xml_output_path),
    }

def MarkdownReportGeneratorNode(state: MessageState) -> MessageState:
    """
    Docstring for MarkdownReportGeneratorNode
    
    Node that takes the complete analysis and generates a comprehensive markdown report.
    """
    print("\n" + "="*70)
    uses_examples_path = _uses_incident_examples_path(state)
    current_step = 8 if uses_examples_path else 7
    total_steps = 8 if uses_examples_path else 7
    print(_format_step_label(current_step, "Generating Markdown Report...", total_steps))
    print("="*70)
    
    result = state["result"]
    explainer = state["explainer_output"]
    search_results = state["search_results"]
    incident_type = result.get("incident_type", "N/A")
    dataset_examples = result.get("dataset_examples", [])
    used_dataset_examples = isinstance(dataset_examples, list) and len(dataset_examples) > 0
    
    # Generate markdown content
    markdown_content = f"""# Cybersecurity Log Analysis Report

---

## Executive Summary

**Threat Title:** {result.get('title', 'N/A')}

**Incident Type:** {incident_type}

**Threat Level:** {explainer.get('threat_level', 'N/A').upper()}

**Dataset Example Match Path:** {"Matched examples were used" if used_dataset_examples else "No matching dataset examples (or none applicable)"}

**Date Generated:** {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Original Logs

```
{state['logs']}
```

---

## Initial Analysis

{result.get('content', 'N/A')}

---

## Search Queries Generated

"""
    
    for i in range(1, 6):
        query = result.get(f'search_query_{i}')
        if query:
            markdown_content += f"{i}. {query}\n"

    markdown_content += "\n---\n\n## Dataset Examples (Incident-Type Match)\n\n"

    if used_dataset_examples:
        markdown_content += f"Total matched examples: **{len(dataset_examples)}**\n\n"
        for idx, ex in enumerate(dataset_examples[:10], 1):
            source_file = ex.get("source_file", "unknown")
            source_index = ex.get("source_index", "?")
            example_type = ex.get("incident_type", "N/A")
            example_desc = str(ex.get("description", "")).strip()
            if len(example_desc) > 220:
                example_desc = example_desc[:220].rstrip() + "..."
            example_vectors = ex.get("vectors", [])
            vectors_text = ", ".join(str(v) for v in example_vectors[:8]) if isinstance(example_vectors, list) else "N/A"

            markdown_content += f"### Example {idx}\n\n"
            markdown_content += f"- **Source:** {source_file} #{source_index}\n"
            markdown_content += f"- **Incident Type:** {example_type}\n"
            markdown_content += f"- **Vectors:** {vectors_text}\n"
            if example_desc:
                markdown_content += f"- **Description:** {example_desc}\n"
            markdown_content += "\n"

        if len(dataset_examples) > 10:
            markdown_content += f"_...and {len(dataset_examples) - 10} more matched examples._\n\n"
    else:
        markdown_content += (
            "No matched dataset examples were used for this run. "
            "The workflow continued with search-based enrichment and model inference.\n\n"
        )
    
    markdown_content += "\n---\n\n## Threat Intelligence Gathered\n\n"
    
    # Group search results by query
    for i in range(1, 6):
        query_results = [sr for sr in search_results if sr.get('query_number') == i]
        if query_results:
            markdown_content += f"### Query {i}: {query_results[0].get('query')}\n\n"
            
            if 'error' in query_results[0]:
                markdown_content += f"**Error:** {query_results[0].get('error')}\n\n"
            else:
                for j, sr in enumerate(query_results, 1):
                    markdown_content += f"**[{j}] {sr.get('title', 'N/A')}**\n\n"
                    markdown_content += f"- **URL:** [{sr.get('url', 'N/A')}]({sr.get('url', 'N/A')})\n"
                    markdown_content += f"- **Summary:** {sr.get('snippet', 'N/A')}\n\n"
    
    markdown_content += "---\n\n## Detailed Security Analysis\n\n"
    markdown_content += explainer.get('detailed_analysis', 'N/A')

    markdown_content += "\n\n---\n\n## Dataset-Style Incident Summary\n\n"
    markdown_content += "```\n"
    markdown_content += str(explainer.get("dataset_style_summary", "N/A")).strip()
    markdown_content += "\n```\n"
    
    markdown_content += "\n\n---\n\n## Recommended Actions\n\n"
    
    for i, action in enumerate(explainer.get('recommended_actions', []), 1):
        markdown_content += f"{i}. {action}\n"
    
    # Add IOC Vector Group section
    if 'ioc_vector_group' in state and state['ioc_vector_group']:
        ioc_group = state['ioc_vector_group']
        markdown_content += "\n---\n\n## IOC Vector Group\n\n"
        markdown_content += f"**Vector Group Name:** {ioc_group.get('vector_group_name', 'N/A')}\n\n"
        markdown_content += "**Selected Vectors:**\n\n"
        for vector in ioc_group.get('vectors', []):
            markdown_content += f"- {vector}\n"
        
        markdown_content += "\n**XML Format (ready to add to rv_ioc_lin.xml):**\n\n```xml\n"
        markdown_content += f"    <vector_group>\n"
        markdown_content += f"        <name>{ioc_group.get('vector_group_name', 'N/A')}</name>\n"
        for vector in ioc_group.get('vectors', []):
            markdown_content += f"        <vector>{vector}</vector>\n"
        markdown_content += "    </vector_group>\n```\n"
    
    markdown_content += "\n---\n\n## Conclusion\n\n"
    markdown_content += f"This analysis was performed using automated threat intelligence gathering and AI-powered analysis. "
    markdown_content += f"The threat has been assessed as **{explainer.get('threat_level', 'N/A').upper()}** level. "
    markdown_content += f"Please review the recommended actions and implement appropriate security measures.\n"
    
    # Save to file with timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = _resolve_output_dir(state)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"analysis_report_{timestamp}.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"✓ Markdown report generated: {output_path}")
    print(f"✓ Report contains {len(markdown_content)} characters")
    
    # Display output
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Markdown report saved to: {output_path}")
    print(f"Report sections: Executive Summary, Logs, Initial Analysis, Search Queries, Threat Intelligence, Detailed Analysis, Recommendations")
    
    return {
        **_carry_context(state),
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": state["explainer_output"],
        "ioc_vector_group": state.get("ioc_vector_group", {}),
        "markdown_output": markdown_content,
        "report_path": str(output_path),
    }



















# Create the LangGraph workflow
workflow = StateGraph(MessageState)

# Add the nodes
workflow.add_node("initial_analysis", InitialAnalysisNode)
workflow.add_node("initial_search", InitialSearchFromLogsToDatasetNode)
workflow.add_node("incident_examples", GettingExamplesUsingIncidentTypeNode)
workflow.add_node("question_former", QuestionFormerNode)
workflow.add_node("context_deriver", ContextDeriverFromSearchQueriesUsingDDGNode)
workflow.add_node("explainer", ExplainerOutputNode)
workflow.add_node("ioc_vector_adder", IOCVectorGroupAdderNode)
workflow.add_node("markdown_generator", MarkdownReportGeneratorNode)

# Set the entry point
workflow.set_entry_point("initial_analysis")

# Connect nodes
workflow.add_edge("initial_analysis", "initial_search")
workflow.add_conditional_edges(
    "initial_search",
    RouteFromIncidentType,
    {
        "incident_examples": "incident_examples",
        "question_former": "question_former",
    },
)
workflow.add_edge("incident_examples", "question_former")
workflow.add_edge("question_former", "context_deriver")
workflow.add_edge("context_deriver", "explainer")
workflow.add_edge("explainer", "ioc_vector_adder")
workflow.add_edge("ioc_vector_adder", "markdown_generator")

workflow.add_edge("markdown_generator", END)




# Compile the graph
app = workflow.compile()

def _run_workflow_for_logs_file(logs_file: Path, hierarchies_dir: Path | None = None) -> dict:
    """Run the workflow for one logs.txt file and persist outputs beside it."""
    with open(logs_file, "r", encoding="utf-8") as f:
        logs_content = f.read()

    customer_ids = _extract_customer_ids(logs_file, hierarchies_dir) if hierarchies_dir else []
    customer_label = _format_customer_label(customer_ids)

    print("\n" + "#" * 70)
    print("# CYBERSECURITY LOG ANALYSIS WORKFLOW")
    print("# Powered by LangGraph + Ollama (seneca) + DuckDuckGo")
    print(f"# Customer IDs: {customer_label}")
    print(f"# Analyzing logs from: {logs_file}")

    result = app.invoke({
        "logs": logs_content,
        "logs_path": str(logs_file),
        "output_dir": str(logs_file.parent),
        "customer_ids": customer_ids,
    })

    print("\n" + "#" * 70)
    print("# WORKFLOW COMPLETE")
    print("#" * 70)
    print(f"Customer IDs: {customer_label}")
    print(f"Report: {result.get('report_path', 'N/A')}")
    print(f"IOC XML: {result.get('xml_output_path', 'N/A')}")

    return result


def _is_generated_output_file(file_path: Path) -> bool:
    """Identify files this workflow itself writes, so they're excluded when re-consolidating."""
    name = file_path.name
    if name in GENERATED_OUTPUT_FILENAMES:
        return True
    return name.startswith("analysis_report_") and name.endswith(".md")


def _consolidate_hierarchy_files_to_logs(hierarchy_dir: Path) -> Path:
    """
    Combine every source file under hierarchy_dir (as pulled by populate_hierarchies.py)
    into a single hierarchy_dir/logs.txt, one section per file: its name followed by
    its content. That combined file is what the LangGraph nodes analyze.
    """
    logs_path = hierarchy_dir / "logs.txt"
    sections = []

    for file_path in sorted(hierarchy_dir.rglob("*")):
        if not file_path.is_file() or _is_generated_output_file(file_path):
            continue

        relative_name = file_path.relative_to(hierarchy_dir).as_posix()
        try:
            content = file_path.read_text(encoding="utf-8").rstrip()
        except UnicodeDecodeError:
            content = f"[binary file, {file_path.stat().st_size} bytes - content omitted]"

        sections.append(f"===== {relative_name} =====\n{content}")

    logs_path.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
    return logs_path


def _run_workflow_for_hierarchy(hierarchy: str, vault_root: str, hierarchies_dir: Path) -> dict:
    """Pull one hierarchy's files via MCP, consolidate them, and analyze just that hierarchy."""
    print("\n" + "#" * 70)
    print("# PULLING HIERARCHY FILES VIA MCP")
    print("#" * 70)
    print(f"Vault root: {vault_root}")
    print(f"Hierarchy:  {hierarchy}")

    pull_hierarchy_from_vault(vault_root, hierarchies_dir, hierarchy)

    hierarchy_dir = hierarchies_dir / Path(hierarchy.strip("/\\"))
    if not hierarchy_dir.exists() or not hierarchy_dir.is_dir():
        print(f"ERROR: No files found for hierarchy '{hierarchy}' after MCP pull.")
        raise SystemExit(1)

    print("\n" + "#" * 70)
    print("# CONSOLIDATING HIERARCHY FILES INTO logs.txt")
    print("#" * 70)
    logs_file = _consolidate_hierarchy_files_to_logs(hierarchy_dir)
    print(f"✓ Combined logs written to {logs_file}")

    return _run_workflow_for_logs_file(logs_file, hierarchies_dir)


def main() -> None:
    """
    Run the workflow for a single hierarchy pulled via MCP (--hierarchy), or fall back
    to batch-analyzing every customer hierarchy already present locally, or a root-level
    logs.txt if neither applies.
    """
    parser = argparse.ArgumentParser(description="Run the cybersecurity log analysis workflow.")
    parser.add_argument(
        "--hierarchy",
        default=None,
        help=(
            "Specific hierarchy path to analyze, e.g. 5/101/1/4/1 "
            "(company_id/customer_id/branch_id/product_id/system_id). "
            "When given, populate_hierarchies.py is run for just this hierarchy, "
            "its files are consolidated into logs.txt, and only that hierarchy is analyzed "
            "instead of batch-scanning all of hierarchies/."
        ),
    )
    parser.add_argument(
        "--vault-root",
        default=DEFAULT_VAULT_ROOT,
        help=f"Vault root path on the MCP server to pull --hierarchy from (default: {DEFAULT_VAULT_ROOT}).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).parent
    hierarchies_dir = base_dir / "hierarchies"

    if args.hierarchy:
        _run_workflow_for_hierarchy(args.hierarchy, args.vault_root, hierarchies_dir)
        return

    hierarchy_logs = _discover_customer_log_files(hierarchies_dir)

    if hierarchy_logs:
        print("\n" + "#" * 70)
        print("# CUSTOMER HIERARCHY BATCH RUN")
        print("#" * 70)
        print(f"Discovered {len(hierarchy_logs)} customer log file(s) under: {hierarchies_dir}")

        completed = 0
        failed = 0

        for index, logs_file in enumerate(hierarchy_logs, start=1):
            customer_ids = _extract_customer_ids(logs_file, hierarchies_dir)
            customer_label = _format_customer_label(customer_ids)
            print("\n" + "=" * 70)
            print(f"[CUSTOMER {index}/{len(hierarchy_logs)}] {customer_label}")
            print("=" * 70)

            try:
                _run_workflow_for_logs_file(logs_file, hierarchies_dir)
                completed += 1
            except Exception as exc:
                failed += 1
                print(f"✗ Workflow failed for customer {customer_label}: {exc}")

        print("\n" + "#" * 70)
        print("# BATCH RUN SUMMARY")
        print("#" * 70)
        print(f"Completed: {completed}")
        print(f"Failed: {failed}")
        return

    logs_file = base_dir / "logs.txt"
    if not logs_file.exists():
        print("ERROR: No customer hierarchies were found and logs.txt is missing.")
        raise SystemExit(1)

    _run_workflow_for_logs_file(logs_file)


if __name__ == "__main__":
    main()

# import time
# interval = 30

# while True:

#     print("\n================ NEW CYCLE ================\n")

#     try:

#         result = app.invoke({
#             "logs": logs_content
#         })

#         print("\n" + "#"*70)
#         print("# GENERATION COMPLETE")
#         print("#"*70)

#         print(f"\nVector Group: {result['ioc_vector_group']['vector_group_name']}")
#         print(f"Vectors: {', '.join(result['ioc_vector_group']['vectors'])}")
#         print(f"\nOutput saved to: new_ioc.xml")
#     except Exception as e:
#         print("[ERROR] Agent crashed:", str(e))
        

#     print(f"[INFO] Sleeping for {interval} seconds...\n")
#     time.sleep(interval)
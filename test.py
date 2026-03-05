from typing import Literal, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from ddgs import DDGS
import asyncio
import os
import httpx
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

model = ChatOllama(model="seneca")

class MessageState(TypedDict):
    logs: str
    result: dict
    search_results: List[dict]
    explainer_output: dict
    ioc_vector_group: dict
    markdown_output: str




class QuestionFormerOutputTemplate(BaseModel):
    title: str = Field(description="An appropriate title of the attack or potential attack after analyzing the logs")
    content: str = Field(description="A 100-200 word initial analysis of the attack or potential attack after analyzing the logs")
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
    vector_group_name: str = Field(description="The name of the IOC vector group to add in camel case format (e.g. 'SuspiciousProcessAndFileChanges')")
    vectors: List[Literal[
        "Ram", 
        "Disk", 
        "Process", 
        "banip", 
        "Unbinary", 
        "MorefilesChanges", 
        "DB_Breach", 
        "DB_Delete", 
        "DB_Modify", 
        "Lib", 
        "Bin", 
        "Malware", 
        "Userbreach", 
        "Ransom", 
        "Honeypot"]] = Field(description="List of IOC vectors from the valid set that are relevant to this attack", min_length=1)

def QuestionFormerNode(state: MessageState) -> MessageState:
    """
    Docstring for QuestionFormerNode
    
    Node that takes in logs and returns a title, an initial analysis, and 5 search queries to find more information about the attack or potential attack.
    """
    print("\n" + "="*70)
    print("[STEP 1/4] Analyzing logs and generating search queries...")
    print("="*70)
    
    # Use with_structured_output for more reliable structured responses
    structured_model = model.with_structured_output(QuestionFormerOutputTemplate)
    
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a cybersecurity analyst. Analyze the following logs and determine if there is an attack or potential attack. Provide an appropriate title, a 100-200 word initial analysis, and 5 search queries to find more information about the attack or potential attack. Ignore the logs that states file does not exist or cannot be found. Focus on the logs that indicate potential security incidents."),
        ("user", "{logs}")
    ])
    
    chain = template | structured_model
    result = chain.invoke({"logs": state["logs"]})
    
    print(f"✓ Generated title: {result.title}")
    print(f"✓ Generated {len([q for q in [result.search_query_1, result.search_query_2, result.search_query_3, result.search_query_4, result.search_query_5] if q])} search queries")
    
    # Display output immediately after completion
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Title: {result.title}")
    print(f"\nInitial Analysis:\n{result.content}")
    print(f"\nSearch Queries Generated:")
    for i, query in enumerate([result.search_query_1, result.search_query_2, result.search_query_3, result.search_query_4, result.search_query_5], 1):
        if query:
            print(f"  {i}. {query}")
    
    return {"logs": state["logs"], "result": result.model_dump()}


def ContextDeriverFromSearchQueriesUsingDDGNode(state: MessageState) -> MessageState:
    """
    Docstring for ContextDeriverFromSearchQueriesUsingDDGNode
    
    Node that takes in the search queries from the previous node and uses them to search on DuckDuckGo to find more information about the attack or potential attack. The results are then used to derive more context about the attack or potential attack.
    """
    print("\n" + "="*70)
    print("[STEP 2/4] Gathering threat intelligence from DuckDuckGo...")
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
    print("[STEP 3/4] Generating comprehensive security analysis...")
    print("="*70)
    
    structured_model = model.with_structured_output(ExplainerOutputTemplate)
    
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
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a senior cybersecurity analyst. Based on the initial analysis and threat intelligence from search results, provide:
1. A threat level (Critical/High/Medium/Low)
2. A comprehensive detailed analysis (300-500 words) explaining the attack, its implications, and technical details from the search results
3. The search results used in your analysis
4. A list of recommended actions to mitigate the threat

Be specific, technical, and actionable in your recommendations."""),
        ("user", """Original Logs:
{logs}

Initial Analysis:
Title: {title}
Content: {content}

Threat Intelligence from Search Results:
{search_context}

Provide your detailed security analysis.""")
    ])
    
    chain = template | structured_model
    result = chain.invoke({
        "logs": state["logs"],
        "title": state["result"].get("title", ""),
        "content": state["result"].get("content", ""),
        "search_context": search_context
    })
    
    print(f"✓ Analysis complete - Threat Level: {result.threat_level}")
    print(f"✓ Generated {len(result.recommended_actions)} mitigation recommendations")
    
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
    
    return {
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": result.model_dump()
    }

def IOCVectorGroupAdderNode(state: MessageState) -> MessageState:
    """
    Docstring for IOCVectorGroupAdderNode

    Node that takes in the detailed analysis and recommended actions from the previous node and generates a new IOC vector group that can be added to the organization's threat intelligence platform. The vector group should be based on the specific indicators of compromise (IOCs) mentioned in the detailed analysis and recommended actions.
    """
    print("\n" + "="*70)
    print("[STEP 4.5/5] Generating IOC Vector Group...")
    print("="*70)
    
    structured_model = model.with_structured_output(IOCVectorGroupAdderTemplate)
    
    explainer = state["explainer_output"]
    result = state["result"]
    5/5
    print("  Analyzing threat patterns and indicators...")
    
    template = ChatPromptTemplate.from_messages([
        ("system", """You are a cybersecurity threat analyst specializing in IOC (Indicator of Compromise) identification.
        
Available IOC vectors with detailed descriptions:

1. Ram - High CPU/Memory usage indicator
   Use when: System shows abnormally high CPU or memory consumption, resource exhaustion attacks, cryptomining, memory leaks, or resource-intensive malicious processes.

2. Disk - Disk space or I/O anomalies
   Use when: Low disk space conditions, unusual disk activity, excessive writes/reads, disk filling attacks, or storage manipulation detected.

3. Process - Abnormal process behavior
   Use when: Unusual process count, suspicious process names, unauthorized processes running, process injection, or abnormal parent-child process relationships.

4. banip - Failed authentication and IP blocking
   Use when: Multiple failed login attempts, brute force attacks detected, IPs being banned, credential stuffing, or authentication abuse patterns.

5. Unbinary - Unauthorized binary detection
   Use when: New unknown binaries detected, unauthorized executables created, suspicious compiled files, or unexpected binary modifications.

6. MorefilesChanges - Extensive file system modifications
   Use when: Large numbers of files created/modified/deleted, mass file encryption (ransomware), file system tampering, or unusual file operation patterns.

7. DB_Breach - Database breach or unauthorized access
   Use when: Unauthorized database access detected, SQL injection attempts, data exfiltration from databases, or database authentication bypass.

8. DB_Delete - Database deletion activity
   Use when: Unauthorized deletion of database records, DROP operations, data destruction attacks, or database sabotage attempts.

9. DB_Modify - Database modification activity
   Use when: Unauthorized changes to database contents, UPDATE/ALTER operations, data manipulation, or database integrity compromise.

10. Lib - Library file tampering
    Use when: System library modifications, shared library (.so/.dll) tampering, library injection attacks, or compromised system libraries.

11. Bin - Binary file tampering
    Use when: System binary modifications, core executable tampering, binary replacement attacks, or critical system file modifications.

12. Malware - Malware detection
    Use when: Known malware signatures detected, malicious software identified, virus/trojan/worm presence, or malware behavior patterns observed.

13. Userbreach - User account compromise
    Use when: User account takeover, compromised credentials, unauthorized user actions, privilege escalation via user accounts, or account abuse.

14. Ransom - Ransomware indicators
    Use when: File encryption patterns, ransom notes detected, ransomware behavior, mass file extension changes, or extortion attempts.

15. Honeypot - Honeypot trigger detection
    Use when: Honeypot systems accessed, trap mechanisms triggered, attacker interaction with decoy systems, or bait file access detected.

SELECTION GUIDELINES:
- Choose vectors that when considered together indicate the most likely attack pattern based on the detailed analysis and recommended actions.
- Prioritize vectors with the strongest evidence in the logs and analysis
- Consider attack chain progression (e.g., Userbreach → Process → MorefilesChanges)
- Combine vectors that represent coordinated attack patterns
- Be specific - only select vectors with clear supporting evidence

Analyze the threat and select 1-5 relevant vectors that match the attack pattern.
Create an appropriate vector group name that describes the attack combination."""),
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
    xml_output = f"""    <vector_group>
        <name>{ioc_result.vector_group_name}</name>"""
    for vector in ioc_result.vectors:
        xml_output += f"\n        <vector>{vector}</vector>"
    xml_output += "\n    </vector_group>"
    
    print(f"\nXML Format (ready to add to rv_ioc_lin.xml):")
    print(xml_output)
    
    return {
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": state["explainer_output"],
        "ioc_vector_group": ioc_result.model_dump()
    }

def MarkdownReportGeneratorNode(state: MessageState) -> MessageState:
    """
    Docstring for MarkdownReportGeneratorNode
    
    Node that takes the complete analysis and generates a comprehensive markdown report.
    """
    print("\n" + "="*70)
    print("[STEP 4/4] Generating Markdown Report...")
    print("="*70)
    
    result = state["result"]
    explainer = state["explainer_output"]
    search_results = state["search_results"]
    
    # Generate markdown content
    markdown_content = f"""# Cybersecurity Log Analysis Report

---

## Executive Summary

**Threat Title:** {result.get('title', 'N/A')}

**Threat Level:** {explainer.get('threat_level', 'N/A').upper()}

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
    timestamp = __import__('datetime').datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_filename = f"analysis_report_{timestamp}.md"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    
    print(f"✓ Markdown report generated: {output_filename}")
    print(f"✓ Report contains {len(markdown_content)} characters")
    
    # Display output
    print("\n" + "-"*70)
    print("NODE OUTPUT:")
    print("-"*70)
    print(f"Markdown report saved to: {output_filename}")
    print(f"Report sections: Executive Summary, Logs, Initial Analysis, Search Queries, Threat Intelligence, Detailed Analysis, Recommendations")
    
    return {
        "logs": state["logs"],
        "result": state["result"],
        "search_results": state["search_results"],
        "explainer_output": state["explainer_output"],
        "ioc_vector_group": state.get("ioc_vector_group", {}),
        "markdown_output": markdown_content
    }



















# Create the LangGraph workflow
workflow = StateGraph(MessageState)

# Add the nodes
workflow.add_node("question_former", QuestionFormerNode)
workflow.add_node("context_deriver", ContextDeriverFromSearchQueriesUsingDDGNode)
workflow.add_node("explainer", ExplainerOutputNode)
workflow.add_node("ioc_vector_adder", IOCVectorGroupAdderNode)
workflow.add_node("markdown_generator", MarkdownReportGeneratorNode)

# Set the entry point
workflow.set_entry_point("question_former")

# Connect nodes
workflow.add_edge("question_former", "context_deriver")
workflow.add_edge("context_deriver", "explainer")
workflow.add_edge("explainer", "ioc_vector_adder")
workflow.add_edge("ioc_vector_adder", "markdown_generator")

workflow.add_edge("markdown_generator", END)




# Compile the graph
app = workflow.compile()

# Read logs from file
logs_file = Path(__file__).parent / "logs.txt"
if logs_file.exists():
    with open(logs_file, 'r', encoding='utf-8') as f:
        logs_content = f.read()
else:
    print("ERROR: logs.txt not found. Please ensure logs.txt exists in the current directory.")
    exit(1)

print("\n" + "#"*70)
print("# CYBERSECURITY LOG ANALYSIS WORKFLOW")
print("# Powered by LangGraph + Ollama (seneca) + DuckDuckGo")
print(f"# Analyzing logs from: {logs_file}")
result = app.invoke({
    "logs": logs_content
})

print("\n" + "#"*70)
print("# WORKFLOW COMPLETE")
print("#"*70)


print(result)
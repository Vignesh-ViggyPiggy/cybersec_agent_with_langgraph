from typing import Literal, TypedDict, List
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from ddgs import DDGS

model = ChatOllama(model="seneca")

class MessageState(TypedDict):
    logs: str
    result: dict
    search_results: List[dict]
    explainer_output: dict
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
    detailed_analysis: str = Field(description="A more detailed analysis of the attack or potential attack based on the search results")
    search_results: List[dict] = Field(description="The search results used to derive the detailed analysis")
    recommended_actions: List[str] = Field(description="Recommended actions to mitigate the attack or potential attack based on the detailed analysis")


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
        ("system", "You are a cybersecurity analyst. Analyze the following logs and determine if there is an attack or potential attack. Provide an appropriate title, a 100-200 word initial analysis, and 5 search queries to find more information about the attack or potential attack."),
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
    
    return {"logs": state["logs"], "result": result.dict()}


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
        "explainer_output": result.dict()
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
        "markdown_output": markdown_content
    }



















# Create the LangGraph workflow
workflow = StateGraph(MessageState)

# Add the nodes
workflow.add_node("question_former", QuestionFormerNode)
workflow.add_node("context_deriver", ContextDeriverFromSearchQueriesUsingDDGNode)
workflow.add_node("explainer", ExplainerOutputNode)
workflow.add_node("markdown_generator", MarkdownReportGeneratorNode)

# Set the entry point
workflow.set_entry_point("question_former")

# Connect nodes
workflow.add_edge("question_former", "context_deriver")
workflow.add_edge("context_deriver", "explainer")
workflow.add_edge("explainer", "markdown_generator")
workflow.add_edge("markdown_generator", END)




# Compile the graph
app = workflow.compile()

print("\n" + "#"*70)
print("# CYBERSECURITY LOG ANALYSIS WORKFLOW")
print("# Powered by LangGraph + Ollama (seneca) + DuckDuckGo")
print("#"*70)

# Read logs from file
with open("cybersec_agent_with_langgraph/logs.txt", "r") as f:
    logs_content = f.read()

# Run the graph
result = app.invoke({
    "logs": logs_content
})

print("\n" + "#"*70)
print("# WORKFLOW COMPLETE")
print("#"*70)


print(result)
# Cybersecurity Log Analysis Agent with LangGraph

An intelligent automated cybersecurity log analysis system that uses LangGraph workflows and Large Language Models to analyze system logs, identify security threats, gather threat intelligence, and generate comprehensive security reports with actionable recommendations.

## Overview

This project implements an AI-powered security analyst that automatically:
- Collects and monitors system logs from remote Linux servers
- Analyzes logs for security incidents and anomalies
- Generates contextual search queries for threat intelligence gathering
- Searches DuckDuckGo for relevant security information
- Performs comprehensive threat analysis with risk assessment
- Generates IOC (Indicator of Compromise) vector groups
- Produces detailed markdown security reports

## Key Features

### 🔍 Automated Log Analysis
- Fetches logs from remote Linux systems via HTTP API
- Detects changes and updates only when new log entries appear
- Analyzes multiple log sources (audit, secure, syslog, etc.)

### 🧠 AI-Powered Threat Detection
- Uses Ollama's Seneca model for intelligent analysis
- Identifies attack patterns and security incidents
- Generates appropriate threat titles and initial assessments

### 🌐 Threat Intelligence Gathering
- Automatically generates 5 targeted search queries per incident
- Searches DuckDuckGo for relevant security information
- Correlates external threat intelligence with local findings

### 📊 Comprehensive Analysis
- Provides detailed 800-1000 word technical security analysis
- Assigns threat levels (Critical/High/Medium/Low)
- Includes minimum 5 actionable security recommendations
- Validates output to ensure complete analysis every time

### 🎯 IOC Vector Group Generation
- Identifies relevant Indicators of Compromise (IOCs)
- Maps incidents to 15 predefined IOC vectors:
  - Ram, Disk, Process, banip, Unbinary, MorefilesChanges
  - DB_Breach, DB_Delete, DB_Modify, Lib, Bin
  - Malware, Userbreach, Ransom, Honeypot
- Generates XML format ready for threat intelligence platforms

### 📝 Professional Reports
- Generates timestamped markdown reports
- Includes executive summary, logs, analysis, and recommendations
- Ready-to-use IOC vector groups in XML format
- Suitable for security team distribution

## Architecture

The system uses a LangGraph workflow with 6 sequential nodes:

```
┌─────────────────────┐
│  LogFileUpdater     │ ← Fetches logs from remote server
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  QuestionFormer     │ ← Analyzes logs, generates search queries
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  ContextDeriver     │ ← Searches DuckDuckGo for threat intel
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  ExplainerOutput    │ ← Detailed analysis & recommendations
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  IOCVectorAdder     │ ← Generates IOC vector groups
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  MarkdownGenerator  │ ← Creates final report
└─────────────────────┘
```

## Requirements

### Dependencies
```
langchain_ollama
langchain_core
pydantic
langgraph
duckduckgo-search (ddgs)
httpx
python-dotenv
```

### Prerequisites
- Python 3.8+
- Ollama installed with Seneca model
- Remote log collection server (optional - will use cached logs if unavailable)

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd cybersec_agent_with_langgraph
   ```

2. **Install dependencies:**
   ```bash
   pip install langchain_ollama langchain_core pydantic langgraph duckduckgo-search httpx python-dotenv
   ```

3. **Install and setup Ollama:**
   ```bash
   # Install Ollama from https://ollama.ai
   ollama pull seneca
   ```

4. **Configure environment variables:**
   Create a `.env` file:
   ```env
   REMOTE_SERVER_URL=http://192.168.8.14:8000
   NUM_LINES=10
   ```

## Configuration

### Environment Variables

- `REMOTE_SERVER_URL`: URL of the remote log collection server (default: `http://localhost:8000`)
- `NUM_LINES`: Number of log lines to fetch per file (default: `10`)

### Supported Log Files

The system monitors these Linux log files:
- `/var/log/audit/audit.log` - Audit system logs
- `/var/log/secure` - Security/authentication logs (RHEL/CentOS)
- `/var/log/auth.log` - Authentication logs (Debian/Ubuntu)
- `/var/log/syslog` - General system logs (Debian/Ubuntu)
- `/var/log/messages` - General system logs (RHEL/CentOS)
- `/var/log/kern.log` - Kernel logs
- `/var/log/cron` - Cron job execution logs
- Web server logs (Apache/Nginx)
- Database logs (MySQL/PostgreSQL)
- `/var/log/fail2ban.log` - Intrusion prevention logs

## Usage

### Basic Execution

```bash
python test.py
```

### Workflow Execution

The system automatically:
1. **Checks for new logs** - Connects to remote server or uses cached logs
2. **Analyzes logs** - Identifies security incidents
3. **Generates search queries** - Creates 5 contextual queries
4. **Gathers intelligence** - Searches DuckDuckGo for threat information
5. **Performs analysis** - 800+ word detailed security analysis
6. **Creates IOC vectors** - Maps to predefined indicator types
7. **Generates report** - Saves markdown report with timestamp

### Output

Each execution produces:
- **Console output** - Step-by-step progress with summaries
- **logs.txt** - Cached log data with timestamp
- **analysis_report_YYYY-MM-DD_HH-MM-SS.md** - Comprehensive security report

### Sample Report Structure

```markdown
# Cybersecurity Log Analysis Report

## Executive Summary
- Threat Title
- Threat Level
- Date Generated

## Original Logs
[Complete log data]

## Initial Analysis
[100-200 word assessment]

## Search Queries Generated
[5 generated queries]

## Threat Intelligence Gathered
[DuckDuckGo search results per query]

## Detailed Security Analysis
[800-1000 word comprehensive analysis]

## Recommended Actions
[Minimum 5 specific actions]

## IOC Vector Group
[Vector group name and selected IOC vectors in XML format]

## Conclusion
[Summary and next steps]
```

## IOC Vector Descriptions

The system selects from 15 IOC vectors:

| Vector | Description |
|--------|-------------|
| **Ram** | High CPU/Memory usage, resource exhaustion attacks |
| **Disk** | Disk space anomalies, excessive I/O |
| **Process** | Abnormal process behavior, suspicious processes |
| **banip** | Failed authentication, brute force attacks |
| **Unbinary** | Unauthorized binary detection |
| **MorefilesChanges** | Extensive file system modifications |
| **DB_Breach** | Database breach or unauthorized access |
| **DB_Delete** | Database deletion activity |
| **DB_Modify** | Database modification activity |
| **Lib** | Library file tampering |
| **Bin** | Binary file tampering |
| **Malware** | Malware detection |
| **Userbreach** | User account compromise |
| **Ransom** | Ransomware indicators |
| **Honeypot** | Honeypot trigger detection |

## Validation & Error Handling

The system includes robust validation:
- **Minimum length requirements** - Ensures detailed_analysis is at least 500 characters
- **Minimum item requirements** - Guarantees at least 5 recommended actions
- **Automatic retry** - Regenerates incomplete analysis
- **Default recommendations** - Adds intelligent defaults if LLM fails
- **Fallback to cached logs** - Uses local logs if remote server unavailable

## Customization

### Changing the LLM Model

```python
model = ChatOllama(model="your-model-name")
```

### Adjusting Analysis Length

Modify the `min_length` parameter in `ExplainerOutputTemplate`:
```python
detailed_analysis: str = Field(..., min_length=800)
```

### Adding Custom IOC Vectors

Edit the `IOCVectorGroupAdderTemplate` Literal type and add descriptions in the system prompt.

## Troubleshooting

### "Failed to connect to server"
- Check if REMOTE_SERVER_URL is correct
- System will use cached logs.txt if available

### "detailed_analysis too short"
- System automatically retries with simplified prompt
- Check Ollama service is running

### Missing recommended_actions
- Validation adds default recommendations automatically
- Verify LLM model is responding correctly

## Future Enhancements

- [ ] Real-time log streaming
- [ ] Multi-server support
- [ ] Integration with SIEM platforms
- [ ] Custom IOC vector definitions
- [ ] Email/Slack notifications
- [ ] Historical trend analysis
- [ ] Machine learning-based anomaly detection

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bugs and feature requests.

## License

[Specify your license here]

## Author

Cybersecurity Log Analysis Agent - Powered by LangGraph, Ollama, and DuckDuckGo

---

**Last Updated:** March 2, 2026 

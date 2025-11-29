# 📊 Table Analysis Agent

An intelligent agent system for table structure understanding and data analysis. Based on a two-agent design pattern:

- **Agent0 (Structure)**: Table structure understanding - detects merged cells, headers, layout
- **Agent1 (Analysis)**: Data analysis & information retrieval - counts, values, calculations

## Project Structure

```
table_agent/
├── __init__.py      # Package initialization
├── tools.py         # Table parsing and analysis tools
├── prompts.py       # System prompts and templates  
├── agent.py         # Main TableAnalysisAgent class
├── main.py          # Entry point (CLI + Web UI)
├── sample_data.json # Sample test data
└── requirements.txt # Dependencies
```

## Installation

```bash
# Install dependencies
pip install pandas beautifulsoup4 lxml
```

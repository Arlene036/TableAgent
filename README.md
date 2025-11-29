# 📊 Table Analysis Agent

An intelligent agent system for table structure understanding and data analysis. Based on a two-agent design pattern:

- **Agent0 (Structure)**: Table structure understanding - detects merged cells, headers, layout
- **Agent1 (Analysis)**: Data analysis & information retrieval - counts, values, calculations

## Features

- 🔍 **Multi-format Support**: HTML tables, Markdown tables, Plain text tables
- 🧠 **Intelligent Query Classification**: Automatically routes queries to appropriate agent
- 📐 **Structure Analysis**: Detects merged cells, multi-row headers, table dimensions
- 📈 **Data Analysis**: Counting, value lookup, comparisons with ReAct reasoning
- 🌐 **Multiple Interfaces**: CLI, Web UI (Gradio/Flask), JSON batch processing

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

# Optional: For web UI
pip install gradio  # or flask
```

## Usage

### 1. Web Interface

```bash
python main.py --ui
# Opens at http://localhost:7860
```

### 2. Command Line

```bash
# Direct input
python main.py --table "| A | B |
|---|---|
| 1 | 2 |" --query "How many rows?"

# From JSON file
python main.py --json sample_data.json
```

### 3. Python API

```python
from agent import create_agent

agent = create_agent(verbose=True)

table = """
| 类别 | 数量 | 金额 |
|------|------|------|
| A    | 100  | 200  |
| B    | 150  | 300  |
"""

# Run analysis
result = agent.run(table, "共有多少个类别？")
print(result.answer)
print(result.mode)  # AgentMode.ANALYSIS or AgentMode.STRUCTURE
```

## Agent Design

### Agent0: Table Structure Understanding

Handles queries about table format, layout, and structure:
- Parse table structure (HTML/Markdown/Plain)
- Detect merged cells
- Identify headers and data regions
- Report dimensions

**Tools**:
- `parse_table_structure`: Parse raw table to standardized format
- `detect_merged_cells`: Find all merged cell regions
- `table_size_detector`: Get dimensions and bounding box

### Agent1: Data Analysis & Retrieval

Handles data queries using ReAct pattern:
1. Parse table structure
2. Convert to DataFrame or JSON tree
3. Execute analysis tools
4. Generate answer with reasoning

**Tools**:
- `parse_to_dataframe`: Convert to pandas DataFrame
- `parse_to_json_tree`: Convert to hierarchical JSON
- `get_cell_value`: Lookup specific cell
- `count_rows_with_condition`: Count matching rows
- `get_column_values`: Extract column data

## Query Classification

The agent automatically classifies queries:

**Structure Queries** (Agent0):
- "这个表格有多少行？"
- "表格中有哪些合并单元格？"
- "What is the table structure?"

**Analysis Queries** (Agent1):
- "本期共有多少个授予对象类别进行了行权？"
- "管理人员的本期解锁金额是多少？"
- "Which category has the highest value?"

## Sample Data Format

Compatible with benchmark dataset format:

```json
{
  "id": "904",
  "task_name": "数值分析",
  "sub_task_name": "统计",
  "context": {
    "context_markdown": "| col1 | col2 |...",
    "context_html": "<table>...</table>"
  },
  "question_list": ["问题1", "问题2"],
  "golden_answer_list": [{"最终答案": ["答案"]}]
}
```

## Example Output

```
📊 Processing sample ID: 904
📋 Task: 数值分析 / 统计
❓ Question: 各项权益工具中，本期共有多少个授予对象类别进行了行权？

🎯 Detected Mode: analysis

💭 Thought: First, I need to parse the table structure.
🔧 Action: parse_table_structure
👁 Observation: Parsed markdown table: 7×9

💭 Thought: This is a counting query...
🔧 Action: count_rows_with_condition
👁 Observation: Found 3 valid entries

📊 FINAL ANSWER:
**答案**: 3个
- 管理人员: 100,840
- 销售人员: 15,658  
- 生产人员: 81,983
```

## CLI Options

```
python main.py --help

Options:
  --ui          Launch web interface
  --json FILE   Process JSON sample file
  --table STR   Table content for CLI mode
  --query STR   Query for CLI mode
  --quiet       Suppress detailed output
  --port NUM    Web UI port (default: 7860)
```

## License

MIT License

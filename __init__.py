"""
Table Analysis Agent Package

An intelligent agent system for table structure understanding and data analysis.

Modules:
    - tools: Table parsing and analysis utilities
    - prompts: System prompts and prompt templates
    - agent: Main TableAnalysisAgent class
    - main: Entry point with CLI and Web UI
"""

from .agent import TableAnalysisAgent, create_agent, AgentResult, AgentMode
from .tools import (
    parse_table_structure,
    detect_merged_cells,
    table_size_detector,
    parse_to_dataframe,
    parse_to_json_tree,
    get_cell_value,
    count_rows_with_condition,
    classify_query
)

__version__ = "1.0.0"
__all__ = [
    "TableAnalysisAgent",
    "create_agent",
    "AgentResult",
    "AgentMode",
    "parse_table_structure",
    "detect_merged_cells",
    "table_size_detector",
    "parse_to_dataframe",
    "parse_to_json_tree",
    "get_cell_value",
    "count_rows_with_condition",
    "classify_query"
]

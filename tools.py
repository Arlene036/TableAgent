"""
Table Analysis Tools Module
Provides tools for table structure understanding and data extraction.
"""

import re
import json
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from bs4 import BeautifulSoup
import pandas as pd


@dataclass
class Cell:
    """Represents a single cell in a table."""
    row: int
    col: int
    value: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class TableStructure:
    """Structured representation of a table."""
    n_rows: int
    n_cols: int
    cells: List[Dict[str, Any]]
    headers: List[str]
    data_rows: List[List[str]]


# =============================================================================
# Agent 0: Table Structure Understanding Tools
# =============================================================================
def parse_table2df(table_input: str):
    """
    Convert raw table text (HTML, Markdown, Plain text) into pandas DataFrame.
    Supports:
        - HTML tables (<table>...</table>)
        - Markdown tables (| --- | --- |)
        - Plain text tables (whitespace / tab separated)
    """
    import pandas as pd
    import re
    from io import StringIO

    # 1. Try HTML first — pandas.read_html is very reliable
    if "<table" in table_input.lower():
        try:
            dfs = pd.read_html(table_input)
            return dfs[0]
        except Exception:
            pass

    # 2. Try Markdown ─ detect "| header |" and "---"
    if "|" in table_input and "---" in table_input:
        try:
            # Normalize markdown table
            md = "\n".join([line for line in table_input.split("\n") if "|" in line])
            df = pd.read_csv(StringIO(md), sep="|", engine="python")
            df = df.dropna(axis=1, how="all")  # drop empty boundary columns
            df = df.applymap(lambda x: str(x).strip() if not pd.isna(x) else x)
            return df
        except Exception:
            pass

    # 3. Plain text — split by tabs or >=2 spaces
    lines = [line.strip() for line in table_input.split("\n") if line.strip()]
    if len(lines) > 1:
        try:
            rows = []
            for line in lines:
                parts = re.split(r"\t+|\s{2,}", line)
                rows.append(parts)
            max_len = max(len(r) for r in rows)

            # pad uneven rows
            rows = [r + [""]*(max_len-len(r)) for r in rows]

            df = pd.DataFrame(rows[1:], columns=rows[0])
            df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
            return df
        except Exception:
            pass

    # 4. Fallback — return single column DataFrame
    return pd.DataFrame({"value": table_input.split("\n")})

def parse_table_structure(table_input: str, format_type: str = "auto") -> Dict[str, Any]:
    """
    Parse raw table into standardized structure.
    
    Args:
        table_input: Raw table content (HTML, markdown, or plain text)
        format_type: "html", "markdown", "auto" (auto-detect)
    
    Returns:
        Standardized structure with cells, dimensions, and metadata
    """
    if format_type == "auto":
        format_type = _detect_format(table_input)
    
    if format_type == "html":
        return _parse_html_table(table_input)
    elif format_type == "markdown":
        return _parse_markdown_table(table_input)
    else:
        return _parse_plain_table(table_input)


def _detect_format(table_input: str) -> str:
    """Auto-detect table format."""
    if "<table" in table_input.lower() or "<tr" in table_input.lower():
        return "html"
    elif "|" in table_input and "---" in table_input:
        return "markdown"
    else:
        return "plain"


def _parse_html_table(html_content: str) -> Dict[str, Any]:
    """Parse HTML table with merged cell support."""
    soup = BeautifulSoup(html_content, 'html.parser')
    table = soup.find('table')
    
    if not table:
        return {"error": "No table found in HTML content"}
    
    rows = table.find_all('tr')
    cells = []
    grid = {}  # Track cell positions for merged cells
    
    max_row = 0
    max_col = 0
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        for cell in row.find_all(['td', 'th']):
            # Skip positions occupied by merged cells
            while (row_idx, col_idx) in grid:
                col_idx += 1
            
            rowspan = int(cell.get('rowspan', 1))
            colspan = int(cell.get('colspan', 1))
            value = cell.get_text(strip=True)
            
            # Record cell
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "value": value,
                "rowspan": rowspan,
                "colspan": colspan,
                "is_header": cell.name == 'th'
            })
            
            # Mark grid positions for merged cells
            for r in range(row_idx, row_idx + rowspan):
                for c in range(col_idx, col_idx + colspan):
                    grid[(r, c)] = value
            
            max_row = max(max_row, row_idx + rowspan)
            max_col = max(max_col, col_idx + colspan)
            col_idx += colspan
    
    return {
        "n_rows": max_row,
        "n_cols": max_col,
        "cells": cells,
        "grid": {f"{k[0]},{k[1]}": v for k, v in grid.items()}
    }


def _parse_markdown_table(md_content: str) -> Dict[str, Any]:
    """Parse markdown table."""
    lines = [line.strip() for line in md_content.strip().split('\n') if line.strip()]
    
    # Filter out separator lines
    data_lines = []
    for line in lines:
        if line.startswith('|') and not re.match(r'^[\|\s\-:]+$', line):
            data_lines.append(line)
        elif '|' in line and not re.match(r'^[\|\s\-:]+$', line):
            data_lines.append(line)
    
    cells = []
    max_col = 0
    
    for row_idx, line in enumerate(data_lines):
        # Split by | and clean
        parts = [p.strip() for p in line.split('|')]
        # Remove empty parts at start/end
        if parts and parts[0] == '':
            parts = parts[1:]
        if parts and parts[-1] == '':
            parts = parts[:-1]
        
        max_col = max(max_col, len(parts))
        
        for col_idx, value in enumerate(parts):
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "value": value,
                "rowspan": 1,
                "colspan": 1,
                "is_header": row_idx == 0
            })
    
    return {
        "n_rows": len(data_lines),
        "n_cols": max_col,
        "cells": cells
    }


def _parse_plain_table(content: str) -> Dict[str, Any]:
    """Parse plain text table (space or tab separated)."""
    lines = [line for line in content.strip().split('\n') if line.strip()]
    cells = []
    max_col = 0
    
    for row_idx, line in enumerate(lines):
        parts = re.split(r'\s{2,}|\t', line.strip())
        max_col = max(max_col, len(parts))
        
        for col_idx, value in enumerate(parts):
            cells.append({
                "row": row_idx,
                "col": col_idx,
                "value": value.strip(),
                "rowspan": 1,
                "colspan": 1
            })
    
    return {
        "n_rows": len(lines),
        "n_cols": max_col,
        "cells": cells
    }


def detect_merged_cells(table_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Detect and return all merged cells from parsed table structure.
    
    Returns:
        List of merged cells with their ranges and content
    """
    merged = []
    for cell in table_structure.get("cells", []):
        if cell.get("rowspan", 1) > 1 or cell.get("colspan", 1) > 1:
            merged.append({
                "start_row": cell["row"],
                "start_col": cell["col"],
                "end_row": cell["row"] + cell.get("rowspan", 1) - 1,
                "end_col": cell["col"] + cell.get("colspan", 1) - 1,
                "value": cell["value"],
                "rowspan": cell.get("rowspan", 1),
                "colspan": cell.get("colspan", 1)
            })
    return merged


def table_size_detector(table_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Detect table dimensions and valid data region.
    
    Returns:
        Table size info including rows, cols, and bounding box
    """
    cells = table_structure.get("cells", [])
    if not cells:
        return {"n_rows": 0, "n_cols": 0, "bounding_box": None}
    
    min_row = min(c["row"] for c in cells)
    max_row = max(c["row"] + c.get("rowspan", 1) - 1 for c in cells)
    min_col = min(c["col"] for c in cells)
    max_col = max(c["col"] + c.get("colspan", 1) - 1 for c in cells)
    
    return {
        "n_rows": table_structure.get("n_rows", max_row + 1),
        "n_cols": table_structure.get("n_cols", max_col + 1),
        "bounding_box": {
            "top_left": (min_row, min_col),
            "bottom_right": (max_row, max_col)
        },
        "total_cells": len(cells)
    }


# =============================================================================
# Agent 1: Table Parsing and Analysis Tools
# =============================================================================

def get_cell_value(df: pd.DataFrame, row_label: str, col_name: str) -> Optional[str]:
    """
    Get value from DataFrame by row label (first column value) and column name.
    
    Args:
        df: pandas DataFrame
        row_label: Value to match in first column
        col_name: Column name to retrieve
    
    Returns:
        Cell value or None if not found
    """
    try:
        first_col = df.columns[0]
        mask = df[first_col].astype(str).str.strip() == row_label.strip()
        if mask.any():
            # Find the best matching column
            matching_cols = [c for c in df.columns if col_name in c]
            if matching_cols:
                return str(df.loc[mask, matching_cols[0]].iloc[0])
            elif col_name in df.columns:
                return str(df.loc[mask, col_name].iloc[0])
        return None
    except Exception as e:
        return f"Error: {str(e)}"


def get_json_value(json_tree: Dict[str, Any], 
                   keys: List[str]) -> Optional[str]:
    """
    Get value from JSON tree by row key and column key.
    
    Args:
        json_tree: JSON tree from parse_to_json_tree
        row_key: Row key (e.g., "管理人员")
        col_key: Column key (partial match supported)
    
    Returns:
        Value or None if not found
    """
    result = json_tree
    for key in keys:
        try:
            result = result[key]
        except:
            return result
    return result


def count_rows_with_condition(df: pd.DataFrame, 
                              column_pattern: str,
                              condition: str = "not_empty") -> int:
    """
    Count rows matching a condition in specified column(s).
    
    Args:
        df: pandas DataFrame
        column_pattern: Pattern to match column names
        condition: "not_empty", "has_value", "numeric"
    
    Returns:
        Count of matching rows
    """
    # Find matching columns
    matching_cols = [c for c in df.columns if column_pattern in c]
    
    if not matching_cols:
        return 0
    
    count = 0
    # Skip the last row if it's a total/合计 row
    check_df = df.copy()
    first_col = df.columns[0]
    if check_df[first_col].iloc[-1] in ['合计', '总计', 'Total', 'Sum']:
        check_df = check_df.iloc[:-1]
    
    for idx, row in check_df.iterrows():
        for col in matching_cols:
            val = str(row[col]).strip()
            if condition == "not_empty":
                if val and val != '-' and val != '':
                    count += 1
                    break
            elif condition == "has_value":
                if val and val != '-' and val != '' and val != '0':
                    count += 1
                    break
            elif condition == "numeric":
                # Check if it's a valid number
                clean_val = val.replace(',', '').replace(' ', '')
                try:
                    float(clean_val)
                    if float(clean_val) != 0:
                        count += 1
                        break
                except:
                    pass
    
    return count


def get_column_values(df: pd.DataFrame, column_pattern: str) -> List[Dict[str, str]]:
    """
    Get all values from columns matching pattern.
    
    Args:
        df: pandas DataFrame
        column_pattern: Pattern to match column names
    
    Returns:
        List of {row_label, column, value} dicts
    """
    matching_cols = [c for c in df.columns if column_pattern in c]
    first_col = df.columns[0]
    
    results = []
    for _, row in df.iterrows():
        row_label = str(row[first_col])
        for col in matching_cols:
            results.append({
                "row_label": row_label,
                "column": col,
                "value": str(row[col])
            })
    
    return results

# =============================================================================
# Tool Registry for Agent
# =============================================================================

TOOL_REGISTRY = {
    # Structure Understanding Tools
    "detect_merged_cells": {
        "func": detect_merged_cells,
        "description": "Detect all merged cells in a parsed table structure",
        "params": []
    },
    "table_size_detector": {
        "func": table_size_detector,
        "description": "Get table dimensions and valid data region bounding box",
        "params": []
    },
    # Analysis Tools
    "get_cell_value": {
        "func": get_cell_value,
        "description": "Get specific cell value from DataFrame by row label and column name",
        "params": ["row_label", "col_name"]
    },
    "get_json_value": {
        "func": get_json_value,
        "description": "Get value from JSON tree by row key and column key",
        "params": ["row_key", "col_key"]
    },
    "count_rows_with_condition": {
        "func": count_rows_with_condition,
        "description": "Count rows matching condition in specified column(s)",
        "params": ["column_pattern", "condition"]
    },
    "get_column_values": {
        "func": get_column_values,
        "description": "Get all values from columns matching a pattern",
        "params": ["column_pattern"]
    }
}


def execute_tool(tool_name: str, table, **kwargs) -> Any:
    # table: either table_structure or df or json
    if tool_name not in TOOL_REGISTRY:
        return {"error": f"Unknown tool: {tool_name}"}
    
    tool = TOOL_REGISTRY[tool_name]
    try:
        result = tool["func"](table, **kwargs)
        return result
    except Exception as e:
        return {"error": f"Tool execution error: {str(e)}"}

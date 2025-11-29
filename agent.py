"""
Table Analysis Agent Module
Main agent class that orchestrates table structure understanding and data analysis.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from tools import (
    parse_table2df,
    parse_table_structure,
    detect_merged_cells,
    table_size_detector,
    parse_to_dataframe,
    parse_to_json_tree,
    get_cell_value,
    count_rows_with_condition,
    get_column_values,
    TOOL_REGISTRY,
    execute_tool
)
from prompts import (
    MAIN_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    CLASSIFICATION_PROMPT,
    TABLE_PARSING_PROMPT
)

class QwenWrapper:
    def __init__(self, model_name='Qwen3-8B', max_new_tokens=4096):
        pass
    
    def call(self, system_prompt, user_prompt) -> str:
        pass

class ReActAgent:
    def __init__(
        self,
        tools: List[str],
        llm_engine = QwenWrapper(),
        direct_system_prompt = DIRECT_SYSTEM_PROMPT,
        max_iterations: int = 6,
        verbose: bool = True,
        memory_verbose: bool = False,
    ):
        self.history = []   

        available_tools = {}
        for tool in tools:
            tool_info = TOOL_REGISTRY[tool].copy()  # shallow copy
            if "func" in tool_info:
                del tool_info["func"]
            available_tools[tool] = tool_info
        tools_str = json.dumps(available_tools, indent=2)

        self.react_system_prompt = MAIN_SYSTEM_PROMPT.format(tools=tools_str)

        print('>>> self.react_system_prompt >>>')
        print(self.react_system_prompt)

        self.direct_system_prompt = direct_system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory_verbose = memory_verbose
        self.tools = tools
        self.llm_engine = llm_engine

    def convert2table_llm(self, table):
        respond = self.llm_engine.call(TABLE_PARSING_PROMPT)
        if 'dataframe' in respond:
            return parse_table2df(table)
        else:
            json_full = json.loads(respond)
            return json_full['data']

    def tool_parsed(self, respond: str):
        """
        Parse tool invocation from LLM output.
        Expected format:
        Action: tool_name
        Action Input: { ... json ... }
        """
        # Extract Action:
        try:
            action_line = re.search(r"Action\s*:\s*([a-zA-Z0-9_]+)", respond)
            tool_name = action_line.group(1).strip()
        except:
            raise ValueError(f"❌ Cannot parse tool name from respond: {respond}")

        # Extract Action Input:
        try:
            input_json = re.search(r"Action Input\s*:\s*(\{.*?\})", respond, re.S).group(1)
            kwargs = json.loads(input_json)
        except:
            kwargs = {}

        print(">>> tool_parsed INPUT>>>")
        print(respond)
        print(">>> tool_parsed OUTPUT>>>")
        print(tool_name)
        print(kwargs)
        return tool_name, kwargs
    
    def convert_table2str(self, input):
        # TODO
        return table_info

    

class StructureAgent(ReActAgent):
    def __init__(self, verbose=True):
        tools = [
            "parse_table_structure",
            "detect_merged_cells",
            "table_size_detector",
        ]
        super().__init__(
            tools=tools,
            verbose=verbose,
        )
    def run(self, table:str, query: str) -> str:
        self.table_structure = parse_table_structure(table)
        # self.table_info = self.convert_table2str(table) # TODO
        self.history.append(
            "Table: " + table # TODO, 可以改成table_info试试
        )
        for i in range(self.max_iterations):
            if i == 0:
                self.history.append(
                    "User: " + query
                )
                respond = self.llm_engine.call(self.react_system_prompt, '\n'.join(self.history))
                if 'Answer:' in respond:
                    return respond.split('Answer:')[-1]
                if 'Action:' in respond:
                    self.history.append(
                        "Action: " + respond.split('Action:')[-1]
                    )
                    tool_name, kwargs = self.tool_parsed(respond)
                    tool_return = execute_tool(tool_name, self.table_structure, **kwargs)
                    self.history.append(
                        "Observation: " + str(tool_return)
                    )
        # if outreach max_iterations
        respond = self.llm_engine.call(self.direct_system_prompt, '\n'.join(self.history))
        return respond

class AnalysisAgent(ReActAgent):
    def __init__(self, verbose=True):
        tools = [
            # "parse_to_dataframe",
            # "parse_to_json_tree",
            "get_cell_value",
            "get_json_value",
            "count_rows_with_condition",
            "get_column_values",
        ]
        super().__init__(
            tools=tools,
            verbose=verbose,
        )

    def run(self, table:str, query: str) -> str:
        self.table = convert2table_llm(table)
        # self.table_info = self.convert_table2str(table) # TODO
        self.history.append(
            "Table: " + table # TODO, 可以改成table_info试试
        )
        for i in range(self.max_iterations):
            if i == 0:
                self.history.append(
                    "User: " + query
                )
                respond = self.llm_engine.call(self.react_system_prompt, '\n'.join(self.history))
                if 'Answer:' in respond:
                    return respond.split('Answer:')[-1]
                if 'Action:' in respond:
                    self.history.append(
                        "Action: " + respond.split('Action:')[-1]
                    )
                    tool_name, kwargs = self.tool_parsed(respond)
                    tool_return = execute_tool(tool_name, self.table, **kwargs)
                    self.history.append(
                        "Observation: " + str(tool_return)
                    )
        # if outreach max_iterations
        respond = self.llm_engine.call(self.direct_system_prompt, '\n'.join(self.history))
        return respond
    

class TableAnalysisAgent:
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.llm = QwenWrapper()
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def run(self, table_input: str, query: str) -> AgentResult:
        self._log(f"\n{'='*60}")
        self._log(f"📋 Table Analysis Agent")
        self._log(f"{'='*60}")
        self._log(f"\n📝 Query: {query}")
        
        query_type = self.llm.call(CLASSIFICATION_PROMPT.format(query=query))

        if "structure" in query_type:
            agent = StructureAgent()
            return agent.run(table_input, query)
        else:
            agent = AnalysisAgent()
            return agent.run(table_input, query)
    
    def run_batch(self, table_input: str, queries: List[str]) -> List[AgentResult]:
        results = []
        for query in queries:
            result = self.run(table_input, query)
            results.append(result)
        return results


def create_agent(verbose: bool = True) -> TableAnalysisAgent:
    return TableAnalysisAgent(verbose=verbose)

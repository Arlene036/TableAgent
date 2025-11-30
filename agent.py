"""
Table Analysis Agent Module
Main agent class that orchestrates table structure understanding and data analysis.
"""

import json
import re
from typing import Dict, Any, List, Optional, Tuple
import os
import openai
from tools import (
    parse_table2df,
    parse_table_structure,
    detect_merged_cells,
    table_size_detector,
    get_cell_value,
    query_table_sql,
    get_column_values,
    TOOL_REGISTRY,
    execute_tool
)
from prompts import (
    MAIN_SYSTEM_PROMPT,
    STRUCTURE_SYSTEM_PROMPT,
    ANALYSIS_SYSTEM_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    CLASSIFICATION_PROMPT,
    TABLE_PARSING_PROMPT,
    GENERAL_SYSTEM_PROMPT
)

os.environ['OPENAI_API_KEY'] = 'YOUR_OPENAI_API_KEY'

class OpenAIWarpper:
    def __init__(self, model_name = 'gpt-4.1-mini', token = ""):
        self.client = openai.OpenAI(
                        api_key=os.environ['OPENAI_API_KEY'],
                        base_url="https://ai-gateway.andrew.cmu.edu/"
                    )
        self.model = model_name
    
    def call(self,system_prompt,user_prompt):
        response = self.client.chat.completions.create(
                model= self.model,
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user","content": user_prompt}
                ]
        )
        answer = response.choices[0].message.content
        return answer

class QwenWrapper:
    def __init__(self, model_name='Qwen3-8B', max_new_tokens=4096):
        # TODO
        self.model = model_name
        self.max_new_tokens = max_new_tokens
    
    def call(self, system_prompt, user_prompt) -> str:
        # TODO
        return "This is a test response from QwenWrapper"

class ReActAgent:
    def __init__(
        self,
        tools: List[str],
        llm_engine = QwenWrapper(),
        mode: str = 'structure',
        direct_system_prompt = DIRECT_SYSTEM_PROMPT,
        max_iterations: int = 6,
        verbose: bool = True,
        memory_verbose: bool = False,
    ):
        self.history = []   
        self.set_react_system_prompt(mode=mode, tools=tools)

        # print('>>> self.react_system_prompt >>>')
        # print(self.react_system_prompt)

        self.direct_system_prompt = direct_system_prompt
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.memory_verbose = memory_verbose
        self.tools = tools
        self.llm_engine = llm_engine

    def set_react_system_prompt(self, mode: str, tools: List[str]):
        available_tools = {}
        for tool in tools:
            tool_info = TOOL_REGISTRY[tool].copy()  # shallow copy
            if "func" in tool_info:
                del tool_info["func"]
            available_tools[tool] = tool_info
        tools_str = json.dumps(available_tools, indent=2)
        if mode == 'structure':
            self.react_system_prompt = STRUCTURE_SYSTEM_PROMPT.format(tools=tools_str)
        else:
            self.react_system_prompt = ANALYSIS_SYSTEM_PROMPT.format(tools=tools_str)

    def _log(self, message: str):
        if self.verbose:
            print(message)

    def convert2table_llm(self, table):
        respond = self.llm_engine.call(system_prompt=TABLE_PARSING_PROMPT, user_prompt=table)
        if 'dataframe' in respond:
            return parse_table2df(table)
        else:
            json_full = json.loads(respond)
            return json_full['data']

    def tool_parsed(self, respond: str):
        # Parse tool name
        try:
            action_line = re.search(r"Action\s*:\s*([a-zA-Z0-9_]+)", respond)
            tool_name = action_line.group(1).strip()
        except:
            raise ValueError(f"❌ Cannot parse tool name from respond: {respond}")

        # Try parsing the standard "Action Input: { ... }"
        kwargs = None
        match = re.search(r"Action Input\s*:\s*(\{.*?\})", respond, re.S)
        if match:
            try:
                kwargs = json.loads(match.group(1))
            except:
                kwargs = None

        # If still None, try parsing a bare `{ ... }` block after the action line
        if kwargs is None:
            # Find the first JSON block
            match = re.search(r"Action\s*:.*?(\{.*?\})", respond, re.S)
            if match:
                try:
                    kwargs = json.loads(match.group(1))
                except:
                    kwargs = {}

        # Final fallback
        if kwargs is None:
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
    def __init__(self, llm_engine, verbose=True, mode: str = 'structure'):
        tools = [
            # "detect_merged_cells",
            # "table_size_detector",
        ]
        super().__init__(
            tools=tools,
            verbose=verbose,
            llm_engine = llm_engine,
            mode=mode,
        )

    def run(self, table:str, query: str) -> str:
        self._log(f"[StructureAgent] 开始处理查询: {query[:50]}...")
        self.table_structure = parse_table_structure(table)
        # self.table_info = self.convert_table2str(table) # TODO
        self.history.append(
            "Table: " + table # TODO, 可以改成table_info试试,
        )
        self.history.append(
            "Table Size: " + str(self.table_structure['n_rows']) + " rows, " + str(self.table_structure['n_cols']) + " cols"
        )
        self.history.append(
            "Table Merged Cells: " + str(self.table_structure['merged_cells'])
        )
        respond = self.llm_engine.call(system_prompt=self.react_system_prompt, user_prompt='\n'.join(self.history))
        if 'Answer:' in respond:
            self._log(f"[StructureAgent] 找到答案")
            return respond.split('Answer:')[-1]
        else:
            respond = self.llm_engine.call(system_prompt=self.direct_system_prompt, user_prompt='\n'.join(self.history))
        return respond

class AnalysisAgent(ReActAgent):
    def __init__(self, llm_engine, verbose=True, mode: str = 'analysis'):
        tools = [
            # "parse_to_dataframe",
            # "parse_to_json_tree",
            "get_cell_value",
            "get_json_value",
            "query_table_sql",
            "get_column_values",
        ]
        super().__init__(
            tools=tools,
            verbose=verbose,
            llm_engine = llm_engine,
            mode=mode,
        )

    def run(self, table:str, query: str) -> str:
        self._log(f"[AnalysisAgent] 开始处理查询: {query[:50]}...")
        self.table = self.convert2table_llm(table)
        # self.table_info = self.convert_table2str(table) # TODO
        self.history.append(
            "Table: " + table # TODO, 可以改成table_info试试
        )
        for i in range(self.max_iterations):
            self._log(f"[AnalysisAgent] 迭代 {i+1}/{self.max_iterations}")
            if i == 0:
                self.history.append(
                    "User: " + query
                )
            respond = self.llm_engine.call(system_prompt=self.react_system_prompt, user_prompt='\n'.join(self.history))
            if 'Answer:' in respond:
                self._log(f"[AnalysisAgent] 找到答案")
                return respond.split('Answer:')[-1]
            if 'Action:' in respond:
                self.history.append(
                    "Action: " + respond.split('Action:')[-1]
                )
                tool_name, kwargs = self.tool_parsed(respond)
                self._log(f"[AnalysisAgent] 执行工具: {tool_name}")
                tool_return = execute_tool(tool_name, self.table, **kwargs)
                self.history.append(
                    "Observation: " + str(tool_return)
                )
                self._log(f"[AnalysisAgent] 工具名称: {tool_name}")
                self._log(f"[AnalysisAgent] 工具参数: {kwargs}")
                self._log(f"[AnalysisAgent] 工具返回: {tool_return}")
        # if outreach max_iterations
        self._log(f"[AnalysisAgent] 达到最大迭代次数")
        respond = self.llm_engine.call(system_prompt=self.direct_system_prompt, user_prompt='\n'.join(self.history))
        return respond
    

class TableAnalysisAgent:
    def __init__(self, llm_type: str = "qwen", verbose: bool = True):
        self.verbose = verbose
        if llm_type == 'qwen':
            self.llm = QwenWrapper()
        else:
            self.llm = OpenAIWarpper()
        
    def _log(self, message: str):
        if self.verbose:
            print(message)
    
    def run(self, table_input: str, query: str) -> str:
        self._log(f"\n{'='*60}")
        self._log(f"📋 Table Analysis Agent")
        self._log(f"{'='*60}")
        self._log(f"\n📝 Query: {query}")
        
        self._log(f"[TableAnalysisAgent] 分类查询类型")
        query_type = self.llm.call(system_prompt=GENERAL_SYSTEM_PROMPT, user_prompt=CLASSIFICATION_PROMPT.format(query=query))
        self._log(f"[TableAnalysisAgent] 查询类型: {query_type[:50]}...")

        if "structure" in query_type[:20]:
            self._log(f"[TableAnalysisAgent] 使用StructureAgent")
            agent = StructureAgent(llm_engine = self.llm, verbose=self.verbose)
            return agent.run(table_input, query)
        else:
            self._log(f"[TableAnalysisAgent] 使用AnalysisAgent")
            agent = AnalysisAgent(llm_engine = self.llm, verbose=self.verbose)
            return agent.run(table_input, query)
    
    def run_batch(self, table_input: str, queries: List[str]) -> List[str]:
        results = []
        for query in queries:
            result = self.run(table_input, query)
            results.append(result)
        return results


def create_agent(llm_type: str = 'qwen', verbose: bool = True) -> TableAnalysisAgent:
    return TableAnalysisAgent(llm_type=llm_type, verbose=verbose)

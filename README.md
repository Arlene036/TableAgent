# 📊 Table Analysis Agent

An intelligent agent system for table structure understanding and data analysis. Based on a two-agent design pattern:

- **Agent0 (Structure)**: Table structure understanding - detects merged cells, headers, layout
- **Agent1 (Analysis)**: Data analysis & information retrieval - counts, values, calculations

We equipped Qwen3-8B with two external tools: (1) A table parser converts noisy HTML or Markdown tables into a structured DataFrame and extracts structural metadata including the number of rows, number of columns, and merged-cell spans. (2) DuckDB provides a SQL interface that allows the LLM to directly interact with the parsed dataframe, performing selection, filtering, arithmetic operations, and aggregation.

The agent follows a ReAct-style interaction loop, generating intermediate thoughts, selecting an action, executing tool calls, and interpreting the returned observations until an answer is produced.
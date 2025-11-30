import json
from pathlib import Path

INPUT_PATH = "/Users/yaqi/cmu/course/11667/TableEval-test.jsonl"
OUTPUT_PATH = "/Users/yaqi/cmu/course/11667/TableEval-results-79-2040.jsonl" # 
MAX_SAMPLE = 1
TYPE = 'openai'

from agent import create_agent

def run_agent_on_dataset():
    agent = create_agent(TYPE)
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    if output_file.exists():
        output_file.unlink()

    count = 0
    with input_file.open("r", encoding="utf8") as f_in, \
         output_file.open("w", encoding="utf8") as f_out:

        for line in f_in:
            if not line.strip():
                continue

            sample = json.loads(line)

            # 获取表格和问题列表
            table = sample["context"]["context_markdown"]   # 如果你 agent 输入 HTML 换成 context_html
            question_list = sample.get("question_list", [])
            
            # 处理所有问题，生成预测列表
            prediction_list = []
            for query in question_list:
                answer = agent.run(table_input=table, query=query)
                prediction_list.append(answer)

            # 构建符合 TableEval 评估格式的结果
            result = {
                "id": sample.get("id", None),
                "task_name": sample.get("task_name", None),
                "sub_task_name": sample.get("sub_task_name", None),
                "golden_answer_list": sample.get("golden_answer_list", []),
                "prediction_list": prediction_list
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            count += 1
            if count == MAX_SAMPLE:
                break
    print("Finish! Results saved to:", OUTPUT_PATH)


def run_agent_on_last_samples():
    agent = create_agent(TYPE)
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    if output_file.exists():
        output_file.unlink()

    # 读取所有行
    with input_file.open("r", encoding="utf8") as f:
        lines = [line for line in f if line.strip()]

    # 倒数 10 条
    last_samples = lines[-MAX_SAMPLE:]

    with output_file.open("w", encoding="utf8") as f_out:
        for line in last_samples:
            sample = json.loads(line)

            # 获取表格和问题列表
            table = sample["context"]["context_markdown"]
            question_list = sample.get("question_list", [])
            
            # 处理所有问题，生成预测列表
            prediction_list = []
            for query in question_list:
                answer = agent.run(table_input=table, query=query)
                prediction_list.append(answer)

            # 构建符合 TableEval 评估格式的结果
            result = {
                "id": sample.get("id", None),
                "task_name": sample.get("task_name", None),
                "sub_task_name": sample.get("sub_task_name", None),
                "golden_answer_list": sample.get("golden_answer_list", []),
                "prediction_list": prediction_list
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Finish! Results saved to:", OUTPUT_PATH)

def run_agent_by_subtask():
    agent = create_agent(TYPE)
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    # 若结果文件存在先清空
    if output_file.exists():
        output_file.unlink()

    # 读取所有 sample
    with input_file.open("r", encoding="utf8") as f:
        samples = [json.loads(line) for line in f if line.strip()]

    # 按 sub_task_name 分组
    grouped = {}
    for s in samples:
        sub = s.get("sub_task_name", "unknown")
        grouped.setdefault(sub, []).append(s)

    # 每类取 2 条
    selected = []
    for sub, items in grouped.items():
        selected.extend(items[:2])  # 取前 2 条

    # 执行 agent
    with output_file.open("w", encoding="utf8") as f_out:
        for sample in selected:
            # 获取表格和问题列表
            table = sample["context"]["context_markdown"]
            question_list = sample.get("question_list", [])
            
            # 处理所有问题，生成预测列表
            prediction_list = []
            for query in question_list:
                answer = agent.run(table_input=table, query=query)
                prediction_list.append(answer)

            # 构建符合 TableEval 评估格式的结果
            result = {
                "id": sample.get("id"),
                "task_name": sample.get("task_name"),
                "sub_task_name": sample.get("sub_task_name"),
                "golden_answer_list": sample.get("golden_answer_list", []),
                "prediction_list": prediction_list
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Finish! Results saved to:", OUTPUT_PATH)



def run_agent_on_indices(indices):
    agent = create_agent(TYPE)
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    if output_file.exists():
        output_file.unlink()

    # 读取所有行
    with input_file.open("r", encoding="utf8") as f:
        lines = [line for line in f if line.strip()]

    # 处理 1-based 和 0-based
    normalized = []
    for idx in indices:
        if idx >= 1:
            normalized.append(idx - 1)
        else:
            normalized.append(idx)

    selected_lines = [lines[i] for i in normalized if 0 <= i < len(lines)]

    with output_file.open("w", encoding="utf8") as f_out:
        for line in selected_lines:
            sample = json.loads(line)
            table = sample["context"]["context_html"]
            question_list = sample.get("question_list", [])
            
            prediction_list = []
            for query in question_list:
                answer = agent.run(table_input=table, query=query)
                prediction_list.append(answer)

            result = {
                "id": sample.get("id"),
                "task_name": sample.get("task_name"),
                "sub_task_name": sample.get("sub_task_name"),
                "golden_answer_list": sample.get("golden_answer_list", []),
                "prediction_list": prediction_list
            }
            f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("Finish! Results saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    # run_agent_on_dataset()
    # run_agent_on_last_samples()
    # run_agent_by_subtask()
    run_agent_on_indices( range(79, 2040))
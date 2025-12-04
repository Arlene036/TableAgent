import json
from pathlib import Path

INPUT_PATH = "TableEval-test.jsonl"
OUTPUT_PATH = "TableEval-results" # 

from agent import create_agent


def run_agent_on_indices(indices):
    agent = create_agent()
    input_file = Path(INPUT_PATH)
    output_file = Path(OUTPUT_PATH)

    if output_file.exists():
        output_file.unlink()

    with input_file.open("r", encoding="utf8") as f:
        lines = [line for line in f if line.strip()]

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
    run_agent_on_indices( range(1804, 2040))
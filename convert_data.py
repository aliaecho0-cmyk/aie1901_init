import json
import os
from pathlib import Path

# 1. 读取你的 jsonl 文件（请确保文件名和路径对）
input_file = "tasks_train.jsonl" 
# 2. 设置输出路径（代码默认去这里找数据）
out_root = Path("data/synthetic_tasks/tau_bench_retail")
out_root.mkdir(parents=True, exist_ok=True)

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

for line in lines:
    if not line.strip():
        continue
    data = json.loads(line.strip())
    
    # 建立独立的文件夹（比如 task_0, task_1）
    task_dir = out_root / f"task_{data['task_id']}"
    task_dir.mkdir(exist_ok=True)
    
    # 提取可用工具（从标准答案的 actions 里提取工具名）
    tools = list(set([action['name'] for action in data.get('actions', [])]))
    
    # === 调格式核心 ===
    
    # 1. 生成 task.json
    task_json = {
        "task_id": f"tau_retail_{data['task_id']}",
        "task_type": "multi_tool", # 用多工具域来跑
        "prompt": data["description"],
        "available_tools": tools,
        "max_steps": 10,
        "cost_budget": 1.0
    }
    (task_dir / "task.json").write_text(json.dumps(task_json, indent=2))
    
    # 2. 生成 rubric.json（把标准答案塞进评判标准里）
    rubric_json = {
        "criteria": [
            {
                "id": "c1",
                "max_score": 1.0,
                "description": f"Agent must execute actions: {json.dumps(data.get('actions', []))}",
                "verifier": "tool_call_valid"
            }
        ]
    }
    (task_dir / "rubric.json").write_text(json.dumps(rubric_json, indent=2))
    
    # 3. 生成 metadata.json（代码要求必须有这个文件，给个空的或简单信息）
    metadata_json = {"source": "tau_bench", "difficulty": "standard"}
    (task_dir / "metadata.json").write_text(json.dumps(metadata_json, indent=2))

print(f"大功告成！已经在 {out_root} 生成了 {len(lines)} 个任务文件夹。")
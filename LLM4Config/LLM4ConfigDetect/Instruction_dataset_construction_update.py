## LLM 的微调一般指指令微调过程。所谓指令微调，是说我们使用的微调数据形如：
# {
#     "instruction":"回答以下用户问题，仅输出答案。",
#     "input":"1+1等于几?",
#     "output":"2"
# }
### 指令模板解释：
# 其中，instruction 是用户指令，告知模型其需要完成的任务；input 是用户输入，是完成用户指令所必须的输入内容；output 是模型应该给出的输出。

### 即我们的核心训练目标是让模型具有理解并遵循用户指令的能力。因此，在指令集构建时，我们应针对我们的目标任务，针对性构建任务指令集。
import json
import yaml
import subprocess
from pathlib import Path
from tqdm import tqdm  # 进度条

# 输入 JSON 文件路径
top_1000_json_path = "LLM4Config/data/top_1000_split_files.json"

# 读取 top_1000_split_files.json 文件
with open(top_1000_json_path, "r", encoding="utf-8") as f:
    top_1000_data = json.load(f)

# 设定 LLaMA 3 指令
instruction_text = (
    "You are a Kubernetes security expert specializing in analyzing the security "
    "of Kubernetes configuration files. Carefully review the following Kubernetes YAML file "
    "and identify any violations of security best practices."
)

# 目标输出 JSON 文件路径
output_json_path = "LLM4Config/data/instruction_templates_update.json"
llama3_data = []

# 函数：使用 Checkov 进行 YAML 静态扫描（针对单个子文档）
def run_checkov(yaml_subdoc):
    temp_yaml = "temp_checkov.yaml"
    
    # 将子文档写入临时 YAML 文件
    with open(temp_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_subdoc)
    
    # 运行 Checkov 扫描
    try:
        result = subprocess.run(
            ["checkov", "-f", temp_yaml, "--output", "json"],
            capture_output=True,
            text=True
        )
        checkov_output = json.loads(result.stdout) if result.stdout else {}
        return [
            {
                "checkov_id": item["check_id"], 
                "checkov_resourceInfo": item["check_result"]["evaluated_keys"]
            }
            for item in checkov_output.get("results", {}).get("failed_checks", [])
        ]
    except Exception as e:
        print(f"⚠️ Checkov scanning error: {e}")
        return []

# 函数：使用 Terrascan 进行 YAML 静态扫描（针对单个子文档）
def run_terrascan(yaml_subdoc):
    temp_yaml = "temp_terrascan.yaml"
    
    # 将子文档写入临时 YAML 文件
    with open(temp_yaml, "w", encoding="utf-8") as f:
        f.write(yaml_subdoc)
    
    # 运行 Terrascan 扫描
    try:
        result = subprocess.run(
            ["terrascan", "scan", "-i", "k8s", "-f", temp_yaml, "--output", "json"],
            capture_output=True,
            text=True
        )
        terrascan_output = json.loads(result.stdout) if result.stdout else {}
        return [
            {
                "terrascan_rule": item["rule_name"], 
                "terrascan_severity": item["severity"],
                "terrascan_resource": item["resource_name"]
            }
            for item in terrascan_output.get("results", [])
        ]
    except Exception as e:
        print(f"⚠️ Terrascan scanning error: {e}")
        return []

# 处理 YAML 文件并进行静态扫描
for item in tqdm(top_1000_data, desc="Processing YAML Files", unit="file"):
    yaml_file = item["files"]
    indices = item["results"]  # 需要提取的子文档索引列表

    # 读取 YAML 文件内容
    yaml_path = Path(yaml_file)
    if not yaml_path.exists():
        print(f"⚠️ Warning: File {yaml_file} not found, skipping...")
        continue

    with open(yaml_file, "r", encoding="utf-8") as f:
        yaml_content = list(yaml.safe_load_all(f))  # 加载所有子文档

    # 遍历需要提取的索引
    for idx in tqdm(indices, desc=f"Scanning sub-documents in {yaml_file}", leave=False, unit="sub-doc"):
        if idx < len(yaml_content):  # 确保索引有效
            yaml_subdoc = yaml.dump(yaml_content[idx], default_flow_style=False)

            # 运行 Checkov 和 Terrascan 进行安全扫描
            checkov_results = run_checkov(yaml_subdoc)
            terrascan_results = run_terrascan(yaml_subdoc)

            # 组合扫描结果
            security_findings = checkov_results + terrascan_results

            # 添加到 LLaMA 3 指令模板
            llama3_data.append({
                "instruction": instruction_text,
                "input": yaml_subdoc,
                "output": security_findings
            })
        else:
            print(f"⚠️ Warning: Index {idx} out of range for file {yaml_file}")

# 保存到 JSON 文件
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(llama3_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ LLaMA 3 instruction data with security scans saved to {output_json_path}")

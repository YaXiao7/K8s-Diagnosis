import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, DataCollatorForSeq2Seq, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType,PeftModel
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm 
import re
# ========== 1. 读取数据集 ==========
df = pd.read_json('LLM4Config/data/instruction_templates_update.json')

# 80% 训练集, 10% 验证集, 10% 测试集
train_size = int(0.8 * len(df))
valid_size = int(0.1 * len(df))
test_size = len(df) - train_size - valid_size

df = df.sample(frac=1, random_state=42)  # 随机打乱数据
train_df = df.iloc[:train_size]
valid_df = df.iloc[train_size:train_size + valid_size]
test_df = df.iloc[train_size + valid_size:]

train_ds = Dataset.from_pandas(train_df)
valid_ds = Dataset.from_pandas(valid_df)
test_ds = Dataset.from_pandas(test_df)

# ========== 2. 加载 Tokenizer ==========
tokenizer = AutoTokenizer.from_pretrained(
    '/root/autodl-tmp/20250225/LLM4Config/model/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B-Instruct',
    use_fast=False, trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ========== 3. 数据预处理 ==========
def process_func(example):
    MAX_LENGTH = 1024
    instruction = tokenizer(f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                            f"You are a Kubernetes security expert specializing in analyzing the security of Kubernetes configuration files."
                            f" Carefully review the following Kubernetes YAML file and identify any violations of security best practices."
                            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", 
                            add_special_tokens=False)
    
    response = tokenizer(f"{json.dumps(example['output'])}<|eot_id|>", add_special_tokens=False)  # JSON 格式
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_ds = train_ds.map(process_func, remove_columns=train_ds.column_names)
valid_ds = valid_ds.map(process_func, remove_columns=valid_ds.column_names)
test_ds = test_ds.map(process_func, remove_columns=test_ds.column_names)

# ========== 4. 配置 LoRA 训练 ==========
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1
)

args = TrainingArguments(
    output_dir="LLM4Config/LLM4ConfigDetect/weights/llama3_finetune_update",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=7,  # 训练 epoch 数
    learning_rate=5e-5,
    save_strategy="no",  # 禁用检查点保存
    save_total_limit=1,  # 只保存最终的模型
    save_on_each_node=True,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    load_best_model_at_end=False  # 不加载最佳模型
)

# ========== 5. 加载模型并微调 ==========
print("🚀 Loading LLaMA3.1 model for fine-tuning...")
model = AutoModelForCausalLM.from_pretrained(
    '/root/autodl-tmp/20250225/LLM4Config/model/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B-Instruct',
    device_map="auto", torch_dtype=torch.float16
)
model.enable_input_require_grads() 
model = get_peft_model(model, config)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=valid_ds,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
trainer.save_model("LLM4Config/LLM4ConfigDetect/weights/llama3_finetune_update")

# ========== 6. 加载训练后的模型进行推理 ==========
print("🔄 Loading fine-tuned model for inference...")
base_model_path = "/root/autodl-tmp/20250225/LLM4Config/model/Meta-Llama-3.1-8B/LLM-Research/Meta-Llama-3___1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
tokenizer.pad_token = tokenizer.eos_token

# ========== 7. 加载 LLaMA3.1 基础模型 ==========
print("🔄 Loading base LLaMA3.1 model...")
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # 16-bit 计算，减少显存
    device_map="auto"
)

# ========== 8. 加载 LoRA 适配器 ==========
lora_adapter_path = "LLM4Config/LLM4ConfigDetect/weights/llama3_finetune_update"
print("🔄 Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_adapter_path)

model.eval()  # 切换到推理模式
print("✅ Model loaded successfully!")

# ========== 9. 运行推理 ==========
def extract_json(text):
    """
    从 LLaMA3.1 生成的 `predicted_text` 中提取 JSON 数据
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)  # 正则匹配 JSON
    if match:
        try:
            return json.loads(match.group(0))  # 解析 JSON
        except json.JSONDecodeError:
            return []  # 无效 JSON 返回空列表
    return []  # 如果没有匹配 JSON，返回空列表
    

def generate_predictions(model, test_dataset):
    predictions = []
    references = []

    print("🚀 Running inference on the test set...")

    for example in tqdm(test_dataset, desc="Processing", unit="sample"):
        input_text = (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a Kubernetes security expert specializing in analyzing the security of Kubernetes configuration files. "
            "Carefully review the following Kubernetes YAML file and identify any violations of security best practices."
            f"<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{example['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(input_ids, max_new_tokens=1024, do_sample=True, top_k=50, top_p=0.95)

        predicted_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        json_output = extract_json(predicted_text)  # 仅提取 JSON

        predictions.append(json_output)  # 仅存 JSON
        references.append(example["output"])  # 解析真实值

    return predictions, references

# 运行推理
predictions, references = generate_predictions(model, test_ds)

# ========== 10. 计算 Precision, Recall, F1 Score ==========
def compute_metrics(predictions, references):
    """
    计算 Precision、Recall、F1 Score，基于 checkov_id 的匹配情况
    """

    def extract_checkov_ids(data):
        """
        从 JSON 数据中提取 checkov_id 列表
        """
        try:
            output_list = json.loads(data) if isinstance(data, str) else data
            return set(item["checkov_id"] for item in output_list if "checkov_id" in item)
        except json.JSONDecodeError:
            return set()

    # 提取模型预测的 checkov_id & 真实的 checkov_id
    pred_ids = [extract_checkov_ids(pred) for pred in predictions]
    ref_ids = [extract_checkov_ids(ref) for ref in references]
    print(pred_ids)
    print(ref_ids)
    # 计算 TP, FP, FN
    TP, FP, FN = 0, 0, 0

    print("\n🔄 Computing evaluation metrics...")
    for pred_set, ref_set in tqdm(zip(pred_ids, ref_ids), total=len(ref_ids), desc="Evaluating", unit="sample"):
        TP += len(pred_set & ref_set)  # 预测正确的
        FP += len(pred_set - ref_set)  # 预测错误的
        FN += len(ref_set - pred_set)  # 漏掉的

    print(f"\n✅ TP: {TP}, FP: {FP}, FN: {FN}")

    # 避免 F1 Score 计算问题
    if TP == 0:
        print("⚠️ No true positives (TP=0). Setting F1 Score to 0.")
        return 0.0, 0.0, 0.0

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

# 计算最终评估指标
precision, recall, f1 = compute_metrics(predictions, references)

print(f"\n📊 Model Evaluation Metrics:")
print(f"✅ Precision: {precision:.4f}")
print(f"✅ Recall: {recall:.4f}")
print(f"✅ F1 Score: {f1:.4f}")
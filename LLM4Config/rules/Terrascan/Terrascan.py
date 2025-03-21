import re
import csv
from pathlib import Path

# 读取 Markdown 文件内容
def read_markdown(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# 提取表格数据的正则表达式
def extract_table_data(md_content):
    # 匹配表格的每一行数据，忽略表头和分隔符行
    table_pattern = r'\| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|'
    
    # 找到所有匹配的行
    matches = re.findall(table_pattern, md_content)
    
    # 过滤掉表头行和分隔符行，返回实际数据行
    data = []
    for match in matches:
        # 如果匹配到的行是表头行（即字段名行）或分隔符行（--------），跳过
        if match == ('Category', 'Resource', 'Severity', 'Description', 'Reference ID', 'ID') or '--------' in match:
            continue
        data.append(match)
    
    return data

# 将数据保存到 CSV 文件
def save_to_csv(data, output_file):
    # 设置 CSV 文件的列头
    headers = ['Category', 'Resource', 'Severity', 'Description', 'Reference ID', 'ID']
    
    # 写入 CSV 文件
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)  # 写入列头
        
        # 写入数据行
        for row in data:
            writer.writerow(row)  # 写入数据行

def main():
    # Markdown 文件路径
    md_file_path = 'LLM4Config/rules/Terrascan/kubernetes.md'  # 根据实际路径修改
    # CSV 输出文件路径
    output_file = 'LLM4Config/rules/Terrascan/terrascan_k8s_rules.csv'  # 根据实际路径修改

    # 读取 Markdown 文件内容
    md_content = read_markdown(md_file_path)

    # 提取表格数据
    table_data = extract_table_data(md_content)

    # 保存到 CSV 文件
    save_to_csv(table_data, output_file)
    print(f"数据已成功保存到 {output_file}")

if __name__ == '__main__':
    main()

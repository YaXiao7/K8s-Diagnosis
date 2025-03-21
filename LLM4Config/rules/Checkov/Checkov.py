"""Module for parsing and processing Checkov security rules from markdown documentation.

This module provides functionality to extract security rule information from Checkov's
markdown documentation and convert it into a structured CSV format.
"""

import re
from pathlib import Path

import pandas as pd

# file = "github/checkov/docs/5.Policy Index/kubernetes.md"
file = "LLM4Config/rules/Checkov/kubernetes.md"

def parse_markdown_table(content):
    """Parse a markdown table containing Checkov security rules.
    
    Args:
        content (str): The markdown content containing the table of security rules
        
    Returns:
        list: A list of dictionaries containing parsed rule information with keys:
            - Id: The Checkov rule ID
            - Type: The type of resource being checked
            - Entity: The entity being checked
            - Policy: The security policy description
            - IaC: The Infrastructure as Code platform
            - Resource Link: Link to the rule implementation
    """
    # Regular expression to match table rows
    row_pattern = r"\|\s*(\d+)\s*\|\s*(CKV[^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|\s*([^|]+)\|"

    # Find all matches in the content
    matches = re.findall(row_pattern, content)

    # Convert matches to list of dictionaries
    data = []
    for match in matches:
        row = {
            "Id": match[1].strip(),
            "Type": match[2].strip(),
            "Entity": match[3].strip(),
            "Policy": match[4].strip(),
            "IaC": match[5].strip(),
            "Resource Link": match[6].strip(),
        }
        data.append(row)

    return data

# Sample usage
with open(file, "r", encoding="utf-8") as f:
    content = f.read()

df = pd.DataFrame(parse_markdown_table(content))

df["id"] = df["Id"]
df["file"] = df["Resource Link"]
df.drop(columns=["Id", "Type", "Entity", "Policy", "IaC", "Resource Link"], inplace=True)
df["file"] = df["file"].apply(lambda x: x.split("(")[-1].replace(")", "").replace("https://github.com/bridgecrewio/checkov/blob/main/checkov", "LLM4Config/rules/Checkov"))
df["type"] = df["file"].apply(lambda x: x.split("/")[-1].split(".")[0])
df = df[["id", "type", "file"]]
df = df.drop_duplicates()

df.to_csv("LLM4Config/rules/Checkov/checkov_k8s_rules.csv", index=False)

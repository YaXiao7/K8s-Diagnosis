"""
YAML file processor for multi-document Kubernetes configurations.
This script splits multi-document YAML files, performs security validation,
and maintains relationships between split documents.

Features:
- Multi-document YAML splitting
- Security validation using Checkov and Terrascan
- Parallel processing for efficiency
- Temporary file handling for processing
"""

import json
import subprocess
import tempfile

import pandas as pd
import yaml
from joblib import Parallel, delayed
from tqdm.rich import tqdm

def check_checkov(path):
    """
    Validates YAML file using Checkov security scanner.

    Args:
        path (str): Path to YAML file to check

    Returns:
        bool: True if security issues found, False otherwise
    """
    scan_output = subprocess.run(["checkov", "-f", path, "--framework", "kubernetes", "-o", "json", "--compact", "--quiet"], capture_output=True)
    try:
        data = json.loads(scan_output.stdout.decode("utf-8"))
        if isinstance(data, list):
            for _data in data:
                summary = _data.get("summary", None)
                if summary and summary["failed"] > 0:
                    return True
            return False
        else:
            summary = data.get("summary", None)
            if summary and summary["failed"] > 0:
                return True
            return False
    except Exception:
        return False

def check_terrascan(path):
    """
    Validates YAML file using Terrascan security scanner.

    Args:
        path (str): Path to YAML file to check

    Returns:
        bool: True if violations found, False otherwise
    """
    scan_output = subprocess.run(["terrascan", "scan", "-i", "k8s", "-f", path, "-o", "json"], capture_output=True)
    try:
        data = json.loads(scan_output.stdout.decode("utf-8"))
        if isinstance(data, list):
            for _data in data:
                result = _data.get("result", None)
                scaned_summary = result.get("scan_summary", None)
                v_count = scaned_summary.get("violated_policies", None)
                if v_count and v_count > 0:
                    return True
            return False
        else:
            result = data.get("results", None)
            scaned_summary = result.get("scan_summary", None)
            v_count = scaned_summary.get("violated_policies", None)
            if v_count and v_count > 0:
                return True
        return False
    except Exception:
        return False

def get_yamls(file_path):
    """
    Extracts YAML documents from a file.

    Args:
        file_path (str): Path to YAML file

    Returns:
        list: List of YAML documents found in the file
    """
    with open(file_path, "r") as f:
        yamls = yaml.load_all(f.read(), Loader=yaml.FullLoader)
    return list(yamls)
    
def check_one(input_yaml):
    """
    Validates a single YAML document using both security scanners.

    Args:
        input_yaml (dict): YAML document to validate

    Returns:
        bool: True if document passes security checks, False otherwise
    """
    with tempfile.NamedTemporaryFile(delete=True, mode="w+", suffix=".yaml") as temp:
        yaml.dump(input_yaml, temp, indent=4)
        temp.seek(0)
        if check_checkov(temp.name) and check_terrascan(temp.name):
            return True
        else:
            return False

def process_yaml(idx, yaml):
    """
    Processes and validates a single YAML document.

    Args:
        idx (int): Document index for tracking
        yaml_doc (dict): YAML document to process

    Returns:
        tuple: (index, document) if valid, None otherwise
    """
    if check_one(yaml):
        return idx
    return None

def process_file(file_path):
    """
    Processes a multi-document YAML file.

    Args:
        file_path (str): Path to YAML file

    Returns:
        list: List of valid YAML documents with their indices
    """
    result = []
    yamls = get_yamls(file_path)
    processed = Parallel(n_jobs=16)(delayed(process_yaml)(idx, yaml) for idx, yaml in enumerate(yamls))
    result = [idx for idx in processed if idx is not None]
    return {"files": file_path, "results": result}

def check_result(result):
    """
    Validates processing results.

    Args:
        result (list): List of processed documents

    Returns:
        bool: True if any valid documents found, False otherwise
    """
    total = sum(len(value) for value in result.values())
    return total >= 1000, total

if __name__ == "__main__":
    # Load list of files to process
    df = pd.read_csv("LLM4Config/data/cleaned_files.csv")
    file_list = df["files"].to_list()

    # Process files in parallel with progress tracking
    results = Parallel(n_jobs=16)(delayed(process_file)(file_path) for file_path in tqdm(file_list))
    
    # Save processed results
    with open("LLM4Config/data/split_files.json", "w") as f:
        json.dump(results, f, indent=2)

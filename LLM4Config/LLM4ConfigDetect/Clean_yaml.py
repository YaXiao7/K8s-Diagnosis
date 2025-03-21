"""
Security scanner for YAML configurations from Helm packages.
This script performs security scanning using Checkov and Terrascan tools,
validates YAML syntax, and filters out invalid configurations.

Dependencies:
    - checkov: For Kubernetes security scanning
    - terrascan: For additional security validation
"""
### assets文件夹中，有的YAML文件没有被检测出问题，因此进行一个清理
import json
import subprocess
from glob import glob

import pandas as pd
from joblib import Parallel, delayed
from tqdm.rich import tqdm

def check_checkov(path):
    """
    Scans a YAML file using Checkov for Kubernetes security issues.

    Args:
        path (str): Path to the YAML file to scan

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
    Scans a YAML file using Terrascan for additional security validation.

    Args:
        path (str): Path to the YAML file to scan

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

def process_item(file):
    """
    Processes a single YAML file through both security scanners.

    Args:
        file (str): Path to YAML file

    Returns:
        str: File path if security issues found, None otherwise
    """
    if check_checkov(file) and check_terrascan(file):
        return file
    return None

if __name__ == "__main__":
    # Load raw package data
    with open("LLM4Config/data/raw_final_results.json") as f:
        data = json.load(f)
    
    # Get all YAML files
    files = glob("LLM4Config/assets/*.yaml")
    
    # Process files in parallel with progress bar
    results = Parallel(n_jobs=-1)(delayed(process_item)(f) for f in tqdm(files))
    
    # Filter out None results and create DataFrame
    df = pd.DataFrame()
    df["files"] = results
    df.dropna(inplace=True)
    
    # Save results to CSV
    df.to_csv("LLM4Config/data/cleaned_files.csv", index=False)

    print(f"Total valid results: {len(df)}")

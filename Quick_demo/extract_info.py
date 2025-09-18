import os
import re
import csv
from typing import Optional, Union
from pathlib import Path
def extract_birads(text: str) -> Optional[Union[int, float]]:
    """
    Extracts a BI-RADS score (0–6, including decimals like 3.5) from text.
    Returns int/float if found, else None.
    """
    # Try targeted patterns first (closest to how people write it)
    patterns = [
        # e.g., "BIRADS score is likely **4**", "BI-RADS: 4", "BIRADS=4C"
        r'(?i)\bbi[-\s]?rads?(?:\s+score|\s+category)?\s*(?:is|=|:)?\s*(?:likely\s*)?\**\s*([0-6](?:\.\d)?)',
        # e.g., "Category 4 BI-RADS"
        r'(?i)\bcategory\s*([0-6](?:\.\d)?)\b[^\n]{0,20}\bbi[-\s]?rads?\b',
        # e.g., "...BI-RADS... (some words) ... 4"
        r'(?i)\bbi[-\s]?rads?\b[^\d]{0,50}\b([0-6](?:\.\d)?)'
    ]
    
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            val = m.group(1)
            # Normalize things like "4A" (keep numeric part if writer put letters after the digit)
            # (We already capture only numeric part, but this is a safe-guard.)
            val = re.match(r'^([0-6](?:\.\d)?)', val).group(1)
            # Return int if whole number, else float
            return int(val) if val.isdigit() or re.match(r'^[0-6]$', val) else float(val)
    return None

def extract_gt_acr(text):
    """
    Extracts the ACR score (A–D) from the given text.
    Handles both formats: (ACR B) and (ACR-B).
    """
    match = re.search(r'\(ACR[-\s]*([A-Da-d])\)', text)
    if match:
        return match.group(1).upper()
    return None

# Define main method, where we read from ../Reports where we have txt files.
# After extracting the string, we will call the above functions to extract birad score and acr denisty
def main():
    report_dir = "../Reports"
    output_csv = "dmid_gt.csv"
    rows = []

    for root, _, files in os.walk(report_dir):
        for f in files:
            if f.endswith(".txt"):
                file_path = os.path.join(root, f)
                with open(file_path, "r") as file:
                    text = file.read()
                    birads = extract_birads(text)
                    acr = extract_gt_acr(text)
                    rows.append([file_path, birads, acr])

    # Write CSV
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["report_path", "birads", "acr"])  # header
        writer.writerows(rows)

    print(f"Saved {len(rows)} report paths with extracted info to {output_csv}")
if __name__ == "__main__":
    main()
import os
import csv
from pathlib import Path

# Path to your DMID dataset
DMID_ROOT = "/home/ab575577/projects_fall_2025/mammochat/datasets/CDD-CESM/CDD-CESM/Low energy images of CDD-CESM"
OUTPUT_CSV = "cdd_test.csv"

def main():
    # Collect all image files (assuming DMID contains .png/.jpg/.jpeg/.dcm converted images)
    image_extensions = {".tif",".png", ".jpg", ".jpeg"}
    rows = []

    for root, _, files in os.walk(DMID_ROOT):
        for f in files:
            if Path(f).suffix.lower() in image_extensions:
                image_path = os.path.join(root, f)
                rows.append([image_path])

    # Write CSV
    with open(OUTPUT_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_path"])  # header
        writer.writerows(rows)

    print(f"Saved {len(rows)} image paths to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()

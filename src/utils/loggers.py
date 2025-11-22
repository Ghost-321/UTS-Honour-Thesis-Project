import csv
from pathlib import Path

class CSVLogger:
    def __init__(self, path, fieldnames):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = fieldnames
        if not self.path.exists():
            with open(self.path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames); w.writeheader()
    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fieldnames); w.writerow(row)

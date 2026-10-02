import csv, random
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../CST440AudioML2/data
BASE = HERE.parent                          # .../CST440AudioML2
root = BASE / "DataSet"

targets = ["yes", "no", "stop", "go", "marvin"]
skip = set(targets) | {"silence", "_background_noise_"}
rows = []

def wavs(folder):
    return [p for p in (root / folder).glob("*.wav")]

def rel(p):
    return p.relative_to(BASE).as_posix()   # e.g. DataSet/yes/abc.wav

for label in targets:
    rows += [[rel(p), label] for p in wavs(label)]

other_words = [d.name for d in root.iterdir() if d.is_dir() and d.name not in skip]
unknown = [p for w in other_words for p in wavs(w)]
random.seed(0)
random.shuffle(unknown)
rows += [[rel(p), "unknown"] for p in unknown[:3000]]

if (root / "silence").is_dir():
    rows += [[rel(p), "silence"] for p in wavs("silence")]

out_path = HERE / "labels.csv"
with out_path.open("w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["filename", "label"])
    w.writerows(rows)

print("rows:", len(rows))
print("saved to:", out_path)
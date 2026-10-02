import csv, random
from pathlib import Path

HERE = Path(__file__).resolve().parent      # .../CST440AudioML2/data
BASE = HERE.parent                          # .../CST440AudioML2
root = BASE / "DataSet"

PER_FOLDER = 250
random.seed(0)

targets = ["yes", "no", "stop", "go", "marvin", "_background_noise_"]
skip = set(targets) | {"silence"}
rows = []

def sample_wavs(folder):
    files = sorted((root / folder).glob("*.wav"))
    return random.sample(files, min(PER_FOLDER, len(files)))

def rel(p):
    return p.relative_to(BASE).as_posix()   # e.g. DataSet/yes/abc.wav

# target words: 100 from each
for label in targets:
    rows += [[rel(p), label] for p in sample_wavs(label)]

# unknown: 100 from each of the other word folders
other_words = [d.name for d in root.iterdir() if d.is_dir() and d.name not in skip]
for w in other_words:
    rows += [[rel(p), "unknown"] for p in sample_wavs(w)]

# silence: 100 from the clips cut from the background noise
if (root / "silence").is_dir():
    rows += [[rel(p), "silence"] for p in sample_wavs("silence")]

out_path = HERE / "labels.csv"
with out_path.open("w", newline="") as out:
    w = csv.writer(out)
    w.writerow(["filename", "label"])
    w.writerows(rows)

print("unknown words used:", len(other_words))
print("rows:", len(rows))
print("saved to:", out_path)
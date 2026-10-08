"""
prepare_data.py - Step 0: clean the .wav files and make the label list.

Reads   DataSet/<word>/*.wav  and  DataSet/_background_noise_/*.wav
Writes  DataSet_clean/<label>/*.wav   (every clip exactly 1 s, 16 kHz, mono, 16-bit)
        DataSet_clean/_background_noise_/*.wav   (copied, used for augmentation)
        data/dataset.csv   columns: filename, label, speaker, split

What it changes
  * Word clips shorter than 1 s are padded with silence at the end (1366 of them).
  * The 6 long background-noise recordings are cut into 1-second "silence" clips at
    different volumes, plus a few almost-silent clips. This is the "silence" class.
  * Every clip is put in train / val / test BY SPEAKER (the part of the file name before
    "_nohash_"), so the test set only has voices the model has never heard. Noise clips
    are split by time: the first 70% of each noise file is train, then 15% val, 15% test.

Run:  python prepare_data.py
"""
import csv
import hashlib
import shutil
import wave
from pathlib import Path

import numpy as np

from features import CLIP_SAMPLES, SAMPLE_RATE, fix_length

BASE = Path(__file__).resolve().parent
SRC = BASE / "DataSet"
DST = BASE / "DataSet_clean"
CSV_OUT = BASE / "data" / "dataset.csv"

WORDS = ["marvin", "yes", "no", "stop", "go"]     # marvin = the device's name (wake word)
NOISE_DIR = "_background_noise_"
SILENCE_HOP = SAMPLE_RATE // 2                     # a new noise clip every 0.5 s
QUIET_CLIPS = 60                                   # extra almost-silent clips
SEED = 440


def read_wav(path):
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"{path}: expected 16 kHz, mono, 16-bit")
        data = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return data.astype(np.float32) / 32768.0


def write_wav(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def split_for(speaker):
    """Same speaker -> same split, every time (like Google's Speech Commands)."""
    bucket = int(hashlib.sha1(speaker.encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else ("val" if bucket < 85 else "test")


def split_by_position(frac):
    return "train" if frac < 0.70 else ("val" if frac < 0.85 else "test")


def main():
    rng = np.random.default_rng(SEED)
    if DST.exists():
        shutil.rmtree(DST)
    rows, padded = [], 0

    # ---- word clips ----
    for word in WORDS:
        files = sorted((SRC / word).glob("*.wav"))
        if not files:
            raise FileNotFoundError(f"No .wav files in {SRC / word}")
        for f in files:
            audio = read_wav(f)
            if len(audio) < CLIP_SAMPLES:
                padded += 1
            out = DST / word / f.name
            write_wav(out, fix_length(audio))
            speaker = f.name.split("_nohash_")[0]
            rows.append([out.relative_to(BASE).as_posix(), word, speaker, split_for(speaker)])

    # ---- silence clips cut from the background noise ----
    noise_files = sorted((SRC / NOISE_DIR).glob("*.wav"))
    (DST / NOISE_DIR).mkdir(parents=True, exist_ok=True)
    n_silence = 0
    for f in noise_files:
        shutil.copy(f, DST / NOISE_DIR / f.name)
        audio = read_wav(f)
        starts = range(0, len(audio) - CLIP_SAMPLES + 1, SILENCE_HOP)
        for start in starts:
            clip = audio[start:start + CLIP_SAMPLES] * rng.uniform(0.05, 1.0)
            # a clip belongs to a split only if it lies fully inside that part of the file
            first = split_by_position(start / len(audio))
            last = split_by_position((start + CLIP_SAMPLES - 1) / len(audio))
            if first != last:
                continue
            out = DST / "silence" / f"{f.stem}_{start // SILENCE_HOP:04d}.wav"
            write_wav(out, clip)
            rows.append([out.relative_to(BASE).as_posix(), "silence", f.stem, first])
            n_silence += 1
    for i in range(QUIET_CLIPS):
        clip = rng.normal(0, rng.uniform(0.0002, 0.003), CLIP_SAMPLES)
        out = DST / "silence" / f"quiet_{i:03d}.wav"
        write_wav(out, clip)
        rows.append([out.relative_to(BASE).as_posix(), "silence", "quiet",
                     split_by_position(i / QUIET_CLIPS)])
        n_silence += 1

    CSV_OUT.parent.mkdir(exist_ok=True)
    with CSV_OUT.open("w", newline="") as out:
        w = csv.writer(out)
        w.writerow(["filename", "label", "speaker", "split"])
        w.writerows(rows)

    # ---- summary ----
    print(f"Padded {padded} short word clips to 1 second")
    print(f"Made {n_silence} silence clips from {len(noise_files)} noise files")
    print(f"\n{'label':<10}{'train':>7}{'val':>7}{'test':>7}{'total':>8}")
    for label in WORDS + ["silence"]:
        counts = [sum(1 for r in rows if r[1] == label and r[3] == s) for s in ("train", "val", "test")]
        print(f"{label:<10}{counts[0]:>7}{counts[1]:>7}{counts[2]:>7}{sum(counts):>8}")
    print(f"\nSaved {len(rows)} rows to {CSV_OUT.relative_to(BASE)}")


if __name__ == "__main__":
    main()

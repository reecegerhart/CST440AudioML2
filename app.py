"""
app.py - Steps 4-5: the speech-command application and its tests.

The device is named "marvin". It sleeps until it hears its name, then listens for one
command (yes / no / stop / go) for 3 seconds, reacts, and goes back to sleep.

Works with any of these model files:
  models/speech_model.keras          (Keras, float)
  models/speech_model_float.tflite   (TensorFlow Lite, float)
  models/speech_model_int8.tflite    (after your teammate quantizes it)

  python app.py --test                          # accuracy report on the test speakers
  python app.py --test --model models/speech_model_int8.tflite   # same, quantized model
  python app.py --wav DataSet_clean/yes/004ae714_nohash_0.wav   # classify .wav files
  python app.py --demo                          # make a demo recording and run the device on it
  python app.py --stream my_recording.wav       # run the device on any long recording
  python app.py --mic                           # live, with your microphone (pip install sounddevice)
  python app.py --mic --gain 4                  # same, mic made 4x louder (see mic_check.py)
"""
import argparse
import csv
import os
import time
import wave
from pathlib import Path

import numpy as np
import tensorflow as tf

import features as F

BASE = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE / "models" / "speech_model_float.tflite"
CSV_PATH = BASE / "data" / "dataset.csv"
WAKE_WORD = "marvin"
COMMANDS = {"yes": "Confirmed!", "no": "Cancelled.", "stop": "Stopping.", "go": "Starting!"}
LISTEN_SECONDS = 3.0      # how long the device listens for a command after its name
HOP = 0.25                # run the model every 250 ms on the last 1 second of audio


# ----------------------------------------------------------------------------
# Model wrapper: same interface for .keras, float .tflite and int8 .tflite
# ----------------------------------------------------------------------------
class SpeechModel:
    def __init__(self, path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path} (run train.py first)")
        self.path, self.size_bytes = path, os.path.getsize(path)
        labels_file = BASE / "models" / "labels.txt"
        self.labels = labels_file.read_text().split()
        if path.suffix == ".keras":
            self.keras = tf.keras.models.load_model(path)
            self.kind = "Keras float32"
        else:
            self.keras = None
            self.interp = tf.lite.Interpreter(model_path=str(path))
            self.interp.allocate_tensors()
            self.inp = self.interp.get_input_details()[0]
            self.out = self.interp.get_output_details()[0]
            self.kind = f"TFLite {np.dtype(self.inp['dtype']).name}"

    def probabilities(self, feats):
        """feats: (n, 49, 40, 1) float32 -> (n, classes) probabilities."""
        if self.keras is not None:
            return self.keras.predict(feats, verbose=0)
        results = []
        for x in feats:
            x = x[np.newaxis].astype(np.float32)
            if self.inp["dtype"] == np.int8:                      # quantize the input
                scale, zero = self.inp["quantization"]
                x = np.clip(np.round(x / scale + zero), -128, 127).astype(np.int8)
            self.interp.set_tensor(self.inp["index"], x)
            self.interp.invoke()
            y = self.interp.get_tensor(self.out["index"])[0].astype(np.float32)
            if self.out["dtype"] == np.int8:                      # dequantize the output
                scale, zero = self.out["quantization"]
                y = (y - zero) * scale
            results.append(y)
        return np.array(results)

    def classify(self, audio):
        """1 second of audio -> (label, confidence, all probabilities)."""
        p = self.probabilities(F.features(F.fix_length(audio)))[0]
        return self.labels[p.argmax()], float(p.max()), p


# ----------------------------------------------------------------------------
# Step 5 - test on the held-out test speakers
# ----------------------------------------------------------------------------
def run_test(model, split="test"):
    from train import LABELS, report_text
    with CSV_PATH.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    waves = np.stack([F.fix_length(F.load_wav(BASE / r["filename"])) for r in rows])
    y_true = np.array([LABELS.index(r["label"]) for r in rows])
    feats = F.features(waves)

    start = time.perf_counter()
    probs = model.probabilities(feats)
    ms = (time.perf_counter() - start) / len(feats) * 1000
    y_pred = probs.argmax(axis=1)

    print(f"\nModel: {model.path.name} ({model.kind}), size {model.size_bytes / 1024:.1f} KB")
    print(f"Average inference time: {ms:.2f} ms per clip (on this computer)\n")
    print(report_text(y_true, y_pred, f"=== {split} set: {len(rows)} clips ==="))

    # wake-word threshold sweep: how sure must the model be before the device wakes up?
    w = LABELS.index(WAKE_WORD)
    said = y_true == w
    print(f"\nWake-word threshold sweep ('{WAKE_WORD}' must be the top word AND above the threshold):")
    print(f"{'threshold':>10}{'missed':>10}{'false wake-ups':>16}")
    for t in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
        woke = (y_pred == w) & (probs[:, w] >= t)
        print(f"{t:>10.2f}{np.mean(~woke[said]) * 100:>9.2f}%{np.mean(woke[~said]) * 100:>15.2f}%")


# ----------------------------------------------------------------------------
# Step 4 - the application: wake word + command, on a stream of audio
# ----------------------------------------------------------------------------
class Device:
    """Sleeps until it hears 'marvin', then accepts one command for LISTEN_SECONDS."""

    def __init__(self, model, threshold):
        self.model, self.threshold = model, threshold
        self.awake_until, self.cooldown_until, self.events = -1.0, -1.0, []

    SMOOTH = 3   # average the last 3 guesses (0.75 s) so a half-heard word can't trigger

    def step(self, window, now):
        _, _, probs = self.model.classify(window)
        self.recent = (getattr(self, "recent", []) + [probs])[-self.SMOOTH:]
        avg = np.mean(self.recent, axis=0)
        label, conf = self.model.labels[int(avg.argmax())], float(avg.max())
        if now < self.cooldown_until or conf < self.threshold:
            if self.awake_until > 0 and now > self.awake_until:
                self._say(now, "(no command heard, going back to sleep)")
                self.awake_until = -1.0
            return
        if self.awake_until < 0 and label == WAKE_WORD:
            self._say(now, f"Heard my name ({conf:.0%}). Listening for a command...")
            self.awake_until = now + LISTEN_SECONDS
            self.cooldown_until = now + 0.75          # don't hear the same word twice
        elif self.awake_until > 0 and label in COMMANDS:
            self._say(now, f"Command '{label}' ({conf:.0%}) -> {COMMANDS[label]}")
            self.awake_until, self.cooldown_until = -1.0, now + 0.75
        elif self.awake_until > 0 and now > self.awake_until:
            self._say(now, "(no command heard, going back to sleep)")
            self.awake_until = -1.0

    def _say(self, now, text):
        self.events.append((now, text))
        print(f"  [{now:6.2f} s] {text}")


def run_stream(model, audio, threshold):
    """Slide a 1-second window over a long recording, like the microcontroller does."""
    device, hop = Device(model, threshold), int(HOP * F.SAMPLE_RATE)
    print(f"\nRunning the device on {len(audio) / F.SAMPLE_RATE:.1f} s of audio "
          f"(threshold {threshold:.0%}):")
    for end in range(F.CLIP_SAMPLES, len(audio) + 1, hop):
        device.step(audio[end - F.CLIP_SAMPLES:end], end / F.SAMPLE_RATE)
    return device.events


def make_demo(path, seed=7):
    """Build a test recording from TEST-set clips: noise, marvin + command, a command
    without marvin first (should be ignored), marvin + command again."""
    rng = np.random.default_rng(seed)
    with CSV_PATH.open(newline="") as f:
        test = [r for r in csv.DictReader(f) if r["split"] == "test"]
    pick = lambda lab: F.load_wav(BASE / rng.choice([r["filename"] for r in test if r["label"] == lab]))
    gap = lambda s: np.zeros(int(s * F.SAMPLE_RATE), np.float32)
    script = ["marvin", "yes", None, "stop", None, "marvin", "go", None, "marvin", "no"]
    parts, said = [gap(1.0)], []
    for item in script:
        if item is None:
            parts.append(gap(1.5)); said.append("(pause)")
        else:
            parts += [F.fix_length(pick(item)), gap(0.3)]; said.append(item)
    audio = np.concatenate(parts)
    noise = F.load_wav(next((BASE / "DataSet_clean" / "_background_noise_").glob("running_tap.wav")))
    audio = audio + 0.03 * np.resize(noise, len(audio))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(F.SAMPLE_RATE)
        w.writeframes(np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes())
    print(f"Saved {path.name}: " + ", ".join(said))
    print("Expected: wake on marvin -> yes; ignore 'stop' (no name first); "
          "wake -> go; wake -> no")
    return audio


def run_mic(model, threshold, gain=1.0, device_id=None):
    try:
        import sounddevice as sd
    except ImportError:
        raise SystemExit("Install the microphone package first:  pip install sounddevice")
    hop = int(HOP * F.SAMPLE_RATE)
    buf = np.zeros(F.CLIP_SAMPLES, np.float32)
    device, start = Device(model, threshold), time.time()
    print(f"Listening... say '{WAKE_WORD}', then yes / no / stop / go.  Ctrl+C to quit.")
    print("(Nothing happening? Run  python mic_check.py  to check your microphone.)")
    with sd.InputStream(samplerate=F.SAMPLE_RATE, channels=1, dtype="float32", blocksize=hop,
                        device=device_id) as s:
        try:
            while True:
                chunk, _ = s.read(hop)
                buf = np.concatenate([buf[hop:], chunk[:, 0] * gain])
                device.step(buf, time.time() - start)
        except KeyboardInterrupt:
            print("\nStopped.")


def main():
    p = argparse.ArgumentParser(description="Speech command app (wake word 'marvin')")
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--test", action="store_true", help="accuracy report on the test set")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--wav", nargs="+", help="classify one or more 1-second .wav files")
    p.add_argument("--demo", action="store_true", help="make demo.wav and run the device on it")
    p.add_argument("--stream", help="run the device on a long .wav recording")
    p.add_argument("--mic", action="store_true", help="live microphone")
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--gain", type=float, default=1.0, help="mic volume boost, e.g. 4")
    p.add_argument("--device", type=int, default=None, help="mic number (python mic_check.py --list)")
    a = p.parse_args()

    model = SpeechModel(a.model)
    print(f"Loaded {model.path.name} ({model.kind}, {model.size_bytes / 1024:.1f} KB)")
    if a.test:
        run_test(model, a.split)
    for path in a.wav or []:
        start = time.perf_counter()
        label, conf, probs = model.classify(F.load_wav(path))
        ms = (time.perf_counter() - start) * 1000
        top = np.argsort(probs)[::-1][:3]
        print(f"{path}: {label} ({conf:.0%})  | top 3: "
              + ", ".join(f"{model.labels[i]} {probs[i]:.0%}" for i in top) + f" | {ms:.1f} ms")
    if a.demo:
        run_stream(model, make_demo(BASE / "demo.wav"), a.threshold)
    if a.stream:
        run_stream(model, F.load_wav(a.stream), a.threshold)
    if a.mic:
        run_mic(model, a.threshold, a.gain, a.device)
    if not (a.test or a.wav or a.demo or a.stream or a.mic):
        p.print_help()


if __name__ == "__main__":
    main()

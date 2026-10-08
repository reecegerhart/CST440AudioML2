"""
mic_check.py - Find out why the microphone demo isn't reacting.

  python mic_check.py              # record 3 seconds, show volume + what the model heard
  python mic_check.py --live       # live view: volume bar + best guess 4 times a second
  python mic_check.py --gain 4     # make the mic 4x louder (if it says "too quiet")
  python mic_check.py --list       # list microphones
  python mic_check.py --device 2   # use microphone number 2 from --list

The 3-second recording is saved as my_recording.wav so you can listen to it.
"""
import argparse
import time
import wave

import numpy as np

import features as F
from app import DEFAULT_MODEL, SpeechModel


def volume_bar(audio, width=30):
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return peak, "#" * int(min(peak, 1.0) * width) + "." * (width - int(min(peak, 1.0) * width))


def best_window(model, audio):
    """Slide a 1-second window over the recording; return the most confident guess."""
    best = (None, 0.0, None, 0.0)
    for end in range(F.CLIP_SAMPLES, len(audio) + 1, F.SAMPLE_RATE // 8):
        label, conf, probs = model.classify(audio[end - F.CLIP_SAMPLES:end])
        if label != "silence" and conf > best[1]:
            best = (label, conf, probs, end / F.SAMPLE_RATE)
    return best


def save_wav(path, audio):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(F.SAMPLE_RATE)
        w.writeframes(np.clip(audio * 32768, -32768, 32767).astype(np.int16).tobytes())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(DEFAULT_MODEL))
    p.add_argument("--live", action="store_true")
    p.add_argument("--gain", type=float, default=1.0)
    p.add_argument("--device", type=int, default=None)
    p.add_argument("--list", action="store_true")
    p.add_argument("--file", help="check a .wav file instead of the microphone")
    a = p.parse_args()

    try:
        import sounddevice as sd
    except (ImportError, OSError):   # OSError: no audio driver found
        sd = None
        if not a.file:
            raise SystemExit("Install it first:  python -m pip install sounddevice")
    if a.list:
        print(sd.query_devices())
        print("\nThe microphone marked '>' is the default. Use --device <number> to pick another.")
        return

    model = SpeechModel(a.model)
    if sd and not a.file:
        dev = sd.query_devices(a.device if a.device is not None else sd.default.device[0])
        print(f"Microphone: {dev['name']}")

    if a.live:
        hop = F.SAMPLE_RATE // 4
        buf = np.zeros(F.CLIP_SAMPLES, np.float32)
        print("Live view. Say 'marvin', 'yes', 'no', 'stop', 'go'. Ctrl+C to quit.\n")
        with sd.InputStream(samplerate=F.SAMPLE_RATE, channels=1, dtype="float32",
                            blocksize=hop, device=a.device) as s:
            try:
                while True:
                    chunk, _ = s.read(hop)
                    buf = np.concatenate([buf[hop:], chunk[:, 0] * a.gain])
                    label, conf, _ = model.classify(buf)
                    peak, bar = volume_bar(chunk[:, 0] * a.gain)
                    print(f"volume [{bar}] {peak:4.2f}   heard: {label:<8} {conf:4.0%}")
            except KeyboardInterrupt:
                print("\nStopped.")
        return

    if a.file:
        audio = F.load_wav(a.file) * a.gain
    else:
        print("\nGet ready... say 'marvin' clearly when you see RECORDING.")
        time.sleep(1.5)
        print("RECORDING (3 seconds)...")
        audio = sd.rec(3 * F.SAMPLE_RATE, samplerate=F.SAMPLE_RATE, channels=1,
                       dtype="float32", device=a.device)
        sd.wait()
        audio = audio[:, 0] * a.gain
        save_wav("my_recording.wav", audio)
        print("Saved my_recording.wav (double-click it to listen)")

    peak, bar = volume_bar(audio)
    print(f"\nLoudest point: [{bar}] {peak:.2f}")
    if peak < 0.01:
        print("-> NO SOUND. The app is not getting your microphone. Check Windows Settings ->"
              " Privacy & security -> Microphone, or try another one with --list / --device.")
        return
    if peak < 0.15:
        print(f"-> TOO QUIET. Try:  python mic_check.py --gain {min(round(0.5 / peak), 20)}")
    elif peak > 0.98:
        print("-> TOO LOUD (clipping). Move back from the mic or try --gain 0.5")
    else:
        print("-> Volume is OK.")

    label, conf, probs, at = best_window(model, audio)
    if label is None:
        print("The model only heard silence/noise. Speak louder or closer, or raise --gain.")
        return
    top = np.argsort(probs)[::-1][:3]
    print(f"Best guess: '{label}' ({conf:.0%}) at {at:.1f} s   | top 3: "
          + ", ".join(f"{model.labels[i]} {probs[i]:.0%}" for i in top))
    m = probs[model.labels.index("marvin")]
    if label == "marvin" and conf >= 0.6:
        print("-> It understood 'marvin'. The --mic demo should work with this volume/gain.")
    else:
        print(f"-> 'marvin' only got {m:.0%}. Try saying it slower: 'MAR-vin', stress on MAR.")


if __name__ == "__main__":
    main()

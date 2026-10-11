# CST-440 Speech Recognition on a Microcontroller

A small neural network that recognizes **5 spoken words plus silence**. The device is named
**"marvin"**: it sleeps until it hears its name, then listens for one command
(**yes, no, stop, go**) and reacts. The model is trained in Python with TensorFlow/Keras,
converted to TensorFlow Lite, and is small enough to quantize to int8 and run on a
microcontroller.

| Class | Role |
|---|---|
| `marvin` | Wake word (the device's name) |
| `yes`, `no`, `stop`, `go` | Commands |
| `silence` | Background noise / nothing said |

## What's in this repo

| File | What it is |
|---|---|
| `features.py` | Turns 1 s of audio into a 49×40 log-mel spectrogram. Shared by all scripts |
| `prepare_data.py` | Step 0: cleans the .wav files and writes `data/dataset.csv` |
| `train.py` | Steps 1-3: designs, builds and trains the model; saves models, plots, results |
| `app.py` | Steps 4-5: the wake-word application and the accuracy tests |
| `quantize.py` | Step 6 prep: int8 quantization, accuracy comparison, C array for the Arduino |
| `mic_check.py` | Checks the microphone: volume, and what the model hears |
| `DataSet/` | Original clips (not changed) |
| `DataSet_clean/` | Cleaned clips (**made by `prepare_data.py`, don't edit by hand**) |
| `data/dataset.csv` | Every clean clip with its label, speaker and train/val/test split |
| `data/representative_features.npy` | 300 sample inputs (50 per class) for int8 quantization |
| `models/speech_model.keras` | Trained model (input to quantization) |
| `models/speech_model_float.tflite` | Float32 TensorFlow Lite model (accuracy baseline) |
| `models/labels.txt` | Class names in the model's output order |
| `models/features.json` | Feature settings the microcontroller code must copy |
| `models/training_history.png`, `models/confusion_matrix.png` | Graphs for the report |
| `models/test_results.txt` | Test-set accuracy report |
| `models/speech_model_int8.tflite` | **int8 model for the Arduino** (made by `quantize.py`) |
| `models/model_data.h` | int8 model as a C array + quantization scales, for the Arduino sketch |
| `models/speech_model_dynamic.tflite` | int8-weights-only model, for comparison in the report |
| `models/quantization_results.txt`, `models/confusion_matrix_int8.png` | Quantization results for the report |

The trained model files are included, so **you don't have to train to run the app**.
You do need to run `prepare_data.py` once, because `DataSet_clean/` is too big to commit.

---

## Installation (VS Code, Windows)

1. Install **Python 3.12** from <https://www.python.org/downloads/> and tick
   **"Add python.exe to PATH"**. (3.10-3.13 also work; don't use 3.14.)
2. Open the project folder in VS Code (**File → Open Folder**).
3. Open a terminal (**Terminal → New Terminal**) and create a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   ```

   If PowerShell says *"running scripts is disabled"*, run
   `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, then activate again.
4. Install the packages:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Press `Ctrl+Shift+P` → **Python: Select Interpreter** → pick the one with `.venv`.

macOS / Linux: use `python3.12 -m venv .venv` and `source .venv/bin/activate`.

---

## Running the project

Run all commands from the project folder with `.venv` active.

### 1. Clean the data (once, about 1 minute)

```powershell
python prepare_data.py
```

This makes every clip exactly 1 second / 16 kHz / mono / 16-bit, pads the 1,366 short
clips, cuts the 6 background-noise recordings into 831 one-second `silence` clips, and
splits by speaker: 70% train, 15% validation, 15% test.

| label | train | val | test | total |
|---|---|---|---|---|
| marvin | 1414 | 342 | 344 | 2100 |
| yes | 2686 | 662 | 696 | 4044 |
| no | 2693 | 591 | 657 | 3941 |
| stop | 2580 | 608 | 684 | 3872 |
| go | 350 | 80 | 62 | 492 |
| silence | 593 | 119 | 119 | 831 |

### 2. Train (optional)

```powershell
python train.py
```

This overwrites everything in `models/` and `data/representative_features.npy`.

### 3. Test the accuracy

```powershell
python app.py --test
```

Prints overall accuracy, precision/recall per word, the wake-word miss and false-wake-up
rates, a threshold sweep, model size and inference time.

### 4. Try the application

```powershell
python app.py --wav DataSet_clean/yes/004ae714_nohash_0.wav   # one or more 1-second clips
python app.py --demo                    # builds demo.wav from test clips and runs the device on it
python app.py --stream my_recording.wav # run the device on any 16 kHz recording
pip install sounddevice
python app.py --mic                     # live microphone: say "marvin", then a command
```

The device averages its last 3 guesses (0.75 s) before acting, and only acts when that
average is above the threshold (default 0.6). Use `--threshold 0.8` to make it harder to wake
up (fewer false wake-ups, more misses).

### 5. Microphone not reacting?

```powershell
python mic_check.py            # records 3 s, shows the volume and what the model heard
python mic_check.py --live     # live volume bar + best guess, 4 times a second
python mic_check.py --gain 4   # try a louder mic if it says TOO QUIET
python mic_check.py --list     # list microphones; pick one with --device <number>
```

Then use the same `--gain` / `--device` with the app, e.g. `python app.py --mic --gain 4`.
A quiet microphone is the most common problem: very quiet speech is often heard as "no".

---

## Results (test set: 2,562 clips from speakers)

| Model | Size | Test accuracy |
|---|---|---|
| `speech_model.keras` / `speech_model_float.tflite` | 243 KB | **96.49%** |
| `speech_model_int8.tflite` (for the Arduino) | 70.5 KB | **96.60%** |

| class | float precision | float recall | int8 precision | int8 recall |
|---|---|---|---|---|
| marvin | 0.982 | 0.927 | 0.982 | 0.933 |
| yes | 0.988 | 0.953 | 0.987 | 0.953 |
| no | 0.903 | 0.992 | 0.904 | 0.994 |
| stop | 0.997 | 0.981 | 0.999 | 0.981 |
| go | 0.960 | 0.774 | 0.980 | 0.806 |
| silence | 0.983 | 1.000 | 0.992 | 0.983 |

Wake word "marvin" at a 0.8 threshold (single clips): misses 14.2% (int8: 13.7%) of the times
it is said, and never wakes up by mistake on the 2,218 other test clips.

Known weak spots: **go** has only 492 recordings (the other words have 2,100-4,000), so it
is sometimes heard as "no". Some loud noise clips are heard as "stop".

`demo.wav` result: woke on each "marvin" and carried out yes / go / no, and ignored a
"stop" that was said without the name first.

---

## Model design

```
input 49 x 40 x 1 log-mel spectrogram (1 s of audio)
Normalization (per mel band)
Conv 3x3, 16 filters -> BatchNorm -> ReLU -> MaxPool 2x2
Conv 3x3, 32 filters -> BatchNorm -> ReLU -> MaxPool 2x2
Conv 3x3, 64 filters -> BatchNorm -> ReLU -> MaxPool 2x2
Conv 3x3, 64 filters -> BatchNorm -> ReLU
GlobalAveragePooling -> Dropout 0.3 -> Dense 6 (softmax)
61,223 parameters
```

Training: Adam (learning rate 0.001, halved when validation loss stalls), batch 64, up to
40 epochs with early stopping. Every training clip is randomly shifted ±100 ms, its
volume changed ±20%, and 80% get background noise mixed in. "go" clips are repeated 4×,
and class weights balance the rest.

---

## Quantization (int8) for the microcontroller

```powershell
python quantize.py                                              # needs prepare_data.py first
python app.py --test --demo --model models/speech_model_int8.tflite   # double-check
```

**Method:** full-integer post-training quantization. `TFLiteConverter` with
`Optimize.DEFAULT`, a representative dataset (`data/representative_features.npy`: 300 clean
training clips, 50 per class, fed one at a time) and `TFLITE_BUILTINS_INT8` with int8
input and output. The converter runs the representative clips through the model to find
the range of every layer's values, then stores each tensor as int8 with a scale and zero
point: `real = scale × (int8 − zero_point)`. BatchNorm is folded into the conv weights and
ReLU is fused into the conv, so every tensor in the final model is int8 (biases int32).

| Model | Weights / math | Size | Test accuracy | Runs on Arduino? |
|---|---|---|---|---|
| `speech_model_float.tflite` | float32 / float32 | 242.8 KB | 96.49% | too big / slow |
| `speech_model_dynamic.tflite` | int8 / float32 | 69.3 KB | 96.49% | no (needs float kernels) |
| **`speech_model_int8.tflite`** | **int8 / int8** | **70.5 KB** | **96.60%** | **yes** |

- 3.4× smaller than float32, and accuracy went *up* 0.12 points (from 2,472 to 2,475 of
  2,562 test clips correct); int8 and float give the same answer on 99.26% of clips.
  **The 95% target is met.**
- `demo.wav` with the int8 model: woke on every "marvin", carried out yes / go / no, and
  ignored the "stop" said without the name.
- Full output: `models/quantization_results.txt`; plot: `models/confusion_matrix_int8.png`.

### For the Arduino sketch

- `#include "model_data.h"`: gives `g_model` / `g_model_len` (16-byte aligned), the
  input/output scales and zero points, and `g_labels`.
- The model uses 7 ops; register exactly these in a `tflite::MicroMutableOpResolver<7>`:
  `AddConv2D`, `AddMaxPool2D`, `AddMean`, `AddFullyConnected`, `AddSoftmax`, `AddSub`, `AddMul`
  (`Sub`/`Mul` are the normalization layer, `Mean` is global average pooling).
- **Input:** compute the log-mel features as float, then
  `int8 = clamp(round(log_mel / 0.07465912) + 57, -128, 127)`.
- **Output:** `probability = (int8 + 128) / 256` (scale 0.00390625, zero point −128).
- The features must match `models/features.json` exactly: 16 kHz audio, 480-sample Hann
  window, 320-sample step, 512-point FFT magnitude, 40 mel bands (20-7600 Hz), natural log
  of (mel + 1e-6). The output order is in `models/labels.txt`.
- Re-run `quantize.py` any time `train.py` is re-run: the scales change with every new model.

---

## Deploying to the Arduino Nano 33 BLE Sense Rev2

The sketch is in `arduino/speech_commands/`. It listens with the board's built-in microphone,
runs the int8 model back to back (~605 ms per run on the board), averages the last 2 guesses,
and uses the same wake-word logic as `app.py`. The mic amplifier is set to +6 dB
(`kPdmGain = 52`); the library default of -10 dB was ~4x too quiet and made "marvin" sound
like "no".

1. Arduino IDE: **Boards Manager → "Arduino Mbed OS Nano Boards"**, **Library Manager →
   "Chirale_TensorFLowLite"**.
2. After any new `quantize.py` run: `python export_arduino.py` (writes `mel_weights.h`, copies
   `model_data.h` into the sketch, and checks the sketch's feature code on the PC: it must
   match `features.py`; the int8 model gets 96.60% with it).
3. Open `arduino/speech_commands/speech_commands.ino`, pick board **Arduino Nano 33 BLE**,
   upload, and open the Serial Monitor at **115200** baud.
4. Say "marvin", then yes / no / stop / go. Orange LED = listening for a command;
   RGB LED: yes = green, no = red, go = blue, stop = yellow.
5. Not reacting? Type `d` + Enter for debug (shows every guess, loudness and inference
   time) and `+` / `-` to change the extra software gain. Speech should reach a loudness of
   ~0.2-0.9.

Build size: 248,544 bytes flash (25%), 152,992 bytes RAM (58%).

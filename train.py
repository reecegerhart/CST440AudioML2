"""
train.py - Steps 1-3: design, build and train the speech-command model.

Needs data/dataset.csv from prepare_data.py.

Step 1  Design : 49x40 log-mel spectrogram -> 4 small conv blocks -> 6 outputs
                 (marvin, yes, no, stop, go, silence). Small enough for a microcontroller.
Step 2  Build  : Keras Sequential model (Conv2D + BatchNorm + ReLU + MaxPool)
Step 3  Train  : Adam, cross-entropy, data augmentation (time shift, background noise,
                 volume), class weights + extra copies of "go" because it has few clips,
                 early stopping on the validation set.

Writes to models/
  speech_model.keras          trained Keras model (give this to the quantization step)
  speech_model_float.tflite   float32 TensorFlow Lite model (accuracy baseline)
  labels.txt                  class names, in the model's output order
  features.json               feature settings the microcontroller must copy
  training_history.png        accuracy / loss curves
  confusion_matrix.png        test-set confusion matrix
  test_results.txt            accuracy report for the scientific report
and data/representative_features.npy  (sample inputs for int8 quantization)

Run:  python train.py
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import features as F

BASE = Path(__file__).resolve().parent
CSV_PATH = BASE / "data" / "dataset.csv"
MODEL_DIR = BASE / "models"
NOISE_DIR = BASE / "DataSet_clean" / "_background_noise_"

LABELS = ["marvin", "yes", "no", "stop", "go", "silence"]   # output order
EXTRA_COPIES = {"go": 4}      # "go" has ~350 training clips vs ~2600 for "yes"
BATCH_SIZE = 64
MAX_EPOCHS = 40
PATIENCE = 6
SHIFT_SAMPLES = 1600          # random time shift of up to +-100 ms
NOISE_PROB = 0.8              # 80% of training clips get background noise mixed in
NOISE_MAX_VOLUME = 0.1
SEED = 42


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def read_split(split):
    with CSV_PATH.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == split]
    paths = [str(BASE / r["filename"]) for r in rows]
    labels = [LABELS.index(r["label"]) for r in rows]
    return paths, np.array(labels)


def load_split_features(split):
    """Clean (not augmented) features for validation / test."""
    paths, y = read_split(split)
    waves = np.stack([F.fix_length(F.load_wav(p)) for p in paths])
    return F.features(waves), y


def training_noise():
    """Only the first 70% of each noise file, the part prepare_data.py gave to 'train'."""
    parts = []
    for f in sorted(NOISE_DIR.glob("*.wav")):
        audio = F.load_wav(f)
        parts.append(audio[: int(len(audio) * 0.70)])
    return tf.constant(np.concatenate(parts), tf.float32)


def make_train_dataset(paths, labels, noise):
    noise_len = int(noise.shape[0])

    def load(path, label):
        audio, _ = tf.audio.decode_wav(tf.io.read_file(path), desired_channels=1)
        audio = tf.squeeze(audio, -1)[: F.CLIP_SAMPLES]
        audio = tf.pad(audio, [[0, F.CLIP_SAMPLES - tf.shape(audio)[0]]])
        return audio, label

    def augment(audio, label):
        # 1) random time shift
        padded = tf.pad(audio, [[SHIFT_SAMPLES, SHIFT_SAMPLES]])
        start = tf.random.uniform([], 0, 2 * SHIFT_SAMPLES + 1, tf.int32)
        audio = padded[start:start + F.CLIP_SAMPLES]
        # 2) random volume
        audio = audio * tf.random.uniform([], 0.8, 1.2)
        # 3) mix in a random piece of background noise
        n_start = tf.random.uniform([], 0, noise_len - F.CLIP_SAMPLES, tf.int32)
        piece = noise[n_start:n_start + F.CLIP_SAMPLES]
        use = tf.cast(tf.random.uniform([]) < NOISE_PROB, tf.float32)
        audio = audio + use * tf.random.uniform([], 0.0, NOISE_MAX_VOLUME) * piece
        return tf.clip_by_value(audio, -1.0, 1.0), label

    def to_features(audio, label):
        return F.tf_features(audio), label

    return (tf.data.Dataset.from_tensor_slices((paths, labels))
            .shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
            .map(load, num_parallel_calls=tf.data.AUTOTUNE)
            .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE)
            .map(to_features, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


# ----------------------------------------------------------------------------
# Steps 1-2 - Design and build the model
# ----------------------------------------------------------------------------
def conv_block(filters, pool=True):
    block = [layers.Conv2D(filters, 3, padding="same", use_bias=False),
             layers.BatchNormalization(),
             layers.ReLU()]
    if pool:
        block.append(layers.MaxPooling2D(2))
    return block


def build_model(norm_sample):
    norm = layers.Normalization(axis=2, name="normalize")   # one mean/std per mel band
    norm.adapt(norm_sample)
    model = keras.Sequential(
        [layers.Input(shape=F.INPUT_SHAPE, name="log_mel"), norm]
        + conv_block(16) + conv_block(32) + conv_block(64) + conv_block(64, pool=False)
        + [layers.GlobalAveragePooling2D(),
           layers.Dropout(0.3),
           layers.Dense(len(LABELS), activation="softmax", name="output")],
        name="speech_commands")
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------------
# Reports and plots
# ----------------------------------------------------------------------------
def confusion(y_true, y_pred):
    m = np.zeros((len(LABELS), len(LABELS)), int)
    for t, p in zip(y_true, y_pred):
        m[t, p] += 1
    return m


def report_text(y_true, y_pred, title):
    m = confusion(y_true, y_pred)
    lines = [title, f"Overall accuracy: {np.mean(y_true == y_pred) * 100:.2f}%  "
                    f"({np.sum(y_true == y_pred)}/{len(y_true)} clips)", "",
             f"{'class':<9}{'clips':>7}{'precision':>11}{'recall':>9}{'f1':>8}"]
    for i, name in enumerate(LABELS):
        tp, support, predicted = m[i, i], m[i].sum(), m[:, i].sum()
        prec = tp / predicted if predicted else 0.0
        rec = tp / support if support else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"{name:<9}{support:>7}{prec:>11.3f}{rec:>9.3f}{f1:>8.3f}")
    # wake word view: is it "marvin" or not?
    w = LABELS.index("marvin")
    is_w, said_w = y_pred == w, y_true == w
    lines += ["", "Wake word 'marvin':",
              f"  missed (false reject): {np.sum(said_w & ~is_w)} of {said_w.sum()} "
              f"= {np.mean(~is_w[said_w]) * 100:.2f}%",
              f"  woke up by mistake (false accept): {np.sum(~said_w & is_w)} of {(~said_w).sum()} "
              f"= {np.mean(is_w[~said_w]) * 100:.2f}%",
              "", "Confusion matrix (rows = true word, columns = predicted word):",
              " " * 9 + "".join(f"{n:>8}" for n in LABELS)]
    lines += [f"{n:<9}" + "".join(f"{v:>8}" for v in row) for n, row in zip(LABELS, m)]
    return "\n".join(lines)


def plot_history(history, path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for key, name in (("accuracy", "train"), ("val_accuracy", "validation")):
        a1.plot(np.arange(1, len(history.history[key]) + 1), history.history[key], label=name)
    for key, name in (("loss", "train"), ("val_loss", "validation")):
        a2.plot(np.arange(1, len(history.history[key]) + 1), history.history[key], label=name)
    a1.set(title="Accuracy", xlabel="Epoch", ylabel="accuracy")
    a2.set(title="Loss (cross-entropy)", xlabel="Epoch", ylabel="loss")
    a1.legend(); a2.legend(); a1.grid(alpha=0.3); a2.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_confusion(y_true, y_pred, path):
    m = confusion(y_true, y_pred)
    pct = m / m.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45)
    ax.set_yticks(range(len(LABELS)), LABELS)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{m[i, j]}\n{pct[i, j]:.0f}%", ha="center", va="center", fontsize=8,
                    color="white" if pct[i, j] > 60 else "black")
    ax.set(xlabel="Predicted word", ylabel="True word", title="Test set confusion matrix")
    fig.colorbar(im, ax=ax, label="% of true word")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


# ----------------------------------------------------------------------------
def main():
    keras.utils.set_random_seed(SEED)
    MODEL_DIR.mkdir(exist_ok=True)

    train_paths, train_y = read_split("train")
    for word, copies in EXTRA_COPIES.items():          # oversample the small class
        idx = np.where(train_y == LABELS.index(word))[0]
        train_paths += [train_paths[i] for i in idx] * (copies - 1)
        train_y = np.concatenate([train_y, train_y[idx].repeat(copies - 1)])
    counts = np.bincount(train_y, minlength=len(LABELS))
    class_weight = {i: len(train_y) / (len(LABELS) * c) for i, c in enumerate(counts)}
    print("Training clips per class (after extra 'go' copies):")
    for name, c in zip(LABELS, counts):
        print(f"  {name:<8} {c:>5}   weight {class_weight[LABELS.index(name)]:.2f}")

    print("Loading validation and test sets...")
    X_val, y_val = load_split_features("val")
    X_test, y_test = load_split_features("test")
    print(f"  val {len(y_val)} clips, test {len(y_test)} clips")

    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(train_paths), 2000, replace=False)
    norm_X = F.features(np.stack([F.fix_length(F.load_wav(train_paths[i])) for i in sample]))

    train_ds = make_train_dataset(train_paths, train_y, training_noise())
    model = build_model(norm_X)
    model.summary()

    history = model.fit(
        train_ds, validation_data=(X_val, y_val), epochs=MAX_EPOCHS,
        class_weight=class_weight, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                 restore_best_weights=True),
                   keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                     patience=3, verbose=1)])

    # ---- evaluate on the held-out test speakers ----
    pred = model.predict(X_test, verbose=0).argmax(axis=1)
    text = report_text(y_test, pred, "=== Test set (speakers never seen in training) ===")
    print("\n" + text)
    (MODEL_DIR / "test_results.txt").write_text(text + "\n")
    plot_history(history, MODEL_DIR / "training_history.png")
    plot_confusion(y_test, pred, MODEL_DIR / "confusion_matrix.png")

    # ---- save everything the next steps need ----
    model.save(MODEL_DIR / "speech_model.keras")
    float_bytes = tf.lite.TFLiteConverter.from_keras_model(model).convert()
    (MODEL_DIR / "speech_model_float.tflite").write_bytes(float_bytes)
    (MODEL_DIR / "labels.txt").write_text("\n".join(LABELS) + "\n")
    (MODEL_DIR / "features.json").write_text(json.dumps(F.settings(), indent=2) + "\n")

    # 50 clean training clips per class for int8 quantization (representative dataset)
    rep_idx = np.concatenate([rng.choice(np.where(train_y == i)[0], 50, replace=False)
                              for i in range(len(LABELS))])
    rep = F.features(np.stack([F.fix_length(F.load_wav(train_paths[i])) for i in rep_idx]))
    np.save(BASE / "data" / "representative_features.npy", rep.astype(np.float32))

    print(f"\nParameters: {model.count_params():,}")
    print(f"Saved models/speech_model.keras and models/speech_model_float.tflite "
          f"({len(float_bytes) / 1024:.1f} KB)")
    print("Saved plots, test_results.txt, labels.txt, features.json, "
          "data/representative_features.npy")


if __name__ == "__main__":
    main()

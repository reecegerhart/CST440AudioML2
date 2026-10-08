"""
features.py - Turns 1 second of audio into the "picture" (log-mel spectrogram) the model sees.

prepare_data.py, train.py and app.py all import this file, so the features are always
computed the same way. The microcontroller code must use the SAME settings (see
models/features.json, written by train.py).

  16 kHz audio, 1 second = 16000 samples
  30 ms window (480 samples), 20 ms step (320 samples), 512-point FFT
  40 mel bands from 20 Hz to 7600 Hz, then log
  -> a 49 x 40 picture (49 time steps, 40 frequency bands)
These are the same sizes as TensorFlow Lite Micro's "micro_speech" example.
"""
import numpy as np
import tensorflow as tf

SAMPLE_RATE = 16000
CLIP_SAMPLES = SAMPLE_RATE          # 1 second
FRAME_LENGTH = 480                  # 30 ms
FRAME_STEP = 320                    # 20 ms
FFT_LENGTH = 512
NUM_MEL_BINS = 40
LOWER_HZ, UPPER_HZ = 20.0, 7600.0
LOG_OFFSET = 1e-6
NUM_FRAMES = 1 + (CLIP_SAMPLES - FRAME_LENGTH) // FRAME_STEP   # 49
INPUT_SHAPE = (NUM_FRAMES, NUM_MEL_BINS, 1)

MEL_MATRIX = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=NUM_MEL_BINS, num_spectrogram_bins=FFT_LENGTH // 2 + 1,
    sample_rate=SAMPLE_RATE, lower_edge_hertz=LOWER_HZ, upper_edge_hertz=UPPER_HZ)


def fix_length(audio):
    """Pad with silence (or cut) so the clip is exactly 1 second."""
    audio = np.asarray(audio, np.float32)[:CLIP_SAMPLES]
    return np.pad(audio, (0, CLIP_SAMPLES - len(audio)))


def load_wav(path):
    """Read a 16 kHz wav as float32 in [-1, 1]."""
    audio, sr = tf.audio.decode_wav(tf.io.read_file(str(path)), desired_channels=1)
    if int(sr) != SAMPLE_RATE:
        raise ValueError(f"{path} is {int(sr)} Hz, expected {SAMPLE_RATE} Hz")
    return tf.squeeze(audio, -1).numpy()


def tf_features(waves):
    """waves: float32 tensor (batch, 16000) -> (batch, 49, 40, 1) log-mel features."""
    stft = tf.signal.stft(waves, frame_length=FRAME_LENGTH, frame_step=FRAME_STEP,
                          fft_length=FFT_LENGTH)
    mel = tf.matmul(tf.abs(stft), MEL_MATRIX)
    return tf.math.log(mel + LOG_OFFSET)[..., tf.newaxis]


def features(waves, batch=500):
    """numpy version of tf_features, done in batches so big datasets fit in memory."""
    waves = np.asarray(waves, np.float32)
    if waves.ndim == 1:
        waves = waves[np.newaxis]
    return np.concatenate([tf_features(tf.constant(waves[i:i + batch])).numpy()
                           for i in range(0, len(waves), batch)])


def settings():
    """Everything the microcontroller needs to reproduce the features."""
    return {
        "sample_rate": SAMPLE_RATE, "clip_samples": CLIP_SAMPLES,
        "frame_length": FRAME_LENGTH, "frame_step": FRAME_STEP, "fft_length": FFT_LENGTH,
        "window": "hann (periodic)", "magnitude": "abs(stft)",
        "num_mel_bins": NUM_MEL_BINS, "lower_hz": LOWER_HZ, "upper_hz": UPPER_HZ,
        "log": f"natural log(mel + {LOG_OFFSET})", "input_shape": list(INPUT_SHAPE),
        "audio_scale": "int16 samples / 32768",
    }

// audio_features.cpp - see audio_features.h. Matches tf.signal.stft + mel matrix + log
// in features.py.
#include "audio_features.h"

#include <math.h>

#include "mel_weights.h"

static float g_window[kFrameLength];      // periodic Hann, like tf.signal.hann_window
static float g_cos[kFftLength / 2];       // FFT twiddle factors
static float g_sin[kFftLength / 2];
static float g_re[kFftLength];
static float g_im[kFftLength];

void features_init() {
  const float pi = 3.14159265358979f;
  for (int n = 0; n < kFrameLength; n++) {
    g_window[n] = 0.5f - 0.5f * cosf(2.0f * pi * n / kFrameLength);
  }
  for (int k = 0; k < kFftLength / 2; k++) {
    g_cos[k] = cosf(2.0f * pi * k / kFftLength);
    g_sin[k] = -sinf(2.0f * pi * k / kFftLength);
  }
}

// In-place radix-2 FFT of g_re / g_im (kFftLength points).
static void fft() {
  for (int i = 1, j = 0; i < kFftLength; i++) {          // bit-reversal reorder
    int bit = kFftLength >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      float t = g_re[i]; g_re[i] = g_re[j]; g_re[j] = t;
      t = g_im[i]; g_im[i] = g_im[j]; g_im[j] = t;
    }
  }
  for (int len = 2; len <= kFftLength; len <<= 1) {     // butterflies
    int half = len >> 1, step = kFftLength / len;
    for (int i = 0; i < kFftLength; i += len) {
      for (int k = 0; k < half; k++) {
        float wr = g_cos[k * step], wi = g_sin[k * step];
        int a = i + k, b = a + half;
        float tr = g_re[b] * wr - g_im[b] * wi;
        float ti = g_re[b] * wi + g_im[b] * wr;
        g_re[b] = g_re[a] - tr; g_im[b] = g_im[a] - ti;
        g_re[a] += tr;          g_im[a] += ti;
      }
    }
  }
}

void features_frame(const int16_t* samples, float gain, float* out) {
  // 1) scale to [-1, 1] like load_wav(), apply gain, clip like the training augmentation,
  //    window, and zero-pad 480 -> 512
  for (int n = 0; n < kFftLength; n++) {
    float x = 0.0f;
    if (n < kFrameLength) {
      x = samples[n] / 32768.0f * gain;
      if (x > 1.0f) x = 1.0f;
      if (x < -1.0f) x = -1.0f;
      x *= g_window[n];
    }
    g_re[n] = x;
    g_im[n] = 0.0f;
  }
  // 2) FFT -> magnitude of bins 0..256 (stored back in g_re)
  fft();
  for (int k = 0; k <= kFftLength / 2; k++) {
    g_re[k] = sqrtf(g_re[k] * g_re[k] + g_im[k] * g_im[k]);
  }
  // 3) mel bands (sparse weights from mel_weights.h), then log
  for (int m = 0; m < kNumMelBins; m++) {
    const float* w = g_mel_weights + g_mel_offset[m];
    float sum = 0.0f;
    for (int i = 0; i < g_mel_count[m]; i++) {
      sum += g_re[g_mel_start[m] + i] * w[i];
    }
    out[m] = logf(sum + kLogOffset);
  }
}

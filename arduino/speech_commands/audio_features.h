// audio_features.h - the log-mel spectrogram, computed the same way as features.py.
//
// Plain C++ (no Arduino code) so export_arduino.py can compile this exact file on the PC
// and check it against features.py before it goes on the board.
//
//   16 kHz audio, 480-sample Hann window (30 ms), 320-sample step (20 ms),
//   512-point FFT magnitude, 40 mel bands (20-7600 Hz), natural log(mel + 1e-6)
//   -> one frame of 40 values every 20 ms; 49 frames = 1 second = one model input.
#ifndef AUDIO_FEATURES_H
#define AUDIO_FEATURES_H

#include <stdint.h>

const int kSampleRate = 16000;
const int kFrameLength = 480;
const int kFrameStep = 320;
const int kFftLength = 512;
const int kNumMelBins = 40;
const int kNumFrames = 49;
const float kLogOffset = 1e-6f;

// Call once before features_frame().
void features_init();

// samples: kFrameLength int16 audio samples. out: kNumMelBins log-mel values.
// gain multiplies the audio first (like --gain in app.py); 1.0 = unchanged.
void features_frame(const int16_t* samples, float gain, float* out);

#endif

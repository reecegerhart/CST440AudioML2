// feature_test.cpp - runs the Arduino feature code (audio_features.cpp) on the PC.
// Used by export_arduino.py; you don't need to run this yourself.
//
//   feature_test <in.raw> <out.raw>
//   in : 1-second clips of int16 audio, back to back (16000 samples each)
//   out: float32 features, 49 x 40 per clip, same layout as features.py
#include <stdio.h>
#include <vector>

#include "audio_features.h"

int main(int argc, char** argv) {
  if (argc != 3) { fprintf(stderr, "usage: feature_test in.raw out.raw\n"); return 1; }
  FILE* in = fopen(argv[1], "rb");
  FILE* out = fopen(argv[2], "wb");
  if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 1; }
  features_init();
  std::vector<int16_t> clip(kSampleRate);
  float frame[kNumMelBins];
  int clips = 0;
  while (fread(clip.data(), sizeof(int16_t), kSampleRate, in) == (size_t)kSampleRate) {
    for (int f = 0; f < kNumFrames; f++) {
      features_frame(clip.data() + f * kFrameStep, 1.0f, frame);
      fwrite(frame, sizeof(float), kNumMelBins, out);
    }
    clips++;
  }
  fclose(in); fclose(out);
  printf("%d clips\n", clips);
  return 0;
}

// speech_commands.ino - Step 6: the "marvin" speech-command device on the
// Arduino Nano 33 BLE Sense Rev2.
//
// Same behaviour as Device in app.py: sleeps until it hears "marvin", then accepts one
// command (yes / no / stop / go) for 3 seconds, reacts, and goes back to sleep.
//
//   microphone (PDM, 16 kHz) -> log-mel frame every 20 ms (audio_features.cpp)
//   -> last 49 frames = 1 s -> int8 model (~0.6 s per run, back to back) -> average of last 2 guesses
//   -> wake word / command logic -> Serial Monitor + LEDs
//
// Output
//   Serial Monitor (115200 baud): what it heard and what it did
//   orange LED: on while listening for a command (after "marvin")
//   RGB LED for 1.5 s: yes = green, no = red, go = blue, stop = yellow
//
// Serial Monitor keys (type and press Enter)
//   d  debug on/off: prints every guess, mic loudness and inference time
//   +  mic gain x1.5      -  mic gain /1.5
//
// Needs library: Chirale_TensorFLowLite (Arduino Library Manager)
// Files made by export_arduino.py: model_data.h, mel_weights.h
#include <PDM.h>
#include <Chirale_TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "audio_features.h"
#include "model_data.h"

// ---- settings (same as app.py) ----
const float kThreshold = 0.6f;          // average confidence needed to act
const int kSmooth = 2;                  // average the last 2 guesses (~1.2 s; inference takes ~0.6 s)
const unsigned long kListenMs = 3000;   // how long to wait for a command after "marvin"
const unsigned long kCooldownMs = 750;  // don't hear the same word twice
const int kFramesPerInference = 1;      // run the model again as soon as the last run is done
const unsigned long kLedMs = 1500;
const int kWakeIndex = 0;               // "marvin" in g_labels
const int kSilenceIndex = 5;

// The mic amplifier: 0 = -20 dB, 40 = 0 dB, 80 = +20 dB (0.5 dB steps). The library default
// is 20 (-10 dB), which made speech ~4x quieter than the training clips and turned quiet
// "marvin"s into "no". 52 = +6 dB (~6x louder than the default).
const int kPdmGain = 52;
float g_gain = 1.0f;                    // extra software gain on top, change with + / -
bool g_debug = false;

// ---- audio: the PDM interrupt writes into a ring buffer, loop() reads it ----
const int kRingSize = 16384;            // ~1 s of audio, power of 2
int16_t g_ring[kRingSize];
volatile uint32_t g_written = 0;        // total samples ever written
int16_t g_pdm_buf[512];

void onPDMdata() {
  int bytes = PDM.available();
  PDM.read(g_pdm_buf, bytes);
  int n = bytes / 2;
  uint32_t w = g_written;
  for (int i = 0; i < n; i++) g_ring[(w + i) & (kRingSize - 1)] = g_pdm_buf[i];
  g_written = w + n;
}

// ---- model ----
const int kArenaSize = 60 * 1024;
alignas(16) uint8_t g_arena[kArenaSize];
tflite::MicroInterpreter* g_interpreter = nullptr;
TfLiteTensor* g_input = nullptr;
TfLiteTensor* g_output = nullptr;

// ---- features: last 49 frames, already quantized to int8 for the model ----
int8_t g_features[kNumFrames * kNumMelBins];
uint32_t g_next_frame = 0;              // sample index where the next frame starts
int g_frames_ready = 0;
int g_frames_since_run = 0;
int16_t g_frame_samples[kFrameLength];
int16_t g_peak = 0;                     // loudest sample since the last inference

// ---- wake word / command state ----
float g_recent[kSmooth][g_num_labels];
int g_recent_count = 0;
bool g_awake = false;
unsigned long g_awake_until = 0, g_cooldown_until = 0, g_led_off_at = 0;

void setRgb(bool r, bool g, bool b) {   // the RGB LED is on when the pin is LOW
  digitalWrite(LEDR, r ? LOW : HIGH);
  digitalWrite(LEDG, g ? LOW : HIGH);
  digitalWrite(LEDB, b ? LOW : HIGH);
}

void fail(const char* msg) {
  while (true) {
    Serial.println(msg);
    setRgb(true, false, false);
    delay(1000);
  }
}

void setup() {
  Serial.begin(115200);
  unsigned long start = millis();
  while (!Serial && millis() - start < 4000) {}   // wait for the Serial Monitor (max 4 s)

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(LEDR, OUTPUT); pinMode(LEDG, OUTPUT); pinMode(LEDB, OUTPUT);
  setRgb(false, false, false);

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) fail("Model schema version mismatch");

  // exactly the 7 operations the int8 model uses (see README)
  static tflite::MicroMutableOpResolver<7> resolver;
  resolver.AddConv2D();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddSoftmax();
  resolver.AddSub();
  resolver.AddMul();

  static tflite::MicroInterpreter interpreter(model, resolver, g_arena, kArenaSize);
  g_interpreter = &interpreter;
  if (g_interpreter->AllocateTensors() != kTfLiteOk) fail("AllocateTensors failed (arena too small?)");
  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  if (g_input->type != kTfLiteInt8 || g_input->bytes != sizeof(g_features)) fail("Unexpected model input");

  features_init();
  memset(g_features, -128, sizeof(g_features));   // -128 = log(1e-6) = total silence

  PDM.onReceive(onPDMdata);
  PDM.setGain(kPdmGain);
  if (!PDM.begin(1, kSampleRate)) fail("Failed to start the microphone (PDM)");

  Serial.println("Speech commands ready.");
  Serial.print("Model "); Serial.print(g_model_len); Serial.print(" bytes, arena used ");
  Serial.print(g_interpreter->arena_used_bytes()); Serial.println(" bytes");
  Serial.println("Say 'marvin', then yes / no / stop / go.  Keys: d = debug, + / - = mic gain");
}

// Pull complete frames out of the ring buffer and turn them into features.
void processAudio() {
  uint32_t written = g_written;
  if (written - g_next_frame > (uint32_t)(kRingSize - kFrameLength)) {   // fell behind
    Serial.println("(audio overrun, skipping ahead)");
    g_next_frame = written - kFrameLength;
    g_next_frame -= g_next_frame % kFrameStep;
  }
  while (written - g_next_frame >= (uint32_t)kFrameLength) {
    for (int i = 0; i < kFrameLength; i++) {
      int16_t s = g_ring[(g_next_frame + i) & (kRingSize - 1)];
      g_frame_samples[i] = s;
      int16_t a = s < 0 ? -s : s;
      if (a > g_peak) g_peak = a;
    }
    float mel[kNumMelBins];
    features_frame(g_frame_samples, g_gain, mel);

    // drop the oldest frame, add the new one at the end, quantized like app.py
    memmove(g_features, g_features + kNumMelBins, (kNumFrames - 1) * kNumMelBins);
    int8_t* dst = g_features + (kNumFrames - 1) * kNumMelBins;
    for (int m = 0; m < kNumMelBins; m++) {
      long q = lroundf(mel[m] / g_input_scale) + g_input_zero_point;
      dst[m] = (int8_t)(q < -128 ? -128 : (q > 127 ? 127 : q));
    }
    g_next_frame += kFrameStep;
    if (g_frames_ready < kNumFrames) g_frames_ready++;
    g_frames_since_run++;
  }
}

void say(unsigned long now, const char* text) {
  Serial.print("["); Serial.print(now / 1000.0f, 2); Serial.print(" s] "); Serial.println(text);
}

// Same logic as Device.step() in app.py.
void decide(const float* probs, unsigned long now) {
  memmove(g_recent[0], g_recent[1], sizeof(float) * g_num_labels * (kSmooth - 1));
  memcpy(g_recent[kSmooth - 1], probs, sizeof(float) * g_num_labels);
  if (g_recent_count < kSmooth) g_recent_count++;
  int best = 0;
  float avg[g_num_labels];
  for (int c = 0; c < g_num_labels; c++) {
    avg[c] = 0;
    for (int i = kSmooth - g_recent_count; i < kSmooth; i++) avg[c] += g_recent[i][c];
    avg[c] /= g_recent_count;
    if (avg[c] > avg[best]) best = c;
  }
  float conf = avg[best];
  char line[96];

  if (now < g_cooldown_until || conf < kThreshold) {
    if (g_awake && now > g_awake_until) {
      say(now, "(no command heard, going back to sleep)");
      g_awake = false;
    }
    return;
  }
  if (!g_awake && best == kWakeIndex) {
    snprintf(line, sizeof(line), "Heard my name (%d%%). Listening for a command...", (int)(conf * 100));
    say(now, line);
    g_awake = true;
    g_awake_until = now + kListenMs;
    g_cooldown_until = now + kCooldownMs;
  } else if (g_awake && best != kWakeIndex && best != kSilenceIndex) {
    const char* reaction = "";
    if (strcmp(g_labels[best], "yes") == 0)  { reaction = "Confirmed!"; setRgb(false, true, false); }
    if (strcmp(g_labels[best], "no") == 0)   { reaction = "Cancelled."; setRgb(true, false, false); }
    if (strcmp(g_labels[best], "stop") == 0) { reaction = "Stopping.";  setRgb(true, true, false); }
    if (strcmp(g_labels[best], "go") == 0)   { reaction = "Starting!";  setRgb(false, false, true); }
    snprintf(line, sizeof(line), "Command '%s' (%d%%) -> %s", g_labels[best], (int)(conf * 100), reaction);
    say(now, line);
    g_led_off_at = now + kLedMs;
    g_awake = false;
    g_cooldown_until = now + kCooldownMs;
  } else if (g_awake && now > g_awake_until) {
    say(now, "(no command heard, going back to sleep)");
    g_awake = false;
  }
}

void handleKeys() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'd') { g_debug = !g_debug; Serial.println(g_debug ? "debug ON" : "debug OFF"); }
    if (c == '+' || c == '-') {
      g_gain = c == '+' ? g_gain * 1.5f : g_gain / 1.5f;
      Serial.print("mic gain "); Serial.println(g_gain, 2);
    }
  }
}

void loop() {
  handleKeys();
  processAudio();
  unsigned long now = millis();
  digitalWrite(LED_BUILTIN, g_awake ? HIGH : LOW);
  if (g_led_off_at && now > g_led_off_at) { setRgb(false, false, false); g_led_off_at = 0; }
  if (g_frames_ready < kNumFrames || g_frames_since_run < kFramesPerInference) return;
  g_frames_since_run = 0;

  unsigned long t0 = micros();
  memcpy(g_input->data.int8, g_features, sizeof(g_features));
  if (g_interpreter->Invoke() != kTfLiteOk) { Serial.println("Invoke failed"); return; }
  unsigned long infer_us = micros() - t0;

  float probs[g_num_labels];
  for (int c = 0; c < g_num_labels; c++) {
    probs[c] = (g_output->data.int8[c] - g_output_zero_point) * g_output_scale;
  }
  if (g_debug) {
    int best = 0;
    for (int c = 1; c < g_num_labels; c++) if (probs[c] > probs[best]) best = c;
    char line[96];
    snprintf(line, sizeof(line), "heard %-8s %3d%%   loudness %.2f   inference %lu ms",
             g_labels[best], (int)(probs[best] * 100), g_peak * g_gain / 32768.0f, infer_us / 1000);
    Serial.println(line);
  }
  g_peak = 0;
  decide(probs, now);
}

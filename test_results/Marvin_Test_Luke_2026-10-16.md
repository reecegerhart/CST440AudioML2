# Marvin Test Results: Luke (10/16/2026)

Board: Arduino Nano 33 BLE, debug log from the Serial Monitor (115200 baud), about 20 tries per word.
Full log with every guess: `Marvin_Test_Log_Luke_2026-10-16.xlsx`

Focus is on what Marvin gets wrong.

## Accuracy

**135 / 138 correct (97.8%)**, not counting 2 guesses that landed while switching between words.

| Word | Guesses | Correct | Avg confidence | Under 60% |
|---|---|---|---|---|
| marvin | 23 | 23 | 86% | 2 |
| yes | 24 | 23 | 77% | 6 |
| no | 24 | 23 | 87% | 3 |
| go | 26 | 26 | 88% | 2 |
| stop | 41 | 40 | 73% | 14 |

## What it got wrong

All wrong guesses were under 50% confidence.

| Said | Marvin heard | Confidence |
|---|---|---|
| yes | marvin | 30% |
| no | marvin | 30% |
| stop | go | 48% |

Switch guesses (not counted above): marvin 24% going from no to go, no 45% going from go to stop.

## Other issues

- **Silent room:** Marvin guesses "marvin" at 23-30% on most idle frames (25 of 29). Never strong enough to wake him, but it is the main false pattern.
- **"stop" is the weakest word:** lowest average confidence and 14 guesses under 60%. Confidence dropped as loudness fell to 0.03-0.04, so it may be quiet or distant speech. More "stop" training clips would help.
- **Wake word:** 5 / 5 wake-ups (60%, 93%, 80%, 91%, 93%).
- **Late command:** one "yes" at 98% came right as the 3 second listening window closed, and Marvin reported "no command heard."

## Not tested

yes / no / go / stop were said without "marvin" first, so this measures word accuracy, not the full wake-then-command flow with LEDs.

# GestureEdge — Real-Time Hand Gesture Control via Edge Inference

Control the Chrome Dino game with hand gestures — 3 gesture classes, 0.038ms inference latency, threaded pipeline decoupled from game rendering.

Built on: MediaPipe Tasks → TFLite MLP → GestureBuffer → Chrome Dino game (monkey-patched).

---

## Demo

<!-- TODO: add demo GIF or video after recording -->

| Gesture | Action |
|---------|--------|
| Open hand | Jump |
| Fist | Duck |
| Pointer finger | Run (cancel jump/duck) |

---

## System Pipeline

```
[Webcam]
   ↓  OpenCV capture + BGR→RGB + flip
[MediaPipe Tasks — hand_landmarker.task]
   ↓  21 landmarks (x, y per point)
[Landmark normalization]
   ↓  translate to wrist origin → scale by max absolute value → 42-dim vector
[TFLite MLP — keypoint_classifier.tflite]
   ↓  3 classes: Open / Fist / Pointer
[GestureBuffer — sliding window majority vote, N=10]
   ↓  stable gesture → thread-safe shared state (threading.Lock)
[Chrome Dino game loop — monkey-patched]
   ↓  inject Up / Down key per frame
[Jump / Duck / Run]
```

The gesture recognition pipeline runs in a **daemon thread**. The game loop runs in the **main thread**. They share state via `threading.Lock()`.

---

## Engineering Decisions

### 1. Landmark normalization — why not raw pixel coordinates?

Raw pixel landmarks shift with hand position and camera distance. Each frame is normalized by:
1. Translating all 21 points relative to landmark 0 (wrist)
2. Scaling by the maximum absolute value in the flattened 42-dim vector

Result: a position- and scale-invariant feature vector. The MLP generalizes across different hand sizes and camera distances without retraining.

### 2. Threading — decouple inference from game rendering

Gesture inference (capture → MediaPipe → MLP → buffer) blocks on I/O and model execution. Running it synchronously in the game loop would stall rendering on slow frames.

Solution: daemon thread for inference, main thread for the game. Shared state is one `_gesture_action` variable guarded by a `Lock`. The game loop reads it every frame at zero blocking cost.

### 3. Temporal smoothing — GestureBuffer

Raw per-frame classification is noisy. A single frame with an occluded finger can flip the gesture class and trigger an unintended jump.

`GestureBuffer` (`gesture_dino.py`) applies a **sliding window majority vote** over the last `window` frames. It also holds the last stable gesture for `hold_frames` consecutive misses — so brief detection gaps don't reset the action mid-game.

```
raw gesture (per frame, or -1 if no hand)
      ↓
GestureBuffer (window=10, hold_frames=5)
  - majority vote over last 10 frames
  - hold last stable gesture for 5 miss frames
      ↓
stable gesture  →  inject key into game loop
```

Three tunable parameters:
- `window=10` — larger = smoother output, ~+33ms lag per extra frame at 30fps
- majority vote threshold — implicit at ≥50% of window
- `hold_frames=5` — grace period for brief occlusion

### 4. TensorRT vs ONNX — profiling-driven decision

I benchmarked both runtimes on the keypoint classifier before choosing one:

| Runtime | Per-sample latency |
|---------|--------------------|
| ONNX CPU | **0.038 ms** |
| TensorRT GPU | 0.062 ms |

TensorRT was **63% slower** per sample. Root cause: the MLP (42 inputs → 3 outputs) is too small for GPU parallelism to overcome CPU↔GPU memory transfer overhead. TensorRT's latency advantage only materializes with heavier architectures — CNNs on raw frames, for example. For this pipeline, ONNX Runtime on CPU is the correct tool.

### 5. Glove robustness — observed problem, hypothesis, scope decision

Testing with thick gloves produced intermittent detection: landmarks present one frame, gone the next, causing uncontrolled jumps in-game.

**Hypothesis (not confirmed via experiment):** distribution shift. MediaPipe's palm detector was trained on bare hands. Thick gloves alter the hand silhouette and occlude the joint features the model relies on for localization.

Two partial mitigations applied:
- `min_hand_detection_confidence=0.4` (lowered from default 0.5)
- `hold_frames=5` in `GestureBuffer` to survive brief detection gaps

This improved stability but did not fully solve thick-glove intermittency. A complete fix would require retraining the palm detector on gloved-hand data or switching to a domain-adapted model — outside the project scope.

---

## Results

| Metric | Value |
|--------|-------|
| Inference latency (ONNX CPU, median) | 0.038 ms |
| Inference latency (TensorRT GPU, median) | 0.062 ms |
| Gesture classes | 3 active (4 trained, 97.2% accuracy) |
| Smoothing window | 10 frames |
| Hold frames on miss | 5 frames |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Hand landmark detection | MediaPipe Tasks |
| Gesture classifier | TFLite MLP (42 → 20 → 10 → 4, 1,114 params) |
| Camera capture + overlay | OpenCV |
| Concurrency | Python `threading` |
| Runtime (final) | ONNX Runtime (CPU) |
| Runtime (benchmarked) | TensorRT (GPU) |
| Game integration | Chrome Dino Runner (monkey-patched) |

---

## Setup

```bash
pip install mediapipe opencv-python numpy
```

Clone the Chrome Dino Runner as an external dependency into the project root:

```bash
# The folder must be named Chrome-Dino-Runner-master
git clone https://github.com/dhhruv/Chrome-Dino-Runner.git Chrome-Dino-Runner-master
```

Run the gesture-controlled game:

```bash
python gesture_dino.py
```

Keyboard controls still work: `Up` / `Space` = jump, `Down` = duck, `p` = pause, `ESC` = close camera.

---

## Credits

- Hand gesture recognition base: [Kazuhito00/hand-gesture-recognition-using-mediapipe](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) (translated by [kinivi](https://github.com/kinivi))
- Chrome Dino game: Chrome-Dino-Runner (external dependency, not included in this repo)

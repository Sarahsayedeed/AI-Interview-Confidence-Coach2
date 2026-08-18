# AI Interview Confidence Coach - Detection Improvement Plan

## Goal
Improve the accuracy of detecting tension signals (hand-to-face and gaze deviation) and add live on-screen text alerts directly inside the recorded video when these events occur.

## User Review Required
The original system used a "10-second baseline" and compared the rest of the video to it. This means if you weren't perfectly centered, or if the baseline was skewed, it wouldn't detect tension well.

**Proposed Change:** We will switch to **absolute threshold detection** for better reliability and real-time alerts.
1. **Hand-to-Face:** Instead of comparing to a baseline, we will directly measure the distance between your wrists/index fingers and your face. If they get too close, it immediately registers as a touch.
2. **Gaze/Head Turn:** We will measure the distance from your nose to your ears. If your head turns left or right, one ear gets much closer to your nose in the camera view. This is a very reliable way to detect looking away.
3. **Live Video Overlays:** Because we are detecting this live frame-by-frame, we will use OpenCV to draw warning text (e.g., "⚠️ Touching Face" or "⚠️ Looking Away") directly on the video right above your head/hands exactly when it happens.

Do you agree with switching to this more direct detection method so we can show the alerts live in the video?

## Proposed Changes

### 1. Update CV Logic
#### [MODIFY] `cv_tracker.py`
- Add absolute distance calculations for hand-to-face using `INDEX_FINGER` and `WRIST` landmarks relative to `NOSE` and `EYE` landmarks.
- Add head-turn calculations using `NOSE` and `EAR` horizontal distances.
- If thresholds are breached during a frame, use `cv2.putText` to draw a prominent alert (e.g., red text) on the video frame at the coordinate of the hand or face.
- Return the boolean flags (`is_touching_face`, `is_looking_away`) in the features dictionary for each frame.

### 2. Update Event Detector
#### [MODIFY] `event_detector.py`
- Remove the complex rolling z-score logic.
- Replace it with a simpler, highly robust grouping logic: if the `is_touching_face` boolean is `True` for more than 0.5 seconds consecutively, record it as a `"hand_touched_face"` event.
- Do the same for `is_looking_away`.

## Verification Plan
1. Run the app locally.
2. Perform intentional movements: touch the face with hands and turn the head away from the screen.
3. Verify that the recorded video displays the text alerts directly over the video when the actions happen.
4. Verify that the AI correctly receives these events and gives coaching notes.

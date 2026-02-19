# Cyberpunk Focus & Fatigue Tracker Pro

This project is an intelligent computer vision system that tracks user attention and fatigue in real-time using a webcam. Written in Python using the **MediaPipe Face Mesh** library.

The program automatically calibrates to the user's face, tracks gaze direction, head turns, blinks, and instantly reacts to microsleep or yawning.

## Features

* **Automatic Calibration:** The program automatically adjusts to your face, lighting, and posture during the first 50 frames.
* **3D Focus Tracking:** Tracks pupil movement (X/Y) and head turns. If you look away from the screen, a warning signal sounds.
* **Microsleep Alarm:** If eyes remain closed for more than 1.5 seconds, a sharp alarm siren goes off.
* **Yawn Detector:** Analyzes mouth shape (MAR) to detect early signs of fatigue.
* **Smart Blink Counter:** Ignores normal squinting and camera micro-noise, recording only full blinks.
* **Focus Score:** At the end of the session, it provides detailed statistics with the percentage of focus on the screen.

## Installation

The project is optimized for **Python 3.12** on Windows. To avoid version conflicts (especially with MediaPipe and NumPy), copy and run this line in your terminal:

```bash
pip install opencv-python mediapipe==0.10.14 numpy==1.26.4 protobuf==3.20.3
```

## Usage Instructions

1. Connect your webcam and run the script:
   ```bash
   python blink_checker.py
   ```
2. **Calibration Phase:** Immediately after launch, a yellow `CALIBRATING` text will appear on the screen. Sit straight, look directly into the camera with a neutral facial expression, and remain quiet for a few seconds.
3. **Active Mode:** After the successful calibration message, work at your computer as usual.
   * Turn your head or look away — you will hear a double beep.
   * Close your eyes for 2 seconds — you will hear a sharp squeak.
4. **How to Exit:** To properly stop the program and get the final statistical report in the terminal, **blink 5 times** (or simply press the `Esc` key on your keyboard).

## Example Final Report

```text
==================================================
CYBERPUNK REPORT:
==================================================
Focus Score: 92.4%
Signs of fatigue detected (yawning).
Blinks: 5/5
Avg. duration: 0.125s | Avg. interval: 8.430s
==================================================
```

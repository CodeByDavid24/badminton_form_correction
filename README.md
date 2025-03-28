# 🏸 AI Overhead Smash Form Coach

An AI-powered **badminton smash analysis tool** that provides real-time feedback on player form using **MediaPipe and OpenCV**. This tool helps players **improve their smash technique** by analyzing body angles, tracking racket motion, and saving training videos with overlaid feedback.

## 🚀 Features

✅ **Real-time form analysis** (shoulder, elbow, knee angles)
✅ **Racket motion tracking** using color detection (red/blue grip)
✅ **Personalized feedback** displayed on-screen
✅ **Video recording** with feedback overlay for review
✅ **Side-view camera analysis** (single camera setup)

## 📂 Installation

Make sure you have Python installed, then install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

## 🎮 Usage

1. **Run the script:**

   ```bash
   python badminton_smash_ai.py
   ```

2. **Position the camera** to capture a **side view** of your smash.
3. **Use a racket with a red or blue grip** for tracking.
4. **Perform an overhead smash.**
5. **Receive real-time feedback** on your form.
6. **Press 'Q' to exit** and review the saved video (`smash_analysis.avi`).

## 📊 How It Works

- **Pose Detection:** Uses **MediaPipe** to analyze joint angles (elbow, shoulder, knee).
- **Racket Tracking:** Detects racket motion based on **color filtering**.
- **Form Feedback:** Compares joint angles to ideal values and provides **corrections**.
- **Video Saving:** Records the session with **feedback overlay**.

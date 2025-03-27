# Badminton Smash Form Correction Application

## Overview

This AI-powered application uses computer vision and pose estimation to help badminton players improve their smash technique through real-time form analysis. By leveraging MediaPipe's pose estimation technology, the application provides immediate feedback on shoulder and elbow positioning during a smash motion.

## Features

- Real-time pose estimation
- Angle calculation for shoulder and elbow joints
- Smash form quality scoring (0-100)
- Personalized technique feedback
- Webcam-based analysis without requiring jumping

## Prerequisites

### System Requirements

- Python 3.10.x
- Windows 10/11 or macOS
- Webcam

### Required Dependencies

- MediaPipe
- OpenCV
- NumPy

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/badminton-form-correction.git
cd badminton-form-correction
```

2. Create a virtual environment:

```bash
python -m venv badminton_pose
source badminton_pose/bin/activate  # Unix/macOS
badminton_pose\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the application:

```bash
python main.py
```

### Interaction

- Position yourself in front of the webcam
- Perform a stationary smash motion
- Receive real-time form feedback
- Press 'q' to exit the application

## Performance Considerations

- Ensure good lighting conditions
- Wear contrasting clothing for better landmark detection
- Perform motion in a clear, unobstructed space

## Future Improvements

- Multi-angle form analysis
- Saved performance logs
- Personalized player profiles
- Advanced machine learning insights

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit changes
4. Push to the branch
5. Create a pull request

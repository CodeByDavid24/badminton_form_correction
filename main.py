import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Video Writer setup (for saving)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('smash_analysis.avi', fourcc, 20.0, (640, 480))

# Function to calculate angle between three points


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

# Function to analyze form


def analyze_form(landmarks):
    feedback = []

    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y]
    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y]
    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y]
    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
           landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y]
    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y]
    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]

    elbow_angle = calculate_angle(shoulder, elbow, wrist)  # Ideal: 100°–130°
    shoulder_angle = calculate_angle(hip, shoulder, elbow)  # Ideal: 160°–180°
    # Ideal: Bent (~110° if jumping)
    knee_angle = calculate_angle(hip, knee, ankle)

    if elbow_angle < 100:
        feedback.append("Bend your elbow more before swinging.")
    elif elbow_angle > 130:
        feedback.append("Avoid overextending your elbow.")

    if shoulder_angle < 160:
        feedback.append("Rotate your shoulder fully for more power.")

    if knee_angle > 120:
        feedback.append("Bend your knees more for better jump power.")

    if wrist[1] > shoulder[1]:  # If wrist is lower than shoulder
        feedback.append("Make contact higher for a steeper smash.")

    return feedback

# Function to track racket motion using color detection (assumes racket grip is red)


def track_racket(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color detection
    lower_red = np.array([0, 120, 70])
    upper_red = np.array([10, 255, 255])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return (x + w // 2, y + h // 2)  # Return center of detected racket
    return None


# OpenCV Video Capture
cap = cv2.VideoCapture(0)  # 0 = Default Webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB for Mediapipe processing
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Convert back to BGR for OpenCV display
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get feedback
        feedback = analyze_form(results.pose_landmarks.landmark)
        for i, msg in enumerate(feedback):
            cv2.putText(image, msg, (10, 30 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Track racket motion
    racket_position = track_racket(frame)
    if racket_position:
        # Draw red circle on detected racket
        cv2.circle(image, racket_position, 10, (0, 0, 255), -1)

    # Save video frame
    out.write(image)

    # Display video feed
    cv2.imshow('Badminton Smash Analysis', image)

    # Exit on 'q' key
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

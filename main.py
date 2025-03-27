import cv2
import mediapipe as mp
import numpy as np


class BadmintonSmashAnalyzer:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def calculate_angle(self, a, b, c):
        """
        Calculate angle between three points
        Args:
            a, b, c: Landmark points (x, y coordinates)
        Returns:
            Angle in degrees
        """
        a = np.array([a.x, a.y])
        b = np.array([b.x, b.y])
        c = np.array([c.x, c.y])

        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
            np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians * 180.0 / np.pi)

        return angle if angle <= 180 else 360 - angle

    def analyze_smash_form(self, frame):
        """
        Analyze badminton smash form from a single frame
        Args:
            frame: Input image frame
        Returns:
            Annotated frame and analysis results
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image
        results = self.pose.process(image_rgb)

        if not results.pose_landmarks:
            return frame, None

        landmarks = results.pose_landmarks.landmark

        # Key points for smash form analysis
        shoulder_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER]
        elbow_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_ELBOW]
        wrist_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_WRIST]
        hip_right = landmarks[self.mp_pose.PoseLandmark.RIGHT_HIP]

        # Calculate key angles
        shoulder_angle = self.calculate_angle(
            hip_right, shoulder_right, elbow_right
        )
        elbow_angle = self.calculate_angle(
            shoulder_right, elbow_right, wrist_right
        )

        # Annotate the frame
        annotated_frame = frame.copy()
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                h, w, _ = annotated_frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 0), -1)

        # Smash form analysis results
        analysis = {
            'shoulder_angle': round(shoulder_angle, 2),
            'elbow_angle': round(elbow_angle, 2),
            'smash_form_quality': self._evaluate_smash_form(
                shoulder_angle, elbow_angle
            )
        }

        return annotated_frame, analysis

    def _evaluate_smash_form(self, shoulder_angle, elbow_angle):
        """
        Evaluate smash form based on key angles
        Args:
            shoulder_angle: Angle between hip, shoulder, and elbow
            elbow_angle: Angle between shoulder, elbow, and wrist
        Returns:
            Form quality assessment
        """
        form_score = 100
        feedback = []

        # Shoulder angle check (ideal: 130-150 degrees)
        if shoulder_angle < 120:
            form_score -= 20
            feedback.append("Rotate shoulders more for better power transfer")
        elif shoulder_angle > 160:
            form_score -= 20
            feedback.append("Reduce excessive shoulder rotation")

        # Elbow angle check (ideal: 100-130 degrees)
        if elbow_angle < 90:
            form_score -= 15
            feedback.append("Extend elbow more for increased racket speed")
        elif elbow_angle > 150:
            form_score -= 15
            feedback.append("Bend elbow slightly for better smash technique")

        return {
            'score': max(0, form_score),
            'feedback': feedback
        }


def main():
    # Initialize video capture (webcam)
    cap = cv2.VideoCapture(0)
    analyzer = BadmintonSmashAnalyzer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame for more natural view
        frame = cv2.flip(frame, 1)

        # Analyze pose
        annotated_frame, analysis = analyzer.analyze_smash_form(frame)

        # Display results
        if analysis:
            # Display form score and feedback
            cv2.putText(
                annotated_frame,
                f"Form Score: {analysis['smash_form_quality']['score']}/100",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            # Display feedback
            y_offset = 60
            for feedback in analysis['smash_form_quality']['feedback']:
                cv2.putText(
                    annotated_frame,
                    feedback,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    1
                )
                y_offset += 30

        cv2.imshow('Badminton Smash Form Analyzer', annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

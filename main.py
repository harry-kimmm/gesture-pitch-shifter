import cv2
import mediapipe as mp
import numpy as np

def main():
    hands = mp.solutions.hands.Hands(max_num_hands=1,
                                     min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0)       

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No webcam frame"); break

        frame = cv2.flip(frame, 1)  
        res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = res.multi_hand_landmarks[0].landmark
            p1 = np.array([lm[4].x * w, lm[4].y * h])   # thumb tip
            p2 = np.array([lm[8].x * w, lm[8].y * h])   # index tip
            dist = np.linalg.norm(p1 - p2)
            semitone = np.interp(dist, [0, 150], [12, -12])

            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     (0, 255, 0), 2)
            cv2.putText(frame, f"{semitone:+.1f} st", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow("Hand-tracking demo  –  ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

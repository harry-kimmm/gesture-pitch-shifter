import cv2, mediapipe as mp

mp_hands = mp.solutions.hands.Hands(max_num_hands=1)
cap = cv2.VideoCapture(0)

while True:
    ok, frame = cap.read()
    if not ok:
        print("No webcam frame"); break
    frame = cv2.flip(frame, 1)                 # mirror for natural feel
    res = mp_hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if res.multi_hand_landmarks:
        for h in res.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, h, mp.solutions.hands.HAND_CONNECTIONS)

    cv2.imshow("ESC to quit", frame)
    if cv2.waitKey(1) & 0xFF == 27:            # Esc key
        break

cap.release(); cv2.destroyAllWindows()

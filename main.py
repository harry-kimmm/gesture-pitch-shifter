import queue, threading, time, numpy as np
import cv2, mediapipe as mp
import soundfile as sf, sounddevice as sd
from   scipy.signal import resample_poly

CALIB_TIME   = 2.0 
BLOCK        = 4096    
SMOOTH_ALPHA = 0.25 
MAX_SHIFT_ST = 6   

pitch_q = queue.Queue(maxsize=1)

def video_loop():
    hands = mp.solutions.hands.Hands(max_num_hands=1,
                                     min_detection_confidence=0.7)
    cap   = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    t0 = time.time()
    d_min, d_max = np.inf, 0
    smoothed     = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No webcam frame"); break
        frame = cv2.flip(frame, 1)
        res   = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm  = res.multi_hand_landmarks[0].landmark
            p1  = np.array([lm[4].x*w, lm[4].y*h])   # thumb tip
            p2  = np.array([lm[8].x*w, lm[8].y*h])   # index tip
            dist = np.linalg.norm(p1 - p2)

            if time.time() - t0 < CALIB_TIME:
                d_min = min(d_min, dist)
                d_max = max(d_max, dist)
                cv2.putText(frame, "Calibrating… pinch & spread", (10,35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                semitone = 0
            else:
                rng = max(10.0, d_max - d_min)          
                norm = np.clip((dist - d_min)/rng, 0, 1)
                raw  = (0.5 - norm) * 2 * MAX_SHIFT_ST
                smoothed = (1-SMOOTH_ALPHA)*smoothed + SMOOTH_ALPHA*raw
                semitone = smoothed

            if pitch_q.full(): pitch_q.get_nowait()
            pitch_q.put_nowait(semitone)

            
            cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)),
                     (0,255,0), 2)
            cv2.putText(frame, f"{semitone:+.1f} st", (10,65),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            mp.solutions.drawing_utils.draw_landmarks(
                frame, res.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS)

        cv2.imshow("ESC to quit", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release(); cv2.destroyAllWindows()

WAV, SR = sf.read("snippet.wav", dtype="float32")
CURSOR  = 0
stop_evt = threading.Event()

def fast_shift(block, st):
    if st == 0:
        return block
    ratio = 2 ** (st / 12)
    if ratio >= 1:
        out = resample_poly(block, int(round(ratio*100)), 100)
    else:
        out = resample_poly(block, 100, int(round(100/ratio)))
    if len(out) > len(block):
        out = out[:len(block)]
    elif len(out) < len(block):
        out = np.pad(out, (0, len(block)-len(out)))
    return out

def audio_worker():
    global CURSOR

    def cb(outdata, frames, tinfo, status):
        global CURSOR
        if status: print(status)

        chunk = WAV[CURSOR:CURSOR+frames]
        if len(chunk) < frames:
            chunk = np.concatenate([chunk, WAV[:frames-len(chunk)]])
            CURSOR = (CURSOR + frames) % len(WAV)
        else:
            CURSOR += frames

        try: shift = pitch_q.get_nowait()
        except queue.Empty: shift = 0

        if chunk.ndim == 2:            
            chunk = chunk.mean(axis=1)
        shifted = fast_shift(chunk, shift)
        outdata[:,0] = np.clip(shifted, -0.95, 0.95)

    with sd.OutputStream(channels=1, samplerate=SR,
                         blocksize=BLOCK, callback=cb):
        while not stop_evt.is_set():
            time.sleep(0.05)

if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    try:
        video_loop()         
    finally:
        stop_evt.set()
        time.sleep(0.2)

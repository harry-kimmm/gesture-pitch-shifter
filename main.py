import queue, threading, time, numpy as np
import cv2, mediapipe as mp
import soundfile as sf, sounddevice as sd
from scipy.signal import resample_poly

CALIB_SECS = 2.0
BLOCK = 2048
SMOOTH_ALPHA = 0.25
MAX_SHIFT_ST = 6
VOL_FLOOR = 0.05
COOLDOWN_S = 0.5
OPEN_SPREAD = 40

pitch_q = queue.Queue(maxsize=1)
volume_q = queue.Queue(maxsize=1)

def thumb_index_dist(lm, w, h):
    p1 = np.array([lm[4].x * w, lm[4].y * h])
    p2 = np.array([lm[8].x * w, lm[8].y * h])
    return np.linalg.norm(p1 - p2)

def fingers_up_count(lm):
    tips = [4, 8, 12, 16, 20]
    mcp = [2, 5, 9, 13, 17]
    return sum(lm[t].y < lm[b].y for t, b in zip(tips, mcp))

def fast_pitch_shift(block, st):
    if st == 0:
        return block
    ratio = 2 ** (st / 12)
    if ratio >= 1:
        out = resample_poly(block, int(round(ratio * 100)), 100)
    else:
        out = resample_poly(block, 100, int(round(100 / ratio)))
    if len(out) > len(block):
        out = out[: len(block)]
    elif len(out) < len(block):
        out = np.pad(out, (0, len(block) - len(out)))
    return out

def video_loop():
    hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    t0 = time.time()
    d_min, d_max = np.inf, 0
    pitch_s = 0.0
    vol_s = 1.0
    mode = "pitch"
    armed = False
    last_toggle = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if res.multi_hand_landmarks:
            h, w, _ = frame.shape
            lm = res.multi_hand_landmarks[0].landmark
            spread = thumb_index_dist(lm, w, h)
            f_up = fingers_up_count(lm)

            now = time.time()
            if f_up <= 1:
                armed = True
            elif (
                f_up == 5
                and spread > OPEN_SPREAD
                and armed
                and now - last_toggle > COOLDOWN_S
            ):
                mode = "volume" if mode == "pitch" else "pitch"
                last_toggle = now
                armed = False
                pitch_s, vol_s = 0, 1

            if now - t0 < CALIB_SECS:
                d_min, d_max = min(d_min, spread), max(d_max, spread)
                cv2.putText(
                    frame,
                    "Calibrating… pinch & spread",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )

            rng = max(10.0, d_max - d_min)
            norm = np.clip((spread - d_min) / rng, 0, 1)

            if mode == "pitch":
                raw = (0.5 - norm) * 2 * MAX_SHIFT_ST
                pitch_s = (1 - SMOOTH_ALPHA) * pitch_s + SMOOTH_ALPHA * raw
                pitch_val, vol_val = pitch_s, 1.0
            else:
                raw = norm
                vol_s = (1 - SMOOTH_ALPHA) * vol_s + SMOOTH_ALPHA * raw
                pitch_val, vol_val = 0.0, max(VOL_FLOOR, vol_s)

            for q, v in ((pitch_q, pitch_val), (volume_q, vol_val)):
                if q.full():
                    q.get_nowait()
                q.put_nowait(v)

            p1 = np.array([lm[4].x * w, lm[4].y * h]).astype(int)
            p2 = np.array([lm[8].x * w, lm[8].y * h]).astype(int)
            cv2.line(frame, tuple(p1), tuple(p2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{pitch_val:+.1f} st",
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Vol {vol_val*100:3.0f}%",
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"[{mode.upper()} MODE]  (high-five toggles)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                res.multi_hand_landmarks[0],
                mp.solutions.hands.HAND_CONNECTIONS,
            )

        cv2.imshow("Gesture Pitch / Volume", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

WAV, SR = sf.read("snippet.wav", dtype="float32")
cursor = 0
stop_evt = threading.Event()

def audio_worker():
    global cursor
    pitch_v, vol_v = 0.0, 1.0

    def cb(out, frames, *_):
        nonlocal pitch_v, vol_v
        global cursor

        chunk = WAV[cursor : cursor + frames]
        if len(chunk) < frames:
            chunk = np.concatenate([chunk, WAV[: frames - len(chunk)]])
            cursor = (cursor + frames) % len(WAV)
        else:
            cursor += frames

        try:
            pitch_v = pitch_q.get_nowait()
        except queue.Empty:
            pass
        try:
            vol_v = volume_q.get_nowait()
        except queue.Empty:
            pass

        mono = chunk.mean(axis=1) if chunk.ndim == 2 else chunk
        shifted = fast_pitch_shift(mono, pitch_v)
        out[:, 0] = np.clip(shifted * vol_v, -0.95, 0.95)

    sd.OutputStream(
        channels=1, samplerate=SR, blocksize=BLOCK, callback=cb
    ).__enter__()
    while not stop_evt.is_set():
        time.sleep(0.05)

if __name__ == "__main__":
    threading.Thread(target=audio_worker, daemon=True).start()
    try:
        video_loop()
    finally:
        stop_evt.set()
        time.sleep(0.2)

import queue, threading, time, numpy as np, cv2, mediapipe as mp
import soundfile as sf, sounddevice as sd
from scipy.signal import resample_poly

CALIB_SECS, BLOCK = 2.0, 2048
ALPHA, MAX_ST, VOL_MIN = 0.25, 6, 0.05
COOLDOWN, OPEN_PX = 0.5, 40

pitch_q = queue.Queue(1)
vol_q   = queue.Queue(1)

def tdist(lm,w,h):
    return np.linalg.norm([lm[4].x*w-lm[8].x*w, lm[4].y*h-lm[8].y*h])

def fcount(lm):
    return sum(lm[t].y < lm[b].y for t,b in zip([4,8,12,16,20],[2,5,9,13,17]))

def rshift(block, st):
    if st == 0: return block
    r = 2**(st/12)
    up,down = (int(round(r*100)),100) if r>=1 else (100,int(round(100/r)))
    out = resample_poly(block, up, down)
    return out[:len(block)] if len(out)>=len(block) else np.pad(out,(0,len(block)-len(out)))

def video():
    hsys = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
    cam  = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    t0 = time.time(); dmin,dmax = np.inf,0
    ps,vs = 0.,1.; mode="pitch"; armed=False; last=0.
    while True:
        ok,f = cam.read(); f=cv2.flip(f,1)
        if not ok: break
        res = hsys.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            H,W,_ = f.shape; lm=res.multi_hand_landmarks[0].landmark
            d, up = tdist(lm,W,H), fcount(lm); now=time.time()
            if up<=1: armed=True
            elif up==5 and d>OPEN_PX and armed and now-last>COOLDOWN:
                mode = "volume" if mode=="pitch" else "pitch"; last=now; armed=False; ps,vs=0,1
            if now-t0<CALIB_SECS: dmin,dmax=min(dmin,d),max(dmax,d)
            rng=max(10.,dmax-dmin); n=np.clip((d-dmin)/rng,0,1)
            if mode=="pitch":
                raw=(0.5-n)*2*MAX_ST
                ps=(1-ALPHA)*ps+ALPHA*raw
                pv,vv=ps,1.
            else:
                raw=n
                vs=(1-ALPHA)*vs+ALPHA*raw
                pv,vv=0.,max(VOL_MIN,vs)
            if pitch_q.full(): pitch_q.get_nowait()
            pitch_q.put_nowait(pv)
            if vol_q.full():   vol_q.get_nowait()
            vol_q.put_nowait(vv)
            a,b=(int(lm[4].x*W),int(lm[4].y*H)),(int(lm[8].x*W),int(lm[8].y*H))
            cv2.line(f,a,b,(0,255,0),2)
            cv2.putText(f,f"{pv:+.1f} st",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
            cv2.putText(f,f"Vol {vv*100:3.0f}%",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.putText(f,f"[{mode.upper()}] high-five toggles",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            mp.solutions.drawing_utils.draw_landmarks(f,res.multi_hand_landmarks[0],mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow("Pitch / Volume",f)
        if cv2.waitKey(1)&0xFF==27: break
    cam.release(); cv2.destroyAllWindows()

W,S = sf.read("snippet.wav",dtype="float32")
cursor=0; stop_evt = threading.Event()

def audio():
    global cursor; pv,vv = 0.,1.
    def cb(out,frames,*_):
        nonlocal pv,vv; global cursor
        seg = W[cursor:cursor+frames]
        if len(seg)<frames:
            seg = np.concatenate([seg, W[:frames-len(seg)]])
            cursor = (cursor+frames)%len(W)
        else:
            cursor += frames
        try: pv = pitch_q.get_nowait()
        except queue.Empty: pass
        try: vv = vol_q.get_nowait()
        except queue.Empty: pass
        mono = seg.mean(axis=1) if seg.ndim==2 else seg
        out[:,0] = np.clip(rshift(mono,pv)*vv, -0.95, 0.95)
    stream = sd.OutputStream(channels=1,samplerate=S,blocksize=BLOCK,callback=cb)
    stream.start()
    while not stop_evt.is_set(): time.sleep(0.05)
    stream.stop(); stream.close()

if __name__=="__main__":
    threading.Thread(target=audio,daemon=True).start()
    try: video()
    finally: stop_evt.set(); time.sleep(0.2)

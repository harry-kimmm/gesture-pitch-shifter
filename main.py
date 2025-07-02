import os, sys, queue, threading, time, numpy as np, cv2, mediapipe as mp
import soundfile as sf, sounddevice as sd
from scipy.signal import resample_poly

def pick_audio():
    folder = os.path.join(os.path.dirname(__file__), "samples")
    files  = [f for f in os.listdir(folder) if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
    if not files:
        print("No audio files in /samples"); sys.exit(1)
    for i, name in enumerate(files, 1):
        print(f"{i}. {name}")
    while True:
        c = input("Select number: ")
        if c.isdigit() and 1 <= int(c) <= len(files):
            return os.path.join(folder, files[int(c)-1])
        print("Try again.")

AUDIO = pick_audio()
CALIB_SECS, BLOCK = 2.0, 2048
ALPHA, MAX_ST, VOL_MIN = 0.25, 6, 0.05
COOLDOWN, OPEN_PX = 0.5, 40

pitch_q, vol_q = queue.Queue(1), queue.Queue(1)

def dist(lm,w,h):
    return np.linalg.norm([lm[4].x*w-lm[8].x*w, lm[4].y*h-lm[8].y*h])

def upcount(lm):
    return sum(lm[t].y < lm[b].y for t,b in zip([4,8,12,16,20],[2,5,9,13,17]))

def rshift(block, st):
    if st == 0: return block
    r = 2**(st/12)
    up, down = (int(r*1000),1000) if r>=1 else (1000,int(1000/r))
    out = resample_poly(block, up, down)
    return out[:len(block)] if len(out)>=len(block) else np.pad(out,(0,len(block)-len(out)))

def cam_loop():
    h = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
    t0=time.time(); dmin,dmax=np.inf,0
    ps,vs,mode,armed,last = 0.,1.,"pitch",False,0.
    while True:
        ok,f=cam.read(); f=cv2.flip(f,1)
        if not ok: break
        res=h.process(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            H,W,_=f.shape; lm=res.multi_hand_landmarks[0].landmark
            d, up = dist(lm,W,H), upcount(lm); now=time.time()
            if up<=1: armed=True
            elif up==5 and d>OPEN_PX and armed and now-last>COOLDOWN:
                mode="volume" if mode=="pitch" else "pitch"; last=now; armed=False; ps,vs=0,1
            if now-t0<CALIB_SECS: dmin,dmax=min(dmin,d),max(dmax,d)
            rng=max(10.,dmax-dmin); n=np.clip((d-dmin)/rng,0,1)
            if mode=="pitch":
                ps=(1-ALPHA)*ps+ALPHA*((0.5-n)*2*MAX_ST); pv,vv=ps,1.
            else:
                vs=(1-ALPHA)*vs+ALPHA*n; pv,vv=0.,max(VOL_MIN,vs)
            if pitch_q.full(): pitch_q.get_nowait()
            pitch_q.put_nowait(pv)
            if vol_q.full(): vol_q.get_nowait()
            vol_q.put_nowait(vv)
            a,b=(int(lm[4].x*W),int(lm[4].y*H)),(int(lm[8].x*W),int(lm[8].y*H))
            cv2.line(f,a,b,(0,255,0),2)
            cv2.putText(f,f"{pv:+.1f} st",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
            cv2.putText(f,f"Vol {vv*100:3.0f}%",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.putText(f,f"[{mode.upper()}] high-five toggles",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            mp.solutions.drawing_utils.draw_landmarks(f,res.multi_hand_landmarks[0],mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow("Pitch/Volume",f)
        if cv2.waitKey(1)&0xFF==27: break
    cam.release(); cv2.destroyAllWindows()

wav,sr=sf.read(AUDIO,dtype="float32")
cur,stop=0,threading.Event()
def audio_loop():
    global cur; pv,vv=0.,1.
    def cb(out,frames,*_):
        nonlocal pv,vv; global cur
        seg=wav[cur:cur+frames]
        if len(seg)<frames:
            seg=np.concatenate([seg,wav[:frames-len(seg)]])
            cur=(cur+frames)%len(wav)
        else: cur+=frames
        try: pv=pitch_q.get_nowait()
        except queue.Empty: pass
        try: vv=vol_q.get_nowait()
        except queue.Empty: pass
        mono=seg.mean(axis=1) if seg.ndim==2 else seg
        out[:,0]=np.clip(rshift(mono,pv)*vv,-0.95,0.95)
    stream=sd.OutputStream(channels=1,samplerate=sr,blocksize=BLOCK,callback=cb)
    stream.start()
    while not stop.is_set(): time.sleep(0.05)
    stream.stop(); stream.close()

if __name__=="__main__":
    threading.Thread(target=audio_loop,daemon=True).start()
    try: cam_loop()
    finally: stop.set(); time.sleep(0.2)

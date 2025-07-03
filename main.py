import os, sys, time, queue, threading, numpy as np
import cv2, mediapipe as mp
import soundfile as sf, sounddevice as sd
from   scipy.signal import resample_poly

def pick_audio():
    folder = os.path.join(os.path.dirname(__file__), "samples")
    files  = [f for f in os.listdir(folder)
              if f.lower().endswith((".wav", ".mp3", ".flac", ".ogg"))]
    if not files:
        print("No audio files in /samples"); sys.exit(1)
    for i,f in enumerate(files,1):
        print(f"{i}. {f}")
    while True:
        c=input("pick: ")
        if c.isdigit() and 1<=int(c)<=len(files):
            return os.path.join(folder,files[int(c)-1])
AUDIO = pick_audio()

CALIB, BLOCK = 2.0, 2048
ALPHA, MAX_ST, VOL_MIN = 0.25, 6, 0.05
COOLDOWN, OPEN = 0.5, 40          

cur_pitch = 0.0
cur_vol   = 1.0
cur_rev   = 0.0                      # 0-dry; 1-wet

def dist(lm,w,h):  return np.linalg.norm([lm[4].x*w-lm[8].x*w,lm[4].y*h-lm[8].y*h])
def up(lm):        return sum(lm[t].y<lm[b].y for t,b in zip([4,8,12,16,20],[2,5,9,13,17]))
def shift(block,st):
    if st==0:return block
    r=2**(st/12); up,down=(int(r*1000),1000) if r>=1 else (1000,int(1000/r))
    out=resample_poly(block,up,down)
    return out[:len(block)] if len(out)>=len(block) else np.pad(out,(0,len(block)-len(out)))

def cam():
    global cur_pitch,cur_vol,cur_rev
    h = mp.solutions.hands.Hands(max_num_hands=1,min_detection_confidence=0.7)
    cam=cv2.VideoCapture(0,cv2.CAP_AVFOUNDATION)
    t0=time.time();dmin,dmax=np.inf,0
    ps,vs=0.,1.; modes=["pitch","volume","reverb"];mi=0
    armed=False;last=0.
    while True:
        ret,frm=cam.read(); frm=cv2.flip(frm,1)
        if not ret: break
        res=h.process(cv2.cvtColor(frm,cv2.COLOR_BGR2RGB))
        if res.multi_hand_landmarks:
            H,W,_=frm.shape; lm=res.multi_hand_landmarks[0].landmark
            d,u=dist(lm,W,H), up(lm); now=time.time()
            if u<=1: armed=True
            elif u==5 and d>OPEN and armed and now-last>COOLDOWN:
                mi=(mi+1)%3; last=now; armed=False; ps,vs=0,1
            if now-t0<CALIB: dmin,dmax=min(dmin,d),max(dmax,d)
            rng=max(10.,dmax-dmin); n=np.clip((d-dmin)/rng,0,1)
            mode=modes[mi]
            if mode=="pitch":
                ps=(1-ALPHA)*ps+ALPHA*((0.5-n)*2*MAX_ST)
                cur_pitch,cur_vol,cur_rev=ps,1.,0.
            elif mode=="volume":
                vs=(1-ALPHA)*vs+ALPHA*n
                cur_pitch,cur_vol,cur_rev=0.,max(VOL_MIN,vs),0.
            else:
                cur_rev   = n
                cur_pitch = 0.0
                cur_vol   = 1.0
            a,b=(int(lm[4].x*W),int(lm[4].y*H)),(int(lm[8].x*W),int(lm[8].y*H))
            cv2.line(frm,a,b,(0,255,0),2)
            cv2.putText(frm,f"{cur_pitch:+.1f} st",(10,65),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)
            cv2.putText(frm,f"Vol {cur_vol*100:3.0f}%",(10,95),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
            cv2.putText(frm,f"Rev {cur_rev*100:3.0f}%",(10,125),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,0,255),2)
            cv2.putText(frm,f"[{mode.upper()}] high-five toggles",(10,25),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            mp.solutions.drawing_utils.draw_landmarks(frm,res.multi_hand_landmarks[0],mp.solutions.hands.HAND_CONNECTIONS)
        cv2.imshow("Pitch/Vol/Reverb",frm)
        if cv2.waitKey(1)&0xFF==27: break
    cam.release(); cv2.destroyAllWindows()

wav,sr=sf.read(AUDIO,dtype="float32")
cursor=0; buf=np.zeros(sr//33,dtype=np.float32); idx=0
stop=threading.Event()
def audio():
    global cursor,idx,cur_pitch,cur_vol,cur_rev
    def cb(out,frames,*_):
        global cursor,idx,cur_pitch,cur_vol,cur_rev
        seg=wav[cursor:cursor+frames]
        if len(seg)<frames:
            seg=np.concatenate([seg,wav[:frames-len(seg)]])
            cursor=(cursor+frames)%len(wav)
        else: cursor+=frames
        mono=seg.mean(axis=1) if seg.ndim==2 else seg
        dry=shift(mono,cur_pitch)*cur_vol
        N=len(buf); mod=(idx+np.arange(frames))%N
        wet=dry+0.8*cur_rev*buf[mod]  
        buf[mod]=wet
        mixed=(1-cur_rev)*dry+cur_rev*wet
        out[:,0]=np.clip(mixed,-0.95,0.95)
        idx=(idx+frames)%N
    stream=sd.OutputStream(channels=1,samplerate=sr,blocksize=BLOCK,callback=cb)
    stream.start(); 
    while not stop.is_set(): time.sleep(0.05)
    stream.stop(); stream.close()

if __name__=="__main__":
    threading.Thread(target=audio,daemon=True).start()
    try: cam()
    finally: stop.set(); time.sleep(0.2)

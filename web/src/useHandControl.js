import { Hands } from "@mediapipe/hands";
import { Camera } from "@mediapipe/camera_utils";

export function startHandControl(videoElt, onCtrl) {
    const st = { dMin: 1e9, dMax: 0, mode: 0, armed: false, last: 0 };

    const hands = new Hands({
        locateFile: f => `/mediapipe-hands/${f}`,
        maxNumHands: 1,
        minDetectionConfidence: 0.7,
        minTrackingConfidence: 0.5,
        modelComplexity: 0
    });

    hands.onResults(r => {
        if (!r.multiHandLandmarks?.length) return;
        const lm = r.multiHandLandmarks[0];
        const dist = Math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y);

        const tips = [4, 8, 12, 16, 20];
        const mcps = [2, 5, 9, 13, 17];
        const upCnt = tips.filter((t, i) => lm[t].y < lm[mcps[i]].y).length;

        if (performance.now() < 2000) {
            st.dMin = Math.min(st.dMin, dist);
            st.dMax = Math.max(st.dMax, dist);
        }
        const norm = (dist - st.dMin) / Math.max(1e-3, st.dMax - st.dMin);
        const clamp = Math.min(1, Math.max(0, norm));

        if (upCnt <= 1) st.armed = true;
        else if (upCnt === 5 && dist > 0.15 && st.armed &&
            performance.now() - st.last > 500) {
            st.mode = (st.mode + 1) % 3;
            st.armed = false;
            st.last = performance.now();
        }

        onCtrl({
            mode: st.mode,
            pitch: (0.5 - clamp) * 12,
            vol: Math.max(0.05, clamp),
            rev: clamp
        });
    });

    const cam = new Camera(videoElt, {
        width: 480, height: 360,
        onFrame: async () => hands.send({ image: videoElt })
    });
    cam.start();

    return () => { cam.stop(); hands.close(); };
}

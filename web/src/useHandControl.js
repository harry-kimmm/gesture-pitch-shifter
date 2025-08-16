import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";

export async function startHandControl(videoEl, onUpdate) {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
    });
    videoEl.srcObject = stream;
    await videoEl.play();

    const wasmBase = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";
    const modelUrl =
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

    const vision = await FilesetResolver.forVisionTasks(wasmBase);
    const hand = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: modelUrl },
        runningMode: "VIDEO",
        numHands: 1,
        minHandDetectionConfidence: 0.5,
        minTrackingConfidence: 0.5,
    });

    let overlay = videoEl.nextSibling;
    if (!(overlay instanceof HTMLCanvasElement)) {
        overlay = document.createElement("canvas");
        overlay.width = videoEl.videoWidth || 640;
        overlay.height = videoEl.videoHeight || 480;
        overlay.style.position = "absolute";
        overlay.style.left = videoEl.offsetLeft + "px";
        overlay.style.top = videoEl.offsetTop + "px";
        overlay.style.pointerEvents = "none";
        videoEl.parentElement?.appendChild(overlay);
    }
    const ctx = overlay.getContext("2d");

    const CALIB_MS = 1500;
    const MAX_ST = 6;
    const ALPHA = 0.25;

    let t0 = performance.now();
    let dMin = Infinity,
        dMax = 0;
    let pitchSm = 0;

    let rafId = 0;
    const tick = () => {
        rafId = requestAnimationFrame(tick);
        if (videoEl.readyState < 2) return;

        const ts = performance.now();
        const res = hand.detectForVideo(videoEl, ts);

        overlay.width = videoEl.videoWidth;
        overlay.height = videoEl.videoHeight;
        const W = overlay.width,
            H = overlay.height;

        ctx.clearRect(0, 0, W, H);
        ctx.drawImage(videoEl, 0, 0, W, H);

        if (!res?.landmarks?.length) {
            onUpdate?.({ pitchSt: pitchSm, dbg: { hasHand: false } });
            return;
        }

        const lm = res.landmarks[0];
        const p = (i) => ({ x: lm[i].x * W, y: lm[i].y * H });
        const pThumb = p(4);
        const pIndex = p(8);

        ctx.strokeStyle = "rgb(0,255,0)";
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pThumb.x, pThumb.y);
        ctx.lineTo(pIndex.x, pIndex.y);
        ctx.stroke();

        const dx = pThumb.x - pIndex.x;
        const dy = pThumb.y - pIndex.y;
        const dist = Math.hypot(dx, dy);

        const now = performance.now();
        if (now - t0 < CALIB_MS) {
            dMin = Math.min(dMin, dist);
            dMax = Math.max(dMax, dist);
            ctx.fillStyle = "rgba(255,0,0,0.9)";
            ctx.font = "20px system-ui, sans-serif";
            ctx.fillText("Calibrating... pinch/spread", 12, 28);
        }

        const rng = Math.max(10, dMax - dMin);
        const n = Math.max(0, Math.min(1, (dist - dMin) / rng));

        const rawPitch = (0.5 - n) * 2 * MAX_ST;
        pitchSm = (1 - ALPHA) * pitchSm + ALPHA * rawPitch;

        ctx.fillStyle = "rgba(0,0,0,0.5)";
        ctx.fillRect(8, H - 48, 170, 40);
        ctx.fillStyle = "#fff";
        ctx.font = "16px system-ui, sans-serif";
        ctx.fillText(`pitch: ${pitchSm.toFixed(1)} st`, 16, H - 20);

        onUpdate?.({
            pitchSt: pitchSm,
            dbg: { hasHand: true, n, dist, dMin, dMax },
        });
    };
    tick();

    return () => {
        cancelAnimationFrame(rafId);
        try {
            hand.close();
        } catch { }
        stream.getTracks().forEach((t) => t.stop());
        if (overlay && overlay.parentElement) overlay.parentElement.removeChild(overlay);
    };
}

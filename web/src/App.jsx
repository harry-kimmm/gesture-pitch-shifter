import { useRef, useState, useEffect, useCallback } from "react";
import { startHandControl } from "./useHandControl";

export default function App() {
  const videoRef = useRef(null);
  const engine = useRef(null);
  const recRef = useRef(null);
  const playRef = useRef(null);

  const [ready, setReady] = useState(false);
  const [recURL, setRecURL] = useState(null);

  function buildGraph(audioBuffer) {
    if (engine.current) {
      engine.current.src?.stop();
      engine.current.ctx.close();
    }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const src = ctx.createBufferSource(); src.buffer = audioBuffer; src.loop = true;
    const vol = ctx.createGain(); vol.gain.value = 1;
    const dry = ctx.createGain(); dry.gain.value = 1;
    const wet = ctx.createGain(); wet.gain.value = 0;
    const del = ctx.createDelay(); del.delayTime.value = 0.03;
    const fb = ctx.createGain(); fb.gain.value = 0.7;

    src.connect(vol).connect(dry).connect(ctx.destination);
    dry.connect(wet).connect(del).connect(fb).connect(wet);
    wet.connect(ctx.destination);

    const dest = ctx.createMediaStreamDestination();
    dry.connect(dest); wet.connect(dest);
    recRef.current = new MediaRecorder(dest.stream);
    recRef.current.ondataavailable = e => setRecURL(URL.createObjectURL(e.data));

    engine.current = { ctx, src, vol, dry, wet };
    ctx.resume().then(() => src.start());
    setReady(true);
  }

  /* ── file input handler ─────────────────────────────── */
  async function handleFile(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    const arr = await file.arrayBuffer();
    const buf = await new AudioContext().decodeAudioData(arr);
    setRecURL(null);                               // clear old recording
    buildGraph(buf);
  }

  /* ── mute / resume loop during playback of take ─────── */
  function stopLoop() { engine.current?.src?.stop(); engine.current.src = null; }
  function resumeLoop() {
    if (engine.current && !engine.current.src) {
      const { ctx, vol } = engine.current;
      const src = ctx.createBufferSource();
      src.buffer = engine.current.vol.context.createBufferSource().buffer;
      src.buffer = engine.current.dry.context.createBufferSource().buffer;
      src.buffer = engine.current.wet.context.createBufferSource().buffer;
      src.buffer = engine.current.dry.context.createBufferSource().buffer;
    }
  }

  useEffect(() => {
    if (!recURL) return;
    const p = playRef.current;
    p.onplay = stopLoop;
    p.onended = resumeLoop;
    return () => { p.onplay = p.onended = null; };
  }, [recURL]);

  const onCtrl = useCallback(({ mode, pitch, vol, rev }) => {
    const g = engine.current;
    if (!g?.src) return;
    if (mode === 0) { g.src.playbackRate.value = 2 ** (pitch / 12); g.vol.gain.value = 1; g.dry.gain.value = 1; g.wet.gain.value = 0; }
    if (mode === 1) { g.src.playbackRate.value = 1; g.vol.gain.value = vol; g.dry.gain.value = 1; g.wet.gain.value = 0; }
    if (mode === 2) { g.src.playbackRate.value = 1; g.vol.gain.value = 1; g.dry.gain.value = 1 - rev; g.wet.gain.value = rev; }
  }, []);

  useEffect(() => {
    if (ready) return startHandControl(videoRef.current, onCtrl);
  }, [ready, onCtrl]);

  return (
    <div style={{ padding: 20, minHeight: "100vh", background: "#222", color: "#eee", fontFamily: "sans-serif" }}>
      <input type="file" accept="audio/*" onChange={handleFile} />
      <br /><br />
      <video ref={videoRef} width="320" autoPlay muted style={{ borderRadius: 8 }} />
      {ready && (
        <>
          <br />
          <button onClick={() => recRef.current.start()}>Start REC</button>
          <button onClick={() => recRef.current.stop()} style={{ marginLeft: 10 }}>Stop REC</button>
        </>
      )}
      {recURL && (
        <>
          <h3>Recording</h3>
          <audio ref={playRef} controls src={recURL} />
          <br />
          <a href={recURL} download="take.webm" style={{ color: "#0af" }}>download</a>
        </>
      )}
    </div>
  );
}

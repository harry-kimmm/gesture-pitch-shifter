import { useEffect, useRef, useState, useCallback } from "react";
import { startHandControl } from "./useHandControl";

export default function App() {
  const videoRef = useRef(null);

  const audioRef = useRef({
    ctx: null,
    src: null,
    gain: null,
    dry: null,
    wet: null,
    convolver: null,
    dest: null,
    rec: null,
    monitor: null,
  });

  const [ready, setReady] = useState(false);
  const [recURL, setRecURL] = useState(null);
  const [status, setStatus] = useState("load an audio file");
  const [muted, setMuted] = useState(false);

  async function handleFile(e) {
    if (!e.target.files.length) return;

    try { audioRef.current?.src?.stop(); audioRef.current?.ctx?.close(); } catch { }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const arrBuf = await e.target.files[0].arrayBuffer();
    const buf = await ctx.decodeAudioData(arrBuf);

    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.loop = true;

    const gain = ctx.createGain();
    gain.gain.value = 1;

    const convolver = ctx.createConvolver();
    convolver.normalize = true;

    const wet = ctx.createGain(); wet.gain.value = 0.0;
    const dry = ctx.createGain(); dry.gain.value = 1.0;

    const dest = ctx.createMediaStreamDestination();

    const monitor = ctx.createGain();
    monitor.gain.value = 1.0;

    src.connect(gain);

    gain.connect(dry);
    dry.connect(monitor);
    dry.connect(dest);

    gain.connect(wet);
    wet.connect(convolver);
    convolver.connect(monitor);
    convolver.connect(dest);

    monitor.connect(ctx.destination);

    const rec = new MediaRecorder(dest.stream);
    rec.ondataavailable = (ev) => {
      if (ev.data?.size) setRecURL((old) => {
        if (old) URL.revokeObjectURL(old);
        return URL.createObjectURL(ev.data);
      });
    };

    audioRef.current = { ctx, src, gain, dry, wet, convolver, dest, rec, monitor };

    await ctx.resume();
    src.start();
    setReady(true);
    setStatus("playing");
  }

  useEffect(() => {
    if (!ready || !videoRef.current) return;
    let stop = null;

    startHandControl(videoRef.current, ({ pitchSt, volume, mode }) => {
      const a = audioRef.current;
      if (!a?.src) return;

      a.src.playbackRate.value = Math.pow(2, pitchSt / 12);

      a.gain.gain.value = volume;

      const mutedFlag = a?.monitor?.gain.value === 0;
      setStatus(
        `mode ${mode} | pitch ${pitchSt.toFixed(1)} st | vol ${(volume * 100).toFixed(0)}% ${mutedFlag ? "(muted)" : ""}`
      );
    }).then((s) => (stop = s));

    return () => stop?.();
  }, [ready]);

  const startRec = useCallback(() => {
    const a = audioRef.current;
    if (!a?.rec) return;
    setRecURL((old) => { if (old) URL.revokeObjectURL(old); return null; });
    a.rec.start();
  }, []);
  const stopRec = useCallback(() => {
    const a = audioRef.current;
    if (!a?.rec) return;
    try { a.rec.stop(); } catch { }
  }, []);

  const toggleMute = useCallback(() => {
    const a = audioRef.current;
    if (!a?.monitor) return;
    const next = a.monitor.gain.value === 0 ? 1 : 0;
    a.monitor.gain.value = next;
    setMuted(next === 0);
  }, []);

  return (
    <div style={{
      padding: 16, minHeight: "100vh", color: "#eee",
      background: "#1f1f1f", fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, sans-serif"
    }}>
      <input type="file" accept="audio/*" onChange={handleFile} />
      <div style={{ height: 12 }} />
      <div style={{ position: "relative", width: 640, maxWidth: "95vw" }}>
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          style={{
            width: "100%",
            borderRadius: 12,
            display: "block",
            filter: "contrast(1.05) saturate(1.05)"
          }}
        />
      </div>

      <div style={{ height: 12 }} />
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        <button onClick={startRec} disabled={!ready} style={btn}>Start REC</button>
        <button onClick={stopRec} disabled={!ready} style={btn}>Stop REC</button>
        <button onClick={toggleMute} disabled={!ready} style={btn}>
          {muted ? "Unmute" : "Mute"}
        </button>
      </div>

      <div style={{ height: 16 }} />
      <div style={{ opacity: ready ? 1 : 0.6 }}>
        <strong>Status:</strong> {status}
      </div>

      {recURL && (
        <>
          <h3>Recording</h3>
          <audio
            controls
            src={recURL}
            onPlay={() => {
              const a = audioRef.current;
              if (a?.monitor) { a.monitor.gain.value = 0; setMuted(true); }
            }}
          />
          <div><a href={recURL} download="take.webm" style={{ color: "#0af" }}>download</a></div>
        </>
      )}
    </div>
  );
}

const btn = {
  padding: "10px 16px",
  background: "#2b2b2b",
  color: "#fff",
  border: "1px solid #444",
  borderRadius: 10,
  cursor: "pointer",
};

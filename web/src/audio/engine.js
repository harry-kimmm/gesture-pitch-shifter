import { SimpleFilter } from "soundtouchjs";

export async function createEngine(ctx, arrayBuffer) {
    const buffer = await ctx.decodeAudioData(arrayBuffer);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.loop = true;

    const vol = ctx.createGain();
    const wet = ctx.createGain();
    const dry = ctx.createGain();
    const comb = ctx.createDelay();
    comb.delayTime.value = 0.03;
    const fbGain = ctx.createGain();
    fbGain.gain.value = 0.4;

    const filter = new SimpleFilter(buffer, 44100);
    const node = ctx.createScriptProcessor(1024, 1, 1);
    node.onaudioprocess = e => {
        const out = e.outputBuffer.getChannelData(0);
        const n = filter.extract(out, 1024);
        if (n < 1024) out.fill(0, n);
    };

    src.connect(vol).connect(node);
    node.connect(dry).connect(ctx.destination);
    dry.connect(wet).connect(comb).connect(fbGain).connect(wet);
    wet.connect(ctx.destination);

    return {
        start: () => src.start(0),
        setPitch: st => filter.tempo = 1 / (2 ** (st / 12)),
        setVolume: v => vol.gain.value = v,
        setReverb: mix => { dry.gain.value = 1 - mix; wet.gain.value = mix; },
    };
}

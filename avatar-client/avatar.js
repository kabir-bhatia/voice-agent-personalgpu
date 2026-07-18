/* Browser lip-sync avatar for the self-hosted voice agent.
 *
 * Connects to the LiveKit room, plays the agent's audio, and animates a cartoon mouth from the
 * viseme timelines the agent publishes on the `lk.viseme` data topic. Timing strategy: the agent
 * publishes one timeline per spoken sentence; we queue them and anchor each to the real audio
 * ONSET detected from the incoming track (Web Audio energy), so shapes stay in step with speech
 * regardless of network/synthesis latency. Audio energy also modulates jaw openness for liveliness.
 */
(() => {
  "use strict";
  const LK = window.LivekitClient;
  const $ = (id) => document.getElementById(id);

  // ---- query params (accept meet.livekit.io style names too) ----
  const q = new URLSearchParams(location.search);
  const URL_ = q.get("url") || q.get("liveKitUrl") || q.get("livekitUrl") || "";
  const TOKEN = q.get("token") || "";
  const DEBUG = q.get("debug") === "1";

  // ---- opt-in tuning overlay (?debug=1): live RMS / envelope / viseme / closure flag ----
  let dbgEl = null;
  if (DEBUG) {
    dbgEl = document.createElement("div");
    dbgEl.style.cssText = "position:fixed;top:8px;left:8px;z-index:9999;font:12px/1.4 monospace;" +
      "background:rgba(0,0,0,.72);color:#7CFC9A;padding:8px 10px;border-radius:8px;white-space:pre;pointer-events:none";
    document.body.appendChild(dbgEl);
  }

  // ---- viseme -> parametric mouth shape (half-width px, openness 0..1, rounding 0..1) ----
  const SHAPE = {
    sil: { hw: 30, open: 0.02, round: 0.2 },
    PP:  { hw: 34, open: 0.03, round: 0.1 },   // lips pressed (म/प/ब)
    FF:  { hw: 30, open: 0.10, round: 0.15 },
    TH:  { hw: 30, open: 0.28, round: 0.2 },
    DD:  { hw: 30, open: 0.34, round: 0.2 },
    kk:  { hw: 30, open: 0.34, round: 0.2 },
    CH:  { hw: 28, open: 0.30, round: 0.45 },
    SS:  { hw: 30, open: 0.20, round: 0.2 },
    nn:  { hw: 30, open: 0.26, round: 0.2 },
    RR:  { hw: 29, open: 0.32, round: 0.3 },
    aa:  { hw: 32, open: 1.00, round: 0.2 },    // wide open
    E:   { hw: 36, open: 0.55, round: 0.1 },    // wide
    ih:  { hw: 35, open: 0.40, round: 0.1 },
    oh:  { hw: 22, open: 0.75, round: 0.85 },   // rounded O
    ou:  { hw: 18, open: 0.45, round: 0.95 },   // pursed
  };
  const MAX_OPEN = 30; // px ry at full open

  // ---- animation state ----
  let speaking = false;
  let mHW = 30, mRY = 4, mRound = 0.2; // smoothed mouth params

  // ---- lip-sync tuning constants (adjust here; watch them live with ?debug=1) ----------------
  const FLOOR    = 0.012; // RMS below this = silence -> mouth hard shut (noise-gate floor)
  const T_OPEN   = 0.035; // RMS above this opens the gate (start speaking)
  const T_CLOSE  = 0.020; // must fall below this to close the gate again (hysteresis)
  const FULL     = 0.13;  // RMS mapped to fully-open mouth
  const ATTACK   = 0.6;   // envelope rise speed (per frame)
  const RELEASE  = 0.8;   // envelope fall speed (per frame) — faster than attack, snappy close
  const HANGOVER_MS = 40; // keep "speaking" this long after a dip so intra-word gaps don't chatter
  const DIP_FRAC = 0.4;   // RMS < DIP_FRAC * recentPeak (while speaking) => bilabial/stop CLOSURE
  const CLOSURE_MS = 85;  // hold the lip-press this long after a detected closure
  const HOLD_MS    = 55;  // min time to keep a chosen vowel shape (anti-flicker)

  // Audio-driven lip sync. We analyse the SAME audio that is played (routed through Web Audio, see
  // TrackSubscribed) so the mouth is sample-aligned with what you hear. Openness comes from an
  // envelope-gated RMS (hard noise-gate => shuts instantly on silence). Bilabials م/ब/प are energy
  // dips, so a sharp drop vs the recent peak snaps the lips shut (PP). Vowel shape comes from the
  // spectrum. sampleRate set on connect.
  let analyser = null, tdata = null, fdata = null, sampleRate = 48000, envE = 0;
  let gateOpen = false, lastLoud = -1e9, closureUntil = 0;
  let heldV = "sil", heldUntil = 0;
  const rmsHist = new Float32Array(8); let rmsIdx = 0; // ~120ms ring @ 60fps

  // exposed for the debug overlay
  const dbg = { rms: 0, env: 0, v: "sil", closure: false, peak: 0 };

  function analyse(now) {
    if (!analyser) return { g: 0, v: "sil" };
    analyser.getFloatTimeDomainData(tdata);
    let s = 0; for (let i = 0; i < tdata.length; i++) s += tdata[i] * tdata[i];
    const rms = Math.sqrt(s / tdata.length);

    // recent peak (for closure detection) BEFORE storing the current sample
    let peak = 0; for (let i = 0; i < rmsHist.length; i++) if (rmsHist[i] > peak) peak = rmsHist[i];
    rmsHist[rmsIdx] = rms; rmsIdx = (rmsIdx + 1) % rmsHist.length;

    // noise-gate with hysteresis
    if (rms > T_OPEN) { gateOpen = true; lastLoud = now; }
    else if (rms < T_CLOSE && (now - lastLoud) > HANGOVER_MS) gateOpen = false;
    const voiced = gateOpen && rms >= FLOOR;

    // envelope openness
    let target = voiced ? Math.min(1, (rms - FLOOR) / (FULL - FLOOR)) : 0;
    if (target < 0) target = 0;
    envE += (target - envE) * (target > envE ? ATTACK : RELEASE);
    if (!voiced && envE < 0.02) envE = 0; // snap fully shut

    dbg.rms = rms; dbg.env = envE; dbg.peak = peak;

    if (!voiced || envE < 0.06) { dbg.v = "sil"; dbg.closure = false; heldV = "sil"; return { g: envE, v: "sil" }; }

    // CLOSURE (bilabial/stop): sharp dip vs recent speech-level peak -> press lips shut
    if (peak > T_OPEN && rms < DIP_FRAC * peak) closureUntil = now + CLOSURE_MS;
    if (now < closureUntil) { dbg.v = "PP"; dbg.closure = true; heldV = "PP"; heldUntil = now + 30;
      return { g: Math.min(envE, 0.25), v: "PP" }; }
    dbg.closure = false;

    // vowel/consonant shape from the spectrum, with a short hold to avoid per-frame flicker
    if (now < heldUntil) { dbg.v = heldV; return { g: envE, v: heldV }; }
    analyser.getByteFrequencyData(fdata);
    const binHz = sampleRate / analyser.fftSize;
    const band = (f1, f2) => {
      let a = 0, n = 0;
      const i0 = Math.max(1, Math.floor(f1 / binHz)), i1 = Math.min(fdata.length - 1, Math.floor(f2 / binHz));
      for (let i = i0; i <= i1; i++) { a += fdata[i]; n++; }
      return n ? a / n : 0;
    };
    const low = band(180, 850), mid = band(850, 2400), high = band(2400, 5000), sib = band(5000, 9500);
    let v;
    if (sib > 55 && sib > mid) v = "SS";                 // s/श/ष hiss -> narrow, teeth
    else if (high > 55 && high > mid * 1.1) v = "CH";     // ch/j/sh
    else if (low > mid * 1.15) v = envE > 0.62 ? "aa" : "oh"; // low-centroid: open/rounded
    else v = envE > 0.5 ? "E" : "ih";                    // mid-centroid: wide
    heldV = v; heldUntil = now + HOLD_MS; dbg.v = v;
    return { g: envE, v };
  }

  const lerp = (a, b, k) => a + (b - a) * k;

  const mouth = $("mouth"), tongue = $("tongue"),
        eyeL = $("eyeL"), eyeR = $("eyeR"), head = $("head");
  let blink = 1, nextBlink = performance.now() + 2500, blinkT = 0;

  function render() {
    const now = performance.now();
    const { g, v } = analyse(now);    // live, sample-aligned openness + shape
    const isSpeech = g > 0.08;
    const sh = SHAPE[v] || SHAPE.sil;
    // Openness is DRIVEN by the audio gate; the viseme shapes it (width, rounding, and how open
    // that phoneme is). Silence => g≈0 => shut, in the same frame the audio goes quiet.
    const tgtHW = sh.hw;
    const tgtRound = sh.round;
    const tgtOpen = g * sh.open;

    // Snappy smoothing; close faster than open so the mouth never lingers open after audio stops.
    const closing = (3 + tgtOpen * MAX_OPEN) < mRY;
    mHW = lerp(mHW, tgtHW, 0.5);
    mRY = lerp(mRY, 3 + tgtOpen * MAX_OPEN, closing ? 0.7 : 0.55);
    mRound = lerp(mRound, tgtRound, 0.45);
    mouth.setAttribute("rx", mHW.toFixed(1));
    mouth.setAttribute("ry", mRY.toFixed(1));
    // tongue shows only when reasonably open
    const to = Math.max(0, mRY - 12);
    tongue.setAttribute("ry", (to * 0.5).toFixed(1));
    tongue.setAttribute("cy", (212 + mRY * 0.35).toFixed(1));
    tongue.setAttribute("rx", (mHW * (0.5 - 0.25 * mRound)).toFixed(1));

    // blink
    if (now > nextBlink) { blinkT = 1; nextBlink = now + 1800 + Math.random() * 3200; }
    if (blinkT > 0) { blinkT = Math.max(0, blinkT - 0.18); }
    blink = 1 - Math.sin(blinkT * Math.PI) * 0.95;
    eyeL.setAttribute("transform", `translate(109 140) scale(1 ${blink.toFixed(3)}) translate(-109 -140)`);
    eyeR.setAttribute("transform", `translate(191 140) scale(1 ${blink.toFixed(3)}) translate(-191 -140)`);

    // subtle idle sway + a little more motion while speaking
    const amp = speaking ? 3.2 : 1.6;
    const dx = Math.sin(now / 1300) * amp, dy = Math.cos(now / 1700) * amp * 0.6;
    const rot = Math.sin(now / 1500) * (speaking ? 1.4 : 0.7);
    head.setAttribute("transform", `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) rotate(${rot.toFixed(2)} 150 160)`);

    speaking = isSpeech;

    if (dbgEl) {
      const bar = (x, m) => "█".repeat(Math.round(Math.min(1, x / m) * 20)).padEnd(20, "·");
      dbgEl.textContent =
        `rms   ${dbg.rms.toFixed(3)} ${bar(dbg.rms, FULL)}\n` +
        `peak  ${dbg.peak.toFixed(3)}\n` +
        `env   ${dbg.env.toFixed(2)}  ${bar(dbg.env, 1)}\n` +
        `viseme ${dbg.v}${dbg.closure ? "  <CLOSURE>" : ""}\n` +
        `gate  ${gateOpen ? "OPEN" : "shut"}   FLOOR ${FLOOR} T_OPEN ${T_OPEN}`;
    }
    requestAnimationFrame(render);
  }

  // ---- LiveKit connection ----
  function setStatus(txt, on) { $("status").textContent = txt; $("dot").classList.toggle("on", !!on); }

  async function connect() {
    if (!URL_ || !TOKEN) { $("err").textContent = "Missing ?url= and ?token= in the link."; return; }
    $("start").disabled = true; setStatus("connecting…", false);
    const room = new LK.Room({ adaptiveStream: true, dynacast: true });

    room.on(LK.RoomEvent.TrackSubscribed, (track) => {
      if (track.kind !== LK.Track.Kind.Audio) return;
      const ms = new MediaStream([track.mediaStreamTrack]);
      // Chrome quirk: a MediaStreamSource only "pulls" audio if the stream is also attached to a
      // (muted) media element. We keep that element muted and route real playback THROUGH Web Audio
      // so the samples we analyse are exactly the samples going to the speaker (perfect alignment).
      const sink = track.attach(); sink.muted = true; sink.autoplay = true;
      sink.setAttribute("playsinline", ""); document.body.appendChild(sink);
      const ac = new (window.AudioContext || window.webkitAudioContext)();
      ac.resume();
      sampleRate = ac.sampleRate;
      const src = ac.createMediaStreamSource(ms);
      analyser = ac.createAnalyser(); analyser.fftSize = 1024; analyser.smoothingTimeConstant = 0.05;
      tdata = new Float32Array(analyser.fftSize);
      fdata = new Uint8Array(analyser.frequencyBinCount);
      src.connect(analyser);
      analyser.connect(ac.destination); // play here — aligned with analysis
      setStatus("connected", true);
    });

    room.on(LK.RoomEvent.Disconnected, () => setStatus("disconnected", false));

    try {
      await room.connect(URL_, TOKEN);
      try {
        await room.localParticipant.setMicrophoneEnabled(true); // user speaks to the agent
      } catch (e) {
        $("err").textContent = "Mic unavailable — avatar will still show the agent. " + (e.message || "");
      }
      $("overlay").style.display = "none";
      setStatus("connected", true);
    } catch (e) {
      $("err").textContent = "Connect failed: " + (e && e.message ? e.message : e);
      $("start").disabled = false; setStatus("idle", false);
    }
  }

  $("start").addEventListener("click", connect);
  requestAnimationFrame(render); // animate immediately (idle sway + blinks) before connecting
})();

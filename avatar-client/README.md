# avatar-client — lip-synced cartoon avatar

A static browser front-end that renders a **viseme-rigged cartoon face** lip-synced to the agent's
speech. Runs entirely in the browser (no GPU, no build step). See `../ARCHITECTURE.md` §9 for the design.

## Files
- `index.html` — the page + the cartoon SVG (head, eyes/blink, parametric mouth).
- `avatar.js` — LiveKit connection + audio-driven lip-sync (RMS → openness, spectrum → shape,
  energy-dip → م/ब/प closures). `?debug=1` shows a live tuning overlay.
- `livekit-client.umd.min.js` — vendored LiveKit browser SDK (v2 UMD; no CDN on the hot path).
  Refresh with: `curl -L https://cdn.jsdelivr.net/npm/livekit-client@2/dist/livekit-client.umd.min.js -o livekit-client.umd.min.js`

## Run
```bash
# localhost is a secure context, so the mic works without HTTPS
python -m http.server 8777 --bind 127.0.0.1
# mint a token on the VM:  python ../livekit/gen_token.py --room demo --identity me --minutes 120
# open in Chrome:
#   http://localhost:8777/index.html?url=wss://<DOMAIN>&token=<TOKEN>
# add &debug=1 for the tuning overlay
```
Click **Connect**, allow the mic, and speak — the agent replies and the mouth lip-syncs.

## Notes
- Requires a secure context for `getUserMedia`: use `localhost` or serve over HTTPS.
- Hosting on the VM over HTTPS (any device, no local server) is a pending task — it does **not** affect
  lip-sync/latency (the audio path is identical); it only changes page delivery.

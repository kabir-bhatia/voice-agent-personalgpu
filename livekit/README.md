# livekit-server — self-hosted LiveKit SFU (Docker + Caddy auto-TLS)

A self-contained, reusable LiveKit deployment. Runs the SFU on a single VM with automatic
Let's Encrypt TLS via Caddy, so browsers can join over `wss://`. Decoupled from any app — drop it
into any project by changing `.env`.

## What's here
| File | Purpose |
|------|---------|
| `docker-compose.yaml` | `livekit-server` + `caddy` (host networking) |
| `livekit.yaml.template` | SFU config (keys, RTC ports, `use_external_ip`, TURN) |
| `caddy.yaml.template` | Caddy layer4: auto-TLS + multiplex `:443` between WSS and TURN/TLS |
| `up.sh` / `down.sh` | render templates from `.env`, then `docker compose up -d` / `down` |
| `gen_token.py` | mint a join token + `meet.livekit.io` URL (no SDK dependency) |
| `.env.example` | copy to `.env`, fill in DOMAIN + key/secret |

## Deploy (fresh VM)
1. **DNS/domain.** Point a hostname at the VM's public IP. No domain? Use sslip.io:
   IP `a.b.c.d` → `a-b-c-d.sslip.io`.
2. **Firewall.** Open ingress on the VM's cloud firewall:
   `443/tcp`, `80/tcp`, `7881/tcp`, `3478/udp`, `50000-60000/udp`.
   (Reserve a **static IP** so the domain/cert survive restarts.)
3. **Config.** `cp .env.example .env` and set `DOMAIN`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET`.
4. **Run.** `./up.sh` — watch the cert with `docker compose logs -f caddy`.
5. **Test.** `python gen_token.py --room test --identity me` → open the printed URL.

## Wiring an app/agent on the SAME VM
- Agent/worker connects over loopback (no TLS, lowest latency): `LIVEKIT_URL=ws://localhost:7880`.
- Browser clients use `wss://<DOMAIN>`.
- Both must use the same `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` as `.env`.

## Reuse in another project
Copy this folder, set a new `DOMAIN` + a fresh key/secret in `.env`, open the firewall, `./up.sh`.
Nothing here references any specific app.

## Ports
| Port | Proto | Purpose |
|------|-------|---------|
| 443 | TCP | WSS signaling + TURN/TLS (Caddy) |
| 80 | TCP | Let's Encrypt ACME challenge |
| 7881 | TCP | WebRTC over TCP (fallback) |
| 3478 | UDP | TURN/UDP |
| 50000-60000 | UDP | WebRTC media |
| 7880 | TCP | SFU HTTP/WS — **internal only** (agent via loopback), not firewalled open |

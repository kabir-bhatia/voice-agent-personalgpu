#!/usr/bin/env bash
# Render config from templates using .env, then start LiveKit + Caddy.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "ERROR: .env not found. cp .env.example .env and edit it."; exit 1; }
set -a; . ./.env; set +a
: "${DOMAIN:?set DOMAIN in .env}"
: "${LIVEKIT_API_KEY:?set LIVEKIT_API_KEY in .env}"
: "${LIVEKIT_API_SECRET:?set LIVEKIT_API_SECRET in .env}"

envsubst '${DOMAIN} ${LIVEKIT_API_KEY} ${LIVEKIT_API_SECRET}' < livekit.yaml.template > livekit.yaml
envsubst '${DOMAIN}' < caddy.yaml.template > caddy.yaml

docker compose up -d
echo "--- containers ---"
docker compose ps
echo
echo "LiveKit up. WSS URL for clients:  wss://${DOMAIN}"
echo "Agent (same host) should use:     ws://localhost:7880"
echo "Watch cert issuance:  docker compose logs -f caddy"

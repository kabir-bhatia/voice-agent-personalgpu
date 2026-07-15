#!/usr/bin/env python3
"""Standalone LiveKit access-token minter. No livekit SDK dependency (pure stdlib HS256 JWT), so
it drops into any project. Reads DOMAIN / LIVEKIT_API_KEY / LIVEKIT_API_SECRET from ./.env.

Usage:
    python gen_token.py --room myroom --identity kabir --minutes 60
Prints a token and a ready-to-open meet.livekit.io join URL (wss://DOMAIN).
"""
import argparse
import base64
import hashlib
import hmac
import json
import os
import time


def _b64url(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_token(key: str, secret: str, room: str, identity: str, minutes: int) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": key,
        "sub": identity,
        "name": identity,
        "nbf": now - 10,
        "exp": now + minutes * 60,
        "video": {
            "roomJoin": True,
            "room": room,
            "canPublish": True,
            "canSubscribe": True,
            "canPublishData": True,
        },
    }
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode())
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode())
    )
    sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return signing_input + "." + _b64url(sig)


def _load_env(path: str = ".env") -> dict:
    env = {}
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), path)
    if os.path.exists(here):
        for line in open(here, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    # environment overrides file
    for k in ("DOMAIN", "LIVEKIT_API_KEY", "LIVEKIT_API_SECRET"):
        if os.environ.get(k):
            env[k] = os.environ[k]
    return env


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--room", default="test-room")
    ap.add_argument("--identity", default="user")
    ap.add_argument("--minutes", type=int, default=60)
    args = ap.parse_args()

    env = _load_env()
    domain = env["DOMAIN"]
    key = env["LIVEKIT_API_KEY"]
    secret = env["LIVEKIT_API_SECRET"]

    token = make_token(key, secret, args.room, args.identity, args.minutes)
    wss = f"wss://{domain}"
    print(f"\nToken (valid {args.minutes} min):\n\n{token}\n")
    print("Join URL (open in browser):\n")
    print(f"https://meet.livekit.io/?liveKitUrl={wss}&token={token}\n")


if __name__ == "__main__":
    main()

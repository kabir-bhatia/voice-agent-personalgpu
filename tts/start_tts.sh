#!/bin/bash
cd ~/tts_compare
set -a; [ -f ~/v6/.env.local ] && source ~/v6/.env.local; set +a
export F5_DTYPE=${F5_DTYPE:-fp32}
export F5_CFG=${F5_CFG:-2.0}
exec .venv-f5/bin/uvicorn tts_server:app --host 0.0.0.0 --port 8002

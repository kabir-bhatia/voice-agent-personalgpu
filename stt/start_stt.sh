#!/bin/bash
cd ~/asr_indic
export LD_LIBRARY_PATH=$(ls -d $PWD/.venv/lib/python3.12/site-packages/nvidia/*/lib | tr '\n' ':')$LD_LIBRARY_PATH
export INDIC_LANG=hi
exec .venv/bin/uvicorn indic_server:app --host 0.0.0.0 --port 8001

#!/bin/bash
# vLLM serving Gemma-3-12B AWQ as the LLM (OpenAI API on 127.0.0.1:8000, served name "gemma2-9b").
# HF weights + the gemma_system.jinja chat template are mounted from the host.
sudo docker run -d --name vllm --restart unless-stopped --runtime runc --gpus all \
  -p 127.0.0.1:8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/../agent/gemma_system.jinja:/chat-templates/gemma_system.jinja \
  -e HF_TOKEN="$HF_TOKEN" -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:latest \
  --model gaunernst/gemma-3-12b-it-int4-awq --quantization awq_marlin \
  --max-model-len 4096 --gpu-memory-utilization 0.55 --served-model-name gemma2-9b \
  --chat-template /chat-templates/gemma_system.jinja

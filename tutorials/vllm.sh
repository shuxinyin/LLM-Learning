python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0\
    --port 6006 \
    --model /root/ckpt/Qwen2.5-7B-Instruct  \
    --served-model-name Qwen2.5-7B-Instruct \
    --max-model-len=32768
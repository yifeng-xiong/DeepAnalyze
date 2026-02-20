export CUDA_VISIBLE_DEVICES=1,2

# Use both GPUs via tensor parallelism; reduce context length to avoid OOM
python -m vllm.entrypoints.openai.api_server \
  --model /home/yifengx4/DeepAnalyze/DeepAnalyze-8B \
  --served-model-name DeepAnalyze-8B \
  --tensor-parallel-size 2 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --trust-remote-code
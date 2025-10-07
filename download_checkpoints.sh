export HTTPS_PROXY=http://10.4.168.171:7890
export https_proxy=http://10.4.168.171:7890
# export HF_ENDPOINT="https://hf-mirror.com"

model_id="Qwen/Qwen3-4B-Instruct-2507"

# export HF_HUB_ENABLE_HF_TRANSFER=1

hf download "$model_id" --local-dir ./huggingface/"$model_id"

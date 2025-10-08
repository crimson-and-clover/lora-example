export HTTPS_PROXY=http://10.4.168.171:7890
export https_proxy=http://10.4.168.171:7890
export HF_ENDPOINT="https://hf-mirror.com"

datasets_id="OpenAssistant/oasst1"

hf download "$datasets_id" --repo-type dataset --local-dir ./huggingface/"$datasets_id"

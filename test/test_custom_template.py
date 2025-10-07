import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

model_name = "huggingface/meta-llama/Llama-3.2-3B-Instruct"
tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

datasets_id = "huggingface/allenai/tulu-3-sft-mixture"

raw_ds = load_dataset(datasets_id, split="train")


def filter_func(x):
    msg = x["messages"]
    if not any(m["role"] == "assistant" and m["content"].strip() != "" for m in msg):
        return False
    src = x["source"]
    allowed_src = ["math", "science", "history", "literature"]
    for allowed in allowed_src:
        if allowed in src:
            return True
    return False


raw_ds = raw_ds.filter(filter_func).flatten_indices()

raw_ds = raw_ds.train_test_split(test_size=0.005, seed=42)

train_ds = raw_ds["train"]
eval_ds = raw_ds["test"]

train_ds = train_ds.shuffle(seed=42)

mini_ds = train_ds.select(range(10))

print("size of train dataset: ", len(train_ds))
print("size of eval dataset: ", len(eval_ds))

my_template = ""

with open("llama-3.2.jinja2", "r", encoding="utf-8") as f:
    my_template = f.read()

msg = mini_ds[0]["messages"]

origin_str = tokenizer.apply_chat_template(
    msg, tokenize=False, add_generation_prompt=True)

my_str = tokenizer.apply_chat_template(
    msg, tokenize=False, chat_template=my_template, add_generation_prompt=True)

assert origin_str == my_str

origin_str = tokenizer.apply_chat_template(msg, tokenize=False)

my_str = tokenizer.apply_chat_template(
    msg, tokenize=False, chat_template=my_template)

assert origin_str == my_str

new_msg = msg + [{"role": "assistant", "content": "Hello, how are you?"}]

origin_str = tokenizer.apply_chat_template(
    new_msg, tokenize=False, add_generation_prompt=True)

my_str = tokenizer.apply_chat_template(
    new_msg, tokenize=False, chat_template=my_template, add_generation_prompt=True)

assert origin_str == my_str

processed = tokenizer.apply_chat_template(
    new_msg, return_dict=True, return_assistant_tokens_mask=True, chat_template=my_template)

assistant_masks = torch.tensor(processed["assistant_masks"]).to(torch.bool)
input_ids = torch.tensor(processed["input_ids"])

print(processed["assistant_masks"])

gen_ids = input_ids[assistant_masks]

decoded_str = tokenizer.decode(gen_ids)
print(decoded_str)


print("test_custom_template passed")

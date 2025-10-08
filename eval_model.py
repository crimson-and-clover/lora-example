from pathlib import Path

import torch
from evaluate import load
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast

from datasets import load_from_disk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16

datasets_path = Path("./datasets/tulu-math")

raw_ds = load_from_disk(datasets_path)

train_size = 10000
test_size = 100
eval_size = 10

raw_ds = raw_ds.train_test_split(
    train_size=train_size, test_size=test_size + eval_size, seed=42)

train_ds = raw_ds["train"]
test_ds = raw_ds["test"]
eval_ds = test_ds.select(range(eval_size))
test_ds = test_ds.select(range(eval_size, eval_size + test_size))


# model_name = "huggingface/meta-llama/Llama-3.2-3B-Instruct"
model_name = "huggingface/Qwen/Qwen3-4B-Instruct-2507"

tokenizer: PreTrainedTokenizerFast = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=dtype, device_map=device)
base_model = base_model.eval()

base_model = base_model.to(device)

model = base_model

batch_size = 4

gt_msgs = []
pred_msgs = []

for start in tqdm(range(0, batch_size, batch_size)):
    end = min(start + batch_size, batch_size)
    batch_msgs = [eval_ds[i]["messages"][:-1] for i in range(start, end)]
    batch_texts = [
        tokenizer.apply_chat_template(
            msg, tokenize=False, add_generation_prompt=True)
        for msg in batch_msgs
    ]

    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    inputs = tokenizer.apply_chat_template(batch_msgs, tokenize=True,
                                           return_dict=True,
                                           add_generation_prompt=True,
                                           padding=True,
                                           return_tensors="pt")
    inputs = inputs.to(device)

    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, dtype=dtype):
            outputs = model.generate(
                **inputs,
                max_new_tokens=4096,
            )

    gen_ids = outputs[:, inputs.input_ids.shape[1]:]
    gen_strs = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    gt_msgs.extend([eval_ds[i]["messages"][-1]["content"]
                   for i in range(start, end)])
    pred_msgs.extend(gen_strs)


metric_bleu = load("bleu")
metric_rouge = load("rouge")
metric_bertscore = load("bertscore")

bleu_result = metric_bleu.compute(
    predictions=pred_msgs,
    references=gt_msgs,
)

print(bleu_result)

rouge_result = metric_rouge.compute(
    predictions=pred_msgs,
    references=gt_msgs,
    use_stemmer=True
)

print(rouge_result)

bertscore_result = metric_bertscore.compute(
    predictions=pred_msgs,
    references=gt_msgs,
    lang="en",
    model_type="roberta-large",
    device=device
)

print(bertscore_result)

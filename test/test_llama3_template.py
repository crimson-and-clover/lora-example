from pathlib import Path

import torch
from transformers import AutoTokenizer

if __name__ == "__main__":
    model_name = "huggingface/meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I am fine, thank you!"},
        {"role": "user", "content": "What is your name?"},
        {"role": "assistant", "content": "My name is Llama."},
    ]

    my_template = ""

    with open("template/llama-3.2.jinja2", "r", encoding="utf-8") as f:
        my_template = f.read()

    origin_str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)

    my_str = tokenizer.apply_chat_template(
        messages, tokenize=False, chat_template=my_template, add_generation_prompt=True)

    assert origin_str == my_str

    origin_str = tokenizer.apply_chat_template(messages, tokenize=False)

    my_str = tokenizer.apply_chat_template(
        messages, tokenize=False, chat_template=my_template)

    assert origin_str == my_str

    processed = tokenizer.apply_chat_template(
        messages, return_dict=True, return_assistant_tokens_mask=True, chat_template=my_template)

    assistant_masks = torch.tensor(processed["assistant_masks"]).to(torch.bool)
    input_ids = torch.tensor(processed["input_ids"])

    gen_ids = input_ids[assistant_masks]

    decoded_str = tokenizer.decode(gen_ids)

    gt_str = "".join(
        [f"{x['content']}<|eot_id|>" for x in messages if x["role"] == "assistant"])

    assert decoded_str == gt_str

    print(f"{Path(__file__).stem} passed")

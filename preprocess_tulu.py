from pathlib import Path

from datasets import load_dataset

if __name__ == "__main__":
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

    raw_ds.save_to_disk(Path("./datasets/tulu-math"))

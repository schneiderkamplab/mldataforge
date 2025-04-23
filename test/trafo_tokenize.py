from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B-Instruct")

def process(sample):
    sample["text"] = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
    sample["input_ids"] = tokenizer(sample["text"], return_tensors="np")["input_ids"][0]
    return sample

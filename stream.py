# type: ignore
from transformers import AutoTokenizer
import torch
import random
from llm import LLM
import glob


@torch.inference_mode()
def stream(model_id: str, query: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model_config: dict = {
        "bunch_size": 8,
        "hidden_size": 512,
        "num_layers": 32,
        "head_size": 64,
    }
    model = LLM(tokenizer.vocab_size, **model_config).to(
        dtype=torch.bfloat16, device="cuda"
    )
    ckpts = glob.glob("models/llm*.pt")
    ckpt = sorted(ckpts, key=lambda x: int(x.split("-")[-1].split(".")[0]))[-1]

    model.load_state_dict(torch.load(ckpt))
    device = torch.device("cuda")
    input_ids = tokenizer.encode(query, return_tensors="pt").to(device)
    past_key_values = None
    for _ in range(512):
        outputs = model(input_ids, past_key_values=past_key_values)
        pred_logits = outputs["logits"]
        past_key_values = outputs["past_key_values"]
        pred_logits = pred_logits[:, -1:].view(-1)
        topk = 10
        probs, token_ids = pred_logits.topk(topk, dim=-1, largest=True, sorted=True)
        probs = (probs * topk).softmax(dim=-1)[probs > 0.01]
        if probs.numel() > 3:
            sample_id = torch.multinomial(probs, 3)
            sample_id = sample_id[random.randint(0, sample_id.numel() - 1)]
        else:
            sample_id = 0
        sample_token_id = token_ids[sample_id]
        pred_strings = tokenizer.decode(sample_token_id)
        if sample_token_id == tokenizer.eos_token_id:
            return pred_strings
        yield pred_strings
        input_ids = torch.as_tensor(sample_token_id).view(-1, 1)


for t in stream(
    "meta-llama/Llama-2-7b-chat-hf",
    "你好",
):
    print(t + " ", end="", flush=True)

print("done")

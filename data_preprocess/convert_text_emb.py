import os
import torch
import numpy as np
from datasets import load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clip_dir = "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
text_encoder = CLIPTextModel.from_pretrained(clip_dir).to(device).eval()
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)

ddd = load_from_disk("/root/mems_dataset").select(range(1000))

@torch.no_grad()
def embed_batch(batch):
    texts = batch["caption"] if "caption" in batch else batch["text"]
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    feats = text_encoder(**inputs).last_hidden_state.mean(dim=1)          # [B,512]
    batch["caption_embedding"] = feats.detach().cpu().numpy().astype(np.float32)
    return batch

dx = ddd.map(embed_batch, batched=True, batch_size=64)
dx.save_to_disk("/root/mems_dataset_embed")
print("Saved:", len(dx))
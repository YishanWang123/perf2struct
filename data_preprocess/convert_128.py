import numpy as np
from datasets import load_from_disk, Dataset
from PIL import Image

SRC = "/root/mems_dataset_embed"      # 里面已经有 caption_embedding
DST = "/root/mems_dataset_embed_256"  # 新数据集

ds = load_from_disk(SRC)

def resize_batch(batch):
    imgs = []
    for img in batch["image"]:
        img = img.convert("RGB").resize((256,256), resample=Image.BICUBIC)
        imgs.append(img)
    batch["image"] = imgs
    return batch

ds2 = ds.map(resize_batch, batched=True, batch_size=64)
ds2.save_to_disk(DST)
print("saved:", len(ds2))
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CFMEmbedDataset(Dataset):
    def __init__(self, dataset, img_size=128, text_key="caption", embed_key="caption_embedding"):
        self.dataset = dataset
        self.text_key = text_key
        self.embed_key = embed_key
        self.transform = transforms.Compose([
            # transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]

        img = ex["image"].convert("RGB")
        caption = ex[self.text_key] if self.text_key in ex else ex.get("text", "")

        emb_raw = ex[self.embed_key]
        # HF datasets often returns list for Array features
        if isinstance(emb_raw, np.ndarray):
            emb = torch.from_numpy(emb_raw).float()
        else:
            emb = torch.tensor(emb_raw, dtype=torch.float32)

        return {
            "image": self.transform(img),
            "caption": caption,
            "caption_embedding": emb,  # [512] float32
        }
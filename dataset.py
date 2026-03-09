import os
import json
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
            transforms.Resize((img_size, img_size)),
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

class CFMLatentDataset(Dataset):
    """仅加载 latent_image 和 caption_embedding，避免不必要的 image 解码（训练不需要）"""
    def __init__(self, dataset, img_size=512, text_key="caption", embed_key="caption_embedding", load_image=False):
        self.dataset = dataset
        self.text_key = text_key
        self.embed_key = embed_key
        self.load_image = load_image  # 训练 latent 时通常不需要原图
        self.transform = transforms.Compose([transforms.ToTensor()]) if load_image else None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        latent_image = ex["latent_image"]
        caption = ex[self.text_key] if self.text_key in ex else ex.get("text", "")

        emb_raw = ex[self.embed_key]
        if isinstance(latent_image, np.ndarray):
            latent_image = torch.from_numpy(latent_image).float()
        else:
            latent_image = torch.tensor(latent_image, dtype=torch.float32)
        if isinstance(emb_raw, np.ndarray):
            emb = torch.from_numpy(emb_raw).float()
        else:
            emb = torch.tensor(emb_raw, dtype=torch.float32)

        out = {"latent_image": latent_image, "caption": caption, "caption_embedding": emb}
        if self.load_image:
            out["image"] = self.transform(ex["image"])
        return out


class CFMFeatureDataset(Dataset):
    """用于 structured feature，接收 list/dict 或 HuggingFace dataset"""
    def __init__(self, dataset, img_size=64, feature_key="feature"):
        self.dataset = dataset
        self.feature_key = feature_key
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ex = self.dataset[idx]
        img = ex["image"].convert("RGB")
        feat_raw = ex[self.feature_key]
        feat = torch.from_numpy(feat_raw).float() if isinstance(feat_raw, np.ndarray) else torch.tensor(feat_raw, dtype=torch.float32)
        return {"image": self.transform(img), "feature": feat}


class CFMFeatureDatasetFromDir(Dataset):
    """从 pipeline_feature 生成的 JSON+npy 格式加载，不依赖 load_from_disk"""
    def __init__(self, root_dir, img_size=64):
        self.root_dir = root_dir
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        with open(os.path.join(root_dir, "metadata.json"), encoding="utf-8") as f:
            self.metadata = json.load(f)
        self.features = np.load(os.path.join(root_dir, "features.npy"))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        from PIL import Image
        path = self.metadata[idx]["image_path"]
        img = Image.open(path).convert("RGB")
        feat = torch.from_numpy(self.features[idx]).float()
        return {"image": self.transform(img), "feature": feat}
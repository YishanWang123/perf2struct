import os
import numpy as np
import torch
from datasets import load_dataset, Image
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from PIL import Image as PILImage
from torchvision import transforms

# =========================
# 路径配置
# =========================
jsonl_path = "/root/memsdata/train/jsonl/labels_n1.jsonl"
png_dir = "/root/memsdata/train/png"
save_dir = "/root/mems_dataset_with_latent_512"

clip_dir = "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
vae_id = "stabilityai/sdxl-vae"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. 构建原始 dataset
# =========================
ds = load_dataset("json", data_files=jsonl_path, split="train")

def add_path(ex):
    ex["image_path"] = os.path.join(png_dir, ex["image_file_name"])
    return ex

ds = ds.map(add_path)
ds = ds.cast_column("image_path", Image())
ds = ds.rename_column("image_path", "image")
ds = ds.rename_column("text_context", "caption")

print("raw keys:", ds[0].keys())

# =========================
# 2. 加载文本编码器
# =========================
tokenizer = CLIPTokenizer.from_pretrained(clip_dir)
text_encoder = CLIPTextModel.from_pretrained(clip_dir).to(device).eval()

@torch.no_grad()
def embed_batch(batch):
    texts = batch["caption"]
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    feats = text_encoder(**inputs).last_hidden_state.mean(dim=1)  # [B, 512]
    batch["caption_embedding"] = feats.cpu().numpy().astype(np.float32)
    return batch

ds = ds.map(embed_batch, batched=True, batch_size=64)

# =========================
# 3. 加载 VAE
# =========================
vae = AutoencoderKL.from_pretrained(vae_id).to(device)
vae.requires_grad_(False)
vae.eval()

image_transform = transforms.Compose([
    transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
])

# 对于 SD / SDXL VAE，输入通常要缩放到 [-1, 1]
def preprocess_image(pil_img):
    img = pil_img.convert("RGB")
    img = image_transform(img)            # [3, 256, 256], in [0,1]
    img = img * 2.0 - 1.0                 # -> [-1,1]
    return img

@torch.no_grad()
def latent_batch(batch):
    imgs = [preprocess_image(img) for img in batch["image"]]
    imgs = torch.stack(imgs, dim=0).to(device)   # [B, 3, 256, 256]

    posterior = vae.encode(imgs).latent_dist
    latents = posterior.sample()

    # 关键：SD/SDXL 通常要乘 scaling_factor
    latents = latents * vae.config.scaling_factor

    batch["latent_image"] = latents.cpu().numpy().astype(np.float32)
    return batch

ds = ds.map(latent_batch, batched=True, batch_size=16)

# =========================
# 4. 保存
# =========================
ds.save_to_disk(save_dir)
print(f"saved to: {save_dir}")
print(ds[0].keys())
print("caption_embedding shape:", np.array(ds[0]["caption_embedding"]).shape)
print("latent_image shape:", np.array(ds[0]["latent_image"]).shape)
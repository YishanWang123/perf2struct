import os
import random
import tempfile
import torch
import numpy as np
from torch.utils.data import DataLoader

# 🔹 随机种子，保证可复现
SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # 复现时关闭，追求速度可改 True
set_seed()
from datasets import load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, utils
import torchvision.transforms.functional as F
from diffusers import AutoencoderKL
import wandb
import torchvision.utils as vutils

from model import UNetModelWithTextEmbedding
from dataset import CFMEmbedDataset, CFMLatentDataset

def tensor_to_image(tensor):
    return F.to_pil_image(tensor.clamp(0, 1))

# 🔹 加载数据
dx = load_from_disk("/root/mems_dataset_with_latent_512")

print(f"Dataset size: {len(dx)} samples")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = CLIPTextModel.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
)
vae = AutoencoderKL.from_pretrained(
    "/root/.cache/huggingface/hub/models--stabilityai--sdxl-vae/snapshots/6f5909a7e596173e25d4e97b07fd19cdf9611c76"
).to(device)
vae.requires_grad_(False)
vae.eval()

train_ds = CFMLatentDataset(dx, load_image=False)  # 不加载原图，显著加速数据 IO
print(f"Number of training samples: {len(train_ds)}")
# num_workers>0 并行加载数据，避免 GPU 空转；pin_memory 加速 CPU->GPU 传输
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)

model = UNetModelWithTextEmbedding(
    dim=(4, 32, 32), num_channels=64, num_res_blocks=1,
    embedding_dim=512, dropout=0.05, num_heads=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# 混合精度训练：torchcfm UNet 与 AMP 不兼容（Input Half vs bias float 报错），暂禁用
use_amp = False  # torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda") if use_amp else None

# 采样间隔：每 2 个 epoch 采样会非常耗时（750 次 VAE decode），改为每 50 个
sample_every_n_epochs = 25

# 🔹 采样时使用的多个 prompt，在此处填写不同 prompt 即可
SAMPLE_PROMPTS = [
    "drive_freq:39800Hz,split:less_than_0.2%,parasitic:berween_5000_and_10000Hz,x_stiffness:7100N/m,nonlinearity:moderate",
    "drive_freq:47900Hz,split:between_3.0%_and_5.0%,parasitic:berween_5000_and_10000Hz,x_stiffness:8300N/m,nonlinearity:moderate",
    "drive_freq:48400Hz,split:between_1.0%_and_3.0%,parasitic:berween_5000_and_10000Hz,x_stiffness:9300N/m,nonlinearity:moderate",
    "drive_freq:37700Hz,split:between_3.0%_and_5.0%,parasitic:berween_5000_and_10000Hz,x_stiffness:5300N/m,nonlinearity:relatively_high",
]

# 🔹 开关：是否启用 W&B
use_wandb = True  # 调试时改成 False

if use_wandb:
    # 🔹 初始化 W&B
    wandb.init(project="cfm-image-generation", name="latent-v1",
            config={"epochs": n_epochs, "batch_size": 64})
else:
    # 提供一个空的 mock wandb，避免代码报错
    class DummyWandb:
        def log(self, *args, **kwargs): pass
        class Image: 
            def __init__(self, *args, **kwargs): pass
        class Video: 
            def __init__(self, *args, **kwargs): pass

    wandb = DummyWandb()

# 🔹 Euler 采样函数
def euler_method(model, text_embedding, t_steps, dt, noise):
    y = noise
    y_values = [y]
    with torch.no_grad():
        for t in t_steps[1:]:
            dy = model(t.to(device), y, text_embeddings=text_embedding)
            y = y + dy * dt
            y_values.append(y)
        y_values = torch.stack(y_values)  # (n_steps, n_samples, 4, 32, 32)
        # VAE 只接受 4D [B,C,H,W]；SDXL VAE 解码 32x32 latent 输出 256x256，勿写死尺寸
        n_steps, n_s, c, h, w = y_values.shape
        decoded = vae.decode(y_values.view(-1, c, h, w)).sample  # (n_steps*n_s, 3, H_out, W_out)
        _, c_out, h_out, w_out = decoded.shape
        y_values = decoded.view(n_steps, n_s, c_out, h_out, w_out)
    return y_values

@torch.no_grad()
def encode_prompt(prompt: str):
    inputs = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(device)
    return text_encoder(**inputs).last_hidden_state.mean(dim=1)  # [1,512]

# Euler 采样步数：10 对于 CFM 通常够用，质量要求高可改为 20~50
EULER_STEPS = 20

def sample_and_log(epoch, text_embeddings, n_samples=20, save_path="sample.gif", tag="train"):
    noise = torch.randn((n_samples, 4, 32, 32), device=device)
    t_steps = torch.linspace(0, 1, EULER_STEPS, device=device)
    dt = t_steps[1] - t_steps[0]

    # 🔹 always eval during sampling
    model.eval()
    with torch.no_grad():
        results = euler_method(model, text_embeddings, t_steps, dt, noise)
    model.train()

    # 取最后一步
    final_batch = results[-1]   # (n_samples, 3, H, W)

    # 拼成网格并 log 到 wandb（不存本地）
    grid = vutils.make_grid(final_batch, nrow=5, normalize=True, value_range=(0, 1))
    grid_img = tensor_to_image(grid.cpu())
    wandb.log({
        f"sample_grid_{tag}": wandb.Image(grid_img, caption=f"{tag}_epoch_{epoch}_{len(t_steps)}")
    })

    # GIF 写入临时文件后 log 到 wandb，随后删除（避免依赖 moviepy）
    frames = [tensor_to_image(results[idx, 0].cpu()) for idx in range(0, results.shape[0], max(1, results.shape[0] // 5))]
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        tmp_path = f.name
    try:
        frames[0].save(tmp_path, save_all=True, append_images=frames[1:], duration=300, loop=0)
        wandb.log({
            f"sample_gif_{len(t_steps)}_{tag}": wandb.Video(tmp_path, fps=4, format="gif")
        })
    finally:
        os.remove(tmp_path)

# 🔹 训练循环
for epoch in tqdm(range(n_epochs)):
    losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        x1 = batch["latent_image"].to(device, non_blocking=True)
        text_embeddings = batch["caption_embedding"].to(device, non_blocking=True)

        x0 = torch.randn_like(x1, device=device)
        t = torch.rand(x0.shape[0], 1, 1, 1, device=device)

        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0
        t = t.squeeze()

        if use_amp:
            with torch.amp.autocast("cuda"):
                vt = model(t, xt, text_embeddings=text_embeddings)
                loss = torch.mean(((vt - ut) ** 2))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            vt = model(t, xt, text_embeddings=text_embeddings)
            loss = torch.mean(((vt - ut) ** 2))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        losses.append(loss.item())

    scheduler.step()
    avg_loss = sum(losses) / len(losses)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    # 降低采样频率：每 sample_every_n_epochs 个 epoch 采样一次（VAE decode 很耗时）
    if (epoch + 1) % sample_every_n_epochs == 0:
        for i, prompt in enumerate(SAMPLE_PROMPTS):
            text_embedding = encode_prompt(prompt)
            tag = f"prompt_{i}"  # wandb 中按 prompt_0, prompt_1, ... 区分
            sample_and_log(epoch, text_embedding, n_samples=1,
                          save_path=f"sample_epoch{epoch}_{tag}.gif", tag=tag)
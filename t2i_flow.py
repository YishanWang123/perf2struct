import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_from_disk
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, utils
import torchvision.transforms.functional as F
import wandb
import torchvision.utils as vutils

from model import UNetModelWithTextEmbedding
from dataset import CFMEmbedDataset

def tensor_to_image(tensor):
    return F.to_pil_image(tensor.clamp(0, 1))

# 🔹 加载数据
dx = load_from_disk("/root/mems_dataset_embed_128")

print(f"Dataset size: {len(dx)} samples")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

text_encoder = CLIPTextModel.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
).to(device)
text_encoder.eval()

tokenizer = CLIPTokenizer.from_pretrained(
    "/root/.cache/huggingface/hub/models--openai--clip-vit-base-patch16/snapshots/57c216476eefef5ab752ec549e440a49ae4ae5f3"
)

train_ds = CFMEmbedDataset(dx)
print(f"Number of training samples: {len(train_ds)}")
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=12, pin_memory=False)

model = UNetModelWithTextEmbedding(
    dim=(3, 128, 128), num_channels=64, num_res_blocks=1,
    embedding_dim=512, dropout=0.05, num_heads=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 1500
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

# 🔹 开关：是否启用 W&B
use_wandb = True  # 调试时改成 False

if use_wandb:
    # 🔹 初始化 W&B
    wandb.init(project="cfm-image-generation", name="flow-v0",
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
    return torch.stack(y_values)

@torch.no_grad()
def encode_prompt(prompt: str):
    inputs = tokenizer([prompt], padding=True, truncation=True, return_tensors="pt").to(device)
    return text_encoder(**inputs).last_hidden_state.mean(dim=1)  # [1,512]

def sample_and_log(epoch, text_embeddings, n_samples=20, save_path="sample.gif", tag="train"):
    noise = torch.randn((n_samples, 3, 128, 128), device=device)
    t_steps = torch.linspace(0, 1,100, device=device)
    dt = t_steps[1] - t_steps[0]

    # 🔹 always eval during sampling
    model.eval()
    with torch.no_grad():
        results = euler_method(model, text_embeddings, t_steps, dt, noise)
    model.train()

    # 取最后一步
    final_batch = results[-1]   # (n_samples, 3, 64, 64)

    # 拼成网格 (nrow=5 -> 4x5 grid)
    grid = vutils.make_grid(final_batch, nrow=5, normalize=True, value_range=(0,1))
    grid_img = tensor_to_image(grid.cpu())
    grid_img.save(f"sample_epoch{epoch}_{tag}_{len(t_steps)}.png")

    wandb.log({
        f"sample_grid_{tag}": wandb.Image(grid_img, caption=f"{tag}_epoch_{epoch}_{len(t_steps)}")
    })

    # 存 GIF
    frames = [tensor_to_image(results[idx, 0].cpu()) for idx in range(0, results.shape[0], 5)]
    frames[0].save(save_path, save_all=True, append_images=frames[1:], duration=300, loop=0)

    # log 到 wandb
    wandb.log({
        f"sample_gif_{len(t_steps)}_{tag}": wandb.Video(save_path, fps=4, format="gif")
    })

# 🔹 训练循环
for epoch in tqdm(range(n_epochs)):
    losses = []
    for batch in train_loader:
        optimizer.zero_grad()
        x1 = batch["image"].to(device)
        text_embeddings = batch["caption_embedding"].to(device)

        x0 = torch.randn_like(x1).to(device)
        t = torch.rand(x0.shape[0], 1, 1, 1).to(device)

        xt = t * x1 + (1 - t) * x0
        ut = x1 - x0
        t = t.squeeze()

        vt = model(t, xt, text_embeddings=text_embeddings)
        loss = torch.mean(((vt - ut) ** 2))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

    avg_loss = sum(losses) / len(losses)
    wandb.log({"train_loss": avg_loss, "epoch": epoch})

    if (epoch + 1) % 300 == 0:
        print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")
        # sample_and_log(epoch, text_embeddings)

        input_prompt = "drive_freq:32600Hz,split:between_3.0%_and_5.0%,parasitic:less_than_5000Hz,x_stiffness:8000N/m,nonlinearity:low"
        text_embedding = encode_prompt(input_prompt)
        sample_and_log(epoch, text_embedding, n_samples=1, save_path=f"sample_prompt_epoch{epoch}.gif", tag="unseen")
# import time
# from collections import defaultdict

# def sync():
#     if device.type == "cuda":
#         torch.cuda.synchronize()

# LOG_EVERY_STEPS = 20   # 打印频率
# PROFILE_STEPS = 100    # 只统计前 N 个 step，避免打印太多

# timers = defaultdict(float)
# counts = 0

# for epoch in tqdm(range(n_epochs)):
#     losses = []
#     # 手动拿 iterator，方便测 “取 batch” 时间
#     it = iter(train_loader)

#     step = 0
#     while True:
#         # ===== 1) DataLoader time =====
#         t0 = time.time()
#         try:
#             batch = next(it)
#         except StopIteration:
#             break
#         timers["dataloader"] += time.time() - t0

#         # ===== 2) H2D copy time =====
#         t0 = time.time()
#         x1 = batch["image"].to(device, non_blocking=True)
#         text_embeddings = batch["caption_embedding"].to(device, non_blocking=True)
#         sync()
#         timers["to_device"] += time.time() - t0

#         optimizer.zero_grad(set_to_none=True)

#         # ===== 3) prep time =====
#         t0 = time.time()
#         x0 = torch.randn_like(x1)
#         t = torch.rand(x0.shape[0], 1, 1, 1, device=device)
#         xt = t * x1 + (1 - t) * x0
#         ut = x1 - x0
#         t_in = t.squeeze()
#         sync()
#         timers["prep"] += time.time() - t0

#         # ===== 4) forward time =====
#         t0 = time.time()
#         vt = model(t_in, xt, text_embeddings=text_embeddings)
#         loss = ((vt - ut) ** 2).mean()
#         sync()
#         timers["forward"] += time.time() - t0

#         # ===== 5) backward time =====
#         t0 = time.time()
#         loss.backward()
#         sync()
#         timers["backward"] += time.time() - t0

#         # ===== 6) step time =====
#         t0 = time.time()
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()
#         sync()
#         timers["optim"] += time.time() - t0

#         losses.append(loss.item())
#         counts += 1
#         step += 1

#         # 打印
#         if counts % LOG_EVERY_STEPS == 0:
#             denom = LOG_EVERY_STEPS
#             print(
#                 f"[profile last {denom} steps] "
#                 f"dl={timers['dataloader']/denom:.3f}s  "
#                 f"h2d={timers['to_device']/denom:.3f}s  "
#                 f"prep={timers['prep']/denom:.3f}s  "
#                 f"fwd={timers['forward']/denom:.3f}s  "
#                 f"bwd={timers['backward']/denom:.3f}s  "
#                 f"opt={timers['optim']/denom:.3f}s  "
#                 f"loss={np.mean(losses[-denom:]):.4f}"
#             )
#             for k in list(timers.keys()):
#                 timers[k] = 0.0

#         # 只 profile 前 PROFILE_STEPS 个 step
#         if counts >= PROFILE_STEPS:
#             break

#     avg_loss = float(np.mean(losses)) if losses else 0.0
#     if use_wandb:
#         wandb.log({"train_loss": avg_loss, "epoch": epoch})
#     scheduler.step()

#     # 只 profile 前 PROFILE_STEPS 后就正常训练
#     if counts >= PROFILE_STEPS:
#         print("Profiling done. Continue training without profiling prints.")
#         break
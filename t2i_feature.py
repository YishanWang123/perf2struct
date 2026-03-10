"""
CFM 训练脚本：pixel 64x64，使用 structured feature (one-hot + 归一化) + MLP + FiLM  conditioning
数据需先运行: python data_preprocess/pipeline_feature.py
"""
import os
import random
import json
import tempfile
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
import torchvision.transforms.functional as F
import wandb
import torchvision.utils as vutils

from model import UNetModelWithFiLM
from dataset import CFMFeatureDatasetFromDir

# 路径（JSON+npy 格式，由 pipeline_feature.py 生成，避免 datasets 版本兼容问题）
DATASET_PATH = "/root/mems_dataset_feature_64"

# 随机种子
SEED = 42
CKPT_DIR = "./checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_image(tensor):
    return F.to_pil_image(tensor.clamp(0, 1))


# 加载数据（从 JSON+npy，不依赖 datasets.load_from_disk）
with open(os.path.join(DATASET_PATH, "feature_config.json")) as f:
    FEATURE_CONFIG = json.load(f)
FEATURE_DIM = FEATURE_CONFIG["feature_dim"]

train_ds = CFMFeatureDatasetFromDir(DATASET_PATH, img_size=64)
print(f"Dataset size: {len(train_ds)} samples, feature dim: {FEATURE_DIM}")
train_loader = DataLoader(
    train_ds, batch_size=64, shuffle=True,
    num_workers=4, pin_memory=True, persistent_workers=True
)

model = UNetModelWithFiLM(
    dim=(3, 64, 64), num_channels=64, num_res_blocks=1,
    feature_dim=FEATURE_DIM, mlp_hidden=128, dropout=0.05, num_heads=4
).to(device)

optimizer = torch.optim.AdamW(model.parameters())
n_epochs = 900
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

EULER_STEPS = 20
sample_every_n_epochs = 30

# 采样用 feature：填写 dataset 中的索引
# 从数据集中取这些 index 的 feature 做采样
# 若需从原始参数构造 feature，可: from data_preprocess.pipeline_feature import features_to_vector
SAMPLE_FEATURE_INDICES = [0, 100, 500, 800]

use_wandb = False

if use_wandb:
    wandb.init(project="cfm-image-generation", name="feature-film-64-ck",
               config={"epochs": n_epochs, "batch_size": 64, "feature_dim": FEATURE_DIM})
else:
    class DummyWandb:
        def log(self, *args, **kwargs): pass

        class Image:
            def __init__(self, *args, **kwargs): pass

        class Video:
            def __init__(self, *args, **kwargs): pass

    wandb = DummyWandb()


def euler_method(model, features, t_steps, dt, noise):
    y = noise
    y_values = [y]
    with torch.no_grad():
        for t in t_steps[1:]:
            dy = model(t.to(device), y, features=features)
            y = y + dy * dt
            y_values.append(y)
    return torch.stack(y_values)


def sample_and_log(epoch, features_batch, n_samples_per_feat=1, tag="sample"):
    """features_batch: [N, feat_dim] 每个 feature 生成 n_samples_per_feat 张图"""
    n_feats = features_batch.shape[0]
    n_total = n_feats * n_samples_per_feat
    features_batch = features_batch.to(device)
    features_repeat = features_batch.repeat_interleave(n_samples_per_feat, dim=0)  # [n_total, feat_dim]

    noise = torch.randn((n_total, 3, 64, 64), device=device)
    t_steps = torch.linspace(0, 1, EULER_STEPS, device=device)
    dt = t_steps[1] - t_steps[0]

    model.eval()
    with torch.no_grad():
        results = euler_method(model, features_repeat, t_steps, dt, noise)
    model.train()

    final_batch = results[-1]
    grid = vutils.make_grid(final_batch, nrow=max(1, n_total // 2), normalize=True, value_range=(0, 1))
    grid_img = tensor_to_image(grid.cpu())
    wandb.log({
        f"sample_grid_{tag}": wandb.Image(grid_img, caption=f"{tag}_epoch_{epoch}")
    })

    frames = [tensor_to_image(results[idx, 0].cpu()) for idx in range(0, results.shape[0], max(1, results.shape[0] // 5))]
    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as f:
        tmp_path = f.name
    try:
        frames[0].save(tmp_path, save_all=True, append_images=frames[1:], duration=300, loop=0)
        wandb.log({f"sample_gif_{tag}": wandb.Video(tmp_path, fps=4, format="gif")})
    finally:
        os.remove(tmp_path)


# 训练循环（仅直接运行时执行，import 时跳过）
if __name__ == "__main__":
    for epoch in tqdm(range(n_epochs)):
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            x1 = batch["image"].to(device, non_blocking=True)
            features = batch["feature"].to(device, non_blocking=True)

            x0 = torch.randn_like(x1, device=device)
            t = torch.rand(x1.shape[0], 1, 1, 1, device=device)

            xt = t * x1 + (1 - t) * x0
            ut = x1 - x0
            t = t.squeeze()

            vt = model(t, xt, features=features)
            loss = torch.mean(((vt - ut) ** 2))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())

        scheduler.step()
        avg_loss = sum(losses) / len(losses)
        wandb.log({"train_loss": avg_loss, "epoch": epoch})

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {avg_loss:.4f}")

        if (epoch + 1) % sample_every_n_epochs == 0:
            indices = [i for i in SAMPLE_FEATURE_INDICES if i < len(train_ds)]
            if indices:
                feats = train_ds.features[indices]
                feats_t = torch.from_numpy(feats).float()
                sample_and_log(epoch, feats_t, n_samples_per_feat=1, tag="unseen")

                ckpt_path = os.path.join(CKPT_DIR, f"epoch_{epoch+1:04d}.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "loss": avg_loss,
                    "feature_dim": FEATURE_DIM,
                }, ckpt_path)

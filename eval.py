import torch
from dataset import CFMFeatureDatasetFromDir
from model.model import UNetModelWithFiLM, UNetModelWithFiLM1714D
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from scripts.t2i_feature import euler_method
import os
DATASET_PATH = "/root/dataset_2/features_64_1"
FEATURE_DIM = 17
SEED = 42
EULER_STEPS = 10
# SAMPLE_FEATURE_INDICES = [0, 100, 500, 800]
# # SAMPLE_FEATURE_INDICES = list(range(200))

# # 200to400
SAMPLE_FEATURE_INDICES = [i for i in range(200)]
print(len(SAMPLE_FEATURE_INDICES))

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
train_ds = CFMFeatureDatasetFromDir(DATASET_PATH, img_size=64)
model = UNetModelWithFiLM1714D(
    dim=(3, 64, 64), num_channels=64, num_res_blocks=1,
    feature_dim=FEATURE_DIM, mlp_hidden=128, dropout=0.05, num_heads=4
).to(device)
ckpt = torch.load("/root/checkpoints/epoch_0840.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])

indices = [i for i in SAMPLE_FEATURE_INDICES if i < len(train_ds)]
if indices:
    feats = train_ds.features[indices]
    # print("indices:", indices)
    # print("feats:", feats)
    # import pdb; pdb.set_trace()
    feats_t = torch.from_numpy(feats).float()
    n_feats = feats_t.shape[0]
    n_total = n_feats * 1
    features_batch = feats_t.to(device)
    features_repeat = features_batch.repeat_interleave(1, dim=0)  # [n_total, feat_dim]

    noise = torch.randn((n_total, 3, 64, 64), device=device)
    t_steps = torch.linspace(0, 1, EULER_STEPS, device=device)
    dt = t_steps[1] - t_steps[0]

    model.eval()
    with torch.no_grad():
        results = euler_method(model, features_batch, t_steps, dt, noise)

    final_images = results[-1].detach().cpu()

    # 保存大图
    grid = vutils.make_grid(
        final_images,
        nrow=10,
        normalize=True,
        value_range=(0, 1)
    )

    plt.figure(figsize=(20, 20))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.savefig("grid_137.png", bbox_inches="tight", pad_inches=0)
    plt.show()

    # 保存单张图
    save_dir = "./generated_200"
    os.makedirs(save_dir, exist_ok=True)

    for idx, img in zip(indices, final_images):
        save_path = os.path.join(save_dir, f"idx_{idx:03d}.png")
        vutils.save_image(img, save_path, normalize=True, value_range=(0, 1))

    print(f"已保存 {len(final_images)} 张图片到: {save_dir}")
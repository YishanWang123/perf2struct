import math
import torch.nn as nn
from torchcfm.models.unet import UNetModel
import torch

def timestep_embedding(timesteps, dim, max_period=10000):
    """Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element. These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
        / half
    )
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

class UNetModelWithTextEmbedding(UNetModel):
    def __init__(self, dim, num_channels, num_res_blocks, embedding_dim, *args, **kwargs):
        super().__init__(dim, num_channels, num_res_blocks, *args, **kwargs)

        self.embedding_layer = nn.Linear(embedding_dim, num_channels*4)
        self.fc = nn.Linear(num_channels*8, num_channels*4)

    def forward(self, t, x, text_embeddings=None):
        """Apply the model to an input batch, incorporating text embeddings."""
        timesteps = t

        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        hs = []
        emb = self.time_embed(
            timestep_embedding(timesteps, self.model_channels)
        )

        if text_embeddings is not None:
            text_embedded = self.embedding_layer(text_embeddings)
            emb = torch.cat([emb, text_embedded], dim=1) # 128*2
            emb = self.fc(emb)

        h = x.type(self.dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class UNetModelWithFiLM(UNetModel):
    """用 MLP 将 feature 转为 embedding，再以 FiLM (scale+shift) 方式注入 UNet。
    支持双分支：x_stiffness（第 x_stiffness_idx 维）单独一个 MLP，其余特征一个 MLP，再 concat 后做 FiLM。
    """
    def __init__(self, dim, num_channels, num_res_blocks, feature_dim, mlp_hidden=128,
                 x_stiffness_idx=11, *args, **kwargs):
        super().__init__(dim, num_channels, num_res_blocks, *args, **kwargs)
        emb_dim = num_channels * 4
        self.feature_dim = feature_dim
        self.x_stiffness_idx = x_stiffness_idx  # 第 12 维（0-based 为 11）
        other_dim = feature_dim - 1  # 除 x_stiffness 外的维度数
        # 其余特征：other_dim -> emb_dim
        self.mlp_other = nn.Sequential(
            nn.Linear(other_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim),
        )
        # x_stiffness 单独一支：1 -> emb_dim
        self.mlp_stiffness = nn.Sequential(
            nn.Linear(1, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim),
        )
        self.mlp_fusion = nn.Sequential(
            nn.Linear(emb_dim*2, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim*2),
        )

    def _split_features(self, features):
        """将 [B, feature_dim] 拆成其余特征 [B, feature_dim-1] 与 x_stiffness [B, 1]。"""
        i = self.x_stiffness_idx
        other = torch.cat([features[:, :i], features[:, i + 1:]], dim=1)
        stiffness = features[:, i : i + 1]
        return other, stiffness

    def forward(self, t, x, features=None):
        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if features is not None:
            features_other, features_stiffness = self._split_features(features)
            film_other = self.mlp_other(features_other)      # [B, emb_dim]
            film_stiffness = self.mlp_stiffness(features_stiffness)  # [B, emb_dim]
            film_pre = torch.cat([film_other, film_stiffness], dim=1)  # [B, emb_dim*2]
            film = self.mlp_fusion(film_pre)
            gamma, beta = film.chunk(2, dim=1)
            emb = emb * (1 + gamma) + beta  # FiLM

        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)


class UNetModelWithFiLM1714D(UNetModel):
    """用 MLP 将 feature 转为 embedding，再以 FiLM (scale+shift) 方式注入 UNet"""
    def __init__(self, dim, num_channels, num_res_blocks, feature_dim, mlp_hidden=128, *args, **kwargs):
        super().__init__(dim, num_channels, num_res_blocks, *args, **kwargs)
        emb_dim = num_channels * 4
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, emb_dim * 2),  # gamma, beta
        )

    def forward(self, t, x, features=None):
        timesteps = t
        while timesteps.dim() > 1:
            timesteps = timesteps[:, 0]
        if timesteps.dim() == 0:
            timesteps = timesteps.repeat(x.shape[0])

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if features is not None:
            film = self.mlp(features)  # [B, emb_dim*2]
            gamma, beta = film.chunk(2, dim=1)
            emb = emb * (1 + gamma) + beta  # FiLM

        h = x.type(self.dtype)
        hs = []
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        return self.out(h)
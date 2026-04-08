"""
从 structured_output.jsonl + png 构建 64x64 像素数据集，feature 为 one-hot + 归一化连续值
使用 JSON + npy 格式保存，避免 HuggingFace datasets 版本兼容问题 (load_from_disk)
"""
import os
import json
import shutil
import numpy as np
from PIL import Image as PILImage

# =========================
# 路径配置
# =========================
JSONL_PATH = "/root/dataset_2/structured_output1.jsonl"
STATS_PATH = "/root/dataset_2/stats1.json"
PNG_DIR = "/root/dataset_2/png"
SAVE_DIR = "/root/dataset_2/features_64_1"

IMG_SIZE = 64

# 用于构建 feature 向量的 key
FEATURE_KEYS = {
    "onehot": ["split_type_id", "parasitic_type_id", "nonlinearity_id"],
    "continuous": [
        "drive_freq_minmax", "x_stiffness_minmax",
        "split_center_minmax", "parasitic_center_minmax",
        "nonlinearity_ord",
    ],
}
N_SPLIT_TYPE = 3
N_PARASITIC_TYPE = 2
N_NONLINEARITY = 7


def onehot(x: int, n_classes: int) -> np.ndarray:
    v = np.zeros(n_classes, dtype=np.float32)
    v[min(x, n_classes - 1)] = 1.0
    return v


def features_to_vector(feats: dict) -> np.ndarray:
    parts = []
    for key in FEATURE_KEYS["onehot"]:
        val = int(feats.get(key, 0))
        if key == "split_type_id":
            parts.append(onehot(val, N_SPLIT_TYPE))
        elif key == "parasitic_type_id":
            parts.append(onehot(val, N_PARASITIC_TYPE))
        elif key == "nonlinearity_id":
            parts.append(onehot(val, N_NONLINEARITY))
    for key in FEATURE_KEYS["continuous"]:
        val = float(feats.get(key, 0.0))
        parts.append(np.array([np.clip(val, 0.0, 1.0)], dtype=np.float32))
    return np.concatenate(parts)


def load_jsonl(path: str) -> list:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    print("Loading structured_output.jsonl ...")
    rows = load_jsonl(JSONL_PATH)

    image_paths = []
    feature_vecs = []
    raw_texts = []
    filenames = []

    for row in rows:
        fname = row["image_file_name"]
        img_path = os.path.join(PNG_DIR, fname)
        if not os.path.exists(img_path):
            print(f"Skip missing: {img_path}")
            continue
        image_paths.append(img_path)
        feature_vecs.append(features_to_vector(row["features"]))
        raw_texts.append(row.get("raw_text", ""))
        filenames.append(fname)

    feature_vecs = np.stack(feature_vecs, axis=0).astype(np.float32)
    feat_dim = feature_vecs.shape[1]

    print(f"Loaded {len(image_paths)} samples, feature dim={feat_dim}")

    os.makedirs(SAVE_DIR, exist_ok=True)

    # 保存 metadata.json (不依赖 datasets 库)
    metadata = [
        {"image_path": p, "raw_text": t, "image_file_name": n}
        for p, t, n in zip(image_paths, raw_texts, filenames)
    ]
    with open(os.path.join(SAVE_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=0, ensure_ascii=False)

    np.save(os.path.join(SAVE_DIR, "features.npy"), feature_vecs)

    if os.path.exists(STATS_PATH):
        shutil.copy(STATS_PATH, os.path.join(SAVE_DIR, "stats.json"))

    config = {"feature_dim": feat_dim, "feature_keys": FEATURE_KEYS, "img_size": IMG_SIZE}
    with open(os.path.join(SAVE_DIR, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved to: {SAVE_DIR}")
    print(f"feature shape: {feature_vecs[0].shape}")


if __name__ == "__main__":
    main()

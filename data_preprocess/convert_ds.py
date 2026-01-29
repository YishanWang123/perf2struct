import os
from datasets import load_dataset, Image

jsonl_path = "/data/memsdata/train/jsonl/labels_n1.jsonl"
png_dir = "/data/memsdata/train/png"

# 读 jsonl
ds = load_dataset("json", data_files=jsonl_path, split="train")

# 补全图片路径
def add_path(ex):
    ex["image_path"] = os.path.join(png_dir, ex["image_file_name"])
    return ex

ds = ds.map(add_path)

# 把 image_path 列变成真正的 Image 列，读取时会自动用 PIL 打开
ds = ds.cast_column("image_path", Image())

# 如果你想让后续代码里用 batch["image"] 这个键名，可以重命名
ds = ds.rename_column("image_path", "image")

# 你的文本如果叫 text_context，而后续 CFMDataset 可能用 caption/text 字段
# 这里也可以顺手统一一下字段名，避免你去改 CFMDataset
# 例如把 text_context 改成 caption
ds = ds.rename_column("text_context", "caption")

# 存成可被 load_from_disk 读取的格式
ds.save_to_disk("/root/mems_dataset")

print(ds[0].keys())
print(ds[0]["image"], ds[0]["caption"])
# Text-Guided-Image-Generation-CFM

This repo implements a **text-to-image (T2I)** generation pipeline based on **Conditional Flow Matching (CFM)**.

- **Backbone**: **UNet** conditioned on text embeddings  
- **Text encoder**: **CLIP** (`openai/clip-vit-base-patch16`) via `CLIPTokenizer` + `CLIPTextModel`  
- **Logging**: **Weights & Biases (wandb)** (metrics + sample grids)

> Note: CLIP weights/tokenizer are downloaded on first use. Make sure your environment has network access (proxy/mirror if needed).

---

## Pipeline

1. Prepare paired **image + text** data (keep the dataset format below).
2. Run preprocessing scripts under `data_preprocess/` **in order**.
3. Run `t2i_flow.py` to start training.
4. Training logs and samples will be synced to **wandb**.

---

## Environment

Coming soon.

---

## Dataset Format

Keep your dataset in the following structure:

```text
/data/memsdata/
  train/
    jsonl/
      labels_n1.jsonl
    png/
      1_00001.png
      1_00002.png
      ...
```
The JSONL file contains one record per line:
```text
{"image_file_name":"1_00001.png","text_context":"drive_freq:..."}
```

---

## Data Preprocess
Preprocessing scripts are located in:
```text
data_preprocess/
  convert_ds.py
  convert_128.py
  convert_text_emb.py
```

and run command
```text
cd data_preprocess
python convert_ds.py
python convert_text_emb.py
python convert_128.py
```
Output datasets are saved on disk and will be loaded by the training script.
Example output name: mems_dataset_embed_128

---

## Train & Evaluaion
you can modify the interval in scripts to control the frequency of eval during training the model
```text
python t2i_flow.py 
```










# CODEX.md

## Purpose

This file is a working memory and operating note for future Codex or human collaborators.
It is not a paper-style README. It is meant to answer:

- what this repo is for
- where the important code lives
- how to run the core workflow
- what assumptions are currently true
- what should be updated when the project changes

## Project Snapshot

- Repo name: `perf2struct`
- Domain: performance-conditioned structural generation for fork-shaped beam design
- Main idea: generate structural images from target performance indicators, then validate downstream with geometry reconstruction and COMSOL-based evaluation
- Current repo focus: preprocessing, model training/inference, and metric computation

## Short-Term Memory

### Current understanding

- The project studies conditional generation from performance labels to 2D fork-shaped beam structures.
- The current workflow is:
  1. prepare paired data from COMSOL and MATLAB
  2. preprocess labels/features
  3. train the conditional generation model
  4. generate output structures
  5. evaluate generated outputs with downstream simulation-derived metrics
- Dataset files are external and are not stored in this repository.
- COMSOL and MATLAB are part of the real workflow, but they are outside this repo.

### Important directories and files

- `README.md`: high-level project description and basic usage
- `data_preprocess/`: data conversion and feature/text preprocessing scripts
- `scripts/`: training or generation entry scripts
- `model/model.py`: core model implementation
- `dataset.py`: dataset loading logic
- `metrics.py`: metric computation for generated outputs
- `vis.ipynb`: notebook for visualization / experiments
- `requirements.txt`: Python dependencies

### Current likely entrypoints

- Training: `python scripts/t2i_feature.py`
- Metrics: `python metrics.py`
- Preprocess:
  - `python data_preprocess/wo_splt_text.py`
  - `python data_preprocess/pipeline_feature.py`

### Environment assumptions

- Recommended Python version from current README: `python==3.8.20`
- Install dependencies with:

```bash
pip install -r requirements.txt
```

### Repository note

- On 2026-03-22, the git history of `main` was rewritten into a fresh single-author history so GitHub contributors would only show the current owner.
- A local backup branch may still exist: `backup_pre_clean_contributors_20260322`

## How To Use This File

When starting a new coding session, read this file first and use it as the compact source of project context.

This file should be updated whenever one of these changes happens:

- training entry script changes
- preprocessing pipeline changes
- dataset format changes
- model architecture changes
- evaluation protocol changes
- environment/version requirements change
- the repo gets reorganized

## Collaboration Rules For Future Codex Sessions

If you are an AI coding agent working in this repo, prefer the following behavior:

- read `README.md` and this file before making assumptions
- inspect the relevant script before proposing changes
- preserve the current research workflow unless explicitly asked to refactor it
- do not fabricate dataset locations; treat data paths as user-provided unless confirmed in code
- do not assume COMSOL or MATLAB steps are reproducible inside this repo
- keep changes minimal and targeted
- when changing training or preprocessing behavior, document the new command in `README.md` and update this file
- when uncertain whether a script is experimental or canonical, ask or inspect recent usage first

## Practical Repo Map

### Preprocessing

- `data_preprocess/convert_ds.py`
- `data_preprocess/convert_text_emb.py`
- `data_preprocess/convert_128.py`
- `data_preprocess/pipeline.py`
- `data_preprocess/pipeline_feature.py`
- `data_preprocess/wo_splt_text.py`
- `data_preprocess/feature_wosplit.py`

### Training / generation scripts

- `scripts/t2i_feature.py`
- `scripts/t2i_flow.py`
- `scripts/t2i_latent.py`

### Modeling

- `model/model.py`
- `model/__init__.py`

### Evaluation

- `metrics.py`

## Suggested Maintenance Pattern

Use this file as a lightweight memory, not as a full manual.
Keep it short and current.
When the project evolves, prefer updating the sections below:

- `Project Snapshot`
- `Short-Term Memory`
- `Current likely entrypoints`
- `Environment assumptions`
- `Open questions / next steps`

## Open Questions / Next Steps

- Clarify which of `t2i_feature.py`, `t2i_flow.py`, and `t2i_latent.py` is the primary training path for the latest experiments.
- Clarify expected dataset directory structure beyond the README example.
- Clarify whether `model/text_guided_image_generation.pth` is a required checkpoint, an example artifact, or a stale file.
- Decide whether the README should document all supported training variants or only the mainline path.

## Template For Future Updates

When updating this file, prefer concise bullets like:

- Current main experiment:
- Current best script:
- Dataset expected at:
- Important config knobs:
- Known broken parts:
- Next thing to implement:


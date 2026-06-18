# FP4D Point-SAM Labeller

A browser tool for manually labelling FieldPheno4D crop point clouds, built on
[Point-SAM](https://github.com/zyc00/Point-SAM). It takes the raw bonndata
FieldPheno4D scans and helps you produce ground-removed rows, isolated single
plants, and per-plant leaf and stem labels, with Point-SAM driving the smart
mask.

The full tool documentation lives in **[`demo/README.md`](./demo/README.md)**.
This page is a short overview.

## What it does

- Opens any chosen dataset folder, including the raw bonndata FieldPheno4D
  download. It auto-detects every plot under the root and lists each one by its
  crop and variety name (Mirza, Popcorn Robust, and so on), not the bare plot id.
- Renders the full-resolution cloud. The Point-SAM encoder runs on a capped
  sample and the mask is propagated back to every point by nearest neighbour, so
  large clouds stay responsive.
- Three labelling modes that form a pipeline: ground removal on the raw row,
  plant separation on the ground-removed vegetation, and leaf and stem labelling
  on the isolated plant.
- A smart mask matched to native Point-SAM. Clicks drive the mask, neutral
  features are fed to the model, and the three subpart, part, and whole
  candidates are exposed for you to cycle with Tab. Include and exclude clicks
  refine a live preview, and you commit with Enter.

## Quick start

```bash
git clone https://github.com/LThannen/Point-SAM.git
cd Point-SAM

# checkpoint (about 1.19 GB, not in the repo)
pip install huggingface_hub
huggingface-cli download yuchen0187/Point-SAM model.safetensors --local-dir pretrained

# environment
python -m venv venv
venv/bin/pip install flask flask-cors hydra-core omegaconf safetensors laspy scipy matplotlib numpy timm
# install torch for your CUDA or driver per https://pytorch.org

# run, pointing at the unzipped FieldPheno4D download
PYTHONPATH=$PWD venv/bin/python demo/app.py --port 5056 \
  --dataset-root /path/to/doi-10.60507-fk2-hyi2ds/
```

Open `http://localhost:5056`, pick a plant from the Plot dropdown, choose a mode,
and start labelling. See [`demo/README.md`](./demo/README.md) for flags, the
staged dataset layout, and the full browser workflow.

## Data

- The raw scans come from the FieldPheno4D dataset, CC BY 4.0,
  [doi:10.60507/FK2/HYI2DS](https://bonndata.uni-bonn.de/dataset.xhtml?persistentId=doi:10.60507/FK2/HYI2DS).
- Pre-labelled clouds (ground removed, isolated plants, leaf and stem labels) are
  packaged and shipped separately as a staged dataset folder. Point
  `--dataset-root` at it to reopen the labels exactly as made.

## Built on Point-SAM

This is a fork of [Point-SAM](https://github.com/zyc00/Point-SAM) by Yuchen Zhou,
Jiayuan Gu, Tung Yen Chiang, Fanbo Xiang, and Hao Su (UC San Diego, Hillbot). The
model code under `pc_sam/`, the configs, and the ViT-L checkpoint are theirs. The
FieldPheno4D labelling tool under `demo/` is the addition in this fork.

Upstream resources: [paper](https://arxiv.org/abs/2406.17741),
[project page](https://point-sam.github.io),
[checkpoint](https://huggingface.co/yuchen0187/Point-SAM/tree/main).

### Citation

```
@inproceedings{
  zhou2025pointsam,
  title={Point-{SAM}: Promptable 3D Segmentation Model for Point Clouds},
  author={Yuchen Zhou and Jiayuan Gu and Tung Yen Chiang and Fanbo Xiang and Hao Su},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=yXCTDhZDh6}
}
```

### Acknowledgement

Point-SAM refers to [SAM](https://github.com/facebookresearch/segment-anything),
[Uni3D](https://github.com/baaivision/Uni3D), and
[OpenShape](https://github.com/Colin97/OpenShape_code).

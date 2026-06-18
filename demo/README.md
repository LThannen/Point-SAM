# FP4D Point-SAM Demo

Run the manual segmentation app from the demo directory:

```bash
PYTHONPATH=/home/lukas/pointr/Point-SAM \
python app.py \
  --dataset-root /path/to/fp4d_pointsam_dataset \
  --ckpt_path /home/lukas/pointr/Point-SAM/pretrained/model.safetensors
```

Useful flags:

- `--dataset-root`: staged dataset folder. The app understands `stage1_ground_removed/`, `stage2_plants_isolated/`, and `stage3_leafstem_labeled/`.
- `--dataset-plot`: optional plot folder to load when a dataset contains multiple plots.
- `--datasets-parent`: optional parent folder scanned for dataset choices in the UI.
- `--model-cap`: maximum points encoded by Point-SAM. Defaults to `POINTSAM_MODEL_CAP` or `400000`; the full cloud is still rendered and labels are propagated back to full resolution.

Dataset contract:

- Stage 1 files are `stage1_ground_removed/PlotXX/<date>.npy` dicts with `xyz_utm`, `xyz_local`, `height`, `epsg`, `plot`, and `date`.
- Stage 2 files are isolated plant clouds under `stage2_plants_isolated/PlotXX/plant_NN/` with matching `_utm.npy` companions when available.
- Stage 3 files are hand labels under `stage3_leafstem_labeled/PlotXX/plant_NN/handlabel_NN_<date>.npy`; QC PNGs sit next to the labels.

Smart Mask workflow:

1. Select Smart Mask.
2. Use Include Click and Exclude Click to refine the preview.
3. Press Enter or Accept mask to commit.
4. Press Esc or Clear Prompts to cancel the preview and reset prompts.

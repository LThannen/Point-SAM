# FP4D Point-SAM Segmentation Tool

This demo is the FP4D manual segmentation workflow built on Point-SAM. It can:

- open a staged FP4D dataset folder,
- render the full-resolution point cloud,
- run Point-SAM on a capped model sample,
- propagate masks back to the full cloud,
- save row vegetation, plant separation, and leaf/stem labels back into the selected dataset folder.

The app is a single-user Flask + Three.js tool. Run it locally, open it in a browser, label data, and export the results next to the dataset you selected.

## Repository Layout

Important paths:

```text
/home/lukas/pointr/
  package_dataset.py                 # builds a clean staged FP4D dataset
  fp4d_pointsam_dataset/             # default packaged dataset location, if created
  Point-SAM/
    pretrained/model.safetensors     # Point-SAM checkpoint
    demo/
      app.py                         # Flask backend
      static/index.html              # browser UI
      static/annotate.js             # labelling logic
      README.md                      # this file
```

The tool expects a staged dataset with this layout:

```text
fp4d_pointsam_dataset/
  README.md
  manifest.json
  row_frame.json                     # optional; if absent, raw row mode loads all raw points
  raw/PlotXX/<date>.las              # optional; needed only for raw full-row mode
  stage1_ground_removed/
    PlotXX/<date>.npy
  stage2_plants_isolated/
    PlotXX/base_centres.npy
    PlotXX/plant_NN/plant_NN_<date>.npy
    PlotXX/plant_NN/plant_NN_<date>_utm.npy
  stage3_leafstem_labeled/
    PlotXX/plant_NN/handlabel_NN_<date>.npy
    PlotXX/plant_NN/handlabel_NN_<date>_qc.png
```

You can also point `--dataset-root` or the UI dataset path directly at a raw FP4D plot folder:

```text
/path/to/doi-10.60507-fk2-hyi2ds/Plot07/
  230516.las
  230525.las
  ...
```

In that case the app derives `plot=Plot07`, discovers dates from the `.las` files, and writes new labels under the selected `Plot07/` folder.

## Install

Use the existing Point-SAM environment when available:

```bash
cd /home/lukas/pointr/Point-SAM
/home/lukas/pointr/venv/bin/python -m pip --version
```

If you need to create a fresh environment, use Python 3.11+ with CUDA-capable PyTorch:

```bash
cd /home/lukas/pointr
python3.11 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install flask flask-cors hydra-core omegaconf safetensors laspy scipy matplotlib numpy timm
```

Install PyTorch for your CUDA/driver setup using the official PyTorch command for the machine. The verified local environment uses:

```bash
/home/lukas/pointr/venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.device_count())
PY
```

The Point-SAM Python package must be importable. The launch commands below set `PYTHONPATH=/home/lukas/pointr/Point-SAM`, which is enough for this repo checkout.

## Checkpoint

The default checkpoint path is:

```text
/home/lukas/pointr/Point-SAM/pretrained/model.safetensors
```

Verify it exists:

```bash
ls -lh /home/lukas/pointr/Point-SAM/pretrained/model.safetensors
```

If you keep the checkpoint somewhere else, pass it explicitly:

```bash
--ckpt_path /path/to/model.safetensors
```

## Package The Existing FP4D Labels

From `/home/lukas/pointr`, build a clean staged dataset:

```bash
cd /home/lukas/pointr
/home/lukas/pointr/venv/bin/python package_dataset.py \
  --src-preprocess /home/lukas/pointr/preprocess \
  --src-plants /home/lukas/pointr/fp4d_plants_basecarry \
  --out /home/lukas/pointr/fp4d_pointsam_dataset \
  --plots Plot03 Plot04
```

The packager is idempotent. Existing files are skipped, and the command prints per-stage counts:

```text
Plot03:
  stage1_ground_removed: seen=9 written=9 skipped=0, points=966960
  stage2_plants_isolated: seen=294 written=294 skipped=0
  stage3_leafstem_labeled: seen=34 written=34 skipped=0
```

Outputs:

- `README.md`: dataset description for users,
- `manifest.json`: source paths and per-stage counts,
- `stage1_ground_removed/`: vegetation-only row clouds,
- `stage2_plants_isolated/`: per-plant clouds and UTM companions,
- `stage3_leafstem_labeled/`: hand-labelled leaf/stem ground truth and QC PNGs.

## Start The App

Run from the demo directory:

```bash
cd /home/lukas/pointr/Point-SAM/demo
PYTHONPATH=/home/lukas/pointr/Point-SAM \
/home/lukas/pointr/venv/bin/python app.py \
  --host 127.0.0.1 \
  --port 5056 \
  --dataset-root /home/lukas/pointr/fp4d_pointsam_dataset \
  --ckpt_path /home/lukas/pointr/Point-SAM/pretrained/model.safetensors
```

Open:

```text
http://127.0.0.1:5056
```

For a lower-memory smoke test, cap the model sample:

```bash
POINTSAM_MODEL_CAP=2048 \
PYTHONPATH=/home/lukas/pointr/Point-SAM \
/home/lukas/pointr/venv/bin/python app.py \
  --host 127.0.0.1 \
  --port 5056 \
  --dataset-root /home/lukas/pointr/fp4d_pointsam_dataset
```

For normal work, use the default cap (`400000`) or tune it:

```bash
--model-cap 400000
```

The full cloud is still rendered and exported. `--model-cap` only limits the Point-SAM encoder sample; masks are propagated back to full resolution.

For raw LAS ground-removal mode, the browser displays a sampled working cloud controlled by `Raw display points` in the UI, or by `--n`/`POINTSAM_N` on startup. This is intentional for very large raw plots such as Plot07, where a single date can contain tens of millions of points.

If `Export full-res` is enabled, row labels are propagated back to the raw LAS by nearest neighbour during export. The backend builds a KD-tree from the sampled working cloud in original UTM coordinates, streams the full raw LAS in chunks, and gives each full-resolution point the label of its nearest sampled point. This keeps browser labelling responsive while still producing a full-resolution LAS. Use a larger raw display sample when boundary precision matters.

## Command-Line Flags

Main flags:

```text
--dataset-root PATH       Staged dataset folder to open.
--dataset-plot PlotXX     Optional plot to select when the dataset has multiple plots.
--datasets-parent PATH    Optional parent scanned for dataset choices in the UI.
--model-cap N             Max points encoded by Point-SAM. Default: POINTSAM_MODEL_CAP or 400000.
--ckpt_path PATH          Point-SAM checkpoint. Default: Point-SAM/pretrained/model.safetensors.
--host HOST               Flask bind host.
--port PORT               Flask bind port.
--date YYMMDD             Initial date.
--export-fullres          Default row export includes full-res propagated LAS.
--n N                     Raw row display/sample points. Default: POINTSAM_N or 300000.
```

`--dataset-root` accepts either a staged dataset root or a direct raw `PlotXX` folder. If you pass the parent of several raw plot folders, also pass `--dataset-plot PlotXX`.

Advanced row-crop flags:

```text
--tlo FLOAT               Lower row-frame crop bound.
--thi FLOAT               Upper row-frame crop bound.
```

If the dataset has no `row_frame.json`, row mode does not crop by row frame and loads all raw LAS points for the selected date.

## Browser Workflow

### Choose Dataset

At the top of the sidebar:

1. Select a dataset from the dropdown, or paste a dataset path.
2. Click `Set dataset`.
3. The Date and Plant dropdowns refresh from the selected folder.

Writes go into the selected dataset root:

- row labels under the dataset label/stage path,
- plant separation under the selected dataset,
- leaf/stem handlabels under the selected dataset.

### Modes

Use the `Mode` dropdown:

```text
Ground removal        Raw row mode: plant vs ground.
Plant separation      Ground-removed row vegetation: assign plant IDs.
Per-plant leaf/stem    Isolated plant mode: stem vs leaf IDs.
```

### Manual Tools

Tools:

```text
Brush Paint     Paint points directly.
Lasso Paint     Draw a polygon and assign all points inside.
Crop Box        Focus row mode on a rectangle.
Delete Points   Remove selected points from the active cloud.
Smart Mask      Point-SAM prompt mode.
```

Navigation:

```text
Middle drag     Rotate.
Right drag      Pan.
Wheel           Zoom.
```

### Smart Mask

Smart Mask is preview-first:

1. Select `Smart Mask`.
2. Select the target class or plant/leaf target.
3. Use `Include Click` for positive prompts.
4. Use `Exclude Click` for negative prompts.
5. Every click updates the cyan preview mask.
6. Press `Enter` or click `Accept mask` to commit.
7. Press `Esc` or `Clear Prompts` to cancel.

The status line shows:

- preview mask point count,
- total prompts,
- positive/negative prompt count,
- predicted IoU.

The backend accumulates prompts until commit or clear. After commit, prompts are reset.

## Export

Click `Export`.

Outputs depend on mode:

```text
Ground removal        2-class NPY/LAS labels.
Plant separation      plant_NN_<date>.npy, plant_NN_<date>_utm.npy, plantsep QC.
Per-plant leaf/stem    handlabel_NN_<date>.npy, gt_otype, gt_leafid, QC PNG, manifest row.
```

In plant mode, handlabels embed `xyz_utm` and `xyz_local`, so already-labelled plants reopen with the exact geometry that was labelled.

## Verification

Package verification:

```bash
cd /home/lukas/pointr
/home/lukas/pointr/venv/bin/python package_dataset.py --out /tmp/fp4d_ds

/home/lukas/pointr/venv/bin/python - <<'PY'
import json
import numpy as np
from pathlib import Path

root = Path('/tmp/fp4d_ds')
assert (root / 'stage1_ground_removed').is_dir()
assert (root / 'stage2_plants_isolated').is_dir()
assert (root / 'stage3_leafstem_labeled').is_dir()
manifest = json.load(open(root / 'manifest.json'))
print(manifest['plots'].keys())

hand = next((root / 'stage3_leafstem_labeled').glob('Plot*/plant_*/handlabel_*.npy'))
obj = np.load(hand, allow_pickle=True).item()
assert 'otype' in obj and 'leafid' in obj
print(hand, len(obj['otype']))
PY
```

App boot verification:

```bash
cd /home/lukas/pointr/Point-SAM/demo
PYTHONPATH=/home/lukas/pointr/Point-SAM \
/home/lukas/pointr/venv/bin/python app.py \
  --host 127.0.0.1 \
  --port 5056 \
  --dataset-root /tmp/fp4d_ds \
  --model-cap 2048
```

In another terminal:

```bash
curl -s http://127.0.0.1:5056/datasets
curl -s http://127.0.0.1:5056/dates
curl -s http://127.0.0.1:5056/pointcloud/current > /tmp/pc_current.json
```

Expected server log for staged stage1 load:

```text
loaded Plot03 230711: rendered 294,084/294,084 points model 2,048/294,084 cap=2,048 from 230711.npy
```

Smart Mask API smoke:

```bash
/home/lukas/pointr/venv/bin/python - <<'PY'
import json
import urllib.request

base = 'http://127.0.0.1:5056'

def post(path, payload):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
        method='POST',
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)

plant = post('/load_plant', {'plant_id': '12', 'date': '230619'})
xyz = plant['xyz']
p0 = xyz[0:3]
p1 = xyz[-3:]

seg1 = post('/segment', {
    'prompt_point': p0,
    'prompt_label': True,
    'active_label': 1,
    'target': {'kind': 'stem'},
    'use_label_context': True,
})
seg2 = post('/segment', {
    'prompt_point': p1,
    'prompt_label': False,
    'active_label': 1,
    'target': {'kind': 'stem'},
    'use_label_context': True,
})
commit = post('/commit', {'label': 1, 'target': {'kind': 'stem'}, 'layer_policy': {}})
print(len(seg1['seg']), sum(seg1['seg']))
print(len(seg2['seg']), sum(seg2['seg']))
print(commit['status'], commit['changed'])
PY
```

## Troubleshooting

### `ModuleNotFoundError: hydra`, `flask`, `laspy`, or `torch`

Use the Point-SAM venv:

```bash
/home/lukas/pointr/venv/bin/python app.py ...
```

or install the missing package in your active environment.

### CUDA out of memory during cloud load

Lower the model cap:

```bash
--model-cap 200000
```

or:

```bash
POINTSAM_MODEL_CAP=200000
```

The app also retries automatically by halving the cap if encoder OOM occurs.

### Raw Plot07 or another raw plot is too large to load

Raw LAS dates can contain tens of millions of points. The app now loads only the sampled working cloud for browser labelling:

```bash
--n 300000
```

or:

```bash
POINTSAM_N=300000
```

In the UI this is the `Raw display points` selector. The server log should show both numbers:

```text
loaded Plot07 230711: rendered 120,000/38,492,213 points ...
```

The first number is the working cloud sent to the browser. The second number is the full raw row point count.

### Dataset opens but raw row mode fails

Raw row mode requires:

```text
raw/PlotXX/<date>.las
```

If your staged dataset only has `stage1_ground_removed`, use `Plant separation` or let the initial load open the staged vegetation cloud.

### `/datasets` is slow or shows too many folders

Pass a narrower parent:

```bash
--datasets-parent /path/to/datasets
```

### Browser shows a blank viewer

Check:

- the Flask terminal for a traceback,
- browser console errors,
- that `/pointcloud/current` returns JSON,
- that the selected dataset has a valid date and points.

### Existing handlabels appear blank

The app first tries the embedded-geometry `handlabel_NN_<date>.npy` path. Confirm the file has:

```text
xyz_utm, xyz_local, otype, leafid
```

Quick check:

```bash
/home/lukas/pointr/venv/bin/python - <<'PY'
import numpy as np
obj = np.load('/path/to/handlabel_12_230619.npy', allow_pickle=True).item()
print(obj.keys())
PY
```

## Notes For Other Users

- The app is intentionally single-user and runs Flask with `threaded=False`; do not use one server instance for multiple simultaneous labellers.
- `--model-cap` does not downsample exports. It only controls the model encoder input.
- `--n` / `Raw display points` controls the sampled working cloud for raw LAS ground-removal mode.
- Point coordinates sent by browser clicks are full-resolution normalized coordinates. No index remapping is needed for prompts.
- If a dataset has `row_frame.json`, row raw mode applies the row crop. If not, raw mode loads all raw points for that date.

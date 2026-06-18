import argparse
import csv
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import hydra
import laspy
import numpy as np
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm
from omegaconf import OmegaConf
from safetensors.torch import load_model
from scipy.spatial import cKDTree
from laspy.vlrs.geotiff import (
    GTModelTypeGeoKey,
    GTRasterTypeGeoKey,
    ModelTypeProjected,
    ProjectedCSTypeGeoKey,
)
from laspy.vlrs.known import GeoKeyDirectoryVlr, GeoKeyEntryStruct

from pc_sam.model.pc_sam import AuxInputs, repeat_interleave
from pc_sam.utils.torch_utils import replace_with_fused_layernorm


POINTSAM_ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREPROCESS_ROOT = Path("/home/lukas/pointr/preprocess")
LEGACY_LABEL_ROOT = LEGACY_PREPROCESS_ROOT / "labels"
LEGACY_LEAFSTEM_LABEL_ROOT = LEGACY_PREPROCESS_ROOT / "labels_leafstem"
STATIC_MODEL_ROOT = Path(__file__).resolve().parent / "static" / "models"
LEGACY_PLANT_ROOT = Path("/home/lukas/pointr/fp4d_plants_basecarry/Plot03")
LEGACY_RAW_ROOT = Path(
    "/home/lukas/PHD/Resources/PHENOROAM DATA ASSIMILATION May 2026/"
    "doi-10.60507-fk2-hyi2ds/Plot03"
)
LEGACY_EPSG = 25832
LEGACY_PLOT = "Plot03"
DEFAULT_DATASET_ROOT = Path("/home/lukas/pointr/fp4d_pointsam_dataset")
LEAF_PALETTE = np.asarray(
    [
        (0.00, 0.45, 1.00),  # blue
        (1.00, 0.55, 0.00),  # orange
        (0.00, 0.70, 0.20),  # green
        (0.95, 0.00, 0.85),  # magenta
        (0.00, 0.75, 0.85),  # cyan
        (0.95, 0.85, 0.00),  # yellow
        (0.55, 0.20, 1.00),  # violet
        (1.00, 0.15, 0.15),  # red
        (0.00, 0.95, 0.55),  # mint
        (1.00, 0.35, 0.65),  # pink
        (0.35, 0.70, 1.00),  # sky
        (0.70, 0.45, 0.00),  # ochre
    ],
    dtype=np.float32,
)
BASE_PLANT_COUNT = 14
REF_DATE = "230621"

sys.path.append(str(LEGACY_PREPROCESS_ROOT))


@dataclass(frozen=True)
class Dataset:
    root: Path
    plot: str
    raw_dir: Path | None
    stage1_dir: Path | None
    plant_root: Path
    leafstem_root: Path
    label_root: Path
    row_frame: dict | None
    dates: tuple[str, ...]
    separation_dates: tuple[str, ...]
    early_dates: tuple[str, ...]
    plants: tuple[str, ...]
    epsg: int


def _npy_dict(path: Path):
    obj = np.load(path, allow_pickle=True)
    if getattr(obj, "shape", None) == ():
        value = obj.item()
        if isinstance(value, dict):
            return value
    return None


def _dates_from_stage1(stage1_dir: Path | None) -> set[str]:
    if not stage1_dir or not stage1_dir.exists():
        return set()
    return {p.stem for p in stage1_dir.glob("*.npy") if p.stem.isdigit()}


def _dates_from_plants(plant_root: Path) -> set[str]:
    dates = set()
    if not plant_root.exists():
        return dates
    for path in plant_root.glob("plant_*/plant_*_*.npy"):
        stem = path.stem
        if stem.endswith("_utm"):
            stem = stem[:-4]
        date = stem.rsplit("_", 1)[-1]
        if date.isdigit():
            dates.add(date)
    return dates


def _plants_from_root(plant_root: Path) -> tuple[str, ...]:
    plants = set(range(BASE_PLANT_COUNT))
    if plant_root.exists():
        for path in plant_root.glob("plant_*"):
            if not path.is_dir():
                continue
            try:
                plants.add(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    return tuple(f"{plant:02d}" for plant in sorted(plants))


def _read_epsg(paths: list[Path]) -> int:
    for path in paths:
        if not path.exists():
            continue
        data = _npy_dict(path)
        if data and "epsg" in data:
            return int(data["epsg"])
    return LEGACY_EPSG


def _read_row_frame(root: Path) -> dict | None:
    path = root / "row_frame.json"
    if path.exists():
        return json.loads(path.read_text())
    if root == LEGACY_PLANT_ROOT and (LEGACY_PREPROCESS_ROOT / "row_frame.json").exists():
        return json.loads((LEGACY_PREPROCESS_ROOT / "row_frame.json").read_text())
    return None


def _choose_plot(root: Path, requested: str | None) -> str:
    if requested:
        return requested
    candidates = []
    for stage in ("stage1_ground_removed", "stage2_plants_isolated", "stage3_leafstem_labeled"):
        stage_dir = root / stage
        if stage_dir.exists():
            candidates.extend(p.name for p in stage_dir.iterdir() if p.is_dir())
    return sorted(candidates)[0] if candidates else LEGACY_PLOT


def _resolve_dataset(root: Path, plot: str | None = None) -> Dataset:
    root = root.expanduser().resolve()
    if not root.exists() and root != DEFAULT_DATASET_ROOT:
        raise FileNotFoundError(root)
    plot = _choose_plot(root, plot)
    staged = (root / "stage1_ground_removed").exists() or (root / "stage2_plants_isolated").exists()
    if staged:
        stage1_dir = root / "stage1_ground_removed" / plot
        plant_root = root / "stage2_plants_isolated" / plot
        leafstem_root = root / "stage3_leafstem_labeled" / plot
        label_root = stage1_dir
        raw_dir = root / "raw" / plot if (root / "raw" / plot).exists() else None
    else:
        stage1_dir = root / "labels" if (root / "labels").exists() else None
        plant_root = root / plot if (root / plot / "base_centres.npy").exists() else root
        leafstem_root = plant_root
        label_root = LEGACY_LABEL_ROOT if root == LEGACY_PLANT_ROOT else root / "labels"
        raw_dir = LEGACY_RAW_ROOT if root == LEGACY_PLANT_ROOT else root / "raw" / plot
        if not raw_dir.exists():
            raw_dir = None
    dates = sorted(_dates_from_stage1(stage1_dir) | _dates_from_plants(plant_root))
    epsg_candidates = []
    if stage1_dir and stage1_dir.exists():
        epsg_candidates.extend(sorted(stage1_dir.glob("*.npy"))[:3])
    if leafstem_root.exists():
        epsg_candidates.extend(sorted(leafstem_root.glob("plant_*/handlabel_*.npy"))[:3])
    epsg = _read_epsg(epsg_candidates)
    early_dates = tuple(d for d in dates if d <= "230621") or tuple(dates[:3])
    return Dataset(
        root=root,
        plot=plot,
        raw_dir=raw_dir,
        stage1_dir=stage1_dir,
        plant_root=plant_root,
        leafstem_root=leafstem_root,
        label_root=label_root,
        row_frame=_read_row_frame(root),
        dates=tuple(dates),
        separation_dates=tuple(dates),
        early_dates=early_dates,
        plants=_plants_from_root(plant_root),
        epsg=epsg,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="localhost")
parser.add_argument("--port", type=int, default=5000)
parser.add_argument("--checkpoint", type=str, default="pretrained/model.safetensors")
parser.add_argument("--pointcloud", type=str, default="")
parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT if DEFAULT_DATASET_ROOT.exists() else LEGACY_PLANT_ROOT)
parser.add_argument("--datasets-parent", type=Path, default=None)
parser.add_argument("--dataset-plot", type=str, default=None)
parser.add_argument("--date", type=str, default="230711")
parser.add_argument("--n", type=int, default=int(os.environ.get("POINTSAM_N", 300000)))
parser.add_argument("--model-cap", type=int, default=int(os.environ.get("POINTSAM_MODEL_CAP", 400000)))
parser.add_argument("--tlo", type=float, default=-0.5)
parser.add_argument("--thi", type=float, default=0.5)
parser.add_argument("--plant-height-threshold", type=float, default=0.05)
parser.add_argument("--use-height-prior", action="store_true")
parser.add_argument("--export-fullres", action="store_true")
parser.add_argument("--config", type=str, default="large", help="path to config file")
parser.add_argument("--config_dir", type=str, default="../configs")
parser.add_argument(
    "--ckpt_path",
    type=str,
    default=str(POINTSAM_ROOT / "pretrained" / "model.safetensors"),
)
args = parser.parse_args()
active_dataset = _resolve_dataset(args.dataset_root, args.dataset_plot)


app = Flask(__name__, static_folder="static")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
CORS(app, origins=f"{args.host}:{args.port}", allow_headers="Access-Control-Allow-Origin")


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


with hydra.initialize(args.config_dir, version_base=None):
    cfg = hydra.compose(config_name=args.config)
    OmegaConf.resolve(cfg)

model = hydra.utils.instantiate(cfg.model)
model.apply(replace_with_fused_layernorm)
load_model(model, args.ckpt_path)
model = model.eval().cuda()


state = {
    "mode": "row",
    "date": None,
    "plant": None,
    "raw_path": None,
    "xyz_utm": None,
    "xyz_local": None,
    "height": None,
    "display_rgb": None,
    "xyz_norm": None,
    "norm_shift": None,
    "norm_scale": None,
    "labels": None,
    "otype": None,
    "leafid": None,
    "plant_id": None,
    "prompt_coords": [],
    "prompt_labels": [],
    "current_mask": None,
    "undo": [],
    "encoder": None,
    "pc_xyz": None,
    "pc_features": None,
    "model_idx": None,
    "model_xyz_norm": None,
    "model_cap": args.model_cap,
    "row_count": 0,
    "n_target": args.n,
    "crop_parent_count": None,
    "stem_base": None,
    "base_markers": None,
    "real_count": 0,
    "loaded_labels_from": None,
}


def _raw_path(date):
    if active_dataset.raw_dir is None:
        raise FileNotFoundError(f"dataset {active_dataset.root} has no raw directory for {active_dataset.plot}")
    path = active_dataset.raw_dir / f"{date}.las"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _row_mask(x, y):
    if active_dataset.row_frame is None:
        return np.ones_like(x, dtype=bool)
    row_origin = np.asarray(active_dataset.row_frame["origin"], dtype=np.float64)
    row_e2 = np.asarray(active_dataset.row_frame["e2"], dtype=np.float64)
    p = np.c_[x, y] - row_origin
    t = p @ row_e2
    return (t > args.tlo) & (t < args.thi)


def _load_row_sample(date, n_target):
    raw_path = _raw_path(date)
    xs, ys, zs, hs = [], [], [], []
    row_count = 0

    with laspy.open(raw_path) as fh:
        extra = set(fh.header.point_format.extra_dimension_names)
        hname = "height" if "height" in extra else None
        for pts in fh.chunk_iterator(4_000_000):
            x = np.asarray(pts.x)
            y = np.asarray(pts.y)
            m = _row_mask(x, y)
            row_count += int(m.sum())
            if not np.any(m):
                continue
            z = np.asarray(pts.z)
            h = np.asarray(pts[hname]).astype(np.float64) if hname else z - z.min()
            xs.append(x[m])
            ys.append(y[m])
            zs.append(z[m])
            hs.append(h[m])

    if not xs:
        raise RuntimeError(f"No row points found for {date}")

    x = np.concatenate(xs)
    y = np.concatenate(ys)
    z = np.concatenate(zs)
    h = np.concatenate(hs)

    xyz_utm = np.column_stack((x, y, z)).astype(np.float64)
    hn = np.clip((h - h.min()) / (np.ptp(h) + 1e-9), 0, 1)
    display_rgb = cm.viridis(hn)[:, :3].astype(np.float32)
    return raw_path, xyz_utm, h.astype(np.float32), display_rgb, row_count


def _normalize_xyz(xyz_utm):
    shift = xyz_utm.mean(axis=0)
    scale = np.linalg.norm(xyz_utm - shift, axis=1).max()
    xyz_norm = ((xyz_utm - shift) / scale).astype(np.float32)
    return xyz_norm, shift, float(scale)


def _encode_pointcloud(pc_xyz, pc_features):
    with torch.no_grad():
        pc_embeddings, patches = model.pc_encoder(pc_xyz, pc_features)
        centers = patches["centers"]
        return {
            "pc_embeddings": pc_embeddings,
            "patches": patches,
            "centers": centers,
            "pc_pe": model.point_encoder.pe_layer(centers),
            "aux_inputs": AuxInputs(coords=pc_xyz, features=pc_features, centers=centers),
        }


def _predict_with_cached_encoder(prompt_coords, prompt_labels, multimask_output=True):
    enc = state["encoder"]
    prompt_masks = None
    sparse_embeddings = model.point_encoder(prompt_coords, prompt_labels)
    dense_embeddings = model.mask_encoder(
        prompt_masks,
        state["pc_xyz"],
        enc["centers"],
        enc["patches"]["knn_idx"],
    )
    dense_embeddings = repeat_interleave(
        dense_embeddings,
        sparse_embeddings.shape[0] // dense_embeddings.shape[0],
        0,
    )
    return model.mask_decoder(
        enc["pc_embeddings"],
        enc["pc_pe"],
        sparse_embeddings,
        dense_embeddings,
        aux_inputs=enc["aux_inputs"],
        multimask_output=multimask_output,
    )


def _context_prompts_for_target(active_label, target=None, max_per_class=64):
    if active_label not in (1, 2) or state["labels"] is None:
        return [], []

    labels = state["labels"]
    if state["mode"] == "separation":
        return [], []
    if state["mode"] == "plant":
        target = target or {}
        if active_label == 1:
            pos_idx = np.flatnonzero(state["otype"] == 1)
            neg_idx = np.flatnonzero(state["otype"] == 2)
        elif target.get("kind") == "leaf" and int(target.get("leafid", 0)) > 0:
            leaf = int(target["leafid"])
            pos_idx = np.flatnonzero((state["otype"] == 2) & (state["leafid"] == leaf))
            neg_idx = np.flatnonzero((state["otype"] != 0) & ~((state["otype"] == 2) & (state["leafid"] == leaf)))
        else:
            pos_idx = np.flatnonzero(state["otype"] == 2)
            neg_idx = np.flatnonzero(state["otype"] == 1)
    else:
        pos_idx = np.flatnonzero(labels == active_label)
        neg_idx = np.flatnonzero((labels != 0) & (labels != active_label))
    rng = np.random.default_rng(17)

    if len(pos_idx) > max_per_class:
        pos_idx = rng.choice(pos_idx, max_per_class, replace=False)
    if len(neg_idx) > max_per_class:
        neg_idx = rng.choice(neg_idx, max_per_class, replace=False)

    coords = []
    prompt_labels = []
    if len(pos_idx):
        coords.extend(state["xyz_norm"][pos_idx].tolist())
        prompt_labels.extend([1] * len(pos_idx))
    if len(neg_idx):
        coords.extend(state["xyz_norm"][neg_idx].tolist())
        prompt_labels.extend([0] * len(neg_idx))
    return coords, prompt_labels


def _set_active_cloud(
    date,
    raw_path,
    xyz_utm,
    height,
    display_rgb,
    row_count,
    n_target,
    crop_parent_count=None,
    mode="row",
    plant=None,
    xyz_local=None,
    display_xyz=None,
    otype=None,
    leafid=None,
    plant_id=None,
    stem_base=None,
    base_markers=None,
    loaded_labels_from=None,
):
    start = time.time()
    display_source = xyz_utm if display_xyz is None else display_xyz
    xyz_norm, shift, scale = _normalize_xyz(display_source)
    height_norm = np.clip((height - height.min()) / (np.ptp(height) + 1e-9), 0, 1).astype(np.float32)

    real_count = len(xyz_norm)
    requested_cap = max(1024, int(state.get("model_cap") or args.model_cap))
    cap = min(requested_cap, real_count)
    rng = np.random.default_rng(0)
    while True:
        if real_count <= cap:
            model_idx = np.arange(real_count, dtype=np.int64)
        else:
            model_idx = np.sort(rng.choice(real_count, cap, replace=False)).astype(np.int64)
        model_xyz_norm = xyz_norm[model_idx]
        model_height_norm = height_norm[model_idx]
        pc_features_np = np.repeat(model_height_norm[:, None], 3, axis=1).astype(np.float32)
        pc_xyz_np = model_xyz_norm
        if len(pc_xyz_np) < 1024:
            pad = 1024 - len(pc_xyz_np)
            pad_idx = np.resize(np.arange(len(pc_xyz_np)), pad)
            pc_xyz_np = np.vstack([pc_xyz_np, model_xyz_norm[pad_idx]])
            pc_features_np = np.vstack([pc_features_np, pc_features_np[pad_idx]])
        pc_xyz = torch.from_numpy(pc_xyz_np).cuda().float().unsqueeze(0)
        pc_features = torch.from_numpy(pc_features_np).cuda().float().unsqueeze(0)
        try:
            encoder = _encode_pointcloud(pc_xyz, pc_features)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or cap <= 1024:
                raise
            torch.cuda.empty_cache()
            cap = max(1024, cap // 2)
            print(f"Point-SAM encode OOM; retrying with model cap {cap:,}", flush=True)

    state.update(
        mode=mode,
        date=date,
        plant=plant,
        raw_path=raw_path,
        xyz_utm=xyz_utm,
        xyz_local=xyz_local if xyz_local is not None else xyz_utm,
        height=height,
        display_rgb=display_rgb,
        xyz_norm=xyz_norm,
        norm_shift=shift,
        norm_scale=scale,
        labels=np.zeros(len(xyz_utm), dtype=np.uint8),
        otype=np.zeros(len(xyz_utm), dtype=np.uint8) if otype is None else otype.astype(np.uint8),
        leafid=np.zeros(len(xyz_utm), dtype=np.int16) if leafid is None else leafid.astype(np.int16),
        plant_id=np.full(len(xyz_utm), -1, dtype=np.int16) if plant_id is None else plant_id.astype(np.int16),
        prompt_coords=[],
        prompt_labels=[],
        current_mask=None,
        undo=[],
        encoder=encoder,
        pc_xyz=pc_xyz,
        pc_features=pc_features,
        model_idx=model_idx,
        model_xyz_norm=model_xyz_norm,
        model_cap=cap,
        row_count=row_count,
        n_target=n_target,
        crop_parent_count=crop_parent_count,
        stem_base=stem_base,
        base_markers=base_markers or [],
        real_count=real_count,
        loaded_labels_from=loaded_labels_from,
    )
    if mode == "plant":
        state["labels"] = state["otype"]
    elif mode == "separation":
        state["labels"] = np.where(state["plant_id"] >= 0, state["plant_id"] + 1, 0).astype(np.uint8)
    print(
        f"loaded {active_dataset.plot} {date}"
        f"{'' if plant is None else f' plant {plant}'}: rendered {len(xyz_utm):,}/{row_count:,} points "
        f"model {len(model_idx):,}/{real_count:,} cap={cap:,} "
        f"from {raw_path.name} in {time.time() - start:.1f}s",
        flush=True,
    )


def _cloud_snapshot():
    return {
        "mode": state["mode"],
        "date": state["date"],
        "plant": state["plant"],
        "raw_path": state["raw_path"],
        "xyz_utm": state["xyz_utm"].copy(),
        "xyz_local": state["xyz_local"].copy(),
        "height": state["height"].copy(),
        "display_rgb": state["display_rgb"].copy(),
        "labels": state["labels"].copy(),
        "otype": state["otype"].copy(),
        "leafid": state["leafid"].copy(),
        "plant_id": state["plant_id"].copy(),
        "row_count": state["row_count"],
        "n_target": state["n_target"],
        "crop_parent_count": state["crop_parent_count"],
        "stem_base": state["stem_base"],
        "base_markers": list(state["base_markers"] or []),
        "loaded_labels_from": state["loaded_labels_from"],
    }


def _restore_cloud_snapshot(snapshot, undo_stack):
    _set_active_cloud(
        snapshot["date"],
        snapshot["raw_path"],
        snapshot["xyz_utm"],
        snapshot["height"],
        snapshot["display_rgb"],
        snapshot["row_count"],
        snapshot["n_target"],
        crop_parent_count=snapshot["crop_parent_count"],
        mode=snapshot.get("mode", "row"),
        plant=snapshot.get("plant"),
        xyz_local=snapshot.get("xyz_local"),
        display_xyz=snapshot.get("xyz_local") if snapshot.get("mode") in ("plant", "separation") else None,
        otype=snapshot.get("otype"),
        leafid=snapshot.get("leafid"),
        plant_id=snapshot.get("plant_id"),
        stem_base=snapshot.get("stem_base"),
        base_markers=snapshot.get("base_markers"),
        loaded_labels_from=snapshot.get("loaded_labels_from"),
    )
    state["labels"] = snapshot["labels"]
    state["undo"] = undo_stack


def _load_date(date, n_target=None):
    n_target = int(n_target or args.n)
    start = time.time()
    raw_path, xyz_utm, height, display_rgb, row_count = _load_row_sample(date, n_target)
    _set_active_cloud(date, raw_path, xyz_utm, height, display_rgb, row_count, n_target)
    print(f"date reload completed in {time.time() - start:.1f}s", flush=True)


def _leafstem_label_dir():
    active_dataset.leafstem_root.mkdir(parents=True, exist_ok=True)
    return active_dataset.leafstem_root


def _row_veg_path(date):
    candidates = []
    if active_dataset.stage1_dir is not None:
        candidates.append(active_dataset.stage1_dir / f"{date}.npy")
    candidates.append(active_dataset.label_root / f"{active_dataset.plot}_{date}_2class.npy")
    candidates.append(active_dataset.label_root / f"{date}.npy")
    path = next((p for p in candidates if p.exists()), candidates[0])
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _separator_ref_xy():
    o = np.load(_row_veg_path(REF_DATE), allow_pickle=True).item()
    if "label" in o:
        xyz = o["xyz_utm"][o["label"] == 1]
    else:
        xyz = o["xyz_utm"]
    if len(xyz) == 0:
        raise RuntimeError(f"no plant points in {_row_veg_path(REF_DATE)}")
    return (xyz[:, :2] * 100.0).mean(0)


def _utm_to_separator_cm(xyz_utm, ref_xy=None):
    ref_xy = _separator_ref_xy() if ref_xy is None else ref_xy
    xyz_local = xyz_utm.astype(np.float64).copy()
    xyz_local[:, 2] -= xyz_local[:, 2].min()
    xyz_local *= 100.0
    xyz_local[:, :2] -= ref_xy
    return xyz_local.astype(np.float32)


def _plant_palette(ids):
    palette = np.asarray(list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors), dtype=np.float32)
    rgb = np.full((len(ids), 3), 0.33, dtype=np.float32)
    for pid in sorted(int(x) for x in np.unique(ids) if int(x) >= 0):
        rgb[ids == pid] = palette[pid % len(palette)]
    return rgb


def _manual_sep_path(date):
    return _leafstem_label_dir() / f"plantsep_{active_dataset.plot}_{date}.npy"


def _base_markers_local():
    base_path = active_dataset.plant_root / "base_centres.npy"
    if not base_path.exists():
        return []
    bases = np.load(base_path)
    return [[float(x), float(y), 0.0] for x, y in bases[:, :2]]


def _seed_plant_ids_from_auto(xyz_utm, date):
    plant_id = np.full(len(xyz_utm), -1, dtype=np.int16)
    auto_xyz = []
    auto_ids = []
    for plant in (int(x) for x in _available_plants()):
        pdir = _plant_dir(plant)
        utm_path = pdir / f"plant_{plant:02d}_{date}_utm.npy"
        if not utm_path.exists():
            continue
        pts = np.load(utm_path).astype(np.float64)
        if len(pts) == 0:
            continue
        auto_xyz.append(pts)
        auto_ids.append(np.full(len(pts), plant, dtype=np.int16))
    if not auto_xyz:
        return plant_id
    auto_xyz = np.vstack(auto_xyz)
    auto_ids = np.concatenate(auto_ids)
    dist, idx = cKDTree(auto_xyz).query(xyz_utm, k=1, workers=-1)
    plant_id[dist < 0.03] = auto_ids[idx[dist < 0.03]]
    return plant_id


def _load_saved_separation(date, n):
    path = _manual_sep_path(date)
    if not path.exists():
        return None, None
    o = np.load(path, allow_pickle=True).item()
    plant_id = np.asarray(o.get("plant_id", []), dtype=np.int16)
    if len(plant_id) != n:
        return None, None
    return plant_id, str(path)


def _load_row_veg(date, seed_auto=True):
    path = _row_veg_path(date)
    o = np.load(path, allow_pickle=True).item()
    if "label" in o:
        keep = np.asarray(o["label"]) == 1
        xyz_utm = np.asarray(o["xyz_utm"], dtype=np.float64)[keep]
    else:
        xyz_utm = np.asarray(o["xyz_utm"], dtype=np.float64)
    if len(xyz_utm) == 0:
        raise RuntimeError(f"no plant points in {path}")
    if "xyz_local" in o and len(o["xyz_local"]) == len(xyz_utm):
        xyz_local = np.asarray(o["xyz_local"], dtype=np.float32)
    else:
        xyz_local = _utm_to_separator_cm(xyz_utm)
    height = (xyz_local[:, 2] - xyz_local[:, 2].min()).astype(np.float32) / 100.0
    plant_id, loaded_from = _load_saved_separation(date, len(xyz_utm))
    if plant_id is None:
        plant_id = _seed_plant_ids_from_auto(xyz_utm, date) if seed_auto else np.full(len(xyz_utm), -1, dtype=np.int16)
        loaded_from = "auto" if seed_auto else None
    display_rgb = _plant_palette(plant_id)
    _set_active_cloud(
        date,
        path,
        xyz_utm,
        height,
        display_rgb,
        len(xyz_utm),
        len(xyz_utm),
        mode="separation",
        xyz_local=xyz_local,
        display_xyz=xyz_local,
        stem_base=None,
        base_markers=_base_markers_local(),
        plant_id=plant_id,
    )
    state["loaded_labels_from"] = loaded_from
    return loaded_from


def _plant_dir(plant):
    return active_dataset.plant_root / f"plant_{int(plant):02d}"


def _plant_paths(plant, date):
    p = int(plant)
    pdir = _plant_dir(p)
    hand_dir = active_dataset.leafstem_root / f"plant_{p:02d}"
    return (
        pdir,
        pdir / f"plant_{p:02d}_{date}.npy",
        pdir / f"plant_{p:02d}_{date}_utm.npy",
        hand_dir / f"handlabel_{p:02d}_{date}.npy",
    )


def _available_plants():
    return active_dataset.plants


def _load_saved_leafstem(plant, date, n):
    _, _, _, hand_path = _plant_paths(plant, date)
    if hand_path.exists():
        o = np.load(hand_path, allow_pickle=True).item()
        otype = np.asarray(o.get("otype", np.zeros(n)), dtype=np.uint8)
        leafid = np.asarray(o.get("leafid", np.zeros(n)), dtype=np.int16)
        if len(otype) == n and len(leafid) == n:
            return otype, leafid, str(hand_path)
    pdir, _, _, _ = _plant_paths(plant, date)
    otype_path = pdir / f"gt_otype_{date}.npy"
    leafid_path = pdir / f"gt_leafid_{date}.npy"
    if otype_path.exists() and leafid_path.exists():
        otype = np.load(otype_path).astype(np.uint8)
        leafid = np.load(leafid_path).astype(np.int16)
        if len(otype) == n and len(leafid) == n:
            return otype, leafid, str(otype_path)
    return np.zeros(n, dtype=np.uint8), np.zeros(n, dtype=np.int16), None


def _stem_base_marker(plant):
    base_path = active_dataset.plant_root / "base_centres.npy"
    if not base_path.exists():
        return None
    bases = np.load(base_path)
    p = int(plant)
    if p < 0 or p >= len(bases):
        return None
    return [float(bases[p, 0]), float(bases[p, 1]), 0.0]


def _leaf_counts():
    if state["leafid"] is None:
        return []
    out = []
    for lid in sorted(int(x) for x in np.unique(state["leafid"]) if int(x) > 0):
        pts = np.flatnonzero((state["otype"] == 2) & (state["leafid"] == lid))
        if len(pts):
            out.append({"id": lid, "points": int(len(pts)), "base_z": float(np.min(state["xyz_local"][pts, 2]))})
    return out


def _plant_id_counts():
    if state["plant_id"] is None:
        return []
    out = []
    for pid in sorted(int(x) for x in np.unique(state["plant_id"]) if int(x) >= 0):
        out.append({"id": pid, "points": int(np.sum(state["plant_id"] == pid))})
    return out


def _separation_export_plants():
    plants = set(range(BASE_PLANT_COUNT))
    if state["plant_id"] is not None:
        plants.update(int(x) for x in np.unique(state["plant_id"]) if int(x) >= 0)
    return sorted(plants)


def _plant_display_rgb(otype, leafid):
    rgb = np.full((len(otype), 3), 0.36, dtype=np.float32)
    rgb[otype == 1] = (0.45, 0.24, 0.08)
    for lid in sorted(int(x) for x in np.unique(leafid) if int(x) > 0):
        rgb[(otype == 2) & (leafid == lid)] = LEAF_PALETTE[(lid - 1) % len(LEAF_PALETTE)]
    return rgb


def _load_handlabel_geometry(hand_path):
    """Restore the exact saved labelling state (points + labels).

    Handlabels embed their own xyz_local/xyz_utm because the user may have
    deleted neighbour-plant points during labelling, so the saved cloud is a
    subset of the on-disk plant_NN_<date>.npy. When that geometry is present we
    must reload from it (not the full on-disk cloud) or the length-based label
    match fails and the plant comes back blank.
    """
    if not hand_path.exists():
        return None
    try:
        o = np.load(hand_path, allow_pickle=True).item()
    except Exception:
        return None
    if not all(k in o for k in ("xyz_local", "xyz_utm", "otype", "leafid")):
        return None
    xl = np.asarray(o["xyz_local"], dtype=np.float32)
    xu = np.asarray(o["xyz_utm"], dtype=np.float64)
    ot = np.asarray(o["otype"], dtype=np.uint8)
    lf = np.asarray(o["leafid"], dtype=np.int16)
    if not (len(xl) == len(xu) == len(ot) == len(lf)) or len(xl) == 0:
        return None
    return xl, xu, ot, lf


def _load_plant(plant, date):
    pdir, local_path, utm_path, hand_path = _plant_paths(plant, date)
    if not local_path.exists():
        raise FileNotFoundError(local_path)
    if not utm_path.exists():
        raise FileNotFoundError(f"{utm_path} missing; run separate_from_labels.py to emit UTM companions")
    embedded = _load_handlabel_geometry(hand_path)
    if embedded is not None:
        xyz_local, xyz_utm, otype, leafid = embedded
        xyz_local = xyz_local.astype(np.float32)
        xyz_utm = xyz_utm.astype(np.float64)
        loaded_from = str(hand_path)
    else:
        xyz_local = np.load(local_path).astype(np.float32)
        xyz_utm = np.load(utm_path).astype(np.float64)
        if xyz_utm.shape != xyz_local.shape:
            raise RuntimeError(f"{utm_path} shape {xyz_utm.shape} does not match {local_path} {xyz_local.shape}")
        otype, leafid, loaded_from = _load_saved_leafstem(plant, date, len(xyz_local))
    display_rgb = _plant_display_rgb(otype, leafid)
    height = (xyz_local[:, 2] - float(np.min(xyz_local[:, 2]))).astype(np.float32) / 100.0
    _set_active_cloud(
        date,
        local_path,
        xyz_utm,
        height,
        display_rgb,
        len(xyz_local),
        len(xyz_local),
        mode="plant",
        plant=f"{int(plant):02d}",
        xyz_local=xyz_local,
        display_xyz=xyz_local,
        otype=otype,
        leafid=leafid,
        stem_base=_stem_base_marker(plant),
        loaded_labels_from=loaded_from,
    )
    return loaded_from


def _ensure_loaded():
    if state["xyz_utm"] is None:
        date = args.date if args.date in active_dataset.dates else (active_dataset.dates[-1] if active_dataset.dates else args.date)
        _load_date(date, args.n)


def _unloaded_status_payload():
    return {
        "mode": state["mode"],
        "date": state["date"],
        "plant": state["plant"],
        "dataset": {
            "root": str(active_dataset.root),
            "plot": active_dataset.plot,
            "raw_dir": None if active_dataset.raw_dir is None else str(active_dataset.raw_dir),
        },
        "n": 0,
        "row_count": 0,
        "counts": {},
        "dates": list(active_dataset.dates),
        "early_dates": list(active_dataset.early_dates),
        "plants": list(_available_plants()),
        "separation_dates": list(active_dataset.separation_dates),
        "leaves": [],
        "plant_instances": [],
        "next_plant_id": None,
        "stem_base": None,
        "base_markers": [],
        "loaded_labels_from": None,
        "export_fullres_default": bool(args.export_fullres),
        "n_target": int(state["n_target"]),
        "crop_parent_count": None,
    }


def _status_payload():
    labels = state["labels"]
    stem_base = None
    next_plant_id = None
    if state["stem_base"] is not None and state["norm_shift"] is not None:
        stem_base = ((np.asarray(state["stem_base"], dtype=np.float64) - state["norm_shift"]) / state["norm_scale"]).tolist()
    base_markers = []
    if state["base_markers"] and state["norm_shift"] is not None:
        base_markers = ((np.asarray(state["base_markers"], dtype=np.float64) - state["norm_shift"]) / state["norm_scale"]).tolist()
    if state["mode"] == "separation":
        plant_id = state["plant_id"]
        counts = {
            "unassigned": int(np.sum(plant_id < 0)),
            "assigned": int(np.sum(plant_id >= 0)),
            "plants": int(len([x for x in np.unique(plant_id) if x >= 0])),
            "unlabeled": int(np.sum(plant_id < 0)),
            "plant": int(np.sum(plant_id >= 0)),
            "ground": 0,
        }
        existing = [int(x) for x in np.unique(plant_id) if int(x) >= 0]
        next_plant_id = (max(existing) + 1) if existing else 0
    elif state["mode"] == "plant":
        otype = state["otype"]
        counts = {
            "unlabeled": int(np.sum(otype == 0)),
            "stem": int(np.sum(otype == 1)),
            "leaf": int(np.sum(otype == 2)),
            "plant": int(np.sum(otype == 1)),
            "ground": int(np.sum(otype == 2)),
        }
    else:
        counts = {
            "unlabeled": int(np.sum(labels == 0)),
            "plant": int(np.sum(labels == 1)),
            "ground": int(np.sum(labels == 2)),
        }
    return {
        "mode": state["mode"],
        "date": state["date"],
        "plant": state["plant"],
        "dataset": {
            "root": str(active_dataset.root),
            "plot": active_dataset.plot,
            "raw_dir": None if active_dataset.raw_dir is None else str(active_dataset.raw_dir),
        },
        "n": int(len(labels)),
        "row_count": int(state["row_count"]),
        "counts": counts,
        "dates": list(active_dataset.dates),
        "early_dates": list(active_dataset.early_dates),
        "plants": list(_available_plants()),
        "separation_dates": list(active_dataset.separation_dates),
        "leaves": _leaf_counts() if state["mode"] == "plant" else [],
        "plant_instances": _plant_id_counts() if state["mode"] == "separation" else [],
        "next_plant_id": next_plant_id,
        "stem_base": stem_base,
        "base_markers": base_markers,
        "loaded_labels_from": state["loaded_labels_from"],
        "export_fullres_default": bool(args.export_fullres),
        "n_target": int(state["n_target"]),
        "crop_parent_count": (
            None if state["crop_parent_count"] is None else int(state["crop_parent_count"])
        ),
    }


def _cloud_payload():
    payload = {
        **_status_payload(),
        "xyz": state["xyz_norm"].reshape(-1).tolist(),
        "rgb": (
            _plant_display_rgb(state["otype"], state["leafid"])
            if state["mode"] == "plant"
            else _plant_palette(state["plant_id"])
            if state["mode"] == "separation"
            else state["display_rgb"]
        ).reshape(-1).tolist(),
        "labels": state["labels"].tolist(),
    }
    if state["mode"] == "plant":
        payload["otype"] = state["otype"].tolist()
        payload["leafid"] = state["leafid"].tolist()
    if state["mode"] == "separation":
        payload["plant_id"] = state["plant_id"].tolist()
    return payload


def _class_rgb(labels):
    rgb = np.zeros((len(labels), 3), dtype=np.uint16)
    rgb[labels == 1] = (0, 65535, 0)
    rgb[labels == 2] = (65535, 0, 0)
    rgb[labels == 0] = (32768, 32768, 32768)
    return rgb


def _target_from_request(data):
    label = int(data.get("label", 1))
    if state["mode"] == "separation":
        target = data.get("target") or {}
        if target.get("kind") in ("eraser", "unassign") or label == 0:
            return "plant_id", -1
        if target.get("kind") == "new_plant":
            existing = [int(x) for x in np.unique(state["plant_id"]) if int(x) >= 0]
            return "plant_id", (max(existing) + 1) if existing else 0
        if target.get("kind") == "plant":
            return "plant_id", int(target.get("plant_id", target.get("plant", 0)))
        return "plant_id", int(data.get("plant_id", max(0, label - 1)))
    if state["mode"] != "plant":
        if label not in (0, 1, 2):
            raise ValueError("label must be 0, 1, or 2")
        return label, 0
    target = data.get("target") or {}
    kind = target.get("kind", "stem" if label == 1 else "eraser" if label == 0 else "leaf")
    if kind == "eraser" or label == 0:
        return 0, 0
    if kind == "stem" or label == 1:
        return 1, 0
    if kind == "new_leaf":
        return 2, int(state["leafid"].max()) + 1
    if kind == "leaf":
        leaf = int(target.get("leafid", data.get("leafid", 0)))
        if leaf <= 0:
            leaf = int(state["leafid"].max()) + 1
        return 2, leaf
    raise ValueError(f"unknown target {kind}")


def _assign_target(idx, otype_value, leaf_value):
    if len(idx) == 0:
        return 0
    if state["mode"] == "separation":
        old = state["plant_id"][idx].copy()
        state["undo"].append(("plant_id", idx, old))
        state["plant_id"][idx] = int(leaf_value)
        state["labels"] = np.where(state["plant_id"] >= 0, state["plant_id"] + 1, 0).astype(np.uint8)
        return int(np.sum(old != state["plant_id"][idx]))
    if state["mode"] != "plant":
        old = state["labels"][idx].copy()
        state["undo"].append(("labels", idx, old))
        state["labels"][idx] = otype_value
        return int(np.sum(old != otype_value))
    old_otype = state["otype"][idx].copy()
    old_leafid = state["leafid"][idx].copy()
    state["undo"].append(("leafstem", idx, old_otype, old_leafid))
    state["otype"][idx] = otype_value
    state["leafid"][idx] = leaf_value if otype_value == 2 else 0
    state["labels"] = state["otype"]
    return int(np.sum((old_otype != state["otype"][idx]) | (old_leafid != state["leafid"][idx])))


def _epsg_geokey_vlr():
    vlr = GeoKeyDirectoryVlr()
    vlr.geo_keys_header.key_directory_version = 1
    vlr.geo_keys_header.key_revision = 1
    vlr.geo_keys_header.minor_revision = 0
    keys = []
    for key_id, value in (
        (GTModelTypeGeoKey.id, ModelTypeProjected),
        (GTRasterTypeGeoKey.id, 1),
        (ProjectedCSTypeGeoKey.id, active_dataset.epsg),
    ):
        key = GeoKeyEntryStruct()
        key.id = key_id
        key.tiff_tag_location = 0
        key.count = 1
        key.value_offset = value
        keys.append(key)
    vlr.geo_keys = keys
    vlr.geo_keys_header.number_of_keys = len(keys)
    return vlr


def _make_las_header(xyz=None, source_header=None):
    if source_header is None:
        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.0001, 0.0001, 0.0001])
        header.offsets = np.floor(xyz.min(axis=0))
    else:
        header = laspy.LasHeader(point_format=3, version=source_header.version)
        header.scales = source_header.scales
        header.offsets = source_header.offsets
    header.vlrs.append(_epsg_geokey_vlr())
    return header


def _write_las(path, xyz, labels, source_header=None):
    header = _make_las_header(xyz=xyz, source_header=source_header)
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.classification = labels.astype(np.uint8)
    rgb = _class_rgb(labels)
    las.red, las.green, las.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    las.write(path)


def _export_sample():
    active_dataset.label_root.mkdir(parents=True, exist_ok=True)
    date = state["date"]
    labels = state["labels"]
    xyz = state["xyz_utm"]
    las_path = active_dataset.label_root / f"{active_dataset.plot}_{date}_2class.las"
    npy_path = active_dataset.label_root / f"{active_dataset.plot}_{date}_2class.npy"
    _write_las(las_path, xyz, labels)
    np.save(
        npy_path,
        {
            "xyz_utm": xyz,
            "label": labels,
            "date": date,
            "plot": active_dataset.plot,
            "epsg": active_dataset.epsg,
            "row_count": state["row_count"],
            "sample_count": len(labels),
        },
    )
    return las_path, npy_path


def _export_fullres():
    date = state["date"]
    out_path = active_dataset.label_root / f"{active_dataset.plot}_{date}_2class_fullres.las"
    tree = cKDTree(state["xyz_utm"])
    total = 0
    labelled = state["labels"]
    with laspy.open(state["raw_path"]) as src:
        header = _make_las_header(source_header=src.header)
        with laspy.open(out_path, mode="w", header=header) as dst:
            for pts in src.chunk_iterator(2_000_000):
                xyz = np.column_stack((np.asarray(pts.x), np.asarray(pts.y), np.asarray(pts.z)))
                _, idx = tree.query(xyz, k=1, workers=-1)
                labels = labelled[idx].astype(np.uint8)
                out = laspy.ScaleAwarePointRecord.zeros(len(xyz), header=header)
                out.x = xyz[:, 0]
                out.y = xyz[:, 1]
                out.z = xyz[:, 2]
                out.classification = labels
                rgb = _class_rgb(labels)
                out.red, out.green, out.blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
                dst.write_points(out)
                total += len(xyz)
    return out_path, total


def _renumber_leaves_by_height():
    leaves = _leaf_counts()
    mapping = {item["id"]: i + 1 for i, item in enumerate(sorted(leaves, key=lambda x: x["base_z"]))}
    if not mapping:
        return mapping
    old_otype = state["otype"].copy()
    old_leafid = state["leafid"].copy()
    state["undo"].append(("leafstem", np.arange(len(state["otype"])), old_otype, old_leafid))
    new_leafid = np.zeros_like(state["leafid"])
    for old, new in mapping.items():
        new_leafid[(state["otype"] == 2) & (state["leafid"] == old)] = new
    state["leafid"] = new_leafid
    state["labels"] = state["otype"]
    return mapping


def _delete_leaf(leaf):
    idx = np.flatnonzero((state["otype"] == 2) & (state["leafid"] == int(leaf)))
    if len(idx) == 0:
        return 0
    _assign_target(idx, 0, 0)
    return int(len(idx))


def _redo_leaf(leaf):
    deleted = _delete_leaf(leaf)
    return deleted


def _write_leafstem_qc(pdir, plant, date):
    colors = _plant_display_rgb(state["otype"], state["leafid"])
    xyz = state["xyz_local"]
    out = pdir / f"handlabel_{plant}_{date}_qc.png"
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, dims, title in ((axs[0], (0, 2), "side"), (axs[1], (0, 1), "top")):
        ax.scatter(xyz[:, dims[0]], xyz[:, dims[1]], s=3, c=colors)
        if state["stem_base"] is not None and title == "top":
            ax.scatter([state["stem_base"][0]], [state["stem_base"][1]], s=80, c="k", marker="+")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title)
    fig.suptitle(f"{active_dataset.plot} plant {plant} {date}")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def _append_leafstem_manifest(pdir, plant, date):
    LEAFSTEM_active_dataset.label_root.mkdir(parents=True, exist_ok=True)
    path = active_dataset.leafstem_root / "manifest.csv"
    row = {
        "plot": active_dataset.plot,
        "plant": plant,
        "date": date,
        "n_leaves": int(len([x for x in np.unique(state["leafid"]) if x > 0])),
        "n_stem_pts": int(np.sum(state["otype"] == 1)),
        "n_leaf_pts": int(np.sum(state["otype"] == 2)),
        "n_unlabeled": int(np.sum(state["otype"] == 0)),
    }
    write_header = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as fh:
        fieldnames = ["plot", "plant", "date", "n_leaves", "n_stem_pts", "n_leaf_pts", "n_unlabeled"]
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)
    return path, row


def _export_plant_labels():
    if state["mode"] != "plant":
        raise RuntimeError("active cloud is not in plant mode")
    plant = state["plant"]
    date = state["date"]
    pdir, _, _, hand_path = _plant_paths(plant, date)
    pdir.mkdir(parents=True, exist_ok=True)
    hand_path.parent.mkdir(parents=True, exist_ok=True)
    gt_otype = pdir / f"gt_otype_{date}.npy"
    gt_leafid = pdir / f"gt_leafid_{date}.npy"
    np.save(gt_otype, state["otype"].astype(np.uint8))
    np.save(gt_leafid, state["leafid"].astype(np.int16))
    np.save(
        hand_path,
        {
            "xyz_utm": state["xyz_utm"].astype(np.float64),
            "xyz_local": state["xyz_local"].astype(np.float32),
            "otype": state["otype"].astype(np.uint8),
            "leafid": state["leafid"].astype(np.int16),
            "plant": plant,
            "date": date,
            "plot": active_dataset.plot,
            "epsg": active_dataset.epsg,
        },
    )
    manifest_path, manifest_row = _append_leafstem_manifest(pdir, plant, date)
    qc_path = _write_leafstem_qc(pdir, plant, date)
    return gt_otype, gt_leafid, hand_path, manifest_path, qc_path, manifest_row


def _backup_current_separation(date):
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_root = active_dataset.plant_root / "auto_backup" / f"{date}_{stamp}"
    backup_root.mkdir(parents=True, exist_ok=True)
    backed = []
    plants = set(_separation_export_plants())
    if active_dataset.plant_root.exists():
        for path in active_dataset.plant_root.glob("plant_*"):
            if not path.is_dir():
                continue
            try:
                plants.add(int(path.name.split("_", 1)[1]))
            except (IndexError, ValueError):
                continue
    for plant in sorted(plants):
        src_dir = _plant_dir(plant)
        if not src_dir.exists():
            continue
        dst_dir = backup_root / f"plant_{plant:02d}"
        dst_dir.mkdir(parents=True, exist_ok=True)
        for name in (
            f"plant_{plant:02d}_{date}.npy",
            f"plant_{plant:02d}_{date}_utm.npy",
            f"gt_otype_{date}.npy",
            f"gt_leafid_{date}.npy",
            f"handlabel_{plant:02d}_{date}.npy",
            f"handlabel_{plant:02d}_{date}_qc.png",
        ):
            src = src_dir / name
            if src.exists():
                shutil.copy2(src, dst_dir / name)
                backed.append(str(src))
    return backup_root, backed


def _write_separation_qc(date):
    out_dir = _leafstem_label_dir()
    out = out_dir / f"plantsep_{active_dataset.plot}_{date}_qc.png"
    colors = _plant_palette(state["plant_id"])
    xyz = state["xyz_local"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.scatter(xyz[:, 0], xyz[:, 1], s=1.0, c=colors)
    bases = np.asarray(state["base_markers"] or [], dtype=np.float64)
    if len(bases):
        ax.scatter(bases[:, 0], bases[:, 1], s=70, c="k", marker="+")
        for i, b in enumerate(bases):
            ax.text(b[0], b[1], str(i), fontsize=8, color="k")
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{active_dataset.plot} {date} manual plant separation")
    fig.tight_layout()
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def _export_manual_separation():
    if state["mode"] != "separation":
        raise RuntimeError("active cloud is not in row plant-separation mode")
    date = state["date"]
    backup_root, backed = _backup_current_separation(date)
    leaf_label_warnings = []
    exported_plants = _separation_export_plants()
    for plant in exported_plants:
        pdir = _plant_dir(plant)
        pdir.mkdir(parents=True, exist_ok=True)
        keep = state["plant_id"] == plant
        np.save(pdir / f"plant_{plant:02d}_{date}.npy", state["xyz_local"][keep].astype(np.float32))
        np.save(pdir / f"plant_{plant:02d}_{date}_utm.npy", state["xyz_utm"][keep].astype(np.float64))
        for name in (
            pdir / f"gt_otype_{date}.npy",
            pdir / f"gt_leafid_{date}.npy",
            pdir / f"handlabel_{plant:02d}_{date}.npy",
        ):
            if name.exists():
                leaf_label_warnings.append(str(name))
    sep_path = _manual_sep_path(date)
    np.save(
        sep_path,
        {
            "xyz_utm": state["xyz_utm"].astype(np.float64),
            "xyz_local": state["xyz_local"].astype(np.float32),
            "plant_id": state["plant_id"].astype(np.int16),
            "date": date,
            "plot": active_dataset.plot,
            "epsg": active_dataset.epsg,
        },
    )
    qc_path = _write_separation_qc(date)
    return backup_root, backed, sep_path, qc_path, leaf_label_warnings


def _reset_active_cloud():
    state.update(
        date=None,
        plant=None,
        raw_path=None,
        xyz_utm=None,
        xyz_local=None,
        height=None,
        display_rgb=None,
        xyz_norm=None,
        norm_shift=None,
        norm_scale=None,
        labels=None,
        otype=None,
        leafid=None,
        plant_id=None,
        prompt_coords=[],
        prompt_labels=[],
        current_mask=None,
        undo=[],
        encoder=None,
        pc_xyz=None,
        pc_features=None,
        model_idx=None,
        model_xyz_norm=None,
        row_count=0,
        crop_parent_count=None,
        stem_base=None,
        base_markers=None,
        real_count=0,
        loaded_labels_from=None,
    )


def _dataset_options():
    roots = [active_dataset.root]
    if DEFAULT_DATASET_ROOT.exists():
        roots.append(DEFAULT_DATASET_ROOT.resolve())
    roots.append(LEGACY_PLANT_ROOT.resolve())
    parent = args.datasets_parent or active_dataset.root.parent
    if parent.exists():
        roots.extend(p.resolve() for p in parent.iterdir() if p.is_dir())
    seen = set()
    out = []
    for root in roots:
        root = root.resolve()
        if root in seen:
            continue
        seen.add(root)
        try:
            ds = _resolve_dataset(root, None)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        out.append(
            {
                "root": str(ds.root),
                "plot": ds.plot,
                "dates": list(ds.dates),
                "plants": list(ds.plants),
                "active": ds.root == active_dataset.root and ds.plot == active_dataset.plot,
            }
        )
    return out


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/static/<path:path>")
def static_server(path):
    return app.send_static_file(path)


@app.route("/mesh/<path:path>")
def mesh_server(path):
    return app.send_static_file(f"models/{path}")


@app.route("/datasets")
def datasets_server():
    return jsonify({"datasets": _dataset_options(), "active": _unloaded_status_payload()["dataset"]})


@app.route("/set_dataset", methods=["POST"])
def set_dataset_server():
    global active_dataset
    data = request.get_json(silent=True) or {}
    root = Path(data.get("root", active_dataset.root))
    plot = data.get("plot")
    try:
        active_dataset = _resolve_dataset(root, plot)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc), "datasets": _dataset_options()}), 400
    _reset_active_cloud()
    return jsonify(
        {
            "status": "dataset_set",
            "dataset": {
                "root": str(active_dataset.root),
                "plot": active_dataset.plot,
                "raw_dir": None if active_dataset.raw_dir is None else str(active_dataset.raw_dir),
            },
            "dates": list(active_dataset.dates),
            "early_dates": list(active_dataset.early_dates),
            "separation_dates": list(active_dataset.separation_dates),
            "plants": list(active_dataset.plants),
        }
    )


@app.route("/dates")
def dates_server():
    if state["xyz_utm"] is None and active_dataset.raw_dir is None:
        return jsonify(_unloaded_status_payload())
    _ensure_loaded()
    return jsonify(_status_payload())


@app.route("/pointcloud/<path:path>")
def pointcloud_server(path):
    _ensure_loaded()
    return jsonify(_cloud_payload())


@app.route("/load_date/<date>", methods=["POST"])
def load_date_server(date):
    data = request.get_json(silent=True) or {}
    n_target = int(data.get("n", state["n_target"] or args.n))
    if n_target < 2048:
        return jsonify({"error": "density must be at least 2048 points"}), 400
    try:
        _load_date(date, n_target)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc), "dates": list(active_dataset.dates)}), 400
    return jsonify(_cloud_payload())


@app.route("/load_plant", methods=["POST"])
def load_plant_server():
    data = request.get_json(silent=True) or {}
    plant = data.get("plant_id", data.get("plant", "06"))
    date = data.get("date", "230619")
    try:
        loaded_from = _load_plant(plant, date)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc), "plants": list(_available_plants()), "dates": list(active_dataset.dates)}), 400
    payload = _cloud_payload()
    payload["loaded_labels_from"] = loaded_from
    return jsonify(payload)


@app.route("/load_row_veg", methods=["POST"])
def load_row_veg_server():
    data = request.get_json(silent=True) or {}
    date = data.get("date", "230619")
    seed_auto = bool(data.get("seed_auto", True))
    try:
        loaded_from = _load_row_veg(date, seed_auto=seed_auto)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        return jsonify({"error": str(exc), "dates": list(active_dataset.dates)}), 400
    payload = _cloud_payload()
    payload["loaded_labels_from"] = loaded_from
    return jsonify(payload)


@app.route("/renumber_leaves", methods=["POST"])
def renumber_leaves():
    _ensure_loaded()
    if state["mode"] != "plant":
        return jsonify({"error": "renumber is only available in plant mode"}), 400
    mapping = _renumber_leaves_by_height()
    return jsonify({"status": "renumbered", "mapping": mapping, **_cloud_payload()})


@app.route("/delete_leaf", methods=["POST"])
def delete_leaf():
    _ensure_loaded()
    if state["mode"] != "plant":
        return jsonify({"error": "delete_leaf is only available in plant mode"}), 400
    data = request.get_json(silent=True) or {}
    deleted = _delete_leaf(int(data.get("leafid", 0)))
    return jsonify({"status": "deleted", "deleted": deleted, **_cloud_payload()})


@app.route("/redo_leaf", methods=["POST"])
def redo_leaf():
    _ensure_loaded()
    if state["mode"] != "plant":
        return jsonify({"error": "redo_leaf is only available in plant mode"}), 400
    data = request.get_json(silent=True) or {}
    leaf = int(data.get("leafid", 0))
    if leaf <= 0:
        return jsonify({"error": "leafid must be positive", **_status_payload()}), 400
    deleted = _redo_leaf(leaf)
    return jsonify({"status": "redo", "deleted": deleted, "target_leafid": leaf, **_cloud_payload()})


@app.route("/clear", methods=["POST"])
def clear():
    state["prompt_coords"] = []
    state["prompt_labels"] = []
    state["current_mask"] = None
    return jsonify({"status": "cleared", **_status_payload()})


@app.route("/reset_labels", methods=["POST"])
def reset_labels():
    if state["mode"] == "separation":
        changed = np.arange(len(state["plant_id"]))
        state["undo"].append(("plant_id", changed, state["plant_id"].copy()))
        state["plant_id"][:] = -1
        state["labels"][:] = 0
    elif state["mode"] == "plant":
        changed = np.flatnonzero((state["otype"] != 0) | (state["leafid"] != 0))
        state["undo"].append(("leafstem", changed, state["otype"][changed].copy(), state["leafid"][changed].copy()))
        state["otype"][:] = 0
        state["leafid"][:] = 0
        state["labels"] = state["otype"]
    else:
        old = state["labels"].copy()
        changed = np.flatnonzero(old)
        state["undo"].append(("labels", changed, old[changed]))
        state["labels"][:] = 0
    state["prompt_coords"] = []
    state["prompt_labels"] = []
    state["current_mask"] = None
    return jsonify({"status": "reset", **_cloud_payload()})


@app.route("/assign_indices", methods=["POST"])
def assign_indices():
    _ensure_loaded()
    data = request.get_json(silent=True) or {}
    try:
        label, leaf = _target_from_request(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    idx = np.asarray(data.get("indices", []), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(state["labels"]))]
    idx = np.unique(idx)
    if len(idx) == 0:
        return jsonify({"error": "no points selected"}), 400

    changed = _assign_target(idx, label, leaf)
    state["prompt_coords"] = []
    state["prompt_labels"] = []
    state["current_mask"] = None
    return jsonify(
        {
            "status": "assigned",
            "label": label,
            "target_leafid": leaf,
            "target_plant_id": int(leaf) if state["mode"] == "separation" else None,
            "changed": int(changed),
            "labels": state["labels"].tolist(),
            "otype": state["otype"].tolist() if state["mode"] == "plant" else None,
            "leafid": state["leafid"].tolist() if state["mode"] == "plant" else None,
            "plant_id": state["plant_id"].tolist() if state["mode"] == "separation" else None,
            **_status_payload(),
        }
    )


@app.route("/crop_indices", methods=["POST"])
def crop_indices():
    _ensure_loaded()
    if state["mode"] == "plant":
        return jsonify({"error": "crop is only available in full-row mode", **_status_payload()}), 400
    data = request.get_json(silent=True) or {}
    idx = np.asarray(data.get("indices", []), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(state["labels"]))]
    idx = np.unique(idx)
    if len(idx) < 2048:
        return jsonify({"error": "crop must contain at least 2048 points"}), 400

    snapshot = _cloud_snapshot()
    undo_stack = state["undo"] + [("cloud", snapshot)]
    parent_count = len(state["labels"])
    raw_path = state["raw_path"]
    xyz_utm = state["xyz_utm"][idx].copy()
    height = state["height"][idx].copy()
    labels = state["labels"][idx].copy()
    hn = np.clip((height - height.min()) / (np.ptp(height) + 1e-9), 0, 1)
    display_rgb = cm.viridis(hn)[:, :3].astype(np.float32)

    _set_active_cloud(
        state["date"],
        raw_path,
        xyz_utm,
        height,
        display_rgb,
        state["row_count"],
        state["n_target"],
        crop_parent_count=parent_count,
    )
    state["labels"] = labels
    state["undo"] = undo_stack
    return jsonify({"status": "cropped", **_cloud_payload()})


@app.route("/delete_indices", methods=["POST"])
def delete_indices():
    _ensure_loaded()
    data = request.get_json(silent=True) or {}
    idx = np.asarray(data.get("indices", []), dtype=np.int64)
    idx = idx[(idx >= 0) & (idx < len(state["labels"]))]
    idx = np.unique(idx)
    if len(idx) == 0:
        return jsonify({"error": "no points selected"}), 400
    min_remaining = 2048 if state["mode"] == "row" else 1
    if len(state["labels"]) - len(idx) < min_remaining:
        return jsonify({"error": f"delete would leave fewer than {min_remaining} points"}), 400

    snapshot = _cloud_snapshot()
    undo_stack = state["undo"] + [("cloud", snapshot)]
    keep = np.ones(len(state["labels"]), dtype=bool)
    keep[idx] = False
    labels = state["labels"][keep].copy()
    mode = state["mode"]
    xyz_local = state["xyz_local"][keep].copy()
    _set_active_cloud(
        state["date"],
        state["raw_path"],
        state["xyz_utm"][keep].copy(),
        state["height"][keep].copy(),
        state["display_rgb"][keep].copy(),
        state["row_count"],
        state["n_target"],
        crop_parent_count=state["crop_parent_count"],
        mode=mode,
        plant=state["plant"],
        xyz_local=xyz_local,
        display_xyz=xyz_local if mode in ("plant", "separation") else None,
        otype=state["otype"][keep].copy(),
        leafid=state["leafid"][keep].copy(),
        plant_id=state["plant_id"][keep].copy(),
        stem_base=state["stem_base"],
        base_markers=state["base_markers"],
        loaded_labels_from=state["loaded_labels_from"],
    )
    state["labels"] = labels
    state["undo"] = undo_stack
    return jsonify(
        {
            "status": "deleted",
            "deleted": int(len(idx)),
            **_cloud_payload(),
        }
    )


@app.route("/reset_crop", methods=["POST"])
def reset_crop():
    _ensure_loaded()
    if state["mode"] == "plant":
        _load_plant(state["plant"], state["date"])
    else:
        _load_date(state["date"], state["n_target"])
    return jsonify({"status": "crop_reset", **_cloud_payload()})


@app.route("/segment", methods=["POST"])
def segment():
    _ensure_loaded()
    data = request.get_json()
    prompt_point = data["prompt_point"]
    prompt_label = int(bool(data["prompt_label"]))
    active_label = int(data.get("active_label", 0))
    target = data.get("target") or {}
    use_label_context = bool(data.get("use_label_context", True))
    use_height_prior = bool(data.get("use_height_prior", args.use_height_prior))
    height_threshold = float(data.get("height_threshold", args.plant_height_threshold))
    flood_points = int(data.get("flood_points", 0) or 0)

    state["prompt_coords"].append(prompt_point)
    state["prompt_labels"].append(prompt_label)

    context_coords, context_labels = (
        _context_prompts_for_target(active_label, target=target) if use_label_context else ([], [])
    )
    all_prompt_coords = context_coords + state["prompt_coords"]
    all_prompt_labels = context_labels + state["prompt_labels"]

    prompt_coords = torch.from_numpy(np.asarray(all_prompt_coords, dtype=np.float32)).cuda()[None, ...]
    prompt_labels = torch.from_numpy(np.asarray(all_prompt_labels, dtype=np.int64)).cuda()[None, ...]

    start = time.time()
    with torch.no_grad():
        masks, iou = _predict_with_cached_encoder(prompt_coords, prompt_labels, multimask_output=True)
    best = int(torch.argmax(iou[0]))
    model_count = len(state["model_idx"])
    model_mask = (masks[0, best] > 0).detach().cpu().numpy()[:model_count]
    if model_count == state["real_count"]:
        mask = model_mask
    else:
        _, nn_idx = cKDTree(state["model_xyz_norm"]).query(state["xyz_norm"], k=1, workers=-1)
        mask = model_mask[nn_idx]
    height_prior_count = 0
    if use_height_prior and active_label == 1:
        before = int(mask.sum())
        mask &= state["height"] >= height_threshold
        height_prior_count = before - int(mask.sum())
    elif use_height_prior and active_label == 2:
        before = int(mask.sum())
        mask &= state["height"] < height_threshold
        height_prior_count = before - int(mask.sum())
    flood_dropped = 0
    flood_kept = 0
    if flood_points > 0 and int(mask.sum()) > flood_points:
        prompt_np = np.asarray(prompt_point, dtype=np.float32)
        mask_idx = np.flatnonzero(mask)
        dist2 = np.sum((state["xyz_norm"][mask_idx] - prompt_np) ** 2, axis=1)
        keep_idx = mask_idx[np.argsort(dist2)[:flood_points]]
        flood_dropped = int(mask.sum()) - int(len(keep_idx))
        flood_kept = int(len(keep_idx))
        mask[:] = False
        mask[keep_idx] = True
    state["current_mask"] = mask
    elapsed = time.time() - start
    print(
        f"segment {state['date']}: prompts={len(state['prompt_coords'])} "
        f"context={len(context_coords)} active={active_label} "
        f"model={model_count:,}/{state['real_count']:,} "
        f"height_prior={use_height_prior} threshold={height_threshold:.3f} removed={height_prior_count:,} "
        f"flood_points={flood_points} flood_dropped={flood_dropped:,} "
        f"mask={int(mask.sum()):,}/{len(mask):,} iou={float(iou[0, best]):.3f} {elapsed:.3f}s",
        flush=True,
    )
    return jsonify(
        {
            "seg": mask.tolist(),
            "iou": float(iou[0, best]),
            "elapsed": elapsed,
            "context_prompts": len(context_coords),
            "height_prior_removed": height_prior_count,
            "flood_points": flood_points,
            "flood_kept": flood_kept,
            "flood_dropped": flood_dropped,
            "plant_height_threshold": height_threshold,
            "use_height_prior": use_height_prior,
        }
    )


@app.route("/commit", methods=["POST"])
def commit():
    _ensure_loaded()
    data = request.get_json(silent=True) or {}
    try:
        label, leaf = _target_from_request(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    overwrite = bool(data.get("overwrite", True))
    if state["current_mask"] is None:
        return jsonify({"error": "no active mask to commit"}), 400

    if state["mode"] == "separation":
        idx = np.flatnonzero(state["current_mask"])
        before = len(idx)
        ghosted = {int(x) for x in data.get("ghosted_plant_ids", [])}
        if ghosted:
            idx = idx[~np.isin(state["plant_id"][idx], list(ghosted))]
        skipped = before - len(idx)
        changed = _assign_target(idx, label, leaf)
        state["prompt_coords"] = []
        state["prompt_labels"] = []
        state["current_mask"] = None
        return jsonify(
            {
                "status": "committed",
                "label": label,
                "target_plant_id": int(leaf),
                "changed": int(changed),
                "skipped": int(skipped),
                "replaced": 0,
                "labels": state["labels"].tolist(),
                "plant_id": state["plant_id"].tolist(),
                **_status_payload(),
            }
        )

    if state["mode"] == "plant":
        idx = np.flatnonzero(state["current_mask"])
        changed = _assign_target(idx, label, leaf)
        state["prompt_coords"] = []
        state["prompt_labels"] = []
        state["current_mask"] = None
        return jsonify(
            {
                "status": "committed",
                "label": label,
                "target_leafid": leaf,
                "changed": int(changed),
                "skipped": 0,
                "replaced": 0,
                "labels": state["labels"].tolist(),
                "otype": state["otype"].tolist(),
                "leafid": state["leafid"].tolist(),
                **_status_payload(),
            }
        )

    raw_policy = data.get("layer_policy")
    if raw_policy is None:
        if overwrite:
            layer_policy = {1: "override", 2: "override"}
        else:
            layer_policy = {1: "protect", 2: "protect"}
            if label in (1, 2):
                layer_policy[label] = "override"
    else:
        layer_policy = {}
        for layer in (1, 2):
            value = raw_policy.get(str(layer), raw_policy.get(layer, "protect"))
            layer_policy[layer] = "override" if value == "override" else "protect"

    labels = state["labels"]
    mask_idx = np.flatnonzero(state["current_mask"])
    mask_old_labels = labels[mask_idx]
    if label == 0:
        set_allowed = (mask_old_labels != 0) & np.array(
            [layer_policy.get(int(old), "protect") == "override" for old in mask_old_labels]
        )
        set_idx = mask_idx[set_allowed]
        clear_idx = np.array([], dtype=np.int64)
    else:
        set_allowed = (mask_old_labels == 0) | np.array(
            [layer_policy.get(int(old), "protect") == "override" for old in mask_old_labels]
        )
        set_idx = mask_idx[set_allowed]
        if layer_policy.get(label, "protect") == "override":
            outside_mask = np.ones(len(labels), dtype=bool)
            outside_mask[mask_idx] = False
            clear_idx = np.flatnonzero((labels == label) & outside_mask)
        else:
            clear_idx = np.array([], dtype=np.int64)

    target = {}
    for i in set_idx:
        target[int(i)] = label
    for i in clear_idx:
        target[int(i)] = 0

    if target:
        idx = np.fromiter(target.keys(), dtype=np.int64)
        new_values = np.fromiter(target.values(), dtype=np.uint8)
        changed_mask = labels[idx] != new_values
        idx = idx[changed_mask]
        new_values = new_values[changed_mask]
    else:
        idx = np.array([], dtype=np.int64)
        new_values = np.array([], dtype=np.uint8)

    protected_mask = (mask_old_labels != 0) & np.array(
        [layer_policy.get(int(old), "protect") == "protect" for old in mask_old_labels]
    )
    if label in (1, 2):
        protected_mask &= mask_old_labels != label
    skipped = int(np.sum(protected_mask))
    replaced = int(len(clear_idx))

    if len(idx) == 0:
        state["prompt_coords"] = []
        state["prompt_labels"] = []
        state["current_mask"] = None
        return jsonify(
            {
                "status": "protected",
                "label": label,
                "changed": 0,
                "skipped": skipped,
                "replaced": replaced,
                "layer_policy": layer_policy,
                "labels": state["labels"].tolist(),
                **_status_payload(),
            }
        )
    old = state["labels"][idx].copy()
    state["undo"].append(("labels", idx, old))
    state["labels"][idx] = new_values
    state["prompt_coords"] = []
    state["prompt_labels"] = []
    state["current_mask"] = None
    return jsonify(
        {
            "status": "committed",
            "label": label,
            "changed": int(len(idx)),
            "skipped": skipped,
            "replaced": replaced,
            "layer_policy": layer_policy,
            "labels": state["labels"].tolist(),
            **_status_payload(),
        }
    )


@app.route("/undo", methods=["POST"])
def undo():
    _ensure_loaded()
    if not state["undo"]:
        return jsonify(
            {
                "status": "empty",
                "cloud_changed": False,
                "labels": state["labels"].tolist(),
                **_status_payload(),
            }
        )
    item = state["undo"].pop()
    cloud_changed = False
    if len(item) == 2 and item[0] == "cloud":
        _restore_cloud_snapshot(item[1], state["undo"])
        cloud_changed = True
    elif len(item) == 4 and item[0] == "leafstem":
        _, idx, old_otype, old_leafid = item
        state["otype"][idx] = old_otype
        state["leafid"][idx] = old_leafid
        state["labels"] = state["otype"]
    elif len(item) == 3 and item[0] == "plant_id":
        _, idx, old = item
        state["plant_id"][idx] = old
        state["labels"] = np.where(state["plant_id"] >= 0, state["plant_id"] + 1, 0).astype(np.uint8)
    else:
        if len(item) == 3 and item[0] == "labels":
            _, idx, old = item
        else:
            idx, old = item
        state["labels"][idx] = old
    state["prompt_coords"] = []
    state["prompt_labels"] = []
    state["current_mask"] = None
    if cloud_changed:
        return jsonify({"status": "undone", "cloud_changed": True, **_cloud_payload()})
    return jsonify(
        {
            "status": "undone",
            "cloud_changed": False,
            "labels": state["labels"].tolist(),
            "otype": state["otype"].tolist() if state["mode"] == "plant" else None,
            "leafid": state["leafid"].tolist() if state["mode"] == "plant" else None,
            "plant_id": state["plant_id"].tolist() if state["mode"] == "separation" else None,
            **_status_payload(),
        }
    )


@app.route("/export", methods=["POST"])
def export():
    _ensure_loaded()
    if state["mode"] == "separation":
        try:
            backup_root, backed, sep_path, qc_path, leaf_label_warnings = _export_manual_separation()
        except RuntimeError as exc:
            return jsonify({"error": str(exc), **_status_payload()}), 400
        payload = {
            "status": "exported",
            "backup_root": str(backup_root),
            "backed_up_files": backed,
            "separation": str(sep_path),
            "qc_png": str(qc_path),
            "leaf_label_warnings": leaf_label_warnings,
            **_cloud_payload(),
        }
        print(
            f"export separation {active_dataset.plot} {state['date']}: backup={backup_root} sep={sep_path} qc={qc_path}",
            flush=True,
        )
        return jsonify(payload)
    if state["mode"] == "plant":
        try:
            gt_otype, gt_leafid, hand_path, manifest_path, qc_path, manifest_row = _export_plant_labels()
        except RuntimeError as exc:
            return jsonify({"error": str(exc), **_status_payload()}), 400
        payload = {
            "status": "exported",
            "gt_otype": str(gt_otype),
            "gt_leafid": str(gt_leafid),
            "handlabel": str(hand_path),
            "manifest": str(manifest_path),
            "qc_png": str(qc_path),
            "manifest_row": manifest_row,
            **_cloud_payload(),
        }
        print(
            f"export plant {active_dataset.plot} plant {state['plant']} {state['date']}: "
            f"otype={gt_otype} leafid={gt_leafid} handlabel={hand_path} qc={qc_path}",
            flush=True,
        )
        return jsonify(payload)
    data = request.get_json(silent=True) or {}
    fullres = bool(data.get("fullres", args.export_fullres))
    if fullres and state["crop_parent_count"] is not None:
        return (
            jsonify(
                {
                    "error": "full-res export is disabled while cropped; reset crop first",
                    **_status_payload(),
                }
            ),
            400,
        )
    las_path, npy_path = _export_sample()
    payload = {
        "status": "exported",
        "las": str(las_path),
        "npy": str(npy_path),
        "sample_count": int(len(state["labels"])),
        "row_count": int(state["row_count"]),
        "sample_dropped_count": int(max(state["row_count"] - len(state["labels"]), 0)),
        "fullres": None,
        **_status_payload(),
    }
    if fullres:
        full_path, total = _export_fullres()
        payload["fullres"] = str(full_path)
        payload["fullres_count"] = int(total)
        payload["fullres_propagated_count"] = int(total)
        payload["fullres_dropped_count"] = 0
    print(f"export {payload}", flush=True)
    return jsonify(payload)


if __name__ == "__main__":
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)

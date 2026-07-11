import json
import re
import shutil
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


DATE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def _base(path):
    xyz = np.load(path)
    low = xyz[:, 2] <= np.quantile(xyz[:, 2], 0.12)
    return np.median(xyz[low], axis=0)


def _instances(plant_root):
    by_date = {}
    for path in sorted(Path(plant_root).glob("plant_*/plant_*_*_utm.npy")):
        match = re.fullmatch(r"plant_(\d+)_(\d{6})_utm", path.stem)
        if match:
            plant, date = int(match.group(1)), match.group(2)
            by_date.setdefault(date, []).append((plant, _base(path)))
    return by_date


def preview_identity_plan(plant_root, plot, reference_date, max_distance_m):
    by_date = _instances(plant_root)
    if reference_date not in by_date:
        raise ValueError(f"reference date {reference_date} has no plant instances")
    if max_distance_m <= 0:
        raise ValueError("identity distance must be positive")

    reference = sorted(by_date[reference_date])
    observations = {plant: [base] for plant, base in reference}
    mappings = {reference_date: {plant: plant for plant, _ in reference}}
    distances = {reference_date: {plant: 0.0 for plant, _ in reference}}
    next_id = max(observations, default=-1) + 1

    reference_count = len(reference)
    dates = sorted(
        (date for date in by_date if date != reference_date),
        key=lambda date: (
            abs(len(by_date[date]) - reference_count),
            abs(int(date) - int(reference_date)),
            date,
        ),
    )
    for date in dates:
        sources = sorted(by_date[date])
        canonical = sorted(observations)
        source_y = np.asarray([base[1] for _, base in sources])
        anchor_y = np.asarray([np.median(np.asarray(observations[plant])[:, 1]) for plant in canonical])
        cost = np.abs(source_y[:, None] - anchor_y[None, :])
        rows, cols = linear_sum_assignment(cost)
        accepted = {
            row: (canonical[col], float(cost[row, col]))
            for row, col in zip(rows, cols)
            if cost[row, col] <= max_distance_m
        }
        date_map = {}
        date_dist = {}
        for row, (source, base) in enumerate(sources):
            if row in accepted:
                plant, distance = accepted[row]
            else:
                plant, distance = next_id, None
                next_id += 1
            date_map[source] = plant
            date_dist[source] = distance
            observations.setdefault(plant, []).append(base)
        mappings[date] = date_map
        distances[date] = date_dist

    coverage = {
        plant: sorted(date for date, mapping in mappings.items() if plant in mapping.values())
        for plant in sorted(observations)
    }
    all_dates = sorted(by_date)
    continuous = [plant for plant, dates in coverage.items() if dates == all_dates]
    anchors = {
        plant: float(np.median(np.asarray(points)[:, 1]))
        for plant, points in observations.items()
    }
    return {
        "schema_version": 1,
        "plot": plot,
        "identity": "fixed physical position along the crop row",
        "reference_date": reference_date,
        "max_base_distance_m": float(max_distance_m),
        "locked_plant_ids": sorted(observations),
        "continuous_plant_ids": continuous,
        "coverage": {str(k): v for k, v in coverage.items()},
        "anchor_y_utm": {str(k): v for k, v in anchors.items()},
        "source_id_to_canonical": {
            date: {str(source): target for source, target in sorted(mapping.items())}
            for date, mapping in sorted(mappings.items())
        },
        "match_distance_m": {
            date: {str(source): distance for source, distance in sorted(values.items())}
            for date, values in sorted(distances.items())
        },
    }


def _mapped_name(name, old, new):
    return name.replace(f"plant_{old:02d}", f"plant_{new:02d}").replace(
        f"handlabel_{old:02d}", f"handlabel_{new:02d}"
    )


def _remap_tree(source_root, target_root, mappings):
    target_root.mkdir(parents=True)
    for path in source_root.iterdir():
        if not path.is_dir() or not re.fullmatch(r"plant_\d+", path.name):
            if path.name != "plant_identity.json":
                target = target_root / path.name
                shutil.copytree(path, target) if path.is_dir() else shutil.copy2(path, target)
            continue
        old = int(path.name[-2:])
        for item in path.iterdir():
            match = DATE_RE.search(item.name)
            if not match:
                targets = [mapping[old] for mapping in mappings.values() if old in mapping]
                if not targets:
                    continue
                new = Counter(targets).most_common(1)[0][0]
                destination = target_root / f"plant_{new:02d}" / _mapped_name(item.name, old, new)
                destination.parent.mkdir(parents=True, exist_ok=True)
                if item.name == "manual_chains.json":
                    chains = json.loads(item.read_text())
                    for chain in chains:
                        chain["obs"] = {
                            date: value
                            for date, value in chain.get("obs", {}).items()
                            if mappings.get(date, {}).get(old) == new
                        }
                    chains = [chain for chain in chains if chain["obs"]]
                    destination.write_text(json.dumps(chains, indent=1) + "\n")
                else:
                    shutil.copy2(item, destination)
                continue
            if old not in mappings.get(match.group(1), {}):
                continue
            new = mappings[match.group(1)][old]
            destination = target_root / f"plant_{new:02d}" / _mapped_name(item.name, old, new)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise RuntimeError(f"identity remap collision: {destination}")
            if item.name.startswith("handlabel_") and item.suffix == ".npy":
                hand = np.load(item, allow_pickle=True).item()
                hand["plant"] = f"{new:02d}"
                np.save(destination, hand, allow_pickle=True)
            else:
                shutil.copy2(item, destination)


def _separation_from_stage1(stage1_path, source_root, date, mapping, plot, epsg):
    obj = np.load(stage1_path, allow_pickle=True).item()
    xyz_all = np.asarray(obj["xyz_utm"], dtype=np.float64)
    keep = np.asarray(obj.get("label", np.ones(len(xyz_all), dtype=np.int8))) == 1
    xyz_utm = xyz_all[keep]
    xyz_local_all = np.asarray(obj.get("xyz_local", xyz_all), dtype=np.float32)
    xyz_local = xyz_local_all[keep] if len(xyz_local_all) == len(xyz_all) else xyz_local_all
    plant_id = np.full(len(xyz_utm), -1, dtype=np.int16)
    tree = cKDTree(xyz_utm)
    claimed = np.zeros(len(xyz_utm), dtype=bool)
    exact = defaultdict(deque)
    for index, point in enumerate(xyz_utm):
        exact[point.tobytes()].append(index)
    for source, canonical in sorted(mapping.items()):
        path = source_root / f"plant_{source:02d}" / f"plant_{source:02d}_{date}_utm.npy"
        points = np.load(path)
        indices = []
        for point in points:
            candidates = exact.get(point.tobytes())
            index = candidates.popleft() if candidates else None
            if index is None:
                nearby = tree.query_ball_point(point, 1e-5)
                index = next((candidate for candidate in nearby if not claimed[candidate]), None)
            if index is None:
                raise RuntimeError(f"{date} plant {source:02d}: points no longer match stage1")
            indices.append(index)
        index = np.asarray(indices, dtype=np.int64)
        plant_id[index] = canonical
        claimed[index] = True
    return {
        "xyz_utm": xyz_utm,
        "xyz_local": xyz_local,
        "plant_id": plant_id,
        "date": date,
        "plot": plot,
        "epsg": epsg,
    }


def apply_identity_plan(dataset_root, plot, plan, epsg=25832):
    dataset_root = Path(dataset_root)
    plant_root = dataset_root / "stage2_plants_isolated" / plot
    leafstem_root = dataset_root / "stage3_leafstem_labeled" / plot
    stage1_root = dataset_root / "stage1_ground_removed" / plot
    if (plant_root / "plant_identity.json").exists():
        raise ValueError("plant identity is already locked")
    mappings = {
        date: {int(source): int(target) for source, target in values.items()}
        for date, values in plan["source_id_to_canonical"].items()
    }

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = dataset_root / "identity_backups" / f"{plot}_{stamp}"
    backup.mkdir(parents=True)
    shutil.copytree(plant_root, backup / "stage2")
    shutil.copytree(leafstem_root, backup / "stage3")

    stage2_tmp = plant_root.parent / f".{plot}_identity_tmp"
    stage3_tmp = leafstem_root.parent / f".{plot}_identity_tmp"
    shutil.rmtree(stage2_tmp, ignore_errors=True)
    shutil.rmtree(stage3_tmp, ignore_errors=True)
    try:
        _remap_tree(plant_root, stage2_tmp, mappings)
        _remap_tree(leafstem_root, stage3_tmp, mappings)
        for date, mapping in mappings.items():
            separation = _separation_from_stage1(
                stage1_root / f"{date}.npy", plant_root, date, mapping, plot, epsg
            )
            np.save(stage3_tmp / f"plantsep_{plot}_{date}.npy", separation, allow_pickle=True)
        plan = dict(plan)
        plan.pop("match_distance_m", None)
        plan["backup"] = str(backup)
        (stage2_tmp / "plant_identity.json").write_text(json.dumps(plan, indent=2) + "\n")
        old_stage2 = plant_root.parent / f".{plot}_identity_old"
        old_stage3 = leafstem_root.parent / f".{plot}_identity_old"
        shutil.rmtree(old_stage2, ignore_errors=True)
        shutil.rmtree(old_stage3, ignore_errors=True)
        plant_root.rename(old_stage2)
        leafstem_root.rename(old_stage3)
        stage2_tmp.rename(plant_root)
        stage3_tmp.rename(leafstem_root)
        shutil.rmtree(old_stage2)
        shutil.rmtree(old_stage3)
    except Exception:
        shutil.rmtree(stage2_tmp, ignore_errors=True)
        shutil.rmtree(stage3_tmp, ignore_errors=True)
        raise
    return backup

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def base_y(path):
    xyz = np.load(path)
    low = xyz[:, 2] <= np.quantile(xyz[:, 2], 0.12)
    return float(np.median(xyz[low, 1]))


def main():
    parser = argparse.ArgumentParser(description="Verify fixed plant identities across dates")
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("plot")
    args = parser.parse_args()

    plant_root = args.dataset_root / "stage2_plants_isolated" / args.plot
    label_root = args.dataset_root / "stage3_leafstem_labeled" / args.plot
    identity = json.loads((plant_root / "plant_identity.json").read_text())
    limit = 2 * float(identity["max_base_distance_m"])

    for plant in identity["continuous_plant_ids"]:
        dates = identity["coverage"][str(plant)]
        assert len(dates) == len(identity["source_id_to_canonical"]), (plant, dates)
        ys = [base_y(plant_root / f"plant_{plant:02d}" / f"plant_{plant:02d}_{date}_utm.npy") for date in dates]
        assert max(ys) - min(ys) <= limit, (plant, max(ys) - min(ys), limit)

    label_count = 0
    for date in identity["source_id_to_canonical"]:
        separation = np.load(label_root / f"plantsep_{args.plot}_{date}.npy", allow_pickle=True).item()
        ids = np.asarray(separation["plant_id"])
        files = list(plant_root.glob(f"plant_*/plant_*_{date}_utm.npy"))
        assert len(files) == len(np.unique(ids[ids >= 0])), date
        assert sum(len(np.load(path)) for path in files) == int(np.sum(ids >= 0)), date

    for path in label_root.glob("plant_*/handlabel_*.npy"):
        hand = np.load(path, allow_pickle=True).item()
        plant, date = int(hand["plant"]), str(hand["date"])
        cloud = np.load(plant_root / f"plant_{plant:02d}" / f"plant_{plant:02d}_{date}_utm.npy")
        distance, _ = cKDTree(cloud).query(np.asarray(hand["xyz_utm"]), k=1, workers=-1)
        assert np.all(distance <= 1e-5), path
        label_count += 1

    print(
        f"PASS {args.plot}: {len(identity['locked_plant_ids'])} physical identities, "
        f"{len(identity['continuous_plant_ids'])} all-date identities, {label_count} label files attached"
    )


if __name__ == "__main__":
    main()

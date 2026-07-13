"""Prepare raw Pheno4D maize scans for manual Point-SAM labelling."""

import argparse
import json
import re
import tempfile
from pathlib import Path

import laspy
import numpy as np


SCAN_RE = re.compile(r"^M(?P<plant>\d+)_(?P<date>\d{4})(?:_a)?\.txt$")


def _write_raw(path: Path, xyz_m: np.ndarray) -> None:
    las = laspy.create(point_format=3, file_version="1.2")
    las.header.scales = np.full(3, 0.0001)
    las.header.offsets = np.floor(xyz_m.min(axis=0))
    las.x, las.y, las.z = xyz_m.T
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(path)


def _point_count(path: Path) -> int:
    with laspy.open(path) as source:
        return source.header.point_count


def _ensure_raw(scans, output: Path, plot: str) -> None:
    for _, date, source in scans:
        path = output / "raw" / plot / f"{date}.las"
        if not path.exists():
            xyz_mm = _load_xyz(source, np.float64)
            _write_raw(path, xyz_mm / 1000.0)


def _scan_files(source: Path, year: str):
    scans = []
    for path in source.glob("*.txt"):
        match = SCAN_RE.match(path.name)
        if match:
            scans.append((int(match["plant"]), f"{year}{match['date']}", path))
    scans.sort(key=lambda item: item[1])
    if not scans:
        raise ValueError(f"no Pheno4D scans found in {source}")
    plants = {plant for plant, _, _ in scans}
    if len(plants) != 1:
        raise ValueError(f"expected one plant, found {sorted(plants)}")
    dates = [date for _, date, _ in scans]
    if len(dates) != len(set(dates)):
        raise ValueError("duplicate scan dates")
    return scans


def _load_xyz(path: Path, dtype) -> np.ndarray:
    xyz = np.loadtxt(path, dtype=dtype, usecols=(0, 1, 2))
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError(f"invalid XYZ data in {path}")
    return xyz


def _prepare_plant(source: Path, output: Path, year: str) -> dict:
    scans = _scan_files(source, year)
    dates = [date for _, date, _ in scans]
    plant = scans[0][0]
    plot = f"Pheno4D_M{plant:02d}"
    _ensure_raw(scans, output, plot)
    plot_root = output / "stage2_plants_isolated" / plot
    plot_root.mkdir(parents=True, exist_ok=True)
    (plot_root / "plant_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "plot": plot,
                "identity": "single isolated Pheno4D plant",
                "reference_date": dates[0],
                "locked_plant_ids": [0],
                "continuous_plant_ids": [0],
                "coverage": {"0": dates},
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "source": str(source.resolve()),
        "plot": plot,
        "scans": [
            {
                "date": date,
                "source": path.name,
                "points": _point_count(output / "raw" / plot / f"{date}.las"),
            }
            for _, date, path in scans
        ],
    }


def prepare(source: Path, output: Path, year: str = "18") -> Path:
    sources = sorted(path for path in source.glob("Maize[0-9][0-9]") if path.is_dir()) or [source]
    manifests = [_prepare_plant(plant_source, output, year) for plant_source in sources]
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset.json").write_text(
        json.dumps({"epsg": None, "coordinate_system": "scanner-local metres"}, indent=2) + "\n"
    )
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "ignored_columns": [3, 4],
                "plants": sorted(manifests, key=lambda item: item["plot"]),
            },
            indent=2,
        )
        + "\n"
    )
    return output


def self_test() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, output = root / "raw", root / "dataset"
        source.mkdir()
        np.savetxt(source / "M01_0313_a.txt", [[10, 20, 30, 99, 98], [20, 40, 50, 97, 96]])
        prepare(source, output)
        assert not (output / "stage2_plants_isolated/Pheno4D_M01/plant_00").exists()
        assert json.loads((output / "dataset.json").read_text())["epsg"] is None
        assert json.loads((output / "manifest.json").read_text())["ignored_columns"] == [3, 4]
        with laspy.open(output / "raw/Pheno4D_M01/180313.las") as raw:
            assert raw.header.point_count == 2
            assert not np.any(raw.read().classification)
        prepare(source, output)
        assert not (output / "stage2_plants_isolated/Pheno4D_M01/plant_00").exists()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--year", default="18", help="two-digit year prefix (default: 18)")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.source and args.out:
        print(prepare(args.source, args.out, args.year))
    else:
        parser.error("source and --out are required unless --self-test is used")

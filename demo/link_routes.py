"""Cross-date leaf LINK routes for the FP4D Point-SAM labeller.

Trace the SAME physical leaf across dates to build trustworthy persistent
identities (chains) + a basal->apical rank, written to
``<leafstem_root>/plant_<NN>/manual_chains.json`` and consumed by
``relink_leaf_identity.py`` (which overrides its unreliable automatic DAG
tracker when that file is present). Self-contained; wire into app.py with:

    from link_routes import register_link_routes
    register_link_routes(app, lambda: active_dataset, _npy_dict)

Per-date ``leafid`` is trusted (the height-labelling filter); this only fixes
the cross-date linking + ranking, which the automatic tracker gets wrong on
dense/occluded plants.
"""
import json
import math
import re
import sys

import numpy as np
from flask import jsonify, request

RELINK_PATH = "/home/lukas/pointr"          # optional: automatic seed source
DEFAULT_CLOUD_POINTS = 50_000
MAX_CLOUD_POINTS = 60_000


def leaf_signatures(dataset, npy_dict, plant):
    """{date: [{leafid, cx, cy, az, length, n}, ...]} collar-centred top-down.

    cx,cy = leaf centroid minus the stem collar (xy, cm); az = compass angle
    (deg) -- a physical leaf keeps ~the same az across dates, the identity cue."""
    p = int(plant)
    pdir = dataset.leafstem_root / f"plant_{p:02d}"
    out = {}
    for date in dataset.dates:
        f = pdir / f"handlabel_{p:02d}_{date}.npy"
        if not f.exists():
            continue
        a = npy_dict(f)
        if not a:
            continue
        xyz = np.asarray(a["xyz_local"], float)
        otype = np.asarray(a["otype"])
        leafid = np.asarray(a["leafid"])
        stem = xyz[otype == 1]
        collar = stem[:, :2].mean(0) if len(stem) else xyz[:, :2].mean(0)
        leaves = []
        for lid in sorted(int(x) for x in np.unique(leafid) if int(x) > 0):
            P = xyz[(otype == 2) & (leafid == lid)]
            if len(P) < 8:
                continue
            cxy = P[:, :2].mean(0) - collar
            Pc = P - P.mean(0)
            try:
                _, _, V = np.linalg.svd(Pc, full_matrices=False)
                length = float(np.ptp(Pc @ V[0]))
            except np.linalg.LinAlgError:
                length = 0.0
            leaves.append(dict(
                leafid=lid, cx=round(float(cxy[0]), 2), cy=round(float(cxy[1]), 2),
                az=round(math.degrees(math.atan2(cxy[1], cxy[0])), 1),
                length=round(length, 1), n=int(len(P)),
            ))
        out[date] = leaves
    return out


def _handlabel_path(dataset, plant, date):
    p = int(plant)
    return dataset.leafstem_root / f"plant_{p:02d}" / f"handlabel_{p:02d}_{date}.npy"


def _sample_evenly(indices, n):
    indices = np.asarray(indices, dtype=np.int64)
    if n >= len(indices):
        return indices
    return indices[np.linspace(0, len(indices) - 1, int(n), dtype=np.int64)]


def _cloud_groups(otype, leafid):
    groups = []
    stem = np.flatnonzero(otype == 1)
    if len(stem):
        groups.append(stem)
    for lid in sorted(int(x) for x in np.unique(leafid) if int(x) > 0):
        pts = np.flatnonzero((otype == 2) & (leafid == lid))
        if len(pts):
            groups.append(pts)
    known = (otype == 1) | ((otype == 2) & (leafid > 0))
    other = np.flatnonzero(~known)
    if len(other):
        groups.append(other)
    return groups


def _stratified_cloud_indices(otype, leafid, max_points):
    n = len(otype)
    target = min(int(max_points), n)
    if target <= 0:
        return np.empty(0, dtype=np.int64)
    if n <= target:
        return np.arange(n, dtype=np.int64)

    groups = _cloud_groups(otype, leafid)
    if not groups:
        return _sample_evenly(np.arange(n), target)

    sizes = np.asarray([len(g) for g in groups], dtype=np.int64)
    if len(groups) >= target:
        keep_groups = sorted(np.argsort(-sizes)[:target])
        sampled = [_sample_evenly(groups[i], 1) for i in keep_groups]
        return np.sort(np.concatenate(sampled)).astype(np.int64)

    floor = min(300, max(1, target // max(1, len(groups) * 4)))
    alloc = np.minimum(sizes, np.maximum(floor, np.rint(target * sizes / sizes.sum()).astype(np.int64)))

    while int(alloc.sum()) > target:
        can_trim = np.where(alloc > 1, alloc, -1)
        i = int(np.argmax(can_trim))
        if can_trim[i] <= 1:
            break
        alloc[i] -= 1
    while int(alloc.sum()) < target:
        deficit = sizes - alloc
        i = int(np.argmax(deficit))
        if deficit[i] <= 0:
            break
        alloc[i] += 1

    sampled = [_sample_evenly(group, keep) for group, keep in zip(groups, alloc) if keep > 0]
    return np.sort(np.concatenate(sampled)).astype(np.int64)


def leaf_clouds(dataset, npy_dict, plant, max_points=DEFAULT_CLOUD_POINTS):
    """Per-date isolated plant clouds sampled for the 3D linker."""
    p = int(plant)
    out = {}
    for date in dataset.dates:
        path = _handlabel_path(dataset, p, date)
        if not path.exists():
            continue
        data = npy_dict(path)
        if not data:
            continue
        missing = [k for k in ("xyz_local", "otype", "leafid") if k not in data]
        if missing:
            raise KeyError(f"{path} missing {', '.join(missing)}")
        xyz = np.asarray(data["xyz_local"], dtype=np.float32)
        otype = np.asarray(data["otype"], dtype=np.uint8)
        leafid = np.asarray(data["leafid"], dtype=np.int16)
        if not (len(xyz) == len(otype) == len(leafid)):
            raise ValueError(f"{path} has mismatched xyz/otype/leafid lengths")
        sel = _stratified_cloud_indices(otype, leafid, max_points)
        out[date] = {
            "xyz": np.round(xyz[sel], 3).tolist(),
            "otype": otype[sel].tolist(),
            "leafid": leafid[sel].tolist(),
            "full_count": int(len(xyz)),
            "sample_count": int(len(sel)),
            "source": str(path),
        }
    return out


def _manual_path(dataset, plant):
    return dataset.leafstem_root / f"plant_{int(plant):02d}" / "manual_chains.json"


def seed_chains(dataset, plant, fresh=False):
    """Chains to seed the UI: saved manual_chains.json if present (unless fresh),
    else the automatic relink tracker (best-effort), else []. Each =
    {rank, obs:{date:leafid}}. fresh=True forces the automatic guess (ignore save)
    and runs the CPD arbiter (TrackPlant3D-style) on the tracker's theft failures --
    this is the one-click "Auto-link" draft, slower (~seconds) but a better start."""
    mp = _manual_path(dataset, plant)
    if mp.exists() and not fresh:
        try:
            return sorted(json.load(open(mp)), key=lambda e: e.get("rank", 1 << 30))
        except (ValueError, OSError):
            pass
    try:
        if RELINK_PATH not in sys.path:
            sys.path.insert(0, RELINK_PATH)
        import relink_leaf_identity as rl
        chains, raw_to_pid = rl.build_persistent_ids(arbitrate=fresh)
        brm = rl.biological_rank_map(arbitrate=fresh)
        pid_rank = {}
        for (d, lid), r in brm.items():
            pid_rank[raw_to_pid[(d, lid)]] = r
        order = sorted(range(len(chains)), key=lambda i: pid_rank[i + 1])
        return [{"rank": r + 1, "obs": {d: int(l) for d, l in chains[i].items()}}
                for r, i in enumerate(order)]
    except Exception as exc:                                    # noqa: BLE001 - best-effort seed
        print(f"[link] automatic seed unavailable ({exc}); starting empty", flush=True)
        return []


def save_chains(dataset, plant, chains):
    """Normalise (drop empty, re-rank 1..N basal->apical) and write manual_chains.json."""
    clean = []
    for i, ch in enumerate(chains, 1):
        obs = {str(d): int(l) for d, l in (ch.get("obs") or {}).items()}
        if obs:
            clean.append({"rank": int(ch.get("rank", i)), "obs": obs})
    clean.sort(key=lambda e: e["rank"])
    for i, e in enumerate(clean, 1):
        e["rank"] = i
    mp = _manual_path(dataset, plant)
    mp.parent.mkdir(parents=True, exist_ok=True)
    json.dump(clean, open(mp, "w"), indent=1)
    print(f"[link] wrote {mp} ({len(clean)} chains)", flush=True)
    return mp, len(clean)


def available_plants(dataset):
    """Plants under leafstem_root that have >=2 dates of hand-labels (linkable)."""
    root = getattr(dataset, "leafstem_root", None)
    out = []
    if not root or not root.exists():
        return out
    for pdir in sorted(root.glob("plant_*")):
        m = re.match(r"plant_(\d+)$", pdir.name)
        if not m:
            continue
        p = int(m.group(1))
        n = sum(1 for d in dataset.dates if (pdir / f"handlabel_{p:02d}_{d}.npy").exists())
        if n >= 2:
            out.append({"plant": f"{p:02d}", "n_dates": n})
    return out


def register_link_routes(app, get_dataset, npy_dict):
    @app.route("/link/plants")
    def link_plants():
        return jsonify({"plants": available_plants(get_dataset())})

    @app.route("/link/clouds")
    def link_clouds():
        plant = request.args.get("plant", "01")
        try:
            max_points = int(request.args.get("max_points", DEFAULT_CLOUD_POINTS))
        except (TypeError, ValueError):
            return jsonify({"error": "max_points must be an integer"}), 400
        max_points = max(1, min(MAX_CLOUD_POINTS, max_points))
        try:
            clouds = leaf_clouds(get_dataset(), npy_dict, plant, max_points=max_points)
        except (OSError, KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        dates = [d for d in get_dataset().dates if d in clouds]
        return jsonify({"plant": f"{int(plant):02d}", "dates": dates, "clouds": clouds})

    @app.route("/link/leaves")
    def link_leaves():
        plant = request.args.get("plant", "01")
        try:
            sigs = leaf_signatures(get_dataset(), npy_dict, plant)
        except (OSError, KeyError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        dates = [d for d in get_dataset().dates if d in sigs]
        return jsonify({"plant": plant, "dates": dates, "leaves": sigs})

    @app.route("/link/chains")
    def link_chains():
        plant = request.args.get("plant", "01")
        fresh = request.args.get("fresh") in ("1", "true", "yes")
        return jsonify({"plant": plant, "chains": seed_chains(get_dataset(), plant, fresh=fresh),
                        "has_manual": _manual_path(get_dataset(), plant).exists()})

    @app.route("/link/save", methods=["POST"])
    def link_save():
        data = request.get_json(silent=True) or {}
        plant = str(data.get("plant", "01"))
        try:
            mp, n = save_chains(get_dataset(), plant, data.get("chains", []))
        except (OSError, ValueError) as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"status": "saved", "path": str(mp), "n": n})

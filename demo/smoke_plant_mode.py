#!/usr/bin/env python3
"""Smoke-check plant mode against a running isolated-Pheno4D labeler."""

import argparse
import json
from urllib.request import Request, urlopen


def post(base_url, path, payload):
    request = Request(
        base_url.rstrip("/") + path,
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
    )
    return json.load(urlopen(request))


parser = argparse.ArgumentParser()
parser.add_argument("date")
parser.add_argument("--url", default="http://127.0.0.1:5056")
parser.add_argument("--plant", default="00")
args = parser.parse_args()

load = {"plant_id": args.plant, "date": args.date, "n": 2048}
cloud = post(args.url, "/load_plant", load)
assert cloud["mode"] == "plant" and cloud["plant"] == args.plant
assert cloud["counts"]["unlabeled"] > 0
seed = next(i for i, value in enumerate(cloud["otype"]) if value == 0)
flood = post(
    args.url,
    "/point_flood",
    {"seed_index": seed, "distance_cm": 1, "max_points": 25, "protect_existing": True},
)
assert all(cloud["otype"][index] == 0 for index in flood["indices"])
leaf = post(
    args.url,
    "/assign_indices",
    {
        "indices": flood["indices"],
        "label": 2,
        "target": {"kind": "new_leaf"},
    },
)
assert any(item["id"] == leaf["target_leafid"] for item in leaf["leaves"])
post(args.url, "/load_plant", load)  # discard the in-memory test assignment
print("plant-mode fallback: OK")

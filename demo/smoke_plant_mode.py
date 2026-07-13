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
leaf = post(
    args.url,
    "/assign_indices",
    {"indices": [0], "label": 2, "target": {"kind": "new_leaf"}},
)
assert any(item["id"] == leaf["target_leafid"] for item in leaf["leaves"])
post(args.url, "/load_plant", load)  # discard the in-memory test assignment
print("plant-mode fallback: OK")

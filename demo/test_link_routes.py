import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from flask import Flask

sys.path.insert(0, str(Path(__file__).parent))
from link_routes import _link_context, leaf_clouds, register_link_routes  # noqa: E402


def npy_dict(path):
    return np.load(path, allow_pickle=True).item()


class LinkRouteSafetyTest(unittest.TestCase):
    def test_cloud_attachment_height_uses_nearest_stem_point(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plant_dir = root / "plant_02"
            plant_dir.mkdir()
            np.save(
                plant_dir / "handlabel_02_230606.npy",
                {
                    "xyz_local": np.array([
                        [0, 0, 0], [0, 0, 5], [0, 0, 15],
                        [0.1, 0, 15], [10, 0, -5],
                        [0.1, 0, 5], [10, 0, 6],
                    ]),
                    "otype": np.array([1, 1, 1, 2, 2, 2, 2]),
                    "leafid": np.array([0, 0, 0, 1, 1, 2, 2]),
                },
            )
            dataset = SimpleNamespace(leafstem_root=root, dates=("230606",))

            cloud = leaf_clouds(dataset, npy_dict, "02")["230606"]

            self.assertEqual(cloud["attachment_z"], {1: 15.0, 2: 5.0})

    def test_noncanonical_seed_and_save_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            leafstem_root = root / "stage3_leafstem_labeled" / "Plot04"
            plant_dir = leafstem_root / "plant_02"
            plant_dir.mkdir(parents=True)
            np.save(
                plant_dir / "handlabel_02_230606.npy",
                {"xyz_local": np.zeros((3, 3)), "otype": np.array([1, 2, 2]), "leafid": np.array([0, 1, 1])},
            )
            dataset = SimpleNamespace(
                root=root,
                plot="Plot04",
                leafstem_root=leafstem_root,
                dates=("230606", "230613"),
            )
            app = Flask(__name__)
            register_link_routes(app, lambda: dataset, npy_dict)
            client = app.test_client()

            loaded = client.get("/link/chains?plant=02").get_json()
            self.assertEqual(loaded["chains"], [])
            self.assertFalse(loaded["auto_available"])
            context = loaded["context"]
            self.assertEqual(client.get("/link/chains?plant=02&fresh=1").get_json()["chains"], [])
            other = SimpleNamespace(**{**dataset.__dict__, "leafstem_root": root / "other"})
            self.assertNotEqual(context, _link_context(other, "02"))

            self.assertEqual(client.post("/link/save", json=[]).status_code, 400)
            foreign = {"plant": "02", "context": context, "chains": [{"rank": 1, "obs": {"230711": 1}}]}
            self.assertEqual(client.post("/link/save", json=foreign).status_code, 400)
            invalid_leaf = {"plant": "02", "context": context, "chains": [{"rank": 1, "obs": {"230606": 2}}]}
            self.assertEqual(client.post("/link/save", json=invalid_leaf).status_code, 400)
            stale = {"plant": "02", "context": "old-dataset", "chains": []}
            self.assertEqual(client.post("/link/save", json=stale).status_code, 409)

            valid = {"plant": "02", "context": context, "chains": [{"rank": 7, "obs": {"230606": 1}}]}
            self.assertEqual(client.post("/link/save", json=valid).status_code, 200)
            self.assertEqual(client.get("/link/chains?plant=02").get_json()["chains"],
                             [{"rank": 1, "obs": {"230606": 1}}])


if __name__ == "__main__":
    unittest.main()

import argparse
from pathlib import Path
from pprint import pprint
import json
from tqdm import tqdm

import torch

from utils.tar import extract, compress
from geometry.reconstruction import perspective_reconstruction


EMPTY_VALUE = {
    "people": [
        {
            "person_id": [-1],
            "pose_keypoints_2d": [0.0] * 25 * 3,
            "face_keypoints_2d": [0.0] * 70 * 3,
            "hand_left_keypoints_2d": [0.0] * 21 * 3,
            "hand_right_keypoints_2d": [0.0] * 21 * 3,
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }
    ]
}


def f(x):
    x = x["pose_keypoints_2d"]
    x = torch.tensor(x).reshape(-1, 3, 1)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_2d_dir", type=Path, required=True)
    parser.add_argument("--poses_2d_suffix", type=str, default=".tar.xz")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--poses_3d", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--min_views", type=int, default=4)
    parser.add_argument("--max_iterations", type=int, default=500)
    parser.add_argument("--pattern", type=str, default="([0-9]{12})+")
    args = parser.parse_args()
    pprint(vars(args))

    with args.calibration.open() as file:
        cameras = json.load(file)

    poses_2d = {}
    for path in sorted(args.poses_2d_dir.glob("*" + args.poses_2d_suffix)):
        view = path.name.split(".")[0]
        poses_2d[view] = {}
        for key, value in extract(path, pattern=args.pattern).items():
            if len(value["people"]) == 0:
                poses_2d[view][key] = EMPTY_VALUE
            else:
                if len(value["people"]) == 1:
                    poses_2d[view][key] = value
                else:
                    cumulated_confidences = []
                    for potential_value in value["people"]:
                        cumulated_confidence = f(potential_value)[:, -1, :].sum()
                        cumulated_confidences.append(cumulated_confidence)
                    cumulated_confidences = torch.stack(cumulated_confidences, dim=0)
                    index = torch.argmax(cumulated_confidences).tolist()
                    poses_2d[view][key] = {"people": [value["people"][index]]}

    views = sorted(poses_2d.keys())
    keys = [v.keys() for v in poses_2d.values()]
    keys = sorted(set(keys[0]).intersection(*keys[1:]))

    parts = ["pose", "face", "hand_left", "hand_right"]
    poses_3d = {}
    for part in parts:
        p = [
            [poses_2d[view][key]["people"][0][f"{part}_keypoints_2d"] for key in keys]
            for view in views
        ]
        R = torch.tensor([cameras[view]["R"] for view in views])
        t = torch.tensor([cameras[view]["t"] for view in views])
        K = torch.tensor([cameras[view]["K"] for view in views])
        dist_coef = torch.tensor([cameras[view]["dist_coef"] for view in views])

        p = torch.tensor(p).reshape(len(views), len(keys), -1, 3, 1)
        p, c = p.split([2, 1], dim=-2)
        v, n, j = p.shape[:3]
        p = p.reshape(v, n * j, 2, 1)
        c = c.reshape(v, n * j, 1, 1)

        P, C = perspective_reconstruction(
            p=p,
            c=c,
            R=R,
            t=t,
            K=K,
            dist_coef=dist_coef,
            threshold=args.threshold,
            min_views=args.min_views,
            max_iterations=args.max_iterations,
            verbose=True,
        )

        P = P.reshape(n, j, 3, 1)
        C = C.reshape(n, j, 1, 1)

        P = torch.cat([P, C], dim=-2)
        poses_3d[part] = P.flatten(start_dim=-3).tolist()

    temp = {}
    for i, key in enumerate(tqdm(keys)):
        temp[key] = {
            "people": [{f"{part}_keypoints_3d": poses_3d[part][i] for part in parts}]
        }
    poses_3d = temp

    compress(args.poses_3d, poses_3d)

import argparse
from pathlib import Path
from pprint import pprint
import json
from tqdm import tqdm

import torch

from utils.tar import extract, compress
from skeleton.BODY_135 import EDGES

from aniposelib.cameras import Camera, CameraGroup
from scipy.spatial.transform import Rotation as R


EMPTY_VALUE = {
    "people": [
        {
            "person_id": [-1],
            "pose_keypoints_2d": [0.0] * 135 * 3,
            "face_keypoints_2d": [],
            "hand_left_keypoints_2d": [],
            "hand_right_keypoints_2d": [],
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }
    ]
}


def f(x):
    x = x["pose_keypoints_2d"]
    x = torch.tensor(x).reshape(135, 3, 1)
    return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poses_2d_dir", type=Path, required=True)
    parser.add_argument("--poses_2d_suffix", type=str, default=".tar.xz")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--poses_3d", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
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
    keys = [poses_2d[view].keys() for view in poses_2d]
    keys = sorted(set(keys[0]).intersection(*keys[1:]))

    p = [
        [poses_2d[view][key]["people"][0]["pose_keypoints_2d"] for key in keys]
        for view in views
    ]

    camera_group = []
    for view, camera in cameras.items():
        if view in views:
            camera_group.append(
                Camera(
                    matrix=camera["K"],
                    dist=camera["dist_coef"],
                    size=camera["resolution"],
                    rvec=R.from_matrix(camera["R"]).as_rotvec(),
                    tvec=camera["t"],
                    name=view,
                )
            )
    camera_group = CameraGroup(camera_group)

    p = torch.tensor(p).reshape(len(views), len(keys), -1, 3)
    p, c = p.split([2, 1], dim=-1)
    v, n, j = p.shape[:3]
    p[(c < args.threshold).expand_as(p)] = float("nan")
    P = camera_group.triangulate_optim(
        points=p.numpy(),
        constraints=EDGES,
        verbose=True,
    )
    P = torch.from_numpy(P)
    P = P.reshape(n, j, 3, 1)
    P[P.isnan()] = 0.0

    c = c.reshape(v, n * j, 1, 1)
    m = c > args.threshold
    C = torch.zeros(n * j, 1, 1)
    for i in range(n * j):
        c_ = c[:, i, :, :]
        m_ = m[:, i, :, :].squeeze()
        if m_.sum() < 2:
            m_[:] = 0.0
        else:
            C[i, :, :] = c_[m_, :, :].mean(dim=-3)
    C = C.reshape(n, j, 1, 1)

    P = torch.cat([P, C], dim=-2)
    P = P.flatten(start_dim=-3).tolist()

    poses_3d = {}
    for i, key in enumerate(tqdm(keys)):
        poses_3d[key] = {"people": [{"pose_keypoints_3d": P[i]}]}

    compress(args.poses_3d, poses_3d)

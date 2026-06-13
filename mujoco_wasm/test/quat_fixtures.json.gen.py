"""Generate numeric fixtures for quat_utils.test.js by calling scipy.spatial.transform.Rotation
on 50 deterministic random quaternions. Run once; output is committed."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    rng = np.random.default_rng(42)
    cases = []
    for i in range(50):
        # Random unit quat in xyzw via random axis + angle
        v = rng.normal(size=3)
        v /= np.linalg.norm(v)
        a = rng.uniform(-np.pi, np.pi)
        h = a / 2
        qXyzw = np.array([v[0] * np.sin(h), v[1] * np.sin(h), v[2] * np.sin(h), np.cos(h)])
        qWxyz = qXyzw[[3, 0, 1, 2]]
        # Vector to rotate
        vec = rng.normal(size=3) * 2.5
        # Reference outputs
        rotated = Rotation.from_quat(qXyzw).apply(vec)
        # Yaw extraction (using both wxyz and xyzw conventions; should match)
        qw, qx, qy, qz = qWxyz
        yaw_ref = float(
            np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        )
        # Composition: a * b
        v2 = rng.normal(size=3)
        v2 /= np.linalg.norm(v2)
        a2 = rng.uniform(-np.pi, np.pi)
        h2 = a2 / 2
        qB = np.array([v2[0] * np.sin(h2), v2[1] * np.sin(h2), v2[2] * np.sin(h2), np.cos(h2)])
        qProd = (Rotation.from_quat(qXyzw) * Rotation.from_quat(qB)).as_quat()
        # Inverse
        qInv = Rotation.from_quat(qXyzw).inv().as_quat()
        # Matrix
        mat = Rotation.from_quat(qXyzw).as_matrix().reshape(-1).tolist()
        cases.append({
            "qXyzw": qXyzw.tolist(),
            "qWxyz": qWxyz.tolist(),
            "vec": vec.tolist(),
            "rotated": rotated.tolist(),
            "yawWxyz": yaw_ref,
            "qB": qB.tolist(),
            "qProdXyzw": qProd.tolist(),
            "qInvXyzw": qInv.tolist(),
            "matRowMajor": mat,
        })

    # Yaw-only construction
    yaw_cases = []
    for ang in [0.0, 0.5, -1.2, np.pi - 0.01, -np.pi + 0.01]:
        q = Rotation.from_euler("z", ang).as_quat()
        yaw_cases.append({"angle": ang, "qXyzw": q.tolist()})

    # SE2 align — random anchors
    se2_cases = []
    for _ in range(10):
        v1 = rng.normal(size=3); v1 /= np.linalg.norm(v1)
        a1 = rng.uniform(-np.pi, np.pi); h1 = a1 / 2
        qFromXyzw = np.array([v1[0] * np.sin(h1), v1[1] * np.sin(h1), v1[2] * np.sin(h1), np.cos(h1)])
        qFromWxyz = qFromXyzw[[3, 0, 1, 2]]
        v2 = rng.normal(size=3); v2 /= np.linalg.norm(v2)
        a2 = rng.uniform(-np.pi, np.pi); h2 = a2 / 2
        qToXyzw = np.array([v2[0] * np.sin(h2), v2[1] * np.sin(h2), v2[2] * np.sin(h2), np.cos(h2)])
        qToWxyz = qToXyzw[[3, 0, 1, 2]]
        pFrom = rng.normal(size=3) * 1.0
        pTo = rng.normal(size=3) * 1.0
        # Reference compute (mirrors stitch_core.compute_se2_align_root)
        def yaw_from_quat_wxyz(q):
            qw, qx, qy, qz = q
            return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
        yawFrom = yaw_from_quat_wxyz(qFromWxyz)
        yawTo = yaw_from_quat_wxyz(qToWxyz)
        rDelta = Rotation.from_euler("z", yawTo - yawFrom).as_quat()
        t = pTo - Rotation.from_quat(rDelta).apply(pFrom)
        # Apply to a probe body
        probePos = rng.normal(size=3) * 2.0
        probeQuatXyzw = np.array([0.1, 0.2, 0.3, 0.9]) / np.linalg.norm(np.array([0.1, 0.2, 0.3, 0.9]))
        probeQuatWxyz = probeQuatXyzw[[3, 0, 1, 2]]
        outPos = Rotation.from_quat(rDelta).apply(probePos) + t
        outQuatXyzw = (Rotation.from_quat(rDelta) * Rotation.from_quat(probeQuatXyzw)).as_quat()
        outQuatWxyz = outQuatXyzw[[3, 0, 1, 2]]
        se2_cases.append({
            "anchorPosFrom": pFrom.tolist(),
            "anchorQuatFromWxyz": qFromWxyz.tolist(),
            "anchorPosTo": pTo.tolist(),
            "anchorQuatToWxyz": qToWxyz.tolist(),
            "rDeltaXyzw": rDelta.tolist(),
            "t": t.tolist(),
            "probePos": probePos.tolist(),
            "probeQuatWxyz": probeQuatWxyz.tolist(),
            "outPos": outPos.tolist(),
            "outQuatWxyz": outQuatWxyz.tolist(),
        })

    out = {"random": cases, "yaw": yaw_cases, "se2": se2_cases}
    out_path = Path(__file__).parent / "quat_fixtures.json"
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path} with {len(cases)} random + {len(yaw_cases)} yaw + {len(se2_cases)} se2 cases")


if __name__ == "__main__":
    main()

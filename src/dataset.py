"""설계 변형 데이터셋 — 무작위 설계안마다 시뮬레이터로 (타점 × 로봇) 판정을 모은다.

실행: python src/dataset.py N [workers]  → data/gen/designs.parquet 대신 npz/json (의존성 최소화)
"""
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
GEN = ROOT / "data" / "gen"

# 설계 변수 범위 [가정 — 단순화 모형 안에서 형상이 깨지지 않는 범위]
SPACE = {
    "flange_width": (0.008, 0.050),
    "member_wall_height": (0.05, 0.25),
    "member_first_spot": (0.00, 0.15),
    "member_y0": (0.08, 0.25),
    "member_x": (0.60, 1.80),
    "pillar_x": (0.90, 1.50),
    "pillar_w": (0.10, 0.26),
    "sill_top": (0.45, 0.65),
    "floor_z": (0.25, 0.35),
}
PARAMS = list(SPACE)


def make_design(v: dict, name="gen"):
    from body import Design
    y0 = v["member_y0"]
    return Design(name, flange_width=v["flange_width"], member_wall_height=v["member_wall_height"],
                  member_first_spot=v["member_first_spot"], member_y=(y0, y0 + 0.68),
                  member_x=v["member_x"], pillar_x=v["pillar_x"], pillar_w=v["pillar_w"],
                  sill_top=v["sill_top"], floor_z=v["floor_z"],
                  sill_spot_z=(v["floor_z"] + v["sill_top"]) / 2)


def sample(rng, n):
    return [{k: float(rng.uniform(*SPACE[k])) for k in PARAMS} for _ in range(n)]


def label(args):
    idx, v = args
    from body import obstacles, spots
    from reach import GUN_LENGTH, check
    from robot import Robot
    from stage1 import BASES
    d = make_design(v)
    S, B = spots(d), obstacles(d)
    robots = [Robot(b, np.pi / 2, GUN_LENGTH) for b in BASES]
    rows = []
    for i, s in enumerate(S):
        for r, rb in enumerate(robots):
            vd = check(rb, s, B)
            rows.append((idx, i, r, vd.ok, vd.reason, vd.tilt if vd.ok else -1))
    return idx, rows


def main(n=2000, workers=8, seed=0):
    GEN.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    designs = sample(rng, n)
    t = time.time()
    out = []
    with Pool(workers) as pool:
        for k, (idx, rows) in enumerate(pool.imap_unordered(label, list(enumerate(designs)), chunksize=4)):
            out.extend(rows)
            if (k + 1) % 200 == 0:
                print(f"{k + 1}/{n}  {time.time() - t:.0f}s", flush=True)
    arr = np.array([(a, b, c, int(d), e) for a, b, c, d, _, e in out], dtype=float)
    reasons = [x[4] for x in out]
    np.savez_compressed(GEN / f"labels_{seed}.npz", arr=arr)
    (GEN / f"designs_{seed}.json").write_text(json.dumps(designs), encoding="utf-8")
    (GEN / f"reasons_{seed}.json").write_text(json.dumps(reasons), encoding="utf-8")
    print(f"done {n} designs, {len(out)} pairs, {time.time() - t:.0f}s, "
          f"{(time.time() - t) / n * workers:.2f} s/design/core")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    w = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    s = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    main(n, w, s)

"""AI 대리 모델로 설계 공간 탐색 → 시뮬레이터로 재검증.

1) 제조성 지도: 기준 설계 A 에서 두 변수(첫 타점 거리 × 실 타점–B필러 간격)만 바꾼 격자 — AI 60×60, 시뮬레이터 8×8 로 대조
2) 설계 탐색: 기준 A 근처 무작위 2만 개를 AI 로 걸러 '못 쏘는 타점 0' 이면서 A 에서 가장 적게 바꾼 안 → 상위 20 개 시뮬레이터 검증
실행: python src/explore.py → results/explore.json, results/fig_map.png
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

import lightgbm as lgb  # noqa: E402

from dataset import PARAMS, SPACE, label  # noqa: E402
from surrogate import design_features  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
BASE_A = {"flange_width": 0.015, "member_wall_height": 0.12, "member_first_spot": 0.03, "member_y0": 0.12,
          "member_x": 1.20, "pillar_x": 1.20, "pillar_w": 0.16, "sill_top": 0.55, "floor_z": 0.30,
          "sill_pillar_gap": 0.03}
MAP_X, MAP_Y = "member_first_spot", "sill_pillar_gap"
from stage1 import BASES  # noqa: E402
N_ROBOTS = len(BASES)


def fast_predict(model, thr, designs):
    X = np.concatenate([design_features(v) for v in designs])
    p = model.predict(X).reshape(len(designs), -1, N_ROBOTS).max(axis=2)
    return (p < thr).sum(axis=1)


def sim_unshootable(results):
    out = {}
    for idx, rows in results:
        ok = {}
        for _, i, _, feas, _, _ in rows:
            ok[i] = ok.get(i, False) or feas
        out[idx] = sum(1 for v in ok.values() if not v)
    return out


def norm_dist(v):
    return float(np.sqrt(sum(((v[k] - BASE_A[k]) / (SPACE[k][1] - SPACE[k][0])) ** 2 for k in PARAMS)))


def main():
    model = lgb.Booster(model_file=str(ROOT / "models" / "surrogate.txt"))
    thr = json.loads((OUT / "surrogate.json").read_text(encoding="utf-8"))["threshold"]
    res = {}

    # 1) 제조성 지도
    fw = np.linspace(*SPACE[MAP_X], 60)
    fs = np.linspace(*SPACE[MAP_Y], 60)
    grid = [dict(BASE_A, **{MAP_X: float(a), MAP_Y: float(b)}) for b in fs for a in fw]
    t = time.time()
    ai_map = fast_predict(model, thr, grid).reshape(len(fs), len(fw))
    ai_s = time.time() - t
    cw = np.linspace(*SPACE[MAP_X], 8)
    cs = np.linspace(*SPACE[MAP_Y], 8)
    coarse = [dict(BASE_A, **{MAP_X: float(a), MAP_Y: float(b)}) for b in cs for a in cw]
    t = time.time()
    with Pool(3) as pool:
        sim = sim_unshootable(pool.map(label, list(enumerate(coarse))))
    sim_s = time.time() - t
    sim_map = np.array([sim[i] for i in range(len(coarse))]).reshape(8, 8)
    ai_coarse = fast_predict(model, thr, coarse).reshape(8, 8)
    res["map"] = {"x_param": MAP_X, "y_param": MAP_Y, "ai_grid": ai_map.tolist(), "fw": fw.tolist(), "fs": fs.tolist(),
                  "sim_grid": sim_map.tolist(), "ai_on_sim_points": ai_coarse.tolist(),
                  "cw": cw.tolist(), "cs": cs.tolist(),
                  "agree_exact": float((sim_map == ai_coarse).mean()),
                  "agree_flag": float(((sim_map > 0) == (ai_coarse > 0)).mean()),
                  "ai_seconds_3600": round(ai_s, 1), "sim_seconds_64_on_3cores": round(sim_s, 1)}
    print("map", {k: v for k, v in res["map"].items() if not isinstance(v, list)}, flush=True)

    # 2) 설계 탐색
    rng = np.random.default_rng(42)
    cands = []
    for _ in range(10000):
        v = {}
        for k in PARAMS:
            lo, hi = SPACE[k]
            v[k] = float(np.clip(BASE_A[k] + rng.normal(0, 0.25) * (hi - lo), lo, hi))
        cands.append(v)
    t = time.time()
    pred = np.concatenate([fast_predict(model, thr, cands[i:i + 500]) for i in range(0, len(cands), 500)])
    search_s = time.time() - t
    ok_idx = [i for i in np.argsort([norm_dist(v) for v in cands]) if pred[i] == 0][:20]
    picks = [cands[i] for i in ok_idx]
    t = time.time()
    with Pool(3) as pool:
        sim = sim_unshootable(pool.map(label, list(enumerate(picks))))
    verify_s = time.time() - t
    res["search"] = {
        "candidates": len(cands), "ai_pass": int((pred == 0).sum()), "ai_seconds": round(search_s, 1),
        "sim_seconds_equiv_est": round(len(cands) * verify_s / len(picks), 0),
        "top20_sim_unshootable": [sim[i] for i in range(len(picks))],
        "top20_verified": int(sum(1 for i in range(len(picks)) if sim[i] == 0)),
        "top": [{"dist": round(norm_dist(p), 3), **{k: round(p[k], 4) for k in PARAMS},
                 "changes": {k: [BASE_A[k], round(p[k], 4)] for k in PARAMS
                             if abs(p[k] - BASE_A[k]) > 0.02 * (SPACE[k][1] - SPACE[k][0])}}
                for p in picks[:5]],
    }
    print("search", {k: v for k, v in res["search"].items() if k != "top"}, flush=True)
    for p in res["search"]["top"]:
        print(p["dist"], p["changes"])
    # 3) 경계 정밀 검사: 변수 하나씩 5 mm 간격 (다른 문제는 풀어 둔 상태에서)
    probes = {}
    for var, fixed in [("member_first_spot", {"sill_pillar_gap": 0.06}), ("sill_pillar_gap", {"member_first_spot": 0.10})]:
        xs = np.round(np.arange(SPACE[var][0], SPACE[var][1] + 1e-9, 0.005), 3)
        vs = [dict(BASE_A, **fixed, **{var: float(x)}) for x in xs]
        ai = fast_predict(model, thr, vs)
        with Pool(3) as pool:
            sim = sim_unshootable(pool.map(label, list(enumerate(vs))))
        probes[var] = [{"mm": round(float(x) * 1000), "ai": int(a), "sim": sim[i]} for i, (x, a) in enumerate(zip(xs, ai))]
        agree = sum(p["ai"] == p["sim"] for p in probes[var])
        print("probe", var, f"{agree}/{len(xs)}", [p for p in probes[var] if p["ai"] != p["sim"]], flush=True)
    res["probes"] = probes

    # 4) 상위 3안은 전체 시뮬레이션(판정 + 로봇 배정 + 접근·이동 경로 + 로봇끼리 간섭)으로 확정
    import stage1
    from dataset import make_design
    full = []
    for p in picks[:3]:
        t = time.time()
        r = stage1.evaluate(make_design(p, "탐색안"), stage1.TARGET_JPH)
        full.append({"unshootable": len(r["unshootable"]), "robots_used": r["robots_used"],
                     "max_robot_time_s": max(r["robot_time_s"].values()) if r["robot_time_s"] else None,
                     "interlock_wait_s": r["interlock_wait_s"], "seconds": round(time.time() - t, 1)})
        print("full", full[-1], flush=True)
    res["search"]["top3_full_sim"] = full
    res["full_sim_seconds_per_design"] = round(float(np.mean([f["seconds"] for f in full])), 1)
    (OUT / "explore.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()

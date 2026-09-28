"""AI 제조성 대리 모델 — 설계 변수만 보고 (타점 × 로봇) 용접 가능 여부를 예측.

학습: data/gen/*_0 (설계 800개), 시험: data/gen/*_1 (학습에 없던 설계 200개).
실행: python src/surrogate.py → results/surrogate.json, models/surrogate.txt
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import lightgbm as lgb  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from body import obstacles, part_bounds, spots  # noqa: E402
from dataset import GEN, PARAMS, make_design  # noqa: E402
from stage1 import BASES  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"
SHOULDER_Z = 0.75   # 로봇 어깨 높이 근사 (a1+a2 원점 z)


def seg_box_dist(a, b, lo, hi, n=16):
    ts = np.linspace(0, 1, n)[:, None]
    p = a + ts * (b - a)
    d = np.maximum(np.maximum(lo - p, p - hi), 0)
    return float(np.min(np.linalg.norm(d, axis=1)))


GUN_TILTS = (0.0, 15.0, 30.0)


def gun_clearances(spot, plates_lo, plates_hi):
    """건 축을 0°/15°/30° 기울였을 때 전극·암(3~20 cm)과 몸체(20~45 cm) 구간이 판재에서 떨어진 거리.
    타점이 붙어 있는 판은 뺀다. 돌려주는 것: (전극·암 0°, 전극·암 최선, 몸체 최선)."""
    n = spot.normal / np.linalg.norm(spot.normal)
    ts_near = np.linspace(0.03, 0.20, 10)
    ts_far = np.linspace(0.20, 0.45, 10)
    near, far = [], []
    for t in GUN_TILTS:
        a = np.cos(np.radians(t)) * n + np.sin(np.radians(t)) * spot.away
        a = a / np.linalg.norm(a)
        for ts, out in ((ts_near, near), (ts_far, far)):
            pts = spot.pos + ts[:, None] * a
            d = np.maximum(np.maximum(plates_lo[None] - pts[:, None], pts[:, None] - plates_hi[None]), 0)
            out.append(float(np.linalg.norm(d, axis=2).min()))
    return near[0], max(near), max(far)


def design_features(v: dict, engineered=True):
    """설계 하나의 (타점 × 로봇) 특징 행렬. 행 순서 = 타점 i, 로봇 r (dataset 과 같음)."""
    d = make_design(v)
    S = spots(d)
    box = part_bounds(d)
    plates = obstacles(d)
    groups = {"sill": 0, "b_pillar": 1, "member": 2}
    rows = []
    for i, s in enumerate(S):
        if engineered:
            keep = [b for b in plates if not (np.all(s.pos >= b.lo - 1e-6) and np.all(s.pos <= b.hi + 1e-6))]
            gun = gun_clearances(s, np.array([b.lo for b in keep]), np.array([b.hi for b in keep]))
        for r, b in enumerate(BASES):
            base, overhead = np.array(b[:3]), float(b[4] != 0)
            rel = s.pos - base
            dist = np.linalg.norm(rel)
            f = [v[k] for k in PARAMS] + [groups[s.group] == g for g in range(3)] + list(s.pos) + list(s.normal)                 + list(base) + [overhead] + list(rel) + [np.hypot(rel[0], rel[1]), dist,
                                                        float(np.dot(s.normal, -rel) / dist)]
            if engineered:
                shoulder = base + np.array([0, 0, -SHOULDER_Z if overhead else SHOULDER_Z])   # 천장 로봇은 아래로
                above = s.pos + s.normal * 0.45            # 건 뒤끝(플랜지) 근처
                f += [
                    seg_box_dist(shoulder, above, box["b_pillar"].lo, box["b_pillar"].hi),  # 팔이 지나는 길과 필러
                    seg_box_dist(shoulder, above, box["sill"].lo, box["sill"].hi),          # 팔이 지나는 길과 실
                    *gun,                                                                   # 건–판재 여유 (기울임 포함)
                    s.pos[0] - v["pillar_x"],
                    min(abs(s.pos[0] - (v["pillar_x"] - v["pillar_w"] / 2)),
                        abs(s.pos[0] - (v["pillar_x"] + v["pillar_w"] / 2 + 0.025))),  # B필러 앞면·뒤 플랜지 끝까지
                    s.pos[1] - 0.08,                              # 실 안쪽 면에서 떨어진 거리
                    v["sill_top"] - s.pos[2],                     # 실 윗면이 타점보다 얼마나 높은가
                ]
            rows.append(f)
    return np.array(rows, dtype=np.float32)


FEATURE_NAMES = (PARAMS + ["is_sill", "is_pillar", "is_member", "spot_x", "spot_y", "spot_z", "n_x", "n_y", "n_z",
                           "robot_x", "robot_y", "robot_z", "robot_overhead", "rel_x", "rel_y", "rel_z",
                           "dist_xy", "dist", "normal_toward_robot"]
                 + ["clear_arm_pillar", "clear_arm_sill", "clear_gun_tilt0", "clear_gun_best", "clear_body_best",
                    "dx_pillar", "dx_pillar_edge", "dy_sill_face", "sill_above_spot"])


def explain(model, X, y, top=10):
    """SHAP(트리 기여도): 실제로 못 쏘는 쌍에서 '못 쏜다' 쪽으로 민 크기를 특징별로 평균."""
    contrib = model.predict(X[y == 0], pred_contrib=True)[:, :-1]
    push = np.clip(-contrib, 0, None).mean(axis=0)
    order = np.argsort(push)[::-1][:top]
    return [(FEATURE_NAMES[i], round(float(push[i]), 3)) for i in order]


def load_split(seed, engineered=True):
    arr = np.load(GEN / f"labels_{seed}.npz")["arr"]
    designs = json.loads((GEN / f"designs_{seed}.json").read_text(encoding="utf-8"))
    arr = arr[np.lexsort((arr[:, 2], arr[:, 1], arr[:, 0]))]
    X = np.concatenate([design_features(v, engineered) for v in designs])
    y = arr[:, 3].astype(int)
    g = arr[:, 0].astype(int)
    return X, y, g, designs


def spot_level(p_pair, g, n_robots=len(BASES)):
    """타점별 '어느 로봇이든 쏠 수 있다' — 로봇 축으로 최대."""
    return p_pair.reshape(-1, n_robots).max(axis=1), g.reshape(-1, n_robots)[:, 0]


def evaluate(model, X, y, g, thr):
    p = model.predict(X)
    ps, gs = spot_level(p, g)
    ys, _ = spot_level(y.astype(float), g)
    pred_bad, true_bad = ps < thr, ys < 0.5
    per_design_true = np.bincount(gs, weights=true_bad)
    per_design_pred = np.bincount(gs, weights=pred_bad)
    return {
        "pair_auc": round(float(roc_auc_score(y, p)), 4),
        "pair_acc": round(float(((p >= thr) == y).mean()), 4),
        "unshootable_spots_true": int(true_bad.sum()),
        "unshootable_recall": round(float((pred_bad & true_bad).sum() / max(true_bad.sum(), 1)), 4),
        "unshootable_precision": round(float((pred_bad & true_bad).sum() / max(pred_bad.sum(), 1)), 4),
        "design_flag_acc": round(float(((per_design_pred > 0) == (per_design_true > 0)).mean()), 4),
        "design_count_exact": round(float((per_design_pred == per_design_true).mean()), 4),
    }


def train(X, y, g, seed=0):
    ids = np.unique(g)
    rng = np.random.default_rng(seed)
    val_ids = set(rng.choice(ids, size=len(ids) // 5, replace=False).tolist())
    va = np.array([x in val_ids for x in g])
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20,
              "feature_fraction": 0.9, "bagging_fraction": 0.9, "bagging_freq": 1, "verbose": -1, "seed": seed,
              "num_threads": 4}
    dtr, dva = lgb.Dataset(X[~va], y[~va]), lgb.Dataset(X[va], y[va])
    m = lgb.train(params, dtr, 2000, valid_sets=[dva], callbacks=[lgb.early_stopping(100, verbose=False)])
    # 임계값: 검증 설계에서 '못 쏘는 타점' 재현율 ≥ 0.95 를 지키는 가장 높은 정밀도 쪽
    p = m.predict(X[va])
    ps, _ = spot_level(p, g[va])
    ys, _ = spot_level(y[va].astype(float), g[va])
    bad = ys < 0.5
    best = 0.5
    for thr in np.linspace(0.05, 0.95, 91):
        rec = ((ps < thr) & bad).sum() / max(bad.sum(), 1)
        if rec >= 0.95:
            best = float(thr)
            break
    return m, best


def main():
    res = {}
    for eng in (True, False):
        tag = "engineered" if eng else "raw"
        t = time.time()
        Xtr, ytr, gtr, _ = load_split(0, eng)
        Xte, yte, gte, _ = load_split(1, eng)
        feat_s = time.time() - t
        model, thr = train(Xtr, ytr, gtr)
        res[tag] = {"threshold": round(thr, 3), "train_pairs": int(len(ytr)), "test_pairs": int(len(yte)),
                    "test_designs": int(len(np.unique(gte))), "trees": model.num_trees(),
                    **evaluate(model, Xte, yte, gte, thr)}
        if eng:
            (ROOT / "models").mkdir(exist_ok=True)
            model.save_model(str(ROOT / "models" / "surrogate.txt"), num_iteration=model.best_iteration)
            res["threshold"] = thr
            # 속도: 설계 1개 = 특징 + 예측
            t = time.time()
            designs = json.loads((GEN / "designs_1.json").read_text(encoding="utf-8"))[:100]
            for v in designs:
                model.predict(design_features(v))
            res["surrogate_s_per_design"] = round((time.time() - t) / 100, 4)
            res["explain_infeasible"] = explain(model, Xte, yte)
        print(tag, res[tag], f"(features {feat_s:.0f}s)", flush=True)
    OUT.mkdir(exist_ok=True)
    (OUT / "surrogate.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

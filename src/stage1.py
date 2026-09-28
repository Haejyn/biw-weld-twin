"""1단계 — 차체 선행구조 검토: 설계안별 타점 판정 → 로봇 배정 → 사이클타임 → 로봇 대수.

실행: python src/stage1.py  → results/stage1.json, results/stage1_spots.csv
"""
import csv
import json
import warnings
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from body import Design, obstacles, spots
from reach import GUN_LENGTH, check
from robot import JOINT_SPEED, Robot

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results"

# ── 가정 (공개 자료 범위의 대략값, 모두 바꿀 수 있게 한 곳에) ──
TARGET_JPH = 60                 # [가정] 목표 시간당 생산 대수 (evaluate 인자로 바꿔 돌린다)
TRANSFER_CLAMP_S = 15.0         # [가정] 차체 이송·지그 클램프·언클램프
WELD_S = 0.7                    # [가정] 타점당 가압·통전·유지·개방
ACCEL_FACTOR = 1.5              # [가정] 최대 관절속도만으로 잰 이동시간에 곱하는 가감속 보정
SETTLE_S = 0.15                 # [가정] 타점 사이 정착
EST_MOVE_S = 0.45               # 배정 단계에서 쓰는 타점당 이동 추정 (순서 확정 뒤 실측으로 검증)
BASES = [(x, -1.9, 0.0) for x in (0.0, 0.8, 1.6, 2.4)]  # 로봇 후보 위치(바닥), 차체 쪽을 본다

HOME = np.radians([0, 0, -90, 0, 0, 0, 0])  # 대기 자세 (a2=0, a3=-90)


def move_time(q0, q1) -> float:
    dq = np.abs(np.asarray(q1)[1:7] - np.asarray(q0)[1:7])
    return float(np.max(dq / JOINT_SPEED)) * ACCEL_FACTOR + SETTLE_S


def sequence(qs: list) -> tuple[list[int], float]:
    """대기 자세에서 출발해 관절 공간 최근접 순서로 돌고 대기 자세로 돌아오는 시간."""
    home = np.concatenate([[0], HOME[:6], [0, 0]])
    left, order, cur, t = list(range(len(qs))), [], home, 0.0
    while left:
        j = min(left, key=lambda k: move_time(cur, qs[k]))
        t += move_time(cur, qs[j]) + WELD_S
        cur = qs[j]; order.append(j); left.remove(j)
    t += move_time(cur, home) - SETTLE_S if order else 0.0
    return order, t


def assign(feasible: dict, n_spots: int, budget: float):
    """CP-SAT: 모든 타점을 쏠 수 있는 로봇에 하나씩, 로봇당 추정시간 ≤ 예산, 로봇 수 최소."""
    m = cp_model.CpModel()
    R = range(len(BASES))
    x = {(s, r): m.NewBoolVar(f"x{s}_{r}") for s in range(n_spots) for r in R if feasible.get((s, r))}
    use = [m.NewBoolVar(f"u{r}") for r in R]
    for s in range(n_spots):
        cand = [x[s, r] for r in R if (s, r) in x]
        if not cand:
            return None
        m.AddExactlyOne(cand)
    per = int((WELD_S + EST_MOVE_S) * 100)
    for r in R:
        mine = [x[s, r] for s in range(n_spots) if (s, r) in x]
        m.Add(per * sum(mine) <= int(budget * 100) * use[r])
    m.Minimize(sum(use) * 1000 + sum(r * use[r] for r in R))
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 20
    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return {r: [s for s in range(n_spots) if (s, r) in x and solver.Value(x[s, r])] for r in R}


def evaluate(design: Design, jph: float = TARGET_JPH) -> dict:
    S, B = spots(design), obstacles(design)
    robots = [Robot(b, np.pi / 2, GUN_LENGTH) for b in BASES]
    verdicts = {(i, r): check(robots[r], s, B) for i, s in enumerate(S) for r in range(len(robots))}
    feasible = {k: v.ok for k, v in verdicts.items()}
    unshootable = [S[i].id for i in range(len(S)) if not any(feasible[i, r] for r in range(len(robots)))]
    shootable = [i for i in range(len(S)) if S[i].id not in unshootable]

    takt = 3600 / jph
    budget = takt - TRANSFER_CLAMP_S
    plan, verified = None, None
    b = budget
    while b > 1 and verified is None:          # 순서를 실제로 잡아 예산을 넘으면 예산을 줄여 다시 배정
        sub = {(k, r): feasible[i, r] for k, i in enumerate(shootable) for r in range(len(robots))}
        plan = assign(sub, len(shootable), b)
        if plan is None:
            break
        times = {}
        for r, ks in plan.items():
            if ks:
                _, t = sequence([verdicts[shootable[k], r].q for k in ks])
                times[r] = t
        if all(t <= budget for t in times.values()):
            verified = times
        else:
            b -= 1.0

    rows = []
    for i, s in enumerate(S):
        owner = None
        if plan and verified is not None:
            owner = next((r for r, ks in plan.items() if any(shootable[k] == i for k in ks)), None)
        v = verdicts[i, owner] if owner is not None else None
        rows.append({
            "design": design.name, "spot": s.id, "group": s.group,
            "reachable_from": sum(feasible[i, r] for r in range(len(robots))),
            "fail_reasons": ",".join(sorted({verdicts[i, r].reason for r in range(len(robots)) if not feasible[i, r]})),
            "robot": owner, "tilt_deg": v.tilt if v else None,
        })
    return {
        "design": design.name, "notes": design.notes, "jph": jph,
        "params": {k: getattr(design, k) for k in ("flange_width", "member_wall_height", "spot_pitch", "member_y", "member_first_spot")},
        "spots": len(S), "unshootable": unshootable,
        "takt_s": takt, "robot_budget_s": budget,
        "robots_used": len(verified) if verified else None,
        "robot_time_s": {f"R{r}@x={BASES[r][0]}": round(t, 1) for r, t in (verified or {}).items()},
        "tilted_spots": sum(1 for row in rows if row["tilt_deg"]),
        "rows": rows,
    }


DESIGNS = [
    Design("A_기준", flange_width=0.015, member_y=(0.12, 0.80),
           notes="크로스멤버 플랜지 15mm, 첫 타점이 실 안쪽 벽에서 70mm"),
    Design("B_플랜지확대", flange_width=0.030, member_y=(0.12, 0.80),
           notes="플랜지 30mm 로 넓힘"),
    Design("C_플랜지확대+첫타점이동", flange_width=0.030, member_first_spot=0.08,
           notes="B + 크로스멤버 첫 타점을 구간 시작에서 80mm 로 (타점 수 13 그대로)"),
]


def main():
    OUT.mkdir(exist_ok=True)
    results = [evaluate(d) for d in DESIGNS]
    sweep = []
    for d in DESIGNS:
        for jph in (50, 55, 60, 65, 70, 75, 80):
            r = evaluate(d, jph)
            sweep.append({"design": d.name, "jph": jph, "robot_budget_s": round(r["robot_budget_s"], 1),
                          "robots_used": r["robots_used"], "unshootable": len(r["unshootable"]),
                          "max_robot_time_s": max(r["robot_time_s"].values()) if r["robot_time_s"] else None})
            print(sweep[-1])
    (OUT / "stage1_jph_sweep.json").write_text(json.dumps(sweep, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT / "stage1_spots.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0]["rows"][0].keys()))
        w.writeheader()
        for r in results:
            w.writerows(r["rows"])
    summary = [{k: v for k, v in r.items() if k != "rows"} for r in results]
    (OUT / "stage1.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in summary:
        print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()

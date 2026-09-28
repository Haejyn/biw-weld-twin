"""1단계 — 차체 선행구조 검토: 설계안별 타점 판정 → 로봇 배정 → 사이클타임 → 로봇 대수.

실행: python src/stage1.py  → results/stage1.json, results/stage1_spots.csv
"""
import csv
from itertools import combinations
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
BASES = [(x, -1.9, 0.0) for x in (-0.75, 0.55, 1.85, 3.15)]  # 로봇 후보 위치(바닥), 1.3 m 간격 — 받침대가 겹치지 않게

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


def balanced_assign(verdicts, idxs, combo):
    """CP-SAT: 조합 안의 로봇에 타점을 하나씩, 가장 바쁜 로봇의 타점 수 최소."""
    m = cp_model.CpModel()
    x = {(i, r): m.NewBoolVar(f"x{i}_{r}") for i in idxs for r in combo if verdicts[i, r].ok}
    for i in idxs:
        m.AddExactlyOne([x[i, r] for r in combo if (i, r) in x])
    top = m.NewIntVar(0, len(idxs), "top")
    for r in combo:
        m.Add(sum(x[i, r] for i in idxs if (i, r) in x) <= top)
    m.Minimize(top)
    solver = cp_model.CpSolver(); solver.parameters.max_time_in_seconds = 10
    solver.parameters.num_workers = 1; solver.parameters.random_seed = 0
    solver.Solve(m)
    return {r: [i for i in idxs if (i, r) in x and solver.Value(x[i, r])] for r in combo}


def robot_times(verdicts, plan):
    return {r: sequence([verdicts[i, r].q for i in ks])[1] for r, ks in plan.items()}


def plan_robots(verdicts, idxs, n_robots, budget, repair_steps=40):
    """로봇 수 k = 1, 2, … 순서로 후보 위치 조합을 전부 시도. 조합마다 균형 배정 → 실제 용접 순서로
    사이클타임 측정 → 예산을 넘으면 가장 바쁜 로봇의 타점을 옮겨 보는 수선. 처음 성공한 k 에서 가장 여유 있는 조합."""
    for k in range(1, n_robots + 1):
        best = None
        for combo in combinations(range(n_robots), k):
            if any(not any(verdicts[i, r].ok for r in combo) for i in idxs):
                continue
            plan = balanced_assign(verdicts, idxs, combo)
            times = robot_times(verdicts, plan)
            for _ in range(repair_steps):
                worst = max(times, key=times.get)
                if times[worst] <= budget:
                    break
                others = sorted((r for r in combo if r != worst), key=times.get)
                moved = False
                for r in others:
                    movable = [i for i in plan[worst] if verdicts[i, r].ok]
                    if movable:
                        i = movable[-1]
                        plan[worst].remove(i); plan[r].append(i)
                        times = robot_times(verdicts, plan)
                        moved = True
                        break
                if not moved:
                    break
            if max(times.values()) <= budget and (best is None or max(times.values()) < max(best[1].values())):
                best = ({r: list(v) for r, v in plan.items()}, times)
        if best:
            return best
    return None, None


def evaluate(design: Design, jph: float = TARGET_JPH) -> dict:
    S, B = spots(design), obstacles(design)
    robots = [Robot(b, np.pi / 2, GUN_LENGTH) for b in BASES]
    verdicts = {(i, r): check(robots[r], s, B) for i, s in enumerate(S) for r in range(len(robots))}
    feasible = {k: v.ok for k, v in verdicts.items()}
    unshootable = [S[i].id for i in range(len(S)) if not any(feasible[i, r] for r in range(len(robots)))]
    shootable = [i for i in range(len(S)) if S[i].id not in unshootable]

    takt = 3600 / jph
    budget = takt - TRANSFER_CLAMP_S
    plan, verified = plan_robots(verdicts, shootable, len(robots), budget)

    rows = []
    for i, s in enumerate(S):
        owner = None
        if plan and verified is not None:
            owner = next((r for r, ks in plan.items() if i in ks), None)
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

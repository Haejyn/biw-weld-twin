"""1단계 — 차체 선행구조 검토: 설계안별 타점 판정 → 로봇 배정 → 경로·로봇끼리 간섭 → 사이클타임 → 로봇 대수.

실행: python src/stage1.py  → results/stage1.json, results/stage1_jph_sweep.json, results/stage1_spots.csv
"""
import csv
import json
import time
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
from ortools.sat.python import cp_model

from body import Design, obstacles, spots
from paths import HOME, SETTLE_S, WELD_S, best_interlock, move_time, plan_sequence
from reach import GUN_LENGTH, check
from robot import Robot

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results"

# ── 가정 (공개 자료 범위의 대략값, 모두 바꿀 수 있게 한 곳에) ──
TARGET_JPH = 60                 # [가정] 목표 시간당 생산 대수
TRANSFER_CLAMP_S = 15.0         # [가정] 차체 이송·지그 클램프·언클램프
# 로봇 후보 위치 (x, y, z, yaw, roll): 바깥쪽 바닥 3곳(1.3 m 간격, 실·B필러) + 천장 거꾸로 2곳(크로스멤버·바닥)
BASES = ([(x, -1.9, 0.0, np.pi / 2, 0.0) for x in (-0.10, 1.20, 2.50)]
         + [(x, 1.8, 2.4, -np.pi / 2, np.pi) for x in (0.60, 1.80)])


def make_robot(b) -> Robot:
    return Robot(b[:3], b[3], GUN_LENGTH, roll=b[4])


def zone_score(pos, normal, b) -> float:
    """타점을 맡기 좋은 정도(작을수록 좋음): 받침대까지 거리 − 받침대가 타점 법선 쪽에 있는 정도."""
    d = np.asarray(b[:3]) - pos
    dist = float(np.linalg.norm(d))
    return dist - float(np.dot(normal, d / dist))


def sequence(qs: list) -> tuple[list[int], float]:
    """경로 검사 없는 간이 순서 (테스트·빠른 추정용): 대기 자세에서 관절 공간 최근접 순서로 돌고 돌아온다."""
    left, order, cur, t = list(range(len(qs))), [], HOME, 0.0
    while left:
        j = min(left, key=lambda k: move_time(cur, qs[k]))
        t += move_time(cur, qs[j]) + WELD_S
        cur = qs[j]; order.append(j); left.remove(j)
    t += move_time(cur, HOME) - SETTLE_S if order else 0.0
    return order, t


def zone_assign(verdicts, idxs, combo, spot_geo):
    """구역 배정: 타점마다 쓰는 로봇 가운데 가깝고 타점 법선 쪽에 있는 로봇(zone_score 최소)이 맡는다.
    그 로봇이 못 쏘면 다음 후보. 이웃 로봇이 서로의 구역으로 팔을 뻗지 않게 한다."""
    plan = {r: [] for r in combo}
    for i in idxs:
        for r in sorted(combo, key=lambda r: zone_score(*spot_geo[i], BASES[r])):
            if verdicts[i, r].ok:
                plan[r].append(i)
                break
    return plan


def balanced_assign(verdicts, idxs, combo, spot_geo=None):
    """CP-SAT: 조합 안의 로봇에 타점을 하나씩, 가장 바쁜 로봇의 타점 수 최소 (구역 정보가 없을 때)."""
    if spot_geo is not None:
        return zone_assign(verdicts, idxs, combo, spot_geo)
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


def robot_times(verdicts, plan, robots=None, boxes=None):
    """로봇별 사이클. robots·boxes 가 있으면 접근·후퇴 + 이동 경로 간섭(경유)까지 넣은 순서로 잰다."""
    if robots is None:
        return {r: sequence([verdicts[i, r].q for i in ks])[1] for r, ks in plan.items()}
    return {r: plan_sequence(robots[r], [verdicts[i, r] for i in ks], boxes)[1] for r, ks in plan.items()}


def timelines(verdicts, plan, robots, boxes):
    out, vias = {}, 0
    for r, ks in plan.items():
        _, _, keys, v = plan_sequence(robots[r], [verdicts[i, r] for i in ks], boxes)
        out[r], vias = keys, vias + v
    return out, vias


def plan_robots(verdicts, idxs, n_robots, budget, robots=None, boxes=None, repair_steps=40, spot_geo=None):
    """로봇 수 k = 1, 2, … 순서로 후보 위치 조합을 전부 시도. 조합마다 균형 배정 → 실제 용접 순서로
    사이클 측정 → 예산을 넘으면 가장 바쁜 로봇의 타점을 옮기는 수선 → (robots 가 있으면) 로봇끼리 간섭을
    인터록 대기로 풀고 대기까지 넣어 예산 확인. 처음 성공한 k 에서 가장 여유 있는 조합.
    돌려주는 것: (배정, 로봇별 사이클(대기 포함), 메타)."""
    for k in range(1, n_robots + 1):
        best = None
        for combo in combinations(range(n_robots), k):
            if any(not any(verdicts[i, r].ok for r in combo) for i in idxs):
                continue
            plan = balanced_assign(verdicts, idxs, combo, spot_geo)
            times = robot_times(verdicts, plan, robots, boxes)
            for _ in range(repair_steps):
                worst = max(times, key=times.get)
                if times[worst] <= budget:
                    break
                others = sorted((r for r in combo if r != worst), key=times.get)
                moved = False
                for r in others:
                    movable = [i for i in plan[worst] if verdicts[i, r].ok]
                    if movable:
                        i = (min(movable, key=lambda k: zone_score(*spot_geo[k], BASES[r])) if spot_geo
                             else movable[-1])
                        plan[worst].remove(i); plan[r].append(i)
                        times = robot_times(verdicts, plan, robots, boxes)
                        moved = True
                        break
                if not moved:
                    break
            if max(times.values()) > budget:
                continue
            meta = {"waits": {r: 0.0 for r in plan}, "vias": 0, "unresolved": 0}
            if robots is not None and k > 1:
                tl, vias = timelines(verdicts, plan, robots, boxes)
                waits, unresolved, order = best_interlock(robots, tl, times)
                times = {r: times[r] + waits[r] for r in times}
                meta = {"waits": waits, "vias": vias, "unresolved": unresolved,
                        "priority": [r + 1 for r in order] if order else None}
                if unresolved or max(times.values()) > budget:
                    continue
            elif robots is not None:
                meta["vias"] = timelines(verdicts, plan, robots, boxes)[1]
            if best is None or max(times.values()) < max(best[1].values()):
                best = ({r: list(v) for r, v in plan.items()}, times, meta)
        if best:
            return best
    return None, None, None


def judge(design: Design):
    S, B = spots(design), obstacles(design)
    robots = [make_robot(b) for b in BASES]
    verdicts = {(i, r): check(robots[r], s, B) for i, s in enumerate(S) for r in range(len(robots))}
    return S, B, robots, verdicts


def evaluate(design: Design, jph: float = TARGET_JPH, judged=None) -> dict:
    t0 = time.time()
    S, B, robots, verdicts = judged or judge(design)
    judge_s = time.time() - t0
    feasible = {k: v.ok for k, v in verdicts.items()}
    unshootable = [S[i].id for i in range(len(S)) if not any(feasible[i, r] for r in range(len(robots)))]
    shootable = [i for i in range(len(S)) if S[i].id not in unshootable]
    takt = 3600 / jph
    budget = takt - TRANSFER_CLAMP_S
    t1 = time.time()
    plan, verified, meta = plan_robots(verdicts, shootable, len(robots), budget, robots, B,
                                       spot_geo={i: (S[i].pos, S[i].normal) for i in range(len(S))})
    plan_s = time.time() - t1

    rows = []
    for i, s in enumerate(S):
        owner = next((r for r, ks in plan.items() if i in ks), None) if plan else None
        v = verdicts[i, owner] if owner is not None else None
        rows.append({
            "design": design.name, "spot": s.id, "group": s.group,
            "reachable_from": sum(feasible[i, r] for r in range(len(robots))),
            "fail_reasons": ",".join(sorted({verdicts[i, r].reason for r in range(len(robots)) if not feasible[i, r]})),
            "robot": owner, "tilt_deg": v.tilt if v else None,
        })
    return {
        "design": design.name, "notes": design.notes, "jph": jph,
        "params": {k: getattr(design, k) for k in ("flange_width", "member_wall_height", "member_first_spot",
                                                   "sill_pillar_gap", "pillar_w", "member_x")},
        "spots": len(S), "unshootable": unshootable,
        "unshootable_reasons": {S[i].id: sorted({verdicts[i, r].reason for r in range(len(robots))})
                                for i in range(len(S)) if S[i].id in unshootable},
        "takt_s": takt, "robot_budget_s": budget,
        "robots_used": len(verified) if verified else None,
        "robot_time_s": {f"R{r + 1}{'천장' if BASES[r][4] else ''}@x={BASES[r][0]}": round(t, 1)
                         for r, t in (verified or {}).items()},
        "interlock_wait_s": {f"R{r + 1}": round(w, 1) for r, w in (meta or {}).get("waits", {}).items()},
        "via_moves": (meta or {}).get("vias"),
        "judge_s": round(judge_s, 1), "plan_s": round(plan_s, 1),
        "rows": rows,
    }


DESIGNS = [
    Design("A_기준", flange_width=0.015, member_first_spot=0.03, sill_pillar_gap=0.03,
           notes="크로스멤버 플랜지 15 mm · 첫 타점 30 mm · 실 타점–B필러 30 mm"),
    Design("B_플랜지확대", flange_width=0.030, member_first_spot=0.03, sill_pillar_gap=0.03,
           notes="A 에서 크로스멤버 플랜지만 30 mm 로"),
    Design("C_개선", flange_width=0.030, member_first_spot=0.08, sill_pillar_gap=0.05,
           notes="B + 크로스멤버 첫 타점 80 mm + 실 타점–B필러 50 mm (타점 수 그대로)"),
]
JPHS = (50, 60, 70)


def _design_run(d):
    judged = judge(d)
    rows, main_r = [], None
    for jph in JPHS:
        r = evaluate(d, jph, judged)
        rows.append({"design": d.name, "jph": jph, "robot_budget_s": round(r["robot_budget_s"], 1),
                     "robots_used": r["robots_used"], "unshootable": len(r["unshootable"]),
                     "max_robot_time_s": max(r["robot_time_s"].values()) if r["robot_time_s"] else None,
                     "robot_time_s": r["robot_time_s"],
                     "interlock_wait_s": r["interlock_wait_s"], "via_moves": r["via_moves"]})
        print(rows[-1], flush=True)
        if jph == TARGET_JPH:
            main_r = r
    return rows, main_r


def main():
    OUT.mkdir(exist_ok=True)
    out = [_design_run(d) for d in DESIGNS]      # 한 프로세스로 (PC 부하를 낮게)
    sweep = [row for rows, _ in out for row in rows]
    results = [r for _, r in out]
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

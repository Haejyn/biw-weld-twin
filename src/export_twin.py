"""3D 디지털 트윈 화면용 데이터 — results/twin_data.json

설계안별: 차체 상자·타점(판정·담당 로봇·기울임)·로봇 위치·로봇별 용접 순서(관절각·시각).
라인: 대안별 스테이션 부하와 처음 60대의 스테이션별 지연·정지.
"""
import json
import warnings
from pathlib import Path

import numpy as np

import stage2
from body import obstacles, spots
from reach import GUN_LENGTH, check
from robot import Robot
from salbp import balance, load, station_loads
from stage1 import BASES, DESIGNS, HOME, WELD_S, TRANSFER_CLAMP_S, move_time, plan_robots, sequence

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent.parent / "results"


def body_payload(design, jph=60):
    S, B = spots(design), obstacles(design)
    robots = [Robot(b, np.pi / 2, GUN_LENGTH) for b in BASES]
    ver = {(i, r): check(robots[r], s, B) for i, s in enumerate(S) for r in range(len(robots))}
    feas = {k: v.ok for k, v in ver.items()}
    shoot = [i for i in range(len(S)) if any(feas[i, r] for r in range(len(robots)))]
    budget = 3600 / jph - TRANSFER_CLAMP_S
    plan, _ = plan_robots(ver, shoot, len(robots), budget)
    owner = {i: r for r, ks in plan.items() for i in ks}
    home = np.concatenate([[0], HOME[:6], [0, 0]])
    paths = []
    for r, ks in plan.items():
        if not ks:
            continue
        qs = [ver[i, r].q for i in ks]
        order, total = sequence(qs)
        t, cur, frames = 0.0, home, [{"t": 0.0, "q": home[1:7].tolist(), "spot": None}]
        for j in order:
            t += move_time(cur, qs[j])
            frames.append({"t": round(t, 3), "q": qs[j][1:7].tolist(), "spot": S[ks[j]].id})
            t += WELD_S
            frames.append({"t": round(t, 3), "q": qs[j][1:7].tolist(), "spot": S[ks[j]].id, "weld_end": True})
            cur = qs[j]
        t += move_time(cur, home) - 0.15
        frames.append({"t": round(t, 3), "q": home[1:7].tolist(), "spot": None})
        paths.append({"robot": r, "cycle_s": round(total, 2), "frames": frames})
    return {
        "name": design.name, "notes": design.notes, "budget_s": budget,
        "boxes": [{"name": bx.name, "lo": bx.lo.tolist(), "hi": bx.hi.tolist()} for bx in B],
        "spots": [{
            "id": s.id, "group": s.group, "pos": s.pos.tolist(), "normal": s.normal.tolist(),
            "robot": owner.get(i), "tilt": ver[i, owner[i]].tilt if i in owner else None,
            "approach": ver[i, owner[i]].approach.tolist() if i in owner else None,
            "reasons": sorted({ver[i, r].reason for r in range(len(robots)) if not feas[i, r]}),
        } for i, s in enumerate(S)],
        "robots": [{"id": r, "base": list(bs), "yaw": np.pi / 2, "used": any(p["robot"] == r for p in paths)}
                   for r, bs in enumerate(BASES)],
        "paths": paths,
    }


def line_payload(n_cars=60):
    stage2.rng = np.random.default_rng(7)
    old, prec = load("ARC83")
    new = stage2.new_car_times(old)
    c = 3600 / stage2.JPH * 100
    n0, a0, _ = balance([old], [1.0], prec, c, max_stations=25)
    nB, aB, _ = balance([old, new], [1 - stage2.MIX_NEW, stage2.MIX_NEW], prec, c,
                        model_cap=c * (1 + stage2.DRIFT), max_stations=30)
    out = {"takt_s": c / 100, "drift": stage2.DRIFT, "alternatives": []}
    for name, n, a, pol in [("A 그대로 혼류", n0, a0, "random"), ("B 재밸런싱+1곳, 고르게 섞기", nB, aB, "even")]:
        lo, ln = station_loads(old, a, n), station_loads(new, a, n)
        seq = stage2.sequence(pol, n_cars, stage2.MIX_NEW)
        window, lag, cars = c * (1 + stage2.DRIFT), np.zeros(n), []
        for is_new in seq:
            need = lag + np.array(ln if is_new else lo)
            stop = float(np.max(np.maximum(need - window, 0)))
            cars.append({"new": bool(is_new), "stop_s": round(stop / 100, 2),
                         "stop_station": int(np.argmax(need)) if stop > 0 else None,
                         "lag_s": [round(x / 100, 2) for x in np.minimum(need, window)]})
            lag = np.maximum(np.minimum(need, window) - c, 0)
        out["alternatives"].append({"name": name, "stations": n,
                                    "load_old_s": [round(x / 100, 2) for x in lo],
                                    "load_new_s": [round(x / 100, 2) for x in ln], "cars": cars})
    return out


if __name__ == "__main__":
    data = {"robot_model": "KUKA KR210 L150 (ROS-Industrial kuka_experimental)", "gun_length": GUN_LENGTH,
            "designs": [body_payload(d) for d in DESIGNS], "line": line_payload()}
    (OUT / "twin_data.json").write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for d in data["designs"]:
        print(d["name"], [(p["robot"], p["cycle_s"], len(p["frames"])) for p in d["paths"]])
    for a in data["line"]["alternatives"]:
        print(a["name"], a["stations"], sum(c["stop_s"] > 0 for c in a["cars"]))

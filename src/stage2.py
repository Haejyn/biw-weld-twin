"""2단계 — 신차 혼류 투입 검토: 대안별 스테이션 수 · 과부하 · 5년 비용(억 원, 공개 자료 단가) · 추천안이 뒤집히는 경계.

기존 라인 = SALBP 공개 벤치마크 ARC83(83작업, 선후관계 그대로), 시간 단위 1 = 0.01 s 로 본다.
신차 = 같은 작업 그래프에 작업별 시간 배수를 씌운 것 [가정 — 실제 신차 데이터 아님].
실행: python src/stage2.py → results/stage2.json
"""
import json
import math
from pathlib import Path

import numpy as np

from salbp import balance, load, station_loads

OUT = Path(__file__).resolve().parent.parent / "results"
rng = np.random.default_rng(7)

# ── 가정 ──
JPH = 60
MIX_NEW = 0.30                 # 신차 비율
DRIFT = 0.20                   # 작업자가 다음 스테이션 쪽으로 넘어가 일할 수 있는 여유(택트 대비)
NEW_CAR_FACTOR = (1.15, 0.15)  # 신차 작업별 시간 배수 ~ 로그정규(평균 1.15, 표준편차 0.15)
CARS = 3000
HOURS_PER_YEAR = 4000          # 2교대 × 8 h × 250 일
YEARS = 5
SHIFTS = 2

# 비용 — 억 원. 출처는 data/costs_sources.json · docs/costs.md (공개 자료), 없는 것만 [가정]
_C = {it["id"]: it for it in json.loads((Path(__file__).resolve().parent.parent / "data" / "costs_sources.json")
                                        .read_text(encoding="utf-8"))["items"]}
EOK = 1e8
LABOUR = {k: _C["labour_cost_per_worker"][f"{k}_krw"] / EOK for k in ("low", "typical", "high")}   # 인·년
LOST_CAR = {k: _C["value_of_lost_car"][f"{k}_krw"] / EOK for k in ("low", "typical", "high")}      # 대
ROBOT_CELL = {k: _C["spot_weld_robot_cell"][f"{k}_krw"] / EOK for k in ("low", "typical", "high")}  # 대
PLANT_PER_CAPACITY = {k: _C["plant_capex_per_capacity"][f"{k}_krw"] / EOK for k in ("low", "typical", "high")}
STATION_CAPEX = 10.0           # [가정 — 공개 단가 없음] 수동 조립 스테이션 1곳(컨베이어 구간·공구·지그), 경계값으로 따로 본다
LINE_SHARE = 0.10              # [가정] 신규 '라인'(조립 한 줄)이 공장 전체 투자에서 차지하는 몫


def new_car_times(base):
    mu, sd = NEW_CAR_FACTOR
    s2 = math.log(1 + (sd / mu) ** 2)
    f = rng.lognormal(math.log(mu) - s2 / 2, math.sqrt(s2), len(base))
    return [t * k for t, k in zip(base, f)]


def sequence(policy: str, n: int, p: float):
    if policy == "random":
        return list(rng.random(n) < p)
    acc, out = 0.0, []            # 고르게 섞기: 누적 비율이 1을 넘을 때마다 신차
    for _ in range(n):
        acc += p
        out.append(acc >= 1.0)
        if acc >= 1.0:
            acc -= 1.0
    return out


def simulate(loads_old, loads_new, cycle, seq):
    """페이스드 라인. 작업자가 여유창(택트×(1+DRIFT)) 안에 못 끝내면 라인 전체가 넘친 만큼 선다."""
    window = cycle * (1 + DRIFT)
    lag = np.zeros(len(loads_old))
    stops, stop_time = 0, 0.0
    for is_new in seq:
        need = lag + np.array(loads_new if is_new else loads_old)
        stop = float(np.max(np.maximum(need - window, 0)))
        if stop > 0:
            stops += 1
            stop_time += stop
        lag = np.maximum(np.minimum(need, window) - cycle, 0)
    total = len(seq) * cycle + stop_time
    eff_jph = len(seq) / (total / 100 / 3600)
    return {"line_stops_per_100": round(100 * stops / len(seq), 1),
            "stop_share_pct": round(100 * stop_time / total, 2),
            "effective_jph": round(eff_jph, 2)}


def lost_cars(eff_jph):
    return max(JPH - eff_jph, 0) * HOURS_PER_YEAR * YEARS


def cost(added, stations, eff_jph, fixed=0.0, station_capex=STATION_CAPEX, labour=None, lost_car=None):
    """5년 비용(억 원) = 신규 설비 + 인건비(스테이션마다 2교대) + 못 만든 차 이익."""
    labour = LABOUR["typical"] if labour is None else labour
    lost_car = LOST_CAR["typical"] if lost_car is None else lost_car
    return round(fixed + added * station_capex + stations * SHIFTS * labour * YEARS + lost_cars(eff_jph) * lost_car, 1)


def new_line_fixed(share=LINE_SHARE, per_capacity=None):
    per_capacity = PLANT_PER_CAPACITY["typical"] if per_capacity is None else per_capacity
    return per_capacity * JPH * MIX_NEW * HOURS_PER_YEAR * share


def main(extra_stations_by_design: dict | None = None):
    old, prec = load("ARC83")
    new = new_car_times(old)
    c = 3600 / JPH * 100
    n0, a0, _ = balance([old], [1.0], prec, c, max_stations=25)
    lo, ln = station_loads(old, a0, n0), station_loads(new, a0, n0)
    res = {"assumptions": {"JPH": JPH, "mix_new": MIX_NEW, "drift": DRIFT, "new_car_factor": NEW_CAR_FACTOR,
                           "unit": "억 원", "station_capex_assumed": STATION_CAPEX, "labour_per_worker_year": LABOUR,
                           "lost_car_value": LOST_CAR, "robot_cell": ROBOT_CELL, "line_share_assumed": LINE_SHARE,
                           "new_line_fixed": round(new_line_fixed(), 1), "years": YEARS, "shifts": SHIFTS},
           "work_content_s": {"old": round(sum(old) / 100, 1), "new": round(sum(new) / 100, 1)},
           "existing_line_stations": n0,
           "new_car_overloaded_stations": sum(l > c * (1 + DRIFT) for l in ln)}

    alts = {}
    for pol in ("random", "even"):
        sim = simulate(lo, ln, c, sequence(pol, CARS, MIX_NEW))
        alts[f"A_혼류_그대로_{pol}"] = {"stations": n0, "added": 0, **sim,
                                    "lost_cars_5y": round(lost_cars(sim["effective_jph"])),
                                    "cost5y": cost(0, n0, sim["effective_jph"])}

    nB, aB, optB = balance([old, new], [1 - MIX_NEW, MIX_NEW], prec, c, model_cap=c * (1 + DRIFT), max_stations=30)
    lbo, lbn = station_loads(old, aB, nB), station_loads(new, aB, nB)
    for pol in ("random", "even"):
        sim = simulate(lbo, lbn, c, sequence(pol, CARS, MIX_NEW))
        alts[f"B_재밸런싱_증설_{pol}"] = {"stations": nB, "added": nB - n0, "proven_optimal": optB, **sim,
                                      "lost_cars_5y": round(lost_cars(sim["effective_jph"])),
                                      "cost5y": cost(nB - n0, nB, sim["effective_jph"])}

    c_old, c_new = 3600 / (JPH * (1 - MIX_NEW)) * 100, 3600 / (JPH * MIX_NEW) * 100
    nCo, _, _ = balance([old], [1.0], prec, c_old, max_stations=25)
    nCn, _, _ = balance([new], [1.0], prec, c_new, max_stations=25)
    alts["C_신규라인"] = {"stations": nCo + nCn, "existing": nCo, "new_line": nCn, "added": nCn,
                        "line_stops_per_100": 0.0, "stop_share_pct": 0.0, "effective_jph": JPH,
                        "lost_cars_5y": 0,
                        "cost5y": cost(nCn, nCo + nCn, JPH, fixed=new_line_fixed())}
    res["alternatives"] = alts

    # 추천안이 뒤집히는 경계 — 공개 단가의 낮음·보통·높음 × 가정 두 개(스테이션 설비비, 신규 라인 몫)
    grid = []
    for sc in (1, 5, 10, 20, 50, 100, 200, 500):
        for share in (0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
            for lk in ("low", "typical", "high"):
                for ck in ("low", "typical", "high"):
                    costs = {k: cost(v["added"], v["stations"], v["effective_jph"],
                                     fixed=new_line_fixed(share) if k.startswith("C") else 0.0,
                                     station_capex=sc, labour=LABOUR[lk], lost_car=LOST_CAR[ck])
                             for k, v in alts.items()}
                    grid.append({"station_capex": sc, "line_share": share, "labour": lk, "lost_car": ck,
                                 "best": min(costs, key=costs.get)})
    res["sensitivity"] = grid
    A, B = alts["A_혼류_그대로_even"], alts["B_재밸런싱_증설_even"]
    res["break_even"] = {
        # A 와 B 가 같아지는 스테이션 설비비: 증설 1곳 설비비 + 작업자 2교대 5년 = A 가 잃는 차 이익
        "station_capex_A_eq_B": round((A["lost_cars_5y"] - B["lost_cars_5y"]) * LOST_CAR["typical"]
                                      - (B["stations"] - A["stations"]) * SHIFTS * LABOUR["typical"] * YEARS, 1),
        "lost_cars_A_5y": A["lost_cars_5y"],
    }

    # 1단계 연결: 로봇 셀 대수 × 공개 단가, 못 쏘는 타점이 있으면 수동 보완 스테이션 1곳 + 2교대 작업자 [가정]
    s1 = json.loads((OUT / "stage1.json").read_text(encoding="utf-8"))
    best = min(alts, key=lambda k: alts[k]["cost5y"])
    res["combined"] = [{
        "body_design": d["design"], "line_alt": best, "unshootable": len(d["unshootable"]),
        "weld_robots": d["robots_used"],
        "robot_capex": round((d["robots_used"] or 0) * ROBOT_CELL["typical"], 1),
        "manual_touchup": round(STATION_CAPEX + SHIFTS * LABOUR["typical"] * YEARS, 1) if d["unshootable"] else 0.0,
        "cost5y": round(alts[best]["cost5y"] + (d["robots_used"] or 0) * ROBOT_CELL["typical"]
                        + ((STATION_CAPEX + SHIFTS * LABOUR["typical"] * YEARS) if d["unshootable"] else 0), 1),
    } for d in s1]
    OUT.mkdir(exist_ok=True)
    (OUT / "stage2.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "sensitivity"}, ensure_ascii=False, indent=1, default=float))
    from collections import Counter
    print("sensitivity winners:", Counter(g["best"] for g in grid))
    print("break even", res["break_even"])
    for c in res["combined"]:
        print(c)


if __name__ == "__main__":
    main()

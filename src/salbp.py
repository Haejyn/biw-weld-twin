"""SALBP 공개 벤치마크(Scholl 1993) 읽기와 CP-SAT 라인밸런싱."""
from pathlib import Path

from ortools.sat.python import cp_model

DATA = Path(__file__).resolve().parent.parent / "data" / "salbp" / "precedence graphs"


def load(name: str):
    lines = [l.strip() for l in (DATA / f"{name}.IN2").read_text().splitlines() if l.strip()]
    n = int(lines[0])
    times = [int(x) for x in lines[1:1 + n]]
    prec = []
    for l in lines[1 + n:]:
        a, b = (int(x) for x in l.split(","))
        if a == -1:
            break
        prec.append((a - 1, b - 1))
    return times, prec


def balance(models: list[list[float]], weights: list[float], prec, cycle: float,
            model_cap: float | None = None, max_stations: int | None = None, time_limit: float = 30):
    """스테이션 수 최소화.

    models: 모델별 작업시간 목록 (한 모델이면 고전 SALBP-1)
    가중 평균 부하 ≤ cycle, (model_cap 이 있으면) 모델별 부하 ≤ model_cap.
    돌려주는 것: (스테이션 수, 작업별 스테이션 번호, 최적 여부)
    """
    n = len(models[0])
    avg = [sum(w * m[i] for w, m in zip(weights, models)) for i in range(n)]
    K = max_stations or n
    m = cp_model.CpModel()
    x = [[m.NewBoolVar(f"x{i}_{k}") for k in range(K)] for i in range(n)]
    st = [m.NewIntVar(0, K - 1, f"s{i}") for i in range(n)]
    for i in range(n):
        m.AddExactlyOne(x[i])
        m.Add(st[i] == sum(k * x[i][k] for k in range(K)))
    for a, b in prec:
        m.Add(st[a] <= st[b])
    scale = 100
    for k in range(K):
        m.Add(sum(int(round(avg[i] * scale)) * x[i][k] for i in range(n)) <= int(round(cycle * scale)))
        if model_cap is not None:
            for mt in models:
                m.Add(sum(int(round(mt[i] * scale)) * x[i][k] for i in range(n)) <= int(round(model_cap * scale)))
    last = m.NewIntVar(0, K - 1, "last")
    m.AddMaxEquality(last, st)
    m.Minimize(last)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    res = solver.Solve(m)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, None, False
    return solver.Value(last) + 1, [solver.Value(s) for s in st], res == cp_model.OPTIMAL


def station_loads(times, assign, n_stations):
    loads = [0.0] * n_stations
    for t, s in zip(times, assign):
        loads[s] += t
    return loads

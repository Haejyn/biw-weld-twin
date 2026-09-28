"""손으로 확인할 수 있는 작은 경우로 판정 로직을 고정한다."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
warnings.filterwarnings("ignore")

from body import Box, Spot  # noqa: E402
from reach import GUN_LENGTH, check, gun_collides  # noqa: E402
from robot import Robot  # noqa: E402
from salbp import balance, load  # noqa: E402
from stage2 import simulate  # noqa: E402

ROBOT = Robot((0, -1.9, 0), np.pi / 2, GUN_LENGTH)


def test_ik_hits_reachable_point():
    q, pos_err, ang_err = ROBOT.solve((0.5, 0, 0.5), (0, 1, 0))
    assert pos_err < 1e-3 and ang_err < 0.5 and ROBOT.within_limits(q)


def test_far_point_is_unreachable():
    spot = Spot("far", "t", np.array([0.0, 2.5, 0.3]), np.array([0, 0, 1.0]))
    assert check(ROBOT, spot, []).reason == "unreachable"


def test_gun_hits_wall_right_next_to_spot():
    wall = Box("w", np.array([0.02, -0.5, 0.0]), np.array([0.022, 0.5, 0.3]))
    assert gun_collides(np.zeros(3), np.array([0, 0, 1.0]), [wall])


def test_tilting_away_clears_the_wall():
    wall = Box("w", np.array([0.02, 0.1, 0.30]), np.array([0.022, 0.8, 0.42]))
    spot = Spot("m", "t", np.array([0.0, 0.4, 0.30]), np.array([0, 0, 1.0]), away=np.array([-1.0, 0, 0]))
    v = check(ROBOT, spot, [wall])
    assert v.ok and v.tilt > 0


SALBP = Path(__file__).resolve().parent.parent / "data" / "salbp" / "precedence graphs"


@pytest.mark.skipif(not SALBP.exists(), reason="python src/fetch_data.py 로 벤치마크를 먼저 받는다")
@pytest.mark.parametrize("name,cycle,known", [("ARC83", 3786, 21), ("ARC83", 4454, 18), ("HAHN", 2004, 8)])
def test_balance_matches_published_optimum(name, cycle, known):
    t, p = load(name)
    k, assign, proven = balance([t], [1.0], p, cycle, max_stations=known + 3)
    assert k == known and proven
    assert all(assign[a] <= assign[b] for a, b in p)


def test_no_stop_when_every_load_fits_the_takt():
    r = simulate([5000, 6000], [5500, 6000], 6000, [True, False] * 50)   # 단위 0.01 s, 택트 60 s
    assert r["line_stops_per_100"] == 0 and r["effective_jph"] == 60


def test_one_overloaded_station_stops_the_line_by_the_overflow():
    r = simulate([60], [80], 60, [True])   # 여유창 72 → 8 만큼 정지
    assert r["line_stops_per_100"] == 100
    assert r["stop_share_pct"] == pytest.approx(100 * 8 / 68, abs=0.01)


# ── 로봇 배정 · AI ──
from reach import Verdict  # noqa: E402
from stage1 import plan_robots  # noqa: E402


def _q(a1_deg):
    q = np.zeros(9)
    q[1] = np.radians(a1_deg)
    q[3] = np.radians(-90)
    return q


def test_plan_uses_one_robot_when_one_can_do_everything_in_budget():
    ver = {(i, r): Verdict(True, "ok", _q(i)) for i in range(5) for r in range(2)}
    plan, times, _ = plan_robots(ver, list(range(5)), 2, budget=45)
    assert len(plan) == 1 and max(times.values()) <= 45


def test_plan_needs_two_robots_when_reach_is_split():
    ver = {}
    for i in range(6):
        for r in range(2):
            ver[i, r] = Verdict((i < 3) == (r == 0), "ok", _q(i)) if (i < 3) == (r == 0) else Verdict(False, "unreachable")
    plan, _, _ = plan_robots(ver, list(range(6)), 2, budget=45)
    assert sorted(len(v) for v in plan.values()) == [3, 3]


def test_ai_features_follow_label_order():
    from dataset import label
    from explore import BASE_A
    from surrogate import design_features
    _, rows = label((0, BASE_A))
    X = design_features(BASE_A)
    from stage1 import BASES
    n = len(BASES)
    assert len(rows) == len(X) == 75 * n
    assert [(i, r) for _, i, r, *_ in rows] == [(i, r) for i in range(75) for r in range(n)]


MODEL = Path(__file__).resolve().parent.parent / "models" / "surrogate.txt"


@pytest.mark.skipif(not MODEL.exists(), reason="python src/surrogate.py 로 모델을 먼저 만든다")
def test_ai_flags_the_bad_first_spot_and_clears_a_far_one():
    import json
    import lightgbm as lgb
    from explore import BASE_A, fast_predict
    thr = json.loads((MODEL.parent.parent / "results" / "surrogate.json").read_text())["threshold"]
    m = lgb.Booster(model_file=str(MODEL))
    near = dict(BASE_A, member_first_spot=0.02, sill_pillar_gap=0.07)
    far = dict(BASE_A, member_first_spot=0.12, sill_pillar_gap=0.07)
    got = fast_predict(m, thr, [near, far])
    assert got[0] >= 1 and got[1] == 0


# ── 판금 형상 · 경로 · 로봇끼리 ──
def test_spots_sit_on_flange_plates():
    from body import Design, obstacles, spots
    d = Design("t")
    boxes = {b.name: b for b in obstacles(d)}
    for s in spots(d):
        plate = {"sill": "sill_flange", "b_pillar": "pillar_flange", "member": "member_flange_rear"}[s.group]
        b = boxes[plate]
        assert np.all(s.pos >= b.lo - 1e-9) and np.all(s.pos <= b.hi + 1e-9), s.id


def test_approach_path_blocked_by_a_plate_in_front():
    from body import Box, Spot
    spot = Spot("p", "t", np.array([0.5, 0.0, 0.5]), np.array([0, -1.0, 0]), away=np.array([0, 0, 1.0]))
    # 타점 앞 100 mm 에 판이 가로막는다 — 용접 자세의 건은 안 닿지만 접근 경로에서 걸린다
    wall = Box("w", np.array([0.3, -0.3, 0.2]), np.array([0.7, -0.29, 0.8]))
    v = check(ROBOT, spot, [wall])
    assert not v.ok and v.reason in ("approach_collision", "gun_collision")


def test_two_robots_reaching_the_same_point_interlock():
    from paths import HOME, resolve_interference
    r1 = Robot((0.0, -1.9, 0), np.pi / 2, GUN_LENGTH)
    r2 = Robot((1.3, -1.9, 0), np.pi / 2, GUN_LENGTH)
    q1, *_ = r1.solve((0.65, -0.2, 0.8), (0, 1, 0))
    q2, *_ = r2.solve((0.70, -0.2, 0.8), (0, 1, 0))
    tl = {0: [(0.0, HOME, None), (1.0, q1, 0), (3.0, q1, 0), (4.0, HOME, None)],
          1: [(0.0, HOME, None), (1.0, q2, 0), (3.0, q2, 0), (4.0, HOME, None)]}
    waits, left = resolve_interference([r1, r2], tl)
    assert left == 0 and waits[1] > 0 and waits[0] == 0

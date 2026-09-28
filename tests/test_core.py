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
    plan, times = plan_robots(ver, list(range(5)), 2, budget=45)
    assert len(plan) == 1 and max(times.values()) <= 45


def test_plan_needs_two_robots_when_reach_is_split():
    ver = {}
    for i in range(6):
        for r in range(2):
            ver[i, r] = Verdict((i < 3) == (r == 0), "ok", _q(i)) if (i < 3) == (r == 0) else Verdict(False, "unreachable")
    plan, _ = plan_robots(ver, list(range(6)), 2, budget=45)
    assert sorted(len(v) for v in plan.values()) == [3, 3]


def test_ai_features_follow_label_order():
    from dataset import label
    from explore import BASE_A
    from surrogate import design_features
    _, rows = label((0, BASE_A))
    X = design_features(BASE_A)
    assert len(rows) == len(X) == 75 * 4
    assert [(i, r) for _, i, r, *_ in rows] == [(i, r) for i in range(75) for r in range(4)]


MODEL = Path(__file__).resolve().parent.parent / "models" / "surrogate.txt"


@pytest.mark.skipif(not MODEL.exists(), reason="python src/surrogate.py 로 모델을 먼저 만든다")
def test_ai_flags_the_bad_first_spot_and_clears_a_far_one():
    import json
    import lightgbm as lgb
    from explore import BASE_A, fast_predict
    thr = json.loads((MODEL.parent.parent / "results" / "surrogate.json").read_text())["threshold"]
    m = lgb.Booster(model_file=str(MODEL))
    near, far = dict(BASE_A, member_first_spot=0.02), dict(BASE_A, member_first_spot=0.12)
    got = fast_predict(m, thr, [near, far])
    assert got[0] >= 1 and got[1] == 0

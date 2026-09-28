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

"""타점 하나를 로봇 하나가 쏠 수 있는지 — 도달(IK) · 관절 한계 · 건/팔 간섭."""
from dataclasses import dataclass

import numpy as np

from body import Box, Spot
from robot import Robot

# 스폿 용접건을 팁부터 세 토막으로 본다: (시작, 끝, 반지름) m  [가정: C형 건의 대략 치수]
GUN_SEGMENTS = [(0.00, 0.05, 0.010), (0.05, 0.25, 0.030), (0.25, 0.45, 0.120)]
GUN_LENGTH = 0.45
TIP_CONTACT = 0.03          # 팁이 판에 닿는 구간은 간섭에서 뺀다
ARM_RADII = [0.0, 0.0, 0.20, 0.15, 0.12, 0.10, 0.08]  # 관절 원점 사이 토막별 팔 반지름 [가정]
TILTS_DEG = [0, 10, 20, 30]  # 법선에서 기울일 수 있는 각
POS_TOL, ANG_TOL = 0.002, 1.0


@dataclass
class Verdict:
    ok: bool
    reason: str               # ok · unreachable · joint_limit · gun_collision · arm_collision
    q: np.ndarray | None = None
    tilt: float | None = None
    approach: np.ndarray | None = None


def seg_box_clearance(a, b, box: Box, samples: int = 12) -> float:
    """선분 위 표본점과 상자 사이 최소 거리."""
    ts = np.linspace(0, 1, samples)[:, None]
    pts = a + ts * (b - a)
    d = np.maximum(np.maximum(box.lo - pts, pts - box.hi), 0)
    return float(np.min(np.linalg.norm(d, axis=1)))


def tilted_axes(spot: Spot):
    """법선을 기울인 후보 방향들 — 벽에서 멀어지는 쪽이 있으면 그쪽만, 없으면 네 방향."""
    n = spot.normal / np.linalg.norm(spot.normal)
    helper = np.array([0, 0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(n, helper); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    dirs = [spot.away] if spot.away is not None else [u, -u, v, -v]
    for t in TILTS_DEG:
        if t == 0:
            yield 0, n
            continue
        for d in dirs:
            a = np.cos(np.radians(t)) * n + np.sin(np.radians(t)) * d
            yield t, a / np.linalg.norm(a)


def gun_collides(tip, out_axis, boxes) -> bool:
    for s0, s1, r in GUN_SEGMENTS:
        a = tip + out_axis * max(s0, TIP_CONTACT)
        b = tip + out_axis * s1
        if any(seg_box_clearance(a, b, bx) < r for bx in boxes):
            return True
    return False


def arm_collides(robot: Robot, q, boxes) -> bool:
    pts = robot.frames_world(q)
    for i in range(2, 7):   # a2 부터 손목까지
        a, b = pts[i], pts[i + 1]
        if any(seg_box_clearance(a, b, bx) < ARM_RADII[i] for bx in boxes):
            return True
    return False


def check(robot: Robot, spot: Spot, boxes: list[Box]) -> Verdict:
    worst = "unreachable"
    rank = {"unreachable": 0, "joint_limit": 1, "gun_collision": 2, "arm_collision": 3}
    seed = None
    for tilt, out_axis in tilted_axes(spot):
        if gun_collides(spot.pos, out_axis, boxes):
            worst = max(worst, "gun_collision", key=rank.get)
            continue
        q, pe, ae = robot.solve(spot.pos, -out_axis, seed)
        if pe > POS_TOL or ae > ANG_TOL:
            continue
        if not robot.within_limits(q):
            worst = max(worst, "joint_limit", key=rank.get)
            continue
        if arm_collides(robot, q, boxes):
            worst = max(worst, "arm_collision", key=rank.get)
            continue
        return Verdict(True, "ok", q, tilt, -out_axis)
    return Verdict(False, worst)

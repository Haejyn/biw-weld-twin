"""타점 하나를 로봇 하나가 쏠 수 있는지 — 도달(IK) · 관절 한계 · 건/팔 간섭 · 접근 경로.

용접 자세만 보지 않고, 전극이 법선을 따라 APPROACH 만큼 떨어진 곳에서 들어오는 직선 경로(=후퇴 경로)의
중간 자세들도 도달·관절 한계·간섭을 모두 통과해야 '쏠 수 있다'.
간섭은 건·팔을 선분 위 표본점 + 반지름(캡슐)으로, 차체를 판재 상자로 보고 한 번의 배열 연산으로 잰다.
"""
from dataclasses import dataclass

import numpy as np

from body import Box, Spot
from robot import Robot

# 스폿 용접건을 팁부터 세 토막으로 본다: (시작, 끝, 반지름) m  [가정: C형 건 전극·암·몸체의 대략 치수]
GUN_SEGMENTS = [(0.00, 0.05, 0.010), (0.05, 0.20, 0.020), (0.20, 0.45, 0.110)]
GUN_LENGTH = 0.45
TIP_CONTACT = 0.03          # 용접 자세에서 팁이 판에 닿는 구간은 간섭에서 뺀다
ARM_RADII = [0.0, 0.0, 0.20, 0.15, 0.12, 0.10, 0.08]  # 관절 원점 사이 토막별 팔 반지름 [가정]
TILTS_DEG = [0, 10, 20, 30]  # 법선에서 기울일 수 있는 각
APPROACH = 0.15              # 접근·후퇴 거리 [가정]
APPROACH_STEPS = (0.05, 0.10, 0.15)
POS_TOL, ANG_TOL = 0.002, 1.0
SAMPLES = 8
RANK = {"unreachable": 0, "joint_limit": 1, "gun_collision": 2, "arm_collision": 3, "approach_collision": 4}


@dataclass
class Verdict:
    ok: bool
    reason: str               # ok · unreachable · joint_limit · gun_collision · arm_collision · approach_collision
    q: np.ndarray | None = None
    tilt: float | None = None
    approach: np.ndarray | None = None
    q_pre: np.ndarray | None = None   # 접근 시작 자세 (타점에서 APPROACH 만큼 뒤)


def box_arrays(boxes: list[Box]):
    if not boxes:
        return None
    return np.array([b.lo for b in boxes]), np.array([b.hi for b in boxes])


def capsules_clear(a, b, r, arrays) -> bool:
    """선분들(a[i]→b[i], 반지름 r[i])이 모든 상자에서 떨어져 있는가."""
    if arrays is None or len(a) == 0:
        return True
    lo, hi = arrays
    ts = np.linspace(0, 1, SAMPLES)[None, :, None]
    pts = (a[:, None, :] + ts * (b - a)[:, None, :]).reshape(-1, 3)          # (S*k, 3)
    rad = np.repeat(np.asarray(r, float), SAMPLES)
    d = np.maximum(np.maximum(lo[None] - pts[:, None], pts[:, None] - hi[None]), 0)   # (P, B, 3)
    dist = np.linalg.norm(d, axis=2).min(axis=1)
    return bool(np.all(dist >= rad))


def seg_box_clearance(a, b, box: Box, samples: int = 12) -> float:
    """선분 위 표본점과 상자 사이 최소 거리 (특징 계산용)."""
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


def gun_capsules(tip, out_axis, contact=TIP_CONTACT):
    a = np.array([tip + out_axis * max(s0, contact) for s0, _, _ in GUN_SEGMENTS])
    b = np.array([tip + out_axis * s1 for _, s1, _ in GUN_SEGMENTS])
    return a, b, [r for _, _, r in GUN_SEGMENTS]


def arm_capsules(robot: Robot, q):
    pts = robot.frames_world(q)
    idx = range(2, 7)                      # a2 → 손목 → 플랜지
    return (np.array([pts[i] for i in idx]), np.array([pts[i + 1] for i in idx]), [ARM_RADII[i] for i in idx])


def gun_collides(tip, out_axis, boxes, contact=TIP_CONTACT) -> bool:
    return not capsules_clear(*gun_capsules(tip, out_axis, contact), box_arrays(boxes))


def arm_collides(robot: Robot, q, boxes, arrays=None) -> bool:
    return not capsules_clear(*arm_capsules(robot, q), arrays if arrays is not None else box_arrays(boxes))


REACH_MAX = 3.3   # 받침대에서 전극 끝까지 이보다 멀면 역기구학 없이 도달 불가 (KR210 L150 팔 + 건 0.45 m 보다 여유)


def check(robot: Robot, spot: Spot, boxes: list[Box]) -> Verdict:
    if np.linalg.norm(spot.pos - robot.base) > REACH_MAX:
        return Verdict(False, "unreachable")
    arrays = box_arrays(boxes)
    worst = "unreachable"
    for tilt, out_axis in tilted_axes(spot):
        if not capsules_clear(*gun_capsules(spot.pos, out_axis), arrays):
            worst = max(worst, "gun_collision", key=RANK.get)
            continue
        q, pe, ae = robot.solve(spot.pos, -out_axis)
        if pe > POS_TOL or ae > ANG_TOL:
            continue
        if not robot.within_limits(q):
            worst = max(worst, "joint_limit", key=RANK.get)
            continue
        if not capsules_clear(*arm_capsules(robot, q), arrays):
            worst = max(worst, "arm_collision", key=RANK.get)
            continue
        # 접근·후퇴 직선 경로: 타점에서 APPROACH 까지 뒤로 물러난 자세들
        seed, ok, q_pre = q, True, None
        for s in APPROACH_STEPS:
            tip = spot.pos + out_axis * s
            qs, pe, ae = robot.solve(tip, -out_axis, seed)
            if (pe > POS_TOL or ae > ANG_TOL or not robot.within_limits(qs)
                    or not capsules_clear(*gun_capsules(tip, out_axis, contact=0.0), arrays)
                    or not capsules_clear(*arm_capsules(robot, qs), arrays)):
                ok = False
                break
            seed, q_pre = qs, qs
        if not ok:
            worst = max(worst, "approach_collision", key=RANK.get)
            continue
        return Verdict(True, "ok", q, tilt, -out_axis, q_pre)
    return Verdict(False, worst)

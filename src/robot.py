"""KUKA KR210 L150 기구학 — ROS-Industrial kuka_experimental 의 kr210l150_macro.xacro 값 그대로.

용접건은 플랜지 x 축 방향으로 뻗은 막대로 본다. 끝점이 전극 팁(TCP).
"""
from math import radians

import numpy as np
from ikpy.chain import Chain
from ikpy.link import OriginLink, URDFLink

# (이름, 원점 xyz, 축, 하한°, 상한°, 속도°/s) — data/robot/kr210l150_macro.xacro 164~204행
JOINTS = [
    ("a1", (-0.00262, 0.00097586, 0.33099), (0, 0, 1), -185, 185, 123),
    ("a2", (0.35277, -0.037476, 0.4192), (0, 1, 0), -45, 85, 115),
    ("a3", (-9.8483e-05, -0.1475, 1.2499), (0, 1, 0), -210, 65, 112),
    ("a4", (0.95795, 0.184, -0.055059), (1, 0, 0), -350, 350, 179),
    ("a5", (0.542, 0, 0), (0, 1, 0), -125, 125, 172),
    ("a6", (0.1925, 0, 0), (1, 0, 0), -350, 350, 219),
]
FLANGE_OFFSET = (0.0375, 0, -0.00023924)
JOINT_SPEED = np.radians([j[5] for j in JOINTS])


def build_chain(gun_length: float) -> Chain:
    links = [OriginLink()]
    for name, xyz, axis, lo, hi, _ in JOINTS:
        links.append(URDFLink(name=name, origin_translation=xyz, origin_orientation=(0, 0, 0),
                              rotation=axis, bounds=(radians(lo), radians(hi))))
    links.append(URDFLink(name="flange", origin_translation=FLANGE_OFFSET,
                          origin_orientation=(0, 0, 0), joint_type="fixed"))
    links.append(URDFLink(name="tcp", origin_translation=(gun_length, 0, 0),
                          origin_orientation=(0, 0, 0), joint_type="fixed"))
    mask = [False] + [True] * 6 + [False, False]
    return Chain(name="kr210l150", links=links, active_links_mask=mask)


class Robot:
    """로봇 한 대. base 는 월드 좌표(x, y, z), yaw 는 z 축 회전, roll 은 x 축 회전(라디안)."""

    def __init__(self, base, yaw: float, gun_length: float, roll: float = 0.0):
        """roll = π 이면 천장에 거꾸로 단 로봇 (바닥 용접용)."""
        self.base = np.asarray(base, float)
        c, s = np.cos(yaw), np.sin(yaw)
        rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        cr, sr = np.cos(roll), np.sin(roll)
        rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        self.rot = rz @ rx
        self.chain = build_chain(gun_length)

    def to_local(self, p):
        return self.rot.T @ (np.asarray(p, float) - self.base)

    def to_world(self, p):
        return self.rot @ np.asarray(p, float) + self.base

    def solve(self, tip_world, approach_world, seed=None):
        """전극 팁을 tip 에, 건 축(플랜지 x)을 approach 방향으로. 관절각과 오차를 돌려준다."""
        target = self.to_local(tip_world)
        axis = self.rot.T @ np.asarray(approach_world, float)
        q = self.chain.inverse_kinematics(target, axis, orientation_mode="X",
                                          initial_position=seed)
        # a6 는 건 축(플랜지 x) 둘레 회전이라 전극 위치·건 방향을 바꾸지 않는다. 건을 축대칭으로 보므로
        # 임의로 도는 a6 를 0 으로 고정해 타점 사이의 쓸데없는 손목 회전을 없앤다.
        q = np.array(q, float); q[6] = 0.0
        frame = self.chain.forward_kinematics(q)
        pos_err = float(np.linalg.norm(frame[:3, 3] - target))
        ang_err = float(np.degrees(np.arccos(np.clip(frame[:3, 0] @ axis, -1, 1))))
        return q, pos_err, ang_err

    def frames_world(self, q):
        """관절 원점들과 팁의 월드 좌표 — 팔 간섭 검사용."""
        return [self.to_world(f[:3, 3]) for f in self.chain.forward_kinematics(q, full_kinematics=True)]

    def within_limits(self, q, margin_deg: float = 2.0) -> bool:
        for (_, _, _, lo, hi, _), v in zip(JOINTS, q[1:7]):
            if not (radians(lo + margin_deg) <= v <= radians(hi - margin_deg)):
                return False
        return True

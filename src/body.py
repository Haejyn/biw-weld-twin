"""단순화한 차체 하위 조립품 — 사이드 실 · B필러 · 플로어 크로스멤버.

실제 차체가 아니다. 공개 BIW CAD 에 타점 정보가 없어 설계 변수로 형상을 만든다.
단위 m. x = 차 길이 방향(라인 진행), y = 차 폭(로봇은 y<0 쪽), z = 높이.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Design:
    name: str
    flange_width: float = 0.020      # 크로스멤버 타점 줄에서 세운 벽까지 거리
    member_wall_height: float = 0.120  # 크로스멤버 세운 벽 높이
    spot_pitch: float = 0.050          # 타점 간격
    floor_z: float = 0.30
    sill_top: float = 0.55
    member_y: tuple = (0.12, 0.80)     # 크로스멤버가 차 안쪽으로 뻗는 구간
    member_first_spot: float = 0.03    # 크로스멤버 첫 타점이 구간 시작에서 떨어진 거리
    member_spots: int = 13
    member_x: float = 1.20             # 크로스멤버가 붙는 차 길이 방향 위치
    pillar_x: float = 1.20             # B필러 중심
    pillar_w: float = 0.16             # B필러 폭(차 길이 방향)
    sill_spot_z: float = 0.42          # 실 바깥 플랜지 타점 높이
    notes: str = ""


@dataclass
class Spot:
    id: str
    group: str
    pos: np.ndarray
    normal: np.ndarray                  # 건이 다가오는 쪽(바깥) 방향
    away: np.ndarray = field(default=None)  # 기울일 때 벽에서 멀어지는 쪽


@dataclass
class Box:
    name: str
    lo: np.ndarray
    hi: np.ndarray


def spots(d: Design) -> list[Spot]:
    out = []
    p = d.spot_pitch
    for i, x in enumerate(np.arange(0.10, 2.30 + 1e-9, p)):   # 실 바깥 플랜지, 옆에서 쏜다
        out.append(Spot(f"S{i:02d}", "sill", np.array([x, -0.11, d.sill_spot_z]), np.array([0, -1.0, 0])))
    for i, z in enumerate(np.arange(0.60, 1.40 + 1e-9, p)):   # B필러 플랜지
        out.append(Spot(f"B{i:02d}", "b_pillar", np.array([d.pillar_x, -0.09, z]), np.array([0, -1.0, 0])))
    y0, y1 = d.member_y
    for i in range(d.member_spots):   # 크로스멤버-플로어, 위에서 쏜다 (타점 수 고정)
        y = y0 + d.member_first_spot + i * p
        out.append(Spot(f"M{i:02d}", "member", np.array([d.member_x, y, d.floor_z]), np.array([0, 0, 1.0]),
                        away=np.array([-1.0, 0, 0])))
    return out


def obstacles(d: Design) -> list[Box]:
    y0, y1 = d.member_y
    wx = d.member_x + d.flange_width
    px0, px1 = d.pillar_x - d.pillar_w / 2, d.pillar_x + d.pillar_w / 2
    return [
        Box("sill", np.array([0.0, -0.08, d.floor_z]), np.array([2.4, 0.08, d.sill_top])),
        Box("b_pillar", np.array([px0, -0.06, d.sill_top]), np.array([px1, 0.06, 1.45])),
        Box("member_wall", np.array([wx, y0, d.floor_z]), np.array([wx + 0.002, y1, d.floor_z + d.member_wall_height])),
        Box("floor", np.array([0.0, 0.08, d.floor_z - 0.002]), np.array([2.4, 0.9, d.floor_z - 0.001])),
    ]

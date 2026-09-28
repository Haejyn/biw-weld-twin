"""차체 하위 조립품 — 사이드 실 · B필러 · 플로어 크로스멤버를 판금 모자형 단면으로.

부품은 두께 1.2 mm 판재(얇은 상자)의 조합이다: 실은 바깥·안쪽 벽 + 윗판·아랫판 + 문 쪽 위 접합 플랜지,
B필러는 바깥판 + 앞뒤 벽 + 뒤 플랜지, 크로스멤버는 양 벽 + 윗판 + 바닥 접합 플랜지, 그리고 플로어 판.
타점은 두 판이 겹치는 플랜지 위에 놓는다. 공개 BIW CAD 에 타점 정보가 없어 치수는 설계 변수로 만든다.
단위 m. x = 차 길이 방향(라인 진행), y = 차 폭(로봇은 y<0 쪽), z = 높이.
"""
from dataclasses import dataclass, field

import numpy as np

T = 0.0012                 # 판 두께
SILL_HALF = 0.08           # 실 단면 반폭 (y)
SILL_DEPTH_BELOW = 0.10    # 실이 바닥 아래로 내려간 깊이
SILL_FLANGE = 0.025        # 실 위 접합 플랜지 높이
PILLAR_HALF_DEPTH = 0.06   # B필러 단면 반깊이 (y)
PILLAR_FLANGE = 0.025      # B필러 뒤 플랜지 폭
MEMBER_HALF_W = 0.04       # 크로스멤버 단면 반폭 (x)
MEMBER_LENGTH = 0.68
X_END = 2.40
PILLAR_TOP = 1.45


@dataclass
class Design:
    name: str
    flange_width: float = 0.020      # 크로스멤버 바닥 접합 플랜지 폭 (타점은 가운데)
    member_wall_height: float = 0.120  # 크로스멤버 높이
    spot_pitch: float = 0.050          # 크로스멤버 타점 간격
    floor_z: float = 0.30
    sill_top: float = 0.55
    member_y: tuple = (0.12, 0.80)     # 크로스멤버가 차 안쪽으로 뻗는 구간 (시작 = 실 안쪽 벽 근처)
    member_first_spot: float = 0.03    # 크로스멤버 첫 타점이 구간 시작에서 떨어진 거리
    member_spots: int = 13
    member_x: float = 1.20             # 크로스멤버 중심의 차 길이 방향 위치
    pillar_x: float = 1.20             # B필러 중심
    pillar_w: float = 0.16             # B필러 폭(차 길이 방향)
    sill_pillar_gap: float = 0.03      # 실 플랜지 타점을 B필러 앞뒤에서 띄우는 거리
    sill_spots: int = 45
    pillar_spots: int = 17
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
    part: str = ""


def _box(name, part, lo, hi):
    return Box(name, np.array(lo, float), np.array(hi, float), part)


def sill_spot_xs(d: Design):
    """실 위 플랜지 타점 x — B필러가 덮는 구간(앞뒤 sill_pillar_gap 여유)을 빼고 개수를 고정해 나눈다."""
    g = d.sill_pillar_gap
    px0, px1 = d.pillar_x - d.pillar_w / 2 - g, d.pillar_x + d.pillar_w / 2 + g
    segs = [(0.10, px0), (px1, X_END - 0.10)]
    lens = [max(b - a, 0.0) for a, b in segs]
    n0 = int(round(d.sill_spots * lens[0] / sum(lens)))
    n1 = d.sill_spots - n0
    xs = []
    for (a, b), n in zip(segs, (n0, n1)):
        if n > 0:
            xs += list(np.linspace(a, b, n)) if n > 1 else [(a + b) / 2]
    return xs


def spots(d: Design) -> list[Spot]:
    out = []
    zf = d.sill_top + SILL_FLANGE / 2
    for i, x in enumerate(sill_spot_xs(d)):          # 실 위 접합 플랜지(바깥·안쪽 판), 문 쪽에서 쏜다
        out.append(Spot(f"S{i:02d}", "sill", np.array([x, -T, zf]), np.array([0, -1.0, 0]),
                        away=np.array([0, 0, 1.0])))
    fx = d.pillar_x + d.pillar_w / 2 + PILLAR_FLANGE / 2
    for i, z in enumerate(np.linspace(d.sill_top + 0.05, PILLAR_TOP - 0.05, d.pillar_spots)):   # B필러 뒤 플랜지
        out.append(Spot(f"B{i:02d}", "b_pillar", np.array([fx, -PILLAR_HALF_DEPTH - T, z]), np.array([0, -1.0, 0]),
                        away=np.array([1.0, 0, 0])))
    y0 = d.member_y[0]
    mx = d.member_x + MEMBER_HALF_W + d.flange_width / 2
    for i in range(d.member_spots):                   # 크로스멤버-플로어 접합 플랜지, 위에서 쏜다
        y = y0 + d.member_first_spot + i * d.spot_pitch
        out.append(Spot(f"M{i:02d}", "member", np.array([mx, y, d.floor_z + T]), np.array([0, 0, 1.0]),
                        away=np.array([1.0, 0, 0])))
    return out


def obstacles(d: Design) -> list[Box]:
    zb = d.floor_z - SILL_DEPTH_BELOW
    st, fz = d.sill_top, d.floor_z
    px0, px1 = d.pillar_x - d.pillar_w / 2, d.pillar_x + d.pillar_w / 2
    y0 = d.member_y[0]
    y1 = y0 + MEMBER_LENGTH
    mx0, mx1 = d.member_x - MEMBER_HALF_W, d.member_x + MEMBER_HALF_W
    H = d.member_wall_height
    return [
        # 사이드 실 — 닫힌 단면 + 위 접합 플랜지(두 겹)
        _box("sill_outer", "sill", [0, -SILL_HALF - T, zb], [X_END, -SILL_HALF, st]),
        _box("sill_inner", "sill", [0, SILL_HALF, zb], [X_END, SILL_HALF + T, st]),
        _box("sill_top", "sill", [0, -SILL_HALF, st - T], [X_END, SILL_HALF, st]),
        _box("sill_bottom", "sill", [0, -SILL_HALF, zb], [X_END, SILL_HALF, zb + T]),
        _box("sill_flange", "sill", [0, -T, st], [X_END, T, st + SILL_FLANGE]),
        # B필러 — 바깥판 + 앞뒤 벽 + 뒤 플랜지
        _box("pillar_outer", "b_pillar", [px0, -PILLAR_HALF_DEPTH - T, st], [px1, -PILLAR_HALF_DEPTH, PILLAR_TOP]),
        _box("pillar_front", "b_pillar", [px0, -PILLAR_HALF_DEPTH, st], [px0 + T, PILLAR_HALF_DEPTH, PILLAR_TOP]),
        _box("pillar_rear", "b_pillar", [px1 - T, -PILLAR_HALF_DEPTH, st], [px1, PILLAR_HALF_DEPTH, PILLAR_TOP]),
        _box("pillar_flange", "b_pillar", [px1, -PILLAR_HALF_DEPTH - T, st], [px1 + PILLAR_FLANGE, -PILLAR_HALF_DEPTH + T, PILLAR_TOP]),
        # 크로스멤버 — 모자형: 양 벽 + 윗판 + 바닥 접합 플랜지(앞뒤)
        _box("member_wall_front", "member", [mx0 - T, y0, fz], [mx0, y1, fz + H]),
        _box("member_wall_rear", "member", [mx1, y0, fz], [mx1 + T, y1, fz + H]),
        _box("member_top", "member", [mx0, y0, fz + H - T], [mx1, y1, fz + H]),
        _box("member_flange_front", "member", [mx0 - d.flange_width, y0, fz], [mx0, y1, fz + T]),
        _box("member_flange_rear", "member", [mx1, y0, fz], [mx1 + d.flange_width, y1, fz + T]),
        # 플로어 판
        _box("floor", "floor", [0, SILL_HALF + T, fz - T], [X_END, 0.9, fz]),
    ]


def part_bounds(d: Design) -> dict:
    """부품별 판재를 감싸는 상자 — AI 특징(여유 거리) 계산용."""
    out = {}
    for b in obstacles(d):
        lo, hi = out.get(b.part, (b.lo, b.hi))
        out[b.part] = (np.minimum(lo, b.lo), np.maximum(hi, b.hi))
    return {k: Box(k, lo, hi, k) for k, (lo, hi) in out.items()}

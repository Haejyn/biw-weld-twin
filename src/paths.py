"""용접 순서의 이동 경로 · 로봇끼리 간섭.

- 타점 사이 이동: 후퇴(타점 → 접근점) → 관절 공간 직선 이동(접근점 → 다음 접근점) → 접근(→ 다음 타점).
  관절 공간 이동 중간 자세를 표본으로 팔·건 간섭을 재고, 걸리면 대기 자세(HOME)를 거쳐 돌아간다.
- 로봇끼리: 로봇마다 (시각, 관절각) 타임라인을 만들고, 우선순위 + 구역 예약 인터록으로 뒤 로봇이
  앞 로봇의 남은 경로와 겹치지 않을 때만 움직이게 한다(대기 시간이 사이클에 더해진다).
"""
import numpy as np

from reach import ARM_RADII, GUN_SEGMENTS, box_arrays, capsules_clear
from robot import JOINT_SPEED

WELD_S = 0.7                    # [가정] 타점당 가압·통전·유지·개방
ACCEL_FACTOR = 1.5              # [가정] 최대 관절속도만으로 잰 이동시간에 곱하는 가감속 보정
SETTLE_S = 0.15                 # [가정] 정착
PATH_SAMPLES = 8
HOME = np.concatenate([[0], np.radians([0, 0, -90, 0, 0, 0]), [0, 0]])
DT = 0.1
ROBOT_GAP = 0.05                # 로봇끼리 최소 여유 [가정]


def move_time(q0, q1) -> float:
    dq = np.abs(np.asarray(q1)[1:7] - np.asarray(q0)[1:7])
    return float(np.max(dq / JOINT_SPEED)) * ACCEL_FACTOR + SETTLE_S


def robot_capsules(robot, q):
    """팔 5토막 + 건 3토막 (월드 좌표)."""
    frames = robot.chain.forward_kinematics(q, full_kinematics=True)
    pts = [robot.to_world(f[:3, 3]) for f in frames]
    a = [pts[i] for i in range(2, 7)]
    b = [pts[i + 1] for i in range(2, 7)]
    r = [ARM_RADII[i] for i in range(2, 7)]
    tip, flange = pts[8], pts[7]
    axis = (flange - tip) / (np.linalg.norm(flange - tip) + 1e-12)
    for s0, s1, rr in GUN_SEGMENTS:
        a.append(tip + axis * s0); b.append(tip + axis * s1); r.append(rr)
    return np.array(a), np.array(b), r


def joint_path_clear(robot, q0, q1, arrays) -> bool:
    for t in np.linspace(0, 1, PATH_SAMPLES)[1:-1]:
        if not capsules_clear(*robot_capsules(robot, q0 + t * (q1 - q0)), arrays):
            return False
    return True


def plan_sequence(robot, verdicts, boxes):
    """타점 판정들(Verdict: q, q_pre)의 용접 순서와 타임라인. 돌려주는 것: (순서, 사이클, 키프레임, 경유 횟수)."""
    arrays = box_arrays(boxes)
    left, order, cur_pre, t = list(range(len(verdicts))), [], HOME, 0.0
    keys = [(0.0, HOME, None)]
    vias = 0
    while left:
        j = min(left, key=lambda k: move_time(cur_pre, verdicts[k].q_pre))
        v = verdicts[j]
        if joint_path_clear(robot, cur_pre, v.q_pre, arrays):
            t += move_time(cur_pre, v.q_pre)
        else:                                   # 대기 자세를 거쳐 돌아간다
            vias += 1
            t += move_time(cur_pre, HOME); keys.append((t, HOME, None))
            t += move_time(HOME, v.q_pre)
        keys.append((t, v.q_pre, None))
        t += move_time(v.q_pre, v.q); keys.append((t, v.q, j))
        t += WELD_S; keys.append((t, v.q, j))
        t += move_time(v.q, v.q_pre); keys.append((t, v.q_pre, None))
        cur_pre = v.q_pre
        order.append(j); left.remove(j)
    t += move_time(cur_pre, HOME) - SETTLE_S
    keys.append((t, HOME, None))
    return order, t, keys, vias


def q_at(keys, t):
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, q0, _), (t1, q1, _) in zip(keys, keys[1:]):
        if t0 <= t <= t1:
            return q0 if t1 == t0 else q0 + (t - t0) / (t1 - t0) * (q1 - q0)
    return keys[-1][1]


def capsule_gap(ca, cb) -> float:
    """두 캡슐 묶음 사이 최소 여유 (표본점 거리 − 두 반지름)."""
    ts = np.linspace(0, 1, 6)[None, :, None]
    pa = (ca[0][:, None] + ts * (ca[1] - ca[0])[:, None]).reshape(-1, 3)
    pb = (cb[0][:, None] + ts * (cb[1] - cb[0])[:, None]).reshape(-1, 3)
    ra, rb = np.repeat(ca[2], 6), np.repeat(cb[2], 6)
    d = np.linalg.norm(pa[:, None] - pb[None], axis=2) - ra[:, None] - rb[None]
    return float(d.min())


def _points(cap, n=6):
    ts = np.linspace(0, 1, n)[None, :, None]
    pts = (cap[0][:, None] + ts * (cap[1] - cap[0])[:, None]).reshape(-1, 3)
    return pts, np.repeat(cap[2], n)


def resolve_interference(robots, timelines, max_wait=40.0, future_step=0.3, window=None, priority=None):
    """우선순위 + 구역 예약 인터록. 번호가 앞선 로봇은 기다리지 않는다. 뒤 로봇은 0.1 s 마다 다음 자세가
    앞선 로봇들의 '남은 경로 전체'(0.3 s 간격 표본)와 ROBOT_GAP 이상 떨어져 있을 때만 그 자세로 들어가고,
    아니면 지금 자세(이미 안전이 확인된 자리)에서 기다린다 → 교착이 생기지 않는다(보수적인 대기).
    timelines: {r: keys}. 돌려주는 것: ({r: 대기 합}, 대기 한도를 넘으면 1 아니면 0)."""
    ids = list(priority) if priority is not None else sorted(timelines)
    waits = {r: 0.0 for r in ids}
    done = {}                                    # r → [(전역 시각, 표본점, 반지름)]
    for r in ids:
        keys = timelines[r]
        end = keys[-1][0]
        others = [(t, p, rad) for o in done for (t, p, rad) in done[o]]
        local, T, track = 0.0, 0.0, []
        pts, rad = _points(robot_capsules(robots[r], q_at(keys, 0.0)))
        track.append((T, pts, rad))
        while local < end:
            nxt = min(local + DT, end)
            npts, nrad = _points(robot_capsules(robots[r], q_at(keys, nxt)))
            hi = np.inf if window is None else T + window
            fut = [(p, rr) for (t, p, rr) in others if T - 1e-9 <= t <= hi]

            def clear_of(q_pts, q_rad):
                if not fut:
                    return True
                P = np.concatenate([p for p, _ in fut]); Rr = np.concatenate([rr for _, rr in fut])
                d = np.linalg.norm(q_pts[:, None] - P[None], axis=2) - q_rad[:, None] - Rr[None]
                return bool(d.min() >= ROBOT_GAP)

            if clear_of(npts, nrad):
                local, pts, rad = nxt, npts, nrad
            else:
                if window is not None and not clear_of(pts, rad):
                    return waits, 1              # 기다리는 자리마저 앞 로봇이 지나간다 → 이 배정은 못 쓴다
                waits[r] += DT
                if waits[r] > max_wait:
                    return waits, 1
            T += DT
            track.append((T, pts, rad))
        done[r] = [x for i, x in enumerate(track) if i % max(1, int(round(future_step / DT))) == 0] + [track[-1]]
    return waits, 0


def best_interlock(robots, timelines, base_times, **kw):
    """우선순위를 몇 가지로 바꿔 보고, 대기를 더한 가장 긴 사이클이 가장 짧은 것을 고른다."""
    ids = sorted(timelines)
    orders = {tuple(ids), tuple(reversed(ids)),                                  # 번호 순 · 역순
              tuple(sorted(ids, key=lambda r: -base_times[r])),                  # 바쁜 로봇 먼저
              tuple(sorted(ids, key=lambda r: base_times[r])),                   # 한가한 로봇 먼저
              tuple(sorted(ids, key=lambda r: abs(r - (ids[0] + ids[-1]) / 2)))}  # 가운데 먼저
    best = None
    for order in orders:
        waits, bad = resolve_interference(robots, timelines, priority=order, **kw)
        if bad:
            continue
        worst = max(base_times[r] + waits[r] for r in ids)
        if best is None or worst < best[0]:
            best = (worst, waits, order)
    if best is None:
        return {r: 0.0 for r in ids}, 1, None
    return best[1], 0, best[2]


ZONE = 1.0          # [가정] 두 로봇이 동시에 용접하면 안 되는 타점 간 차 길이 방향 거리 (간섭 구역 폭)


def visits_from_keys(keys):
    """plan_sequence 키프레임 → [(이동 시작, 방문 시작=접근점 도착, 방문 끝=후퇴 끝, 타점 j)]"""
    out, prev_end = [], 0.0
    for a in range(len(keys) - 3):
        (t0, _, j0), (t1, _, j1), (t2, _, j2), (t3, _, j3) = keys[a:a + 4]
        if j0 is None and j1 is not None and j2 == j1 and j3 is None:
            out.append((prev_end, t0, t3, j1))
            prev_end = t3
    return out


def zone_interlock(timelines, spot_xs, zone=ZONE):
    """간섭 구역 인터록. 로봇이 타점을 접근·용접·후퇴하는 동안 그 타점 둘레(±zone/2 가 아니라, 다른 로봇
    타점과의 거리 < zone)를 예약한다. 다른 로봇은 접근점 앞에서 기다린다. 기다리는 로봇은 구역을 쥐지 않아
    교착이 없다. timelines: {r: keys}, spot_xs: {r: [로봇 r 의 j번째 타점 x]}.
    돌려주는 것: ({r: 대기 합}, {r: 대기를 넣은 새 키프레임})."""
    plans = {r: visits_from_keys(k) for r, k in timelines.items()}
    ids = sorted(plans)
    idx = {r: 0 for r in ids}                 # 다음 방문 번호
    shift = {r: 0.0 for r in ids}             # 지금까지 누적된 대기
    ready = {r: (plans[r][0][1] if plans[r] else None) for r in ids}   # 다음 방문을 시작할 수 있는 시각
    busy = {}                                 # r → (끝 시각, x)
    inserted = {r: [] for r in ids}           # (원래 방문 시작 시각, 대기)
    T = 0.0
    while any(idx[r] < len(plans[r]) for r in ids):
        for r in ids:
            if r in busy and busy[r][0] <= T + 1e-9:
                del busy[r]
        for r in ids:
            if idx[r] >= len(plans[r]) or r in busy or ready[r] > T + 1e-9:
                continue
            _, v0, v1, j = plans[r][idx[r]]
            x = spot_xs[r][j]
            if all(abs(x - bx) >= zone for o, (_, bx) in busy.items() if o != r):
                wait = T - ready[r]
                if wait > 1e-9:
                    inserted[r].append((v0, wait))
                    shift[r] += wait
                busy[r] = (T + (v1 - v0), x)
                idx[r] += 1
                if idx[r] < len(plans[r]):
                    nxt = plans[r][idx[r]]
                    ready[r] = T + (v1 - v0) + (nxt[1] - v1)
        T = round(T + DT, 6)
    new_keys = {}
    for r, keys in timelines.items():
        out, add = [], 0.0
        pend = sorted(inserted[r])
        for t, q, j in keys:
            while pend and pend[0][0] <= t + 1e-9:
                v0, w = pend.pop(0)
                out.append((v0 + add, q_at(keys, v0), None))      # 접근점에서 기다린다
                add += w
            out.append((t + add, q, j))
        new_keys[r] = out
    return shift, new_keys

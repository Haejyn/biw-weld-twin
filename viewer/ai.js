// AI 제조성 대리 모델 (LightGBM) — 브라우저 추론.
// 특징은 src/surrogate.py design_features(engineered=True) 를 그대로 옮겼다.
// 형상은 src/dataset.py make_design + src/body.py v2 (판재 15장) spots()/obstacles()/part_bounds() 와 같다.
// 파이썬과 같은 순서로 같은 식을 계산해 float32 로 넣는다 — 값이 비트 단위로 같아야 트리 분기가 같다.

// src/body.py 상수
const T = 0.0012;
const SILL_HALF = 0.08;
const SILL_DEPTH_BELOW = 0.10;
const SILL_FLANGE = 0.025;
const PILLAR_HALF_DEPTH = 0.06;
const PILLAR_FLANGE = 0.025;
const MEMBER_HALF_W = 0.04;
const MEMBER_LENGTH = 0.68;
const X_END = 2.40;
const PILLAR_TOP = 1.45;
const SPOT_PITCH = 0.050;
const SILL_SPOTS = 45, PILLAR_SPOTS = 17, MEMBER_SPOTS = 13;
// seg_box_dist 표본 수 — src/surrogate.py 의 n 기본값. parity 특징 행으로 확인한다.
export const SEG_SAMPLES = 16;

// numpy.linspace(a, b, n): a + i*step, 마지막은 정확히 b
function linspace(a, b, n) {
  if (n === 1) return [a];
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => (i === n - 1 ? b : i * step + a));
}
// 파이썬 round() — 반올림에서 딱 .5 는 짝수 쪽
function pyRound(x) {
  const f = Math.floor(x), diff = x - f;
  if (diff > 0.5) return f + 1;
  if (diff < 0.5) return f;
  return f % 2 === 0 ? f : f + 1;
}
const id2 = (p, i) => `${p}${String(i).padStart(2, "0")}`;

export function makeDesign(v) {
  const y0 = v.member_y0;
  return {
    flange_width: v.flange_width, member_wall_height: v.member_wall_height, member_first_spot: v.member_first_spot,
    member_y: [y0, y0 + 0.68], member_x: v.member_x, pillar_x: v.pillar_x, pillar_w: v.pillar_w,
    sill_top: v.sill_top, floor_z: v.floor_z, sill_pillar_gap: v.sill_pillar_gap,
  };
}

export function sillSpotXs(d) {
  const g = d.sill_pillar_gap;
  const px0 = d.pillar_x - d.pillar_w / 2 - g, px1 = d.pillar_x + d.pillar_w / 2 + g;
  const segs = [[0.10, px0], [px1, X_END - 0.10]];
  const lens = segs.map(([a, b]) => Math.max(b - a, 0.0));
  const n0 = pyRound(SILL_SPOTS * lens[0] / (0 + lens[0] + lens[1]));
  const n1 = SILL_SPOTS - n0;
  const xs = [];
  segs.forEach(([a, b], k) => {
    const n = k === 0 ? n0 : n1;
    if (n > 0) xs.push(...(n > 1 ? linspace(a, b, n) : [(a + b) / 2]));
  });
  return xs;
}

export function spots(d) {
  const out = [];
  const zf = d.sill_top + SILL_FLANGE / 2;
  sillSpotXs(d).forEach((x, i) => out.push({ id: id2("S", i), group: "sill", pos: [x, -T, zf], normal: [0, -1, 0], away: [0, 0, 1] }));
  const fx = d.pillar_x + d.pillar_w / 2 + PILLAR_FLANGE / 2;
  linspace(d.sill_top + 0.05, PILLAR_TOP - 0.05, PILLAR_SPOTS).forEach((z, i) =>
    out.push({ id: id2("B", i), group: "b_pillar", pos: [fx, -PILLAR_HALF_DEPTH - T, z], normal: [0, -1, 0], away: [1, 0, 0] }));
  const y0 = d.member_y[0];
  const mx = d.member_x + MEMBER_HALF_W + d.flange_width / 2;
  for (let i = 0; i < MEMBER_SPOTS; i++) {
    const y = y0 + d.member_first_spot + i * SPOT_PITCH;
    out.push({ id: id2("M", i), group: "member", pos: [mx, y, d.floor_z + T], normal: [0, 0, 1], away: [1, 0, 0] });
  }
  return out;
}

export function obstacles(d) {
  const zb = d.floor_z - SILL_DEPTH_BELOW;
  const st = d.sill_top, fz = d.floor_z;
  const px0 = d.pillar_x - d.pillar_w / 2, px1 = d.pillar_x + d.pillar_w / 2;
  const y0 = d.member_y[0];
  const y1 = y0 + MEMBER_LENGTH;
  const mx0 = d.member_x - MEMBER_HALF_W, mx1 = d.member_x + MEMBER_HALF_W;
  const H = d.member_wall_height;
  const b = (name, part, lo, hi) => ({ name, part, lo, hi });
  return [
    b("sill_outer", "sill", [0, -SILL_HALF - T, zb], [X_END, -SILL_HALF, st]),
    b("sill_inner", "sill", [0, SILL_HALF, zb], [X_END, SILL_HALF + T, st]),
    b("sill_top", "sill", [0, -SILL_HALF, st - T], [X_END, SILL_HALF, st]),
    b("sill_bottom", "sill", [0, -SILL_HALF, zb], [X_END, SILL_HALF, zb + T]),
    b("sill_flange", "sill", [0, -T, st], [X_END, T, st + SILL_FLANGE]),
    b("pillar_outer", "b_pillar", [px0, -PILLAR_HALF_DEPTH - T, st], [px1, -PILLAR_HALF_DEPTH, PILLAR_TOP]),
    b("pillar_front", "b_pillar", [px0, -PILLAR_HALF_DEPTH, st], [px0 + T, PILLAR_HALF_DEPTH, PILLAR_TOP]),
    b("pillar_rear", "b_pillar", [px1 - T, -PILLAR_HALF_DEPTH, st], [px1, PILLAR_HALF_DEPTH, PILLAR_TOP]),
    b("pillar_flange", "b_pillar", [px1, -PILLAR_HALF_DEPTH - T, st], [px1 + PILLAR_FLANGE, -PILLAR_HALF_DEPTH + T, PILLAR_TOP]),
    b("member_wall_front", "member", [mx0 - T, y0, fz], [mx0, y1, fz + H]),
    b("member_wall_rear", "member", [mx1, y0, fz], [mx1 + T, y1, fz + H]),
    b("member_top", "member", [mx0, y0, fz + H - T], [mx1, y1, fz + H]),
    b("member_flange_front", "member", [mx0 - d.flange_width, y0, fz], [mx0, y1, fz + T]),
    b("member_flange_rear", "member", [mx1, y0, fz], [mx1 + d.flange_width, y1, fz + T]),
    b("floor", "floor", [0, SILL_HALF + T, fz - T], [X_END, 0.9, fz]),
  ];
}

export function partBounds(boxes) {
  const out = {};
  for (const b of boxes) {
    const cur = out[b.part];
    out[b.part] = cur
      ? { lo: cur.lo.map((x, k) => Math.min(x, b.lo[k])), hi: cur.hi.map((x, k) => Math.max(x, b.hi[k])) }
      : { lo: [...b.lo], hi: [...b.hi] };
  }
  return out;
}

// seg_box_dist: 선분을 n 점으로 나눠 상자까지 가장 가까운 거리
function segBoxDist(a, b, lo, hi, n = SEG_SAMPLES) {
  let best = Infinity;
  for (const t of linspace(0, 1, n)) {
    let s = 0;
    for (let k = 0; k < 3; k++) {
      const p = a[k] + t * (b[k] - a[k]);
      const d = Math.max(Math.max(lo[k] - p, p - hi[k]), 0);
      s += d * d;
    }
    best = Math.min(best, Math.sqrt(s));
  }
  return best;
}

const GUN_TILTS = [0.0, 15.0, 30.0];
const TS_NEAR = linspace(0.03, 0.20, 10);   // 전극·암
const TS_FAR = linspace(0.20, 0.45, 10);    // 건 몸체
const norm3 = (v) => Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);

/** 건 축을 0°/15°/30° 기울였을 때 판재까지 여유 — 타점이 붙어 있는 판은 뺀다.
 *  돌려주는 것: [전극·암 0°, 전극·암 최선, 몸체 최선] */
export function gunClearances(spot, plates) {
  const pos = spot.pos;
  const keep = plates.filter((b) => !(pos.every((x, k) => x >= b.lo[k] - 1e-6) && pos.every((x, k) => x <= b.hi[k] + 1e-6)));
  const nn = norm3(spot.normal);
  const n = spot.normal.map((x) => x / nn);
  const near = [], far = [];
  for (const t of GUN_TILTS) {
    const r = t * (Math.PI / 180);
    const c = Math.cos(r), sn = Math.sin(r);
    let a = [c * n[0] + sn * spot.away[0], c * n[1] + sn * spot.away[1], c * n[2] + sn * spot.away[2]];
    const an = norm3(a);
    a = a.map((x) => x / an);
    for (const [ts, out] of [[TS_NEAR, near], [TS_FAR, far]]) {
      let best = Infinity;
      for (const tt of ts) {
        const pt = [pos[0] + tt * a[0], pos[1] + tt * a[1], pos[2] + tt * a[2]];
        for (const b of keep) {
          let q = 0;
          for (let k = 0; k < 3; k++) {
            const d = Math.max(Math.max(b.lo[k] - pt[k], pt[k] - b.hi[k]), 0);
            q += d * d;
          }
          best = Math.min(best, Math.sqrt(q));
        }
      }
      out.push(best);
    }
  }
  return [near[0], Math.max(...near), Math.max(...far)];
}

const GROUPS = ["sill", "b_pillar", "member"];

/** (타점 × 로봇) 특징 행렬 — Float32Array, 행 = 타점 i 주, 로봇 r 부 */
export function designFeatures(v, meta, segSamples = SEG_SAMPLES) {
  const d = makeDesign(v);
  const S = spots(d), B = obstacles(d);
  const box = partBounds(B);
  const nF = meta.feature_names.length, nR = meta.bases.length;
  const X = new Float32Array(S.length * nR * nF);
  const sz = meta.shoulder_z;
  let o = 0;
  for (const s of S) {
    const gun = gunClearances(s, B);   // 타점마다 한 번
    for (const bb of meta.bases) {
      const base = bb.slice(0, 3);
      const overhead = bb[4] !== 0 ? 1 : 0;
      const rel = [s.pos[0] - base[0], s.pos[1] - base[1], s.pos[2] - base[2]];
      const dist = Math.sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
      const toward = (s.normal[0] * -rel[0] + s.normal[1] * -rel[1] + s.normal[2] * -rel[2]) / dist;
      const shoulder = [base[0] + 0, base[1] + 0, base[2] + (overhead ? -sz : sz)];
      const above = s.pos.map((p, k) => p + s.normal[k] * 0.45);
      const x = s.pos[0];
      const row = [
        ...meta.params.map((k) => v[k]),
        ...GROUPS.map((g) => (s.group === g ? 1 : 0)),
        ...s.pos, ...s.normal,
        ...base, overhead, ...rel,
        Math.hypot(rel[0], rel[1]), dist, toward,
        segBoxDist(shoulder, above, box.b_pillar.lo, box.b_pillar.hi, segSamples),
        segBoxDist(shoulder, above, box.sill.lo, box.sill.hi, segSamples),
        ...gun,
        x - v.pillar_x,
        Math.min(Math.abs(x - (v.pillar_x - v.pillar_w / 2)), Math.abs(x - (v.pillar_x + v.pillar_w / 2 + 0.025))),
        s.pos[1] - 0.08,
        v.sill_top - s.pos[2],
      ];
      X.set(row, o);
      o += nF;
    }
  }
  return { X, spots: S, boxes: B, nF, nR };
}

/** trees.json → 한 덩어리 배열. 자식: 0 이상 = 전역 노드 번호, 음수 = ~전역 잎 번호 */
export function compileModel(json) {
  let nNodes = 0, nLeaves = 0;
  for (const t of json.trees) { nNodes += t.f.length; nLeaves += t.v.length; }
  const feat = new Uint8Array(nNodes), thr = new Float32Array(nNodes);
  const left = new Int32Array(nNodes), right = new Int32Array(nNodes);
  const leaf = new Float64Array(nLeaves), root = new Int32Array(json.trees.length);
  let no = 0, lo = 0;
  json.trees.forEach((t, ti) => {
    const map = (c) => (c >= 0 ? no + c : ~(lo + ~c));
    root[ti] = t.f.length ? no : ~lo;
    for (let i = 0; i < t.f.length; i++) {
      feat[no + i] = t.f[i];
      thr[no + i] = t.t[i];            // float32 로 저장 → 비교가 파이썬(float32 입력)과 같다
      left[no + i] = map(t.l[i]);
      right[no + i] = map(t.r[i]);
    }
    for (let i = 0; i < t.v.length; i++) leaf[lo + i] = t.v[i];
    no += t.f.length;
    lo += t.v.length;
  });
  return { feat, thr, left, right, leaf, root, sigmoid: json.sigmoid };
}

export function predict(model, X, nF) {
  const n = X.length / nF;
  const out = new Float64Array(n);
  const { feat, thr, left, right, leaf, root, sigmoid } = model;
  for (let r = 0; r < n; r++) {
    const base = r * nF;
    let sum = 0;
    for (let t = 0; t < root.length; t++) {
      let node = root[t];
      while (node >= 0) node = X[base + feat[node]] <= thr[node] ? left[node] : right[node];
      sum += leaf[~node];
    }
    out[r] = 1 / (1 + Math.exp(-sigmoid * sum));
  }
  return out;
}

/** 설계 하나 판정: 타점마다 네 로봇 중 가장 높은 확률 */
export function judge(model, meta, v) {
  const { X, spots: S, boxes, nF, nR } = designFeatures(v, meta);
  const p = predict(model, X, nF);
  const best = S.map((_, i) => Math.max(...p.subarray(i * nR, i * nR + nR)));
  return { spots: S, boxes, pairProb: p, best, bad: best.map((b) => b < meta.threshold), X, nF };
}

export function parityCheck(model, meta) {
  let maxP = 0, maxF = 0, spotMismatch = 0, spotsTotal = 0, designMismatch = 0;
  for (const d of meta.parity) {
    const r = judge(model, meta, d.design);
    r.pairProb.forEach((p, k) => { maxP = Math.max(maxP, Math.abs(p - d.prob[k])); });
    d.features.forEach((row, k) => row.forEach((x, j) => { maxF = Math.max(maxF, Math.abs(r.X[k * r.nF + j] - x)); }));
    const nR = meta.bases.length;
    let anyDiff = false;
    for (let i = 0; i < r.spots.length; i++) {
      const py = Math.max(...d.prob.slice(i * nR, i * nR + nR)) < meta.threshold;
      spotsTotal++;
      if (py !== r.bad[i]) { spotMismatch++; anyDiff = true; }
    }
    if (anyDiff) designMismatch++;
  }
  return { designs: meta.parity.length, maxAbsProbDiff: maxP, maxAbsFeatureDiff: maxF, spotsTotal, spotMismatch, designMismatch };
}

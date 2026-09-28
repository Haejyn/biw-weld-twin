// AI 제조성 대리 모델 (LightGBM) — 브라우저 추론.
// 특징은 src/surrogate.py design_features(engineered=True) 를 그대로 옮겼다.
// 형상은 src/dataset.py make_design + src/body.py spots()/obstacles() 와 같다.

const PITCH = 0.05;

// numpy.arange(start, stop, step) 과 같은 값: delta = (start + step) - start, x_i = start + i * delta
function arange(start, stop, step) {
  const n = Math.ceil((stop - start) / step);
  const delta = (start + step) - start;
  return Array.from({ length: n }, (_, i) => start + i * delta);
}

export function makeDesign(v) {
  const y0 = v.member_y0;
  return {
    flange_width: v.flange_width, member_wall_height: v.member_wall_height, member_first_spot: v.member_first_spot,
    member_y: [y0, y0 + 0.68], member_x: v.member_x, pillar_x: v.pillar_x, pillar_w: v.pillar_w,
    sill_top: v.sill_top, floor_z: v.floor_z, sill_spot_z: (v.floor_z + v.sill_top) / 2, member_spots: 13,
  };
}

export function spots(d) {
  const out = [];
  arange(0.10, 2.30 + 1e-9, PITCH).forEach((x, i) =>
    out.push({ id: `S${String(i).padStart(2, "0")}`, group: "sill", pos: [x, -0.11, d.sill_spot_z], normal: [0, -1, 0] }));
  arange(0.60, 1.40 + 1e-9, PITCH).forEach((z, i) =>
    out.push({ id: `B${String(i).padStart(2, "0")}`, group: "b_pillar", pos: [d.pillar_x, -0.09, z], normal: [0, -1, 0] }));
  const y0 = d.member_y[0];
  for (let i = 0; i < d.member_spots; i++) {
    const y = y0 + d.member_first_spot + i * PITCH;
    out.push({ id: `M${String(i).padStart(2, "0")}`, group: "member", pos: [d.member_x, y, d.floor_z], normal: [0, 0, 1] });
  }
  return out;
}

export function obstacles(d) {
  const [y0, y1] = d.member_y;
  const wx = d.member_x + d.flange_width;
  const px0 = d.pillar_x - d.pillar_w / 2, px1 = d.pillar_x + d.pillar_w / 2;
  return [
    { name: "sill", lo: [0.0, -0.08, d.floor_z], hi: [2.4, 0.08, d.sill_top] },
    { name: "b_pillar", lo: [px0, -0.06, d.sill_top], hi: [px1, 0.06, 1.45] },
    { name: "member_wall", lo: [wx, y0, d.floor_z], hi: [wx + 0.002, y1, d.floor_z + d.member_wall_height] },
    { name: "floor", lo: [0.0, 0.08, d.floor_z - 0.002], hi: [2.4, 0.9, d.floor_z - 0.001] },
  ];
}

// numpy.linspace(0, 1, 16): i * (1/15), 마지막은 정확히 1
const TS = Array.from({ length: 16 }, (_, i) => (i === 15 ? 1 : i * (1 / 15)));
function segBoxDist(a, b, lo, hi) {
  let best = Infinity;
  for (const t of TS) {
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

const GROUPS = ["sill", "b_pillar", "member"];

/** (타점 × 로봇) 특징 행렬 — Float32Array, 행 = 타점 i 주, 로봇 r 부 */
export function designFeatures(v, meta) {
  const d = makeDesign(v);
  const S = spots(d), B = obstacles(d);
  const box = Object.fromEntries(B.map((b) => [b.name, b]));
  const nF = meta.feature_names.length, nR = meta.bases.length;
  const X = new Float32Array(S.length * nR * nF);
  let o = 0;
  for (const s of S) {
    for (const base of meta.bases) {
      const rel = [s.pos[0] - base[0], s.pos[1] - base[1], s.pos[2] - base[2]];
      const shoulder = [base[0], base[1], base[2] + meta.shoulder_z];
      const above = s.pos.map((p, k) => p + s.normal[k] * 0.45);
      const gunStart = s.pos.map((p, k) => p + s.normal[k] * 0.05);
      const row = [
        ...meta.params.map((k) => v[k]),
        ...GROUPS.map((g) => (s.group === g ? 1 : 0)),
        ...s.pos, ...s.normal,
        base[0], ...rel,
        Math.hypot(rel[0], rel[1]), Math.sqrt(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]),
        segBoxDist(shoulder, above, box.b_pillar.lo, box.b_pillar.hi),
        segBoxDist(shoulder, above, box.sill.lo, box.sill.hi),
        segBoxDist(gunStart, above, box.sill.lo, box.sill.hi),
        segBoxDist(gunStart, above, box.member_wall.lo, box.member_wall.hi),
        s.pos[0] - v.pillar_x,
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

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import * as AI from "./ai.js";

// KUKA KR210 L150 — data/robot/kr210l150_macro.xacro, src/robot.py 와 같은 값
const JOINTS = [
  { xyz: [-0.00262, 0.00097586, 0.33099], axis: "z" },
  { xyz: [0.35277, -0.037476, 0.4192], axis: "y" },
  { xyz: [-9.8483e-05, -0.1475, 1.2499], axis: "y" },
  { xyz: [0.95795, 0.184, -0.055059], axis: "x" },
  { xyz: [0.542, 0, 0], axis: "y" },
  { xyz: [0.1925, 0, 0], axis: "x" },
];
const FLANGE = [0.0375, 0, -0.00023924];
const MESHES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"];

const COLOR = {
  robots: [0x4da3ff, 0xa98bff],
  bad: 0xff5a4e,
  accent: 0x5ee1d4,
  body: 0xc9d2e0,
  arm: 0xd27a3c,
};
const GROUP_KO = { sill: "사이드실", b_pillar: "B필러", member: "크로스멤버" };
const REASON_KO = {
  unreachable: "도달 불가", gun_collision: "건 간섭", arm_collision: "팔 간섭", joint_limit: "관절 한계",
};

const $ = (id) => document.getElementById(id);
const DESIGN_LABEL = { A: "A 기준", B: "B 플랜지 확대", C: "C 첫 타점 이동" };
const designLabel = (name) => DESIGN_LABEL[name[0]] ?? name.replace(/_/, " ");

/* ---------- scene ---------- */
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x05070c, 9, 22);
const camera = new THREE.PerspectiveCamera(38, 1, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(-2.9, 3.9, 4.3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1.2, -0.95, 0.8);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 1.2;
controls.maxDistance = 14;
controls.update();

scene.add(new THREE.HemisphereLight(0xcfe0ff, 0x1a1410, 1.1));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(3, -4, 6);
scene.add(sun);
const rim = new THREE.DirectionalLight(0x88aaff, 0.8);
rim.position.set(-4, 5, 3);
scene.add(rim);

const grid = new THREE.GridHelper(14, 56, 0x2a3346, 0x151b27);
grid.rotation.x = Math.PI / 2;
grid.position.set(1.2, -0.6, 0);
grid.material.transparent = true;
grid.material.opacity = 0.55;
scene.add(grid);
const floor = new THREE.Mesh(
  new THREE.CircleGeometry(7, 64),
  new THREE.MeshBasicMaterial({ color: 0x0a0e16, transparent: true, opacity: 0.6, depthWrite: false }),
);
floor.position.set(1.2, -0.6, -0.002);
scene.add(floor);

function glowTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  const grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grad.addColorStop(0, "rgba(255,255,255,1)");
  grad.addColorStop(0.25, "rgba(255,240,200,0.8)");
  grad.addColorStop(1, "rgba(255,200,120,0)");
  g.fillStyle = grad;
  g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}
function crossTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.strokeStyle = "#ff5a4e";
  g.lineWidth = 12;
  g.lineCap = "round";
  g.beginPath();
  g.moveTo(14, 14); g.lineTo(50, 50);
  g.moveTo(50, 14); g.lineTo(14, 50);
  g.stroke();
  return new THREE.CanvasTexture(c);
}
const GLOW = glowTexture();
const CROSS = crossTexture();

/* ---------- robot ---------- */
const armMat = new THREE.MeshStandardMaterial({ color: COLOR.arm, roughness: 0.55, metalness: 0.15 });
const baseMat = new THREE.MeshStandardMaterial({ color: 0x3b3f47, roughness: 0.7, metalness: 0.2 });
const ghostMat = new THREE.MeshBasicMaterial({ color: 0x9fb4d6, transparent: true, opacity: 0.12, depthWrite: false });
const gunMat = new THREE.MeshStandardMaterial({ color: 0x3a404b, roughness: 0.5, metalness: 0.5 });
const tipMat = new THREE.MeshStandardMaterial({ color: 0xd49a68, roughness: 0.35, metalness: 0.8 });

function makeGun(len) {
  const gun = new THREE.Group();
  const along = (geom, x0, x1) => { geom.rotateZ(-Math.PI / 2); geom.translate((x0 + x1) / 2, 0, 0); return geom; };
  gun.add(new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.1, 0.1).translate(0.065, 0, 0), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.03, 0.04, 0.06, 20), 0.13, 0.19), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.014, 0.018, len - 0.24, 16), 0.19, len - 0.05), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.004, 0.013, 0.05, 16), len - 0.05, len), tipMat));
  // 열린 C 모양 고정 암
  const cArm = new THREE.Group();
  const seg = (a, b, r) => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, va.distanceTo(vb), 12), gunMat);
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    cArm.add(m);
  };
  seg([0.1, 0, -0.04], [0.1, 0, -0.15], 0.016);
  seg([0.1, 0, -0.15], [len + 0.03, 0, -0.15], 0.014);
  seg([len + 0.03, 0, -0.15], [len + 0.03, 0, -0.07], 0.012);
  gun.add(cArm);
  return gun;
}

// 링크 모양: 링크 좌표계 기준 Object3D 를 만드는 함수. 처음엔 STL, .dae 가 오면 바꾼다.
const shared = new Set();
const visuals = {};
// URDF <collision> 원점: base_link 만 roll -90°
function stlVisual(geo, name) {
  shared.add(geo);
  return (mat) => {
    const m = new THREE.Mesh(geo, mat);
    if (name === "base_link") m.rotation.x = -Math.PI / 2;
    return m;
  };
}
// URDF <visual> 원점: base_link, link_2~5 는 roll -90°. dae 는 inch 단위·Z_UP 이지만
// ColladaLoader 가 붙이는 위축 회전은 버리고 0.0254 배율과 URDF 회전만 쓴다(STL 과 겹치는지 확인함).
const VISUAL_ROLL = new Set(["base_link", "link_2", "link_3", "link_4", "link_5"]);
function daeVisual(collada, name) {
  const src = collada.scene;
  src.rotation.set(0, 0, 0);
  src.scale.setScalar(0.0254);
  src.traverse((o) => {
    if (!o.isMesh) return;
    shared.add(o.geometry);
    const mats = Array.isArray(o.material) ? o.material : [o.material];
    // 원래 색이 어두운 부품(모터·케이블)은 짙은 회색으로, 나머지는 로봇 몸체 색으로
    o.userData.dark = mats.every((m) => !m.color || m.color.r * 0.3 + m.color.g * 0.59 + m.color.b * 0.11 < 0.25);
  });
  return (mat) => {
    const wrap = new THREE.Group();
    if (VISUAL_ROLL.has(name)) wrap.rotation.x = -Math.PI / 2;
    const obj = src.clone(true);
    obj.traverse((o) => {
      if (o.isMesh) o.material = mat === armMat && o.userData.dark ? baseMat : mat === baseMat && !o.userData.dark ? baseLightMat : mat;
    });
    wrap.add(obj);
    return wrap;
  };
}
const baseLightMat = new THREE.MeshStandardMaterial({ color: 0x4a4f58, roughness: 0.6, metalness: 0.25 });

function buildRobot(robot, gunLength, ghost) {
  const root = new THREE.Group();
  root.position.set(...robot.base);
  root.rotation.z = robot.yaw;
  const mat = (i) => (ghost ? ghostMat : i === 0 ? baseMat : armMat);
  root.add(visuals.base_link(mat(0)));
  let parent = root;
  const rot = [];
  JOINTS.forEach((j, i) => {
    const origin = new THREE.Group();
    origin.position.set(...j.xyz);
    const r = new THREE.Group();
    origin.add(r);
    r.add(visuals[MESHES[i + 1]](mat(i + 1)));
    parent.add(origin);
    rot.push(r);
    parent = r;
  });
  const flange = new THREE.Group();
  flange.position.set(...FLANGE);
  parent.add(flange);
  if (!ghost) flange.add(makeGun(gunLength));
  const tcp = new THREE.Object3D();
  tcp.position.set(gunLength, 0, 0);
  flange.add(tcp);
  return { root, rot, tcp };
}
function setPose(r, q) {
  JOINTS.forEach((j, i) => { r.rot[i].rotation[j.axis] = q[i]; });
}

/* ---------- state ---------- */
let data, design, view = null;
let t = 0, playing = true, speed = 4;

let mode = "twin";            // "twin" = 설계안 재생, "ai" = AI 설계 검토
const ai = { ready: null, model: null, meta: null, v: null, robots: null };

function clearView() {
  if (!view) return;
  scene.remove(view.group);
  if (view.keep) view.group.remove(view.keep);   // AI 모드 로봇은 슬라이더마다 다시 만들지 않는다
  view.group.traverse((o) => { if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose(); });
}

// 차체 판넬 (반투명)
const panelMat = new THREE.MeshStandardMaterial({
  color: COLOR.body, transparent: true, opacity: 0.16, roughness: 0.6, depthWrite: false, side: THREE.DoubleSide,
});
const edgeMat = new THREE.LineBasicMaterial({ color: 0xaab6c8, transparent: true, opacity: 0.18 });
function addBoxes(group, boxes) {
  for (const b of boxes) {
    const size = b.hi.map((h, i) => Math.max(h - b.lo[i], 0.003));
    const geo = new THREE.BoxGeometry(...size);
    const m = new THREE.Mesh(geo, panelMat);
    m.position.set(...b.lo.map((l, i) => l + size[i] / 2));
    m.renderOrder = 2;
    group.add(m);
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
    e.position.copy(m.position);
    group.add(e);
  }
}

function buildDesign(d) {
  clearView();
  const group = new THREE.Group();
  const gunLength = data.gun_length;

  addBoxes(group, d.boxes);

  // 로봇
  const robots = {};
  for (const r of d.robots) {
    if (r.used) {
      robots[r.id] = buildRobot(r, gunLength, false);
      group.add(robots[r.id].root);
    } else {
      const g = new THREE.Group();
      g.add(visuals.base_link(ghostMat));
      g.position.set(...r.base);
      g.rotation.z = r.yaw;
      group.add(g);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.42, 0.5, 48),
        new THREE.MeshBasicMaterial({ color: 0x9fb4d6, transparent: true, opacity: 0.12, depthWrite: false }),
      );
      ring.position.set(r.base[0], r.base[1], 0.002);
      group.add(ring);
    }
  }
  const colorOf = {};
  d.paths.forEach((p, i) => { colorOf[p.robot] = COLOR.robots[i % COLOR.robots.length]; });

  // 타점
  const spotGeo = new THREE.SphereGeometry(0.014, 16, 12);
  const hitGeo = new THREE.SphereGeometry(0.03, 8, 6);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const spots = {}, hits = [];
  for (const s of d.spots) {
    const pos = new THREE.Vector3(...s.pos);
    const entry = { s, pos };
    if (s.robot == null) {
      const x = new THREE.Sprite(new THREE.SpriteMaterial({ map: CROSS, depthTest: false, transparent: true }));
      x.scale.setScalar(0.07);
      x.position.copy(pos);
      x.renderOrder = 10;
      group.add(x);
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: GLOW, color: COLOR.bad, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      halo.scale.setScalar(0.22);
      halo.position.copy(pos);
      group.add(halo);
      entry.halo = halo;
    } else {
      const base = new THREE.Color(colorOf[s.robot] ?? 0xffffff);
      const mat = new THREE.MeshStandardMaterial({ color: base, emissive: base, emissiveIntensity: 0.2, roughness: 0.4 });
      const m = new THREE.Mesh(spotGeo, mat);
      m.position.copy(pos);
      group.add(m);
      entry.mesh = m;
      entry.base = base;
    }
    const hit = new THREE.Mesh(hitGeo, hitMat);
    hit.position.copy(pos);
    hit.userData.id = s.id;
    group.add(hit);
    hits.push(hit);
    spots[s.id] = entry;
  }

  // 용접 이벤트: 도착 시각 ~ weld_end 시각
  const welds = [];
  for (const p of d.paths) {
    p.frames.forEach((f, i) => {
      if (f.weld_end) welds.push({ id: f.spot, t0: p.frames[i - 1].t, t1: f.t, robot: p.robot });
    });
  }
  const sparks = d.paths.map(() => {
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({
      map: GLOW, color: 0xffe2a8, transparent: true, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending,
    }));
    sp.visible = false;
    sp.renderOrder = 11;
    group.add(sp);
    return sp;
  });

  scene.add(group);
  const duration = Math.max(...d.paths.map((p) => p.frames[p.frames.length - 1].t)) + 1.5;
  view = { group, robots, spots, hits, welds, sparks, duration };
  if (!window.__fk?.[d.name]) verifyFK(d);
  $("timeSlider").max = duration.toFixed(2);
  renderNumbers(d);
}

function verifyFK(d) {
  let max = 0, maxAng = 0, n = 0, worst = null;
  const w = new THREE.Vector3();
  const byId = Object.fromEntries(d.spots.map((s) => [s.id, s]));
  for (const p of d.paths) {
    const r = view.robots[p.robot];
    for (const f of p.frames) {
      if (!f.spot) continue;
      setPose(r, f.q);
      r.root.updateMatrixWorld(true);
      r.tcp.getWorldPosition(w);
      const e = w.distanceTo(new THREE.Vector3(...byId[f.spot].pos));
      const fl = r.tcp.parent.getWorldPosition(new THREE.Vector3());
      const axisErr = THREE.MathUtils.radToDeg(w.clone().sub(fl).normalize().angleTo(new THREE.Vector3(...byId[f.spot].approach)));
      maxAng = Math.max(maxAng, axisErr);
      n++;
      if (e > max) { max = e; worst = f.spot; }
    }
  }
  const mm = max * 1000;
  window.__fk = window.__fk || {};
  window.__fk[d.name] = { maxErrMm: mm, maxAxisDeg: maxAng, frames: n, worst };
  console.log(`[FK] ${d.name}: TCP vs spot.pos max error ${mm.toFixed(4)} mm, gun axis vs approach ${maxAng.toFixed(3)} deg over ${n} frames (worst ${worst})`);
  if (mm > 2) console.warn(`[FK] ${d.name}: error above 2 mm`);
}

function poseAt(frames, time) {
  if (time <= frames[0].t) return frames[0].q;
  for (let i = 1; i < frames.length; i++) {
    const b = frames[i];
    if (time <= b.t) {
      const a = frames[i - 1];
      const u = b.t > a.t ? (time - a.t) / (b.t - a.t) : 1;
      return a.q.map((v, k) => v + (b.q[k] - v) * u);
    }
  }
  return frames[frames.length - 1].q;
}

function applyTime(time) {
  const d = design;
  for (const p of d.paths) setPose(view.robots[p.robot], poseAt(p.frames, time));
  for (const e of Object.values(view.spots)) {
    if (!e.mesh) continue;
    e.state = 0;
  }
  const active = {};
  for (const w of view.welds) {
    const e = view.spots[w.id];
    if (time >= w.t1) e.state = Math.max(e.state, 2);
    else if (time >= w.t0) { e.state = 1; active[w.robot] = { e, u: (time - w.t0) / (w.t1 - w.t0) }; }
  }
  for (const e of Object.values(view.spots)) {
    if (!e.mesh) continue;
    const m = e.mesh.material;
    if (e.state === 0) { m.emissiveIntensity = 0.12; m.color.copy(e.base).multiplyScalar(0.45); e.mesh.scale.setScalar(1); }
    else if (e.state === 1) { m.color.set(0xffffff); m.emissiveIntensity = 1.6; e.mesh.scale.setScalar(1.7); }
    else { m.color.copy(e.base); m.emissiveIntensity = 0.9; e.mesh.scale.setScalar(1.15); }
  }
  d.paths.forEach((p, i) => {
    const sp = view.sparks[i];
    const a = active[p.robot];
    sp.visible = !!a;
    if (a) {
      sp.position.copy(a.e.pos);
      sp.scale.setScalar(0.12 + 0.1 * Math.sin(a.u * Math.PI));
    }
  });
  $("timeSlider").value = time;
  $("timeLabel").textContent = `${time.toFixed(1)} / ${view.duration.toFixed(1)} s`;
}

/* ---------- numbers card ---------- */
function renderNumbers(d) {
  d.paths.slice(0, 2).forEach((p, i) => { $(`lg${i}`).textContent = `로봇 ${p.robot + 1} 담당`; });
  const bad = d.spots.filter((s) => s.robot == null).length;
  const cyc = d.paths.map((p, i) => {
    const pct = Math.min(100, (p.cycle_s / d.budget_s) * 100);
    const slack = d.budget_s - p.cycle_s;
    return `<div class="cyc">
      <div class="row"><span>로봇 ${p.robot + 1}</span><span>${p.cycle_s.toFixed(1)} / ${d.budget_s.toFixed(0)} s</span></div>
      <div class="bar"><i class="r${i}" style="width:${pct}%"></i></div>
      <div class="slack">여유 ${slack.toFixed(1)} s · 타점 ${p.frames.filter((f) => f.weld_end).length}개</div>
    </div>`;
  }).join("");
  $("numbers").innerHTML = `
    <div class="big-row">
      <div class="big"><div class="v">${d.spots.length}</div><div class="k">타점 수</div></div>
      <div class="big ${bad ? "bad" : ""}"><div class="v">${bad}</div><div class="k">못 쏘는 타점</div></div>
      <div class="big"><div class="v">${d.paths.length}</div><div class="k">로봇 대수</div></div>
    </div>
    <p class="change">${d.notes}</p>
    <h3>로봇별 사이클타임 · 예산 ${d.budget_s.toFixed(0)} s</h3>
    ${cyc}`;
}

/* ---------- tooltip ---------- */
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let pinned = null;
function pick(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
  const hit = raycaster.intersectObjects(view.hits, false)[0];
  return hit ? view.spots[hit.object.userData.id] : null;
}
function showTip(e, ev) {
  const tip = $("tip");
  if (!e) { tip.hidden = true; return; }
  const s = e.s;
  const rows = [["부위", GROUP_KO[s.group] ?? s.group]];
  if (s.ai != null) {
    rows.push(["AI 판정", s.bad ? "못 쏨" : s.border ? "경계 근처" : "쏠 수 있음"]);
    rows.push(["가장 높은 확률", `${s.ai.toFixed(2)} (기준 ${ai.meta.threshold})`]);
    rows.push(["그 로봇", `후보 ${s.bestRobot + 1}`]);
  } else if (s.robot != null) {
    rows.push(["담당 로봇", `로봇 ${s.robot + 1}`]);
    rows.push(["기울임", s.tilt ? `${s.tilt}°` : "없음"]);
  } else {
    rows.push(["담당 로봇", "없음"]);
  }
  let html = `<b>${s.id}</b>` + rows.map(([k, v]) => `<div class="row"><span>${k}</span><span>${v}</span></div>`).join("");
  if (s.robot == null && s.reasons) {
    html += `<div class="row"><span>못 쏘는 이유</span><span class="reason">${s.reasons.map((r) => REASON_KO[r] ?? r).join(", ")}</span></div>`;
  }
  tip.innerHTML = html;
  tip.hidden = false;
  const rect = stage.getBoundingClientRect();
  let x = ev.clientX - rect.left + 14, y = ev.clientY - rect.top + 14;
  x = Math.min(x, rect.width - tip.offsetWidth - 8);
  y = Math.min(y, rect.height - tip.offsetHeight - 70);
  tip.style.left = `${Math.max(8, x)}px`;
  tip.style.top = `${Math.max(8, y)}px`;
}
renderer.domElement.addEventListener("pointermove", (ev) => {
  if (!view || pinned) return;
  const e = pick(ev);
  renderer.domElement.style.cursor = e ? "pointer" : "";
  showTip(e, ev);
});
renderer.domElement.addEventListener("click", (ev) => {
  if (!view) return;
  const e = pick(ev);
  pinned = e;
  showTip(e, ev);
});
renderer.domElement.addEventListener("pointerleave", () => { if (!pinned) $("tip").hidden = true; });

/* ---------- controls ---------- */
const playBtn = $("playBtn");
function setPlaying(v) { playing = v; playBtn.classList.toggle("playing", v); }
playBtn.onclick = () => { if (t >= view.duration) t = 0; setPlaying(!playing); };
$("timeSlider").addEventListener("input", (ev) => { t = parseFloat(ev.target.value); applyTime(t); });
$("speeds").addEventListener("click", (ev) => {
  const b = ev.target.closest("button");
  if (!b) return;
  speed = +b.dataset.speed;
  for (const x of $("speeds").children) x.classList.toggle("on", x === b);
});

function setTab(i) {
  for (const [k, b] of [...$("designTabs").children].entries()) b.classList.toggle("on", k === i);
}
const TWIN_LEGEND = $("legend").innerHTML;

// 모드마다 기본 시점: 재생은 로봇과 차체 전체, AI 검토는 차체 타점 가까이
const VIEWS = {
  twin: { pos: [-2.9, 3.9, 4.3], target: [1.2, -0.95, 0.8] },
  ai: { pos: [-0.9, 2.9, 3.0], target: [1.15, -0.2, 0.55] },
};
function setView(name) {
  camera.position.set(...VIEWS[name].pos);
  controls.target.set(...VIEWS[name].target);
  controls.update();
}

function selectDesign(i) {
  if (mode === "ai") setView("twin");
  mode = "twin";
  stage.classList.remove("ai");
  $("legend").innerHTML = TWIN_LEGEND;
  design = data.designs[i];
  pinned = null;
  $("tip").hidden = true;
  buildDesign(design);
  t = 0;
  applyTime(t);
  setTab(i);
}

/* ---------- AI 설계 검토 ---------- */
const PARAM_KO = {
  flange_width: "플랜지 폭", member_wall_height: "크로스멤버 벽 높이", member_first_spot: "첫 타점 거리",
  member_y0: "크로스멤버 시작 위치", member_x: "크로스멤버 위치", pillar_x: "B필러 위치", pillar_w: "B필러 폭",
  sill_top: "실 높이", floor_z: "바닥 높이",
};
const AI_NOTE = "AI 1차 판정 · 경계 근처는 시뮬레이터로 확정 (시험 400개 설계: 못 쏘는 타점 재현율 96.3%, 설계 판정 정확도 98.3%)";
const BORDER = 0.15;   // 기준보다 이만큼 위까지는 '경계 근처'로 칠한다

function loadAI() {
  ai.ready ??= Promise.all([
    fetch("ai/trees.json").then((r) => r.json()),
    fetch("ai/meta.json").then((r) => r.json()),
  ]).then(([trees, meta]) => {
    ai.model = AI.compileModel(trees);
    ai.meta = meta;
    const t0 = performance.now();
    const par = AI.parityCheck(ai.model, meta);
    par.ms = +(performance.now() - t0).toFixed(1);
    par.pass = par.maxAbsProbDiff < 1e-4 && par.spotMismatch === 0;
    window.__aiParity = par;
    console.log(`[AI parity] ${par.designs} designs: max |p_js - p_py| = ${par.maxAbsProbDiff.toExponential(2)}, ` +
      `max feature diff ${par.maxAbsFeatureDiff.toExponential(2)}, spot decisions mismatched ${par.spotMismatch}/${par.spotsTotal} ` +
      `(designs ${par.designMismatch}) → ${par.pass ? "PASS" : "FAIL"}`);
    if (!par.pass) console.warn("[AI parity] 파이썬과 어긋남");
  });
  return ai.ready;
}

async function enterAI() {
  await loadAI();
  if (mode !== "ai") setView("ai");
  mode = "ai";
  setPlaying(false);
  stage.classList.add("ai");
  pinned = null;
  $("tip").hidden = true;
  setTab(data.designs.length);
  ai.v ??= { ...ai.meta.base };
  renderAIPanel();
  $("legend").innerHTML = `
    <div class="lg"><i class="x">×</i>AI 예측 못 쏨</div>
    <div class="lg"><i class="dot border"></i>경계 근처</div>
    <div class="lg"><i class="dot okay"></i>쏠 수 있음</div>
    <div class="lg"><i class="dot ghost"></i>후보 로봇 위치 4곳</div>`;
  runAI();
}

function renderAIPanel() {
  const { meta } = ai;
  const mm = (x) => `${Math.round(x * 1000)} mm`;
  $("numbers").innerHTML = `
    <div class="big-row two">
      <div class="big" id="aiBadBox"><div class="v" id="aiBad">–</div><div class="k">AI 예측 못 쏘는 타점</div></div>
      <div class="big"><div class="v" id="aiMs">–</div><div class="k">판정 시간</div></div>
    </div>
    <div class="sliders">
      ${meta.params.map((k) => `
        <label class="sl">
          <span class="sl-top"><span>${PARAM_KO[k] ?? k}</span><b id="val_${k}">${mm(ai.v[k])}</b></span>
          <input type="range" data-k="${k}" min="${meta.space[k][0]}" max="${meta.space[k][1]}" step="0.001" value="${ai.v[k]}">
        </label>`).join("")}
    </div>
    <div class="ai-foot">
      <button class="chip" id="aiReset">A안으로 되돌리기</button>
    </div>
    <p class="note">${AI_NOTE}</p>`;
  $("numbers").querySelectorAll("input[type=range]").forEach((el) => {
    el.addEventListener("input", () => {
      ai.v[el.dataset.k] = parseFloat(el.value);
      $(`val_${el.dataset.k}`).textContent = mm(ai.v[el.dataset.k]);
      runAI();
    });
  });
  $("aiReset").onclick = () => { ai.v = { ...ai.meta.base }; renderAIPanel(); runAI(); };
}

function runAI() {
  const t0 = performance.now();
  const res = AI.judge(ai.model, ai.meta, ai.v);
  const ms = performance.now() - t0;
  const nBad = res.bad.filter(Boolean).length;
  $("aiBad").textContent = `${nBad}개`;
  $("aiBadBox").classList.toggle("bad", nBad > 0);
  $("aiMs").textContent = `${ms < 10 ? ms.toFixed(1) : Math.round(ms)} ms`;
  buildAIScene(res);
  window.__aiLast = { bad: nBad, ms, v: { ...ai.v } };
}

function buildAIScene(res) {
  clearView();
  const group = new THREE.Group();
  if (!ai.robots) {
    ai.robots = new THREE.Group();
    for (const [i, base] of ai.meta.bases.entries()) {
      const r = buildRobot({ base, yaw: Math.PI / 2 }, data.gun_length, false);
      setPose(r, [0, 0, -Math.PI / 2, 0, 0, 0]);
      ai.robots.add(r.root);
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.42, 0.5, 48),
        new THREE.MeshBasicMaterial({ color: 0x9fb4d6, transparent: true, opacity: 0.12, depthWrite: false }),
      );
      ring.position.set(base[0], base[1], 0.002);
      ai.robots.add(ring);
    }
  }
  group.add(ai.robots);
  addBoxes(group, res.boxes);

  const spotGeo = new THREE.SphereGeometry(0.014, 16, 12);
  const hitGeo = new THREE.SphereGeometry(0.03, 8, 6);
  const hitMat = new THREE.MeshBasicMaterial({ visible: false });
  const okMat = new THREE.MeshStandardMaterial({ color: 0xc9d2e0, emissive: 0xc9d2e0, emissiveIntensity: 0.15, roughness: 0.4 });
  const borderMat = new THREE.MeshStandardMaterial({ color: 0xffb24a, emissive: 0xffb24a, emissiveIntensity: 0.6, roughness: 0.4 });
  const spots = {}, hits = [];
  const nR = ai.meta.bases.length;
  res.spots.forEach((s, i) => {
    const pos = new THREE.Vector3(...s.pos);
    const probs = res.pairProb.subarray(i * nR, i * nR + nR);
    const bestRobot = probs.indexOf(Math.max(...probs));
    const border = !res.bad[i] && res.best[i] < ai.meta.threshold + BORDER;
    if (res.bad[i]) {
      const x = new THREE.Sprite(new THREE.SpriteMaterial({ map: CROSS, depthTest: false, transparent: true }));
      x.scale.setScalar(0.07);
      x.position.copy(pos);
      x.renderOrder = 10;
      group.add(x);
      const halo = new THREE.Sprite(new THREE.SpriteMaterial({
        map: GLOW, color: COLOR.bad, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending,
      }));
      halo.scale.setScalar(0.22);
      halo.position.copy(pos);
      group.add(halo);
    } else {
      const m = new THREE.Mesh(spotGeo, border ? borderMat : okMat);
      m.position.copy(pos);
      group.add(m);
    }
    const hit = new THREE.Mesh(hitGeo, hitMat);
    hit.position.copy(pos);
    hit.userData.id = s.id;
    group.add(hit);
    hits.push(hit);
    spots[s.id] = { s: { ...s, ai: res.best[i], bad: res.bad[i], border, bestRobot }, pos };
  });
  scene.add(group);
  view = { group, keep: ai.robots, robots: {}, spots, hits, welds: [], sparks: [], duration: 1 };
}

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  // 좁은 화면에서는 뒤로 물러나 차체 전체가 들어오게
  camera.fov = w < 600 ? 52 : 38;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

/* ---------- line panel ---------- */
const line = { s: 0, playing: false, rate: 2.5, alts: [] };
const linePlay = $("linePlay");
function setLinePlaying(v) { line.playing = v; linePlay.classList.toggle("playing", v); }

function buildLine() {
  const L = data.line;
  const box = $("alts");
  box.innerHTML = "";
  const stats = L.alternatives.map((a) => ({
    stops: a.cars.filter((c) => c.stop_s > 0).length,
    total: a.cars.reduce((x, c) => x + c.stop_s, 0),
  }));
  const bestIdx = stats.reduce((bi, s, i) => (s.total < stats[bi].total ? i : bi), 0);
  line.alts = L.alternatives.map((a, i) => {
    const el = document.createElement("div");
    el.className = "alt" + (i === bestIdx ? " best" : "");
    const newShare = Math.round((a.cars.filter((c) => c.new).length / a.cars.length) * 100);
    el.innerHTML = `
      <div class="alt-top">
        <div><div class="alt-name">${a.name}</div>
        <div class="alt-meta">스테이션 ${a.stations}곳 · 택트 ${L.takt_s} s · 여유창 ${Math.round(L.drift * 100)}% · 신차 ${newShare}%</div></div>
        <div class="counters">
          <div class="counter" data-k="stops"><div class="v">0</div><div class="k">라인 정지 횟수</div></div>
          <div class="counter" data-k="total"><div class="v">0.0 s</div><div class="k">누적 정지 시간</div></div>
          <div class="counter" data-k="cars"><div class="v">0</div><div class="k">투입 / 60대</div></div>
        </div>
      </div>
      <canvas></canvas>`;
    box.appendChild(el);
    return { a, el, canvas: el.querySelector("canvas"), stops: el.querySelector('[data-k="stops"]'),
      total: el.querySelector('[data-k="total"]'), cars: el.querySelector('[data-k="cars"] .v') };
  });
}

function drawLine() {
  const L = data.line;
  const takt = L.takt_s, win = takt * (1 + L.drift);
  const s = line.s;
  const accent = "#5ee1d4";
  for (const A of line.alts) {
    const { a, canvas } = A;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (canvas.width !== Math.round(W * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const n = a.stations, gap = W < 420 ? 3 : 5;
    const bw = (W - gap * (n - 1)) / n;
    const top = 22, bh = H - 72, taktY = top + bh * (1 - takt / win);
    const cur = Math.floor(s);

    let stops = 0, total = 0;
    a.cars.forEach((c, k) => {
      if (c.stop_s > 0 && k + c.stop_station <= cur) { stops++; total += c.stop_s; }
    });

    for (let j = 0; j < n; j++) {
      const x = j * (bw + gap);
      const k = cur - j;
      const car = k >= 0 && k < a.cars.length ? a.cars[k] : null;
      const lag = car ? car.lag_s[j] : 0;
      const stopHere = car && car.stop_s > 0 && car.stop_station === j;
      // 바탕: 택트 아래는 조금 밝게, 여유창 구간은 어둡게
      roundRect(g, x, top, bw, bh, 6);
      g.fillStyle = "rgba(255,255,255,0.035)";
      g.fill();
      g.save();
      roundRect(g, x, top, bw, bh, 6);
      g.clip();
      g.fillStyle = "rgba(255,255,255,0.045)";
      g.fillRect(x, taktY, bw, top + bh - taktY);
      if (lag > 0) {
        const hOk = (Math.min(lag, takt) / win) * bh;
        g.fillStyle = car.new ? "rgba(94,225,212,0.42)" : "rgba(238,241,246,0.26)";
        g.fillRect(x, top + bh - hOk, bw, hOk);
        if (lag > takt) {
          const hOver = ((lag - takt) / win) * bh;
          g.fillStyle = "rgba(255,178,74,0.85)";
          g.fillRect(x, taktY - hOver, bw, hOver);
        }
      }
      if (stopHere) {
        const pulse = 1 - (s - cur);
        g.fillStyle = `rgba(255,90,78,${0.35 + 0.5 * pulse})`;
        g.fillRect(x, top, bw, bh);
      }
      g.restore();
      if (stopHere) {
        g.fillStyle = "#ff5a4e";
        g.font = "600 11px Pretendard, sans-serif";
        g.textAlign = "center";
        g.fillText(`정지 +${car.stop_s.toFixed(1)}s`, clamp(x + bw / 2, 30, W - 30), 13);
      }
      g.fillStyle = "rgba(154,163,178,0.9)";
      g.font = "11px Pretendard, sans-serif";
      g.textAlign = "center";
      if (bw > 14 || j % 2 === 0) g.fillText(String(j + 1), x + bw / 2, H - 6);
    }

    // 차: 스테이션 아래 줄에서 흐른다
    const carY = top + bh + 10, carH = 12;
    for (let k = Math.max(0, cur - n); k <= Math.min(a.cars.length - 1, cur); k++) {
      const p = s - k;
      if (p < 0 || p >= n) continue;
      // 택트 대부분은 스테이션에 서 있고 마지막 20% 에 다음 칸으로 옮겨 간다
      const j = Math.floor(p), u = clamp((p - j - 0.8) / 0.2, 0, 1);
      const slide = u * u * (3 - 2 * u);
      const cx = (j + slide) * (bw + gap);
      const w = Math.max(6, bw * 0.62);
      const x = cx + (bw - w) / 2;
      g.globalAlpha = j === n - 1 ? 1 - slide : 1;
      roundRect(g, x, carY, w, carH, 3);
      g.fillStyle = a.cars[k].new ? accent : "rgba(238,241,246,0.6)";
      g.fill();
      g.globalAlpha = 1;
    }

    A.stops.querySelector(".v").textContent = stops;
    A.total.querySelector(".v").textContent = `${total.toFixed(1)} s`;
    A.stops.classList.toggle("hot", stops > 0);
    A.total.classList.toggle("hot", total > 0);
    A.cars.textContent = Math.min(a.cars.length, Math.max(0, cur + 1));
  }
}
function roundRect(g, x, y, w, h, r) {
  g.beginPath();
  g.roundRect ? g.roundRect(x, y, w, h, r) : g.rect(x, y, w, h);
}
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

linePlay.onclick = () => {
  const end = 60 + Math.max(...data.line.alternatives.map((a) => a.stations));
  if (line.s >= end) line.s = 0;
  setLinePlaying(!line.playing);
};
$("lineReset").onclick = () => { line.s = 0; drawLine(); };
new ResizeObserver(() => data && drawLine()).observe($("alts"));

/* ---------- loop ---------- */
const clock = new THREE.Clock();
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (view && playing && mode === "twin") {
    t += dt * speed;
    if (t > view.duration) t = 0;
    applyTime(t);
  }
  if (data && line.playing) {
    const end = 60 + Math.max(...data.line.alternatives.map((a) => a.stations));
    line.s = Math.min(end, line.s + dt * line.rate);
    if (line.s >= end) setLinePlaying(false);
    drawLine();
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

/* ---------- boot ---------- */
async function boot() {
  const loader = new STLLoader();
  const [json, ...meshes] = await Promise.all([
    fetch("data/twin_data.json").then((r) => r.json()),
    ...MESHES.map((m) => loader.loadAsync(`meshes/${m}.stl`)),
  ]);
  data = json;
  MESHES.forEach((m, i) => { meshes[i].computeVertexNormals(); visuals[m] = stlVisual(meshes[i], m); });

  const tabs = $("designTabs");
  data.designs.forEach((d, i) => {
    const b = document.createElement("button");
    b.textContent = designLabel(d.name);
    b.setAttribute("role", "tab");
    b.onclick = () => selectDesign(i);
    tabs.appendChild(b);
  });
  const aiTab = document.createElement("button");
  aiTab.textContent = "AI 설계 검토";
  aiTab.className = "ai-tab";
  aiTab.setAttribute("role", "tab");
  aiTab.onclick = () => enterAI();
  tabs.appendChild(aiTab);
  loadAI().catch((err) => { console.error("[AI] 모델을 못 읽음", err); aiTab.disabled = true; });
  resize();
  // 모든 설계안 FK 검증 — 콘솔에 남기고 window.__fk 로 노출
  for (const d of data.designs) buildDesign(d);
  selectDesign(0);
  setPlaying(true);
  buildLine();
  drawLine();
  setLinePlaying(true);
  $("loading").remove();
  const all = Object.values(window.__fk).map((x) => x.maxErrMm);
  console.log(`[FK] overall max error ${Math.max(...all).toFixed(6)} mm`);
  frame();
  loadSmoothMeshes();
}

// 매끈한 .dae(약 14 MB)는 STL 로 먼저 띄운 뒤 뒤에서 받아 바꾼다. 하나라도 실패하면 STL 유지.
async function loadSmoothMeshes() {
  try {
    const { ColladaLoader } = await import("three/addons/loaders/ColladaLoader.js");
    const loaded = await Promise.all(MESHES.map((m) => new ColladaLoader().loadAsync(`meshes/${m}.dae`)));
    MESHES.forEach((m, i) => { visuals[m] = daeVisual(loaded[i], m); });
    if (ai.robots) { ai.robots.traverse((o) => { if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose(); }); ai.robots = null; }
    if (mode === "ai") {
      runAI();
    } else {
      const keepT = t;
      buildDesign(design);
      t = keepT;
      applyTime(t);
    }
    window.__meshes = "dae";
  } catch (err) {
    console.warn("[meshes] .dae 를 못 읽어 STL 로 둡니다", err);
    window.__meshes = "stl";
  }
}
boot().catch((err) => {
  console.error(err);
  $("loading").textContent = "데이터를 불러오지 못했습니다";
});

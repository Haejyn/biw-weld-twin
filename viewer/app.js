import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { STLLoader } from "three/addons/loaders/STLLoader.js";
import * as AI from "./ai.js";

/* =====================================================================
   상수 — KUKA KR210 L150 (data/robot/kr210l150_macro.xacro, src/robot.py 와 같은 값)
   ===================================================================== */
const JOINTS = [
  { xyz: [-0.00262, 0.00097586, 0.33099], axis: "z" },
  { xyz: [0.35277, -0.037476, 0.4192], axis: "y" },
  { xyz: [-9.8483e-05, -0.1475, 1.2499], axis: "y" },
  { xyz: [0.95795, 0.184, -0.055059], axis: "x" },
  { xyz: [0.542, 0, 0], axis: "y" },
  { xyz: [0.1925, 0, 0], axis: "x" },
];
const FLANGE = [0.0375, 0, -0.00023924];
const HOME = [0, 0, -Math.PI / 2, 0, 0, 0];
const MESHES = ["base_link", "link_1", "link_2", "link_3", "link_4", "link_5", "link_6"];
const ROBOT_HUES = [0x4da3ff, 0xa98bff];
const ROBOT_CSS = ["var(--r0)", "var(--r1)"];
const ACCENT = 0x5ee1d4;
const BAD = 0xff5a4e;

const GROUP_KO = { sill: "실", b_pillar: "B필러", member: "크로스멤버" };
const GROUP_PREFIX = { sill: "S", b_pillar: "B", member: "M" };
const BOX_KO = { sill: "실", b_pillar: "B필러", member_wall: "크로스멤버 벽", floor: "바닥" };
const REASON_KO = { unreachable: "도달 불가", gun_collision: "건 간섭", arm_collision: "팔 간섭", joint_limit: "관절 한계" };
const DESIGN_LABEL = { A: "A 기준", B: "B 플랜지 확대", C: "C 첫 타점 이동" };
const designLabel = (name) => DESIGN_LABEL[name[0]] ?? name.replace(/_/, " ");
const PARAM_KO = {
  flange_width: "플랜지 폭", member_wall_height: "크로스멤버 벽 높이", member_first_spot: "첫 타점 거리",
  member_y0: "크로스멤버 시작 위치", member_x: "크로스멤버 위치", pillar_x: "B필러 위치", pillar_w: "B필러 폭",
  sill_top: "실 높이", floor_z: "바닥 높이",
};
const PARAMS = Object.keys(PARAM_KO);
const AI_NOTE = "AI 1차 판정 · 경계 근처는 시뮬레이터로 확정 (시험 400개 설계: 못 쏘는 타점 재현율 96.3%, 설계 판정 정확도 98.3%)";
const BORDER = 0.15;   // 기준 확률보다 이만큼 위까지는 '경계 근처'
// 도달 범위(근사, 플랜지까지 — 건 제외): 어깨 오프셋 + 상완 + 전완·손목 + 플랜지
const REACH = 0.35277 + 1.2499 + Math.hypot(0.95795 + 0.542, 0.055) + 0.1925 + 0.0375;

const ICONS = {
  cursor: '<path d="M3.5 2.5 12 7.2l-3.9 1.1-1.6 3.9z"/>',
  orbit: '<ellipse cx="8" cy="8" rx="6.2" ry="2.6"/><path d="M8 1.8v12.4"/><path d="m11.8 3.9 1.9.9-.5 2"/>',
  pan: '<path d="M8 1.8v12.4M1.8 8h12.4M8 1.8 6.4 3.4M8 1.8l1.6 1.6M8 14.2l-1.6-1.6M8 14.2l1.6-1.6M1.8 8l1.6-1.6M1.8 8l1.6 1.6M14.2 8l-1.6-1.6M14.2 8l-1.6 1.6"/>',
  fit: '<path d="M2 5.5V2h3.5M10.5 2H14v3.5M14 10.5V14h-3.5M5.5 14H2v-3.5"/><rect x="5.5" y="5.5" width="5" height="5"/>',
  front: '<path d="M3 5h8v8H3zM3 5l2.5-2.5h8v8L11 13M11 5l2.5-2.5"/><rect x="3" y="5" width="8" height="8" fill="currentColor" fill-opacity=".3" stroke="none"/>',
  side: '<path d="M3 5h8v8H3zM3 5l2.5-2.5h8v8L11 13M11 5l2.5-2.5"/><path d="M11 5l2.5-2.5v8L11 13z" fill="currentColor" fill-opacity=".35" stroke="none"/>',
  top: '<path d="M3 5h8v8H3zM3 5l2.5-2.5h8v8L11 13M11 5l2.5-2.5"/><path d="M3 5l2.5-2.5h8L11 5z" fill="currentColor" fill-opacity=".35" stroke="none"/>',
  iso: '<path d="M8 1.8 13.6 5v6.1L8 14.2l-5.6-3.1V5z"/><path d="M2.4 5 8 8.1 13.6 5M8 8.1v6.1"/>',
  body: '<path d="M1.8 7h12.4v4.5H1.8z"/><path d="M6 7V2.8h2.6V7"/>',
  spots: '<circle cx="3.8" cy="8" r="1.6"/><circle cx="8" cy="8" r="1.6"/><circle cx="12.2" cy="8" r="1.6"/>',
  robot: '<path d="M2.5 14h7M6 14v-2.5l2.6-5 4.4 1.2"/><circle cx="6" cy="11.3" r="1.1"/><circle cx="8.6" cy="6.3" r="1.1"/><path d="m13 7.7.8 1.8"/>',
  reach: '<circle cx="8" cy="8" r="6" stroke-dasharray="2 2"/><circle cx="8" cy="8" r="1.2"/>',
  label: '<path d="M2.5 2.5h5l6 6-5 5-6-6z"/><circle cx="5.5" cy="5.5" r="1"/>',
  play: '<path d="M5 3.3v9.4L12.5 8z" fill="currentColor"/>',
  pause: '<path d="M5.5 3.5v9M10.5 3.5v9" stroke-width="2.2"/>',
  rewind: '<path d="M4 3.5v9"/><path d="M12.5 3.5v9L6 8z" fill="currentColor"/>',
  dockLeft: '<rect x="2" y="3" width="12" height="10" rx="1"/><path d="M6.2 3v10"/>',
  dockRight: '<rect x="2" y="3" width="12" height="10" rx="1"/><path d="M9.8 3v10"/>',
  eye: '<path d="M1.5 8S4 3.8 8 3.8 14.5 8 14.5 8 12 12.2 8 12.2 1.5 8 1.5 8z"/><circle cx="8" cy="8" r="2"/>',
  eyeOff: '<path d="M1.5 8S4 3.8 8 3.8 14.5 8 14.5 8 12 12.2 8 12.2 1.5 8 1.5 8z" opacity=".45"/><path d="M2.5 13.5 13.5 2.5"/>',
  folder: '<path d="M2 4.5h4l1.2 1.3H14V12.5H2z"/>',
  box: '<path d="M2.5 5 8 2.5 13.5 5v6L8 13.5 2.5 11z"/><path d="M2.5 5 8 7.5 13.5 5M8 7.5v6"/>',
  spot: '<circle cx="8" cy="8" r="3"/>',
  doc: '<path d="M4 1.8h5.5L12 4.3v9.9H4z"/><path d="M9.5 1.8v2.5H12"/>',
  ai: '<path d="M8 1.8 9.4 6.6 14.2 8 9.4 9.4 8 14.2 6.6 9.4 1.8 8l4.8-1.4z"/>',
};
const icon = (name) => `<svg class="ic" viewBox="0 0 16 16">${ICONS[name] ?? ""}</svg>`;

const $ = (id) => document.getElementById(id);
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const mm = (m, d = 1) => (m * 1000).toFixed(d);
const vecMm = (v, d = 1) => v.map((x) => mm(x, d)).join(", ");
const esc = (s) => String(s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);

/* =====================================================================
   로그
   ===================================================================== */
const LV = { info: "INFO", ok: "OK", warn: "WARN", err: "ERR" };
function log(level, msg) {
  const li = document.createElement("li");
  li.className = level;
  const ts = new Date().toTimeString().slice(0, 8);
  li.innerHTML = `<span class="ts">${ts}</span><span class="lv">${LV[level]}</span><span>${esc(msg)}</span>`;
  const ol = $("log");
  ol.appendChild(li);
  if (ol.children.length > 400) ol.firstChild.remove();
  ol.scrollTop = ol.scrollHeight;
  $("logCount").textContent = ol.children.length;
}

/* =====================================================================
   렌더러 · 장면 · 카메라
   ===================================================================== */
const stage = $("stage");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.setClearColor(0x000000, 0);
stage.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(36, 1, 0.05, 200);
camera.up.set(0, 0, 1);
camera.position.set(-2.9, 3.9, 4.3);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1.2, -0.95, 0.8);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
controls.maxDistance = 40;
controls.update();

scene.add(new THREE.HemisphereLight(0xdfe6f2, 0x24201c, 1.05));
const sun = new THREE.DirectionalLight(0xffffff, 2.0);
sun.position.set(3, -4, 6);
scene.add(sun);
const rim = new THREE.DirectionalLight(0xa8b8d8, 0.7);
rim.position.set(-4, 5, 3);
scene.add(rim);

const grid = new THREE.GridHelper(16, 64, 0x44464d, 0x2c2e33);
grid.rotation.x = Math.PI / 2;
grid.position.set(1.2, -0.6, 0);
grid.material.transparent = true;
grid.material.opacity = 0.8;
scene.add(grid);

// 좌표축 기즈모 — 작은 별도 장면을 왼쪽 아래에 겹쳐 그린다
const gizmo = { scene: new THREE.Scene(), cam: new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0.1, 10), size: 84 };
function textSprite(text, color) {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.fillStyle = color;
  g.font = "bold 40px sans-serif";
  g.textAlign = "center";
  g.textBaseline = "middle";
  g.fillText(text, 32, 34);
  const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  s.scale.setScalar(0.55);
  return s;
}
[["X", [1, 0, 0], "#e5605a"], ["Y", [0, 1, 0], "#7cc36e"], ["Z", [0, 0, 1], "#5b9df0"]].forEach(([n, d, col]) => {
  const dir = new THREE.Vector3(...d);
  gizmo.scene.add(new THREE.ArrowHelper(dir, new THREE.Vector3(), 1, new THREE.Color(col), 0.28, 0.14));
  const s = textSprite(n, col);
  s.position.copy(dir.clone().multiplyScalar(1.32));
  gizmo.scene.add(s);
});

/* =====================================================================
   질감 · 재질 · 로봇
   ===================================================================== */
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
function ringTexture() {
  const c = document.createElement("canvas");
  c.width = c.height = 64;
  const g = c.getContext("2d");
  g.strokeStyle = "#5ee1d4";
  g.lineWidth = 6;
  g.beginPath();
  g.arc(32, 32, 24, 0, Math.PI * 2);
  g.stroke();
  return new THREE.CanvasTexture(c);
}
const GLOW = glowTexture(), CROSS = crossTexture(), RING = ringTexture();

const armMat = new THREE.MeshStandardMaterial({ color: 0xd27a3c, roughness: 0.55, metalness: 0.15 });
const baseMat = new THREE.MeshStandardMaterial({ color: 0x3b3f47, roughness: 0.7, metalness: 0.2 });
const baseLightMat = new THREE.MeshStandardMaterial({ color: 0x4a4f58, roughness: 0.6, metalness: 0.25 });
const ghostMat = new THREE.MeshBasicMaterial({ color: 0xa9b3c4, transparent: true, opacity: 0.14, depthWrite: false });
const gunMat = new THREE.MeshStandardMaterial({ color: 0x3a404b, roughness: 0.5, metalness: 0.5 });
const tipMat = new THREE.MeshStandardMaterial({ color: 0xd49a68, roughness: 0.35, metalness: 0.8 });
const panelMat = new THREE.MeshStandardMaterial({
  color: 0xc9d2e0, transparent: true, opacity: 0.16, roughness: 0.6, depthWrite: false, side: THREE.DoubleSide,
});
const edgeMat = new THREE.LineBasicMaterial({ color: 0xaab6c8, transparent: true, opacity: 0.28 });
const selEdgeMat = new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity: 0.95, depthTest: false });

function makeGun(len) {
  const gun = new THREE.Group();
  const along = (geom, x0, x1) => { geom.rotateZ(-Math.PI / 2); geom.translate((x0 + x1) / 2, 0, 0); return geom; };
  gun.add(new THREE.Mesh(new THREE.BoxGeometry(0.13, 0.1, 0.1).translate(0.065, 0, 0), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.03, 0.04, 0.06, 20), 0.13, 0.19), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.014, 0.018, len - 0.24, 16), 0.19, len - 0.05), gunMat));
  gun.add(new THREE.Mesh(along(new THREE.CylinderGeometry(0.004, 0.013, 0.05, 16), len - 0.05, len), tipMat));
  const seg = (a, b, r) => {
    const va = new THREE.Vector3(...a), vb = new THREE.Vector3(...b);
    const m = new THREE.Mesh(new THREE.CylinderGeometry(r, r, va.distanceTo(vb), 12), gunMat);
    m.position.copy(va).add(vb).multiplyScalar(0.5);
    m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), vb.clone().sub(va).normalize());
    gun.add(m);
  };
  // 열린 C 모양 고정 암
  seg([0.1, 0, -0.04], [0.1, 0, -0.15], 0.016);
  seg([0.1, 0, -0.15], [len + 0.03, 0, -0.15], 0.014);
  seg([len + 0.03, 0, -0.15], [len + 0.03, 0, -0.07], 0.012);
  return gun;
}

// 링크 모양: 링크 좌표계 기준 Object3D 를 만드는 함수.
const shared = new Set();
const visualSets = { stl: null, dae: null };
let visuals = {};
let meshKind = "stl";
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

function buildRobot(base, yaw, gunLength, ghost) {
  const root = new THREE.Group();
  root.position.set(...base);
  root.rotation.z = yaw;
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
  return { root, rot, tcp, q: [...HOME] };
}
function setPose(r, q) {
  JOINTS.forEach((j, i) => { r.rot[i].rotation[j.axis] = q[i]; });
  r.q = q;
}

/* =====================================================================
   상태
   ===================================================================== */
let data = null;
let docIndex = 0;              // 0..n-1 = 설계안, n = AI 설계 검토
let mode = "twin";
let design = null;
let view = null;               // 지금 그려진 장면
let t = 0, playing = true, speed = 4;
let sel = null;                // 선택된 트리 노드 키 (예: "spot:M00")
let tool = "select";
const vis = { body: true, spots: true, robots: true, reach: false, labels: false };
const openNodes = new Set(["body", "spots", "robots"]);
const ai = { ready: null, model: null, meta: null, v: null, res: null, robots: null, robotsKind: null, ms: 0 };
const fkAll = {};

const isVisible = (key) => vis[key] !== false;

/* =====================================================================
   장면 만들기
   ===================================================================== */
function disposeView() {
  if (!view) return;
  scene.remove(view.group);
  if (view.keep) view.group.remove(view.keep);
  view.group.traverse((o) => { if (o.geometry && !shared.has(o.geometry)) o.geometry.dispose(); });
  view = null;
}

function addBoxes(group, boxes) {
  const out = {};
  for (const b of boxes) {
    const size = b.hi.map((h, i) => Math.max(h - b.lo[i], 0.003));
    const geo = new THREE.BoxGeometry(...size);
    const m = new THREE.Mesh(geo, panelMat);
    m.position.set(...b.lo.map((l, i) => l + size[i] / 2));
    m.renderOrder = 2;
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(geo), edgeMat);
    e.position.copy(m.position);
    group.add(m, e);
    out[b.name] = { b, objs: [m, e], geo, pos: m.position.clone() };
  }
  return out;
}

function addBadMarker(group, pos) {
  const x = new THREE.Sprite(new THREE.SpriteMaterial({ map: CROSS, depthTest: false, transparent: true }));
  x.scale.setScalar(0.07);
  x.position.copy(pos);
  x.renderOrder = 10;
  const halo = new THREE.Sprite(new THREE.SpriteMaterial({
    map: GLOW, color: BAD, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending,
  }));
  halo.scale.setScalar(0.22);
  halo.position.copy(pos);
  group.add(x, halo);
  return [x, halo];
}

const spotGeo = new THREE.SphereGeometry(0.014, 16, 12);
const hitGeo = new THREE.SphereGeometry(0.03, 8, 6);
const hitMat = new THREE.MeshBasicMaterial({ visible: false });
shared.add(spotGeo); shared.add(hitGeo);

function addHit(group, id, pos) {
  const hit = new THREE.Mesh(hitGeo, hitMat);
  hit.position.copy(pos);
  hit.userData.id = id;
  group.add(hit);
  return hit;
}

function reachRing(base) {
  const pts = [];
  for (let i = 0; i <= 128; i++) {
    const a = (i / 128) * Math.PI * 2;
    pts.push(new THREE.Vector3(base[0] + REACH * Math.cos(a), base[1] + REACH * Math.sin(a), 0.004));
  }
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const line = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: ACCENT, dashSize: 0.08, gapSize: 0.06, transparent: true, opacity: 0.55 }));
  line.computeLineDistances();
  return line;
}

function buildTwin(d) {
  disposeView();
  const group = new THREE.Group();
  const boxes = addBoxes(group, d.boxes);
  const reach = new THREE.Group();
  group.add(reach);

  const robots = {};
  const hueOf = {};
  d.paths.forEach((p, i) => { hueOf[p.robot] = i % ROBOT_HUES.length; });
  for (const r of d.robots) {
    if (r.used) {
      const rig = buildRobot(r.base, r.yaw, data.gun_length, false);
      group.add(rig.root);
      robots[r.id] = { id: r.id, base: r.base, used: true, rig, objs: [rig.root], hue: hueOf[r.id] };
      reach.add(reachRing(r.base));
    } else {
      const g = new THREE.Group();
      g.add(visuals.base_link(ghostMat));
      g.position.set(...r.base);
      g.rotation.z = r.yaw;
      const ring = new THREE.Mesh(
        new THREE.RingGeometry(0.42, 0.5, 48),
        new THREE.MeshBasicMaterial({ color: 0xa9b3c4, transparent: true, opacity: 0.14, depthWrite: false }),
      );
      ring.position.set(r.base[0], r.base[1], 0.002);
      group.add(g, ring);
      robots[r.id] = { id: r.id, base: r.base, used: false, objs: [g, ring], ghost: g };
    }
  }

  const spots = {}, hits = [];
  for (const s of d.spots) {
    const pos = new THREE.Vector3(...s.pos);
    const e = { s, pos, objs: [], group: s.group, bad: s.robot == null };
    if (e.bad) e.objs = addBadMarker(group, pos);
    else {
      const baseCol = new THREE.Color(ROBOT_HUES[hueOf[s.robot] ?? 0]);
      const mat = new THREE.MeshStandardMaterial({ color: baseCol, emissive: baseCol, emissiveIntensity: 0.2, roughness: 0.4 });
      e.mesh = new THREE.Mesh(spotGeo, mat);
      e.mesh.position.copy(pos);
      e.baseCol = baseCol;
      group.add(e.mesh);
      e.objs = [e.mesh];
    }
    e.hit = addHit(group, s.id, pos);
    hits.push(e.hit);
    spots[s.id] = e;
  }

  const welds = [];
  for (const p of d.paths) {
    p.frames.forEach((f, i) => {
      if (f.weld_end) {
        welds.push({ id: f.spot, t0: p.frames[i - 1].t, t1: f.t, robot: p.robot });
        spots[f.spot].weld = { t0: p.frames[i - 1].t, t1: f.t };
      }
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
  view = { kind: "twin", group, boxes, robots, spots, hits, welds, sparks, reach, duration, hueOf };
  if (!fkAll[d.name]) verifyFK(d);
}

/* =====================================================================
   FK 검증 — 전극 끝(TCP)이 타점 좌표와 같은지, 건 축이 접근 방향과 같은지
   ===================================================================== */
function verifyFK(d) {
  let max = 0, maxAng = 0, n = 0, worst = null;
  const w = new THREE.Vector3(), fl = new THREE.Vector3();
  const byId = Object.fromEntries(d.spots.map((s) => [s.id, s]));
  for (const p of d.paths) {
    const r = view.robots[p.robot].rig;
    for (const f of p.frames) {
      if (!f.spot) continue;
      setPose(r, f.q);
      r.root.updateMatrixWorld(true);
      r.tcp.getWorldPosition(w);
      r.tcp.parent.getWorldPosition(fl);
      const e = w.distanceTo(new THREE.Vector3(...byId[f.spot].pos));
      const ang = THREE.MathUtils.radToDeg(w.clone().sub(fl).normalize().angleTo(new THREE.Vector3(...byId[f.spot].approach)));
      maxAng = Math.max(maxAng, ang);
      n++;
      if (e > max) { max = e; worst = f.spot; }
    }
  }
  const errMm = max * 1000;
  fkAll[d.name] = { maxErrMm: errMm, maxAxisDeg: maxAng, frames: n, worst };
  window.__fk = fkAll;
  const msg = `[FK] ${d.name}: TCP vs spot.pos max error ${errMm.toFixed(4)} mm, gun axis vs approach ${maxAng.toFixed(3)} deg over ${n} frames (worst ${worst})`;
  console.log(msg);
  log(errMm < 2 ? "ok" : "err", `FK 검증 ${designLabel(d.name)}: 전극 끝-타점 최대 오차 ${errMm.toExponential(2)} mm, 건 축 ${maxAng.toFixed(3)}° (${n} 프레임)`);
}

/* =====================================================================
   재생
   ===================================================================== */
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
  if (!view || view.kind !== "twin") return;
  for (const p of design.paths) setPose(view.robots[p.robot].rig, poseAt(p.frames, time));
  const active = {};
  for (const e of Object.values(view.spots)) e.state = 0;
  for (const w of view.welds) {
    const e = view.spots[w.id];
    if (time >= w.t1) e.state = 2;
    else if (time >= w.t0) { e.state = 1; active[w.robot] = { e, u: (time - w.t0) / (w.t1 - w.t0) }; }
  }
  for (const e of Object.values(view.spots)) {
    if (!e.mesh) continue;
    const m = e.mesh.material;
    if (e.state === 0) { m.emissiveIntensity = 0.12; m.color.copy(e.baseCol).multiplyScalar(0.45); e.mesh.scale.setScalar(1); }
    else if (e.state === 1) { m.color.set(0xffffff); m.emissiveIntensity = 1.6; e.mesh.scale.setScalar(1.7); }
    else { m.color.copy(e.baseCol); m.emissiveIntensity = 0.9; e.mesh.scale.setScalar(1.15); }
  }
  design.paths.forEach((p, i) => {
    const sp = view.sparks[i], a = active[p.robot];
    sp.visible = !!a && vis.spots;
    if (a) { sp.position.copy(a.e.pos); sp.scale.setScalar(0.12 + 0.1 * Math.sin(a.u * Math.PI)); }
  });
  $("timeLabel").textContent = `${time.toFixed(2)} / ${view.duration.toFixed(2)} s`;
  drawScrub();
}

function setPlaying(v) {
  playing = v && mode === "twin";
  for (const id of ["tbPlay", "tlPlay"]) $(id).innerHTML = icon(playing ? "pause" : "play");
}
function setSpeed(s) {
  speed = s;
  for (const b of $("speeds").children) b.classList.toggle("on", +b.dataset.speed === s);
  syncChecks();
}

/* =====================================================================
   보이기 · 장면 트리
   ===================================================================== */
function applyVisibility() {
  if (!view) return;
  for (const [name, b] of Object.entries(view.boxes)) b.objs.forEach((o) => { o.visible = vis.body && isVisible(`box:${name}`); });
  for (const e of Object.values(view.spots)) {
    const on = vis.spots && isVisible(`grp:${e.group}`);
    e.objs.forEach((o) => { o.visible = on; });
    e.hit.visible = on;
  }
  for (const r of Object.values(view.robots)) r.objs.forEach((o) => { o.visible = vis.robots && isVisible(`robot:${r.id}`); });
  if (view.reach) view.reach.visible = vis.reach && vis.robots;
  document.querySelectorAll("[data-act^='toggle:']").forEach((b) => {
    const k = b.dataset.act.split(":")[1];
    b.classList.toggle("on", !!vis[k]);
  });
  syncChecks();
  updateSelectionMarkers();
}

function spotBadCount(group) {
  return Object.values(view.spots).filter((e) => e.group === group && e.bad).length;
}

function renderTree() {
  if (!view) return;
  const row = (key, depth, ico, name, extra = "", opts = {}) => {
    const hasKids = opts.kids;
    const closed = hasKids && !openNodes.has(key);
    const eyeKey = opts.eye;
    const hidden = eyeKey && vis[eyeKey] === false;
    return `<div class="node${sel === key ? " sel" : ""}${closed ? " closed" : ""}${hidden ? " hidden-obj" : ""}" data-node="${key}" style="padding-left:${4 + depth * 12}px">
      <span class="tw">${hasKids ? '<i class="caret"></i>' : ""}</span>
      <span class="ico">${ico}</span><span class="nm">${name}</span>${extra}
      ${eyeKey ? `<button class="eye" data-eye="${eyeKey}" aria-label="보이기">${icon(hidden ? "eyeOff" : "eye")}</button>` : ""}
    </div>`;
  };
  const kids = (key, html) => `<div class="children${openNodes.has(key) ? "" : " closed"}">${html}</div>`;
  let h = "";
  h += row("body", 0, icon("folder"), "차체", "", { kids: true, eye: "body" });
  h += kids("body", Object.keys(view.boxes).map((n) => row(`box:${n}`, 1, icon("box"), BOX_KO[n] ?? n, "", { eye: `box:${n}` })).join(""));
  const nSpots = Object.keys(view.spots).length;
  h += row("spots", 0, icon("folder"), "타점", `<span class="badge">${nSpots}</span>`, { kids: true, eye: "spots" });
  let sh = "";
  for (const g of ["sill", "b_pillar", "member"]) {
    const list = Object.values(view.spots).filter((e) => e.group === g);
    const bad = list.filter((e) => e.bad).length;
    const badge = `${bad ? `<span class="badge bad">${bad} 못 쏨</span>` : ""}<span class="badge">${list.length}</span>`;
    sh += row(`grp:${g}`, 1, icon("spots"), `${GROUP_KO[g]} ${GROUP_PREFIX[g]}`, badge, { kids: true, eye: `grp:${g}` });
    sh += kids(`grp:${g}`, list.map((e) => row(`spot:${e.s.id}`, 2,
      `<i class="dot-i" style="background:${e.bad ? "var(--bad)" : e.border ? "var(--warn)" : view.kind === "twin" ? ROBOT_CSS[view.hueOf[e.s.robot] ?? 0] : "#c9d2e0"}"></i>`,
      e.s.id, e.bad ? '<span class="tag" style="color:var(--bad)">못 쏨</span>' : "")).join(""));
  }
  h += kids("spots", sh);
  h += row("robots", 0, icon("folder"), "로봇", "", { kids: true, eye: "robots" });
  h += kids("robots", Object.values(view.robots).map((r) => row(`robot:${r.id}`, 1, icon("robot"), `R${r.id + 1}`,
    `<span class="tag">${view.kind === "ai" ? "후보" : r.used ? "사용" : "빈 자리"}</span>`, { eye: `robot:${r.id}` })).join(""));
  $("tree").innerHTML = h;
}

$("tree").addEventListener("click", (ev) => {
  const eye = ev.target.closest("[data-eye]");
  const node = ev.target.closest(".node");
  if (!node) return;
  const key = node.dataset.node;
  if (eye) {
    const k = eye.dataset.eye;
    vis[k] = vis[k] === false;
    applyVisibility();
    renderTree();
    return;
  }
  if (ev.target.closest(".tw") && node.querySelector(".caret")) {
    openNodes.has(key) ? openNodes.delete(key) : openNodes.add(key);
    renderTree();
    return;
  }
  if (key.startsWith("grp:") || ["body", "spots", "robots"].includes(key)) openNodes.add(key);
  if (key.startsWith("spot:")) { openNodes.add("spots"); openNodes.add(`grp:${view.spots[key.slice(5)]?.group}`); }
  select(key);
});

function renderDocs() {
  const items = data.designs.map((d, i) => {
    const bad = d.spots.filter((s) => s.robot == null).length;
    return `<button class="doc${docIndex === i ? " on" : ""}" data-doc="${i}"><span class="ico">${icon("doc")}</span>
      <span class="nm">${designLabel(d.name)}</span><span class="badge${bad ? " bad" : ""}">${bad ? `${bad} 못 쏨` : "0"}</span></button>`;
  });
  const n = data.designs.length;
  items.push(`<button class="doc${docIndex === n ? " on" : ""}" data-doc="${n}"><span class="ico">${icon("ai")}</span>
    <span class="nm">AI 설계 검토</span><span class="badge ai">AI</span></button>`);
  $("docList").innerHTML = items.join("");
}
$("docList").addEventListener("click", (ev) => {
  const b = ev.target.closest("[data-doc]");
  if (b) openDoc(+b.dataset.doc);
});

/* =====================================================================
   선택 · 강조 · 속성
   ===================================================================== */
const selGroup = new THREE.Group();
scene.add(selGroup);
let selBoxHelper = null;

function select(key) {
  sel = key;
  renderTree();
  updateSelectionMarkers();
  renderInspector();
  $("stSel").textContent = key ? `선택 ${selName(key)}` : "선택 없음";
  const node = document.querySelector(`.node[data-node="${CSS.escape(key ?? "")}"]`);
  node?.scrollIntoView({ block: "nearest" });
}
function selName(key) {
  if (!key) return "";
  const [kind, id] = key.split(":");
  if (kind === "spot") return `${id} (${GROUP_KO[view.spots[id]?.group] ?? ""} 타점)`;
  if (kind === "grp") return `${GROUP_KO[id]} 타점 그룹`;
  if (kind === "box") return BOX_KO[id] ?? id;
  if (kind === "robot") return `R${+id + 1}`;
  return { body: "차체", spots: "타점 전체", robots: "로봇 전체" }[kind] ?? key;
}

function updateSelectionMarkers() {
  selGroup.clear();
  selBoxHelper = null;
  if (!view || !sel) return;
  const [kind, id] = sel.split(":");
  const ring = (pos, size) => {
    const s = new THREE.Sprite(new THREE.SpriteMaterial({ map: RING, depthTest: false, transparent: true }));
    s.position.copy(pos);
    s.scale.setScalar(size);
    s.renderOrder = 12;
    selGroup.add(s);
  };
  if (kind === "spot" && view.spots[id]) ring(view.spots[id].pos, 0.09);
  if (kind === "grp") Object.values(view.spots).filter((e) => e.group === id).forEach((e) => ring(e.pos, 0.055));
  const edgeOf = (b) => {
    const e = new THREE.LineSegments(new THREE.EdgesGeometry(b.geo), selEdgeMat);
    e.position.copy(b.pos);
    e.renderOrder = 12;
    selGroup.add(e);
  };
  if (kind === "box" && view.boxes[id]) edgeOf(view.boxes[id]);
  if (kind === "body") Object.values(view.boxes).forEach(edgeOf);
  if (kind === "robot" && view.robots[id]) {
    selBoxHelper = new THREE.BoxHelper(view.robots[id].objs[0], ACCENT);
    selBoxHelper.material.depthTest = false;
    selBoxHelper.material.transparent = true;
    selGroup.add(selBoxHelper);
  }
}

const kvRow = (k, v, cls = "v") => `<tr><td>${k}</td><td class="${cls}">${v}</td></tr>`;
const meter = (label, value, max, cls, text, mark) => `<div class="meter">
  <div class="row"><span>${label}</span><span>${text}</span></div>
  <div class="bar"><i class="${cls}" style="width:${clamp((value / max) * 100, 0, 100)}%"></i>${mark != null ? `<span class="mark" style="left:${clamp(mark * 100, 0, 100)}%"></span>` : ""}</div></div>`;

function deriveParams(d) {
  const box = (n) => d.boxes.find((b) => b.name === n);
  const wall = box("member_wall"), pil = box("b_pillar"), sill = box("sill");
  const m0 = d.spots.find((s) => s.id === "M00");
  const member_x = m0.pos[0], y0 = wall.lo[1];
  return {
    flange_width: wall.lo[0] - member_x, member_wall_height: wall.hi[2] - wall.lo[2], member_first_spot: m0.pos[1] - y0,
    member_y0: y0, member_x, pillar_x: (pil.lo[0] + pil.hi[0]) / 2, pillar_w: pil.hi[0] - pil.lo[0],
    sill_top: sill.hi[2], floor_z: sill.lo[2],
  };
}

function robotPath(id) { return design?.paths.find((p) => p.robot === id); }

function renderInspector() {
  const el = $("inspector");
  if (!view) { el.innerHTML = ""; return; }
  const [kind, id] = (sel ?? "").split(":");
  if (kind === "spot" && view.spots[id]) { el.innerHTML = inspectSpot(view.spots[id]); return; }
  if (kind === "grp") { el.innerHTML = inspectGroup(id); return; }
  if (kind === "box" && view.boxes[id]) { el.innerHTML = inspectBox(view.boxes[id].b); return; }
  if (kind === "robot" && view.robots[id]) { el.innerHTML = inspectRobot(view.robots[id]); return; }
  el.innerHTML = inspectDesign();
}

function inspectSpot(e) {
  const s = e.s;
  let h = `<div class="insp-title"><b class="mono">${s.id}</b><span class="kind">${GROUP_KO[s.group]} 타점</span></div><table class="kv">`;
  h += kvRow("좌표 (mm)", vecMm(s.pos));
  h += kvRow("법선", s.normal.map((x) => x.toFixed(2)).join(", "));
  if (view.kind === "twin") {
    h += kvRow("판정", e.bad ? "못 쏨" : "쏠 수 있음", e.bad ? "v bad" : "v");
    h += kvRow("담당 로봇", s.robot != null ? `R${s.robot + 1}` : "없음");
    h += kvRow("기울임", s.tilt != null ? `${s.tilt}°` : "—");
    if (s.approach) h += kvRow("건 접근 방향", s.approach.map((x) => x.toFixed(3)).join(", "));
    if (e.weld) h += kvRow("용접 시각", `${e.weld.t0.toFixed(2)} – ${e.weld.t1.toFixed(2)} s`);
    if (e.bad) h += kvRow("못 쏘는 이유", s.reasons.map((r) => REASON_KO[r] ?? r).join(", ") || "—", "v bad");
    h += "</table>";
    return h;
  }
  h += kvRow("AI 판정", e.bad ? "못 쏨" : e.border ? "경계 근처" : "쏠 수 있음", e.bad ? "v bad" : "v");
  h += kvRow("가장 높은 확률", `${e.best.toFixed(3)} (기준 ${ai.meta.threshold})`);
  h += "</table><div class=\"sub-h\">후보 로봇별 쏠 확률</div>";
  e.probs.forEach((p, r) => {
    h += meter(`R${r + 1}`, p, 1, p >= ai.meta.threshold ? "acc" : "warn", p.toFixed(3), ai.meta.threshold);
  });
  return h;
}

function inspectGroup(g) {
  const list = Object.values(view.spots).filter((e) => e.group === g);
  const bad = list.filter((e) => e.bad);
  let h = `<div class="insp-title"><b>${GROUP_KO[g]} ${GROUP_PREFIX[g]}</b><span class="kind">타점 그룹</span></div><table class="kv">`;
  h += kvRow("타점 수", list.length);
  h += kvRow(view.kind === "ai" ? "AI 예측 못 쏨" : "못 쏘는 타점", bad.length ? bad.map((e) => e.s.id).join(", ") : "0", bad.length ? "v bad" : "v");
  if (view.kind === "twin") {
    const by = {};
    list.forEach((e) => { if (e.s.robot != null) by[e.s.robot] = (by[e.s.robot] ?? 0) + 1; });
    h += kvRow("담당", Object.entries(by).map(([r, n]) => `R${+r + 1} ${n}`).join(" · ") || "—");
  } else {
    h += kvRow("경계 근처", list.filter((e) => e.border).map((e) => e.s.id).join(", ") || "0");
  }
  return h + "</table>";
}

function inspectBox(b) {
  const size = b.hi.map((h, i) => h - b.lo[i]);
  return `<div class="insp-title"><b>${BOX_KO[b.name] ?? b.name}</b><span class="kind">차체 판넬 (상자)</span></div><table class="kv">
    ${kvRow("최소 (mm)", vecMm(b.lo))}${kvRow("최대 (mm)", vecMm(b.hi))}${kvRow("크기 (mm)", vecMm(size))}</table>`;
}

function inspectRobot(r) {
  let h = `<div class="insp-title"><b>R${r.id + 1}</b><span class="kind">KUKA KR210 L150</span></div><table class="kv">`;
  h += kvRow("베이스 (mm)", vecMm(r.base, 0));
  h += kvRow("회전 (z)", "90°");
  h += kvRow("상태", view.kind === "ai" ? "후보 위치" : r.used ? "사용" : "빈 자리");
  const p = view.kind === "twin" ? robotPath(r.id) : null;
  if (p) h += kvRow("담당 타점", `${p.frames.filter((f) => f.weld_end).length}개`);
  h += kvRow("도달 범위 (플랜지, 근사)", `${mm(REACH, 0)} mm`);
  h += "</table>";
  if (p) h += `<div class="sub-h">사이클타임</div>` + meter(`예산 ${design.budget_s.toFixed(0)} s`, p.cycle_s, design.budget_s,
    `r${r.hue ?? 0}`, `${p.cycle_s.toFixed(2)} s · 여유 ${(design.budget_s - p.cycle_s).toFixed(2)} s`);
  if (r.rig) {
    h += `<div class="sub-h">관절각 (°)</div><table class="kv" id="jointTable">`;
    r.rig.q.forEach((q, i) => { h += `<tr><td>a${i + 1}</td><td class="v" data-j="${i}">${THREE.MathUtils.radToDeg(q).toFixed(1)}</td></tr>`; });
    h += "</table>";
  }
  return h;
}

function inspectDesign() {
  const isAI = view.kind === "ai";
  const v = isAI ? ai.v : deriveParams(design);
  let h = `<div class="insp-title"><b>${isAI ? "AI 설계 검토" : designLabel(design.name)}</b><span class="kind">설계안</span></div>`;
  if (!isAI) h += `<div class="note" style="margin-top:0">${esc(design.notes)}</div>`;
  h += `<div class="sub-h">설계 변수</div><table class="kv">`;
  PARAMS.forEach((k) => { h += kvRow(PARAM_KO[k], `${mm(v[k])} mm`); });
  return h + "</table>";
}

function renderResults() {
  const el = $("results");
  if (!view) return;
  if (view.kind === "ai") {
    $("resultsTitle").textContent = "AI 판정";
    const bad = ai.res.bad.filter(Boolean).length;
    const ids = ai.res.spots.filter((_, i) => ai.res.bad[i]).map((s) => s.id);
    const border = Object.values(view.spots).filter((e) => e.border).map((e) => e.s.id);
    el.innerHTML = `<div class="stats two">
        <div class="stat ${bad ? "bad" : "good"}"><div class="v" id="aiBad">${bad}개</div><div class="k">AI 예측 못 쏘는 타점</div></div>
        <div class="stat"><div class="v" id="aiMs">${ai.ms < 10 ? ai.ms.toFixed(1) : Math.round(ai.ms)} ms</div><div class="k">판정 시간</div></div>
      </div><table class="kv">
      ${kvRow("못 쏨", ids.join(", ") || "없음", ids.length ? "v bad" : "v")}
      ${kvRow("경계 근처", border.join(", ") || "없음")}
      ${kvRow("기준 확률", ai.meta.threshold)}</table>
      <div class="note">${AI_NOTE}</div>`;
    return;
  }
  $("resultsTitle").textContent = "해석 결과";
  const d = design;
  const bad = d.spots.filter((s) => s.robot == null).length;
  let h = `<div class="stats">
    <div class="stat"><div class="v">${d.spots.length}</div><div class="k">타점 수</div></div>
    <div class="stat ${bad ? "bad" : "good"}"><div class="v">${bad}</div><div class="k">못 쏘는 타점</div></div>
    <div class="stat"><div class="v">${d.paths.length}</div><div class="k">로봇 대수</div></div></div>
    <div class="sub-h">로봇별 사이클타임 · 예산 ${d.budget_s.toFixed(0)} s</div>`;
  d.paths.forEach((p, i) => {
    h += meter(`R${p.robot + 1} · ${p.frames.filter((f) => f.weld_end).length}점`, p.cycle_s, d.budget_s, `r${i % 2}`,
      `${p.cycle_s.toFixed(2)} / ${d.budget_s.toFixed(0)} s`);
  });
  el.innerHTML = h;
}

/* =====================================================================
   포인터 — 선택 · 떠 있는 설명 · 커서 좌표
   ===================================================================== */
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const floorPlane = new THREE.Plane(new THREE.Vector3(0, 0, 1), 0);
function setPointer(ev) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.set(((ev.clientX - rect.left) / rect.width) * 2 - 1, -((ev.clientY - rect.top) / rect.height) * 2 + 1);
  raycaster.setFromCamera(pointer, camera);
}
function pickSpot() {
  if (!view) return null;
  const hit = raycaster.intersectObjects(view.hits.filter((h) => h.visible), false)[0];
  return hit ? view.spots[hit.object.userData.id] : null;
}
function pickRobot() {
  if (!view || !vis.robots) return null;
  const roots = Object.values(view.robots).filter((r) => r.objs[0].visible);
  const hit = raycaster.intersectObjects(roots.map((r) => r.objs[0]), true)[0];
  if (!hit) return null;
  return roots.find((r) => { let o = hit.object; while (o) { if (o === r.objs[0]) return true; o = o.parent; } return false; });
}
function showTip(e, ev) {
  const tip = $("tip");
  if (!e) { tip.hidden = true; return; }
  const s = e.s;
  const rows = [["부위", GROUP_KO[s.group]]];
  if (view.kind === "twin") {
    rows.push(["담당", s.robot != null ? `R${s.robot + 1}` : "없음"]);
    if (s.tilt) rows.push(["기울임", `${s.tilt}°`]);
    if (e.bad) rows.push(["이유", `<span class="reason">${s.reasons.map((r) => REASON_KO[r] ?? r).join(", ")}</span>`]);
  } else {
    rows.push(["AI", e.bad ? '<span class="reason">못 쏨</span>' : e.border ? "경계 근처" : "쏠 수 있음"]);
    rows.push(["최대 확률", e.best.toFixed(2)]);
  }
  tip.innerHTML = `<b>${s.id}</b>` + rows.map(([k, v]) => `<div class="row"><span>${k}</span><span>${v}</span></div>`).join("");
  tip.hidden = false;
  const rect = stage.getBoundingClientRect();
  const x = Math.min(ev.clientX - rect.left + 14, rect.width - tip.offsetWidth - 6);
  const y = Math.min(ev.clientY - rect.top + 14, rect.height - tip.offsetHeight - 6);
  tip.style.left = `${Math.max(6, x)}px`;
  tip.style.top = `${Math.max(6, y)}px`;
}

let downAt = null;
renderer.domElement.addEventListener("pointerdown", (ev) => { downAt = [ev.clientX, ev.clientY]; });
renderer.domElement.addEventListener("pointermove", (ev) => {
  setPointer(ev);
  const e = pickSpot();
  renderer.domElement.style.cursor = e && tool === "select" ? "pointer" : "";
  showTip(e, ev);
  const p = new THREE.Vector3();
  if (e) $("stCursor").textContent = `X ${mm(e.pos.x)}  Y ${mm(e.pos.y)}  Z ${mm(e.pos.z)}`;
  else if (raycaster.ray.intersectPlane(floorPlane, p)) $("stCursor").textContent = `X ${mm(p.x)}  Y ${mm(p.y)}  Z 0.0`;
});
renderer.domElement.addEventListener("pointerleave", () => { $("tip").hidden = true; $("stCursor").textContent = "X — Y —"; });
renderer.domElement.addEventListener("pointerup", (ev) => {
  if (!downAt || Math.hypot(ev.clientX - downAt[0], ev.clientY - downAt[1]) > 4 || tool !== "select" || ev.button !== 0) return;
  setPointer(ev);
  const e = pickSpot();
  if (e) { openNodes.add("spots"); openNodes.add(`grp:${e.group}`); select(`spot:${e.s.id}`); return; }
  const r = pickRobot();
  if (r) { openNodes.add("robots"); select(`robot:${r.id}`); return; }
  select(null);
});

/* =====================================================================
   시점 · 도구
   ===================================================================== */
const VIEW_DIRS = {
  front: { dir: [-1, 0, 0.0001], name: "정면" },
  side: { dir: [0, 1, 0.0001], name: "측면" },
  top: { dir: [0, -0.0001, 1], name: "위" },
  iso: { dir: [-4.1, 4.85, 3.5], name: "등각" },
};
let viewName = "iso";
const FIT = 0.74;   // 경계 구는 상자보다 넉넉해서 줄여 맞춘다
function visibleBounds(focusBody) {
  const box = new THREE.Box3();
  if (!view) return box.setFromCenterAndSize(new THREE.Vector3(1.2, 0, 0.6), new THREE.Vector3(3, 3, 2));
  if (focusBody) {
    Object.values(view.boxes).forEach((b) => b.objs[0].visible && box.expandByObject(b.objs[0]));
    Object.values(view.spots).forEach((e) => e.hit.visible && box.expandByPoint(e.pos));
  } else {
    view.group.updateMatrixWorld(true);
    view.group.traverse((o) => {
      if (!(o.isMesh || o.isSprite) || o === view.reach || !o.visible) return;
      let p = o, shown = true;
      while (p && p !== view.group) { if (!p.visible) { shown = false; break; } p = p.parent; }
      if (shown && (o.isMesh ? o.material.visible !== false : true)) box.expandByObject(o);
    });
  }
  if (box.isEmpty()) box.setFromCenterAndSize(new THREE.Vector3(1.2, 0, 0.6), new THREE.Vector3(3, 3, 2));
  return box;
}
function setView(name, focusBody = view?.kind === "ai") {
  viewName = name;
  const box = visibleBounds(focusBody);
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  const dir = new THREE.Vector3(...VIEW_DIRS[name].dir).normalize();
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const fitH = radius / Math.sin(fov / 2);
  const fitW = radius / Math.sin(Math.atan(Math.tan(fov / 2) * camera.aspect));
  const dist = Math.max(fitH, fitW) * FIT * (focusBody ? 1.25 : 1);
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(dir, dist);
  controls.update();
  $("vpLabel").textContent = `${VIEW_DIRS[name].name} · 투시`;
}
function fit() {
  const dir = camera.position.clone().sub(controls.target).normalize();
  const box = visibleBounds(view?.kind === "ai");
  const center = box.getCenter(new THREE.Vector3());
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  const fov = THREE.MathUtils.degToRad(camera.fov);
  const dist = Math.max(radius / Math.sin(fov / 2), radius / Math.sin(Math.atan(Math.tan(fov / 2) * camera.aspect))) * FIT;
  controls.target.copy(center);
  camera.position.copy(center).addScaledVector(dir, dist);
  controls.update();
}
controls.addEventListener("start", () => { $("vpLabel").textContent = "사용자 · 투시"; });

function setTool(name) {
  tool = name;
  controls.mouseButtons.LEFT = name === "pan" ? THREE.MOUSE.PAN : THREE.MOUSE.ROTATE;
  document.querySelectorAll("[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === name));
}

/* =====================================================================
   설계안 열기 · AI 모드
   ===================================================================== */
function openDoc(i) {
  const n = data.designs.length;
  if (i === n) { enterAI(); return; }
  const wasAI = mode === "ai";
  mode = "twin";
  docIndex = i;
  design = data.designs[i];
  $("aiPanel").hidden = true;
  $("vpMode").textContent = "";
  $("tlOff").hidden = true;
  $("scrub").hidden = false;
  buildTwin(design);
  t = 0;
  applyTime(0);
  setPlaying(true);
  keepSelection();
  applyVisibility();
  renderTree();
  renderDocs();
  renderInspector();
  renderResults();
  renderTimelineLegend();
  if (wasAI) setView("iso");
  $("docTitle").innerHTML = `설계안 <b>${designLabel(design.name)}</b> · KR210 L150 × ${design.paths.length}`;
  const bad = design.spots.filter((s) => s.robot == null);
  log("info", `설계안 열기: ${designLabel(design.name)} — 타점 ${design.spots.length}, 로봇 ${design.paths.length}대, 못 쏘는 타점 ${bad.length}`);
  bad.forEach((s) => log("warn", `${s.id} ${s.reasons.map((r) => REASON_KO[r] ?? r).join(", ")} — 네 후보 로봇 모두 못 쏨`));
}
function keepSelection() {
  if (!sel) return;
  const [kind, id] = sel.split(":");
  const ok = (kind === "spot" && view.spots[id]) || (kind === "box" && view.boxes[id]) || (kind === "robot" && view.robots[id])
    || kind === "grp" || ["body", "spots", "robots"].includes(kind);
  if (!ok) sel = null;
  updateSelectionMarkers();
  $("stSel").textContent = sel ? `선택 ${selName(sel)}` : "선택 없음";
}

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
    log(par.pass ? "ok" : "err", `AI 파리티 ${par.pass ? "PASS" : "FAIL"}: 설계 ${par.designs}개, 확률 최대 차이 ${par.maxAbsProbDiff.toExponential(2)}, 타점 판정 불일치 ${par.spotMismatch}/${par.spotsTotal}`);
    $("stAi").innerHTML = `<span class="${par.pass ? "ok" : "bad"}">AI 파리티 ${par.pass ? "PASS" : "FAIL"}</span>`;
  });
  return ai.ready;
}

async function enterAI() {
  await loadAI();
  const wasAI = mode === "ai";
  mode = "ai";
  docIndex = data.designs.length;
  setPlaying(false);
  ai.v ??= { ...ai.meta.base };
  $("aiPanel").hidden = false;
  $("vpMode").textContent = "AI 설계 검토";
  $("tlOff").hidden = false;
  $("scrub").hidden = true;
  $("timeLabel").textContent = "— s";
  $("tlLegend").innerHTML = "";
  renderAIFields();
  runAI(true);
  renderDocs();
  if (!wasAI) {
    setView("iso", true);
    log("info", "AI 설계 검토 모드 — 설계 변수를 바꾸면 브라우저 안에서 LightGBM 이 바로 판정");
  }
  $("docTitle").innerHTML = `<b>AI 설계 검토</b> · 후보 로봇 ${ai.meta.bases.length}곳`;
}

function renderAIFields() {
  const { meta } = ai;
  const rows = meta.params.map((k) => {
    const [lo, hi] = meta.space[k];
    return `<div class="fld" data-k="${k}"><label title="${PARAM_KO[k]}">${PARAM_KO[k]}</label>
      <input type="range" data-k="${k}" min="${lo}" max="${hi}" step="0.001" value="${ai.v[k]}" aria-label="${PARAM_KO[k]}">
      <span class="num"><input type="number" data-k="${k}" min="${Math.round(lo * 1000)}" max="${Math.round(hi * 1000)}" step="1" value="${Math.round(ai.v[k] * 1000)}"><span class="u">mm</span></span></div>`;
  }).join("");
  $("aiFields").innerHTML = rows + `<div class="btn-row"><button class="btn" data-act="ai:reset">A안으로 되돌리기</button></div>`;
  syncFieldState();
}
function syncFieldState() {
  document.querySelectorAll("#aiFields .fld").forEach((f) => {
    const k = f.dataset.k;
    f.classList.toggle("changed", Math.abs(ai.v[k] - ai.meta.base[k]) > 5e-4);
  });
}
$("aiFields").addEventListener("input", (ev) => {
  const el = ev.target;
  const k = el.dataset.k;
  if (!k) return;
  const [lo, hi] = ai.meta.space[k];
  if (el.type === "range") {
    ai.v[k] = parseFloat(el.value);
    document.querySelector(`#aiFields input[type=number][data-k=${k}]`).value = Math.round(ai.v[k] * 1000);
  } else {
    const val = parseFloat(el.value);
    if (!Number.isFinite(val)) return;
    ai.v[k] = clamp(val / 1000, lo, hi);
    document.querySelector(`#aiFields input[type=range][data-k=${k}]`).value = ai.v[k];
  }
  syncFieldState();
  runAI();
});
$("aiFields").addEventListener("change", (ev) => {
  const el = ev.target;
  const k = el.dataset.k;
  if (!k) return;
  if (el.type === "number") el.value = Math.round(ai.v[k] * 1000);
  log("info", `AI 판정: ${PARAM_KO[k]} ${mm(ai.v[k], 0)} mm → 못 쏘는 타점 ${ai.res.bad.filter(Boolean).length}개 (${ai.ms.toFixed(1)} ms)`);
});

function runAI(full = false) {
  const t0 = performance.now();
  const res = AI.judge(ai.model, ai.meta, ai.v);
  ai.ms = performance.now() - t0;
  ai.res = res;
  buildAIScene(res);
  applyVisibility();
  keepSelection();
  renderTree();
  renderResults();
  if (full || !sel || sel.startsWith("spot") || sel.startsWith("grp") || sel === "body") renderInspector();
  window.__aiLast = { bad: res.bad.filter(Boolean).length, ms: ai.ms, v: { ...ai.v } };
}

function buildAIScene(res) {
  disposeView();
  const group = new THREE.Group();
  if (!ai.robots || ai.robotsKind !== meshKind) {
    ai.robots = new THREE.Group();
    ai.robotsKind = meshKind;
    ai.rigs = ai.meta.bases.map((base) => {
      const rig = buildRobot(base, Math.PI / 2, data.gun_length, false);
      setPose(rig, HOME);
      ai.robots.add(rig.root);
      return rig;
    });
    ai.reach = new THREE.Group();
    ai.meta.bases.forEach((b) => ai.reach.add(reachRing(b)));
    ai.robots.add(ai.reach);
  }
  group.add(ai.robots);
  const boxes = addBoxes(group, res.boxes);
  const okMat = new THREE.MeshStandardMaterial({ color: 0xc9d2e0, emissive: 0xc9d2e0, emissiveIntensity: 0.15, roughness: 0.4 });
  const borderMat = new THREE.MeshStandardMaterial({ color: 0xffa53d, emissive: 0xffa53d, emissiveIntensity: 0.6, roughness: 0.4 });
  const spots = {}, hits = [];
  const nR = ai.meta.bases.length;
  res.spots.forEach((s, i) => {
    const pos = new THREE.Vector3(...s.pos);
    const probs = Array.from(res.pairProb.subarray(i * nR, i * nR + nR));
    const border = !res.bad[i] && res.best[i] < ai.meta.threshold + BORDER;
    const e = { s, pos, group: s.group, bad: res.bad[i], border, best: res.best[i], probs, objs: [] };
    if (e.bad) e.objs = addBadMarker(group, pos);
    else {
      const m = new THREE.Mesh(spotGeo, border ? borderMat : okMat);
      m.position.copy(pos);
      group.add(m);
      e.objs = [m];
    }
    e.hit = addHit(group, s.id, pos);
    hits.push(e.hit);
    spots[s.id] = e;
  });
  const robots = {};
  ai.rigs.forEach((rig, i) => { robots[i] = { id: i, base: ai.meta.bases[i], used: true, rig, objs: [rig.root] }; });
  scene.add(group);
  view = { kind: "ai", group, keep: ai.robots, boxes, robots, spots, hits, welds: [], sparks: [], reach: ai.reach, duration: 0 };
}

/* =====================================================================
   타임라인
   ===================================================================== */
const scrub = $("scrub");
function renderTimelineLegend() {
  $("tlLegend").innerHTML = design.paths.map((p, i) => `<i class="sw r${i % 2}"></i>R${p.robot + 1}`).join(" ")
    + ` <i class="sw" style="background:var(--muted);width:2px"></i>예산 ${design.budget_s.toFixed(0)} s`;
}
const TL = { padL: 64, padR: 14 };
function drawScrub() {
  if (!view || view.kind !== "twin" || scrub.hidden) return;
  const dpr = Math.min(window.devicePixelRatio, 2);
  const W = scrub.clientWidth, H = scrub.clientHeight;
  if (!W || !H) return;
  if (scrub.width !== Math.round(W * dpr) || scrub.height !== Math.round(H * dpr)) { scrub.width = Math.round(W * dpr); scrub.height = Math.round(H * dpr); }
  const g = scrub.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const T = view.duration;
  const x = (tt) => TL.padL + (tt / T) * (W - TL.padL - TL.padR);
  // 눈금자
  g.fillStyle = "#1f2023";
  g.fillRect(0, 0, W, 18);
  g.font = "10px 'JetBrains Mono', Consolas, Pretendard, monospace";
  g.textAlign = "center";
  for (let s = 0; s <= T; s += 1) {
    const major = s % 5 === 0;
    g.fillStyle = major ? "#8b8f97" : "#4a4d54";
    g.fillRect(Math.round(x(s)), major ? 10 : 14, 1, major ? 8 : 4);
    if (major) g.fillText(`${s}`, x(s), 9);
  }
  const rows = design.paths.length, top = 24, rowH = Math.min(26, (H - top - 6) / rows);
  design.paths.forEach((p, i) => {
    const y = top + i * rowH;
    g.fillStyle = "#8b8f97";
    g.textAlign = "left";
    g.fillText(`R${p.robot + 1}`, 10, y + rowH / 2 + 3);
    g.fillStyle = "#2c2d31";
    g.fillRect(TL.padL, y + rowH * 0.3, x(p.cycle_s) - TL.padL, rowH * 0.4);
    const hue = i % 2 === 0 ? [77, 163, 255] : [169, 139, 255];
    for (const w of view.welds) {
      if (w.robot !== p.robot) continue;
      const done = t >= w.t1;
      g.fillStyle = `rgba(${hue.join(",")},${done ? 1 : 0.4})`;
      g.fillRect(x(w.t0), y + rowH * 0.18, Math.max(2, x(w.t1) - x(w.t0)), rowH * 0.64);
    }
  });
  // 예산선
  const bx = Math.round(x(design.budget_s)) + 0.5;
  g.strokeStyle = "rgba(216,218,222,.45)";
  g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(bx, 18); g.lineTo(bx, H); g.stroke();
  g.setLineDash([]);
  // 재생 헤드
  const px = Math.round(x(t)) + 0.5;
  g.strokeStyle = "#5ee1d4";
  g.lineWidth = 1.5;
  g.beginPath(); g.moveTo(px, 4); g.lineTo(px, H); g.stroke();
  g.lineWidth = 1;
  g.fillStyle = "#5ee1d4";
  g.beginPath(); g.moveTo(px - 5, 0); g.lineTo(px + 5, 0); g.lineTo(px, 7); g.fill();
}
let scrubbing = false;
function scrubTo(ev) {
  const rect = scrub.getBoundingClientRect();
  const u = (ev.clientX - rect.left - TL.padL) / (rect.width - TL.padL - TL.padR);
  t = clamp(u, 0, 1) * view.duration;
  applyTime(t);
}
scrub.addEventListener("pointerdown", (ev) => { if (view?.kind !== "twin") return; scrubbing = true; scrub.setPointerCapture(ev.pointerId); scrubTo(ev); });
scrub.addEventListener("pointermove", (ev) => { if (scrubbing) scrubTo(ev); });
scrub.addEventListener("pointerup", () => { scrubbing = false; });

/* =====================================================================
   혼류 라인
   ===================================================================== */
const line = { s: 0, playing: false, rate: 2.5, alts: [], lastCur: -1 };
function setLinePlaying(v) {
  line.playing = v;
  $("linePlay").innerHTML = icon(v ? "pause" : "play");
}
function buildLine() {
  const L = data.line;
  const stats = L.alternatives.map((a) => a.cars.reduce((x, c) => x + c.stop_s, 0));
  const best = stats.indexOf(Math.min(...stats));
  $("alts").innerHTML = "";
  line.alts = L.alternatives.map((a, i) => {
    const el = document.createElement("div");
    el.className = "alt" + (i === best ? " best" : "");
    const newShare = Math.round((a.cars.filter((c) => c.new).length / a.cars.length) * 100);
    el.innerHTML = `<div class="alt-top"><span class="alt-name">${esc(a.name)}</span>
      <span class="alt-meta">스테이션 ${a.stations} · 택트 ${L.takt_s} s · 여유창 ${Math.round(L.drift * 100)}% · 신차 ${newShare}%</span>
      <span class="alt-stats"><span data-k="stops">정지 <b>0</b>회</span><span data-k="total">누적 <b>0.0</b> s</span><span>투입 <b data-k="cars">0</b>/60</span></span></div>
      <canvas></canvas>`;
    $("alts").appendChild(el);
    return { a, el, canvas: el.querySelector("canvas"), stops: el.querySelector('[data-k="stops"]'), total: el.querySelector('[data-k="total"]'), cars: el.querySelector('[data-k="cars"]') };
  });
  $("lineInfo").textContent = `60대 · ${L.takt_s} s 택트`;
}
function roundRect(g, x, y, w, h, r) { g.beginPath(); g.roundRect ? g.roundRect(x, y, w, h, r) : g.rect(x, y, w, h); }
function drawLine() {
  const L = data.line;
  const takt = L.takt_s, win = takt * (1 + L.drift);
  const s = line.s, cur = Math.floor(s);
  for (const A of line.alts) {
    const { a, canvas } = A;
    const dpr = Math.min(window.devicePixelRatio, 2);
    const W = canvas.clientWidth, H = canvas.clientHeight;
    if (!W || !H) continue;
    if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) { canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr); }
    const g = canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const n = a.stations, gap = 3;
    const bw = (W - gap * (n - 1)) / n;
    const top = 14, bh = Math.max(20, H - top - 26), taktY = top + bh * (1 - takt / win);
    let stops = 0, total = 0;
    a.cars.forEach((c, k) => { if (c.stop_s > 0 && k + c.stop_station <= cur) { stops++; total += c.stop_s; } });
    g.font = "10px 'JetBrains Mono', Consolas, Pretendard, monospace";
    g.textAlign = "center";
    for (let j = 0; j < n; j++) {
      const x = j * (bw + gap);
      const k = cur - j;
      const car = k >= 0 && k < a.cars.length ? a.cars[k] : null;
      const lag = car ? car.lag_s[j] : 0;
      const stopHere = car && car.stop_s > 0 && car.stop_station === j;
      g.fillStyle = "#2a2b2f";
      g.fillRect(x, top, bw, bh);
      g.fillStyle = "#303136";
      g.fillRect(x, taktY, bw, top + bh - taktY);
      if (lag > 0) {
        const hOk = (Math.min(lag, takt) / win) * bh;
        g.fillStyle = car.new ? "rgba(94,225,212,0.55)" : "rgba(216,218,222,0.32)";
        g.fillRect(x, top + bh - hOk, bw, hOk);
        if (lag > takt) {
          const hOver = ((lag - takt) / win) * bh;
          g.fillStyle = "rgba(255,165,61,0.9)";
          g.fillRect(x, taktY - hOver, bw, hOver);
        }
      }
      if (stopHere) {
        g.fillStyle = `rgba(255,90,78,${0.35 + 0.5 * (1 - (s - cur))})`;
        g.fillRect(x, top, bw, bh);
        g.fillStyle = "#ff5a4e";
        g.fillText(`+${car.stop_s.toFixed(1)}s`, clamp(x + bw / 2, 22, W - 22), 10);
      }
      g.fillStyle = "#62666e";
      if (bw > 16 || j % 2 === 0) g.fillText(String(j + 1), x + bw / 2, H - 2);
    }
    const carY = top + bh + 4, carH = 8;
    for (let k = Math.max(0, cur - n); k <= Math.min(a.cars.length - 1, cur); k++) {
      const p = s - k;
      if (p < 0 || p >= n) continue;
      const j = Math.floor(p), u = clamp((p - j - 0.8) / 0.2, 0, 1);
      const slide = u * u * (3 - 2 * u);
      const w = Math.max(5, bw * 0.62);
      g.globalAlpha = j === n - 1 ? 1 - slide : 1;
      roundRect(g, (j + slide) * (bw + gap) + (bw - w) / 2, carY, w, carH, 2);
      g.fillStyle = a.cars[k].new ? "#5ee1d4" : "rgba(216,218,222,0.6)";
      g.fill();
      g.globalAlpha = 1;
    }
    A.stops.querySelector("b").textContent = stops;
    A.total.querySelector("b").textContent = total.toFixed(1);
    A.stops.classList.toggle("hot", stops > 0);
    A.total.classList.toggle("hot", total > 0);
    A.cars.textContent = Math.min(a.cars.length, Math.max(0, cur + 1));
  }
  // 정지 순간을 로그에
  if (cur !== line.lastCur) {
    if (cur > line.lastCur) {
      for (let c = line.lastCur + 1; c <= cur; c++) {
        for (const A of line.alts) {
          A.a.cars.forEach((car, k) => {
            if (car.stop_s > 0 && k + car.stop_station === c) log("warn", `라인 ${A.a.name.split(" ")[0]}: ${k + 1}번째 차(${car.new ? "신차" : "기존"}) 스테이션 ${car.stop_station + 1} 정지 +${car.stop_s.toFixed(1)} s`);
          });
        }
      }
    }
    line.lastCur = cur;
  }
}
const lineEnd = () => 60 + Math.max(...data.line.alternatives.map((a) => a.stations));

/* =====================================================================
   라벨 겹쳐 그리기
   ===================================================================== */
const labelEls = new Map();
function updateLabels() {
  const box = $("labels");
  if (!vis.labels || !view) { if (labelEls.size) { box.innerHTML = ""; labelEls.clear(); } return; }
  const W = stage.clientWidth, H = stage.clientHeight;
  const v = new THREE.Vector3();
  const seen = new Set();
  for (const e of Object.values(view.spots)) {
    if (!e.hit.visible) continue;
    v.copy(e.pos).project(camera);
    if (v.z > 1 || v.x < -1.05 || v.x > 1.05 || v.y < -1.05 || v.y > 1.05) continue;
    let el = labelEls.get(e.s.id);
    if (!el) { el = document.createElement("span"); el.textContent = e.s.id; box.appendChild(el); labelEls.set(e.s.id, el); }
    el.className = e.bad ? "bad" : "";
    el.style.left = `${((v.x + 1) / 2) * W}px`;
    el.style.top = `${((1 - v.y) / 2) * H}px`;
    seen.add(e.s.id);
  }
  for (const [id, el] of labelEls) if (!seen.has(id)) { el.remove(); labelEls.delete(id); }
}

/* =====================================================================
   메뉴 · 도구 막대 · 단축키
   ===================================================================== */
function act(a) {
  const [cmd, arg] = a.split(":");
  switch (cmd) {
    case "view": setView(arg); break;
    case "fit": fit(); break;
    case "toggle":
      vis[arg] = !vis[arg];
      applyVisibility();
      renderTree();
      log("info", `${{ body: "차체", spots: "타점", robots: "로봇", reach: "도달 범위", labels: "타점 라벨" }[arg]} ${vis[arg] ? "보이기" : "숨기기"}`);
      break;
    case "play":
      if (mode !== "twin") break;
      if (t >= view.duration) t = 0;
      setPlaying(!playing);
      break;
    case "rewind": if (mode === "twin") { t = 0; applyTime(0); } break;
    case "speed": setSpeed(+arg); break;
    case "line":
      if (arg === "play") { if (line.s >= lineEnd()) { line.s = 0; line.lastCur = -1; } setLinePlaying(!line.playing); }
      if (arg === "reset") { line.s = 0; line.lastCur = -1; drawLine(); }
      break;
    case "ai":
      if (arg === "reset") {
        if (mode !== "ai") { enterAI(); break; }
        ai.v = { ...ai.meta.base };
        renderAIFields();
        runAI(true);
        log("info", "설계 변수를 A안으로 되돌림");
      } else enterAI();
      break;
    case "log": showTab("log"); break;
    case "mesh": setMeshKind(arg); break;
    case "dock": toggleDock(arg); break;
    case "drawer": toggleDrawer(arg); break;
    case "about": $("aboutDlg").showModal(); break;
    case "keys": $("keysDlg").showModal(); break;
    case "snapshot": snapshot(); break;
    case "reload": location.reload(); break;
  }
  syncChecks();
}

function syncChecks() {
  document.querySelectorAll(".menu-pop [data-check]").forEach((b) => {
    const [cmd, arg] = b.dataset.act.split(":");
    let on = false;
    if (cmd === "toggle") on = !!vis[arg];
    if (cmd === "dock") on = !$("work").classList.contains(`no-${arg}`);
    b.classList.toggle("checked", on);
  });
  document.querySelectorAll(".menu-pop [data-radio]").forEach((b) => {
    const [cmd, arg] = b.dataset.act.split(":");
    const on = (cmd === "mesh" && arg === meshKind) || (cmd === "speed" && +arg === speed);
    b.classList.toggle("checked", on);
    if (cmd === "mesh" && arg === "dae") b.disabled = !visualSets.dae;
  });
}

document.addEventListener("click", (ev) => {
  const menuBtn = ev.target.closest(".menu-btn");
  const menus = document.querySelectorAll(".menu");
  if (menuBtn) {
    const m = menuBtn.parentElement;
    const was = m.classList.contains("open");
    menus.forEach((x) => x.classList.remove("open"));
    if (!was) m.classList.add("open");
    return;
  }
  const a = ev.target.closest("[data-act]");
  if (!ev.target.closest(".menu-pop") || a) menus.forEach((x) => x.classList.remove("open"));
  if (a && !a.disabled) act(a.dataset.act);
  const tb = ev.target.closest("[data-tool]");
  if (tb) setTool(tb.dataset.tool);
  const sp = ev.target.closest("#speeds [data-speed]");
  if (sp) setSpeed(+sp.dataset.speed);
  if (ev.target.closest("[data-close]")) ev.target.closest("dialog").close();
});
document.querySelectorAll(".menu").forEach((m) => m.addEventListener("mouseenter", () => {
  if (document.querySelector(".menu.open") && !m.classList.contains("open")) {
    document.querySelectorAll(".menu").forEach((x) => x.classList.remove("open"));
    m.classList.add("open");
  }
}));
document.addEventListener("keydown", (ev) => {
  if (ev.target.closest("input, textarea") || ev.ctrlKey || ev.metaKey || ev.altKey) return;
  const k = ev.key;
  if (k === " ") { ev.preventDefault(); act("play"); }
  else if (k === "Home") act("rewind");
  else if (k === "1") act("view:front");
  else if (k === "2") act("view:side");
  else if (k === "3") act("view:top");
  else if (k === "4") act("view:iso");
  else if (k === "f" || k === "F") act("fit");
  else if (k === "Escape") { select(null); document.querySelectorAll(".menu").forEach((x) => x.classList.remove("open")); }
});

// 도구 막대 설명 풍선
const ttip = document.createElement("div");
ttip.className = "ttip";
ttip.hidden = true;
document.body.appendChild(ttip);
document.addEventListener("pointerover", (ev) => {
  const el = ev.target.closest("[data-tip]");
  if (!el || ev.pointerType === "touch") { ttip.hidden = true; return; }
  ttip.textContent = el.dataset.tip;
  ttip.hidden = false;
  const r = el.getBoundingClientRect();
  ttip.style.left = `${clamp(r.left + r.width / 2 - ttip.offsetWidth / 2, 4, innerWidth - ttip.offsetWidth - 4)}px`;
  ttip.style.top = `${r.bottom + 6}px`;
});

function snapshot() {
  renderer.render(scene, camera);
  const a = document.createElement("a");
  a.href = renderer.domElement.toDataURL("image/png");
  a.download = `biw-weld-twin-${mode === "ai" ? "ai" : design.name[0]}.png`;
  a.click();
  log("info", "뷰포트 이미지 저장");
}

function setMeshKind(kind) {
  if (kind === "dae" && !visualSets.dae) return;
  meshKind = kind;
  visuals = visualSets[kind];
  window.__meshes = kind;
  rebuildCurrent();
  log("info", `로봇 메시: ${kind === "dae" ? "DAE 외형" : "STL 충돌용"}`);
}
function rebuildCurrent() {
  if (mode === "ai") runAI(true);
  else {
    const keepT = t;
    buildTwin(design);
    t = keepT;
    applyTime(t);
    keepSelection();
    applyVisibility();
    renderInspector();
  }
}

/* =====================================================================
   도크 · 탭 · 분할 막대
   ===================================================================== */
function toggleDock(which) { $("work").classList.toggle(`no-${which}`); requestAnimationFrame(resize); }
function toggleDrawer(which) {
  const w = $("work");
  const cls = `open-${which}`;
  const on = !w.classList.contains(cls);
  w.classList.remove("open-left", "open-right");
  if (on) w.classList.add(cls);
}
$("scrim").addEventListener("click", () => $("work").classList.remove("open-left", "open-right"));
document.querySelectorAll(".panel-h").forEach((h) => h.addEventListener("click", () => h.parentElement.classList.toggle("collapsed")));

function showTab(name) {
  document.querySelectorAll("#bottomTabs [data-tab]").forEach((b) => b.classList.toggle("on", b.dataset.tab === name));
  document.querySelectorAll(".dock-bottom [data-body]").forEach((b) => { b.hidden = b.dataset.body !== name; });
  if ($("work").classList.contains("no-bottom")) toggleDock("bottom");
  requestAnimationFrame(() => { drawScrub(); if (data) drawLine(); });
}
$("bottomTabs").addEventListener("click", (ev) => { const b = ev.target.closest("[data-tab]"); if (b) showTab(b.dataset.tab); });

(() => {
  const sp = $("splitter");
  let startY = 0, startH = 0;
  sp.addEventListener("pointerdown", (ev) => {
    startY = ev.clientY;
    startH = $("dockBottom").getBoundingClientRect().height;
    sp.setPointerCapture(ev.pointerId);
    sp.onpointermove = (e) => {
      const max = $("work").clientHeight * 0.7;
      document.documentElement.style.setProperty("--bottom-h", `${clamp(startH + (startY - e.clientY), 90, max)}px`);
    };
  });
  sp.addEventListener("pointerup", () => { sp.onpointermove = null; });
})();

function resize() {
  const w = stage.clientWidth, h = stage.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  drawScrub();
  if (data) drawLine();
}
new ResizeObserver(resize).observe(stage);
new ResizeObserver(() => { drawScrub(); if (data) drawLine(); }).observe($("dockBottom"));

/* =====================================================================
   프레임 루프
   ===================================================================== */
const clock = new THREE.Clock();
const fps = { n: 0, acc: 0 };
let jointAcc = 0;
function frame() {
  const dt = Math.min(clock.getDelta(), 0.1);
  if (view && playing && mode === "twin") {
    t += dt * speed;
    if (t > view.duration) t = 0;
    applyTime(t);
  }
  if (data && line.playing) {
    line.s = Math.min(lineEnd(), line.s + dt * line.rate);
    if (line.s >= lineEnd()) setLinePlaying(false);
    if (!document.querySelector('[data-body="line"]').hidden) drawLine();
    else if (Math.floor(line.s) !== line.lastCur) drawLine();
  }
  if (selBoxHelper) selBoxHelper.update();
  jointAcc += dt;
  if (jointAcc > 0.15 && sel?.startsWith("robot:")) {
    jointAcc = 0;
    const r = view?.robots[sel.split(":")[1]];
    if (r?.rig) document.querySelectorAll("#jointTable [data-j]").forEach((td) => { td.textContent = THREE.MathUtils.radToDeg(r.rig.q[+td.dataset.j]).toFixed(1); });
  }
  controls.update();
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setViewport(0, 0, w, h);
  renderer.render(scene, camera);
  // 좌표축 기즈모
  const s = gizmo.size;
  gizmo.cam.position.copy(camera.position).sub(controls.target).normalize().multiplyScalar(4);
  gizmo.cam.up.copy(camera.up);
  gizmo.cam.lookAt(0, 0, 0);
  renderer.autoClear = false;
  renderer.clearDepth();
  renderer.setScissorTest(true);
  renderer.setScissor(8, 8, s, s);
  renderer.setViewport(8, 8, s, s);
  renderer.render(gizmo.scene, gizmo.cam);
  renderer.setScissorTest(false);
  renderer.setViewport(0, 0, w, h);
  renderer.autoClear = true;
  updateLabels();
  fps.n++; fps.acc += dt;
  if (fps.acc >= 0.5) { $("stFps").textContent = `${Math.round(fps.n / fps.acc)} fps`; fps.n = 0; fps.acc = 0; }
  requestAnimationFrame(frame);
}

/* =====================================================================
   시작
   ===================================================================== */
async function boot() {
  document.querySelectorAll("[data-icon]").forEach((b) => { b.innerHTML = icon(b.dataset.icon); b.setAttribute("aria-label", b.dataset.tip ?? b.dataset.icon); });
  log("info", "BIW Weld Twin 시작");
  const loader = new STLLoader();
  const [json, ...meshes] = await Promise.all([
    fetch("data/twin_data.json").then((r) => r.json()),
    ...MESHES.map((m) => loader.loadAsync(`meshes/${m}.stl`)),
  ]);
  data = json;
  visualSets.stl = {};
  MESHES.forEach((m, i) => { meshes[i].computeVertexNormals(); visualSets.stl[m] = stlVisual(meshes[i], m); });
  visuals = visualSets.stl;
  log("info", `데이터: 설계안 ${data.designs.length}개, 로봇 ${data.robot_model}, 건 길이 ${mm(data.gun_length, 0)} mm`);
  loadAI().catch((err) => { console.error("[AI] 모델을 못 읽음", err); log("err", "AI 모델을 못 읽음"); });

  resize();
  // 모든 설계안 FK 검증 — 콘솔·로그·상태 막대
  for (const d of data.designs) { design = d; buildTwin(d); }
  const all = Object.values(fkAll).map((x) => x.maxErrMm);
  const fkMax = Math.max(...all);
  console.log(`[FK] overall max error ${fkMax.toFixed(6)} mm`);
  log(fkMax < 2 ? "ok" : "err", `FK 전체: 최대 오차 ${fkMax.toFixed(6)} mm (기준 2 mm)`);
  $("stFk").innerHTML = `<span class="${fkMax < 2 ? "ok" : "bad"}">FK 오차 ${fkMax.toFixed(6)} mm</span>`;

  setTool("select");
  setSpeed(4);
  buildLine();
  openDoc(0);
  setView("iso");
  drawLine();
  setLinePlaying(true);
  $("loading").remove();
  frame();
  loadSmoothMeshes();
}

// 매끈한 .dae(13.4 MB)는 STL 로 먼저 띄운 뒤 뒤에서 받아 바꾼다. 실패하면 STL 유지.
async function loadSmoothMeshes() {
  const t0 = performance.now();
  try {
    const { ColladaLoader } = await import("three/addons/loaders/ColladaLoader.js");
    const loaded = await Promise.all(MESHES.map((m) => new ColladaLoader().loadAsync(`meshes/${m}.dae`)));
    visualSets.dae = {};
    MESHES.forEach((m, i) => { visualSets.dae[m] = daeVisual(loaded[i], m); });
    log("ok", `로봇 외형 메시(DAE) 불러옴 — ${((performance.now() - t0) / 1000).toFixed(1)} s`);
    if (meshKind === "stl") setMeshKind("dae");
  } catch (err) {
    console.warn("[meshes] .dae 를 못 읽어 STL 로 둡니다", err);
    log("warn", "DAE 메시를 못 읽어 STL 로 표시");
    window.__meshes = "stl";
  }
  syncChecks();
}

/* =====================================================================
   녹화 모드 (?record) — README 데모 GIF 를 프레임 단위로 찍을 때만 켜진다.
   실시간 재생을 멈추고, 시각·카메라·라인 위치를 바깥에서 한 칸씩 정한다.
   ===================================================================== */
if (new URLSearchParams(location.search).has("record")) {
  controls.enableDamping = false;
  $("stFps").style.display = "none";   // headless 소프트웨어 렌더링 fps 는 실제와 달라 녹화에서 뺀다
  const zAxis = new THREE.Vector3(0, 0, 1);
  window.__rec = {
    ready: () => !!(data && view && window.__meshes && window.__aiParity),
    pause() { setPlaying(false); setLinePlaying(false); },
    duration: () => view?.duration ?? 0,
    setTime(x) { t = x; applyTime(t); },
    orbit(deg) {
      const off = camera.position.clone().sub(controls.target).applyAxisAngle(zAxis, THREE.MathUtils.degToRad(deg));
      camera.position.copy(controls.target).add(off);
      controls.update();
    },
    dolly(k) {
      const off = camera.position.clone().sub(controls.target).multiplyScalar(k);
      camera.position.copy(controls.target).add(off);
      controls.update();
    },
    act,
    select,
    showTab,
    setLine(s) { line.s = s; drawLine(); },
    lineEnd,
    setBottom(px) { document.documentElement.style.setProperty("--bottom-h", `${px}px`); resize(); },
    hideTip() { $("tip").hidden = true; ttip.hidden = true; },
    frame: () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))),
  };
}

boot().catch((err) => {
  console.error(err);
  $("loading").textContent = "데이터를 불러오지 못했습니다";
});

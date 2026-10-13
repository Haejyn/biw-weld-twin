"""설계 수정 에이전트 — 못 쏘는 타점이 있는 설계를 제약 안에서 최소한으로 고친다.

LLM 은 한국어 요청을 제약 JSON 으로 옮기고, 도구(predict · sensitivity · simulate · submit)를 불러 수정안을 찾는다.
제약 검증, 범위 자르기, 호출 한도, '시뮬레이터로 확인한 설계만 제출 가능' 은 전부 코드가 맡는다.

설계는 mm 단위 dict (키 10개, dataset.SPACE 와 같은 이름). 시뮬레이터·대리 모델에는 m 로 바꿔 넘긴다.
실행: python -m src.agent --design A --request "플랜지 폭은 그대로, 첫 타점 거리는 60 이하"
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import PARAMS, SPACE  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "http://localhost:11434"
MODEL = "qwen3.5:4b"
MAX_CALLS = 12          # 도구 호출 전체 한도
MAX_SIM = 4             # simulate 한도
MAX_TURNS = 24          # LLM 응답 횟수 한도 (도구 없이 말만 하는 경우 포함)
TOL = 0.06              # mm. 같은 설계로 보는 허용 오차 (표시용 반올림 때문)
SWEEP_POINTS = 9

VARS = list(PARAMS)
RANGE = {k: (SPACE[k][0] * 1000, SPACE[k][1] * 1000) for k in VARS}   # mm
LABEL = {
    "flange_width": "플랜지 폭",
    "member_wall_height": "크로스멤버 높이",
    "member_first_spot": "첫 타점 거리",
    "member_y0": "크로스멤버 시작 위치(y)",
    "member_x": "크로스멤버 위치(x)",
    "pillar_x": "B필러 위치(x)",
    "pillar_w": "B필러 폭",
    "sill_top": "실 윗면 높이",
    "floor_z": "바닥 높이",
    "sill_pillar_gap": "실 타점-B필러 간격",
}
REASON_KO = {"unreachable": "도달 불가", "joint_limit": "관절 한계", "gun_collision": "건 간섭",
             "arm_collision": "팔 간섭", "approach_collision": "접근 경로 간섭"}

# 기준 설계 (README 의 A, B, C). 단위 mm
DESIGN_A = {"flange_width": 15.0, "member_wall_height": 120.0, "member_first_spot": 30.0, "member_y0": 120.0,
            "member_x": 1200.0, "pillar_x": 1200.0, "pillar_w": 160.0, "sill_top": 550.0, "floor_z": 300.0,
            "sill_pillar_gap": 30.0}
NAMED = {"A": DESIGN_A,
         "B": dict(DESIGN_A, flange_width=30.0),
         "C": dict(DESIGN_A, flange_width=30.0, member_first_spot=80.0, sill_pillar_gap=50.0)}


class ConstraintError(ValueError):
    pass


# ── 설계 · 제약 ──────────────────────────────────────────────────────────────
def to_m(design_mm: dict) -> dict:
    return {k: design_mm[k] / 1000 for k in VARS}


def to_mm(design_m: dict) -> dict:
    return {k: design_m[k] * 1000 for k in VARS}


def norm_dist(a: dict, b: dict) -> float:
    """두 설계(mm) 사이 거리 — 변수마다 범위 폭으로 나눈 뒤 유클리드 (explore.norm_dist 와 같은 식)."""
    return float(sum(((a[k] - b[k]) / (RANGE[k][1] - RANGE[k][0])) ** 2 for k in VARS) ** 0.5)


def same_design(a: dict, b: dict) -> bool:
    return all(abs(a[k] - b[k]) <= TOL for k in VARS)


def show(design: dict, ref: dict | None = None) -> dict:
    """LLM 에 보이는 설계: 0.1 mm 로 반올림, ref 가 있으면 ref 와 다른 변수만."""
    return {k: round(design[k], 1) for k in VARS if ref is None or abs(design[k] - ref[k]) > TOL}


class Constraints:
    """변수별 {fixed, min, max} (mm). 비어 있으면 제약 없음 (범위 RANGE 만 지킨다)."""

    def __init__(self, spec: dict | None = None):
        self.spec = {k: dict(v) for k, v in (spec or {}).items()}

    def bounds(self, var: str) -> tuple[float, float]:
        lo, hi = RANGE[var]
        c = self.spec.get(var, {})
        if "fixed" in c:
            return c["fixed"], c["fixed"]
        return max(lo, c.get("min", lo)), min(hi, c.get("max", hi))

    def violations(self, design: dict) -> list[str]:
        out = []
        for k in VARS:
            v = design[k]
            lo, hi = RANGE[k]
            if v < lo - TOL or v > hi + TOL:
                out.append(f"{k} {v:.1f} 은 허용 범위 {lo:g}~{hi:g} 밖")
            c = self.spec.get(k, {})
            if "fixed" in c and abs(v - c["fixed"]) > TOL:
                out.append(f"{k} 는 {c['fixed']:.1f} 로 고정인데 {v:.1f}")
            if "min" in c and v < c["min"] - TOL:
                out.append(f"{k} {v:.1f} 이 최소 {c['min']:.1f} 미만")
            if "max" in c and v > c["max"] + TOL:
                out.append(f"{k} {v:.1f} 이 최대 {c['max']:.1f} 초과")
        return out

    def clamp(self, design: dict) -> tuple[dict, dict]:
        """제약과 범위 안으로 자른다. 돌려주는 것: (잘린 설계, {변수: [원래, 바뀐 값]})"""
        out, notes = {}, {}
        for k in VARS:
            lo, hi = self.bounds(k)
            v = lo if lo == hi else min(max(round(design[k], 1), lo), hi)
            out[k] = v
            if abs(v - design[k]) > TOL:
                notes[k] = [round(design[k], 1), round(v, 1)]
        return out, notes

    def describe(self) -> str:
        if not self.spec:
            return "없음"
        parts = []
        for k, c in self.spec.items():
            if "fixed" in c:
                parts.append(f"{LABEL[k]}({k}) = {c['fixed']:g} 고정")
            else:
                if "min" in c:
                    parts.append(f"{LABEL[k]}({k}) >= {c['min']:g}")
                if "max" in c:
                    parts.append(f"{LABEL[k]}({k}) <= {c['max']:g}")
        return ", ".join(parts)

    def key(self) -> dict:
        """비교용: {변수: (fixed|min|max 값들)}"""
        return {k: {a: round(b, 3) for a, b in c.items()} for k, c in self.spec.items()}


def parse_constraints(raw, start: dict) -> Constraints:
    """LLM 이 낸 {"constraints": [{"var", "type", "value"}]} 를 검증해 Constraints 로. 틀리면 ConstraintError."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ConstraintError(f"JSON 이 아님: {e}")
    items = raw.get("constraints") if isinstance(raw, dict) else raw
    if not isinstance(items, list):
        raise ConstraintError("constraints 목록이 없음")
    spec: dict = {}
    for it in items:
        if not isinstance(it, dict):
            raise ConstraintError(f"항목 형식 오류: {it!r}")
        var, typ, val = it.get("var"), it.get("type"), it.get("value")
        if var not in RANGE:
            raise ConstraintError(f"모르는 변수 {var!r}. 가능한 변수: {', '.join(VARS)}")
        if typ not in ("fixed", "min", "max"):
            raise ConstraintError(f"{var}: type 은 fixed/min/max 중 하나여야 함 ({typ!r})")
        if val is None:
            if typ != "fixed":
                raise ConstraintError(f"{var}: {typ} 에는 value 가 필요함")
            val = start[var]
        try:
            val = float(val)
        except (TypeError, ValueError):
            raise ConstraintError(f"{var}: value 가 숫자가 아님 ({val!r})")
        lo, hi = RANGE[var]
        if not (lo - 1e-9 <= val <= hi + 1e-9):
            raise ConstraintError(f"{var}: {val:g} mm 은 허용 범위 {lo:g}~{hi:g} 밖")
        c = spec.setdefault(var, {})
        if typ in c and abs(c[typ] - val) > 1e-9:
            raise ConstraintError(f"{var}: {typ} 가 두 번 다른 값으로 나옴")
        c[typ] = val
    for var, c in spec.items():
        if "fixed" in c:
            if ("min" in c and c["fixed"] < c["min"]) or ("max" in c and c["fixed"] > c["max"]):
                raise ConstraintError(f"{var}: 고정값이 min/max 와 어긋남")
            spec[var] = {"fixed": c["fixed"]}
        elif "min" in c and "max" in c and c["min"] > c["max"]:
            raise ConstraintError(f"{var}: min {c['min']:g} > max {c['max']:g}")
    return Constraints(spec)


# ── 도구 뒷단 (실제 / 가짜 교체 가능) ───────────────────────────────────────────
class RealBackend:
    """대리 모델과 시뮬레이터. 무거운 모듈은 처음 쓸 때 불러온다."""

    def __init__(self):
        self._model = None
        self._thr = None

    def _load(self):
        if self._model is None:
            import warnings
            warnings.filterwarnings("ignore")
            import lightgbm as lgb
            self._model = lgb.Booster(model_file=str(ROOT / "models" / "surrogate.txt"))
            self._thr = json.loads((ROOT / "results" / "surrogate.json").read_text(encoding="utf-8"))["threshold"]

    def predict(self, design_mm: dict) -> dict:
        """{타점 id: 쏠 수 있을 확률(로봇 중 최대)}"""
        self._load()
        import numpy as np
        from body import spots
        from dataset import make_design
        from stage1 import BASES
        from surrogate import design_features
        v = to_m(design_mm)
        p = self._model.predict(design_features(v), num_threads=1).reshape(-1, len(BASES)).max(axis=1)
        ids = [s.id for s in spots(make_design(v))]
        return {i: float(x) for i, x in zip(ids, np.asarray(p))}

    @property
    def threshold(self) -> float:
        self._load()
        return self._thr

    def simulate(self, design_mm: dict) -> dict:
        """{못 쏘는 타점 id: {"group": .., "reason": 가장 멀리 간 실패 이유, "reasons": {이유: 로봇 수}}}"""
        from dataset import make_design
        from reach import RANK
        from stage1 import judge
        S, _, robots, verdicts = judge(make_design(to_m(design_mm)))
        out = {}
        for i, s in enumerate(S):
            vs = [verdicts[i, r] for r in range(len(robots))]
            if not any(v.ok for v in vs):
                counts: dict = {}
                for v in vs:
                    counts[v.reason] = counts.get(v.reason, 0) + 1
                out[s.id] = {"group": s.group, "reason": max(counts, key=lambda r: RANK[r]), "reasons": counts}
        return out


# ── LLM ────────────────────────────────────────────────────────────────────────
def ollama_chat(messages, tools=None, fmt=None, model=MODEL, url=OLLAMA_URL, timeout=600) -> dict:
    """Ollama /api/chat 한 번. 돌려주는 것은 응답 message (role · content · tool_calls)."""
    body = {"model": model, "messages": messages, "stream": False, "think": False, "keep_alive": "30m",
            "options": {"temperature": 0.2}}
    if tools:
        body["tools"] = tools
    if fmt:
        body["format"] = fmt
    req = urllib.request.Request(url + "/api/chat", json.dumps(body).encode("utf-8"),
                                 {"Content-Type": "application/json"})
    last = None
    for _ in range(2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))["message"]
        except OSError as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f"Ollama 호출 실패: {last}")


DESIGN_PROP = {k: {"type": "number", "description": f"{LABEL[k]} (mm)"} for k in VARS}
TOOLS = [
    {"type": "function", "function": {
        "name": "predict",
        "description": "대리 모델로 못 쏘는 타점을 예측한다 (빠름, 근사). 적지 않은 변수는 시작 설계 값을 쓴다.",
        "parameters": {"type": "object", "properties": {"design": {"type": "object", "properties": DESIGN_PROP}},
                       "required": ["design"]}}},
    {"type": "function", "function": {
        "name": "sensitivity",
        "description": "design 의 다른 변수는 두고 var 하나만 허용 범위에서 9점 훑어, 점마다 예측된 못 쏘는 타점 수를 돌려준다 (빠름).",
        "parameters": {"type": "object", "properties": {
            "design": {"type": "object", "properties": DESIGN_PROP},
            "var": {"type": "string", "enum": VARS}}, "required": ["design", "var"]}}},
    {"type": "function", "function": {
        "name": "simulate",
        "description": f"시뮬레이터로 정확히 판정한다. 못 쏘는 타점과 이유를 돌려준다 (약 10초, 최대 {MAX_SIM}회).",
        "parameters": {"type": "object", "properties": {"design": {"type": "object", "properties": DESIGN_PROP}},
                       "required": ["design"]}}},
    {"type": "function", "function": {
        "name": "submit",
        "description": "최종안 제출. 가장 최근 simulate 한 설계와 같고 못 쏘는 타점이 0 이고 제약을 지킬 때만 받는다.",
        "parameters": {"type": "object", "properties": {
            "design": {"type": "object", "properties": DESIGN_PROP},
            "rationale": {"type": "string", "description": "무엇을 왜 바꿨는지 한두 문장"}},
            "required": ["design", "rationale"]}}},
]

CONSTRAINT_SCHEMA = {
    "type": "object",
    "properties": {"constraints": {"type": "array", "items": {
        "type": "object",
        "properties": {"var": {"type": "string", "enum": VARS},
                       "type": {"type": "string", "enum": ["fixed", "min", "max"]},
                       "value": {"type": ["number", "null"]}},
        "required": ["var", "type", "value"]}}},
    "required": ["constraints"]}

PARSE_SYSTEM = (
    "사용자의 설계 수정 요청에서 제약만 뽑아 JSON 으로 낸다. 길이는 모두 mm.\n"
    "항목: var (변수 이름), type (fixed = 바꾸지 말 것 / min = 이 값 이상 / max = 이 값 이하), value (mm, 그대로 두라는 말이면 null).\n"
    "변수:\n" + "\n".join(f"- {k}: {LABEL[k]}, 허용 {RANGE[k][0]:g}~{RANGE[k][1]:g}" for k in VARS) + "\n"
    "요청에 제약이 없으면 constraints 는 빈 목록. 요청에 없는 제약은 만들지 않는다.\n"
    "'그대로', '고정', '바꾸지 말 것' 은 fixed 이고 value 는 null. 요청에 숫자가 없으면 value 에 숫자를 쓰지 않는다.\n"
    "예) '크로스멤버 높이는 건드리지 말고 실 윗면 높이는 600 이하로' -> "
    '{"constraints":[{"var":"member_wall_height","type":"fixed","value":null},{"var":"sill_top","type":"max","value":600}]}\n'
    "예) '바닥 높이 280 이상 300 이하로' -> "
    '{"constraints":[{"var":"floor_z","type":"min","value":280},{"var":"floor_z","type":"max","value":300}]}\n'
    "예) '전부 쏠 수 있게 고쳐줘' -> "
    '{"constraints":[]}'
)

ACTION_SCHEMA = {
    "type": "object",
    "properties": {"tool": {"type": "string", "enum": ["predict", "sensitivity", "simulate", "submit"]},
                   "design": {"type": "object", "properties": DESIGN_PROP},
                   "var": {"type": "string", "enum": VARS},
                   "rationale": {"type": "string"}},
    "required": ["tool", "design"]}

AGENT_SYSTEM = (
    "너는 차체 설계 수정 도우미다. 못 쏘는 타점이 있는 설계를 제약 안에서 가장 적게 바꿔 모든 타점을 쏠 수 있게 만든다.\n"
    "길이는 mm. 도구에 설계를 넘길 때 바꾼 변수만 적어도 된다 (나머지는 시작 설계 값).\n"
    "순서: simulate 로 시작 설계의 못 쏘는 타점과 이유를 보고, sensitivity 로 어느 변수가 효과 있는지 보고, "
    "predict 로 후보를 거른 뒤, simulate 로 확인해서 0개면 submit 한다.\n"
    "타점 이름: S = 실 위 플랜지(변수: 실 타점-B필러 간격, B필러 위치·폭, 실 윗면 높이), B = B필러 뒤 플랜지, "
    "M = 크로스멤버 플랜지(플랜지 폭, 첫 타점 거리, 높이, 위치).\n"
    "이유: unreachable 도달 불가, joint_limit 관절 한계, gun_collision 건 간섭, arm_collision 팔 간섭, approach_collision 접근 경로 간섭.\n"
    f"한도: 도구 호출 전체 {MAX_CALLS}회, simulate {MAX_SIM}회. 제약은 코드가 강제하며 어긴 값은 잘리거나 거부된다. "
    "고정된 변수는 건드리지 않는다. 변경량이 작을수록 좋다. predict 는 근사라서 submit 전에 반드시 simulate 로 확인한다.\n"
    "못 쏘는 타점마다 관련 변수를 sensitivity 로 훑는다. 한 변수로 안 풀리면 다른 변수를 함께 바꾼다. "
    "predict 가 0개를 보이면 바로 simulate 로 확인한다. 말만 하지 말고 매번 도구를 부른다."
)


# ── 에이전트 ───────────────────────────────────────────────────────────────────
def parse_request(request: str, start: dict, llm) -> tuple[Constraints, dict]:
    """한국어 요청 -> Constraints. 한 번 틀리면 오류를 알려 주고 한 번 더 시킨다."""
    msgs = [{"role": "system", "content": PARSE_SYSTEM}, {"role": "user", "content": request}]
    info = {"attempts": 0, "raw": None, "error": None}
    for _ in range(2):
        info["attempts"] += 1
        msg = llm(msgs, None, CONSTRAINT_SCHEMA)
        info["raw"] = msg.get("content")
        try:
            return parse_constraints(info["raw"], start), info
        except ConstraintError as e:
            info["error"] = str(e)
            msgs += [{"role": "assistant", "content": msg.get("content") or ""},
                     {"role": "user", "content": f"제약이 잘못됐다: {e}. 고쳐서 JSON 만 다시 낸다."}]
    raise ConstraintError(info["error"])


class DesignAgent:
    def __init__(self, backend, llm=ollama_chat, max_calls=MAX_CALLS, max_sim=MAX_SIM):
        self.backend, self.llm = backend, llm
        self.max_calls, self.max_sim = max_calls, max_sim

    # 상태 초기화
    def _reset(self, start: dict, constraints: Constraints):
        self.start, self.cons = dict(start), constraints
        self.trace, self.n_calls, self.n_sim, self.n_fallback = [], 0, 0, 0
        self.last_sim = None        # {"design", "n", "spots"}
        self.sims = []              # 지금까지 simulate 한 것 (best 찾기용)
        self.accepted, self.rationale = None, None

    # ── 도구 ──
    def _design(self, args: dict):
        """도구 인자 -> 전체 설계(mm). 적지 않은 변수는 시작 설계 값. 오류면 문자열."""
        d = args.get("design") if isinstance(args.get("design"), dict) else None
        if d is None and any(k in args for k in VARS):
            d = {k: args[k] for k in VARS if k in args}
        if d is None:
            return None, "design 객체가 필요함"
        unknown = [k for k in d if k not in RANGE]
        if unknown:
            return None, f"모르는 변수 {unknown}. 가능한 변수: {', '.join(VARS)}"
        out = dict(self.start)
        for k, v in d.items():
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                return None, f"{k} 값이 숫자가 아님 ({v!r})"
        return out, None

    def _clamped(self, args: dict):
        d, err = self._design(args)
        if err:
            return None, None, err
        d, notes = self.cons.clamp(d)
        return d, notes, None

    def t_predict(self, args):
        d, notes, err = self._clamped(args)
        if err:
            return {"error": err}
        p = self.backend.predict(d)
        bad = sorted(((i, x) for i, x in p.items() if x < self.backend.threshold), key=lambda t: t[1])
        out = {"design": show(d, self.start), "unshootable_count": len(bad),
               "unshootable": [{"id": i, "p": round(x, 3)} for i, x in bad[:12]]}
        if notes:
            out["clamped"] = notes
        return out

    def t_sensitivity(self, args):
        d, notes, err = self._clamped(args)
        if err:
            return {"error": err}
        var = args.get("var")
        if var not in RANGE:
            return {"error": f"var 는 다음 중 하나: {', '.join(VARS)}"}
        lo, hi = self.cons.bounds(var)
        if lo == hi:
            return {"error": f"{var} 는 {lo:g} 로 고정이라 훑을 수 없음"}
        thr = self.backend.threshold
        sweep = []
        for x in [lo + (hi - lo) * i / (SWEEP_POINTS - 1) for i in range(SWEEP_POINTS)]:
            x = round(x, 1)
            x = min(max(x, lo), hi)
            p = self.backend.predict(dict(d, **{var: x}))
            ids = sorted(i for i, v in p.items() if v < thr)
            sweep.append([x, len(ids)] + ([ids] if 0 < len(ids) <= 3 else []))
        out = {"var": var, "range": [round(lo, 1), round(hi, 1)], "sweep": sweep,
               "note": "[값 mm, 예측 못 쏘는 타점 수, (3개 이하면 타점 id)]. 다른 변수는 입력 설계 그대로"}
        if notes:
            out["clamped"] = notes
        return out

    def t_simulate(self, args):
        if self.n_sim >= self.max_sim:
            return {"error": f"simulate 한도 {self.max_sim}회를 다 썼음"}
        d, notes, err = self._clamped(args)
        if err:
            return {"error": err}
        self.n_sim += 1
        res = self.backend.simulate(d)
        self.last_sim = {"design": d, "n": len(res), "spots": res}
        self.sims.append(self.last_sim)
        counts: dict = {}
        for v in res.values():
            counts[v["reason"]] = counts.get(v["reason"], 0) + 1
        out = {"design": show(d, self.start), "unshootable_count": len(res),
               "unshootable": [{"id": i, "reason": v["reason"]} for i, v in list(res.items())[:15]],
               "reason_counts": counts, "simulate_left": self.max_sim - self.n_sim}
        if notes:
            out["clamped"] = notes
        return out

    def t_submit(self, args):
        d, err = self._design(args)
        if err:
            return {"accepted": False, "problems": [err]}
        problems = []
        if self.last_sim is None:
            problems.append("simulate 로 확인한 설계가 없음")
        elif not same_design(d, self.last_sim["design"]):
            problems.append("가장 최근 simulate 한 설계와 다름. 이 설계로 simulate 를 먼저 할 것")
        elif self.last_sim["n"] > 0:
            problems.append(f"simulate 결과 못 쏘는 타점 {self.last_sim['n']}개")
        problems += self.cons.violations(d)
        if problems:
            return {"accepted": False, "problems": problems}
        self.accepted = dict(self.last_sim["design"])
        self.rationale = str(args.get("rationale", ""))[:300]
        return {"accepted": True}

    def _summary(self, tool, res) -> str:
        if "error" in res:
            return "오류: " + res["error"]
        if tool == "predict":
            return f"예측 못 쏘는 타점 {res['unshootable_count']}개 {[x['id'] for x in res['unshootable']]}"
        if tool == "sensitivity":
            return f"{res['var']} 훑기 " + " ".join(f"{p[0]:g}:{p[1]}" for p in res["sweep"])
        if tool == "simulate":
            return f"시뮬레이션 못 쏘는 타점 {res['unshootable_count']}개 {res['reason_counts']}"
        return "제출 승인" if res.get("accepted") else "제출 거부: " + "; ".join(res["problems"])

    def call(self, tool: str, args: dict) -> dict:
        """도구 한 번 실행 + 한도 계산 + trace 기록."""
        t0 = time.time()
        self.n_calls += 1
        fn = {"predict": self.t_predict, "sensitivity": self.t_sensitivity,
              "simulate": self.t_simulate, "submit": self.t_submit}.get(tool)
        res = fn(args) if fn else {"error": f"모르는 도구 {tool}"}
        self.trace.append({"n": self.n_calls, "tool": tool, "args": args, "summary": self._summary(tool, res),
                           "result": res, "ms": round((time.time() - t0) * 1000)})
        return res

    # ── 전체 실행 ──
    def run(self, start: dict, request: str, constraints: Constraints | None = None) -> dict:
        t_all = time.time()
        parse_info = None
        if constraints is None:
            try:
                constraints, parse_info = parse_request(request, start, self.llm)
            except ConstraintError as e:
                return {"status": "실패: 제약 해석 오류", "success": False, "request": request,
                        "constraints": None, "parse_error": str(e), "start": show(start), "final": None,
                        "trace": [], "tool_calls": 0, "sim_calls": 0, "wall_s": round(time.time() - t_all, 1)}
        self._reset(start, constraints)

        try:
            p0 = self.backend.predict(self.start)
            thr = self.backend.threshold
            pre = sorted(i for i, x in p0.items() if x < thr)
            pre_txt = f"{len(pre)}개 {pre[:12]}"
        except Exception as e:  # 대리 모델이 없어도 에이전트는 돈다
            pre_txt = f"알 수 없음 ({e})"
        user = (f"요청: {request}\n시작 설계(mm): {json.dumps(show(start), ensure_ascii=False)}\n"
                f"제약: {constraints.describe()}\n시작 설계에서 대리 모델이 예측한 못 쏘는 타점: {pre_txt}")
        messages = [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user}]

        status, idle = None, 0
        for _ in range(MAX_TURNS):
            msg = self.llm(messages, TOOLS)
            calls = msg.get("tool_calls") or []
            via = "tools"
            if not calls:
                # 소형 모델이 도구 결과 뒤에 빈 응답을 내거나 설명만 하고 끝내는 일이 잦다.
                # user 메시지로 재촉하면 빈 응답이 더 나와서, 같은 대화를 JSON 형식 강제로 한 번 더 받는다.
                act = self._structured_action(messages)
                if act:
                    calls, via = [{"function": act}], "format"
                    msg = {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls}
                    self.n_fallback += 1
            messages.append({k: v for k, v in msg.items() if k in ("role", "content", "tool_calls")})
            if not calls:
                idle += 1
                if idle >= 3:
                    status = "실패: 제출 없이 종료"
                    break
                continue
            idle = 0
            for c in calls:
                fn = c.get("function", {})
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                res = self.call(fn.get("name", ""), args)
                self.trace[-1]["via"] = via
                messages.append({"role": "tool", "tool_name": fn.get("name", ""),
                                 "content": json.dumps(res, ensure_ascii=False)})
                if self.accepted:
                    break
                if self.n_calls >= self.max_calls:
                    status = "실패: 한도 초과"
                    break
            if self.accepted or status:
                break
        else:
            status = "실패: 제출 없이 종료"

        out = self._result(request, status)
        out["wall_s"] = round(time.time() - t_all, 1)
        out["parse"] = parse_info
        return out

    def _structured_action(self, messages):
        """도구 호출 없는 응답 뒤에, 다음 행동을 {tool, design, var, rationale} JSON 으로 강제해 받는다. 실패하면 None."""
        try:
            m = self.llm(messages, None, ACTION_SCHEMA)
            j = json.loads(m.get("content") or "")
            name = j["tool"]
            if name not in ("predict", "sensitivity", "simulate", "submit"):
                return None
            args = {k: j[k] for k in ("design", "var", "rationale") if k in j}
            if name != "sensitivity":
                args.pop("var", None)
            if name != "submit":
                args.pop("rationale", None)
            return {"name": name, "arguments": args}
        except (ValueError, KeyError, TypeError, RuntimeError):
            return None

    def _result(self, request, status) -> dict:
        ok = self.accepted is not None
        best = None
        if self.sims:
            b = min(self.sims, key=lambda s: (s["n"], norm_dist(s["design"], self.start)))
            best = {"design": show(b["design"]), "unshootable": b["n"]}
        final = self.accepted or (self.sims and min(self.sims, key=lambda s: (s["n"], norm_dist(s["design"], self.start)))["design"]) or None
        return {
            "status": "성공" if ok else status, "success": ok, "request": request,
            "constraints": self.cons.key(), "constraints_text": self.cons.describe(),
            "start": show(self.start), "final": show(final) if final else None,
            "changes": ({k: [round(self.start[k], 1), round(final[k], 1)] for k in VARS
                         if abs(final[k] - self.start[k]) > TOL} if final else None),
            "distance": round(norm_dist(final, self.start), 4) if final else None,
            "best_simulated": best, "rationale": self.rationale,
            "tool_calls": self.n_calls, "sim_calls": self.n_sim, "format_fallbacks": self.n_fallback,
            "trace": self.trace,
        }


def print_result(r: dict):
    print(f"요청: {r['request']}")
    print(f"해석한 제약: {r.get('constraints_text') or r.get('parse_error')}")
    for t in r["trace"]:
        print(f"{t['n']:>2}. {t['tool']:<11} {t['summary']}  ({t['ms']} ms)" + ("  [형식 강제]" if t.get("via") == "format" else ""))
    print(f"결과: {r['status']}")
    if r.get("changes"):
        print("바뀐 변수(mm):", ", ".join(f"{k} {a:g}->{b:g}" for k, (a, b) in r["changes"].items()))
    if r.get("distance") is not None:
        print(f"변경 거리 {r['distance']:.3f} · 도구 {r['tool_calls']}회 · simulate {r['sim_calls']}회 · {r['wall_s']} s")
    if r.get("rationale"):
        print("근거:", r["rationale"])
    if not r["success"] and r.get("best_simulated"):
        print("시뮬레이션한 것 중 가장 나은 설계:", json.dumps(r["best_simulated"], ensure_ascii=False))


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description="설계 수정 에이전트")
    ap.add_argument("--design", required=True, help="A | B | C | JSON (mm, 변수 10개, 빠진 변수는 A 값)")
    ap.add_argument("--request", required=True, help="한국어 요청")
    ap.add_argument("--trace", default=str(ROOT / "results" / "agent_trace_last.json"), help="trace 저장 경로")
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args(argv)
    if a.design.upper() in NAMED:
        start = dict(NAMED[a.design.upper()])
    else:
        start = dict(DESIGN_A, **{k: float(v) for k, v in json.loads(a.design).items()})
    bad = [k for k in start if k not in RANGE]
    if bad:
        sys.exit(f"모르는 변수 {bad}")
    out_of_range = Constraints().violations(start)
    if out_of_range:
        sys.exit("설계가 허용 범위 밖: " + "; ".join(out_of_range))
    llm = lambda m, t=None, f=None: ollama_chat(m, t, f, model=a.model)  # noqa: E731
    r = DesignAgent(RealBackend(), llm).run(start, a.request)
    print_result(r)
    Path(a.trace).parent.mkdir(parents=True, exist_ok=True)
    Path(a.trace).write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"trace 저장: {a.trace}")
    return 0 if r["success"] else 1


if __name__ == "__main__":
    sys.exit(main())

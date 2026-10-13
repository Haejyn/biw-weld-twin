"""설계 수정 에이전트 평가 — 에이전트 대 기존 탐색(explore.py 방식) 비교.

시작 설계: 시험용 설계(data/gen/designs_1.json) 중 시뮬레이터 판정에서 못 쏘는 타점이 1개 이상인 것 N 개 + README 의 A, B.
조건 1: 제약 없음. 조건 2: 기준선이 그 시작 설계에서 가장 많이 바꾼 변수를 '그대로 두라' 는 제약 한 개 (한국어 요청).
기준선: 시작 설계 주변 후보 N 만 개를 대리 모델로 거르고, 제약을 만족하는 것 중 시작 설계에 가까운 순으로 시뮬레이터에 확인.

실행: python scripts/agent_eval.py [--n 20] [--seed 0] [--workers 10]   (중간 결과는 results/agent_eval_partial.json 에 쌓이며 이어서 돈다)
      python scripts/agent_eval.py --report-only      (results/agent_eval.json -> docs/agent_eval.md 만 다시 쓴다)
"""
import argparse
import json
import platform
import subprocess
import sys
import time
import urllib.request
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import agent  # noqa: E402
from agent import LABEL, RANGE, VARS, Constraints, DesignAgent, RealBackend, norm_dist  # noqa: E402

RESULTS = ROOT / "results"
PARTIAL = RESULTS / "agent_eval_partial.json"
OUT_JSON = RESULTS / "agent_eval.json"
OUT_MD = ROOT / "docs" / "agent_eval.md"
MAX_BASELINE_SIM = 20      # 기준선이 시뮬레이터로 확인해 보는 후보 수 상한 (explore.py 의 상위 20 과 같음)

REQ_FREE = ["못 쏘는 타점이 없게 최소한만 고쳐줘",
            "이 설계에서 못 쏘는 타점 없애줘. 가능하면 조금만 바꾸고",
            "전 타점 다 쏠 수 있게 수정해줘, 변경은 최소로"]
REQ_FIX = ["{label} 값은 그대로 두고 못 쏘는 타점이 없게 최소한만 고쳐줘",
           "{label} 값은 바꾸면 안 돼. 나머지로 해결해줘",
           "{label} 고정. 못 쏘는 타점 없게 최소한으로 바꿔줘"]


# ── 시작 설계 ──────────────────────────────────────────────────────────────
def pick_starts(n, seed):
    gen = ROOT / "data" / "gen"
    designs = json.loads((gen / "designs_1.json").read_text(encoding="utf-8"))
    arr = np.load(gen / "labels_1.npz")["arr"]
    ok = {}
    for d, i, _, f, _ in arr:
        k = (int(d), int(i))
        ok[k] = ok.get(k, False) or bool(f)
    bad_n = {}
    for (d, _), v in ok.items():
        if not v:
            bad_n[d] = bad_n.get(d, 0) + 1
    ids = sorted(bad_n)
    rng = np.random.default_rng(seed)
    pick = sorted(rng.choice(ids, size=min(n, len(ids)), replace=False).tolist())
    starts = [{"name": "A", "design": dict(agent.NAMED["A"]), "label_unshootable": 2},
              {"name": "B", "design": dict(agent.NAMED["B"]), "label_unshootable": 2}]
    for d in pick:
        starts.append({"name": f"T{d:03d}", "design": agent.to_mm({k: designs[d][k] for k in VARS}),
                       "label_unshootable": bad_n[d]})
    return starts


# ── 기준선 ─────────────────────────────────────────────────────────────────
def baseline_task(a):
    """후보 샘플 -> 대리 모델 거름 -> 거리순으로 시뮬레이터 확인. (작업자 프로세스에서 실행)"""
    key, start, spec, seed, ncand = a
    t_all = time.time()
    be = RealBackend()
    cons = Constraints(spec)
    rng = np.random.default_rng(seed)
    cands = []
    for _ in range(ncand):
        v = {}
        for k in VARS:
            lo, hi = cons.bounds(k)
            v[k] = lo if lo == hi else float(np.clip(start[k] + rng.normal(0, 0.25) * (RANGE[k][1] - RANGE[k][0]), lo, hi))
        cands.append(v)
    t = time.time()
    thr = be.threshold
    passed = [i for i, v in enumerate(cands) if all(x >= thr for x in be.predict(v).values())]
    scan_s = time.time() - t
    passed.sort(key=lambda i: norm_dist(cands[i], start))
    tried, found, sim_s = [], None, 0.0
    for i in passed[:MAX_BASELINE_SIM]:
        t = time.time()
        n = len(be.simulate(cands[i]))
        sim_s += time.time() - t
        tried.append([round(norm_dist(cands[i], start), 4), n])
        if n == 0:
            found = cands[i]
            break
    ref = found or (cands[passed[0]] if passed else None)
    return {"key": key, "success": found is not None, "sim_calls": len(tried), "scan_s": round(scan_s, 1),
            "sim_s": round(sim_s, 1), "wall_s": round(time.time() - t_all, 1), "surrogate_pass": len(passed),
            "tried": tried, "final": {k: round(found[k], 1) for k in VARS} if found else None,
            "distance": round(norm_dist(found, start), 4) if found else None,
            "most_changed": (max(VARS, key=lambda k: abs(ref[k] - start[k]) / (RANGE[k][1] - RANGE[k][0]))
                             if ref else None),
            "most_changed_from_verified": found is not None}


def run_baselines(tasks, workers, done, on_done=None):
    todo = [t for t in tasks if t[0] not in done]
    out = {}
    if not todo:
        return out
    t0 = time.time()
    with Pool(min(workers, len(todo))) as pool:
        for r in pool.imap_unordered(baseline_task, todo):
            out[r["key"]] = r
            if on_done:
                on_done(r)
            print(f"[baseline] {r['key']} 성공={r['success']} 시뮬 {r['sim_calls']}회 {r['wall_s']} s "
                  f"({len(out)}/{len(todo)}, 경과 {time.time() - t0:.0f} s)", flush=True)
    return out


# ── 환경 ───────────────────────────────────────────────────────────────────
def env_info():
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    import lightgbm
    info = {"python": platform.python_version(), "os": platform.platform(), "lightgbm": lightgbm.__version__,
            "cpu_count": __import__("os").cpu_count(),
            "gpu": sh(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]),
            "ollama": sh(["ollama", "--version"]), "model": agent.MODEL, "date": time.strftime("%Y-%m-%d")}
    try:
        tags = json.loads(urllib.request.urlopen(agent.OLLAMA_URL + "/api/tags", timeout=10).read())
        m = next(x for x in tags["models"] if x["name"] == agent.MODEL)
        info["model_detail"] = f"{m['details']['parameter_size']} {m['details']['quantization_level']}"
    except Exception:
        pass
    return info


# ── 집계 · 보고서 ───────────────────────────────────────────────────────────
def med(xs):
    return float(np.median(xs)) if xs else None


def mean(xs):
    return float(np.mean(xs)) if xs else None


def effective_match(r):
    """해석한 제약이 의도와 같은가 — 허용 범위 전체를 그대로 두는 min/max(실제로는 제약이 아닌 것)는 뺀 뒤 비교."""
    got = {}
    for k, c in (r["agent"]["constraints"] or {}).items():
        c = {t: v for t, v in c.items()
             if not (t == "min" and v <= RANGE[k][0] + 1e-6) and not (t == "max" and v >= RANGE[k][1] - 1e-6)}
        if c:
            got[k] = c
    var = r["intended_var"]
    if var is None:
        return not got
    return set(got) == {var} and set(got[var]) == {"fixed"} and abs(got[var]["fixed"] - r["start_design"][var]) < 0.06


def summarize(runs):
    out = {}
    for cond in (1, 2):
        rs = [r for r in runs if r["cond"] == cond]
        if not rs:
            continue
        ag = [r["agent"] for r in rs]
        bl = [r["baseline"] for r in rs]
        both = [(r["agent"]["distance"], r["baseline"]["distance"]) for r in rs
                if r["agent"]["success"] and r["baseline"]["success"]]
        out[cond] = {
            "n": len(rs),
            "agent_success": sum(a["success"] for a in ag), "baseline_success": sum(b["success"] for b in bl),
            "agent_sim_mean": mean([a["sim_calls"] for a in ag]), "agent_sim_median": med([a["sim_calls"] for a in ag]),
            "baseline_sim_mean": mean([b["sim_calls"] for b in bl]), "baseline_sim_median": med([b["sim_calls"] for b in bl]),
            "agent_wall_mean": mean([a["wall_s"] for a in ag]), "agent_wall_median": med([a["wall_s"] for a in ag]),
            "baseline_wall_mean": mean([b["wall_s"] for b in bl]), "baseline_wall_median": med([b["wall_s"] for b in bl]),
            "agent_tool_calls_mean": mean([a["tool_calls"] for a in ag]),
            "agent_tool_calls_total": sum(a["tool_calls"] for a in ag),
            "agent_format_calls_total": sum(a.get("format_fallbacks", 0) for a in ag),
            "baseline_scan_mean": mean([b["scan_s"] for b in bl]), "baseline_sim_s_mean": mean([b["sim_s"] for b in bl]),
            "baseline_surrogate_pass_zero": sum(1 for b in bl if b["surrogate_pass"] == 0),
            "dist_pairs": len(both),
            "agent_dist_median_paired": med([x for x, _ in both]), "baseline_dist_median_paired": med([y for _, y in both]),
            "agent_dist_mean_paired": mean([x for x, _ in both]), "baseline_dist_mean_paired": mean([y for _, y in both]),
            "agent_dist_median_all": med([a["distance"] for a in ag if a["success"]]),
            "baseline_dist_median_all": med([b["distance"] for b in bl if b["success"]]),
            "parse_match": sum(a["parse_match"] for a in ag),
            "parse_match_effective": sum(effective_match(r) for r in rs if r["agent"]["constraints"] is not None),
            "parse_error": sum(a["status"] == "실패: 제약 해석 오류" for a in ag),
            "agent_status": {s: sum(1 for a in ag if a["status"] == s) for s in sorted({a["status"] for a in ag})},
        }
    return out


def fmt(x, d=1):
    return "-" if x is None else f"{x:.{d}f}"


def trace_block(r):
    a = r["agent"]
    lines = [f"시작 {r['start']} · 조건 {r['cond']} · 요청: {a['request']}",
             f"해석한 제약: {a['constraints_text']}"]
    for t in a["trace"]:
        lines.append(f"{t['n']:>2}. {t['tool']:<11} {t['summary']} ({t['ms']} ms)")
    lines.append(f"결과: {a['status']}" + (f" · 바뀐 변수(mm): " + ", ".join(
        f"{k} {x:g}->{y:g}" for k, (x, y) in a["changes"].items()) if a.get("changes") and a["success"] else ""))
    if a.get("rationale") and a["success"]:
        lines.append(f"근거: {a['rationale']}")
    if not a["success"] and a.get("best_simulated"):
        lines.append(f"시뮬레이션한 것 중 가장 나은 설계의 못 쏘는 타점 {a['best_simulated']['unshootable']}개")
    return "```\n" + "\n".join(lines) + "\n```"


def write_report(data):
    env, cfg, runs, S = data["env"], data["config"], data["runs"], data["summary"]
    L = ["# 설계 수정 에이전트 평가", "",
         "에이전트(`src/agent.py`)와 기존 탐색 방식(`src/explore.py` 를 시작 설계별로 바꾼 것)을 같은 시작 설계 · 같은 제약에서 비교했다. "
         "수치는 모두 `results/agent_eval.json` 에서 만들었다 (`python scripts/agent_eval.py --report-only`).", "",
         "## 환경", "",
         f"- 날짜 {env['date']} · Python {env['python']} · LightGBM {env['lightgbm']} · CPU {env['cpu_count']} 스레드",
         f"- GPU {env.get('gpu', '-')} (다른 작업과 공유) · Ollama {env.get('ollama', '-')}",
         f"- 모델 {env['model']} ({env.get('model_detail', '-')}) · think 끔 · temperature 0.2",
         f"- 시드 {cfg['seed']} · 시작 설계 {cfg['n_starts']}개 (시험 설계 {cfg['n_test']} + README 설계 A, B) × 조건 2 = 실행 {len(runs)}회",
         f"- 기준선 후보 {cfg['candidates']:,}개 · 시뮬레이터 확인 상한 {MAX_BASELINE_SIM}개",
         "", "## 방법", "",
         f"- 시작 설계: `data/gen/designs_1.json`(학습에 쓰지 않은 시험 설계 200개) 중 시뮬레이터 판정에서 못 쏘는 타점이 1개 이상인 설계 "
         f"{cfg['n_eligible']}개에서 {cfg['n_drawn']}개를 시드 {cfg['seed']} 로 뽑고 시간 때문에 앞의 {cfg['n_test']}개만 썼다. 여기에 README 의 A, B.",
         "- 조건 1: 제약 없음. 조건 2: 같은 시작 설계에서 기준선(조건 1)이 가장 많이 바꾼 변수(변수 범위 폭 대비)를 그대로 두라는 한국어 요청 한 문장. "
         "요청 문장은 템플릿 3개를 번갈아 쓴다.",
         "- 기준선: 시작 설계 주변(변수별 정규분포, 범위 폭의 0.25배)에서 후보를 뽑아 제약 안으로 자르고, 대리 모델 예측이 0개인 것을 "
         f"시작 설계와의 거리(`norm_dist`) 순으로 시뮬레이터에 확인해 처음으로 0개인 설계를 고른다. 최대 {MAX_BASELINE_SIM}개까지 확인한다.",
         "- 성공: 시뮬레이터가 0개로 확인했고 의도한 제약(조건 2 는 해당 변수가 시작값 그대로)을 지킨 경우. "
         "에이전트는 `submit` 이 승인된 경우만 성공으로 센다.",
         "- 변경 거리: 변수마다 허용 범위 폭으로 나눈 차이의 유클리드 거리. 두 방식이 모두 성공한 실행끼리만 짝지어 비교한다.",
         "- 시뮬레이터 호출: 에이전트는 `simulate` 호출 수, 기준선은 확인한 후보 수. 대리 모델 호출은 세지 않는다.",
         "- 시간: 에이전트는 요청 해석부터 종료까지(LLM 대기 포함). 기준선은 후보 대리 모델 거름 + 시뮬레이터 확인이며 "
         f"작업자 {cfg['workers']}개를 동시에 돌린 상태에서 잰 값이라 단독 실행보다 길 수 있다.",
         "", "## 결과", ""]
    L += ["| 항목 | 조건 1 (제약 없음) | 조건 2 (변수 하나 고정) |", "|---|---|---|"]

    def row(name, f):
        L.append(f"| {name} | " + " | ".join(f(S[c]) if str(c) in S or c in S else "-" for c in (1, 2)) + " |")
    S = {int(k): v for k, v in S.items()}
    row("실행 수", lambda s: str(s["n"]))
    row("성공률 에이전트", lambda s: f"{s['agent_success']}/{s['n']} ({100 * s['agent_success'] / s['n']:.0f} %)")
    row("성공률 기준선", lambda s: f"{s['baseline_success']}/{s['n']} ({100 * s['baseline_success'] / s['n']:.0f} %)")
    row("시뮬레이터 호출 에이전트 (평균 / 중앙값)", lambda s: f"{fmt(s['agent_sim_mean'])} / {fmt(s['agent_sim_median'], 0)}")
    row("시뮬레이터 호출 기준선 (평균 / 중앙값)", lambda s: f"{fmt(s['baseline_sim_mean'])} / {fmt(s['baseline_sim_median'], 0)}")
    row("걸린 시간 에이전트 (평균 / 중앙값, 초)", lambda s: f"{fmt(s['agent_wall_mean'], 0)} / {fmt(s['agent_wall_median'], 0)}")
    row("걸린 시간 기준선 (평균 / 중앙값, 초)", lambda s: f"{fmt(s['baseline_wall_mean'], 0)} / {fmt(s['baseline_wall_median'], 0)}")
    row("변경 거리 에이전트 (중앙값, 둘 다 성공한 실행)", lambda s: f"{fmt(s['agent_dist_median_paired'], 3)} (n={s['dist_pairs']})")
    row("변경 거리 기준선 (중앙값, 같은 실행)", lambda s: f"{fmt(s['baseline_dist_median_paired'], 3)} (n={s['dist_pairs']})")
    row("변경 거리 에이전트 (평균, 같은 실행)", lambda s: fmt(s['agent_dist_mean_paired'], 3))
    row("변경 거리 기준선 (평균, 같은 실행)", lambda s: fmt(s['baseline_dist_mean_paired'], 3))
    row("제약 해석이 의도와 일치", lambda s: f"{s['parse_match']}/{s['n']}")
    row("제약 해석 일치, 범위 전체를 뜻하는 min/max 는 무시", lambda s: f"{s['parse_match_effective']}/{s['n']}")
    row("제약 해석 오류로 중단", lambda s: str(s["parse_error"]))
    row("에이전트 도구 호출 (평균)", lambda s: fmt(s['agent_tool_calls_mean']))
    row("그중 JSON 형식 강제로 받은 호출", lambda s: f"{s['agent_format_calls_total']}/{s['agent_tool_calls_total']}")
    row("기준선: 대리 모델 거름 시간 (평균, 초)", lambda s: fmt(s['baseline_scan_mean']))
    row("기준선: 시뮬레이터 확인 시간 (평균, 초)", lambda s: fmt(s['baseline_sim_s_mean']))
    row("기준선: 대리 모델을 통과한 후보가 0개인 실행", lambda s: str(s['baseline_surrogate_pass_zero']))
    L += ["", "에이전트 종료 상태:", ""]
    for c in (1, 2):
        if c in S:
            L.append(f"- 조건 {c}: " + ", ".join(f"{k} {v}회" for k, v in S[c]["agent_status"].items()))

    L += ["", "## 실행별 결과", "",
          "| 시작 | 조건 | 시작 설계의 못 쏘는 타점 | 에이전트 | simulate 수 | 거리 | 기준선 | 확인한 후보 수 | 거리 | 제약 해석 일치 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        a, b = r["agent"], r["baseline"]
        L.append(f"| {r['start']} | {r['cond']} | {r['label_unshootable']} | {a['status']} | {a['sim_calls']} | {fmt(a['distance'] if a['success'] else None, 3)} | "
                 f"{'성공' if b['success'] else '실패'} | {b['sim_calls']} | {fmt(b['distance'], 3)} | {'o' if a['parse_match'] else 'x'} |")

    ok = [r for r in runs if r["agent"]["success"]]
    ex_ok = [next((r for r in ok if r["cond"] == c), None) for c in (1, 2)]
    ex_ok = [r for r in ex_ok if r]
    fails = [r for r in runs if not r["agent"]["success"] and r["agent"]["trace"]]
    L += ["", "## trace 예시", ""]
    for r in ex_ok:
        L += [f"성공 ({r['start']}, 조건 {r['cond']})", "", trace_block(r), ""]
    for r in fails[:1]:
        L += [f"실패 ({r['start']}, 조건 {r['cond']})", "", trace_block(r), ""]
    if not ex_ok:
        L += ["성공한 실행이 없다.", ""]
    L += ["<!-- LIMITS -->"]
    existing = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    if "<!-- LIMITS -->" in existing:      # 손으로 쓴 한계 절은 보존
        tail = existing.split("<!-- LIMITS -->", 1)[1]
        L[-1] = "<!-- LIMITS -->" + tail.rstrip("\n")
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


# ── 실행 ───────────────────────────────────────────────────────────────────
def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=10000)
    ap.add_argument("--limit", type=int, default=None, help="뽑은 시험 설계 중 앞에서 이만큼만 쓴다 (시간이 모자랄 때)")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if a.report_only:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        data["summary"] = summarize(data["runs"])
        OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        write_report(data)
        return

    RESULTS.mkdir(exist_ok=True)
    starts = pick_starts(a.n, a.seed)
    if a.limit is not None:
        starts = starts[:2 + a.limit]
    state = json.loads(PARTIAL.read_text(encoding="utf-8")) if PARTIAL.exists() else {"baseline": {}, "runs": {}}
    if state.get("cfg") not in (None, [a.n, a.seed, a.candidates, a.limit]):
        sys.exit("results/agent_eval_partial.json 이 다른 설정으로 만들어졌다. 지우고 다시 시작한다.")
    state["cfg"] = [a.n, a.seed, a.candidates, a.limit]

    def save():
        PARTIAL.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")

    idx = {s["name"]: i for i, s in enumerate(starts)}
    # 1) 기준선 조건 1
    t1 = [(f"{s['name']}|1", s["design"], {}, a.seed * 1000 + idx[s["name"]], a.candidates) for s in starts]
    run_baselines(t1, a.workers, state["baseline"], lambda r: (state["baseline"].__setitem__(r["key"], r), save()))
    # 2) 조건 2: 기준선이 가장 많이 바꾼 변수를 고정하는 요청
    intended = {}
    for s in starts:
        b1 = state["baseline"][f"{s['name']}|1"]
        var = b1["most_changed"] or "member_first_spot"
        intended[s["name"]] = var
    t2 = [(f"{s['name']}|2", s["design"], {intended[s["name"]]: {"fixed": s["design"][intended[s["name"]]]}},
           a.seed * 1000 + 500 + idx[s["name"]], a.candidates) for s in starts]
    run_baselines(t2, a.workers, state["baseline"], lambda r: (state["baseline"].__setitem__(r["key"], r), save()))

    # 3) 에이전트 (순서대로 한 번에 하나, Ollama 는 요청을 줄 세운다)
    be = RealBackend()
    for s in starts:
        for cond in (1, 2):
            key = f"{s['name']}|{cond}"
            if key in state["runs"]:
                continue
            var = intended[s["name"]]
            if cond == 1:
                request = REQ_FREE[idx[s["name"]] % 3]
                want = {}
            else:
                request = REQ_FIX[idx[s["name"]] % 3].format(label=LABEL[var])
                want = {var: {"fixed": round(s["design"][var], 3)}}
            t = time.time()
            r = DesignAgent(be).run(s["design"], request)
            got = r["constraints"]
            r["parse_match"] = got == want
            r["intended_var"] = var if cond == 2 else None
            fin_ok = bool(r["success"]) and (cond == 1 or var not in (r["changes"] or {}))
            r["success_intended"] = fin_ok
            r["success_parsed"] = r["success"]
            r["success"] = fin_ok
            r["wall_s"] = round(time.time() - t, 1)
            state["runs"][key] = r
            save()
            print(f"[agent] {key} {r['status']} 시뮬 {r['sim_calls']}회 도구 {r['tool_calls']}회 {r['wall_s']} s "
                  f"제약일치={r['parse_match']} ({len(state['runs'])}/{2 * len(starts)})", flush=True)

    runs = []
    for s in starts:
        for cond in (1, 2):
            key = f"{s['name']}|{cond}"
            r, b = state["runs"][key], state["baseline"][key]
            runs.append({"start": s["name"], "cond": cond, "label_unshootable": s["label_unshootable"],
                         "start_design": {k: round(s["design"][k], 1) for k in VARS},
                         "intended_var": intended[s["name"]] if cond == 2 else None, "agent": r, "baseline": b})
    data = {"env": env_info(),
            "config": {"seed": a.seed, "n_test": len(starts) - 2, "n_drawn": a.n, "n_starts": len(starts), "candidates": a.candidates,
                       "workers": a.workers, "n_eligible": _eligible_count()},
            "runs": runs, "summary": summarize(runs)}
    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    write_report(data)
    print(json.dumps(data["summary"], ensure_ascii=False, indent=1))


def _eligible_count():
    arr = np.load(ROOT / "data" / "gen" / "labels_1.npz")["arr"]
    ok = {}
    for d, i, _, f, _ in arr:
        k = (int(d), int(i))
        ok[k] = ok.get(k, False) or bool(f)
    return len({d for (d, _), v in ok.items() if not v})


if __name__ == "__main__":
    main()

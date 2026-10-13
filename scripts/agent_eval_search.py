"""search 도구를 넣은 에이전트 평가 — scripts/agent_eval.py 의 이전 실행(results/agent_eval.json)과 같은 시작 설계 · 같은 요청 · 같은 고정 변수.

기준선도 같이 보강한다: 이전 기준선(전역 후보 10,000개)이 못 찾은 경우를 공정하게 비교하려고, 기준선에도 search 와 같은 국소 탐색
(search_candidates, k=20)을 붙인다. 이전 기준선 결과에 국소 탐색 후보를 더한 합집합으로 계산한다
(시뮬레이터 확인 수 = 이전 확인 수 + 국소 후보 확인 수, 실제 합집합 순서로 돌렸을 때보다 같거나 많다).

실행: python scripts/agent_eval_search.py [--starts 12] [--tag smoke]    (--tag 가 있으면 문서를 건드리지 않는다)
"""
import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import agent  # noqa: E402
import agent_eval as ev  # noqa: E402
from agent import RANGE, VARS, Constraints, DesignAgent, RealBackend, norm_dist, search_candidates  # noqa: E402

PREV = ROOT / "results" / "agent_eval.json"
DOC = ROOT / "docs" / "agent_eval.md"
M0, M1 = "<!-- SEARCH-RUN-START -->", "<!-- SEARCH-RUN-END -->"
K_BASE = 20


def local_baseline_task(a):
    """이전 기준선 + 국소 탐색 후보. (작업자 프로세스)"""
    key, start, spec, seed, old = a
    t_all = time.time()
    be = RealBackend()
    cons = Constraints(spec)
    base, _ = cons.clamp(start)
    t = time.time()
    top, scanned = search_candidates(lambda ds: [be._count_local(d) for d in ds], start, base, cons, K_BASE, seed)
    scan_s = time.time() - t
    old_dist = old["distance"] if old["success"] else float("inf")
    todo = [(dist, c) for dist, n, c in top if n == 0 and dist < old_dist][:K_BASE]
    tried, found, sim_s = [], None, 0.0
    for dist, c in todo:
        t = time.time()
        n = len(be.simulate(c))
        sim_s += time.time() - t
        tried.append([round(dist, 4), n])
        if n == 0:
            found = (dist, c)
            break
    success = found is not None or old["success"]
    dist = found[0] if found else (old["distance"] if old["success"] else None)
    return {"key": key, "success": success, "distance": round(dist, 4) if dist is not None else None,
            "sim_calls": old["sim_calls"] + len(tried), "wall_s": round(old["wall_s"] + time.time() - t_all, 1),
            "old_success": old["success"], "old_sim_calls": old["sim_calls"], "local_sim_calls": len(tried),
            "local_scan_s": round(scan_s, 1), "local_sim_s": round(sim_s, 1), "local_scanned": scanned,
            "local_zero_candidates": sum(1 for _, n, _ in top if n == 0), "from_local": found is not None,
            "final": {k: round(found[1][k], 1) for k in VARS} if found else None}


# ── 집계 ──────────────────────────────────────────────────────────────────
def stat(entries):
    """entries: success · sim_calls · wall_s · distance 를 가진 dict 목록."""
    ok = [e for e in entries if e["success"]]
    return {"n": len(entries), "success": len(ok),
            "sim_mean": ev.mean([e["sim_calls"] for e in entries]), "sim_median": ev.med([e["sim_calls"] for e in entries]),
            "wall_mean": ev.mean([e["wall_s"] for e in entries]), "wall_median": ev.med([e["wall_s"] for e in entries]),
            "dist_median": ev.med([e["distance"] for e in ok]), "dist_n": len(ok)}


def build_summary(runs):
    out = {}
    for cond in (1, 2):
        rs = [r for r in runs if r["cond"] == cond]
        pairs = [(r["agent"]["distance"], r["baseline_local"]["distance"]) for r in rs
                 if r["agent"]["success"] and r["baseline_local"]["success"]]
        ag = [r["agent"] for r in rs]
        out[cond] = {
            "agent_new": stat(ag), "agent_prev": stat([r["agent_prev"] for r in rs]),
            "baseline_prev": stat([r["baseline_prev"] for r in rs]), "baseline_local": stat([r["baseline_local"] for r in rs]),
            "pair_n": len(pairs), "pair_agent_dist_median": ev.med([x for x, _ in pairs]),
            "pair_baseline_dist_median": ev.med([y for _, y in pairs]),
            "parse_match": sum(a["parse_match"] for a in ag), "parse_match_prev": sum(r["agent_prev"]["parse_match"] for r in rs),
            "tool_calls_mean": ev.mean([a["tool_calls"] for a in ag]),
            "tool_calls_mean_prev": ev.mean([r["agent_prev"]["tool_calls"] for r in rs]),
            "format_calls": sum(a.get("format_fallbacks", 0) for a in ag), "tool_calls_total": sum(a["tool_calls"] for a in ag),
            "status": {s: sum(1 for a in ag if a["status"] == s) for s in sorted({a["status"] for a in ag})},
            "baseline_local_found": sum(1 for r in rs if r["baseline_local"]["from_local"]),
        }
    return out


def f(x, d=1):
    return ev.fmt(x, d)


def section(data):
    S, env = {int(k): v for k, v in data["summary"].items()}, data["env"]
    L = [M0, "", "## 이번 실행: search 도구 추가", "",
         "에이전트에 `search(design, k)` 도구를 더한 뒤 같은 시작 설계 12개 × 조건 2 를 다시 돌렸다. 시작 설계, 요청 문장, 조건 2 에서 고정한 변수는 "
         "아래 \"이전\" 실행과 같다. `search` 는 제약을 만족하는 후보를 시작 설계 주변에서 반경을 넓혀 가며(범위 폭의 3 % 에서 40 %, 반경마다 300개) "
         "대리 모델로 훑고, 예측 0개인 후보가 k개 모이면 멈춘다. 모자라면 `explore.py` 와 같은 전역 후보 1,500개를 더한다. "
         "예측 0개인 상위 후보는 필요 없는 변경을 하나씩 시작값으로 되돌려 본다. 결과는 시작 설계에 가까운 순서이고 대리 모델 예측이 캐시된다.",
         "",
         "기준선도 보강했다. 이전 기준선(전역 후보 10,000개)이 못 찾는 경우가 많아서, 같은 국소 탐색(`search_candidates`, k=20)에서 나온 후보를 합쳤다. "
         "시뮬레이터 확인 수는 이전 확인 수 + 국소 후보 확인 수이며, 실제 합집합 순서로 돌렸을 때보다 같거나 많다. "
         "걸린 시간은 이전 기준선 시간 + 국소 탐색 시간이다.", "",
         f"- 실행 환경: {env['date']} · 모델 {env['model']} · 이번 실행 전체 걸린 시간 {data['total_wall_s']:.0f} s "
         f"(기준선 국소 탐색 {data['phase_wall_s']['baseline']:.0f} s + 에이전트 {data['phase_wall_s']['agent']:.0f} s)",
         "- 다른 GPU · Ollama 작업 없이 이 평가만 돌렸다.", ""]
    names = [("에이전트 (search 추가)", "agent_new"), ("에이전트 이전 (search 없음)", "agent_prev"),
             ("기준선 이전", "baseline_prev"), ("기준선 + 국소 탐색", "baseline_local")]
    for cond, title in ((1, "조건 1 (제약 없음)"), (2, "조건 2 (변수 하나 고정)")):
        s = S[cond]
        L += [f"### {title}", "", "| 항목 | " + " | ".join(n for n, _ in names) + " |", "|---|---|---|---|---|"]
        rows = [("성공", lambda x: f"{x['success']}/{x['n']} ({100 * x['success'] / x['n']:.0f} %)"),
                ("시뮬레이터 호출 (평균 / 중앙값)", lambda x: f"{f(x['sim_mean'])} / {f(x['sim_median'], 0)}"),
                ("걸린 시간 (평균 / 중앙값, 초)", lambda x: f"{f(x['wall_mean'], 0)} / {f(x['wall_median'], 0)}"),
                ("변경 거리 (성공한 실행의 중앙값)", lambda x: f"{f(x['dist_median'], 3)} (n={x['dist_n']})")]
        for label, fn in rows:
            L.append(f"| {label} | " + " | ".join(fn(s[k]) for _, k in names) + " |")
        L += ["", f"- 새 에이전트와 새 기준선이 모두 성공한 {s['pair_n']}건의 변경 거리 중앙값: 에이전트 {f(s['pair_agent_dist_median'], 3)}, "
              f"기준선 {f(s['pair_baseline_dist_median'], 3)}",
              f"- 제약 해석 일치: 이번 {s['parse_match']}/{s['agent_new']['n']}, 이전 {s['parse_match_prev']}/{s['agent_new']['n']}",
              f"- 에이전트 도구 호출 평균: 이번 {f(s['tool_calls_mean'])}, 이전 {f(s['tool_calls_mean_prev'])}. "
              f"그중 JSON 형식 강제 {s['format_calls']}/{s['tool_calls_total']}",
              "- 에이전트 종료 상태: " + ", ".join(f"{k} {v}회" for k, v in s["status"].items()),
              f"- 보강 기준선이 국소 탐색 후보로 새로 찾았거나 더 가까운 설계로 바꾼 실행: {s['baseline_local_found']}/{s['agent_new']['n']}", ""]
    L += ["### 실행별 결과", "",
          "| 시작 | 조건 | 에이전트 | simulate 수 | 도구 호출 | 거리 | 기준선+국소 | 확인 수 | 거리 | 제약 해석 일치 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in data["runs"]:
        a, b = r["agent"], r["baseline_local"]
        L.append(f"| {r['start']} | {r['cond']} | {a['status']} | {a['sim_calls']} | {a['tool_calls']} | "
                 f"{f(a['distance'] if a['success'] else None, 3)} | {'성공' if b['success'] else '실패'} | {b['sim_calls']} | "
                 f"{f(b['distance'], 3)} | {'o' if a['parse_match'] else 'x'} |")
    ok = [r for r in data["runs"] if r["agent"]["success"]]
    fails = [r for r in data["runs"] if not r["agent"]["success"] and r["agent"]["trace"]]
    L += ["", "### trace 예시", ""]
    ex = [next((r for r in ok if r["cond"] == c), None) for c in (1, 2)]
    for r in [x for x in ex if x]:
        L += [f"성공 ({r['start']}, 조건 {r['cond']})", "", ev.trace_block({"start": r["start"], "cond": r["cond"], "agent": r["agent"]}), ""]
    for r in fails[:1]:
        L += [f"실패 ({r['start']}, 조건 {r['cond']})", "", ev.trace_block({"start": r["start"], "cond": r["cond"], "agent": r["agent"]}), ""]
    L += [M1]
    return "\n".join(L)


def splice(text):
    doc = DOC.read_text(encoding="utf-8")
    if M0 in doc:
        doc = doc.split(M0)[0] + text + doc.split(M1, 1)[1]
    else:
        anchor = "## 환경"
        doc = doc.replace(anchor, text + "\n\n" + anchor, 1)
    DOC.write_text(doc, encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    out_json = ROOT / "results" / f"agent_eval_search{'_' + a.tag if a.tag else ''}.json"
    if a.report_only:
        data = json.loads(out_json.read_text(encoding="utf-8"))
        data["summary"] = build_summary(data["runs"])
        out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        splice(section(data))
        return
    t_all = time.time()
    prev = json.loads(PREV.read_text(encoding="utf-8"))
    prev_by = {(r["start"], r["cond"]): r for r in prev["runs"]}
    starts = ev.pick_starts(prev["config"]["n_drawn"], prev["config"]["seed"])[:a.starts]
    seed = prev["config"]["seed"]

    # 1) 보강 기준선
    t0 = time.time()
    tasks = []
    for i, s in enumerate(starts):
        for cond in (1, 2):
            p = prev_by[(s["name"], cond)]
            var = p["intended_var"]
            spec = {} if cond == 1 else {var: {"fixed": s["design"][var]}}
            tasks.append((f"{s['name']}|{cond}", s["design"], spec, seed * 1000 + (0 if cond == 1 else 500) + i, p["baseline"]))
    base = {}
    with Pool(min(a.workers, len(tasks))) as pool:
        for r in pool.imap_unordered(local_baseline_task, tasks):
            base[r["key"]] = r
            print(f"[baseline+local] {r['key']} 성공={r['success']} (이전 {r['old_success']}) 국소 확인 {r['local_sim_calls']}회 "
                  f"({len(base)}/{len(tasks)})", flush=True)
    t_base = time.time() - t0

    # 2) 에이전트
    t0 = time.time()
    be = RealBackend()
    runs = []
    for s in starts:
        for cond in (1, 2):
            p = prev_by[(s["name"], cond)]
            var = p["intended_var"]
            request = p["agent"]["request"]
            want = {} if cond == 1 else {var: {"fixed": round(s["design"][var], 3)}}
            t = time.time()
            r = DesignAgent(be).run(s["design"], request)
            r["parse_match"] = r["constraints"] == want
            r["success"] = bool(r["success"]) and (cond == 1 or var not in (r["changes"] or {}))
            r["wall_s"] = round(time.time() - t, 1)
            runs.append({"start": s["name"], "cond": cond, "intended_var": var, "agent": r,
                         "agent_prev": p["agent"], "baseline_prev": p["baseline"], "baseline_local": base[f"{s['name']}|{cond}"]})
            print(f"[agent] {s['name']}|{cond} {r['status']} simulate {r['sim_calls']}회 도구 {r['tool_calls']}회 {r['wall_s']} s "
                  f"제약일치={r['parse_match']} ({len(runs)}/{2 * len(starts)})", flush=True)
            out_json.with_suffix(".partial").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    be.close()
    t_agent = time.time() - t0
    env = ev.env_info()
    data = {"env": env, "total_wall_s": time.time() - t_all, "phase_wall_s": {"baseline": t_base, "agent": t_agent},
            "config": {"starts": [s["name"] for s in starts]}, "runs": runs}
    data["summary"] = build_summary(runs)
    out_json.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    if not a.tag:
        splice(section(data))
    print(json.dumps(data["summary"], ensure_ascii=False, indent=1))
    print(f"전체 {time.time() - t_all:.0f} s (기준선 {t_base:.0f} s, 에이전트 {t_agent:.0f} s)")


if __name__ == "__main__":
    main()

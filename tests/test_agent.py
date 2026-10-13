"""설계 수정 에이전트 — 가짜 LLM 과 가짜 시뮬레이터로 도구 규칙·한도·제약 검증을 고정한다. Ollama 는 필요 없다."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import agent  # noqa: E402
from agent import Constraints, ConstraintError, DesignAgent, parse_constraints  # noqa: E402

START = dict(agent.DESIGN_A)


class FakeBackend:
    """플랜지 폭이 20 mm 미만이면 타점 2개를 못 쏜다. 시뮬레이터 호출 횟수를 센다."""
    threshold = 0.05

    def __init__(self):
        self.sim_calls = 0

    def predict(self, d):
        bad = d["flange_width"] < 20
        return {"S01": 0.01 if bad else 0.99, "M00": 0.02 if bad else 0.99}

    def simulate(self, d):
        self.sim_calls += 1
        if d["flange_width"] < 20:
            return {"S01": {"group": "sill", "reason": "gun_collision", "reasons": {"gun_collision": 5}},
                    "M00": {"group": "member", "reason": "arm_collision", "reasons": {"arm_collision": 5}}}
        return {}


def call(name, **args):
    return {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": name, "arguments": args}}]}


class ScriptedLLM:
    """미리 정한 응답을 차례로 돌려준다. 다 쓰면 마지막 응답을 반복."""

    def __init__(self, *msgs):
        self.msgs, self.n = list(msgs), 0

    def __call__(self, messages, tools=None, fmt=None):
        m = self.msgs[min(self.n, len(self.msgs) - 1)]
        self.n += 1
        return m


def agent_with(llm, cons=None, backend=None):
    a = DesignAgent(backend or FakeBackend(), llm)
    a._reset(START, cons or Constraints())
    return a


# ── submit ────────────────────────────────────────────────────────────────
def test_submit_rejects_unsimulated_design():
    a = agent_with(None)
    r = a.call("submit", {"design": {"flange_width": 25}, "rationale": "x"})
    assert r["accepted"] is False and "simulate" in r["problems"][0]
    # 다른 설계로 simulate 했어도 마찬가지
    a.call("simulate", {"design": {"flange_width": 30}})
    r = a.call("submit", {"design": {"flange_width": 25}, "rationale": "x"})
    assert r["accepted"] is False and "다름" in r["problems"][0]


def test_submit_rejects_design_with_unshootable_spots():
    a = agent_with(None)
    a.call("simulate", {"design": {"flange_width": 15}})
    r = a.call("submit", {"design": {"flange_width": 15}, "rationale": "x"})
    assert r["accepted"] is False and "2개" in r["problems"][0]


def test_submit_rejects_constraint_violation():
    a = agent_with(None, Constraints({"flange_width": {"fixed": 15.0}}))
    # 제약을 어긴 설계가 시뮬레이션 기록과 같다고 해도 (상태를 직접 심어 둔다) 제출은 거부된다
    bad = dict(START, flange_width=30.0)
    a.last_sim = {"design": bad, "n": 0, "spots": {}}
    r = a.call("submit", {"design": {"flange_width": 30}, "rationale": "x"})
    assert r["accepted"] is False
    assert any("고정" in p for p in r["problems"])
    assert a.accepted is None


def test_submit_accepts_simulated_clean_design():
    a = agent_with(None)
    a.call("simulate", {"design": {"flange_width": 30}})
    r = a.call("submit", {"design": {"flange_width": 30.04}, "rationale": "플랜지 확대"})
    assert r["accepted"] is True and a.accepted["flange_width"] == 30


# ── 범위 자르기 ───────────────────────────────────────────────────────────
def test_tools_clamp_to_constraints_and_ranges():
    a = agent_with(None, Constraints({"flange_width": {"fixed": 15.0}, "member_first_spot": {"max": 60.0}}))
    r = a.call("predict", {"design": {"flange_width": 40, "member_first_spot": 120, "sill_top": 9999}})
    assert r["clamped"]["flange_width"] == [40.0, 15.0]
    assert r["clamped"]["member_first_spot"] == [120.0, 60.0]
    assert r["clamped"]["sill_top"][1] == 650.0
    r = a.call("sensitivity", {"design": {}, "var": "flange_width"})
    assert "고정" in r["error"]
    r = a.call("sensitivity", {"design": {}, "var": "member_first_spot"})
    assert len(r["sweep"]) == 9 and r["sweep"][-1][0] == 60.0


# ── 한도 ──────────────────────────────────────────────────────────────────
def test_simulate_budget_is_four():
    a = agent_with(None)
    for i in range(4):
        assert "error" not in a.call("simulate", {"design": {"flange_width": 10 + i}})
    r = a.call("simulate", {"design": {"flange_width": 30}})
    assert "한도" in r["error"] and a.backend.sim_calls == 4


def test_total_call_budget_returns_failure_with_best_simulated_design():
    # 시작 설계를 한 번 simulate 한 뒤 predict 만 계속 부르는 LLM
    llm = ScriptedLLM(call("simulate", design={"flange_width": 12}), call("predict", design={"flange_width": 14}))
    be = FakeBackend()
    r = DesignAgent(be, llm).run(START, "x", Constraints())
    assert r["status"] == "실패: 한도 초과"
    assert r["tool_calls"] == agent.MAX_CALLS == 12 and r["success"] is False
    assert r["best_simulated"]["unshootable"] == 2 and be.sim_calls == 1
    assert len(r["trace"]) == 12 and {"tool", "args", "summary", "ms"} <= set(r["trace"][0])


def test_run_success_path_and_trace():
    llm = ScriptedLLM(
        call("simulate", design={}),
        call("sensitivity", design={}, var="flange_width"),
        call("simulate", design={"flange_width": 25}),
        call("submit", design={"flange_width": 25}, rationale="플랜지 폭 15 -> 25"))
    be = FakeBackend()
    r = DesignAgent(be, llm).run(START, "x", Constraints())
    assert r["status"] == "성공" and r["success"] and r["changes"] == {"flange_width": [15.0, 25.0]}
    assert r["sim_calls"] == 2 and be.sim_calls == 2 and r["tool_calls"] == 4


def test_run_stops_when_llm_never_calls_tools():
    r = DesignAgent(FakeBackend(), ScriptedLLM({"role": "assistant", "content": "음"})).run(START, "x", Constraints())
    assert r["status"] == "실패: 제출 없이 종료"


def test_run_parses_request_through_llm():
    def llm(messages, tools=None, fmt=None):
        if tools is None:
            return {"role": "assistant", "content": json.dumps(
                {"constraints": [{"var": "flange_width", "type": "fixed", "value": None}]})}
        return call("simulate", design={})
    r = DesignAgent(FakeBackend(), llm).run(START, "플랜지 폭은 그대로", None)
    assert r["constraints"] == {"flange_width": {"fixed": 15.0}}


# ── 제약 해석 ─────────────────────────────────────────────────────────────
def test_parser_accepts_valid_constraints():
    c = parse_constraints({"constraints": [{"var": "flange_width", "type": "fixed", "value": None},
                                           {"var": "member_first_spot", "type": "max", "value": 60}]}, START)
    assert c.spec == {"flange_width": {"fixed": 15.0}, "member_first_spot": {"max": 60.0}}
    assert c.bounds("member_first_spot") == (0.0, 60.0)


@pytest.mark.parametrize("item", [
    {"var": "member_first_spot", "type": "max", "value": 500},        # 범위 150 초과
    {"var": "flange_width", "type": "min", "value": 1},               # 범위 8 미만
    {"var": "flange_width", "type": "fixed", "value": 100},
    {"var": "wheel_size", "type": "fixed", "value": None},            # 없는 변수
    {"var": "flange_width", "type": "around", "value": 20},           # 없는 종류
    {"var": "flange_width", "type": "max", "value": None},            # 값 없음
    {"var": "flange_width", "type": "max", "value": "많이"},
])
def test_parser_rejects_bad_items(item):
    with pytest.raises(ConstraintError):
        parse_constraints({"constraints": [item]}, START)


def test_parser_rejects_min_above_max():
    with pytest.raises(ConstraintError):
        parse_constraints({"constraints": [{"var": "pillar_w", "type": "min", "value": 200},
                                           {"var": "pillar_w", "type": "max", "value": 150}]}, START)


def test_parse_request_retries_once_then_fails():
    bad = {"role": "assistant", "content": json.dumps({"constraints": [{"var": "pillar_w", "type": "max", "value": 999}]})}
    llm = ScriptedLLM(bad)
    with pytest.raises(ConstraintError):
        agent.parse_request("B필러 폭 999 이하", START, llm)
    assert llm.n == 2
    r = DesignAgent(FakeBackend(), llm).run(START, "B필러 폭 999 이하")
    assert r["status"] == "실패: 제약 해석 오류" and r["constraints"] is None


def test_empty_response_falls_back_to_forced_json_action():
    """도구 호출 없이 빈 응답이 오면 JSON 형식 강제로 다음 행동을 다시 받는다. 형식 강제 응답도 같은 도구 규칙을 거친다."""
    actions = iter([{"tool": "simulate", "design": {"flange_width": 25}},
                    {"tool": "submit", "design": {"flange_width": 25}, "rationale": "x"}])

    def llm(messages, tools=None, fmt=None):
        if fmt is agent.ACTION_SCHEMA:
            return {"role": "assistant", "content": json.dumps(next(actions))}
        return {"role": "assistant", "content": ""}
    r = DesignAgent(FakeBackend(), llm).run(START, "x", Constraints())
    assert r["status"] == "성공" and r["tool_calls"] == 2 and r["format_fallbacks"] == 2
    assert [t["via"] for t in r["trace"]] == ["format", "format"]


def test_forced_action_cannot_bypass_submit_rules():
    def llm(messages, tools=None, fmt=None):
        if fmt is agent.ACTION_SCHEMA:
            return {"role": "assistant", "content": json.dumps(
                {"tool": "submit", "design": {"flange_width": 30}, "rationale": "바로 제출"})}
        return {"role": "assistant", "content": ""}
    r = DesignAgent(FakeBackend(), llm).run(START, "x", Constraints())
    assert r["success"] is False and r["trace"][0]["result"]["accepted"] is False


def test_search_returns_constraint_satisfying_candidates_nearest_first():
    a = agent_with(None, Constraints({"member_first_spot": {"max": 60.0}}))
    r = a.call("search", {"k": 4})
    assert "error" not in r and len(r["candidates"]) == 4
    dists = [c["dist"] for c in r["candidates"]]
    assert dists == sorted(dists)
    assert all(c["predicted_unshootable"] == 0 for c in r["candidates"])
    for c in r["candidates"]:
        d = dict(START, **c["design"])
        assert not a.cons.violations(d)
        assert c["design"].get("flange_width", 0) >= 20      # 가짜 시뮬레이터에서 고쳐지는 변수


def test_search_keeps_fixed_variable_and_reports_when_nothing_passes():
    a = agent_with(None, Constraints({"flange_width": {"fixed": 15.0}}))
    r = a.call("search", {})
    assert r["candidates"] and all(c["predicted_unshootable"] > 0 for c in r["candidates"])
    assert all("flange_width" not in c["design"] for c in r["candidates"])


def test_search_then_simulate_then_submit_flow():
    llm = ScriptedLLM(call("search", k=3))
    a = DesignAgent(FakeBackend(), llm)
    a._reset(START, Constraints())
    top = a.call("search", {"k": 3})["candidates"][0]["design"]
    a.call("simulate", {"design": top})
    assert a.call("submit", {"design": top, "rationale": "x"})["accepted"] is True

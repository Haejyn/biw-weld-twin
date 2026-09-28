"""README 그림 — docs/img/{ai_map,ai_eval,line,robots}.png. 원자료는 results/*.json (src 파이프라인을 먼저 돌린다).
Pretendard 가 docs/fonts 에 없으면 받는다. 사용: python scripts/make_readme_figures.py"""
import json
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"
OUT = ROOT / "docs" / "img"
OUT.mkdir(parents=True, exist_ok=True)
BLACK, GRAY, LIGHT, BLUE, RED = "#111111", "#6b6b6b", "#d9d9d9", "#1f4e79", "#b03a2e"
PRETENDARD_URL = "https://github.com/orioncactus/pretendard/releases/download/v1.3.9/Pretendard-1.3.9.zip"


def setup_style():
    fonts = ROOT / "docs" / "fonts"
    files = list(fonts.glob("Pretendard-*.ttf")) if fonts.exists() else []
    if not files:
        fonts.mkdir(parents=True, exist_ok=True)
        archive = fonts / "Pretendard.zip"
        urllib.request.urlretrieve(PRETENDARD_URL, archive)
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if name.startswith("public/static/alternative/") and name.endswith(".ttf"):
                    (fonts / Path(name).name).write_bytes(z.read(name))
        files = list(fonts.glob("Pretendard-*.ttf"))
    for f in files:
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "Pretendard", "font.size": 9, "axes.unicode_minus": False,
        "axes.linewidth": 0.8, "axes.edgecolor": BLACK, "axes.labelcolor": BLACK,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.color": BLACK, "ytick.color": BLACK,
        "legend.frameon": False, "legend.fontsize": 8, "figure.dpi": 200, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05, "lines.linewidth": 1.0,
    })


def panel(ax, letter):
    ax.text(-0.02, 1.02, f"({letter})", transform=ax.transAxes, ha="right", va="bottom", fontsize=10,
            fontweight="semibold")


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def ai_map_fig():
    ex = load("explore.json")
    m = ex["map"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})
    fw, fs = np.array(m["fw"]) * 1000, np.array(m["fs"]) * 1000
    cmap = ListedColormap(["#ffffff", "#e8c4bd", RED])
    a.pcolormesh(fw, fs, np.clip(np.array(m["ai_grid"]), 0, 2), cmap=cmap, vmin=0, vmax=2, shading="nearest")
    sim = np.array(m["sim_grid"])
    for j, y in enumerate(np.array(m["cs"]) * 1000):
        for i, x in enumerate(np.array(m["cw"]) * 1000):
            same = int(sim[j, i]) == int(np.array(m["ai_on_sim_points"])[j, i])
            a.plot(x, y, "o" if same else "x", ms=3.2, mfc="white", mec=BLACK, mew=0.7)
    a.plot(30, 30, "*", ms=9, color=BLUE, mec="white", mew=0.6)
    a.text(33, 22, "기준 A", fontsize=7.5, color="white")
    a.plot(80, 50, "^", ms=6, color=BLUE, mec="white", mew=0.6)
    a.text(83, 53, "C안", fontsize=7.5, color=BLUE)
    a.set_xlabel("크로스멤버 첫 타점 거리 (mm)")
    a.set_ylabel("실 타점–B필러 간격 (mm)")
    panel(a, "a")
    styles = {"member_first_spot": (BLUE, "첫 타점 거리 (간격 60 mm)"), "sill_pillar_gap": (GRAY, "실–B필러 간격 (첫 타점 100 mm)")}
    for var, probe in ex["probes"].items():
        c, lab = styles[var]
        x = [p["mm"] for p in probe]
        b.step(x, [p["ai"] for p in probe], where="mid", color=c, label=f"AI · {lab}")
        b.plot(x, [p["sim"] for p in probe], "o", ms=3, mfc="white", mec=c, mew=0.7)
        for p in probe:
            if p["ai"] != p["sim"]:
                b.annotate("", (p["mm"], p["ai"] - 0.08), (p["mm"], p["sim"] + 0.1),
                           arrowprops=dict(arrowstyle="->", color=RED, lw=0.8))
    b.set_xlabel("변수 값 (mm) · 점 = 시뮬레이터")
    b.set_ylabel("못 쏘는 타점 (개)")
    b.set_yticks([0, 1, 2])
    b.set_ylim(-0.3, 2.6)
    b.legend(loc="upper right", fontsize=7)
    panel(b, "b")
    fig.tight_layout()
    fig.savefig(OUT / "ai_map.png")


KO_FEAT = {"is_member": "크로스멤버 타점", "is_sill": "실 타점", "is_pillar": "B필러 타점",
           "clear_arm_pillar": "팔 경로–B필러 여유", "clear_arm_sill": "팔 경로–실 여유",
           "clear_gun_sill": "건–실 여유", "clear_gun_member": "건–크로스멤버 여유",
           "dist_xy": "로봇 수평 거리", "dist": "로봇 거리", "rel_x": "로봇 기준 x", "rel_y": "로봇 기준 y",
           "rel_z": "로봇 기준 z", "dx_pillar": "B필러와 x 거리", "dx_pillar_edge": "B필러 가장자리 거리",
           "spot_y": "타점 y", "spot_x": "타점 x", "spot_z": "타점 z", "robot_x": "로봇 x", "robot_y": "로봇 y",
           "robot_z": "로봇 높이", "robot_overhead": "천장 로봇", "normal_toward_robot": "법선–로봇 방향",
           "sill_above_spot": "실 윗면 높이 차", "dy_sill_face": "실 면과 y 거리",
           "member_first_spot": "첫 타점 거리", "sill_pillar_gap": "실–B필러 간격", "flange_width": "플랜지 폭",
           "n_z": "법선 z", "n_y": "법선 y",
           "clear_gun_tilt0": "건–판재 여유 (0°)", "clear_gun_best": "건 전극·암–판재 여유", "clear_body_best": "건 몸체–판재 여유"}


def ai_eval_fig():
    s = load("surrogate.json")
    keys = [("unshootable_recall", "재현율"), ("unshootable_precision", "정밀도"),
            ("design_flag_acc", "설계 판정"), ("design_count_exact", "개수 정확")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw={"width_ratios": [1, 1.1]})
    xs = np.arange(len(keys))
    for off, tag, color, label in [(-0.18, "raw", LIGHT, "좌표만"), (0.18, "engineered", BLUE, "+ 공학 특징")]:
        vals = [round(s[tag][k] * 100 + 1e-9, 1) for k, _ in keys]
        a.bar(xs + off, vals, 0.34, color=color, edgecolor=BLACK, linewidth=0.6, label=label)
        for x, v in zip(xs + off, vals):
            a.text(x, v + 0.6, f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)
    a.set_xticks(xs, [k for _, k in keys])
    a.set_ylim(40, 125)
    a.set_ylabel(f"시험 설계 {s['engineered']['test_designs']}개 (%)")
    a.legend(loc="upper left", ncol=2)
    panel(a, "a")
    ex = s["explain_infeasible"][:8][::-1]
    b.barh([KO_FEAT.get(n, n) for n, _ in ex], [v for _, v in ex], color=BLUE, edgecolor=BLACK, linewidth=0.6,
           height=0.6)
    b.set_xlabel("'못 쏜다' 쪽 평균 SHAP 기여")
    b.tick_params(axis="y", which="both", right=False, labelsize=7.5)
    panel(b, "b")
    fig.tight_layout()
    fig.savefig(OUT / "ai_eval.png")


def line_fig():
    r = load("stage2.json")["alternatives"]
    keys = ["A_혼류_그대로_random", "B_재밸런싱_증설_random", "B_재밸런싱_증설_even", "C_신규라인"]
    labels = ["A 그대로", "B 재밸런싱\n무작위 순서", "B 재밸런싱\n고르게 섞기", "C 신규 라인"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.6))
    colors = [LIGHT, LIGHT, BLUE, LIGHT]
    for ax, key, ylab, fmt in [(a, "line_stops_per_100", "라인 정지 (100대당)", "{:.1f}"),
                               (b, "cost5y", "5년 비용 (억 원)", "{:,.0f}")]:
        vals = [r[k][key] for k in keys]
        ax.bar(range(4), vals, 0.6, color=colors, edgecolor=BLACK, linewidth=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.02, fmt.format(v), ha="center", va="bottom", fontsize=7)
        ax.set_xticks(range(4), labels, fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, max(vals) * 1.18)
    panel(a, "a")
    panel(b, "b")
    fig.tight_layout()
    fig.savefig(OUT / "line.png")


def robots_fig():
    sw = load("stage1_jph_sweep.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.5))
    styles = {"A": (BLACK, "o", "A 기준"), "B": (GRAY, "s", "B 플랜지 확대"), "C": (BLUE, "^", "C 개선")}
    for k, (c, mk, lab) in styles.items():
        rows = [x for x in sw if x["design"].startswith(k)]
        jph = [x["jph"] for x in rows]
        off = {"A": -0.6, "B": 0, "C": 0.6}[k]
        ok = [x for x in rows if x["robots_used"] is not None]
        a.plot(np.array([x["jph"] for x in ok]) + off, [x["robots_used"] for x in ok], mk, ms=4, color=c,
               mfc="white", mew=0.9, label=lab)
        ok = [x for x in rows if x["max_robot_time_s"] is not None]
        b.plot([x["jph"] for x in ok], [x["max_robot_time_s"] for x in ok], mk + "-", ms=3.5, color=c, mfc="white",
               mew=0.9, label=lab)
    jph = sorted({x["jph"] for x in sw})
    b.plot(jph, [3600 / j - 15 for j in jph], "--", color=RED, lw=0.9, label="예산 (택트 − 15 s)")
    for k, (c, mk, lab) in styles.items():
        for x in sw:
            if x["design"].startswith(k) and x["robots_used"] is None:
                a.plot(x["jph"] + {"A": -0.6, "B": 0, "C": 0.6}[k], 4.9, "x", ms=5, color=RED)
    a.text(70, 4.55, "후보 5곳으로 불가", ha="center", fontsize=7, color=RED)
    a.set_xlabel("목표 JPH")
    a.set_ylabel("필요 로봇 (대)")
    a.set_yticks([2, 3, 4, 5])
    a.set_ylim(1.5, 5.3)
    a.legend(loc="upper left")
    b.set_xlabel("목표 JPH")
    b.set_ylabel("가장 바쁜 로봇 사이클 (s)")
    b.legend(loc="upper right", fontsize=7)
    panel(a, "a")
    panel(b, "b")
    fig.tight_layout()
    fig.savefig(OUT / "robots.png")


if __name__ == "__main__":
    setup_style()
    ai_map_fig()
    ai_eval_fig()
    line_fig()
    robots_fig()
    print("ok", sorted(p.name for p in OUT.glob("*.png")))

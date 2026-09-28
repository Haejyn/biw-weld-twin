"""결과 그림 두 장 — results/*.json 을 읽어 results/fig_*.png 로."""
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent.parent / "results"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#b4b2a8", "#1a1a19", "#6b6a63"
for name in ("Malgun Gothic", "Pretendard"):
    if any(f.name == name for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = name
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRAY,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10})


def fig_stage1():
    rows = list(csv.DictReader(open(OUT / "stage1_spots.csv", encoding="utf-8-sig")))
    designs = ["A_기준", "C_플랜지확대+첫타점이동"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    from body import spots
    from stage1 import DESIGNS
    for ax, dn in zip(axes, designs):
        d = next(x for x in DESIGNS if x.name == dn)
        pos = {s.id: s.pos for s in spots(d) if s.group == "member"}
        mine = [r for r in rows if r["design"] == dn and r["group"] == "member"]
        for r in mine:
            y = pos[r["spot"]][1] * 1000
            if r["robot"] == "":
                ax.scatter(y, 0, s=90, marker="x", color=ORANGE, linewidths=2.5, zorder=3)
            else:
                tilt = float(r["tilt_deg"] or 0)
                ax.scatter(y, tilt, s=60, color=BLUE, edgecolor="white", linewidth=2, zorder=3)
        ax.axvspan(0, 80, color=GRAY, alpha=0.25, lw=0)
        ax.text(40, 33, "실(sill)", ha="center", color=MUTED, fontsize=9)
        n_x = sum(1 for r in mine if r["robot"] == "")
        ax.set_title(f"{dn.split('_')[0]}안 — 못 쏘는 타점 {n_x}개", loc="left", color=INK, fontsize=11)
        ax.set_xlabel("크로스멤버 타점 위치 (차 안쪽으로, mm)")
        ax.set_ylim(-5, 38)
    axes[0].set_ylabel("건 기울임 (°)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage1_member.png", dpi=180)


def fig_stage2():
    r = json.loads((OUT / "stage2.json").read_text(encoding="utf-8"))
    a = r["alternatives"]
    keys = ["A_혼류_그대로_random", "B_재밸런싱_증설_random", "B_재밸런싱_증설_even", "C_신규라인"]
    labels = ["A 그대로 혼류", "B 재밸런싱\n+1곳, 무작위 순서", "B 재밸런싱\n+1곳, 고르게 섞기", "C 신규 라인"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ax = axes[0]
    vals = [a[k]["line_stops_per_100"] for k in keys]
    bars = ax.bar(labels, vals, color=[GRAY, GRAY, BLUE, GRAY], width=0.6)
    ax.set_title("라인 정지 (100대당 횟수)", loc="left", color=INK, fontsize=11)
    for b, v, k in zip(bars, vals, keys):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}회\nJPH {a[k]['effective_jph']:.1f}",
                ha="center", color=INK, fontsize=8.5)
    ax.set_ylim(0, max(vals) * 1.3)
    ax = axes[1]
    vals = [a[k]["cost5y"] for k in keys]
    bars = ax.bar(labels, vals, color=[GRAY, GRAY, BLUE, GRAY], width=0.6)
    ax.set_title("5년 비용 (상대 단위, 스테이션 1곳 설비 = 1)", loc="left", color=INK, fontsize=11)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}", ha="center", color=INK, fontsize=9)
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_stage2_alternatives.png", dpi=180)


def fig_map():
    import numpy as np
    from matplotlib.colors import ListedColormap
    r = json.loads((OUT / "explore.json").read_text(encoding="utf-8"))["map"]
    ai = np.array(r["ai_grid"])
    fw, fs = np.array(r["fw"]) * 1000, np.array(r["fs"]) * 1000
    cmap = ListedColormap(["#1c2b3a", "#8a4a2c", "#c4562a", "#eb6834"])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.pcolormesh(fw, fs, np.clip(ai, 0, 3), cmap=cmap, vmin=0, vmax=3, shading="nearest")
    sim = np.array(r["sim_grid"])
    for j, y in enumerate(np.array(r["cs"]) * 1000):
        for i, x in enumerate(np.array(r["cw"]) * 1000):
            ok = sim[j, i] == np.array(r["ai_on_sim_points"])[j, i]
            ax.scatter(x, y, s=46, marker="o" if ok else "X", color="white" if ok else "#ffd23f",
                       edgecolor="#1a1a19", linewidth=0.8, zorder=3)
            ax.text(x + 0.7, y + 3, str(int(sim[j, i])), color="white", fontsize=7, zorder=4)
    ax.scatter([15], [30], s=160, marker="*", color="#2a78d6", edgecolor="white", linewidth=1.2, zorder=5)
    ax.text(16.5, 33, "기준 A", color="white", fontsize=9, zorder=5)
    ax.set_xlabel("크로스멤버 플랜지 폭 (mm)")
    ax.set_ylabel("첫 타점 거리 (mm)")
    ax.set_title("AI 제조성 지도 — 색: AI 예측 못 쏘는 타점 수 (0 짙음 → 3+ 주황)\n점: 시뮬레이터 검증 (숫자 = 실제 값, ✕ = 불일치)",
                 loc="left", color=INK, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_map.png", dpi=180)


if __name__ == "__main__":
    import sys
    fig_stage1()
    fig_stage2()
    if (OUT / "explore.json").exists():
        fig_map()

"""설계안별 차체 STEP + 타점 목록 CSV — CATIA 등 CAD 에서 열어 볼 수 있게.

results/cad/<설계안>.step  : 부품(실·B필러·크로스멤버 벽·바닥)은 이름 붙은 솔리드, 타점은 지름 6 mm 구
results/cad/<설계안>_spots.csv : 타점 id · 부위 · 좌표(mm) · 법선 · 판정 · 담당 로봇 · 못 쏘는 이유
다시 읽어 부품 수와 경계 상자를 확인한다.
"""
import csv
import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

import cadquery as cq  # noqa: E402

from body import obstacles, spots  # noqa: E402
from stage1 import DESIGNS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cad"
MM = 1000.0
NAMES = {"sill": "SIDE_SILL", "b_pillar": "B_PILLAR", "member_wall": "CROSSMEMBER_WALL", "floor": "FLOOR"}  # STEP 은 ASCII 이름
COLORS = {"sill": (0.62, 0.64, 0.68), "b_pillar": (0.55, 0.58, 0.63), "member_wall": (0.70, 0.66, 0.58),
          "floor": (0.45, 0.47, 0.50)}


def build(design, twin_design):
    asm = cq.Assembly(name=f"BIW_{design.name.split('_')[0]}")
    for b in obstacles(design):
        size = (b.hi - b.lo) * MM
        center = (b.hi + b.lo) / 2 * MM
        solid = cq.Workplane("XY").box(*size).translate(tuple(center))
        asm.add(solid, name=NAMES[b.name], color=cq.Color(*COLORS[b.name]))
    judged = {s["id"]: s for s in twin_design["spots"]}
    for s in spots(design):
        bad = judged[s.id]["robot"] is None
        ball = cq.Workplane("XY").sphere(3.0).translate(tuple(s.pos * MM))
        asm.add(ball, name=f"SPOT_{s.id}", color=cq.Color(0.92, 0.35, 0.2) if bad else cq.Color(0.2, 0.5, 0.85))
    return asm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    twin = {d["name"]: d for d in json.loads((ROOT / "results" / "twin_data.json").read_text(encoding="utf-8"))["designs"]}
    reason_ko = {"unreachable": "도달 불가", "gun_collision": "건 간섭", "arm_collision": "팔 간섭", "joint_limit": "관절 한계"}
    for d in DESIGNS:
        tag = d.name.split("_")[0]
        asm = build(d, twin[d.name])
        path = OUT / f"{tag}.step"
        asm.save(str(path), exportType="STEP")
        with open(OUT / f"{tag}_spots.csv", "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["id", "부위", "x_mm", "y_mm", "z_mm", "nx", "ny", "nz", "판정", "담당로봇", "못쏘는이유"])
            for s in twin[d.name]["spots"]:
                x, y, z = (round(v * MM, 1) for v in s["pos"])
                w.writerow([s["id"], s["group"], x, y, z, *s["normal"],
                            "OK" if s["robot"] is not None else "NG",
                            "" if s["robot"] is None else f"R{s['robot'] + 1}",
                            "/".join(reason_ko.get(r, r) for r in s["reasons"]) if s["robot"] is None else ""])
        back = cq.importers.importStep(str(path))
        solids = back.solids().vals()
        bb = back.val().BoundingBox()
        print(f"{tag}: {path.name} {path.stat().st_size // 1024} KB, 다시 읽은 솔리드 {len(solids)}개 "
              f"(부품 4 + 타점 {len(spots(d))}), 경계 {bb.xlen:.0f}×{bb.ylen:.0f}×{bb.zlen:.0f} mm")


if __name__ == "__main__":
    main()

"""LightGBM dump_model() JSON → 브라우저용 압축 트리 (viewer/ai/trees.json).

트리마다 평평한 배열: f(분할 특징), t(임계값), l/r(자식: 0 이상 = 내부 노드, 음수 = ~잎 번호), v(잎 값).
- 분할은 전부 '<=', missing_type None 인지 확인한다(아니면 멈춤).
- 특징은 float32 로 들어오므로(surrogate.design_features) 임계값을 '그 값 이하인 가장 큰 float32' 로 바꿔도
  모든 float32 입력에서 비교 결과가 같다 — 짧게 적을 수 있고 정확하다.
- 확률 = 1 / (1 + exp(-sigmoid * 잎 값 합)).
실행: .venv\\Scripts\\python viewer\\ai\\convert.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def f32_floor(t: float) -> np.float32:
    x = np.float32(t)
    if float(x) > t:
        x = np.nextafter(x, np.float32(-np.inf))
    return x


def short(x, digits=10):
    return float(f"{x:.{digits}g}")


def flatten(root):
    f, t, l, r, v = [], [], [], [], []

    def visit(n):
        if "leaf_value" in n:
            v.append(short(n["leaf_value"]))
            return ~(len(v) - 1)
        assert n["decision_type"] == "<=" and n["missing_type"] == "None", n["decision_type"]
        i = len(f)
        f.append(n["split_feature"])
        t.append(float(str(f32_floor(n["threshold"]))))   # float32 최단 표기
        l.append(None)
        r.append(None)
        l[i] = visit(n["left_child"])
        r[i] = visit(n["right_child"])
        return i

    if "leaf_value" in root:       # 잎 하나뿐인 트리
        return {"f": [], "t": [], "l": [], "r": [], "v": [short(root["leaf_value"])]}
    visit(root)
    return {"f": f, "t": t, "l": l, "r": r, "v": v}


def predict(trees, sig, x):
    s = 0.0
    for tr in trees:
        if not tr["f"]:
            s += tr["v"][0]
            continue
        n = 0
        while n >= 0:
            n = tr["l"][n] if x[tr["f"][n]] <= np.float32(tr["t"][n]) else tr["r"][n]
        s += tr["v"][~n]
    return 1 / (1 + np.exp(-sig * s))


def main():
    m = json.loads((HERE / "model.json").read_text(encoding="utf-8"))
    obj = m["objective"].split()
    assert obj[0] == "binary", obj
    sig = float(obj[1].split(":")[1]) if len(obj) > 1 else 1.0
    trees = [flatten(t["tree_structure"]) for t in m["tree_info"]]
    out = {"objective": "binary", "sigmoid": sig, "n_features": m["max_feature_idx"] + 1,
           "feature_names": m["feature_names"], "trees": trees}
    path = HERE / "trees.json"
    path.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"{len(trees)} trees, {sum(len(t['f']) for t in trees)} splits → {path.stat().st_size / 1e6:.2f} MB")

    # 자체 검증: 메타의 parity 설계 첫 8행 특징으로 파이썬 확률과 비교
    meta = json.loads((HERE / "meta.json").read_text(encoding="utf-8"))
    err = 0.0
    for d in meta["parity"]:
        for k, row in enumerate(d["features"]):
            x = np.array(row, dtype=np.float32)
            err = max(err, abs(predict(trees, sig, x) - d["prob"][k]))
    print(f"parity (first-8 rows × {len(meta['parity'])} designs): max |p - p_py| = {err:.2e}")


if __name__ == "__main__":
    main()

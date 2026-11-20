"""重賞の出走馬で、血統の道悪倍率の効き方を細かく見る（全馬の血統が取れるまでの目安）.

    npx tsx scripts/ml/pedigree_ancestors.ts && (cd scripts/ml && python analyze_heavy_graded.py)
    → data/ml/heavy_graded_analysis.json

1段目の確率は train_stack.py と同じく「そのレースの年を除いて学習したモデル」の予測（data/ml/graded_oof.csv に保存し、2回目からは再利用）。
補正は logit(p) + Σ β·log(倍率)。β は年を1つ外して推定し、外した年で logloss を測る（合計で比べる）。
- by_going：稍重・重・不良ごと（重・不良の経験なしの馬）
- by_ratio：倍率の大きさごとの、実際の3着内率と予測の平均（補正なし・β=1）
- weights：父・母父・母母父の重みの付け方の比較（経験なし・稍重〜不良）
- by_experience：重・不良の出走数（0・1・2・3〜）ごとの β（経験がある馬にも血統が効くか）
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PED = REPO / "data" / "pedigree"
OOF = OUT / "graded_oof.csv"
POS = ["F", "MF", "MMF"]


def load_oof() -> dict:
    if not OOF.exists():
        from train_stack import oof_base  # 重い（1段目を年ごとに4回学習する）
        ped = list(csv.DictReader((PED / "graded_features.csv").open(encoding="utf-8")))
        keys = {(r["race_id"], int(r["umaban"])) for r in ped}
        base, context = oof_base(keys, sorted({r["date"][:4] for r in ped}))
        with OOF.open("w", encoding="utf-8") as f:
            f.write("race_id,umaban,p,going,heavy_n,wet_n\n")
            for k in sorted(base):
                g, h, w = context[k]
                f.write(f"{k[0]},{k[1]},{base[k]},{'' if g is None else g},{h},{w}\n")
    out = {}
    for r in csv.DictReader(OOF.open(encoding="utf-8")):
        out[(r["race_id"], int(r["umaban"]))] = {
            "p": float(r["p"]), "going": None if r["going"] == "" else int(float(r["going"])),
            "heavy_n": int(float(r["heavy_n"])), "wet_n": int(float(r["wet_n"]))}
    return out


def fit(Z, X, y, idx):
    """logit = Z + X·b のロジスティック回帰（切片なし）をニュートン法で。戻り値 b, 標準誤差."""
    k = X.shape[1]
    b = np.zeros(k)
    for _ in range(50):
        q = 1 / (1 + np.exp(-(Z[idx] + X[idx] @ b)))
        H = (X[idx] * (q * (1 - q))[:, None]).T @ X[idx] + 1e-9 * np.eye(k)
        step = np.linalg.solve(H, X[idx].T @ (y[idx] - q))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(H)))


def nll(Z, X, y, b, idx) -> float:
    q = 1 / (1 + np.exp(-(Z[idx] + X[idx] @ b)))
    q = np.clip(q, 1e-12, 1 - 1e-12)
    return float(-np.sum(y[idx] * np.log(q) + (1 - y[idx]) * np.log(1 - q)))


def loyo(Z, X, y, yr, idx):
    """年を1つ外して β を推定 → 外した年で logloss。補正なし・β=1（X が1列のとき）も同じ年で出す."""
    res = {"none": 0.0, "estimated": 0.0}
    if X.shape[1] == 1:
        res["beta1"] = 0.0
    years_better = 0
    for yy in sorted(set(yr[idx])):
        te = idx[yr[idx] == yy]
        tr = idx[yr[idx] != yy]
        b, _ = fit(Z, X, y, tr)
        none = nll(Z, X, y, np.zeros(X.shape[1]), te)
        est = nll(Z, X, y, b, te)
        res["none"] += none
        res["estimated"] += est
        years_better += est < none
        if X.shape[1] == 1:
            res["beta1"] += nll(Z, X, y, np.ones(1), te)
    b, se = fit(Z, X, y, idx)
    return {"runners": int(len(idx)), "coef": [round(float(v), 3) for v in b], "se": [round(float(v), 3) for v in se],
            **{f"logloss_{k}": round(v, 3) for k, v in res.items()}, "years_estimated_better": years_better}


def main() -> None:
    oof = load_oof()
    feats = {(r["race_id"], int(r["umaban"])): r for r in csv.DictReader((PED / "graded_features.csv").open(encoding="utf-8"))}
    anc = {(r["race_id"], int(r["umaban"])): r for r in csv.DictReader((PED / "heavy_ancestors.csv").open(encoding="utf-8"))}

    rows = []
    for k, o in oof.items():
        f, a = feats.get(k), anc.get(k)
        if not f or not a or o["going"] is None or o["going"] < 1 or not f["heavy_ratio"]:
            continue
        p = min(1 - 1e-6, max(1e-6, o["p"]))
        rows.append({
            "year": f["date"][:4], "y": int(f["label"]), "z": math.log(p / (1 - p)), "p": p,
            "going": o["going"], "heavy_n": o["heavy_n"], "ratio": float(f["heavy_ratio"]),
            **{f"{s}_r": float(a[f"{s}_ratio"]) if a[f"{s}_ratio"] else None for s in POS},
            **{f"{s}_n": float(a[f"{s}_starts"]) if a[f"{s}_starts"] else 0.0 for s in POS},
        })
    y = np.array([r["y"] for r in rows]); Z = np.array([r["z"] for r in rows]); yr = np.array([r["year"] for r in rows])
    going = np.array([r["going"] for r in rows]); hn = np.array([r["heavy_n"] for r in rows])
    logr = np.log(np.array([r["ratio"] for r in rows]))[:, None]
    inexp = np.where(hn == 0)[0]
    result: dict = {"note": "重賞（2023〜2026年）の稍重〜不良。logloss は年を1つ外した検証の合計。数が少ないので目安", "wet_runners": int(len(rows))}

    # 馬場ごと
    result["by_going"] = {name: loyo(Z, logr, y, yr, inexp[going[inexp] == code]) for code, name in ((1, "稍重"), (2, "重"), (3, "不良"))}
    result["by_going"]["稍重〜不良"] = loyo(Z, logr, y, yr, inexp)

    # 倍率の大きさごと
    bins = [(0, 0.85, "〜0.85"), (0.85, 0.95, "0.85〜0.95"), (0.95, 1.05, "0.95〜1.05"), (1.05, 1.15, "1.05〜1.15"), (1.15, 99, "1.15〜")]
    ratio = np.exp(logr[:, 0])
    p0 = 1 / (1 + np.exp(-Z)); p1 = 1 / (1 + np.exp(-(Z + logr[:, 0])))
    result["by_ratio"] = {}
    for lo, hi, name in bins:
        idx = inexp[(ratio[inexp] >= lo) & (ratio[inexp] < hi)]
        if len(idx):
            result["by_ratio"][name] = {"runners": int(len(idx)), "actual_top3": round(float(y[idx].mean()), 4),
                                        "pred_none": round(float(p0[idx].mean()), 4), "pred_beta1": round(float(p1[idx].mean()), 4)}

    # 重みの付け方（3頭とも倍率があり、経験なしの馬）
    full = np.array([all(r[f"{s}_r"] for s in POS) for r in rows])
    idx_w = inexp[full[inexp]]
    L = {s: np.log(np.array([r[f"{s}_r"] or 1.0 for r in rows])) for s in POS}
    N = {s: np.array([r[f"{s}_n"] for r in rows]) for s in POS}

    def combo(w):
        tot = sum(w.values())
        return (sum(w[s] * L[s] for s in POS) / tot)[:, None]

    def combo_ratio(w):  # 今のアプリと同じく、倍率そのものを重み付き平均してから log
        tot = sum(w.values())
        return np.log(sum(w[s] * np.exp(L[s]) for s in POS) / tot)[:, None]

    nw = N["F"] + N["MF"] + N["MMF"]
    result["weights"] = {
        "今 50:25:12.5（倍率の平均）": loyo(Z, combo_ratio({"F": 4, "MF": 2, "MMF": 1}), y, yr, idx_w),
        "40:35:25": loyo(Z, combo({"F": 40, "MF": 35, "MMF": 25}), y, yr, idx_w),
        "均等 1:1:1": loyo(Z, combo({"F": 1, "MF": 1, "MMF": 1}), y, yr, idx_w),
        "産駒の重・不良の出走数に比例": loyo(Z, (sum(N[s] * L[s] for s in POS) / np.maximum(nw, 1))[:, None], y, yr, idx_w),
        "父だけ": loyo(Z, L["F"][:, None], y, yr, idx_w),
        "母父だけ": loyo(Z, L["MF"][:, None], y, yr, idx_w),
        "母母父だけ": loyo(Z, L["MMF"][:, None], y, yr, idx_w),
        "3頭を別々に推定": loyo(Z, np.stack([L[s] for s in POS], axis=1), y, yr, idx_w),
    }
    result["weights_note"] = "coef は β（3頭を別々に推定のときは 父・母父・母母父の順）。1列のときの logloss_beta1 は β=1"

    # 経験の量ごと（稍重〜不良）
    result["by_experience"] = {}
    for lo, hi, name in ((0, 0, "0戦"), (1, 1, "1戦"), (2, 2, "2戦"), (3, 999, "3戦〜")):
        idx = np.where((hn >= lo) & (hn <= hi))[0]
        result["by_experience"][name] = loyo(Z, logr, y, yr, idx)

    (OUT / "heavy_graded_analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

"""今回の距離に近い走ほど重く見る能力指数の比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 196項目）.

    cd scripts/ml && python compare_abdist.py   # → data/ml/feature_comparison_abdist.json

依頼者の指摘（2026-11-22、マイルCS のスターアニス）：今の能力指数の直近加重平均（同じ芝ダの近5走、0.8^k）は、
今回と距離がまったく違う走（オークス 2400m 12着 −0.45、スプリンターズS 1200m +0.41）も同じ重さで入り、平均が下がる（+0.46、18頭中17位）。
1600m の2走は +0.82・+1.19 でどちらも勝っている。
H1 距離で重み付け：同じ芝ダの近10走で、重み＝0.8^k × exp(−|今回の距離 − その走の距離| ÷ 400m)（同じ距離 1、±400m 0.37、±800m 0.14）の加重平均
H2 近い距離だけ：同じ芝ダの近10走のうち、今回の距離 ±400m 以内の走（新しい順に最大5走）の 0.8^k 加重平均。なければ欠損
それぞれ、出走馬の中での順位÷頭数・最高との差・3番手との差も足す
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
SCALE = 400.0


def relative(rows_by_race, name):
    for rs in rows_by_race.values():
        vals = sorted((x[name] for x in rs if not math.isnan(x[name])), reverse=True)
        for x in rs:
            v = x[name]
            if math.isnan(v) or not vals:
                x[name + "_rank"] = x[name + "_dmax"] = x[name + "_d3"] = NAN
                continue
            x[name + "_rank"] = (1 + sum(1 for o in vals if o > v)) / len(rs)
            x[name + "_dmax"] = v - vals[0]
            x[name + "_d3"] = v - (vals[2] if len(vals) >= 3 else vals[-1])


def main() -> None:
    rows, hist = build_dataset(DATA)
    by_race = defaultdict(list)
    for r in rows:
        d = r["distance"]
        runs = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"] and x["surface"] == r["surface"] and x["ab"] is not None][::-1][:10]
        if runs and not math.isnan(d):
            w = [0.8 ** k * math.exp(-abs(d - x["distance"]) / SCALE) for k, x in enumerate(runs)]
            r["abd_w"] = sum(a * x["ab"] for a, x in zip(w, runs)) / sum(w)
            near = [x for x in runs if abs(d - x["distance"]) <= 400][:5]
            wn = [0.8 ** k for k in range(len(near))]
            r["abn_w"] = sum(a * x["ab"] for a, x in zip(wn, near)) / sum(wn) if near else NAN
        else:
            r["abd_w"] = r["abn_w"] = NAN
        by_race[r["race_id"]].append(r)
    relative(by_race, "abd_w")
    relative(by_race, "abn_w")
    H1 = ["abd_w", "abd_w_rank", "abd_w_dmax", "abd_w_d3"]
    H2 = ["abn_w", "abn_w_rank", "abn_w_dmax", "abn_w_d3"]
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + H1 + H2
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(196)": FEATURES, "+H1 距離で重み付け": FEATURES + H1, "+H2 近い距離だけ": FEATURES + H2, "+両方": cols}
    out = {"scale_m": SCALE, "coverage": {f: round(float(np.mean(~np.isnan(X[:, cols.index(f)]))), 3) for f in ("abd_w", "abn_w")}, "sets": {}}
    first = None
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < first[t]["logloss_sum3"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_abdist.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

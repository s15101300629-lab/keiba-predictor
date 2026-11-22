"""距離の違う走が混ざる問題の、能力指数以外の項目の比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 196項目）.

    cd scripts/ml && python compare_distmix.py   # → data/ml/feature_comparison_distmix.json

依頼者の指摘（2026-11-22、マイルCS のスターアニス）を受けた点検で、近走をそのまま平均する項目に同じ問題があった
- 調子（近5走の着順・タイム差の成分）：オークス 2400m 12着が入る
- タイム指数の近5走平均：13位（最高は 8位）
- 距離の変化（前走との差）：前走が 1200m（スプリンターズS）なので「400m 延長」として下がる。勝った 1600m に戻るだけ
J1 調子を距離で重み付け：近10走（芝ダ問わず）で、重み＝0.8^k × exp(−|今回の距離 − その走の距離| ÷ 400m) の着順成分・タイム差成分の平均
J2 タイム指数を距離で重み付け：同じ芝ダの近10走で同じ重みの平均
J3 ふだんの距離との差：今回の距離 − 近5走の距離の平均、今回の距離 − 能力指数が一番高かった走の距離
J1・J2 は出走馬の中での順位÷頭数も足す
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, run_components  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
SCALE = 400.0


def wavg(vals, w):
    return sum(a * b for a, b in zip(vals, w)) / sum(w) if w and sum(w) > 0 else NAN


def rank_in_race(by_race, name):
    for rs in by_race.values():
        vals = [x[name] for x in rs if not math.isnan(x[name])]
        for x in rs:
            v = x[name]
            x[name + "_rank"] = NAN if math.isnan(v) else (1 + sum(1 for o in vals if o > v)) / len(rs)


def main() -> None:
    rows, hist = build_dataset(DATA)
    by_race = defaultdict(list)
    for r in rows:
        d = r["distance"]
        all_runs = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"]][::-1]
        r10 = all_runs[:10]
        if r10 and not math.isnan(d):
            w = [0.8 ** k * math.exp(-abs(d - x["distance"]) / SCALE) for k, x in enumerate(r10)]
            comps = [run_components(x["finish"], x["field"], x["time_diff"]) for x in r10]
            r["fdw_fin"] = wavg([c[0] for c in comps], w)
            r["fdw_time"] = wavg([c[1] for c in comps], w)
        else:
            r["fdw_fin"] = r["fdw_time"] = NAN
        same = [x for x in all_runs if x["surface"] == r["surface"] and x["tidx"] is not None][:10]
        if same and not math.isnan(d):
            r["tdw"] = wavg([x["tidx"] for x in same], [0.8 ** k * math.exp(-abs(d - x["distance"]) / SCALE) for k, x in enumerate(same)])
        else:
            r["tdw"] = NAN
        r5 = [x for x in all_runs[:5] if x["distance"]]
        r["dist_vs_usual"] = d - sum(x["distance"] for x in r5) / len(r5) if r5 and not math.isnan(d) else NAN
        ab = [x for x in all_runs if x["surface"] == r["surface"] and x["ab"] is not None]
        r["dist_vs_best"] = d - max(ab, key=lambda x: x["ab"])["distance"] if ab and not math.isnan(d) else NAN
        by_race[r["race_id"]].append(r)
    for name in ("fdw_fin", "fdw_time", "tdw"):
        rank_in_race(by_race, name)
    J1 = ["fdw_fin", "fdw_time", "fdw_fin_rank", "fdw_time_rank"]
    J2 = ["tdw", "tdw_rank"]
    J3 = ["dist_vs_usual", "dist_vs_best"]
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + J1 + J2 + J3
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(196)": FEATURES, "+J1 調子を距離で重み付け": FEATURES + J1, "+J2 タイム指数を距離で重み付け": FEATURES + J2,
            "+J3 ふだんの距離との差": FEATURES + J3, "+全部": cols}
    out = {"scale_m": SCALE, "sets": {}}
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
        (OUT / "feature_comparison_distmix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

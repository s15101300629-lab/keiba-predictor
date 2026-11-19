"""走行位置の効率性の比較（1年ずつ外す）.

    cd scripts/ml && python compare_position.py   # → data/ml/feature_comparison_position.json

成績表の通過順位（コーナーごとの位置）と着順から、近5走で作る。位置はどれも頭数で割る。
- 位置取り変化量：pos_change_mean5 =（最初のコーナーの位置 − 最後のコーナーの位置）÷頭数 の平均（前に上がるほど +）
- 4角からの順位改善：late_gain_mean5 / late_gain_last1 =（最後のコーナーの位置 − 着順）÷頭数（ゴールまでに抜いたほど +）
- 位置取り維持：hold_n = 序盤に前3分の1にいた走の数、hold_rate = そのうち最後のコーナーでも前3分の1にいた割合、
  front_late_gain = その走での4角からの順位改善の平均（先行して粘れたか）
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")


def mean(xs):
    return sum(xs) / len(xs) if xs else NAN


def main() -> None:
    info = {}
    for r in load_races(DATA):
        n = r["field_size"]
        for h in r["runners"]:
            cps = [x for x in (h.get("corner_positions") or []) if x]
            if not cps or h["finish"] is None or n < 2:
                info[(r["race_id"], h["horse"])] = None
                continue
            first, last = cps[0] / n, cps[-1] / n
            info[(r["race_id"], h["horse"])] = {"change": first - last, "late": (cps[-1] - h["finish"]) / n,
                                               "front": first <= 1 / 3, "hold": last <= 1 / 3}
    rows, _ = build_dataset(DATA)
    by_horse: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_horse[r["horse"]].append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        past: list[dict] = []
        for r in runs:
            last5 = past[-5:]
            r["pos_change_mean5"] = mean([p["change"] for p in last5])
            r["late_gain_mean5"] = mean([p["late"] for p in last5])
            r["late_gain_last1"] = past[-1]["late"] if past else NAN
            fronts = [p for p in last5 if p["front"]]
            r["hold_n"] = len(fronts) if last5 else NAN
            r["hold_rate"] = mean([1.0 if p["hold"] else 0.0 for p in fronts])
            r["front_late_gain"] = mean([p["late"] for p in fronts])
            v = info.get((r["race_id"], r["horse"]))
            if v is not None:
                past.append(v)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    change = ["pos_change_mean5"]
    late = ["late_gain_mean5", "late_gain_last1"]
    hold = ["hold_n", "hold_rate", "front_late_gain"]
    sets = {"base": FEATURES, "+位置取り変化量": FEATURES + change, "+4角からの順位改善": FEATURES + late,
            "+位置取り維持": FEATURES + hold, "+全部": FEATURES + change + late + hold}
    feats_all = change + late + hold
    coverage = {f: round(sum(1 for r in rows if not (isinstance(r[f], float) and math.isnan(r[f]))) / len(rows), 3) for f in feats_all}
    print("coverage", coverage, file=sys.stderr, flush=True)
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {"coverage": coverage, "sets": {}}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            bst = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = bst.predict(m(te))
            y = [r["label"] for r in te]
            per[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result["sets"][name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_position.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

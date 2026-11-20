"""馬体重の発表前（前日まで）の計算方法の検証.

    cd scripts/ml && python eval_body_weight.py   # → data/ml/body_weight_eval.json

学習データには馬体重の欠けがないため、発表前に空のまま予測すると確率が大きく下がる。
前年までで学習して、その年（2024〜2026年）を次の3通りで予測して比べる。
- actual：本当の馬体重・増減（当日の発表後と同じ）
- prev：前走の馬体重・増減0（前走がない馬は、学習データの初出走馬の性別ごとの中央値）
- missing：空のまま（今のアプリの発表前）
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
import features  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix, race_top3_hit  # noqa: E402


def prev_body_weights():
    """(race_id, 馬名) → 前走の馬体重（なければ None）と、初出走馬の性別ごとの馬体重の中央値."""
    prev, last = {}, {}
    debut = defaultdict(list)
    for race in features.load_races(DATA):
        for h in race["runners"]:
            prev[(race["race_id"], h["horse"])] = last.get(h["horse"])
            if h["horse"] not in last and h.get("body_weight") and race["date"] >= TRAIN_FROM:
                debut[h.get("sex")].append(h["body_weight"])
        for h in race["runners"]:
            if h.get("body_weight"):
                last[h["horse"]] = h["body_weight"]
    return prev, {k: float(np.median(v)) for k, v in debut.items() if k}


def main() -> None:
    prev, debut_median = prev_body_weights()
    rows, _ = features.build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    i_bw, i_diff, i_sex = features.FEATURES.index("body_weight"), features.FEATURES.index("body_weight_diff"), features.FEATURES.index("sex")
    sex_code = {v: k for k, v in getattr(features, "SEX", {}).items()} if hasattr(features, "SEX") else {}
    result = {"rounds": rounds, "debut_median_by_sex": debut_median, "per_year": {}}
    for year in ["2024", "2025", "2026"]:
        tr = [r for r in rows if r["date"][:4] < year]
        te = [r for r in rows if r["date"][:4] == year]
        booster = lgb.train(PARAMS, lgb.Dataset(matrix(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
        y = [r["label"] for r in te]
        X = matrix(te)
        Xp = X.copy(); Xm = X.copy()
        no_prev = 0
        for j, r in enumerate(te):
            bw = prev.get((r["race_id"], r["horse"]))
            if bw is None:
                no_prev += 1
                sex = sex_code.get(X[j, i_sex])
                bw = debut_median.get(sex, float(np.median(list(debut_median.values()))))
            Xp[j, i_bw], Xp[j, i_diff] = bw, 0.0
            Xm[j, i_bw], Xm[j, i_diff] = math.nan, math.nan
        per = {"runners": len(te), "no_previous_weight": no_prev}
        for name, M in (("actual", X), ("prev", Xp), ("missing", Xm)):
            p = booster.predict(M)
            per[name] = {"auc": roc_auc_score(y, p), "logloss": log_loss(y, p), "top3_hit": race_top3_hit(te, p),
                         "mean_sum_per_race": float(np.sum(p) / len({r["race_id"] for r in te}))}
        result["per_year"][year] = per
        print(year, json.dumps(per, ensure_ascii=False), file=sys.stderr, flush=True)
    (OUT / "body_weight_eval.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

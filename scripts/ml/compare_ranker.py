"""LightGBM Ranker（lambdarank）と今の Classifier（3着以内）の比較（1年ずつ外す）.

    cd scripts/ml && python compare_ranker.py   # → data/ml/ranker_comparison.json

- Ranker の正解：1着=3、2着=2、3着=1、それ以外=0。レースごとに並びを学習する
- 木の本数は、学習に使う年のうち一番新しい年で決める（評価する年は使わない）
- 比べる順位：Classifier / Ranker / 2つの順位の平均
- 指標：上位3頭の3着内率、1位の馬の勝率・3着内率、1位の馬の単勝・複勝の回収率
"""

from __future__ import annotations

import csv
import glob
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RANK_PARAMS = {**PARAMS, "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [3]}


def relevance(r) -> int:
    f = r["finish"]
    return 0 if f is None or f > 3 else 4 - f


def umaban_of(r):
    u = r["umaban"]
    return None if u is None or (isinstance(u, float) and math.isnan(u)) else int(u)


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    rows.sort(key=lambda r: (r["race_id"], r["umaban"] if umaban_of(r) is not None else 99))
    place_pay, win_pay = {}, {}
    for f in glob.glob(str(ROOT / "data/jra-seiseki/csv/*/payouts.csv")):
        for p in csv.DictReader(open(f, encoding="utf-8")):
            if p["type"] in ("複勝", "単勝") and p["combo"] and p["payout"]:
                try:
                    (place_pay if p["type"] == "複勝" else win_pay)[(p["race_id"], int(p["combo"]))] = int(p["payout"].replace(",", ""))
                except ValueError:
                    pass
    mat = lambda rs: np.array([[r[f] for f in FEATURES] for r in rs], dtype=np.float64)  # noqa: E731

    def groups(rs):
        sizes, last = [], None
        for r in rs:
            if r["race_id"] != last:
                sizes.append(0)
                last = r["race_id"]
            sizes[-1] += 1
        return sizes

    def evaluate(rs, score):
        by = {}
        for r, s in zip(rs, score):
            by.setdefault(r["race_id"], []).append((s, r))
        n = hit3 = win1 = place1 = wret = pret = 0
        for items in by.values():
            items.sort(key=lambda t: -t[0])
            n += 1
            hit3 += sum(r["label"] for _, r in items[:3]) / min(3, len(items))
            top = items[0][1]
            u = umaban_of(top)
            win1 += top["finish"] == 1
            place1 += top["label"]
            wret += win_pay.get((top["race_id"], u), 0) if top["finish"] == 1 else 0
            pret += place_pay.get((top["race_id"], u), 0) if top["label"] else 0
        return {"races": n, "top3_hit": hit3 / n, "top1_win": win1 / n, "top1_place": place1 / n,
                "top1_win_roi": wret / n / 100, "top1_place_roi": pret / n / 100}

    def rank_in_race(rs, score):
        out = np.zeros(len(rs))
        by = {}
        for i, (r, s) in enumerate(zip(rs, score)):
            by.setdefault(r["race_id"], []).append((s, i))
        for items in by.values():
            items.sort(key=lambda t: -t[0])
            for k, (_, i) in enumerate(items):
                out[i] = k + 1
        return out

    years = sorted({r["date"][:4] for r in rows})
    result = {"per_year": {}}
    for year in years:
        tr = [r for r in rows if r["date"][:4] != year]
        te = [r for r in rows if r["date"][:4] == year]
        val_year = max(y for y in years if y != year)
        fit = [r for r in tr if r["date"][:4] != val_year]
        val = [r for r in tr if r["date"][:4] == val_year]
        # Classifier（3着以内）
        dfit = lgb.Dataset(mat(fit), label=[r["label"] for r in fit])
        b = lgb.train(PARAMS, dfit, 3000, valid_sets=[lgb.Dataset(mat(val), label=[r["label"] for r in val], reference=dfit)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        clf = lgb.train(PARAMS, lgb.Dataset(mat(tr), label=[r["label"] for r in tr]), max(b.best_iteration, 50))
        # Ranker
        dfit = lgb.Dataset(mat(fit), label=[relevance(r) for r in fit], group=groups(fit))
        dval = lgb.Dataset(mat(val), label=[relevance(r) for r in val], group=groups(val), reference=dfit)
        b = lgb.train(RANK_PARAMS, dfit, 3000, valid_sets=[dval], callbacks=[lgb.early_stopping(100, verbose=False)])
        rk = lgb.train(RANK_PARAMS, lgb.Dataset(mat(tr), label=[relevance(r) for r in tr], group=groups(tr)), max(b.best_iteration, 50))
        p_clf = clf.predict(mat(te))
        p_rk = rk.predict(mat(te))
        avg = -(rank_in_race(te, p_clf) + rank_in_race(te, p_rk))  # 順位の平均（小さいほど上）
        result["per_year"][year] = {"ranker_rounds": b.best_iteration, "Classifier": evaluate(te, p_clf),
                                    "Ranker": evaluate(te, p_rk), "平均": evaluate(te, avg + 1e-6 * p_clf)}
        print(year, json.dumps(result["per_year"][year], ensure_ascii=False), file=sys.stderr, flush=True)
    keys = ["top3_hit", "top1_win", "top1_place", "top1_win_roi", "top1_place_roi"]
    result["mean"] = {m: {k: float(np.mean([v[m][k] for v in result["per_year"].values()])) for k in keys}
                      for m in ("Classifier", "Ranker", "平均")}
    (OUT / "ranker_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["mean"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

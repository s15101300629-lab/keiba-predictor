"""seed 5本の平均の効果を、表示を遅くせずに取り入れる比較（2026-11-25、依頼者の案）.

    cd scripts/ml && python cache_dataset.py
    cd scripts/ml && python compare_distill.py top3    # → data/ml/distill_top3.json（win・top2 も同じ）

比べるもの（どれも前年までで学習して 2024〜2026年を予測。2026-11-21 以降は使わない）
- 今の1本：train.PARAMS（seed 42）。木の本数は前の年を検証にして決める（compare_round4.fit_year と同じ）
- 5本の平均：seed 1〜5 の5本（それぞれ同じ決め方）の確率の平均
- 案1（蒸留）：5本の平均の確率（0〜1）を答えにして、1本を学び直す（objective cross_entropy）。
  答えは学習に使う行での5本の予測（学習した行そのものへの予測）。木の本数は、前々年までで学習した5本の平均を答えにして学び、
  前の年の本当の答え（0/1）での cross_entropy が一番良いところで決める
- 案2：学習率を5倍（0.03→0.15）にして、1本の木の数を今の1本の約1/5（今の本数÷5）にした5本（seed 1〜5）の平均。合計の木の数は今と同じ
logloss は合計（3着以内 3・連対 2・勝率 1）にそろえた後。的中率は、3着以内＝上位3頭の3着内率、連対＝上位2頭の連対率、勝率＝1位の馬の勝率
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_round4 import YEARS, load, write  # noqa: E402
from train import PARAMS  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

SEEDS = [1, 2, 3, 4, 5]
TARGETS = {"top3": 3, "top2": 2, "win": 1}


def topk_hit(race_id, y, p, k) -> float:
    order = np.lexsort((-p, race_id))
    rid, yy = race_id[order], y[order]
    start = np.r_[0, np.flatnonzero(rid[1:] != rid[:-1]) + 1]
    rank = np.arange(len(rid)) - np.repeat(start, np.diff(np.r_[start, len(rid)]))
    sel = rank < k
    return float(yy[sel].sum() / sel.sum())


def metrics(y, p, rid, k) -> dict:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    pn = normalize_sum(rid, p, float(k))
    return {"logloss_sum": round(float(log_loss(y, pn)), 5), "auc": round(float(roc_auc_score(y, p)), 5), "hit": round(topk_hit(rid, y, p, k), 5)}


def with_seed(params, seed):
    p = dict(params)
    if seed is not None:
        p.update(seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    return p


def fit_es(X, y, fit, val, params):
    """fit で学び、val で早めに止める（本数を決める）. 戻り値は止めたモデルと本数."""
    dfit = lgb.Dataset(X[fit], y[fit])
    b = lgb.train(params, dfit, 6000, valid_sets=[lgb.Dataset(X[val], y[val], reference=dfit)], callbacks=[lgb.early_stopping(100, verbose=False)])
    return b, max(b.best_iteration, 50)


def run(target: str) -> None:
    k = TARGETS[target]
    d = load()
    X, year, rid, fin = d["X"], d["year"], d["race_id"], d["finish"]
    y = (fin <= k).astype(float)
    out = {"target": target, "note": __doc__.split("\n\n")[1].strip(), "years": {}}
    for ty in YEARS:
        vy = str(int(ty) - 1)
        fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
        res = {}
        # 今の1本
        _, r1 = fit_es(X, y, fit, val, PARAMS)
        m1 = lgb.train(PARAMS, lgb.Dataset(X[tr], y[tr]), r1)
        res["今の1本"] = {"trees": r1, **metrics(y[te], m1.predict(X[te]), rid[te], k)}
        print(target, ty, "今の1本", res["今の1本"], file=sys.stderr, flush=True)
        # 5本の平均（前々年までで学んだ5本＝蒸留の本数決め用、前年までで学んだ5本＝予測と蒸留の答え）
        tfit, ttr, pte, trees5 = np.zeros(fit.sum()), np.zeros(tr.sum()), np.zeros(te.sum()), []
        for s in SEEDS:
            ps = with_seed(PARAMS, s)
            b, r = fit_es(X, y, fit, val, ps)
            tfit += b.predict(X[fit], num_iteration=b.best_iteration) / len(SEEDS)
            m = lgb.train(ps, lgb.Dataset(X[tr], y[tr]), r)
            ttr += m.predict(X[tr]) / len(SEEDS)
            pte += m.predict(X[te]) / len(SEEDS)
            trees5.append(r)
        res["5本の平均"] = {"trees": int(sum(trees5)), **metrics(y[te], pte, rid[te], k)}
        print(target, ty, "5本の平均", res["5本の平均"], file=sys.stderr, flush=True)
        # 案1：蒸留（答え＝5本の平均の確率。cross_entropy）
        pd_ = dict(PARAMS, objective="cross_entropy", metric="cross_entropy")
        dfit = lgb.Dataset(X[fit], np.clip(tfit, 0, 1))
        b = lgb.train(pd_, dfit, 6000, valid_sets=[lgb.Dataset(X[val], y[val], reference=dfit)], callbacks=[lgb.early_stopping(100, verbose=False)])
        rd = max(b.best_iteration, 50)
        md = lgb.train(pd_, lgb.Dataset(X[tr], np.clip(ttr, 0, 1)), rd)
        res["案1 蒸留"] = {"trees": rd, **metrics(y[te], md.predict(X[te]), rid[te], k)}
        print(target, ty, "案1", res["案1 蒸留"], file=sys.stderr, flush=True)
        # 案2：学習率5倍・1本の木を今の1本の1/5にした5本の平均
        r2 = max(int(round(r1 / len(SEEDS))), 10)
        p2 = np.zeros(te.sum())
        for s in SEEDS:
            ps = with_seed(dict(PARAMS, learning_rate=PARAMS["learning_rate"] * len(SEEDS)), s)
            p2 += lgb.train(ps, lgb.Dataset(X[tr], y[tr]), r2).predict(X[te]) / len(SEEDS)
        res["案2 速い5本"] = {"trees": r2 * len(SEEDS), "trees_each": r2, **metrics(y[te], p2, rid[te], k)}
        print(target, ty, "案2", res["案2 速い5本"], file=sys.stderr, flush=True)
        # 5本の平均の改善のうち、どれだけ残ったか（logloss）
        gain = res["今の1本"]["logloss_sum"] - res["5本の平均"]["logloss_sum"]
        for name in ("案1 蒸留", "案2 速い5本"):
            res[name]["kept_of_gain"] = round((res["今の1本"]["logloss_sum"] - res[name]["logloss_sum"]) / gain, 3) if gain > 0 else None
        out["years"][ty] = res
        write(f"distill_{target}.json", out)
    names = ["今の1本", "5本の平均", "案1 蒸留", "案2 速い5本"]
    out["mean"] = {n: {m: round(float(np.mean([out["years"][t][n][m] for t in YEARS])), 5) for m in ("logloss_sum", "auc", "hit")} for n in names}
    g = out["mean"]["今の1本"]["logloss_sum"] - out["mean"]["5本の平均"]["logloss_sum"]
    for n in ("案1 蒸留", "案2 速い5本"):
        out["mean"][n]["kept_of_gain"] = round((out["mean"]["今の1本"]["logloss_sum"] - out["mean"][n]["logloss_sum"]) / g, 3) if g > 0 else None
    write(f"distill_{target}.json", out)


if __name__ == "__main__":
    run(sys.argv[1])

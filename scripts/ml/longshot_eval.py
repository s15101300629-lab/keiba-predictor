"""人気薄（6番人気以下）が3着以内に来るのを、AI がどれだけ当てられているか。と、人気薄専用の2段目モデルの試し.

    cd scripts/ml && python longshot_eval.py   # → data/ml/longshot_eval.json

年ごとに前年までで学習した本体の予測（backtest_value.prepare。今の 182項目・PARAMS）と確定オッズを使う。人気は確定単勝オッズの順位
1. 人気の帯ごとに：頭数・実際の3着内率・AI と市場（ハービル）の3着以内確率の平均、帯の中での AUC（AI・市場・合わせた確率）
2. 人気薄の中で、点数の高い順に上位 5%・10%・20% を複勝で買った回収率（AI・市場・合わせた確率・2段目）
3. 2段目（人気薄専用）：6番人気以下だけで、本体の182項目＋AI の確率＋市場の確率・オッズから3着以内を学習（LightGBM。試す年より前の年だけ）
   合わせた確率は、AI と市場の3着以内確率の logit をロジスティック回帰で合わせたもの（試す年より前の年の全馬で当てはめ）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from train import OUT  # noqa: E402

bv.PRED_YEARS = ["2022", "2023", "2024", "2025", "2026"]  # 2段目の学習に前の年を多めに使う
TEST_YEARS = ["2024", "2025", "2026"]
BANDS = {"1〜3番人気": (1, 3), "4〜5番人気": (4, 5), "6〜9番人気": (6, 9), "10番人気〜": (10, 99)}
LONG = 6
QS = [0.05, 0.1, 0.2]
P2 = {"objective": "binary", "learning_rate": 0.03, "num_leaves": 15, "min_data_in_leaf": 200, "feature_fraction": 0.5,
      "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 10.0, "verbose": -1, "seed": 7}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, s):
    return round(float(roc_auc_score(y, s)), 4) if 0 < y.sum() < len(y) else None


def main() -> None:
    d = bv.prepare()
    n = d["n"]
    pop = np.zeros(n, int)
    for g in d["gs"]:
        order = np.argsort(d["odds"][g], kind="stable")
        pop[g[order]] = np.arange(1, len(g) + 1)
    y = d["is_top3"].astype(float)
    year = d["year"]
    ai, mkt = d["ai_top3"], d["mkt_top3"]
    place = d["place_ret"] / 100.0  # 100円あたりの払戻 → 倍
    X2 = np.column_stack([d["X"], logit(ai), logit(mkt), np.log(d["odds"]), logit(d["ai_win"]), logit(d["mkt_win"]), pop])
    out = {"note": "人気は確定単勝オッズの順位。回収率は確定の複勝払戻。2段目は6番人気以下だけで学習", "years": {}}
    for ty in TEST_YEARS:
        tr, te = np.isin(year, [x for x in bv.PRED_YEARS if x < ty]), year == ty
        # 合わせた確率（試す年より前の全馬で当てはめ）
        lr = LogisticRegression(C=1e3, max_iter=1000).fit(np.column_stack([logit(ai[tr]), logit(mkt[tr])]), y[tr])
        comb = lr.predict_proba(np.column_stack([logit(ai), logit(mkt)]))[:, 1]
        # 2段目：人気薄だけ。最後の学習年で木の本数を決めてから、学習年すべてで学び直す
        lt, le = tr & (pop >= LONG), te & (pop >= LONG)
        last = max(x for x in bv.PRED_YEARS if x < ty)
        fit, val = lt & (year != last), lt & (year == last)
        b = lgb.train(P2, lgb.Dataset(X2[fit], y[fit]), 3000, valid_sets=[lgb.Dataset(X2[val], y[val])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m2 = lgb.train(P2, lgb.Dataset(X2[lt], y[lt]), max(b.best_iteration, 50))
        s2 = np.full(n, np.nan)
        s2[le] = m2.predict(X2[le])
        res = {"stage2_rounds": int(b.best_iteration), "bands": {}, "longshot": {}}
        for name, (lo, hi) in BANDS.items():
            sel = te & (pop >= lo) & (pop <= hi)
            res["bands"][name] = {"n": int(sel.sum()), "actual": round(float(y[sel].mean()), 4), "ai_mean": round(float(ai[sel].mean()), 4),
                                  "mkt_mean": round(float(mkt[sel].mean()), 4), "auc_ai": auc(y[sel], ai[sel]), "auc_mkt": auc(y[sel], mkt[sel]),
                                  "auc_comb": auc(y[sel], comb[sel]), "place_roi_all": round(float(place[sel].mean()), 4)}
        scores = {"AI": ai, "市場": mkt, "合わせた確率": comb, "2段目": s2,
                  "AI÷市場": ai / mkt, "合わせた÷市場": comb / mkt, "2段目÷市場": s2 / mkt}
        ls = {"n": int(le.sum()), "actual": round(float(y[le].mean()), 4), "place_roi_all": round(float(place[le].mean()), 4),
              "logloss": {k: round(float(log_loss(y[le], np.clip(v[le], 1e-6, 1 - 1e-6))), 5) for k, v in
                          {"AI": ai, "市場": mkt, "合わせた確率": comb, "2段目": s2}.items()},
              "auc": {k: auc(y[le], v[le]) for k, v in scores.items()}, "top": {}}
        idx = np.flatnonzero(le)
        for k, v in scores.items():
            order = idx[np.argsort(-v[idx], kind="stable")]
            ls["top"][k] = {f"{int(q * 100)}%": {"n": int(round(len(order) * q)), "hit": round(float(y[order[:int(round(len(order) * q))]].mean()), 4),
                                                   "place_roi": round(float(place[order[:int(round(len(order) * q))]].mean()), 4)} for q in QS}
        res["longshot"] = ls
        out["years"][ty] = res
        print(ty, json.dumps(ls["auc"], ensure_ascii=False), file=sys.stderr, flush=True)
        (OUT / "longshot_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

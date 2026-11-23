"""10番人気以下の3着以内を当てる2段目モデルの比較（依頼者の案、2026-11-23）.

    cd scripts/ml && python longshot10_eval.py   # → data/ml/longshot10_eval.json

本体の予測は backtest_value.prepare（年ごとに前年までで学習）。確定オッズ・確定の複勝払戻。人気は確定単勝オッズの順位
試す年 2024・2025・2026（2段目は試す年より前の年だけで学習。木の本数はその前の年を検証にして決める）
比べるモデル（どれも入力は train_longshot.py と同じ：本体の全項目＋AI・市場の確率・log オッズ・人気）
- 市場：補正した市場（市場の3着以内確率の logit だけのロジスティック回帰）
- 6〜：今の☆の2段目（6番人気以下で学習）
- 10〜：10番人気以下だけで学習
- 10〜強め：10番人気以下だけで、葉を大きく・木を浅く（数が少ないので）
評価はどれも10番人気以下の馬だけ
- logloss・AUC
- 1レースで一番高い馬を1頭（10番人気以下がいるレース）：3着内率・複勝回収率
- 「2段目 ÷ 補正した市場」がしきい値以上：点数・3着内率・複勝回収率と、市場の確率の帯をそろえた基準
- 上位 5%・10%・20%（2段目の確率の順）：3着内率・複勝回収率
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
from longshot_eval import P2, logit  # noqa: E402
from train import OUT  # noqa: E402
from train_longshot import popularity, stage2_matrix  # noqa: E402

bv.PRED_YEARS = ["2022", "2023", "2024", "2025", "2026"]
TEST_YEARS = ["2024", "2025", "2026"]
P2_STRONG = {**P2, "num_leaves": 7, "min_data_in_leaf": 400, "lambda_l2": 30.0}
MODELS = {"6〜": (6, P2), "10〜": (10, P2), "10〜強め": (10, P2_STRONG)}
THRESHOLDS = [1.2, 1.5, 2.0]
QS = [0.05, 0.1, 0.2]
LONG10 = 10


def fit_predict(X2, y, year, train_mask, test_mask, ty, params):
    last = max(x for x in bv.PRED_YEARS if x < ty)
    fit, val = train_mask & (year != last), train_mask & (year == last)
    b = lgb.train(params, lgb.Dataset(X2[fit], y[fit]), 3000, valid_sets=[lgb.Dataset(X2[val], y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(b.best_iteration, 50)
    m = lgb.train(params, lgb.Dataset(X2[train_mask], y[train_mask]), rounds)
    out = np.full(len(y), np.nan)
    out[test_mask] = m.predict(X2[test_mask])
    return out, rounds


def basic(sel, y, ret) -> dict:
    n = int(sel.sum())
    if n == 0:
        return {"bets": 0}
    return {"bets": n, "hit_rate": round(float(y[sel].mean()), 4), "roi": round(float(ret[sel].sum() / (100 * n)), 4)}


def main() -> None:
    d = bv.prepare()
    pop = popularity(d)
    y = d["is_top3"].astype(float)
    year, mkt, ret = d["year"], d["mkt_top3"], d["place_ret"]
    X2 = stage2_matrix(d, pop)
    rng = np.random.default_rng(0)
    out = {"note": "10番人気以下の3着以内。評価は10番人気以下の馬だけ。回収率は確定の複勝", "years": {}}
    for ty in TEST_YEARS:
        tr, te = np.isin(year, [x for x in bv.PRED_YEARS if x < ty]), year == ty
        ev = te & (pop >= LONG10)
        lm = LogisticRegression(C=1e3, max_iter=1000).fit(logit(mkt[tr])[:, None], y[tr])
        mcal = lm.predict_proba(logit(mkt)[:, None])[:, 1]
        scores = {"市場": mcal}
        rounds = {}
        for name, (min_pop, params) in MODELS.items():
            s, r = fit_predict(X2, y, year, tr & (pop >= min_pop), te & (pop >= min_pop), ty, params)
            scores[name], rounds[name] = s, r
        res = {"horses": int(ev.sum()), "top3": int(y[ev].sum()), "base_rate": round(float(y[ev].mean()), 4),
               "roi_all": round(float(ret[ev].sum() / (100 * ev.sum())), 4), "rounds": rounds, "models": {}}
        for name, s in scores.items():
            m = {"logloss": round(float(log_loss(y[ev], np.clip(s[ev], 1e-6, 1 - 1e-6))), 5),
                 "auc": round(float(roc_auc_score(y[ev], s[ev])), 4), "mean_pred": round(float(s[ev].mean()), 4)}
            # 1レース1頭：10番人気以下で一番高い馬
            best = np.zeros(d["n"], bool)
            for g in d["gs"]:
                g = g[ev[g]]
                if len(g):
                    best[g[np.argmax(s[g])]] = True
            m["best_per_race"] = basic(best, y, ret)
            # 上位 q（その年の10番人気以下の中で）
            for q in QS:
                cut = np.quantile(s[ev], 1 - q)
                m[f"top{int(q * 100)}%"] = basic(ev & (s >= cut), y, ret)
            if name != "市場":
                for t in THRESHOLDS:
                    sel = ev & (s / mcal >= t)
                    r = bv.result(sel, ret, d["race_idx"], rng, pool=te, mkt=mkt)
                    if r.get("bets"):
                        r = {"bets": r["bets"], "hit_rate": round(r["hits"] / r["bets"], 4), "roi": round(r["roi"], 4),
                             "roi_market_matched": round(r["roi_market_matched"], 4), "roi_95": [round(v, 3) for v in r["roi_95"]]}
                    m[f"ratio≥{t}"] = r
            res["models"][name] = m
        out["years"][ty] = res
        print(ty, {k: (v["logloss"], v["auc"], v["best_per_race"].get("hit_rate")) for k, v in res["models"].items()}, file=sys.stderr, flush=True)
        (OUT / "longshot10_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

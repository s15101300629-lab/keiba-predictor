"""AI印の☆（妙味）の決め方の比較：今の「合わせた確率 ÷ 補正した市場 ≥ 1.1」と、2段目のモデルを使う形.

    cd scripts/ml && python star_eval.py   # → data/ml/star_eval.json

本体の予測は backtest_value.prepare（年ごとに前年までで学習。182項目）。確定オッズ・確定の複勝払戻
- 今：合わせた確率（AI と市場の3着以内確率の logit をロジスティック回帰）÷ 補正した市場（市場の logit だけのロジスティック回帰）
- 2段目（全馬）：本体の182項目＋AI・市場の確率・オッズ・人気から3着以内を学習（LightGBM。試す年より前の年だけ）
- 2段目（人気薄）：同じものを6番人気以下だけで学習（longshot_eval.py と同じ）。6番人気以下にだけ付く
どれも「2段目の確率 ÷ 補正した市場 ≥ しきい値」で☆。成績は複勝の回収率と、市場の確率の帯をそろえて全馬を買った基準（bv.result）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from longshot_eval import P2, logit  # noqa: E402
from train import OUT  # noqa: E402

bv.PRED_YEARS = ["2022", "2023", "2024", "2025", "2026"]
TEST_YEARS = ["2024", "2025", "2026"]
THRESHOLDS = [1.1, 1.2, 1.3, 1.5]


def stage2(X2, y, year, train_mask, test_mask, ty):
    last = max(x for x in bv.PRED_YEARS if x < ty)
    fit, val = train_mask & (year != last), train_mask & (year == last)
    b = lgb.train(P2, lgb.Dataset(X2[fit], y[fit]), 3000, valid_sets=[lgb.Dataset(X2[val], y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    m = lgb.train(P2, lgb.Dataset(X2[train_mask], y[train_mask]), max(b.best_iteration, 50))
    out = np.full(len(y), np.nan)
    out[test_mask] = m.predict(X2[test_mask])
    return out, int(b.best_iteration)


def main() -> None:
    d = bv.prepare()
    n = d["n"]
    pop = np.zeros(n, int)
    for g in d["gs"]:
        pop[g[np.argsort(d["odds"][g], kind="stable")]] = np.arange(1, len(g) + 1)
    y = d["is_top3"].astype(float)
    year, ai, mkt = d["year"], d["ai_top3"], d["mkt_top3"]
    X2 = np.column_stack([d["X"], logit(ai), logit(mkt), np.log(d["odds"]), logit(d["ai_win"]), logit(d["mkt_win"]), pop])
    rng = np.random.default_rng(0)
    out = {"note": "☆の決め方の比較。回収率は確定の複勝。roi_market_matched は市場の確率の帯をそろえて全馬を買った基準", "years": {}}
    for ty in TEST_YEARS:
        tr, te = np.isin(year, [x for x in bv.PRED_YEARS if x < ty]), year == ty
        lm = LogisticRegression(C=1e3, max_iter=1000).fit(logit(mkt[tr])[:, None], y[tr])
        mcal = lm.predict_proba(logit(mkt)[:, None])[:, 1]
        lc = LogisticRegression(C=1e3, max_iter=1000).fit(np.column_stack([logit(ai[tr]), logit(mkt[tr])]), y[tr])
        comb = lc.predict_proba(np.column_stack([logit(ai), logit(mkt)]))[:, 1]
        s_all, r_all = stage2(X2, y, year, tr, te, ty)
        s_long, r_long = stage2(X2, y, year, tr & (pop >= 6), te & (pop >= 6), ty)
        races = len(np.unique(d["race_idx"][te]))
        res = {"rounds": {"all": r_all, "long": r_long}, "races": races, "rules": {}}
        rules = {f"今（合わせた÷市場≥{t}）": te & (comb / mcal >= t) for t in THRESHOLDS}
        rules |= {f"2段目全馬÷市場≥{t}": te & (s_all / mcal >= t) for t in THRESHOLDS}
        rules |= {f"2段目人気薄÷市場≥{t}": te & (pop >= 6) & (s_long / mcal >= t) for t in THRESHOLDS}
        for name, sel in rules.items():
            r = bv.result(sel, d["place_ret"], d["race_idx"], rng, pool=te, mkt=mkt)
            if r.get("bets"):
                r["per_race"] = round(r["bets"] / races, 3)
                r["hit_rate"] = round(r["hits"] / r["bets"], 4)
                r["mean_pop"] = round(float(pop[sel].mean()), 2)
            res["rules"][name] = r
        out["years"][ty] = res
        print(ty, {k: (v.get("bets"), round(v.get("roi", 0), 3), round(v.get("roi_market_matched", 0), 3)) for k, v in res["rules"].items()},
              file=sys.stderr, flush=True)
        (OUT / "star_eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""人気薄専用の2段目モデル（AI印の☆）を学習して data/ml/model_longshot.json に書き出す.

    cd scripts/ml && python train_longshot.py   # → data/ml/model_longshot.json

- 6番人気以下（確定単勝オッズの順位）の馬だけで、3着以内を学習する（LightGBM。star_eval.py の「2段目人気薄」と同じ形）
- 入力：本体の182項目 ＋ AI の3着以内確率・市場の3着以内確率（ハービル）・log 単勝オッズ・AI の勝率・市場の勝率（どれも logit）・人気
  - AI の確率は、年ごとに前年までで学習した本体の予測（backtest_value.prepare）。アプリでは今の本体の予測を使う
- ☆：2段目の確率 ÷ 補正した市場（市場の3着以内確率の logit だけのロジスティック回帰）が THRESHOLD 以上
  - star_eval.json（2024〜2026年、確定オッズ）：1.2 以上で複勝回収率 89・86・95%（市場の確率の帯をそろえた基準 74・73・75%）
- 学習：2022〜2026年の人気薄。木の本数は 2026年を検証にして決め、全部で学び直す。補正した市場の係数も全年で当てはめる
- アプリ（app/lib/longshot.ts）との一致を確かめるため、2026年の一部のレースの入力と予測を parity に入れる（scripts/ml/parity_longshot.ts）
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
from features import FEATURES  # noqa: E402
from longshot_eval import P2, logit  # noqa: E402
from train import OUT, export_model  # noqa: E402

bv.PRED_YEARS = ["2022", "2023", "2024", "2025", "2026"]
MIN_POP = 6
THRESHOLD = 1.2
EXTRA = ["ls_ai_top3", "ls_mkt_top3", "ls_log_odds", "ls_ai_win", "ls_mkt_win", "ls_pop"]
PARITY_RACES = 40


def popularity(d) -> np.ndarray:
    pop = np.zeros(d["n"], int)
    for g in d["gs"]:
        pop[g[np.argsort(d["odds"][g], kind="stable")]] = np.arange(1, len(g) + 1)
    return pop


def stage2_matrix(d, pop) -> np.ndarray:
    """star_eval.py・longshot_eval.py と同じ並び."""
    return np.column_stack([d["X"], logit(d["ai_top3"]), logit(d["mkt_top3"]), np.log(d["odds"]), logit(d["ai_win"]), logit(d["mkt_win"]), pop])


def fit_stage2(X2, y, year, train_mask, last_year: str):
    fit, val = train_mask & (year != last_year), train_mask & (year == last_year)
    b = lgb.train(P2, lgb.Dataset(X2[fit], y[fit]), 3000, valid_sets=[lgb.Dataset(X2[val], y[val])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    rounds = max(b.best_iteration, 50)
    return lgb.train(P2, lgb.Dataset(X2[train_mask], y[train_mask]), rounds), rounds


def fit_market_cal(mkt, y, mask) -> LogisticRegression:
    return LogisticRegression(C=1e3, max_iter=1000).fit(logit(mkt[mask])[:, None], y[mask])


def main() -> None:
    d = bv.prepare()
    pop = popularity(d)
    y = d["is_top3"].astype(float)
    year = d["year"]
    X2 = stage2_matrix(d, pop)
    long = np.isin(year, bv.PRED_YEARS) & (pop >= MIN_POP)
    model, rounds = fit_stage2(X2, y, year, long, bv.PRED_YEARS[-1])
    lm = fit_market_cal(d["mkt_top3"], y, np.isin(year, bv.PRED_YEARS))
    out = export_model(model)
    out["features"] = FEATURES + EXTRA
    out.update({
        "min_pop": MIN_POP, "threshold": THRESHOLD, "rounds": rounds, "train_rows": int(long.sum()), "years": bv.PRED_YEARS,
        "market_cal": {"coef": float(lm.coef_[0][0]), "intercept": float(lm.intercept_[0])},
    })
    # 一致テスト用：2026年の最後のほうのレース（同じオッズの馬がいないレース）
    pred = model.predict(X2)
    mcal = lm.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
    parity = []
    for g in [g for g in d["gs"] if year[g[0]] == bv.PRED_YEARS[-1]][::-1]:
        if len(parity) >= PARITY_RACES:
            break
        if len(set(d["odds"][g].tolist())) < len(g):
            continue
        parity.append({"race_id": str(d["race_id"][g[0]]), "horses": [{
            "umaban": int(d["umaban"][i]), "horse": str(d["horse"][i]),
            "x": [None if np.isnan(v) else float(v) for v in d["X"][i]],
            "top3": float(d["ptop3_raw"][i]), "win": float(d["pwin_raw"][i]), "odds": float(d["odds"][i]),
            "extra": [float(v) for v in X2[i, len(FEATURES):]], "p": float(pred[i]), "market": float(mcal[i]),
        } for i in g]})
    out["parity"] = parity
    (OUT / "model_longshot.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    sel = long & (pred / mcal >= THRESHOLD)
    print(json.dumps({"rounds": rounds, "train_rows": int(long.sum()), "market_cal": out["market_cal"],
                      "star_share_in_sample": round(float(sel.sum() / long.sum()), 4), "parity_races": len(parity)}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()

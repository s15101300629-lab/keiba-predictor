"""2026年のレースを、答えを見ていない設定で予測するためのモデル一式（2026-11-24、買い方の成績の過去分）.

    cd scripts/ml && python prev_models.py   # → data/ml/prev/

今の本番のモデル（data/ml/model.json など）は 2026年のレースも学習しているので、過去の開催日の買い方の成績には使えない。
ここでは 2025年までのデータだけで、本番と同じ作り方をする：
- 本体（3着以内・勝率・連対率）：2025年までで学習。木の本数は 2025年を検証にして決める（train.py の split_rows(…, "2026")）
- メリハリの補正 T・オッズと合わせる係数：2025年の予測（2024年までで学習したモデル）で当てはめ（backtest_value.prepare と同じ）
- ☆の2段目：2022〜2025年の予測で学習（木の本数は 2025年を検証にして決める）
check_jra_day.py --model=../../data/ml/prev/model.json で使い、day_bets.ts は同じフォルダの係数を読む
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
import train  # noqa: E402
from features import FEATURES, build_dataset  # noqa: E402
from longshot_eval import logit  # noqa: E402
from train_longshot import EXTRA, MIN_POP, THRESHOLD, fit_market_cal, fit_stage2, popularity, stage2_matrix  # noqa: E402

CUT = "2026"
PREV = train.OUT / "prev"


def main_models() -> None:
    rows, _ = build_dataset(train.DATA)
    rows = [r for r in rows if r["date"] >= train.TRAIN_FROM]
    fit, val, tr, _ = train.split_rows(rows, CUT)
    for name, k in (("model", 3), ("model_win", 1), ("model_top2", 2)):
        label = lambda rs: [1 if r["finish"] is not None and r["finish"] <= k else 0 for r in rs]  # noqa: E731
        best = train.best_rounds(fit, val, label(fit), label(val))
        b = lgb.train(train.PARAMS, lgb.Dataset(train.matrix(tr), label=label(tr), feature_name=FEATURES), num_boost_round=best)
        (PREV / f"{name}.json").write_text(json.dumps(train.export_model(b), separators=(",", ":")), encoding="utf-8")
        print(name, "rounds", best, "rows", len(tr), file=sys.stderr, flush=True)


def combine_and_longshot() -> None:
    bv.PRED_YEARS = ["2022", "2023", "2024", "2025"]
    bv.OUT = PREV  # calibration.json（T は 2025年の予測で当てはめ）と予測のためおきを prev に書く
    d = bv.prepare()
    year = d["year"]
    last = year == bv.PRED_YEARS[-1]
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(np.column_stack([logit(d["ai_top3"][last]), logit(d["mkt_top3"][last])]), d["is_top3"][last])
    lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][last])[:, None], d["is_top3"][last])
    wa, wb = bv.fit_benter([g for g in d["gs"] if year[g[0]] == bv.PRED_YEARS[-1]], d["la"], d["lm"], d["is_win"])
    combine = {"fitted_year": bv.PRED_YEARS[-1], "n_features": len(FEATURES),
               "win": {"ai": float(wa), "market": float(wb), "fitted_year": bv.PRED_YEARS[-1]},
               "combined": {"ai": float(lr.coef_[0][0]), "market": float(lr.coef_[0][1]), "intercept": float(lr.intercept_[0])},
               "market_only": {"market": float(lr_m.coef_[0][0]), "intercept": float(lr_m.intercept_[0])},
               "bands": {k: list(v) for k, v in bv.LABEL_BANDS.items()}}
    (PREV / "value_combine.json").write_text(json.dumps(combine, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    pop = popularity(d)
    y = d["is_top3"].astype(float)
    X2 = stage2_matrix(d, pop)
    long = np.isin(year, bv.PRED_YEARS) & (pop >= MIN_POP)
    model, rounds = fit_stage2(X2, y, year, long, bv.PRED_YEARS[-1])
    lm = fit_market_cal(d["mkt_top3"], y, np.isin(year, bv.PRED_YEARS))
    out = train.export_model(model)
    out["features"] = FEATURES + EXTRA
    out.update({"min_pop": MIN_POP, "threshold": THRESHOLD, "rounds": rounds, "train_rows": int(long.sum()), "years": bv.PRED_YEARS,
                "market_cal": {"coef": float(lm.coef_[0][0]), "intercept": float(lm.intercept_[0])}})
    (PREV / "model_longshot.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print("combine", combine["combined"], "win", combine["win"], "longshot rounds", rounds, file=sys.stderr, flush=True)


def main() -> None:
    PREV.mkdir(parents=True, exist_ok=True)
    main_models()
    combine_and_longshot()


if __name__ == "__main__":
    main()

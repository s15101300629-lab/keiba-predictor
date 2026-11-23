"""seed を変えたモデルの平均（アンサンブル）と、新しい年ほど重く学習する（時間の重み）の比較（2026-11-23、依頼者への提案）.

    cd scripts/ml && python compare_ensemble.py   # → data/ml/ensemble_comparison.json

前年までで学習して 2024〜2026年を予測（tune_params.evaluate と同じ作り。木の本数は前の年を検証にして決める）
- 今：seed 42 の1本
- 平均5本：seed 42〜46 の5本（木の本数は今と同じ）の確率の平均
- 時間の重み d：学習の行の重み = d^(試す年 − 1 − その行の年)（試す年の前の年が 1）。d = 0.9・0.8・0.7。木の本数も同じ重みで決め直す
- 両方が効いたら、合わせた形は後で比べる
logloss・AUC は、レースごとに3着以内確率の合計を3にそろえた後
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import normalize_sum, race_top3_hit  # noqa: E402

TEST_YEARS = ["2024", "2025", "2026"]
SEEDS = [42, 43, 44, 45, 46]
DECAYS = [0.9, 0.8, 0.7]


def fit(X, y, w, fit_m, val_m, tr_m, params, rounds=None):
    if rounds is None:
        d = lgb.Dataset(X[fit_m], y[fit_m], weight=None if w is None else w[fit_m])
        b = lgb.train(params, d, 6000, valid_sets=[lgb.Dataset(X[val_m], y[val_m], reference=d)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        rounds = max(b.best_iteration, 50)
    m = lgb.train(params, lgb.Dataset(X[tr_m], y[tr_m], weight=None if w is None else w[tr_m]), rounds)
    return m, rounds


def score(y, p, race_id, finish) -> dict:
    pn = normalize_sum(race_id, p)
    return {"logloss_sum3": float(log_loss(y, pn)), "auc_sum3": float(roc_auc_score(y, pn)), "top3_hit": race_top3_hit(race_id, y, p),
            "top1_win": float(np.mean([finish[race_id == r][np.argmax(p[race_id == r])] == 1 for r in np.unique(race_id)]))}


def main() -> None:
    rows, _ = features.build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    X = np.array([[r[f] for f in features.FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], float)
    year = np.array([r["date"][:4] for r in rows]).astype(int)
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    del rows
    out = {"note": "前年までで学習して各年を予測。logloss・AUC は合計3にそろえた後", "years": {}}
    for ty in map(int, TEST_YEARS):
        fit_m, val_m, tr_m, te = year < ty - 1, year == ty - 1, year < ty, year == ty
        res = {}
        preds = {}
        m, rounds = fit(X, y, None, fit_m, val_m, tr_m, PARAMS)
        preds["今"] = m.predict(X[te])
        ens = [preds["今"]]
        for s in SEEDS[1:]:
            ms, _ = fit(X, y, None, fit_m, val_m, tr_m, {**PARAMS, "seed": s}, rounds)
            ens.append(ms.predict(X[te]))
        preds["平均5本"] = np.mean(ens, axis=0)
        for d in DECAYS:
            # 本数を決めるとき（前の年が検証）は、検証の年の前の年を1にする
            w_fit = d ** np.clip(ty - 2 - year, 0, None).astype(float)
            w_tr = d ** np.clip(ty - 1 - year, 0, None).astype(float)
            dd = lgb.Dataset(X[fit_m], y[fit_m], weight=w_fit[fit_m])
            b = lgb.train(PARAMS, dd, 6000, valid_sets=[lgb.Dataset(X[val_m], y[val_m], reference=dd)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            r_d = max(b.best_iteration, 50)
            md = lgb.train(PARAMS, lgb.Dataset(X[tr_m], y[tr_m], weight=w_tr[tr_m]), r_d)
            preds[f"時間の重み{d}"] = md.predict(X[te])
            res[f"時間の重み{d}_rounds"] = r_d
        for k, p in preds.items():
            res[k] = score(y[te], p, race_id[te], finish[te])
        res["rounds"] = rounds
        out["years"][str(ty)] = res
        print(ty, {k: round(v["logloss_sum3"], 5) for k, v in res.items() if isinstance(v, dict)}, file=sys.stderr, flush=True)
        (OUT / "ensemble_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    names = [k for k, v in out["years"][TEST_YEARS[0]].items() if isinstance(v, dict)]
    out["mean"] = {k: {m: float(np.mean([out["years"][t][k][m] for t in TEST_YEARS])) for m in out["years"][TEST_YEARS[0]][k]} for k in names}
    out["years_better_than_now"] = {k: sum(out["years"][t][k]["logloss_sum3"] < out["years"][t]["今"]["logloss_sum3"] for t in TEST_YEARS) for k in names}
    (OUT / "ensemble_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

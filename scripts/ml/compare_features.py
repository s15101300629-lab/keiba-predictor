"""特徴量セットの比較: 各年を1年ずつ外して学習 → 外した年で評価（logloss・AUC・上位3頭の的中率）.

    cd scripts/ml && python compare_features.py

追加した特徴量が本当に効くかを、1年分の検証より頑健に確かめるために使う。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

PACE = ["pos_first_mean", "pos_last_mean", "nige_rate", "senko_rate", "hp_n", "hp_top3", "sp_n", "sp_top3",
        "field_nige_cands", "field_front", "field_front_ratio", "rel_pos_rank"]
SETS = {"without_pace": [f for f in FEATURES if f not in PACE], "with_pace": FEATURES}


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {}
    for name, feats in SETS.items():
        per_year = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            y = [r["label"] for r in te]
            per_year[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per_year[year], file=sys.stderr)
        result[name] = {"per_year": per_year,
                        "mean": {k: float(np.mean([v[k] for v in per_year.values()])) for k in ("logloss", "auc", "top3_hit")}}
    print(json.dumps({k: v["mean"] for k, v in result.items()}, indent=1), file=sys.stderr)
    (OUT / "feature_comparison.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

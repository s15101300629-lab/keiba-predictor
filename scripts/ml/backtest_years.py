"""年ごとの的中率（前年までで学習して、その年を予測する。評価する年は学習に使わない）.

    cd scripts/ml && python backtest_years.py   # → data/ml/backtest_years.json

- 今の特徴量（FEATURES）。木の本数は年ごとに、その前年を検証にして決める（2026-11-21 から。前は metrics.json の本数を全部の年に使っていた）
- 2023年は、それより前の学習データ（2022年は過去走の蓄積用）がないので対象外
- 指標：AUC・logloss、上位3頭の3着内率（人気上位3頭と比べる）、1位の馬の勝率・3着内率
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, best_rounds, matrix, race_top3_hit, split_rows  # noqa: E402


def top1(rows, score):
    by = {}
    for r, s in zip(rows, score):
        by.setdefault(r["race_id"], []).append((s, r))
    n = win = place = 0
    for items in by.values():
        items.sort(key=lambda t: -t[0])
        n += 1
        win += items[0][1]["finish"] == 1
        place += items[0][1]["label"]
    return win / n, place / n, n


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    result = {"rounds": {}, "per_year": {}}
    for year in ["2024", "2025", "2026"]:
        # 木の本数は、評価する年の前年を検証にして決める（評価する年を本数の決定に使わない。train.py の split_rows・best_rounds と同じ）
        fit, val, tr, te = split_rows(rows, year)
        rounds = best_rounds(fit, val, [r["label"] for r in fit], [r["label"] for r in val])
        result["rounds"][year] = rounds
        b = lgb.train(PARAMS, lgb.Dataset(matrix(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
        p = b.predict(matrix(te))
        y = [r["label"] for r in te]
        pop = [-(r["popularity"] or 99) for r in te]
        w, pl, n = top1(te, p)
        pw, ppl, _ = top1(te, pop)
        result["per_year"][year] = {
            "races": n, "auc": roc_auc_score(y, p), "logloss": log_loss(y, p),
            "top3_hit": race_top3_hit(te, p), "top3_hit_popularity": race_top3_hit(te, pop),
            "top1_win": w, "top1_place": pl, "top1_win_popularity": pw, "top1_place_popularity": ppl,
        }
        print(year, result["per_year"][year], file=sys.stderr, flush=True)
    (OUT / "backtest_years.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

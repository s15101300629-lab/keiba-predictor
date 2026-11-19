"""成績表の範囲（過去走の蓄積・学習に使う年）の比較.

    cd scripts/ml && python compare_range.py   # → data/ml/range_comparison.json

- A：今と同じ（過去走は 2022年から、学習は 2023年から）
- B：過去走は 2019年から、学習は 2023年から（昔の走が過去走に入る効果）
- C：過去走は 2019年から、学習は 2021年から
- D：過去走は 2019年から、学習は 2020年から
- 評価は前年までで学習して翌年を予測（2024・2025・2026年）。木の本数は metrics.json の値で固定
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
from train import DATA, OUT, PARAMS, matrix, race_top3_hit  # noqa: E402

_orig_load = features.load_races


def dataset(history_from: str):
    features.load_races = lambda root: [r for r in _orig_load(root) if r["date"] >= history_from]
    try:
        rows, _ = features.build_dataset(DATA)
    finally:
        features.load_races = _orig_load
    return rows


def main() -> None:
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    variants = {"A 過去走2022〜・学習2023〜": ("2022-01-01", "2023-01-01"),
                "B 過去走2019〜・学習2023〜": ("2019-01-01", "2023-01-01"),
                "C 過去走2019〜・学習2021〜": ("2019-01-01", "2021-01-01"),
                "D 過去走2019〜・学習2020〜": ("2019-01-01", "2020-01-01")}
    cache: dict[str, list] = {}
    result = {"rounds": rounds, "variants": {}}
    for name, (hist_from, train_from) in variants.items():
        rows = cache.setdefault(hist_from, dataset(hist_from))
        rows_t = [r for r in rows if r["date"] >= train_from]
        per = {}
        for year in ["2024", "2025", "2026"]:
            tr = [r for r in rows_t if r["date"][:4] < year]
            te = [r for r in rows_t if r["date"][:4] == year]
            b = lgb.train(PARAMS, lgb.Dataset(matrix(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(matrix(te))
            y = [r["label"] for r in te]
            per[year] = {"train_rows": len(tr), "auc": roc_auc_score(y, p), "logloss": log_loss(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result["variants"][name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("auc", "logloss", "top3_hit")}}
    (OUT / "range_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["variants"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

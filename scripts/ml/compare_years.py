"""成績表の年数を増やす比較（2017・2018年を足す。前年までで学習して 2024〜2026年を予測。今の本番の特徴量）.

    cd scripts/ml && python compare_years.py   # → data/ml/years_comparison.json

- 今：成績表は 2019年から（過去走・基準タイムなどの元）、学習は 2020年から
- 過去走だけ増やす：成績表は 2017年から、学習は 2020年から
- 学習も増やす：成績表は 2017年から、学習は 2018年から
評価は tune_params.evaluate（木の本数は年ごとに前年で決める）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from train import DATA, OUT, PARAMS  # noqa: E402
from tune_params import evaluate  # noqa: E402

VARIANTS = {"今（成績表2019〜・学習2020〜）": ("2019", "2020-01-01"), "過去走だけ増やす（2017〜・学習2020〜）": ("2017", "2020-01-01"),
            "学習も増やす（2017〜・学習2018〜）": ("2017", "2018-01-01")}


def main() -> None:
    out = {"variants": {}}
    built = {}
    first = None
    for name, (hist_from, train_from) in VARIANTS.items():
        if hist_from not in built:
            features.HISTORY_FROM = hist_from
            rows, _ = features.build_dataset(DATA)
            built[hist_from] = rows
            print(hist_from, len(rows), file=sys.stderr, flush=True)
        rows = [r for r in built[hist_from] if r["date"] >= train_from]
        X = np.array([[r[f] for f in features.FEATURES] for r in rows], dtype=np.float64)
        y = np.array([r["label"] for r in rows], dtype=float)
        year = np.array([r["date"][:4] for r in rows])
        race_id = np.array([r["race_id"] for r in rows])
        finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X, y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < first[t]["logloss_sum3"] for t in ["2024", "2025", "2026"])
        out["variants"][name] = res
        (OUT / "years_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

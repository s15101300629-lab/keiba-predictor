"""調教評価の過去分（2020〜2024年、2026-11-25 に取得）を入れたときの比較.

    cd scripts/ml && python cache_dataset.py && python compare_training_past.py   # → data/ml/training_past.json

- 今の形：調教評価は 2025年からだけ（それより前の行の train_rank・train_rank_rel・train_rank_pos は欠損）
- 全年：取った 2020年〜の評価をそのまま使う（2018・2019年は欠損）
前年までで学習して 2024〜2026年を予測（compare_round4.fit_year と同じ。2026-11-21 以降は使わない）。全体・初出走で比べる
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_round4 import YEARS, fit_year, load, metrics, write  # noqa: E402
from compare_siblings import subset_metrics  # noqa: E402
from train import PARAMS  # noqa: E402

TRAIN_COLS = ["train_rank", "train_rank_rel", "train_rank_pos"]


def main() -> None:
    d = load()
    cols = [d["features"].index(c) for c in TRAIN_COLS]
    now = d["X"].copy()
    now[np.ix_(d["date"] < "2025-01-01", cols)] = np.nan
    have = {y: round(float((~np.isnan(d["X"][d["year"] == y][:, cols[0]])).mean()), 4) for y in sorted(set(d["year"]))}
    print("評価がある割合", have, file=sys.stderr, flush=True)
    first = d["X"][:, d["features"].index("n_prior")] == 0
    rid, year, y = d["race_id"], d["year"], d["y"]
    out = {"note": __doc__.split("\n\n")[1].strip(), "coverage_by_year": have, "years": {}}
    for ty in YEARS:
        te = year == ty
        res = {}
        for name, X in (("今の形", now), ("全年", d["X"])):
            m, rounds = fit_year(X, y, year, ty, PARAMS)
            p = m.predict(X[te])
            res[name] = {"rounds": rounds, "全体": metrics(y[te], p, rid[te], d["finish"][te]), "初出走": subset_metrics(y[te], p, rid[te], first[te])}
            print(ty, name, json.dumps(res[name], ensure_ascii=False), file=sys.stderr, flush=True)
        out["years"][ty] = res
        write("training_past.json", out)


if __name__ == "__main__":
    main()

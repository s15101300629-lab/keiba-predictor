"""馬名の「（12頭）」を外す修正（2026-11-25、features.clean_horse_name）の前と後の比較.

    cd scripts/ml && python cache_dataset.py && python compare_name_fix.py   # → data/ml/name_fix.json

直した後の表は data/ml/cache/dataset.npz（cache_dataset.py）。直す前の表は、clean_horse_name を何もしない関数に置き換えてここで作る。
どちらも前年までで学習して 2024〜2026年を予測（compare_round4.fit_year と同じ。2026-11-21 以降は使わない）。
3着以内・連対・勝率の logloss（合計 3・2・1 にそろえた後）・AUC・的中率と、入力が変わった行の数
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from compare_distill import TARGETS, metrics  # noqa: E402
from compare_round4 import HOLDOUT, YEARS, fit_year, load, write  # noqa: E402
from train import DATA, PARAMS, TRAIN_FROM, matrix  # noqa: E402


def unfixed():
    features.clean_horse_name = lambda name: name  # 直す前と同じ（馬名をそのまま使う）
    rows, _ = features.build_dataset(DATA)
    rows = [r for r in rows if TRAIN_FROM <= r["date"] < HOLDOUT]
    return matrix(rows), np.array([r["race_id"] for r in rows])


def main() -> None:
    d = load()
    Xb, rid_b = unfixed()
    assert len(rid_b) == len(d["race_id"]) and (rid_b == d["race_id"]).all(), "行の並びが違う"
    diff = ~((d["X"] == Xb) | (np.isnan(d["X"]) & np.isnan(Xb)))
    changed = diff.any(1)
    out = {"note": __doc__.split("\n\n")[2].strip(), "rows": int(len(changed)), "rows_changed": int(changed.sum()),
           "rows_changed_by_year": {y: int(changed[d["year"] == y].sum()) for y in YEARS},
           "features_changed_most": sorted(((f, int(diff[:, k].sum())) for k, f in enumerate(d["features"])), key=lambda x: -x[1])[:15],
           "years": {}}
    print("入力が変わった行", out["rows_changed"], out["rows_changed_by_year"], file=sys.stderr, flush=True)
    year, rid, fin = d["year"], d["race_id"], d["finish"]
    for ty in YEARS:
        te = year == ty
        res = {}
        for target, k in TARGETS.items():
            y = (fin <= k).astype(float)
            for name, X in (("直す前", Xb), ("直した後", d["X"])):
                m, rounds = fit_year(X, y, year, ty, PARAMS)
                res.setdefault(target, {})[name] = {"rounds": rounds, **metrics(y[te], m.predict(X[te]), rid[te], k),
                                                    "changed_rows": int(changed[te].sum())}
                print(ty, target, name, res[target][name], file=sys.stderr, flush=True)
        out["years"][ty] = res
        write("name_fix.json", out)
    for target in TARGETS:
        out.setdefault("improved_years", {})[target] = sum(out["years"][t][target]["直した後"]["logloss_sum"] < out["years"][t][target]["直す前"]["logloss_sum"] for t in YEARS)
    write("name_fix.json", out)


if __name__ == "__main__":
    main()

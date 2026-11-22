"""採用した2つ（距離で重み付けした能力指数 8項目・血統の追加 5項目）を合わせた確かめ（前年までで学習して 2024〜2026年を予測）.

    cd scripts/ml && python compare_abdist_ped2.py   # → data/ml/feature_comparison_abdist_ped2.json
基準は 196項目（features.FEATURES_BEFORE_ABDIST）、比べるのは本番の 209項目（features.FEATURES）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, FEATURES_BEFORE_ABDIST, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    X = np.array([[r[f] for f in FEATURES] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(196)": FEATURES_BEFORE_ABDIST, f"本番({len(FEATURES)})": FEATURES}
    out = {"sets": {}}
    first = None
    for name, fs in sets.items():
        idx = [FEATURES.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < first[t]["logloss_sum3"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_abdist_ped2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

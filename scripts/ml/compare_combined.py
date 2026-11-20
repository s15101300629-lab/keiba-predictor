"""良かった追加を合わせたときの比較.

    cd scripts/ml && python compare_combined.py [k]   # → data/ml/feature_comparison_combined.json（k は平滑化の強さ。省略で平滑化なし）

- 能力指数（馬場差＋斤量の補正、集計、レース内の比較）と推定上がり：compare_ability.py の ability.npz
- 前走レベル・直接対決・似たコース：compare_candidates.py の candidates.npz（行の並びは同じ。race_id で確認する）
- 3着内率の平滑化：compare_smoothing.py の smooth（置き換え）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ability import agari_columns, columns  # noqa: E402
from compare_candidates import EXTRA  # noqa: E402
from compare_smoothing import PAIRS, smooth  # noqa: E402
from features import FEATURES_BEFORE_ABILITY as FEATURES  # noqa: E402
from train import OUT, PARAMS  # noqa: E402
from tune_params import evaluate  # noqa: E402


def main() -> None:
    k = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    a = np.load(OUT / "cache" / "ability.npz")
    c = np.load(OUT / "cache" / "candidates.npz")
    assert (a["race_id"] == c["race_id"]).all()
    acol, ccol = list(a["features"]), list(c["features"])
    base = a["X"][:, [acol.index(f) for f in FEATURES]]
    if k > 0:
        rate_idx = [FEATURES.index(r) for r, _, _ in PAIRS]
        base[:, rate_idx] = smooth(base, FEATURES, k)
    ab = a["X"][:, [acol.index(f) for f in columns("ab2") + agari_columns()]]
    cand = c["X"][:, [ccol.index(f) for f in EXTRA]]
    y, race_id, finish = a["label"].astype(float), a["race_id"], a["finish"]
    year = np.array([s[:4] for s in a["date"]])
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    sets = {"base": base, "+能力+推定上がり": np.hstack([base, ab]), "+能力+推定上がり+前走レベル・直接対決・似たコース": np.hstack([base, ab, cand])}
    out = {"smoothing_k": k, "sets": {}}
    path = OUT / ("feature_comparison_combined.json" if k == 0 else f"feature_comparison_combined_k{int(k)}.json")
    for name, X in sets.items():
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X, y, year, race_id, finish, params)
        if name != "base":
            res["years_logloss_better"] = sum(res[t]["logloss"] < out["sets"]["base"][t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""最終の組み合わせの比較（能力指数＋推定上がり＋4つの指数 に、前走レベル・直接対決・似たコースを足すか）.

    cd scripts/ml && python compare_final.py   # → data/ml/feature_comparison_final.json
    （先に cache_dataset / compare_ability build / compare_candidates build / compare_ability2 build）

評価：前年までで学習して 2024〜2026年を予測（tune_params.evaluate）。パラメータは param_tuning.json の調整後
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_ability import CACHE as AB_CACHE  # noqa: E402
from compare_ability import agari_columns, columns  # noqa: E402
from compare_ability2 import CACHE as AB2_CACHE  # noqa: E402
from compare_ability2 import COND, GROWTH, PACE, REL  # noqa: E402
from compare_candidates import CACHE as CAND_CACHE  # noqa: E402
from compare_candidates import COURSE, H2H, LEVEL  # noqa: E402
from features import FEATURES_BEFORE_ABILITY as FEATURES  # noqa: E402
from train import OUT, PARAMS  # noqa: E402
from tune_params import evaluate  # noqa: E402


def main() -> None:
    a, b, c = np.load(AB_CACHE), np.load(AB2_CACHE), np.load(CAND_CACHE)
    assert (a["race_id"] == b["race_id"]).all() and (a["race_id"] == c["race_id"]).all()
    pick = lambda d, fs: d["X"][:, [list(d["features"]).index(f) for f in fs]]  # noqa: E731
    base = np.hstack([pick(a, FEATURES + columns("ab2") + agari_columns()), pick(b, GROWTH + COND + PACE + REL)])
    y, race_id, finish = a["label"].astype(float), a["race_id"], a["finish"]
    year = np.array([s[:4] for s in a["date"]])
    params = dict(PARAMS, **json.loads((OUT / "param_tuning.json").read_text())["tuning"]["best_params"])
    sets = {"基準(能力指数＋推定上がり＋4つの指数)": base,
            "+似たコース": np.hstack([base, pick(c, COURSE)]),
            "+前走レベル・直接対決": np.hstack([base, pick(c, LEVEL + H2H)]),
            "+全部": np.hstack([base, pick(c, LEVEL + H2H + COURSE)])}
    out = {"sets": {}}
    first = None
    for name, X in sets.items():
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X, y, year, race_id, finish, params)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss"] < first[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_final.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

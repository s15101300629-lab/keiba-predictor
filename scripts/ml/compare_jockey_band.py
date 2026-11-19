"""騎手×距離帯の比較（1年ずつ外す）.

    cd scripts/ml && python compare_jockey_band.py   # → data/ml/feature_comparison_jockey_band.json

- 騎手の、同じ芝ダ×距離帯（〜1400 / 〜1800 / 〜2200 / 2201〜m）での騎乗数と3着内率。競馬場はまたいで数える
- 集計の期間：今年（ほかの騎手の項目と同じ） / 直近365日
"""

from __future__ import annotations

import bisect
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, norm_jockey  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")


def band(distance) -> int | None:
    if distance is None:
        return None
    return 0 if distance <= 1400 else 1 if distance <= 1800 else 2 if distance <= 2200 else 3


def main() -> None:
    rows, hist = build_dataset(DATA)
    rides = {k: sorted(v, key=lambda r: r["date"]) for k, v in hist.jockeys.items()}
    dates = {k: [r["date"] for r in v] for k, v in rides.items()}
    for r in rows:
        key = norm_jockey(r.get("jockey"))
        rs, ds = rides.get(key, []), dates.get(key, [])
        end = bisect.bisect_left(ds, r["date"])
        b = band(r["distance"])
        starts = {
            "year": bisect.bisect_left(ds, r["date"][:4] + "-01-01"),
            "365": bisect.bisect_left(ds, (date.fromisoformat(r["date"]) - timedelta(days=365)).isoformat()),
        }
        for name, start in starts.items():
            sel = [x for x in rs[start:end] if x["surface"] == r["surface"] and band(x["distance"]) == b]
            r[f"jockey_band_{name}_n"] = len(sel)
            r[f"jockey_band_{name}_top3"] = sum(x["top3"] for x in sel) / len(sel) if sel else NAN
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    sets = {"base": FEATURES,
            "+騎手×距離帯(今年)": FEATURES + ["jockey_band_year_n", "jockey_band_year_top3"],
            "+騎手×距離帯(直近365日)": FEATURES + ["jockey_band_365_n", "jockey_band_365_top3"]}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            bst = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = bst.predict(m(te))
            y = [r["label"] for r in te]
            per[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result[name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_jockey_band.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

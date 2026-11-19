"""重賞成績の比較（1年ずつ外す）.

    cd scripts/ml && python compare_graded.py   # → data/ml/feature_comparison_graded.json

- その馬の過去の重賞（成績表のレース名の（ＧⅠ）（ＧⅡ）（ＧⅢ））での出走数と、勝率・連対率・3着内率
- 比べるもの：今のモデル / +出走数と3着内率 / +出走数と勝率・連対率・3着内率
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")
GRADE = re.compile(r"[（(]\s*[GＧ]\s*[ⅠⅡⅢ123１２３]\s*[)）]")


def main() -> None:
    graded = set()
    for path in sorted((DATA / "json").glob("*/*.json")):
        day = json.loads(path.read_text(encoding="utf-8"))
        for r in day["races"]:
            if r.get("race_id") and GRADE.search(r.get("race_name") or ""):
                graded.add(r["race_id"])
    print("graded races", len(graded), file=sys.stderr)
    rows, _ = build_dataset(DATA)
    by_horse: dict[str, list[dict]] = {}
    for r in rows:
        by_horse.setdefault(r["horse"], []).append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        prev: list[dict] = []
        for r in runs:
            n = len(prev)
            r["graded_n"] = n
            fin = [p["finish"] for p in prev]
            r["graded_win"] = sum(1 for f in fin if f == 1) / n if n else NAN
            r["graded_top2"] = sum(1 for f in fin if f is not None and f <= 2) / n if n else NAN
            r["graded_top3"] = sum(1 for f in fin if f is not None and f <= 3) / n if n else NAN
            if r["race_id"] in graded:
                prev.append(r)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    sets = {"base": FEATURES, "+重賞(出走数・3着内率)": FEATURES + ["graded_n", "graded_top3"],
            "+重賞(出走数・勝率・連対率・3着内率)": FEATURES + ["graded_n", "graded_win", "graded_top2", "graded_top3"]}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            y = [r["label"] for r in te]
            per[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result[name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_graded.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

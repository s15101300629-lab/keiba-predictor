"""血統の道悪補正に使う祖先の範囲（data/pedigree/heavy_variants.csv の各列）を比べる.

    cd scripts/ml && python compare_pedigree_depth.py

train_stack.py と同じ条件（稍重〜不良・重不良未経験・年を外した1段目確率）で、
各パターンの β を「年を外して推定 → 外した年で評価」し、対数損失の合計を比べる。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix  # noqa: E402
from train_stack import heavy_adjustment  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VARIANTS = REPO / "data" / "pedigree" / "heavy_variants.csv"
FEATS = REPO / "data" / "pedigree" / "graded_features.csv"


def main() -> None:
    variants = list(csv.DictReader(VARIANTS.open(encoding="utf-8")))
    names = [c for c in variants[0] if c not in ("race_id", "umaban")]
    data = list(csv.DictReader(FEATS.open(encoding="utf-8")))
    keys = {(r["race_id"], int(r["umaban"])) for r in data}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    base, context = {}, {}
    for year in sorted({r["date"][:4] for r in data}):
        train = [r for r in rows if r["date"][:4] != year]
        target = [r for r in rows if r["date"][:4] == year and (r["race_id"], int(r["umaban"])) in keys]
        booster = lgb.train(PARAMS, lgb.Dataset(matrix(train), label=[r["label"] for r in train]), num_boost_round=rounds)
        for r, p in zip(target, booster.predict(matrix(target))):
            k = (r["race_id"], int(r["umaban"]))
            base[k] = float(p)
            context[k] = (r["going"], r["heavy_n"], r["wet_n"])
    data = [r for r in data if (r["race_id"], int(r["umaban"])) in base]
    result = {}
    for name in names:
        # 全パターンで同じ馬を比べる: 今の方式(v0)で倍率が出る馬。そのパターンで出ない馬は 1倍（補正なし）
        ratios = {(v["race_id"], int(v["umaban"])): float(v[name]) if v[name] else 1.0 for v in variants if v[names[0]]}
        res = heavy_adjustment(data, base, context, ratios)
        result[name] = {k: res[k] for k in ("beta", "chosen", "runners", "mle_beta", "mle_ci95", "cv_logloss", "years_improved")}
        print(name, json.dumps(result[name], ensure_ascii=False), file=sys.stderr)
    (REPO / "data" / "pedigree" / "depth_comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

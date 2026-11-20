"""相談中の特徴量（前走レベル・直接対決・似たコース）を、今のモデルと正直な検証で比べ直す.

    cd scripts/ml && python compare_candidates.py build   # 特徴量を作って data/ml/cache/candidates.npz に保存
    cd scripts/ml && python compare_candidates.py eval    # → data/ml/feature_comparison_candidates.json

- 列の定義は compare_level.py（前走レベル・直接対決）と compare_similar_course.py（コースの数値・似たコースでの成績）と同じ
- 評価：前年までで学習して 2024〜2026年を予測（木の本数は前年を検証にして決める。tune_params.evaluate）
- パラメータは data/ml/param_tuning.json の調整後があればそれ、なければ今の PARAMS
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES_BEFORE_ABILITY as FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402

CACHE = OUT / "cache" / "candidates.npz"
LEVEL = ["lv_field_form", "lv_field_tidx", "lv_after_n", "lv_after_top3", "lv_beaten_form", "lv_lost_form"]
H2H = ["h2h_n", "h2h_ahead", "h2h_adj", "h2h_margin"]
COURSE = ["c_straight", "c_elev", "c_lap", "sim_straight_n", "sim_straight_top3", "band_straight_n", "band_straight_top3",
          "band_elev_n", "band_elev_top3", "band_both_n", "band_both_top3"]
EXTRA = LEVEL + H2H + COURSE


def build() -> None:
    from compare_level import add_level_features
    from compare_similar_course import add_course_features

    rows, _ = build_dataset(DATA)
    add_level_features(rows)
    add_course_features(rows)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + EXTRA
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE, X=np.array([[r[f] for f in cols] for r in rows], dtype=np.float64), features=np.array(cols),
        label=np.array([r["label"] for r in rows]),
        finish=np.array([r["finish"] if r["finish"] is not None else 99 for r in rows]),
        date=np.array([r["date"] for r in rows]), race_id=np.array([r["race_id"] for r in rows]))
    print(f"wrote {CACHE} rows {len(rows)}", file=sys.stderr)


def evaluate_sets() -> None:
    from tune_params import evaluate

    d = np.load(CACHE)
    cols = list(d["features"])
    X, y, race_id, finish = d["X"], d["label"].astype(float), d["race_id"], d["finish"]
    year = np.array([s[:4] for s in d["date"]])
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    sets = {"base": FEATURES, "+前走レベル": FEATURES + LEVEL, "+直接対決": FEATURES + H2H, "+似たコース": FEATURES + COURSE,
            "+全部": FEATURES + EXTRA}
    out = {"params": {k: v for k, v in params.items() if k not in ("verbose", "deterministic", "num_threads")}, "sets": {}}
    for name, feats in sets.items():
        idx = [cols.index(f) for f in feats]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, params)
        if name != "base":
            base = out["sets"]["base"]
            res["years_logloss_better"] = sum(res[t]["logloss"] < base[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_candidates.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build() if sys.argv[1] == "build" else evaluate_sets()

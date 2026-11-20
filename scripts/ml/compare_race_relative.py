"""レース内の比較の検証（出走馬どうしの相対的な強さを学ばせる）.

    cd scripts/ml && python compare_race_relative.py   # → data/ml/race_relative_comparison.json

比べるもの（評価は tune_params.evaluate と同じ。前年までで学習して 2024〜2026年を予測）
- base：今の特徴量。あわせて「レース内で確率の合計を 3 にそろえる」補正（logloss_sum3）も出る
- +相対：重要度の高い項目のうち、まだ出走馬との比較がないものについて、出走馬平均との差と順位÷頭数を足す
- 順位学習の目的関数（xendcg。1着=3・2着=2・3着=1・他=0）は compare_ranker.py で差がなかったので、ここでは扱わない
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT, PARAMS  # noqa: E402
from tune_params import evaluate  # noqa: E402

# すでに出走馬との比較になっている項目（form_dev・rel_*・rank_*・train_rank_*・field_*）は除く
RELATIVE = ["last1_finish_c", "jockey_top3", "form_finish_mean", "form_time_mean", "surface_top3", "trainer_top3", "ped_top3",
            "jockey_venue_top3", "dist_top3", "days_since_last", "tidx_last1", "venue_top3", "jockey_course_top3",
            "trainer_venue_top3", "weight", "age", "body_weight"]


def race_relative(X, race_id, cols):
    """各列について、出走馬平均との差と、順位（大きい順）÷頭数. 欠損は平均・順位の計算から外す."""
    _, inv = np.unique(race_id, return_inverse=True)
    n_r = np.bincount(inv)
    srt = np.argsort(inv, kind="stable")
    groups = np.split(srt, np.flatnonzero(np.diff(inv[srt])) + 1)
    diff, rank = [], []
    for j in cols:
        v = X[:, j]
        ok = ~np.isnan(v)
        s = np.bincount(inv[ok], weights=v[ok], minlength=len(n_r))
        c = np.bincount(inv[ok], minlength=len(n_r))
        mean = np.where(c > 0, s / np.maximum(c, 1), np.nan)
        diff.append(np.where(ok, v - mean[inv], np.nan))
        # 順位：同じレース内で自分より大きい値の数 + 1（同じ値は同じ順位。行の並び＝着順が漏れないように）
        r = np.full(len(v), np.nan)
        for idx in groups:
            idx = idx[ok[idx]]
            vals = v[idx]
            r[idx] = 1 + (vals[None, :] > vals[:, None]).sum(1)
        rank.append(np.where(ok, r / n_r[inv], np.nan))
    return np.column_stack(diff + rank)


def main() -> None:
    d = np.load(OUT / "cache" / "dataset.npz")
    feats = list(d["features"])
    X, y, race_id, finish = d["X"], d["label"].astype(float), d["race_id"], d["finish"]
    year = np.array([s[:4] for s in d["date"]])
    cols = [feats.index(f) for f in RELATIVE if f in feats]
    missing = [f for f in RELATIVE if f not in feats]
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    out = {"relative_features": [f for f in RELATIVE if f in feats], "missing": missing, "sets": {}}
    out["sets"]["base"] = evaluate(X, y, year, race_id, finish, params)
    Xr = np.hstack([X, race_relative(X, race_id, cols)])
    res = evaluate(Xr, y, year, race_id, finish, params)
    res["years_logloss_better"] = sum(res[t]["logloss"] < out["sets"]["base"][t]["logloss"] for t in ["2024", "2025", "2026"])
    out["sets"]["+相対"] = res
    (OUT / "race_relative_comparison.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

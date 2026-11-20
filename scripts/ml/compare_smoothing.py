"""少ない出走数の3着内率をベイズ的に縮める（平滑化）検証.

    cd scripts/ml && python compare_smoothing.py   # → data/ml/feature_comparison_smoothing.json

- 平滑化：(3着内数 + k × 事前の率) ÷ (出走数 + k)。3着内数は 率 × 出走数 で戻す（データを作り直さない）
- 事前の率（段階的に縮める）
  - jockey_venue → jockey_top3、jockey_course → （平滑化した）jockey_venue、trainer_venue → trainer_top3
  - jockey_top3・trainer_top3・本人の surface/dist/venue/wet/hp/sp → 学習データ全体の平均（その列の出走数で重み付け）
  - 上位の率が欠損なら全体の平均
- k = 5・10・20・50 で、置き換えた版と元の列に足した版を比べる
- 評価：前年までで学習して 2024〜2026年を予測（tune_params.evaluate）。パラメータは param_tuning.json の調整後があればそれ
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT, PARAMS  # noqa: E402

# (率の列, 出走数の列, 事前の率の列（None=全体の平均）)。上から順に計算する（事前に使う列を先に平滑化）
PAIRS = [
    ("jockey_top3", "jockey_n", None),
    ("jockey_venue_top3", "jockey_venue_n", "jockey_top3"),
    ("jockey_course_top3", "jockey_course_n", "jockey_venue_top3"),
    ("trainer_top3", "trainer_n", None),
    ("trainer_venue_top3", "trainer_venue_n", "trainer_top3"),
    ("surface_top3", "surface_n", None),
    ("dist_top3", "dist_n", None),
    ("venue_top3", "venue_n", None),
    ("wet_top3", "wet_n", None),
    ("hp_top3", "hp_n", None),
    ("sp_top3", "sp_n", None),
]


def smooth(X: np.ndarray, feats: list[str], k: float) -> np.ndarray:
    """PAIRS の率を平滑化した列（PAIRS の順）を返す."""
    col = {f: i for i, f in enumerate(feats)}
    done: dict[str, np.ndarray] = {}
    out = []
    for rate_f, n_f, prior_f in PAIRS:
        r, n = X[:, col[rate_f]], np.nan_to_num(X[:, col[n_f]])
        ok = (n > 0) & ~np.isnan(r)
        glob = float((r[ok] * n[ok]).sum() / n[ok].sum())
        prior = np.full(len(r), glob)
        if prior_f is not None:
            p = done[prior_f]
            prior = np.where(np.isnan(p), glob, p)
        top3 = np.where(ok, r * n, 0.0)
        s = (top3 + k * prior) / (n + k)
        done[rate_f] = s
        out.append(s)
    return np.column_stack(out)


def main() -> None:
    from tune_params import evaluate

    d = np.load(OUT / "cache" / "dataset.npz")
    feats = list(d["features"])
    X, y, race_id, finish = d["X"], d["label"].astype(float), d["race_id"], d["finish"]
    year = np.array([s[:4] for s in d["date"]])
    tuned = OUT / "param_tuning.json"
    params = dict(PARAMS, **json.loads(tuned.read_text())["tuning"]["best_params"]) if tuned.exists() else PARAMS
    rate_idx = [feats.index(r) for r, _, _ in PAIRS]
    out = {"pairs": PAIRS, "sets": {}}
    path = OUT / "feature_comparison_smoothing.json"
    out["sets"]["base"] = evaluate(X, y, year, race_id, finish, params)
    for k in (5, 10, 20, 50):
        S = smooth(X, feats, k)
        Xr = X.copy()
        Xr[:, rate_idx] = S
        for name, M in ((f"置き換え k={k}", Xr), (f"追加 k={k}", np.hstack([X, S]))):
            print(name, file=sys.stderr, flush=True)
            res = evaluate(M, y, year, race_id, finish, params)
            res["years_logloss_better"] = sum(res[t]["logloss"] < out["sets"]["base"][t]["logloss"] for t in ["2024", "2025", "2026"])
            out["sets"][name] = res
            path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

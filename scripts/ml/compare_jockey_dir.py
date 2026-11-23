"""乗り替わりの「向き」を足す比較（2026-11-23、依頼者の案）.

    cd scripts/ml && python compare_jockey_dir.py   # → data/ml/feature_comparison_jockey_dir.json

今の入力は「騎手が替わったか」（jockey_change）の 0/1 だけ。前走の騎手の成績と今回の騎手の成績の差を足す
- jc_prev_top3_py：前走の騎手の、前の年の3着内率（今回のレースの年の前の年。今回の騎手の jockey_top3_py と同じ物差し）
- jc_delta_py：今回の騎手の jockey_top3_py − 前走の騎手の前の年の3着内率（同じ騎手なら 0）
- jc_delta_ty：今年（今回のレースの日より前）の3着内率の差
前走がない馬は欠損。前年までで学習して 2024〜2026年を予測（tune_params.evaluate）
"""

from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from features import norm_jockey, rate  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")


def main() -> None:
    rows, hist = features.build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    # 騎手ごと：騎乗の日付（並べたもの）と3着以内の累積
    jk = {}
    for name, rides in hist.jockeys.items():
        rides = sorted(rides, key=lambda r: r["date"])
        dates = [r["date"] for r in rides]
        cum = np.concatenate([[0], np.cumsum([r["top3"] for r in rides])])
        jk[name] = (dates, cum)

    def stats(name, lo, hi):
        """name の騎手の、日付が [lo, hi) の騎乗数と3着以内."""
        if name not in jk:
            return 0, 0
        dates, cum = jk[name]
        a, b = bisect.bisect_left(dates, lo), bisect.bisect_left(dates, hi)
        return b - a, int(cum[b] - cum[a])

    horse_runs = {h: sorted(runs, key=lambda r: r["date"]) for h, runs in hist.horses.items()}
    extra = {k: [] for k in ("jc_prev_top3_py", "jc_delta_py", "jc_delta_ty")}
    for r in rows:
        runs = horse_runs.get(r["horse"], [])
        i = bisect.bisect_left([x["date"] for x in runs], r["date"]) if runs else 0
        last = runs[i - 1] if i > 0 else None
        cur = norm_jockey(r.get("jockey"))
        if not last or not last.get("jockey") or not cur:
            for k in extra:
                extra[k].append(NAN)
            continue
        y = int(r["date"][:4])
        n_py, t_py = stats(last["jockey"], f"{y - 1}-01-01", f"{y}-01-01")
        n_ty, t_ty = stats(last["jockey"], f"{y}-01-01", r["date"])
        prev_py, prev_ty = rate(t_py, n_py), rate(t_ty, n_ty)
        extra["jc_prev_top3_py"].append(prev_py)
        extra["jc_delta_py"].append(0.0 if cur == last["jockey"] else r["jockey_top3_py"] - prev_py)
        extra["jc_delta_ty"].append(0.0 if cur == last["jockey"] else r["jockey_top3"] - prev_ty)
    X0 = np.array([[r[f] for f in features.FEATURES] for r in rows], dtype=np.float64)
    E = {k: np.array(v, dtype=np.float64) for k, v in extra.items()}
    y = np.array([r["label"] for r in rows], float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    variants = {"今": [], "+前年の差": ["jc_prev_top3_py", "jc_delta_py"], "+全部": ["jc_prev_top3_py", "jc_delta_py", "jc_delta_ty"]}
    out = {"variants": {}}
    base = None
    for name, ks in variants.items():
        X = np.column_stack([X0] + [E[k] for k in ks]) if ks else X0
        res = evaluate(X, y, year, race_id, finish, PARAMS)
        if base is None:
            base = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < base[t]["logloss_sum3"] for t in ("2024", "2025", "2026"))
        out["variants"][name] = res
        (OUT / "feature_comparison_jockey_dir.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""本番の入力の見張り（2026-11-24、依頼者の案 4）.

    cd scripts/ml && python drift.py build     # 学習データの直近1年の項目ごとの統計としきい値 → data/ml/feature_stats.json

その日の本番の入力（check_jra_day.py が作る行。アプリと同じ計算かは parity_day.ts で確かめる）の、項目ごとの
欠けの割合・平均を、学習データの直近1年と比べる（check(…)）。
- ずれ：欠けの割合の差（ポイント）と、平均の差 ÷ 標準偏差
- しきい値：基本は 欠け 10ポイント・平均 0.5。ただし、その日の条件で毎日大きく動く項目（距離・馬場・競馬場など）が毎回引っかからないように、
  直近1年の開催日ごとの同じずれの 99.5% 点がそれより大きい項目は、その値をしきい値にする（過去のデータで決め直す）
"""

from __future__ import annotations

import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT  # noqa: E402

BASE_MISSING = 0.10
BASE_MEAN_Z = 0.5
Q = 99.5
PATH = OUT / "feature_stats.json"


def day_measures(X: np.ndarray, mu: np.ndarray, sd: np.ndarray, miss: np.ndarray):
    """1日分の行 X（行×項目）の、欠けの差と平均のずれ（標準偏差の何倍）."""
    m = np.isnan(X)
    dmiss = m.mean(0) - miss
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(m.all(0), np.nan, np.nanmean(np.where(m, np.nan, X), 0))
        z = (mean - mu) / np.where(sd > 0, sd, np.nan)
    return dmiss, z


def build(X: np.ndarray, dates: np.ndarray, features: list[str]) -> dict:
    last = max(dates)
    since = (date.fromisoformat(last) - timedelta(days=365)).isoformat()
    sel = dates > since
    Y = X[sel]
    miss = np.isnan(Y).mean(0)
    with np.errstate(invalid="ignore"):
        mu, sd = np.nanmean(Y, 0), np.nanstd(Y, 0)
    lo, hi = np.nanmin(np.where(np.isnan(Y), np.inf, Y), 0), np.nanmax(np.where(np.isnan(Y), -np.inf, Y), 0)
    dm, dz = [], []
    for d in sorted(set(dates[sel])):
        a, b = day_measures(Y[dates[sel] == d], mu, sd, miss)
        dm.append(np.abs(a)); dz.append(np.abs(b))
    qm, qz = np.nanpercentile(np.array(dm), Q, axis=0), np.nanpercentile(np.array(dz), Q, axis=0)
    out = {"since": since, "until": last, "rows": int(sel.sum()), "days": len(dm), "base": {"missing": BASE_MISSING, "mean_z": BASE_MEAN_Z, "q": Q},
           "features": {}}
    for k, f in enumerate(features):
        out["features"][f] = {
            "missing": round(float(miss[k]), 5), "mean": None if math.isnan(mu[k]) else float(mu[k]), "sd": None if math.isnan(sd[k]) else float(sd[k]),
            "min": None if not math.isfinite(lo[k]) else float(lo[k]), "max": None if not math.isfinite(hi[k]) else float(hi[k]),
            "th_missing": round(float(max(BASE_MISSING, qm[k] if not math.isnan(qm[k]) else 0)), 4),
            "th_mean_z": round(float(max(BASE_MEAN_Z, qz[k] if not math.isnan(qz[k]) else 0)), 4)}
    return out


def check(X: np.ndarray, features: list[str], stats: dict | None = None) -> list[dict]:
    """その日の行のずれが大きい項目（しきい値を超えたもの）."""
    stats = stats or json.loads(PATH.read_text(encoding="utf-8"))
    fs = stats["features"]
    names = [f for f in features if f in fs]
    idx = [features.index(f) for f in names]
    mu = np.array([fs[f]["mean"] if fs[f]["mean"] is not None else np.nan for f in names])
    sd = np.array([fs[f]["sd"] if fs[f]["sd"] is not None else np.nan for f in names])
    miss = np.array([fs[f]["missing"] for f in names])
    dmiss, z = day_measures(X[:, idx], mu, sd, miss)
    out = []
    for k, f in enumerate(names):
        flags = []
        if abs(dmiss[k]) >= fs[f]["th_missing"]:
            flags.append("欠け")
        if not math.isnan(z[k]) and abs(z[k]) >= fs[f]["th_mean_z"]:
            flags.append("平均")
        if flags:
            out.append({"feature": f, "flags": flags, "missing_day": round(float(miss[k] + dmiss[k]), 3), "missing_train": fs[f]["missing"],
                        "mean_z": None if math.isnan(z[k]) else round(float(z[k]), 2),
                        "th_missing": fs[f]["th_missing"], "th_mean_z": fs[f]["th_mean_z"]})
    return sorted(out, key=lambda x: -max(abs(x["missing_day"] - x["missing_train"]) / x["th_missing"], abs(x["mean_z"] or 0) / x["th_mean_z"]))


def main() -> None:
    if sys.argv[1:2] != ["build"]:
        print(__doc__)
        return
    cache = OUT / "cache" / "dataset.npz"
    if cache.exists():
        d = np.load(cache)
        X, dates, features = d["X"], d["date"].astype(str), [str(f) for f in d["features"]]
    else:
        from features import FEATURES, build_dataset
        from train import DATA, TRAIN_FROM, matrix
        rows, _ = build_dataset(DATA)
        rows = [r for r in rows if r["date"] >= TRAIN_FROM]
        X, dates, features = matrix(rows), np.array([r["date"] for r in rows]), FEATURES
    stats = build(X, dates, features)
    PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    loose = sorted(((f, v["th_missing"], v["th_mean_z"]) for f, v in stats["features"].items()), key=lambda x: -x[2])[:15]
    print("しきい値が大きい項目（日ごとに大きく動く）:", loose, file=sys.stderr)


if __name__ == "__main__":
    main()

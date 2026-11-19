"""2段目モデル: 1段目の AI 確率 + netkeiba 戦績の特徴量（上がり・位置取り・全キャリア）→ 3着以内確率.

    npx tsx scripts/ml/history_features.ts && (cd scripts/ml && python train_history_stack.py)

1段目の確率は、そのレースの年を除いて学習したモデルの予測（train_stack.oof_base）。
年を1つずつ外して学習 → 外した年で評価し、1段目だけ（補正のみ）より良くなったときだけ
data/ml/history_stack.json の adopted を true にする。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import OUT  # noqa: E402
from train_stack import oof_base, top3_hit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "pedigree" / "history_features.csv"
HIST = ["h_agari_mean", "h_agari_last1", "h_pos_first_mean", "h_gain_mean", "h_career_n", "h_career_top3"]
FEATURES = ["base_logit", *HIST]


class Prep:
    """欠損を学習データの中央値で埋めて標準化（TypeScript 側で同じ計算をするため自前で持つ）."""

    def fit(self, X):
        self.median = np.nanmedian(X, axis=0)
        filled = np.where(np.isnan(X), self.median, X)
        self.mean = filled.mean(axis=0)
        self.scale = filled.std(axis=0)
        self.scale[self.scale == 0] = 1.0
        return self

    def transform(self, X):
        return (np.where(np.isnan(X), self.median, X) - self.mean) / self.scale


def fit_lr(X, y, cols):
    prep = Prep().fit(X[:, cols])
    return prep, LogisticRegression(C=1.0, max_iter=1000).fit(prep.transform(X[:, cols]), y)


def main() -> None:
    rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
    keys = {(r["race_id"], int(r["umaban"])) for r in rows}
    years = sorted({r["date"][:4] for r in rows})
    base, _ = oof_base(keys, years)
    rows = [r for r in rows if (r["race_id"], int(r["umaban"])) in base]

    def val(r, f):
        if f == "base_logit":
            p = min(1 - 1e-6, max(1e-6, base[(r["race_id"], int(r["umaban"]))]))
            return math.log(p / (1 - p))
        return float(r[f]) if r[f] != "" else math.nan

    X = np.array([[val(r, f) for f in FEATURES] for r in rows])
    y = np.array([int(r["label"]) for r in rows])
    race = np.array([r["race_id"] for r in rows])
    yr = np.array([r["date"][:4] for r in rows])

    variants = {"base_only": [0], "with_history": list(range(len(FEATURES)))}
    oof = {k: np.zeros(len(y)) for k in variants}
    per_year = {}
    for year in years:
        te, tr = yr == year, yr != year
        per_year[year] = {}
        for name, cols in variants.items():
            prep, lr = fit_lr(X[tr], y[tr], cols)
            oof[name][te] = lr.predict_proba(prep.transform(X[te][:, cols]))[:, 1]
            per_year[year][name] = log_loss(y[te], oof[name][te], labels=[0, 1])

    def score(p):
        return {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "race_top3_hit": top3_hit(race, y, p)}

    result = {
        "races": len(set(race)), "runners": len(y), "years": years,
        "base_only": score(oof["base_only"]), "with_history": score(oof["with_history"]),
        "per_year_logloss": per_year,
    }
    adopted = (result["with_history"]["logloss"] < result["base_only"]["logloss"]
               and result["with_history"]["race_top3_hit"] >= result["base_only"]["race_top3_hit"])
    prep, lr = fit_lr(X, y, variants["with_history"])
    result["adopted"] = adopted
    result["years_improved"] = sum(1 for v in per_year.values() if v["with_history"] < v["base_only"])
    result["coefficients"] = dict(zip(FEATURES, (round(float(c), 4) for c in lr.coef_[0])))
    print(json.dumps(result, ensure_ascii=False, indent=1), file=sys.stderr)
    (OUT / "history_stack.json").write_text(json.dumps({
        "adopted": adopted, "features": FEATURES,
        "median": prep.median.tolist(), "mean": prep.mean.tolist(), "scale": prep.scale.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "validation": result,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

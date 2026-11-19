"""2段目モデル: 1段目の AI 確率 + 血統特徴量 → 3着以内確率（ロジスティック回帰）.

    cd scripts/ml && python train_stack.py

入力: data/pedigree/graded_features.csv（npx tsx scripts/ml/pedigree_features.ts で作成）
1段目の確率は「そのレースの年を除いて学習したモデル」で予測した値を使う（結果を見ていない予測）。
レース単位の交差検証で、血統を足すと良くなる場合だけ data/ml/stack.json の adopted を true にする。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
PED = REPO / "data" / "pedigree" / "graded_features.csv"
STACK_FEATURES = ["base_logit", "ped_dev", "dist_fit", "heavy_fit"]


def top3_hit(race_ids, labels, scores) -> float:
    by: dict[str, list] = {}
    for r, y, s in zip(race_ids, labels, scores):
        by.setdefault(r, []).append((s, y))
    hits = total = 0
    for items in by.values():
        items.sort(key=lambda x: -x[0])
        hits += sum(y for _, y in items[:3])
        total += min(3, len(items))
    return hits / total


HEAVY_BETA = 1.0  # 採用値: オッズ × 血統の道悪倍率^β（β=1 = 血統の倍率どおり）


def heavy_adjustment(data, base, context) -> dict:
    """道悪未経験の馬が重・不良を走るときの補正の検証.

    対象: 重・不良の重賞で、その芝ダで重・不良の経験がない馬。
    補正: logit(p) + β·log(血統の道悪倍率)。β=1 を基準に、最尤推定値と年ごとの対数損失を記録する。
    """
    rows = []
    for r in data:
        k = (r["race_id"], int(r["umaban"]))
        going, heavy_n = context[k]
        if going is None or going < 2 or heavy_n != 0 or not r["heavy_fit"]:
            continue
        p = base[k]
        rows.append((r["date"][:4], math.log(p / (1 - p)), math.log(float(r["heavy_fit"])), int(r["label"]), r["race_id"]))
    if not rows:
        return {"beta": HEAVY_BETA, "runners": 0}
    yr = np.array([x[0] for x in rows]); z0 = np.array([x[1] for x in rows])
    x = np.array([x[2] for x in rows]); y = np.array([x[3] for x in rows])

    def nll(b, idx):
        q = 1 / (1 + np.exp(-(z0[idx] + b * x[idx])))
        return float(-np.sum(y[idx] * np.log(q) + (1 - y[idx]) * np.log(1 - q)))

    b = 0.0
    for _ in range(50):  # ニュートン法で最尤推定
        q = 1 / (1 + np.exp(-(z0 + b * x)))
        h = float(np.sum(q * (1 - q) * x ** 2))
        b += float(np.sum((y - q) * x)) / h
    se = 1 / math.sqrt(h)
    idx_all = np.arange(len(y))
    per_year = {yy: {"runners": int((yr == yy).sum()), "logloss_none": nll(0, np.where(yr == yy)[0]),
                     "logloss_adopted": nll(HEAVY_BETA, np.where(yr == yy)[0])} for yy in sorted(set(yr))}
    return {
        "beta": HEAVY_BETA, "races": len({x[4] for x in rows}), "runners": len(rows),
        "mle_beta": b, "mle_ci95": [b - 1.96 * se, b + 1.96 * se],
        "logloss_none": nll(0, idx_all), "logloss_adopted": nll(HEAVY_BETA, idx_all),
        "years_improved": sum(1 for v in per_year.values() if v["logloss_adopted"] < v["logloss_none"]),
        "per_year": per_year,
    }


def main() -> None:
    ped = list(csv.DictReader(PED.open(encoding="utf-8")))
    keys = {(r["race_id"], int(r["umaban"])) for r in ped}
    years = sorted({r["date"][:4] for r in ped})
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]

    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    base: dict[tuple[str, int], float] = {}
    context: dict[tuple[str, int], tuple] = {}
    for year in years:
        train = [r for r in rows if r["date"][:4] != year]
        target = [r for r in rows if r["date"][:4] == year and (r["race_id"], int(r["umaban"])) in keys]
        booster = lgb.train(PARAMS, lgb.Dataset(matrix(train), label=[r["label"] for r in train]), num_boost_round=rounds)
        for r, p in zip(target, booster.predict(matrix(target))):
            base[(r["race_id"], int(r["umaban"]))] = float(p)
            context[(r["race_id"], int(r["umaban"]))] = (r["going"], r["heavy_n"])
        print(f"base model without {year}: {len(target)} graded runners", file=sys.stderr)

    data = [r for r in ped if (r["race_id"], int(r["umaban"])) in base]
    for r in data:
        p = min(1 - 1e-6, max(1e-6, base[(r["race_id"], int(r["umaban"]))]))
        r["base_p"] = p
        r["base_logit"] = math.log(p / (1 - p))
    X = np.array([[float(r[f]) for f in STACK_FEATURES] for r in data])
    y = np.array([int(r["label"]) for r in data])
    groups = np.array([r["race_id"] for r in data])
    base_p = np.array([r["base_p"] for r in data])

    oof = {"base_only": np.zeros(len(y)), "with_pedigree": np.zeros(len(y))}
    for train_idx, test_idx in GroupKFold(n_splits=5).split(X, y, groups):
        for name, cols in (("base_only", [0]), ("with_pedigree", list(range(len(STACK_FEATURES))))):
            scaler = StandardScaler().fit(X[train_idx][:, cols])
            lr = LogisticRegression(C=1.0).fit(scaler.transform(X[train_idx][:, cols]), y[train_idx])
            oof[name][test_idx] = lr.predict_proba(scaler.transform(X[test_idx][:, cols]))[:, 1]

    def score(p):
        return {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "race_top3_hit": top3_hit(groups, y, p)}

    result = {
        "races": len(set(groups)), "runners": len(y), "years": years,
        "stage1_raw": score(base_p),
        "stage1_recalibrated": score(oof["base_only"]),
        "stage1_plus_pedigree": score(oof["with_pedigree"]),
    }
    adopted = (result["stage1_plus_pedigree"]["logloss"] < result["stage1_recalibrated"]["logloss"]
               and result["stage1_plus_pedigree"]["race_top3_hit"] >= result["stage1_recalibrated"]["race_top3_hit"])
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(C=1.0).fit(scaler.transform(X), y)
    result["adopted"] = adopted
    result["coefficients"] = dict(zip(STACK_FEATURES, (round(float(c), 4) for c in lr.coef_[0])))
    print(json.dumps(result, ensure_ascii=False, indent=1), file=sys.stderr)

    heavy = heavy_adjustment(data, base, context)
    print(json.dumps(heavy, ensure_ascii=False, indent=1), file=sys.stderr)
    (OUT / "heavy_adjustment.json").write_text(json.dumps(heavy, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    (OUT / "stack.json").write_text(json.dumps({
        "adopted": adopted, "features": STACK_FEATURES,
        "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
        "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
        "validation": result,
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

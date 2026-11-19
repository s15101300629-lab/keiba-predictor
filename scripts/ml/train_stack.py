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


HEAVY_BETA_DEFAULT = 1.0  # 基準: オッズ × 血統の道悪倍率（倍率どおり）
HEAVY_BETA_MAX = 3.0


def heavy_adjustment(data, base, context, ratios, experience: str = "heavy") -> dict:
    """道悪（稍重・重・不良）で、重・不良の経験がない馬の補正 logit(p) + β·log(血統の道悪倍率) を検証して β を決める.

    β は「ある年を外して推定 → 外した年で評価」で、β=1 と補正なしの両方に勝ったときだけ推定値を使う。
    勝てなければ β=1、β=1 も補正なしに負けるなら 0（補正しない）。
    """
    rows = []
    for r in data:
        k = (r["race_id"], int(r["umaban"]))
        going, heavy_n, wet_n = context[k]
        # 経験の判定: heavy = 重・不良を走ったことがない / wet = 稍重以上を走ったことがない
        experienced = heavy_n if experience == "heavy" else wet_n
        if going is None or going < 1 or experienced != 0 or not ratios.get(k):
            continue
        p = base[k]
        rows.append((r["date"][:4], math.log(p / (1 - p)), math.log(ratios[k]), int(r["label"]), r["race_id"], going))
    if not rows:
        return {"beta": 0.0, "runners": 0}
    yr = np.array([x[0] for x in rows]); z0 = np.array([x[1] for x in rows])
    x = np.array([x[2] for x in rows]); y = np.array([x[3] for x in rows]); going = np.array([x[5] for x in rows])

    def nll(b, idx):
        q = 1 / (1 + np.exp(-(z0[idx] + b * x[idx])))
        return float(-np.sum(y[idx] * np.log(q) + (1 - y[idx]) * np.log(1 - q)))

    def mle(idx):
        b = 0.0
        for _ in range(50):  # ニュートン法
            q = 1 / (1 + np.exp(-(z0[idx] + b * x[idx])))
            h = float(np.sum(q * (1 - q) * x[idx] ** 2))
            b += float(np.sum((y[idx] - q) * x[idx])) / h
        return b, 1 / math.sqrt(h)

    all_idx = np.arange(len(y))
    b_all, se = mle(all_idx)
    cv = {"none": 0.0, "beta1": 0.0, "estimated": 0.0}
    per_year = {}
    for yy in sorted(set(yr)):
        te, tr = np.where(yr == yy)[0], np.where(yr != yy)[0]
        b_tr = min(HEAVY_BETA_MAX, max(0.0, mle(tr)[0]))
        row = {"runners": int(len(te)), "beta_without_year": b_tr,
               "logloss_none": nll(0, te), "logloss_beta1": nll(HEAVY_BETA_DEFAULT, te), "logloss_estimated": nll(b_tr, te)}
        per_year[yy] = row
        cv["none"] += row["logloss_none"]; cv["beta1"] += row["logloss_beta1"]; cv["estimated"] += row["logloss_estimated"]
    # 推定値を使うのは、β=1 より良く、しかも全部の年で補正なしより良いときだけ（2026-11 依頼者の判断。ぶれの大きい推定値で決めない）
    est_all_years = all(v["logloss_estimated"] < v["logloss_none"] for v in per_year.values())
    if cv["estimated"] < cv["beta1"] and cv["estimated"] < cv["none"] and est_all_years:
        beta, chosen = min(HEAVY_BETA_MAX, max(0.0, b_all)), "estimated"
    elif cv["beta1"] < cv["none"]:
        beta, chosen = HEAVY_BETA_DEFAULT, "beta1"
    else:
        beta, chosen = 0.0, "none"
    by_going = {}
    for code, name in ((1, "稍重"), (2, "重"), (3, "不良")):
        idx = np.where(going == code)[0]
        if len(idx):
            by_going[name] = {"runners": int(len(idx)), "logloss_none": nll(0, idx), "logloss_adopted": nll(beta, idx)}
    return {
        "beta": beta, "chosen": chosen, "goings": ["稍重", "重", "不良"],
        "races": len({x[4] for x in rows}), "runners": len(rows),
        "mle_beta": b_all, "mle_ci95": [b_all - 1.96 * se, b_all + 1.96 * se],
        "cv_logloss": cv, "years_improved": sum(1 for v in per_year.values() if v["logloss_estimated" if chosen == "estimated" else "logloss_beta1"] < v["logloss_none"]),
        "per_year": per_year, "by_going": by_going,
    }


def oof_base(keys: set, years: list[str]):
    """重賞の出走馬について、そのレースの年を除いて学習した1段目の確率（結果を見ていない予測）.

    戻り値: base[(race_id, umaban)] = 確率, context[(race_id, umaban)] = (馬場コード, 重不良の出走数)
    """
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
            context[(r["race_id"], int(r["umaban"]))] = (r["going"], r["heavy_n"], r["wet_n"])
        print(f"base model without {year}: {len(target)} graded runners", file=sys.stderr)
    return base, context


def main() -> None:
    ped = list(csv.DictReader(PED.open(encoding="utf-8")))
    keys = {(r["race_id"], int(r["umaban"])) for r in ped}
    years = sorted({r["date"][:4] for r in ped})
    base, context = oof_base(keys, years)

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

    ratios = {(r["race_id"], int(r["umaban"])): float(r["heavy_ratio"]) for r in data if r.get("heavy_ratio")}
    heavy = heavy_adjustment(data, base, context, ratios)
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

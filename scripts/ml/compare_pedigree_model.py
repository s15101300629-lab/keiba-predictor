"""血統の特徴量を本体モデルに入れる比較（1年ずつ外す）。今の「後からの道悪補正」とも比べる.

    npx tsx scripts/ml/pedigree_all_features.ts && (cd scripts/ml && python compare_pedigree_model.py)
    → data/ml/feature_comparison_pedigree.json

- 血統の元データ：父・母父・母母父の産駒成績（レース前年まで。data/pedigree/all/runner_features.csv）
  - 道悪倍率（重・不良 ÷ 良）、稍重〜不良 ÷ 良、今回の芝ダでの産駒の3着内率、今回の距離区分の得意さ（区分の3着内率 ÷ 全区分の平均）、産駒の出走数
- 比べるもの
  - base：今のモデル
  - base+後からの補正：今のアプリと同じ。重・不良の経験がない馬が稍重〜不良を走るとき logit + 1×log(道悪倍率 50:25:12.5)
  - +血統（3頭別々）：祖先ごとの値をそのまま入れる。本人の経験（稍重以上の出走数・3着内率など）と一緒に学習するので、経験が少ないほど血統を重く見るかはモデルが決める
  - +血統（まとめ・均等）：3頭の平均（1:1:1）
  - +血統（まとめ・50:25:12.5）：今のアプリと同じ重み
- 評価：2023〜2026年を1年ずつ外す。全体と、道悪のレース・そのうち重・不良の経験なし・初出走・初ダート／初芝・初めての距離
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CSV = REPO / "data" / "pedigree" / "all" / "runner_features.csv"
NAN = float("nan")
POS = ["F", "MF", "MMF"]
KINDS = ["heavy", "wet", "top3", "dist"]
SEPARATE = [f"{p}_{k}" for p in POS for k in KINDS] + [f"{p}_starts" for p in POS]


def weighted(r, kind, w):
    s = tot = 0.0
    for p, wp in zip(POS, w):
        v = r.get(f"{p}_{kind}", NAN)
        if v == v:
            s += wp * v
            tot += wp
    return s / tot if tot else NAN


def main() -> None:
    ped = {}
    with CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ped[(r["race_id"], int(r["umaban"]))] = {k: float(v) if v != "" else NAN for k, v in r.items() if k not in ("race_id", "umaban", "horse_id")}
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    for r in rows:
        u = r["umaban"]
        p = ped.get((r["race_id"], int(u))) if u == u else None
        for c in SEPARATE:
            r[c] = p.get(c, NAN) if p else NAN
        for name, w in (("eq", (1, 1, 1)), ("421", (4, 2, 1))):
            for k in KINDS:
                r[f"ped_{k}_{name}"] = weighted(r, k, w) if p else NAN
    sets = {
        "base": FEATURES,
        "+血統（3頭別々）": FEATURES + SEPARATE,
        "+血統（まとめ・均等）": FEATURES + [f"ped_{k}_eq" for k in KINDS],
        "+血統（まとめ・50:25:12.5）": FEATURES + [f"ped_{k}_421" for k in KINDS],
    }
    subsets = {
        "all": lambda r: True,
        "wet": lambda r: r["going"] == r["going"] and r["going"] >= 1,
        "wet_no_heavy_exp": lambda r: r["going"] == r["going"] and r["going"] >= 1 and r["heavy_n"] == 0,
        "debut": lambda r: r["n_prior"] == 0,
        "first_surface": lambda r: r["n_prior"] > 0 and r["surface_n"] == 0,
        "first_distance": lambda r: r["n_prior"] > 0 and r["surface_n"] > 0 and r["dist_n"] == 0,
    }
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = ["2023", "2024", "2025", "2026"]
    coverage = {c: round(sum(1 for r in rows if r[c] == r[c]) / len(rows), 3) for c in ("F_heavy", "MF_heavy", "MMF_heavy", "ped_heavy_eq")}

    def score(rs, p):
        y = [r["label"] for r in rs]
        if not rs or len(set(y)) < 2:
            return None
        return {"rows": len(rs), "logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(rs, p)}

    def scores(te, p):
        out = {}
        for name, cond in subsets.items():
            idx = [i for i, r in enumerate(te) if cond(r)]
            sub = [te[i] for i in idx]
            # 部分集合の上位3頭の的中率は、その馬だけで並べた値になるので logloss・AUC を見る
            out[name] = score(sub, p[idx]) if idx else None
        return out

    result: dict = {"rounds": rounds, "coverage": coverage, "sets": {}}
    preds_base = {}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            per[year] = scores(te, p)
            if name == "base":
                preds_base[year] = (te, p)
            print(name, year, json.dumps(per[year]["all"]), file=sys.stderr, flush=True)
        result["sets"][name] = {"per_year": per}
    # 今のアプリの後からの補正（重・不良の経験なし × 稍重〜不良、β=1、50:25:12.5 の道悪倍率）
    per = {}
    for year, (te, p) in preds_base.items():
        q = p.copy()
        for i, r in enumerate(te):
            ratio = r["ped_heavy_421"]
            if subsets["wet_no_heavy_exp"](r) and ratio == ratio and ratio > 0:
                z = math.log(q[i] / (1 - q[i])) + math.log(ratio)
                q[i] = 1 / (1 + math.exp(-z))
        per[year] = scores(te, q)
    result["sets"]["base+後からの補正"] = {"per_year": per}

    for name, v in result["sets"].items():
        mean = {}
        for sub in subsets:
            vals = [v["per_year"][y][sub] for y in years if v["per_year"][y][sub]]
            if vals:
                mean[sub] = {k: float(np.mean([x[k] for x in vals])) for k in ("logloss", "auc", "top3_hit")} | {"rows": int(sum(x["rows"] for x in vals))}
        v["mean"] = mean
        if name != "base":
            base = result["sets"]["base"]["per_year"]
            v["years_logloss_better"] = {sub: sum(1 for y in years if v["per_year"][y][sub] and base[y][sub]
                                                  and v["per_year"][y][sub]["logloss"] < base[y][sub]["logloss"]) for sub in subsets}
    (OUT / "feature_comparison_pedigree.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

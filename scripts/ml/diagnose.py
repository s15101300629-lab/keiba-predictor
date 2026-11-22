"""AI が当たらない原因を調べる（2024〜2026年、年ごとに前年までで学習した予測。確定オッズ）.

    cd scripts/ml && python diagnose.py   # → data/ml/diagnose.json

1. 年ごと：AI・人気順・AIと市場を合わせた確率の、上位3頭の3着内率・1位の勝率と3着内率・logloss
2. レースの種類ごと（クラス・芝ダ・距離・頭数・競馬場・馬場）：同じ指標と、人気順との差
3. 馬の種類ごと（人気・過去の出走数・間隔・年齢）：AI の確率の平均と実際（ずれ）、logloss の AI と市場の差
4. 市場が知っていて AI が知らないこと：logit(補正した市場) − logit(AI) を本体の入力から当てるモデル（2024・2025年で学習、2026年で R²）と、効いている項目
5. 1日分（約24レース）での的中率のぶれ：1日の結果がどのくらい上下するのが普通か
合わせた確率・補正した市場の係数は、その年の前の年で当てはめる（backtest_value.py と同じ）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from features import FEATURES  # noqa: E402
from train import OUT  # noqa: E402

TEST = ["2024", "2025", "2026"]
col = {f: i for i, f in enumerate(FEATURES)}
VENUES = {1: "札幌", 2: "函館", 3: "福島", 4: "新潟", 5: "東京", 6: "中山", 7: "中京", 8: "京都", 9: "阪神", 10: "小倉"}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main() -> None:
    d = bv.prepare()
    n, gs, year, y = d["n"], d["gs"], d["year"], d["is_top3"].astype(float)
    X = d["X"]
    pop = np.zeros(n, int)
    for g in gs:
        pop[g[np.argsort(d["odds"][g], kind="stable")]] = np.arange(1, len(g) + 1)
    ai, mkt = d["ai_top3"], d["mkt_top3"]
    comb, mcal = np.full(n, np.nan), np.full(n, np.nan)
    for ty in TEST:
        prev, te = year == str(int(ty) - 1), year == ty
        lc = LogisticRegression(C=1e3, max_iter=1000).fit(np.column_stack([logit(ai[prev]), logit(mkt[prev])]), y[prev])
        lm = LogisticRegression(C=1e3, max_iter=1000).fit(logit(mkt[prev])[:, None], y[prev])
        comb[te] = lc.predict_proba(np.column_stack([logit(ai[te]), logit(mkt[te])]))[:, 1]
        mcal[te] = lm.predict_proba(logit(mkt[te])[:, None])[:, 1]
    test = np.isin(year, TEST)
    tgs = [g for g in gs if test[g[0]]]

    # レースごとの属性
    def race_attr(g):
        x = X[g[0]]
        prize = x[col["prize_1st"]]
        debut = bool(np.all(X[g, col["n_prior"]] == 0))
        cls = ("新馬" if debut else "未勝利" if prize <= 600 else "1勝" if prize <= 1100 else "2勝" if prize <= 1600
               else "3勝" if prize <= 2100 else "OP・L" if prize < 3800 else "重賞")
        dist = x[col["distance"]]
        fs = len(g)
        going = x[col["going"]]
        return {
            "クラス（1着賞金の帯）": cls,
            "芝ダ": "芝" if x[col["surface"]] == 0 else "ダート",
            "距離": "〜1400" if dist <= 1400 else "〜1800" if dist <= 1800 else "〜2200" if dist <= 2200 else "2201〜",
            "頭数": "〜10頭" if fs <= 10 else "11〜14頭" if fs <= 14 else "15頭〜",
            "競馬場": VENUES.get(int(x[col["venue"]]), "?"),
            "馬場": "良" if going == 0 else "稍重" if going == 1 else "重・不良" if going >= 2 else "?",
        }

    def top3_hit(g, score):
        o = g[np.argsort(-score[g], kind="stable")][:3]
        return float(y[o].sum() / 3)

    def race_metrics(groups):
        if not groups:
            return None
        idx = np.concatenate(groups)
        r = {"races": len(groups), "runners": int(len(idx))}
        for name, s in (("AI", ai), ("人気", -pop.astype(float)), ("合わせた", comb)):
            r[f"上位3頭_{name}"] = round(float(np.mean([top3_hit(g, s) for g in groups])), 4)
        r["1位の勝率_AI"] = round(float(np.mean([d["is_win"][g[np.argmax(ai[g])]] for g in groups])), 4)
        r["1位の勝率_人気"] = round(float(np.mean([d["is_win"][g[np.argmin(pop[g])]] for g in groups])), 4)
        r["logloss_AI"] = round(ll(y[idx], ai[idx]), 5)
        r["logloss_市場"] = round(ll(y[idx], mcal[idx]), 5)
        r["logloss_合わせた"] = round(ll(y[idx], comb[idx]), 5)
        r["差_上位3頭_AI−人気"] = round(r["上位3頭_AI"] - r["上位3頭_人気"], 4)
        return r

    out: dict = {"note": __doc__.split("\n")[0], "years": {}, "race_segments": {}, "runner_segments": {}}
    for ty in TEST:
        out["years"][ty] = race_metrics([g for g in tgs if year[g[0]] == ty])
    out["all"] = race_metrics(tgs)

    attrs = [race_attr(g) for g in tgs]
    for key in attrs[0]:
        seg = {}
        for v in sorted({a[key] for a in attrs}):
            seg[v] = race_metrics([g for g, a in zip(tgs, attrs) if a[key] == v])
        out["race_segments"][key] = seg

    # 馬の種類ごと
    def runner_seg(name, values, labels):
        seg = {}
        for lab, sel in zip(labels, values):
            sel = sel & test
            if sel.sum() < 200:
                continue
            seg[lab] = {"runners": int(sel.sum()), "実際": round(float(y[sel].mean()), 4), "AIの平均": round(float(ai[sel].mean()), 4),
                        "市場の平均": round(float(mcal[sel].mean()), 4), "合わせたの平均": round(float(comb[sel].mean()), 4),
                        "logloss_AI": round(ll(y[sel], ai[sel]), 5), "logloss_市場": round(ll(y[sel], mcal[sel]), 5),
                        "logloss_合わせた": round(ll(y[sel], comb[sel]), 5)}
        out["runner_segments"][name] = seg

    runner_seg("人気", [pop == 1, pop == 2, pop == 3, (pop >= 4) & (pop <= 5), (pop >= 6) & (pop <= 9), pop >= 10],
               ["1番人気", "2番人気", "3番人気", "4〜5番人気", "6〜9番人気", "10番人気〜"])
    npri = X[:, col["n_prior"]]
    runner_seg("過去の出走数", [npri == 0, npri == 1, (npri >= 2) & (npri <= 4), (npri >= 5) & (npri <= 9), npri >= 10],
               ["0（初出走）", "1", "2〜4", "5〜9", "10〜"])
    gap = X[:, col["days_since_last"]]
    runner_seg("前走からの間隔", [(gap > 0) & (gap <= 14), (gap > 14) & (gap <= 35), (gap > 35) & (gap <= 70), (gap > 70) & (gap <= 180), gap > 180],
               ["〜2週", "〜5週", "〜10週", "〜半年", "半年〜"])
    age = X[:, col["age"]]
    runner_seg("年齢", [age == 2, age == 3, age == 4, age == 5, age >= 6], ["2歳", "3歳", "4歳", "5歳", "6歳〜"])
    tr = X[:, col["train_rank"]]
    runner_seg("調教評価", [np.isnan(tr), tr == 4, tr == 3, tr == 2, tr == 1], ["なし", "A", "B", "C", "D"])

    # 4. 市場と AI の差を、本体の入力から当てる
    target = logit(mcal) - logit(ai)
    trm, tem = np.isin(year, ["2024", "2025"]), year == "2026"
    m = lgb.train({"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 100, "verbose": -1, "seed": 1},
                  lgb.Dataset(X[trm], target[trm]), 400)
    pred = m.predict(X[tem])
    r2 = 1 - np.mean((target[tem] - pred) ** 2) / np.var(target[tem])
    imp = m.feature_importance("gain")
    out["market_minus_ai"] = {
        "note": "logit(補正した市場) − logit(AI)。R² が低いほど、市場は AI の入力にない情報を使っている",
        "r2_2026": round(float(r2), 4), "sd": round(float(np.std(target[tem])), 4),
        "top_features": [[FEATURES[i], round(float(imp[i] / imp.sum()), 4)] for i in np.argsort(-imp)[:20]],
    }
    # 差の大きさごとに、実際はどちらに近いか
    q = np.quantile(target[test], [0.1, 0.9])
    for lab, sel in (("市場がAIよりずっと高い（上位10%）", test & (target >= q[1])), ("市場がAIよりずっと低い（下位10%）", test & (target <= q[0]))):
        out["market_minus_ai"][lab] = {"runners": int(sel.sum()), "実際": round(float(y[sel].mean()), 4),
                                       "AI": round(float(ai[sel].mean()), 4), "市場": round(float(mcal[sel].mean()), 4)}

    # 5. 1日（24レース）での上位3頭の3着内率のぶれ
    rng = np.random.default_rng(0)
    hits_ai = np.array([top3_hit(g, ai) for g in tgs if year[g[0]] == "2026"])
    hits_pop = np.array([top3_hit(g, -pop.astype(float)) for g in tgs if year[g[0]] == "2026"])
    sims = [rng.choice(len(hits_ai), 24) for _ in range(5000)]
    out["daily_noise"] = {
        "races_per_day": 24,
        "AI_mean": round(float(hits_ai.mean()), 4),
        "AI_5_95": [round(float(np.quantile([hits_ai[s].mean() for s in sims], q)), 4) for q in (0.05, 0.95)],
        "AI−人気_5_95": [round(float(np.quantile([(hits_ai[s] - hits_pop[s]).mean() for s in sims], q)), 4) for q in (0.05, 0.95)],
        "AI−人気_mean": round(float((hits_ai - hits_pop).mean()), 4),
    }
    (OUT / "diagnose.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("all", "daily_noise")}, ensure_ascii=False), file=sys.stderr)


if __name__ == "__main__":
    main()

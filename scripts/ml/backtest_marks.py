"""AI の印（◎○▲☆△消）を過去レースに付けて、印ごとの成績を出す（前年までで学習して、その年を予測する）.

    cd scripts/ml && python backtest_marks.py   # → data/ml/marks_backtest.json

印の決め方（上から順に。1頭に印は1つ。▲☆△消は複数頭に付いてよい）
1. ◎：AI の勝率1位。「割高」なら次の馬
2. ○：残りで AIスコア（3着以内確率）1位
3. ▲：3通りを比べる（A：AI の勝率上位4頭以内で目安の倍率1.0以上、複数可／B：勝率上位4頭以内で単勝の回収見込みが一番高い1頭／C：残りで勝率1位）
   最初の版（勝率上位4頭以内で単勝の回収見込み 0.85〜1.0以上）は、3年で十数頭しか付かなかった
4. ☆：残りで、6番人気以下かつ「人気薄専用の2段目の3着以内確率 ÷ 市場だけを補正した3着以内」が 1.2 以上（2026-11-22 から。train_longshot.py と同じ形を、
   その年より前の年だけで学習）。それまでは「目安の倍率 1.1 以上」
5. △：残りで、AIスコアが6位以内
6. 消：残りで、3番人気以内かつ「割高」（倍率 0.9 以下）
- 合わせ方の係数（単勝：条件付きロジット、3着以内：ロジスティック回帰）は、毎年その前の年で決める（backtest_value.py と同じ）
- 印ごとに：頭数、勝率、3着内率、単勝・複勝回収率、市場の確率の帯をそろえた全馬の複勝回収率（人気の偏りを除いた基準）
- 消と比べるため、3番人気以内の馬全体と、そのうち割高な馬全体の成績も出す。確定オッズなので、レース前のオッズでは少し悪くなる
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_value import TEST_YEARS, benter, fit_benter, prepare  # noqa: E402
from train_longshot import MIN_POP, THRESHOLD, fit_market_cal, fit_stage2, popularity, stage2_matrix  # noqa: E402  （PRED_YEARS を 2022年からにする）
import backtest_value as bv  # noqa: E402
from train import OUT  # noqa: E402

MARKS = ["◎", "○", "▲", "☆", "△", "消", "無印"]
VARIANTS = ["A", "B", "C"]
VALUE, EXPENSIVE = 1.1, 0.9


def assign(g, ai_win, ai_top3, win_ev, ratio, pop, variant, star) -> list[str]:
    """1レース分の印（g の並び）."""
    k = len(g)
    mark = ["無印"] * k
    used = np.zeros(k, bool)
    order_win = np.argsort(-ai_win[g], kind="stable")
    order_top3 = np.argsort(-ai_top3[g], kind="stable")
    win_rank = np.empty(k, int)
    win_rank[order_win] = np.arange(k)
    top3_rank = np.empty(k, int)
    top3_rank[order_top3] = np.arange(k)
    r = ratio[g]
    for i in order_win:  # ◎
        if r[i] > EXPENSIVE:
            mark[i] = "◎"; used[i] = True; break
    for i in order_top3:  # ○
        if not used[i]:
            mark[i] = "○"; used[i] = True; break
    # ▲：A＝勝率上位4頭以内で倍率1.0以上（複数可）、B＝勝率上位4頭以内で単勝の回収見込みが一番高い1頭、C＝勝率3位（残りの中で勝率1位）
    cand = [i for i in range(k) if not used[i] and win_rank[i] < 4]
    if variant == "A":
        for i in cand:
            if r[i] >= 1.0:
                mark[i] = "▲"; used[i] = True
    elif variant == "B" and cand:
        i = max(cand, key=lambda j: win_ev[g][j])
        mark[i] = "▲"; used[i] = True
    elif variant == "C":
        for i in order_win:
            if not used[i]:
                mark[i] = "▲"; used[i] = True; break
    for i in range(k):  # ☆（人気薄専用の2段目）
        if not used[i] and pop[g][i] >= MIN_POP and star[g][i] >= THRESHOLD:
            mark[i] = "☆"; used[i] = True
    for i in range(k):  # △
        if not used[i] and top3_rank[i] < 6:
            mark[i] = "△"; used[i] = True
    for i in range(k):  # 消
        if not used[i] and pop[g][i] <= 3 and r[i] <= EXPENSIVE:
            mark[i] = "消"; used[i] = True
    return mark


def stats(sel, d, mkt_cal, pool) -> dict:
    n = int(sel.sum())
    if not n:
        return {"n": 0}
    out = {"n": n, "per_race": float(n / len(np.unique(d["race_idx"][pool]))),
           "win_rate": float(d["is_win"][sel].mean()), "top3_rate": float(d["is_top3"][sel].mean()),
           "win_roi": float(d["win_ret"][sel].sum() / (100 * n)), "place_roi": float(d["place_ret"][sel].sum() / (100 * n))}
    edges = np.quantile(mkt_cal[pool], np.linspace(0, 1, 21)[1:-1])
    band = np.searchsorted(edges, mkt_cal)
    all_roi = np.array([d["place_ret"][pool & (band == b)].sum() / (100 * max(1, (pool & (band == b)).sum())) for b in range(20)])
    cnt = np.bincount(band[sel], minlength=20)
    out["place_roi_market_matched"] = float((cnt * all_roi).sum() / cnt.sum())
    return out


def main() -> None:
    d = prepare()
    year, gs, odds = d["year"], d["gs"], d["odds"]
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    pop = popularity(d)
    X2 = stage2_matrix(d, pop)
    y3 = d["is_top3"].astype(float)
    out = {"rules": {"value": VALUE, "expensive": EXPENSIVE, "triangle_variants": VARIANTS,
                     "star": {"min_pop": MIN_POP, "threshold": THRESHOLD, "model": "人気薄専用の2段目（train_longshot.py の形）"}}, "years": {}}
    for y in TEST_YEARS:
        prev = str(int(y) - 1)
        mp, mt = year == prev, year == y
        gp = [g for g in gs if year[g[0]] == prev]
        gt = [g for g in gs if year[g[0]] == y]
        a, b = fit_benter(gp, d["la"], d["lm"], d["is_win"])
        comb_win = benter(gs, d["la"], d["lm"], a, b, d["n"])
        X = np.column_stack([logit(d["ai_top3"]), logit(d["mkt_top3"])])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X[mp], d["is_top3"][mp])
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][mp])[:, None], d["is_top3"][mp])
        ratio = lr.predict_proba(X)[:, 1] / lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        mkt_cal = lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        win_ev = comb_win * odds
        # ☆：人気薄専用の2段目（その年より前の年だけで学習）÷ 補正した市場（その年より前の年で当てはめ）
        before = np.isin(year, [x for x in bv.PRED_YEARS if x < y])
        m2, _ = fit_stage2(X2, y3, year, before & (pop >= MIN_POP), max(x for x in bv.PRED_YEARS if x < y))
        star = np.zeros(d["n"])
        star[mt] = m2.predict(X2[mt]) / fit_market_cal(d["mkt_top3"], y3, before).predict_proba(np.log(np.clip(d["mkt_top3"][mt], 1e-6, 1 - 1e-6) / (1 - np.clip(d["mkt_top3"][mt], 1e-6, 1 - 1e-6)))[:, None])[:, 1]
        yr = {"races": len(gt), "all": stats(mt, d, mkt_cal, mt), "pop_le3": stats(mt & (pop <= 3), d, mkt_cal, mt),
              "pop_le3_expensive": stats(mt & (pop <= 3) & (ratio <= EXPENSIVE), d, mkt_cal, mt)}
        for v in VARIANTS:
            marks = np.empty(d["n"], dtype=object)
            for g in gt:
                marks[g] = assign(g, d["ai_win"], d["ai_top3"], win_ev, ratio, pop, v, star)
            yr[f"triangle_{v}"] = {m: stats(mt & (marks == m), d, mkt_cal, mt) for m in MARKS}
        out["years"][y] = yr
        print(y, json.dumps({m: (v["n"], round(v.get("top3_rate", 0), 3), round(v.get("place_roi", 0), 3))
                             for m, v in yr["triangle_A"].items()}, ensure_ascii=False), file=sys.stderr, flush=True)
    (OUT / "marks_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

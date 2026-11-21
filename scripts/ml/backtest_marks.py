"""AI の印（◎○▲☆△消）を過去レースに付けて、印ごとの成績を出す（前年までで学習して、その年を予測する）.

    cd scripts/ml && python backtest_marks.py   # → data/ml/marks_backtest.json

印の決め方（上から順に。1頭に印は1つ。▲☆△消は複数頭に付いてよい）
1. ◎：AI の勝率1位。「割高」なら次の馬
2. ○：残りで AIスコア（3着以内確率）1位
3. ▲：残りで、AI の勝率が上位4頭以内、かつ単勝の回収見込み（AI と市場を合わせた勝率 × 単勝オッズ）が WIN_EV 以上
4. ☆：残りで、目安の倍率（AI と市場を合わせた3着以内 ÷ 市場だけを補正した3着以内）が 1.1 以上
5. △：残りで、AIスコアが6位以内
6. 消：残りで、3番人気以内かつ「割高」（倍率 0.9 以下）
- 合わせ方の係数（単勝：条件付きロジット、3着以内：ロジスティック回帰）は、毎年その前の年で決める（backtest_value.py と同じ）
- 印ごとに：頭数、勝率、3着内率、単勝・複勝回収率、市場の確率の帯をそろえた全馬の複勝回収率（人気の偏りを除いた基準）
- WIN_EV は 0.85・0.90・1.00 を比べる。確定オッズなので、レース前のオッズでは少し悪くなる
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest_value import TEST_YEARS, benter, fit_benter, prepare  # noqa: E402
from train import OUT  # noqa: E402

MARKS = ["◎", "○", "▲", "☆", "△", "消", "無印"]
WIN_EVS = [0.85, 0.90, 1.00]
VALUE, EXPENSIVE = 1.1, 0.9


def assign(g, ai_win, ai_top3, win_ev, ratio, pop, win_ev_min) -> list[str]:
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
    for i in range(k):  # ▲
        if not used[i] and win_rank[i] < 4 and win_ev[g][i] >= win_ev_min:
            mark[i] = "▲"; used[i] = True
    for i in range(k):  # ☆
        if not used[i] and r[i] >= VALUE:
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
    pop = np.zeros(d["n"], int)
    for g in gs:
        pop[g[np.argsort(odds[g], kind="stable")]] = np.arange(1, len(g) + 1)
    out = {"rules": {"value": VALUE, "expensive": EXPENSIVE, "win_ev": WIN_EVS}, "years": {}}
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
        yr = {"races": len(gt), "all": stats(mt, d, mkt_cal, mt)}
        for ev_min in WIN_EVS:
            marks = np.empty(d["n"], dtype=object)
            for g in gt:
                marks[g] = assign(g, d["ai_win"], d["ai_top3"], win_ev, ratio, pop, ev_min)
            yr[f"win_ev_{ev_min}"] = {m: stats(mt & (marks == m), d, mkt_cal, mt) for m in MARKS}
        out["years"][y] = yr
        print(y, json.dumps({m: (v["n"], round(v.get("top3_rate", 0), 3), round(v.get("place_roi", 0), 3))
                             for m, v in yr["win_ev_0.9"].items()}, ensure_ascii=False), file=sys.stderr, flush=True)
    (OUT / "marks_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""AI印から作る買い目（ワイド・馬連・馬単・3連複・3連単。単勝・複勝はなし）の過去の成績（2026-11-23、依頼者の案）.

    cd scripts/ml && python backtest_mark_bets.py   # → data/ml/mark_bets_backtest.json

印は backtest_marks.py の決め方 B（アプリと同じ）。年ごとに前年までで学習した AI、確定オッズ・確定の払戻（成績表の payouts.csv）
組み合わせの確率はハービルの式（1着の確率に比例して順に決まる）。勝率は次の2つを比べる
- 合わせた勝率：AI と市場の勝率を条件付きロジットで合わせたもの（係数は前の年で決める。backtest_value.py と同じ）
- AI の勝率
買い目の作り方（どれも1点100円）
- 印：ワイド・馬連は ◎○▲ のボックス（3点）、馬単は ◎→○▲（2点）、3連複は ◎-○▲-○▲☆△、3連単は ◎→○▲→○▲☆△
- 確率上位（印）：印の付いた馬（消を除く）の組み合わせから、確率の高い順に「印」と同じ点数
- 確率上位（全馬）：全馬の組み合わせから、確率の高い順に同じ点数（印で絞る意味があるかを見る）
- 確率上位 N点：印の付いた馬から、券種ごとに決めた点数（ワイド 3・馬連 3・馬単 3・3連複 5・3連単 6）
成績：1レースの点数、レースの的中率（どれか当たった割合）、回収率、買い目の確率の合計の平均（見込みの的中率）
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from backtest_marks import assign  # noqa: E402
from backtest_value import TEST_YEARS, benter, fit_benter, prepare  # noqa: E402
from train import DATA, OUT  # noqa: E402
from train_longshot import MIN_POP, fit_market_cal, fit_stage2, popularity, stage2_matrix  # noqa: E402

TYPES = ["ワイド", "馬連", "馬単", "3連複", "3連単"]
ORDERED = {"馬単", "3連単"}
FIXED_N = {"ワイド": 3, "馬連": 3, "馬単": 3, "3連複": 5, "3連単": 6}
USE = ["◎", "○", "▲", "☆", "△"]


def load_payouts(years) -> dict:
    out: dict[str, dict[str, dict[str, float]]] = {}
    for y in years:
        path = DATA / "csv" / y / "payouts.csv"
        with path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["type"] in TYPES and r["combo"] and r["payout"]:
                    t = out.setdefault(r["race_id"], {}).setdefault(r["type"], {})
                    t[r["combo"]] = t.get(r["combo"], 0) + float(r["payout"])
    return out


def ticket_probs(p: np.ndarray, t: str, horses: list[int]) -> dict[tuple, float]:
    """horses（レース内の添字）の組み合わせの確率（ハービル）。p はレースの全馬の勝率（合計1）."""
    n = len(p)

    def tri(i, j, k):  # i→j→k の順
        return p[i] * p[j] / (1 - p[i]) * p[k] / (1 - p[i] - p[j])

    out = {}
    if t == "馬単":
        for i, j in itertools.permutations(horses, 2):
            out[(i, j)] = p[i] * p[j] / (1 - p[i])
    elif t == "馬連":
        for i, j in itertools.combinations(horses, 2):
            out[(i, j)] = p[i] * p[j] / (1 - p[i]) + p[j] * p[i] / (1 - p[j])
    elif t == "3連単":
        for i, j, k in itertools.permutations(horses, 3):
            out[(i, j, k)] = tri(i, j, k)
    elif t == "3連複":
        for c in itertools.combinations(horses, 3):
            out[c] = sum(tri(*q) for q in itertools.permutations(c))
    elif t == "ワイド":
        for i, j in itertools.combinations(horses, 2):
            out[(i, j)] = sum(tri(*q) for k in range(n) if k not in (i, j) for q in itertools.permutations((i, j, k)))
    return out


def mark_tickets(mark: list[str], t: str) -> list[tuple]:
    idx = lambda ms: [i for i, m in enumerate(mark) if m in ms]  # noqa: E731
    a, b, c = idx(["◎"]), idx(["○", "▲"]), idx(["○", "▲", "☆", "△"])
    if t in ("ワイド", "馬連"):
        return list(itertools.combinations(sorted(a + b), 2))
    if t == "馬単":
        return [(i, j) for i in a for j in b]
    if t == "3連単":
        return [(i, j, k) for i in a for j in b for k in c if len({i, j, k}) == 3]
    if t == "3連複":
        return sorted({tuple(sorted((i, j, k))) for i in a for j in b for k in c if len({i, j, k}) == 3})
    return []


def key(t: str, nums: list[int]) -> str:
    return "-".join(str(x) for x in (nums if t in ORDERED else sorted(nums)))


def main() -> None:
    d = prepare()
    year, gs, odds = d["year"], d["gs"], d["odds"]
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    pop = popularity(d)
    X2 = stage2_matrix(d, pop)
    y3 = d["is_top3"].astype(float)
    pay = load_payouts(TEST_YEARS)
    out = {"note": "印から作る買い目。1点100円、確定の払戻。見込みはハービルの式の確率の合計", "years": {}}
    for y in TEST_YEARS:
        prev = str(int(y) - 1)
        mp = year == prev
        gp = [g for g in gs if year[g[0]] == prev]
        gt = [g for g in gs if year[g[0]] == y and str(d["race_id"][g[0]]) in pay and (d["umaban"][g] > 0).all()]
        a, b = fit_benter(gp, d["la"], d["lm"], d["is_win"])
        comb_win = benter(gs, d["la"], d["lm"], a, b, d["n"])
        X = np.column_stack([logit(d["ai_top3"]), logit(d["mkt_top3"])])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X[mp], d["is_top3"][mp])
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][mp])[:, None], d["is_top3"][mp])
        ratio = lr.predict_proba(X)[:, 1] / lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        win_ev = comb_win * odds
        before = np.isin(year, [x for x in bv.PRED_YEARS if x < y])
        m2, _ = fit_stage2(X2, y3, year, before & (pop >= MIN_POP), max(x for x in bv.PRED_YEARS if x < y))
        mt = year == y
        star = np.zeros(d["n"])
        mc = fit_market_cal(d["mkt_top3"], y3, before).predict_proba(logit(np.clip(d["mkt_top3"][mt], 1e-6, 1 - 1e-6))[:, None])[:, 1]
        star[mt] = m2.predict(X2[mt]) / mc
        acc: dict[str, dict[str, dict[str, float]]] = {}
        for g in gt:
            mark = assign(g, d["ai_win"], d["ai_top3"], win_ev, ratio, pop, "B", star)
            rp = pay[str(d["race_id"][g[0]])]
            nums = [int(x) for x in d["umaban"][g]]
            marked = [i for i, m in enumerate(mark) if m in USE]
            for wname, wsrc in (("合わせた勝率", comb_win), ("AIの勝率", d["ai_win"])):
                p = wsrc[g] / wsrc[g].sum()
                for t in TYPES:
                    base = mark_tickets(mark, t)
                    pm = ticket_probs(p, t, marked)
                    pa = ticket_probs(p, t, list(range(len(g)))) if t != "3連単" or len(g) <= 18 else {}
                    rules = {"印": base,
                             "確率上位（印）": sorted(pm, key=lambda c: -pm[c])[:len(base)],
                             "確率上位（全馬）": sorted(pa, key=lambda c: -pa[c])[:len(base)],
                             f"確率上位 {FIXED_N[t]}点": sorted(pm, key=lambda c: -pm[c])[:FIXED_N[t]]}
                    for rname, tickets in rules.items():
                        if wname == "AIの勝率" and rname == "印":
                            continue  # 印そのものは勝率の選び方に関係ない
                        s = acc.setdefault(f"{wname}｜{rname}", {}).setdefault(t, {"races": 0, "points": 0, "hit_races": 0, "ret": 0.0, "exp": 0.0})
                        s["races"] += 1
                        s["points"] += len(tickets)
                        r = sum(rp.get(t, {}).get(key(t, [nums[i] for i in c]), 0) for c in tickets)
                        s["ret"] += r
                        s["hit_races"] += r > 0
                        s["exp"] += sum((pa if rname == "確率上位（全馬）" else pm)[c] for c in tickets)
        res = {}
        for rule, by_t in acc.items():
            res[rule] = {t: {"points_per_race": round(s["points"] / s["races"], 2), "hit_rate": round(s["hit_races"] / s["races"], 4),
                             "roi": round(s["ret"] / max(1, 100 * s["points"]), 4), "expected_hit": round(s["exp"] / s["races"], 4)}
                         for t, s in by_t.items()}
        out["years"][y] = {"races": len(gt), "rules": res}
        print(y, len(gt), json.dumps({r: {t: (v["points_per_race"], v["hit_rate"], v["roi"]) for t, v in x.items()} for r, x in res.items()},
                                     ensure_ascii=False), file=sys.stderr, flush=True)
        (OUT / "mark_bets_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

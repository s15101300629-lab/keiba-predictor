"""3連系の確率の補正（Benter の補正）の比較（2026-11-23、依頼者の案）.

    cd scripts/ml && python benter_exotic.py   # → data/ml/benter_exotic.json

ハービルの式：2着は残りの馬の勝率に比例、3着も同じ。Benter の補正：2着は p^λ2、3着は p^λ3 に比例（λ < 1 で平らになる）
- 勝率は印の買い目と同じ「AI と市場を合わせた勝率」（backtest_value の条件付きロジット。係数は前の年で決める）
- λ2・λ3 は前の年の実際の着順で決める（2着は「1着を除いた中で2着の馬」の対数尤度、3着も同じ。それぞれ 0.40〜1.20 を 0.02 刻み）
- 試す年（2024〜2026）で：2着・3着の対数尤度（1頭あたり）、実際の 1-2-3 着の並び・3着以内の3頭の組の確率の対数尤度、
  印から作る買い目（backtest_mark_bets.py の「合わせた勝率｜確率上位 N点」）の的中率・回収率を、ハービル（λ=1）と比べる
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from backtest_mark_bets import FIXED_N, ORDERED, TYPES, USE, key, load_payouts  # noqa: E402
from backtest_marks import assign  # noqa: E402
from backtest_value import TEST_YEARS, benter, fit_benter, prepare  # noqa: E402
from train import OUT  # noqa: E402
from train_longshot import MIN_POP, fit_market_cal, fit_stage2, popularity, stage2_matrix  # noqa: E402

GRID = np.round(np.arange(0.40, 1.2001, 0.02), 2)


def cond_ll(p, order, lam, k):
    """着順 order（レース内の添字、1着から）のうち k 着（k=2 or 3）の、それより前の馬を除いた条件付き対数尤度."""
    used = order[:k - 1]
    w = np.power(p, lam)
    w[used] = 0
    return float(np.log(w[order[k - 1]] / w.sum()))


def fit_lam(races, k) -> float:
    best, bl = 1.0, -1e18
    for lam in GRID:
        ll = sum(cond_ll(p, o, lam, k) for p, o in races)
        if ll > bl:
            best, bl = float(lam), ll
    return best


def tri_probs(p, l2, l3):
    """全部の 1→2→3 の並びの確率（辞書）."""
    n = len(p)
    w2, w3 = np.power(p, l2), np.power(p, l3)
    out = {}
    for i in range(n):
        s2 = w2.sum() - w2[i]
        for j in range(n):
            if j == i:
                continue
            pj = p[i] * w2[j] / s2
            s3 = w3.sum() - w3[i] - w3[j]
            for k in range(n):
                if k != i and k != j:
                    out[(i, j, k)] = pj * w3[k] / s3
    return out


def ticket_probs(tp, t, horses):
    hs = set(horses)
    out: dict[tuple, float] = {}
    for (i, j, k), v in tp.items():
        if t == "3連単":
            if {i, j, k} <= hs:
                out[(i, j, k)] = out.get((i, j, k), 0) + v
        elif t == "3連複":
            if {i, j, k} <= hs:
                c = tuple(sorted((i, j, k)))
                out[c] = out.get(c, 0) + v
        elif t == "馬単":
            if {i, j} <= hs:
                out[(i, j)] = out.get((i, j), 0) + v
        elif t == "馬連":
            if {i, j} <= hs:
                c = tuple(sorted((i, j)))
                out[c] = out.get(c, 0) + v
        elif t == "ワイド":
            for a, b in ((i, j), (i, k), (j, k)):
                if {a, b} <= hs:
                    c = tuple(sorted((a, b)))
                    out[c] = out.get(c, 0) + v
    return out


def main() -> None:
    d = prepare()
    year, gs, odds = d["year"], d["gs"], d["odds"]
    logit = lambda p: np.log(p / (1 - p))  # noqa: E731
    pop = popularity(d)
    X2 = stage2_matrix(d, pop)
    y3 = d["is_top3"].astype(float)
    pay = load_payouts(TEST_YEARS)
    fin = d["finish"]
    out = {"note": "2着・3着の確率を p^λ に比例させる補正。λ は前の年で決める。1点100円・確定の払戻", "years": {}}
    for y in TEST_YEARS:
        prev = str(int(y) - 1)
        mp = year == prev
        a, b = fit_benter([g for g in gs if year[g[0]] == prev], d["la"], d["lm"], d["is_win"])
        cw = benter(gs, d["la"], d["lm"], a, b, d["n"])

        def races_of(yy):
            rs = []
            for g in gs:
                if year[g[0]] != yy:
                    continue
                f = fin[g]
                if not all((f == k).sum() == 1 for k in (1, 2, 3)):
                    continue  # 同着・中止で 1〜3着がそろわないレースは除く
                order = [int(np.flatnonzero(f == k)[0]) for k in (1, 2, 3)]
                rs.append((g, cw[g] / cw[g].sum(), order))
            return rs

        rp, rt = races_of(prev), races_of(y)
        l2 = fit_lam([(p, o) for _, p, o in rp], 2)
        l3 = fit_lam([(p, o) for _, p, o in rp], 3)
        res = {"lambda2": l2, "lambda3": l3, "races": len(rt)}
        for name, (x2, x3) in {"ハービル": (1.0, 1.0), "補正": (l2, l3)}.items():
            ll2 = np.mean([cond_ll(p, o, x2, 2) for _, p, o in rt])
            ll3 = np.mean([cond_ll(p, o, x3, 3) for _, p, o in rt])
            res[name] = {"ll_2nd": round(float(ll2), 5), "ll_3rd": round(float(ll3), 5)}
        # 印から作る買い目
        X = np.column_stack([logit(d["ai_top3"]), logit(d["mkt_top3"])])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X[mp], d["is_top3"][mp])
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][mp])[:, None], d["is_top3"][mp])
        ratio = lr.predict_proba(X)[:, 1] / lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        before = np.isin(year, [x for x in bv.PRED_YEARS if x < y])
        m2, _ = fit_stage2(X2, y3, year, before & (pop >= MIN_POP), max(x for x in bv.PRED_YEARS if x < y))
        mt = year == y
        star = np.zeros(d["n"])
        mc = fit_market_cal(d["mkt_top3"], y3, before).predict_proba(logit(np.clip(d["mkt_top3"][mt], 1e-6, 1 - 1e-6))[:, None])[:, 1]
        star[mt] = m2.predict(X2[mt]) / mc
        acc = {nm: {t: {"races": 0, "points": 0, "hit": 0, "ret": 0.0, "ll": 0.0} for t in TYPES} for nm in ("ハービル", "補正")}
        for g, p, order in rt:
            rid = str(d["race_id"][g[0]])
            if rid not in pay or not (d["umaban"][g] > 0).all():
                continue
            mark = assign(g, d["ai_win"], d["ai_top3"], cw * odds, ratio, pop, "B", star)
            marked = [i for i, m in enumerate(mark) if m in USE]
            nums = [int(x) for x in d["umaban"][g]]
            for nm, (x2, x3) in {"ハービル": (1.0, 1.0), "補正": (l2, l3)}.items():
                tp = tri_probs(p, x2, x3)
                for t in TYPES:
                    pm = ticket_probs(tp, t, marked)
                    tk = sorted(pm, key=lambda c: -pm[c])[:FIXED_N[t]]
                    s = acc[nm][t]
                    r = sum(pay[rid].get(t, {}).get(key(t, [nums[i] for i in c]), 0) for c in tk)
                    s["races"] += 1; s["points"] += len(tk); s["ret"] += r; s["hit"] += r > 0
                # 実際の並び・組の確率
                acc[nm]["3連単"]["ll"] += float(np.log(tp[tuple(order)]))
                acc[nm]["3連複"]["ll"] += float(np.log(sum(tp[q] for q in itertools.permutations(order))))
        for nm in acc:
            res[nm]["bets"] = {t: {"hit_rate": round(s["hit"] / max(1, s["races"]), 4), "roi": round(s["ret"] / max(1, 100 * s["points"]), 4)}
                               for t, s in acc[nm].items()}
            res[nm]["ll_trifecta"] = round(acc[nm]["3連単"]["ll"] / max(1, acc[nm]["3連単"]["races"]), 4)
            res[nm]["ll_trio"] = round(acc[nm]["3連複"]["ll"] / max(1, acc[nm]["3連複"]["races"]), 4)
        out["years"][y] = res
        print(y, json.dumps(res, ensure_ascii=False), file=sys.stderr, flush=True)
        (OUT / "benter_exotic.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

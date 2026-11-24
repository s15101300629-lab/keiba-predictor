"""印から作る買い目の改善の比較（2026-11-24、依頼者の案 1・3・4・5・6・7・10）.

    cd scripts/ml && python backtest_bets2.py   # → data/ml/bets2_backtest.json・data/ml/mark_bets_backtest.json（画面の過去の成績）

2024・2025・2026年を、年ごとに前の年までで決めた設定で計算する。確定の払戻・100円均一
- 前：今のアプリ（AI と市場の勝率を条件付きロジットで合わせた勝率、Benter の補正 λ は前の年、印の馬から確率の高い順に決まった点数）
- 案1：3着以内の合わせた確率（value_combine の combined の形。前の年で当てはめ、合計を3にそろえる）に合う各馬の強さを逆算し、
  その強さから組み合わせの確率を出す（2着・3着は強さの λ 乗に比例＝Benter）
- 案6：点数をレースごとに変える。「どれか当たる確率」が目標に届くまで確率の高い順に足す（上限は今の点数の2倍）。
  目標は、前の年で「今の点数と同じくらいの的中率」になる値に決める
- 案7：ハービル（λ=1）と Benter（λ は前の年）を比べる（案1の確率で）
- 案3：クラス・頭数ごとの的中率・回収率。見送り：新馬、または前の年の回収率が全体より低かったクラス
- 案4：1〜2番人気に「消」が付いたレース（勝負レース）だけの回収率
- 案5：☆の馬の複勝、ワイド ◎-☆・○-☆
- 案10：AI の確率のメリハリの補正（backtest_value.prepare の T）の、logloss の前後
"""

from __future__ import annotations

import csv
import itertools
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import backtest_value as bv  # noqa: E402
from backtest_mark_bets import FIXED_N, TYPES, USE, key, load_payouts  # noqa: E402
from backtest_marks import assign  # noqa: E402
from backtest_value import TEST_YEARS, benter, fit_benter, prepare  # noqa: E402
from benter_exotic import cond_ll, fit_lam  # noqa: E402
from train import DATA, OUT  # noqa: E402
from train_longshot import MIN_POP, fit_market_cal, fit_stage2, popularity, stage2_matrix  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

CLASSES = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP", "L", "重賞"]
TARGET_GRID = np.round(np.arange(0.05, 0.951, 0.01), 2)


def race_class(name: str, cond: str) -> str:
    if re.search(r"（Ｇ[ⅠⅡⅢ]+）|\(G[123I]+\)|（Ｊ・Ｇ", name):
        return "重賞"
    if re.search(r"（Ｌ）|\(L\)", name):
        return "L"
    if "新馬" in name:
        return "新馬"
    if "未勝利" in name:
        return "未勝利"
    for k, pat in (("1勝", r"1勝クラス|500万"), ("2勝", r"2勝クラス|1000万"), ("3勝", r"3勝クラス|1600万")):
        if re.search(pat, cond) or re.search(pat, name):
            return k
    return "OP"


def load_classes(years) -> dict[str, str]:
    out = {}
    for y in years:
        with (DATA / "csv" / y / "races.csv").open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["race_id"]] = race_class(r["race_name"], r["conditions"])
    return out


def top3_from_strength(s, l2, l3):
    """強さ s（合計1）から、各馬が3着以内に入る確率（1着は s に比例、2着・3着は s^λ に比例）."""
    n = len(s)
    w2, w3 = s ** l2, s ** l3
    p12 = s[:, None] * w2[None, :] / (w2.sum() - w2)[:, None]  # i が1着・j が2着
    np.fill_diagonal(p12, 0)
    denom = w3.sum() - w3[:, None] - w3[None, :]  # i・j を除いた残り
    p123 = p12[:, :, None] * w3[None, None, :] / denom[:, :, None]
    idx = np.arange(n)
    p123[idx, :, idx] = 0
    p123[:, idx, idx] = 0
    first, second, third = s, p12.sum(0), p123.sum((0, 1))
    return first + second + third, p12, p123


def invert_strength(t, l2, l3, iters=60):
    """3着以内の確率 t（合計 min(3,頭数)）に合う強さ（合計1）を、比を掛ける繰り返しで求める."""
    s = np.clip(t, 1e-4, None) / t.sum()
    for _ in range(iters):
        cur, _, _ = top3_from_strength(s, l2, l3)
        s = s * np.power(np.clip(t, 1e-6, None) / np.clip(cur, 1e-9, None), 0.8)
        s = s / s.sum()
    return s


def ticket_probs(s, l2, l3, horses, t):
    """馬の添字 horses の中の組み合わせの確率（券種 t）."""
    _, p12, p123 = top3_from_strength(s, l2, l3)
    hs = list(horses)
    out = {}
    if t == "馬単":
        for i, j in itertools.permutations(hs, 2):
            out[(i, j)] = p12[i, j]
    elif t == "馬連":
        for i, j in itertools.combinations(hs, 2):
            out[(i, j)] = p12[i, j] + p12[j, i]
    elif t == "3連単":
        for i, j, k in itertools.permutations(hs, 3):
            out[(i, j, k)] = p123[i, j, k]
    elif t == "3連複":
        for c in itertools.combinations(hs, 3):
            out[c] = sum(p123[q] for q in itertools.permutations(c))
    elif t == "ワイド":
        # 3着以内に i と j がどちらも入る確率：全部の3頭の組で数える
        tri = p123 + p123.transpose(0, 2, 1) + p123.transpose(1, 0, 2) + p123.transpose(1, 2, 0) + p123.transpose(2, 0, 1) + p123.transpose(2, 1, 0)
        for i, j in itertools.combinations(hs, 2):
            out[(i, j)] = tri[i, j, :].sum() / 1.0
    return out


def wide_hit(s, l2, l3, picks):
    _, _, p123 = top3_from_strength(s, l2, l3)
    n = len(s)
    tot = 0.0
    for a, b, c in itertools.combinations(range(n), 3):
        st = {a, b, c}
        if any(i in st and j in st for i, j in picks):
            tot += sum(p123[q] for q in itertools.permutations((a, b, c)))
    return tot


def choose(probs: dict, t: str, n_fixed: int, target: float | None, s=None, l2=None, l3=None):
    ranked = sorted(probs, key=lambda c: -probs[c])
    if target is None:
        return ranked[:n_fixed]
    out = []
    for c in ranked[:2 * n_fixed]:
        out.append(c)
        hit = wide_hit(s, l2, l3, out) if t == "ワイド" else sum(probs[x] for x in out)
        if hit >= target:
            break
    return out


def main() -> None:
    bv.PRED_YEARS = ["2022", "2023", "2024", "2025", "2026"]
    d = prepare()
    year, gs, odds = d["year"], d["gs"], d["odds"]
    logit = lambda p: np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))  # noqa: E731
    pop = popularity(d)
    X2 = stage2_matrix(d, pop)
    y3 = d["is_top3"].astype(float)
    pay = load_payouts(TEST_YEARS + [str(int(TEST_YEARS[0]) - 1)])
    classes = load_classes(TEST_YEARS + [str(int(TEST_YEARS[0]) - 1)])
    out = {"note": "印から作る買い目の改善の比較。1点100円、確定の払戻。年ごとに前の年までで決めた設定", "temperature": bv.TEMPS, "years": {}}
    for y in TEST_YEARS:
        prev = str(int(y) - 1)
        res = {}
        # 案10：メリハリの補正の前後の logloss（3着以内・勝率・連対）
        mt = year == y
        rid_t = d["race_id"][mt]
        cal = {}
        for k, tgt, lab, raw in (("top3", 3.0, d["is_top3"], d["ptop3_raw"]), ("win", 1.0, d["is_win"], d["pwin_raw"]),
                                 ("top2", 2.0, d["finish"] <= 2, d["ptop2_raw"])):
            T = bv.TEMPS[k][y]
            before = 1 / (1 + np.exp(-logit(raw[mt]) / T))  # T を外す
            cal[k] = {"T": T, "logloss_before": round(float(log_loss(lab[mt], normalize_sum(rid_t, before, tgt))), 5),
                      "logloss_after": round(float(log_loss(lab[mt], normalize_sum(rid_t, raw[mt], tgt))), 5)}
        res["calibration"] = cal
        # 合わせ方（前の年で当てはめ）
        a, b = fit_benter([g for g in gs if year[g[0]] == prev], d["la"], d["lm"], d["is_win"])
        cw = benter(gs, d["la"], d["lm"], a, b, d["n"])
        mp = year == prev
        X = np.column_stack([logit(d["ai_top3"]), logit(d["mkt_top3"])])
        lr = LogisticRegression(C=1e6, max_iter=1000).fit(X[mp], d["is_top3"][mp])
        comb3 = lr.predict_proba(X)[:, 1]
        lr_m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(d["mkt_top3"][mp])[:, None], d["is_top3"][mp])
        ratio = comb3 / lr_m.predict_proba(logit(d["mkt_top3"])[:, None])[:, 1]
        # λ（前の年の着順で）
        def orders(yy):
            rs = []
            for g in gs:
                if year[g[0]] != yy:
                    continue
                f = d["finish"][g]
                if all((f == k).sum() == 1 for k in (1, 2, 3)):
                    rs.append((cw[g] / cw[g].sum(), [int(np.flatnonzero(f == k)[0]) for k in (1, 2, 3)]))
            return rs
        rp = orders(prev)
        l2, l3 = fit_lam(rp, 2), fit_lam(rp, 3)
        res["lambda"] = [l2, l3]
        # ☆
        before = np.isin(year, [x for x in bv.PRED_YEARS if x < y])
        m2, _ = fit_stage2(X2, y3, year, before & (pop >= MIN_POP), max(x for x in bv.PRED_YEARS if x < y))
        star = np.zeros(d["n"])
        for mm in (mt, mp):
            mc = fit_market_cal(d["mkt_top3"], y3, before).predict_proba(logit(d["mkt_top3"][mm])[:, None])[:, 1]
            star[mm] = m2.predict(X2[mm]) / mc

        def races_of(yy):
            out_r = []
            for g in gs:
                rid = str(d["race_id"][g[0]])
                if year[g[0]] != yy or rid not in pay or not (d["umaban"][g] > 0).all():
                    continue
                mark = assign(g, d["ai_win"], d["ai_top3"], cw * odds, ratio, pop, "B", star)
                marked = [i for i, m in enumerate(mark) if m in USE]
                if len(marked) < 3:
                    continue
                t3 = normalize_sum(np.zeros(len(g)), comb3[g], 3.0)
                out_r.append({"g": g, "rid": rid, "mark": mark, "marked": marked, "nums": [int(x) for x in d["umaban"][g]],
                              "pw": cw[g] / cw[g].sum(), "s": invert_strength(t3, l2, l3), "s_h": invert_strength(t3, 1.0, 1.0),
                              "cls": classes.get(rid, "OP"), "n": len(g), "pop": pop[g]})
            return out_r

        def prepare_race(r, prob_of):
            """券種ごとに、確率の高い順の買い目・それまでの「どれか当たる確率」・払戻（目標を変えても計算し直さないため）."""
            sv, lam2, lam3 = prob_of(r)
            _, p12, p123 = top3_from_strength(sv, lam2, lam3)
            n = len(sv)
            tri = sum(p123.transpose(q) for q in itertools.permutations(range(3)))  # 3頭の組（並び問わず）の確率
            A, B, C = np.array(list(itertools.combinations(range(n), 3))).T
            P = tri[A, B, C]
            hs = r["marked"]
            outp = {}
            for t in TYPES:
                if t == "馬単":
                    probs = {(i, j): p12[i, j] for i, j in itertools.permutations(hs, 2)}
                elif t == "馬連":
                    probs = {(i, j): p12[i, j] + p12[j, i] for i, j in itertools.combinations(hs, 2)}
                elif t == "3連単":
                    probs = {(i, j, k): p123[i, j, k] for i, j, k in itertools.permutations(hs, 3)}
                elif t == "3連複":
                    probs = {c: tri[c] for c in itertools.combinations(hs, 3)}
                else:
                    probs = {(i, j): float(P[((A == i) & (B == j)) | ((A == i) & (C == j)) | ((B == i) & (C == j))].sum()) for i, j in itertools.combinations(hs, 2)}
                ranked = sorted(probs, key=lambda c: -probs[c])[:2 * FIXED_N[t]]
                if t == "ワイド":
                    mask = np.zeros(len(P), bool)
                    cum = []
                    for i, j in ranked:
                        mask |= ((A == i) & (B == j)) | ((A == i) & (C == j)) | ((B == i) & (C == j))
                        cum.append(float(P[mask].sum()))
                else:
                    cum = list(np.cumsum([probs[c] for c in ranked]))
                rets = [pay[r["rid"]].get(t, {}).get(key(t, [r["nums"][i] for i in c]), 0) for c in ranked]
                outp[t] = (cum, rets)
            return outp

        def evaluate(pre, target=None):
            """券種ごと・レースごとの [点数, 払戻]. target なしは決まった点数."""
            recs = {t: [] for t in TYPES}
            for pr in pre:
                for t in TYPES:
                    cum, rets = pr[t]
                    if target is None:
                        k = min(FIXED_N[t], len(rets))
                    else:
                        k = next((i + 1 for i, c in enumerate(cum) if c >= target[t]), len(cum))
                    recs[t].append((k, sum(rets[:k])))
            return recs

        def summary(recs, sel=None):
            o = {}
            for t, rr in recs.items():
                rr = [x for i, x in enumerate(rr) if sel is None or sel[i]]
                pts = sum(x[0] for x in rr)
                o[t] = {"races": len(rr), "points_per_race": round(pts / max(1, len(rr)), 2),
                        "hit_rate": round(sum(1 for x in rr if x[1] > 0) / max(1, len(rr)), 4), "roi": round(sum(x[1] for x in rr) / max(1, 100 * pts), 4)}
            pts = sum(x[0] for rr in recs.values() for i, x in enumerate(rr) if sel is None or sel[i])
            ret = sum(x[1] for rr in recs.values() for i, x in enumerate(rr) if sel is None or sel[i])
            o["全券種"] = {"roi": round(ret / max(1, 100 * pts), 4), "points": pts}
            return o

        rt, rpv = races_of(y), races_of(prev)
        base = lambda r: (r["pw"], l2, l3)  # noqa: E731  前（勝率から）
        harv = lambda r: (r["s_h"], 1.0, 1.0)  # noqa: E731  案1・ハービル
        new = lambda r: (r["s"], l2, l3)  # noqa: E731  案1・Benter
        pre_new = [prepare_race(r, new) for r in rt]
        rec_base, rec_h, rec_new = evaluate([prepare_race(r, base) for r in rt]), evaluate([prepare_race(r, harv) for r in rt]), evaluate(pre_new)
        res["前（勝率から・決まった点数）"] = summary(rec_base)
        res["案1・ハービル（決まった点数）"] = summary(rec_h)
        res["案1・Benter（決まった点数）"] = summary(rec_new)
        # 案6：目標を前の年で決める（前の年の「案1・決まった点数」の的中率に一番近い目標。二分法）
        pre_prev = [prepare_race(r, new) for r in rpv]
        rec_prev_fixed = evaluate(pre_prev)
        targets = {}
        for t in TYPES:
            want = np.mean([x[1] > 0 for x in rec_prev_fixed[t]])
            lo, hi = 0.0, 1.0
            for _ in range(14):
                mid = (lo + hi) / 2
                hr = np.mean([x[1] > 0 for x in evaluate(pre_prev, {u: mid for u in TYPES})[t]])
                if hr < want:
                    lo = mid
                else:
                    hi = mid
            targets[t] = round(hi, 3)
        res["案6の目標"] = targets
        rec_var = evaluate(pre_new, targets)
        res["案1＋6（点数をレースごとに）"] = summary(rec_var)
        # 案3：クラス・頭数ごと（最終の形＝案1＋6）
        by_cls = {}
        for c in CLASSES:
            sel = [r["cls"] == c for r in rt]
            by_cls[c] = summary(rec_var, sel)["全券種"] | {"races": int(sum(sel))}
        res["案3・クラスごと"] = by_cls
        by_n = {}
        for lo_n, hi_n in ((2, 10), (11, 14), (15, 18)):
            sel = [lo_n <= r["n"] <= hi_n for r in rt]
            by_n[f"{lo_n}〜{hi_n}頭"] = summary(rec_var, sel)["全券種"] | {"races": int(sum(sel))}
        res["案3・頭数ごと"] = by_n
        prev_var = evaluate(pre_prev, targets)
        prev_all = summary(prev_var)["全券種"]["roi"]
        weak = [c for c in CLASSES if (lambda sm: sm["points"] > 0 and sm["roi"] < prev_all)(summary(prev_var, [r["cls"] == c for r in rpv])["全券種"])]
        res["案3・見送り（前の年に全体より回収率が低かったクラス）"] = {"classes": weak, **summary(rec_var, [r["cls"] not in weak for r in rt])["全券種"]}
        res["案3・見送り（新馬だけ）"] = summary(rec_var, [r["cls"] != "新馬" for r in rt])["全券種"]
        res["案3・見送りなし"] = summary(rec_var)["全券種"]
        # 案4：勝負レース（1〜2番人気に消）
        shobu = [any(r["mark"][i] == "消" and r["pop"][i] <= 2 for i in range(r["n"])) for r in rt]
        res["案4・勝負レース"] = {"races": int(sum(shobu)), **summary(rec_var, shobu)}
        res["案4・勝負レース以外"] = summary(rec_var, [not x for x in shobu])["全券種"]
        # 案5：☆の複勝・ワイド ◎-☆・○-☆
        k5 = {"複勝☆": [0, 0.0, 0], "ワイド◎-☆": [0, 0.0, 0], "ワイド○-☆": [0, 0.0, 0]}
        for r in rt:
            stars = [i for i, m in enumerate(r["mark"]) if m == "☆"]
            for i in stars:
                gi = r["g"][i]
                k5["複勝☆"][0] += 1; k5["複勝☆"][1] += d["place_ret"][gi]; k5["複勝☆"][2] += d["place_ret"][gi] > 0
                for top, name in (("◎", "ワイド◎-☆"), ("○", "ワイド○-☆")):
                    j = next((x for x, m in enumerate(r["mark"]) if m == top), None)
                    if j is None:
                        continue
                    v = pay[r["rid"]].get("ワイド", {}).get(key("ワイド", [r["nums"][i], r["nums"][j]]), 0)
                    k5[name][0] += 1; k5[name][1] += v; k5[name][2] += v > 0
        res["案5"] = {k: {"points": v[0], "hit_rate": round(v[2] / max(1, v[0]), 4), "roi": round(v[1] / max(1, 100 * v[0]), 4)} for k, v in k5.items()}
        res["races"] = len(rt)
        out["years"][y] = res
        print(y, json.dumps({k: v for k, v in res.items() if k in ("calibration", "lambda", "案6の目標", "races")}, ensure_ascii=False), file=sys.stderr, flush=True)
        (OUT / "bets2_backtest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

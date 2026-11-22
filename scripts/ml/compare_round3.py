"""依頼者の案の4つの比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 182項目）.

    cd scripts/ml && python compare_round3.py   # → data/ml/feature_comparison_round3.json

A 近走1走ごとの相対：その走の出走馬の中での能力指数の偏差（(自分 − 平均) ÷ ばらつき）を1走ずつ出し、近5走（同じ芝ダ）で
  直近加重平均（0.8^k）・最高・前走。相手の顔ぶれに対してどれだけ走れたか（能力指数そのものはコースの基準タイムとの差）
B コース×4角の位置：コース（競馬場・芝ダ・距離）ごとに、最後のコーナーの位置（÷頭数）を5つの帯に分けた3着内率（その年より前の年の走だけ）。
  各馬の近5走の最後のコーナーの位置の平均 → 帯 → このコースでのその帯の3着内率、コース全体との差、前の帯と後ろの帯の差（このコースの前残りの強さ）
C 能力指数の分解：各走の補正の量（タイム指数 − 能力指数 ＝ その日の馬場差＋斤量の補正）の近5走平均・前走
D 持ちタイム・勝ちタイム：近5走に限らず全部の過去走（同じ芝ダ）での能力指数の最高、今回の距離±200m での最高、勝った走での最高、
  距離±200m の最高の出走馬の中の順位÷頭数と最高との差
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
A = ["rz_wmean", "rz_best", "rz_last", "rz_n"]
B = ["c4_rate", "c4_adv", "c4_bias", "c4_n"]
C = ["corr_wmean", "corr_last"]
D = ["mt_best_all", "mt_best_dist", "mt_win_best", "mt_dist_rank", "mt_dist_dmax"]
BINS = 5


def wmean(vals):
    w = [0.8 ** k for k in range(len(vals))]
    return sum(v * x for v, x in zip(vals, w)) / sum(w) if vals else NAN


def main() -> None:
    rows, hist = build_dataset(DATA)
    # A：その走の出走馬の中での能力指数の偏差
    rz = {}
    for rid, entries in hist.by_race.items():
        abs_ = [run["ab"] for _, run in entries if run["ab"] is not None]
        if len(abs_) < 3:
            continue
        m = sum(abs_) / len(abs_)
        sd = math.sqrt(sum((a - m) * (a - m) for a in abs_) / len(abs_))
        for h, run in entries:
            if run["ab"] is not None and sd > 0:
                rz[(rid, h)] = (run["ab"] - m) / sd
    # B：コース×4角の位置の帯の3着内率（年ごと。その年より前の年だけを足し合わせて使う）
    per_year: dict[tuple, list] = defaultdict(lambda: [[0, 0] for _ in range(BINS)])
    for h, runs in hist.horses.items():
        for r in runs:
            if r["pos_last"] and r["field"] and r["field"] > 1 and r["finish"] is not None:
                b = min(BINS - 1, int((r["pos_last"] - 1) / r["field"] * BINS))
                c = per_year[(r["date"][:4], r["venue"], r["surface"], r["distance"])][b]
                c[0] += 1
                c[1] += r["finish"] <= 3
    years = sorted({k[0] for k in per_year})
    course_table: dict[tuple, list] = {}  # (年, コース) → その年より前の合計
    courses = {k[1:] for k in per_year}
    for co in courses:
        acc = [[0, 0] for _ in range(BINS)]
        for y in years:
            course_table[(y,) + co] = [list(x) for x in acc]
            for b in range(BINS):
                acc[b][0] += per_year[(y,) + co][b][0] if (y,) + co in per_year else 0
                acc[b][1] += per_year[(y,) + co][b][1] if (y,) + co in per_year else 0
        course_table[(str(int(years[-1]) + 1),) + co] = acc

    by_race_rows: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        runs = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"]]
        same = [x for x in runs if x["surface"] == r["surface"]]
        rec = same[::-1]
        # A
        z = [rz[(x["race_id"], r["horse"])] for x in rec[:5] if (x["race_id"], r["horse"]) in rz]
        r["rz_wmean"], r["rz_best"] = wmean(z), (max(z) if z else NAN)
        r["rz_last"] = rz.get((rec[0]["race_id"], r["horse"]), NAN) if rec else NAN
        r["rz_n"] = len(z)
        # B
        t = course_table.get((r["date"][:4], r["venue"], r["surface"], r["distance"]))
        pos = [x["pos_last"] / x["field"] for x in runs[-5:] if x["pos_last"] and x["field"]]
        if t and sum(c[0] for c in t) >= 200 and pos:
            pm = sum(pos) / len(pos)
            b = min(BINS - 1, int(max(pm - 1e-9, 0) * BINS))
            rate = [(c[1] + 1) / (c[0] + 5) for c in t]  # 少ない帯は全体に寄せる
            tot = sum(c[1] for c in t) / sum(c[0] for c in t)
            r["c4_rate"], r["c4_adv"], r["c4_bias"], r["c4_n"] = rate[b], rate[b] - tot, rate[0] - rate[-1], t[b][0]
        else:
            r["c4_rate"] = r["c4_adv"] = r["c4_bias"] = NAN
            r["c4_n"] = 0
        # C
        corr = [x["tidx"] - x["ab"] for x in rec[:5] if x["tidx"] is not None and x["ab"] is not None]
        r["corr_wmean"] = sum(corr) / len(corr) if corr else NAN
        r["corr_last"] = corr[0] if corr else NAN
        # D
        ab_all = [x["ab"] for x in same if x["ab"] is not None]
        d = r["distance"]
        ab_dist = [x["ab"] for x in same if x["ab"] is not None and x["distance"] and not math.isnan(d) and abs(x["distance"] - d) <= 200]
        ab_win = [x["ab"] for x in same if x["ab"] is not None and x["finish"] == 1]
        r["mt_best_all"] = max(ab_all) if ab_all else NAN
        r["mt_best_dist"] = max(ab_dist) if ab_dist else NAN
        r["mt_win_best"] = max(ab_win) if ab_win else NAN
        by_race_rows[r["race_id"]].append(r)
    for rid, rs in by_race_rows.items():
        vals = [x["mt_best_dist"] for x in rs if not math.isnan(x["mt_best_dist"])]
        top = max(vals) if vals else NAN
        for x in rs:
            v = x["mt_best_dist"]
            x["mt_dist_rank"] = (1 + sum(1 for o in vals if o > v)) / len(rs) if not math.isnan(v) else NAN
            x["mt_dist_dmax"] = v - top if not math.isnan(v) else NAN

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + A + B + C + D
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(182)": FEATURES, "+A 近走1走ごとの相対": FEATURES + A, "+B コース×4角の位置": FEATURES + B,
            "+C 能力指数の分解": FEATURES + C, "+D 持ちタイム・勝ちタイム": FEATURES + D, "+全部": FEATURES + A + B + C + D}
    out = {"coverage": {f: round(float(np.mean(~np.isnan(X[:, cols.index(f)]))), 3) for f in ("rz_wmean", "c4_rate", "corr_wmean", "mt_best_dist", "mt_win_best")},
           "sets": {}}
    first = None
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < first[t]["logloss_sum3"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_round3.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

"""相手の強さを考えた強さの比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 182項目）.

    cd scripts/ml && python compare_rating.py   # → data/ml/feature_comparison_rating.json

依頼者の指摘（2026-11-22）：「強い相手に接戦した走」と「弱い相手に楽勝した走」が、タイムが同じなら同じ評価になるのはよくない。
その走の出走馬の中で、どれだけ抜けていたか（相手の強さこみ）を測る。

E1 レーティング（着順）：全レースを日付順に、出走馬どうしの勝ち負け（着順）で強さの点数を更新する（イロレーティングの多頭版）。
   強い相手に先着すると大きく上がり、弱い相手に負けると大きく下がる。同じ日のレースは前日までの点数で計算し、日の終わりにまとめて更新
   K=32 を頭数−1 で割る。初めは 1500。競走中止は最下位
E2 レーティング（着差）：勝ち負けの代わりに、2頭の勝ち馬との差（秒）の差から勝ちの度合い（0〜1、0.3秒で約73%）を使う。接戦の負けは少ししか下がらない
E3 芝ダ別のレーティング（着差）
   それぞれ、レースの前日までの点数・出走数、出走馬の中での順位÷頭数・偏差・最高との差
F 相手の強さで補正した能力指数：その走の能力指数 − 相手の強さ（他の馬のそのときの調子の平均）から見た見込み（全走で1本の直線）。
   近5走（同じ芝ダ）の直近加重平均・最高
"""

from __future__ import annotations

import bisect
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
K = 32.0
MARGIN_SCALE = 0.3  # 秒


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else NAN


def ratings(hist, mode: str, by_surface: bool):
    """馬 → [(日付, その日のレース後の点数, それまでの出走数)]。mode："finish"（着順）か "margin"（着差）."""
    races = []
    for rid, entries in hist.by_race.items():
        races.append((entries[0][1]["date"], rid, entries[0][1]["surface"], entries))
    races.sort(key=lambda x: (x[0], x[1]))
    cur: dict = defaultdict(lambda: 1500.0)
    cnt: dict = defaultdict(int)
    out: dict = defaultdict(list)
    i = 0
    while i < len(races):
        date = races[i][0]
        delta: dict = defaultdict(float)
        touched = set()
        while i < len(races) and races[i][0] == date:
            _, rid, surf, entries = races[i]
            i += 1
            hs = [(h, run) for h, run in entries]
            n = len(hs)
            if n < 2:
                continue
            key = (lambda h: (h, surf)) if by_surface else (lambda h: h)
            r = [cur[key(h)] for h, _ in hs]
            fin = [run["finish"] if run["finish"] is not None else 99 for _, run in hs]
            td = [run["time_diff"] if run["time_diff"] is not None else (9.9 if run["finish"] is None else NAN) for _, run in hs]
            for a in range(n):
                s = 0.0
                for b in range(n):
                    if a == b:
                        continue
                    e = 1.0 / (1.0 + 10 ** ((r[b] - r[a]) / 400.0))
                    if mode == "margin" and not math.isnan(td[a]) and not math.isnan(td[b]):
                        w = 1.0 / (1.0 + math.exp(-(td[b] - td[a]) / MARGIN_SCALE))
                    else:
                        w = 1.0 if fin[a] < fin[b] else 0.0 if fin[a] > fin[b] else 0.5
                    s += w - e
                delta[key(hs[a][0])] += K / (n - 1) * s
                touched.add(key(hs[a][0]))
        for k in touched:
            cur[k] += delta[k]
            cnt[k] += 1
            out[k].append((date, cur[k], cnt[k]))
    return out


def before(series, date):
    """date より前の最後の点数と出走数。なければ (1500, 0)."""
    if not series:
        return 1500.0, 0
    j = bisect.bisect_left([s[0] for s in series], date) - 1
    return (series[j][1], series[j][2]) if j >= 0 else (1500.0, 0)


def add_field(rows_by_race, name):
    for rs in rows_by_race.values():
        vals = [x[name] for x in rs]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((v - m) * (v - m) for v in vals) / len(vals))
        top = max(vals)
        for x in rs:
            v = x[name]
            x[name + "_rank"] = (1 + sum(1 for o in vals if o > v)) / len(rs)
            x[name + "_z"] = (v - m) / sd if sd > 0 else 0.0
            x[name + "_dmax"] = v - top


def main() -> None:
    rows, hist = build_dataset(DATA)
    print("ratings", file=sys.stderr, flush=True)
    series = {"elo": ratings(hist, "finish", False), "elom": ratings(hist, "margin", False), "elos": ratings(hist, "margin", True)}
    # F：相手の強さ（他の馬のそのときの調子の平均）と、そこから見た能力指数の見込み
    strength = {}
    for rid, entries in hist.by_race.items():
        forms = [(h, run["form_at"]) for h, run in entries]
        for h, run in entries:
            strength[(rid, h)] = mean([f for o, f in forms if o != h])
    xs, ys = [], []
    for h, runs in hist.horses.items():
        for r in runs:
            s = strength.get((r["race_id"], h), NAN)
            if r["ab"] is not None and not math.isnan(s):
                xs.append(s); ys.append(r["ab"])
    slope, intercept = np.polyfit(xs, ys, 1)
    print("ab ~ field form:", slope, intercept, file=sys.stderr, flush=True)

    by_race = defaultdict(list)
    for r in rows:
        for name, s in series.items():
            key = (r["horse"], r["surface"]) if name == "elos" else r["horse"]
            v, n = before(s.get(key, []), r["date"])
            r[name], r[name + "_n"] = v, n
        rec = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"] and x["surface"] == r["surface"]][::-1][:5]
        adj = [x["ab"] - (slope * strength[(x["race_id"], r["horse"])] + intercept) for x in rec
               if x["ab"] is not None and not math.isnan(strength.get((x["race_id"], r["horse"]), NAN))]
        w = [0.8 ** k for k in range(len(adj))]
        r["oab_wmean"] = sum(a * b for a, b in zip(adj, w)) / sum(w) if adj else NAN
        r["oab_best"] = max(adj) if adj else NAN
        by_race[r["race_id"]].append(r)
    for name in series:
        add_field(by_race, name)

    E1 = ["elo", "elo_n", "elo_rank", "elo_z", "elo_dmax"]
    E2 = ["elom", "elom_n", "elom_rank", "elom_z", "elom_dmax"]
    E3 = ["elos", "elos_n", "elos_rank", "elos_z", "elos_dmax"]
    F = ["oab_wmean", "oab_best"]
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + E1 + E2 + E3 + F
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(182)": FEATURES, "+E1 レーティング（着順）": FEATURES + E1, "+E2 レーティング（着差）": FEATURES + E2,
            "+E2+E3 芝ダ別も": FEATURES + E2 + E3, "+F 相手補正の能力指数": FEATURES + F, "+全部": cols}
    out = {"k": K, "margin_scale": MARGIN_SCALE, "sets": {}}
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
        (OUT / "feature_comparison_rating.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

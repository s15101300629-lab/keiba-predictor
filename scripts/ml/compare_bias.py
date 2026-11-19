"""展開を考えた成績・トラックバイアスの比較（1年ずつ外す）.

    cd scripts/ml && python compare_bias.py   # → data/ml/feature_comparison_bias.json

A. 展開を考えた成績
- 過去の各走で「この流れ（ペース：スロー・ミドル・ハイ）×この位置取り（最初のコーナーの位置÷頭数を4区分）×芝ダ」なら
  着順の成分（1−(着順−1)/(頭数−1)）は平均いくつか、を見込みとし、実際との差を出す
- 見込みは 2022年（学習・評価に使わない年）の走から作る（評価する年の結果が混ざらないように）
- pace_adj_last1：前走の差、pace_adj_mean5：近5走の差の平均

B. トラックバイアス（内外の通り道の記録はないので、最後のコーナーの位置取りと枠番を目安にする）
- その日・同じ競馬場・同じ芝ダの他のレースで
  - 前残りの度合い：最後のコーナーで前3分の1にいた馬と後ろ3分の1にいた馬の、着順の成分の差の平均
  - 内枠有利の度合い：1〜4枠と5〜8枠の、着順の成分の差の平均
- 過去走の補正（その日の全レースを使う）：その走が、その日のバイアスに沿っていたか（バイアスの向き×位置取り・枠）と、
  バイアスに逆らったのにどれだけ走れたか（着順の成分 − バイアスから見た見込み）。前走と近5走の平均
- 今回のレース（同じ日の、それより前のレースだけを使う）：その日のバイアスと、この馬の脚質・枠がバイアスに合うか
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")
EXPECT_YEAR = "2022"


def isnan(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def mean(xs):
    xs = [x for x in xs if not isnan(x)]
    return sum(xs) / len(xs) if xs else NAN


def finish_c(finish, field) -> float:
    if finish is None or not field or field < 2:
        return NAN
    return 1 - (finish - 1) / (field - 1)


def pos_bucket(ratio) -> int | None:
    if isnan(ratio):
        return None
    return 0 if ratio <= 0.25 else 1 if ratio <= 0.5 else 2 if ratio <= 0.75 else 3


def main() -> None:
    races = load_races(DATA)
    info = {}  # (race_id, horse) -> その走の結果と位置
    race_meta = {}
    for r in races:
        race_meta[r["race_id"]] = {"date": r["date"], "venue": r["venue"], "surface": r["surface"], "pace": r["pace"],
                                   "no": int(r["race_id"][-2:])}
        n = r["field_size"]
        for h in r["runners"]:
            cps = [x for x in (h.get("corner_positions") or []) if x]
            info[(r["race_id"], h["horse"])] = {
                "fc": finish_c(h["finish"], n), "waku": h.get("waku"),
                "first": cps[0] / n if cps else NAN, "last": cps[-1] / n if cps else NAN,
            }
    # A. 見込み（2022年の走から）
    cell: dict[tuple, list[float]] = defaultdict(list)
    for (rid, _), v in info.items():
        m = race_meta[rid]
        b = pos_bucket(v["first"])
        if m["date"][:4] == EXPECT_YEAR and b is not None and m["pace"] is not None and not isnan(v["fc"]):
            cell[(m["surface"], m["pace"], b)].append(v["fc"])
    expect = {k: sum(v) / len(v) for k, v in cell.items()}
    for (rid, _), v in info.items():
        m = race_meta[rid]
        e = expect.get((m["surface"], m["pace"], pos_bucket(v["first"])))
        v["pace_adj"] = v["fc"] - e if e is not None and not isnan(v["fc"]) else NAN

    # B. レースごとのバイアス（前残り・内枠）
    race_bias = {}
    by_race: dict[str, list[dict]] = defaultdict(list)
    for (rid, _), v in info.items():
        by_race[rid].append(v)
    for rid, vs in by_race.items():
        front = [v["fc"] for v in vs if not isnan(v["last"]) and v["last"] <= 1 / 3]
        back = [v["fc"] for v in vs if not isnan(v["last"]) and v["last"] > 2 / 3]
        inner = [v["fc"] for v in vs if v["waku"] and v["waku"] <= 4]
        outer = [v["fc"] for v in vs if v["waku"] and v["waku"] >= 5]
        race_bias[rid] = (mean(front) - mean(back) if front and back else NAN, mean(inner) - mean(outer) if inner and outer else NAN)
    day_races: dict[tuple, list[str]] = defaultdict(list)
    for rid, m in race_meta.items():
        day_races[(m["date"], m["venue"], m["surface"])].append(rid)

    def day_bias(rid: str, earlier_only: bool):
        m = race_meta[rid]
        others = [o for o in day_races[(m["date"], m["venue"], m["surface"])] if o != rid and (not earlier_only or race_meta[o]["no"] < m["no"])]
        return mean([race_bias[o][0] for o in others]), mean([race_bias[o][1] for o in others])

    for (rid, _), v in info.items():
        fb, ib = day_bias(rid, earlier_only=False)
        lastpos = 0.5 - v["last"] if not isnan(v["last"]) else NAN  # 前にいるほど +
        inner = (0.5 if v["waku"] <= 4 else -0.5) if v["waku"] else NAN
        fit = (0 if isnan(fb) or isnan(lastpos) else fb * lastpos) + (0 if isnan(ib) or isnan(inner) else ib * inner)
        v["bias_fit"] = fit if not (isnan(fb) and isnan(ib)) else NAN
        v["bias_adj"] = v["fc"] - fit if not isnan(v["fc"]) and not isnan(v["bias_fit"]) else NAN

    rows, _ = build_dataset(DATA)
    by_horse: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_horse[r["horse"]].append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: r["date"])
        past: list[dict] = []
        for r in runs:
            last5 = past[-5:]
            last = past[-1] if past else None
            r["pace_adj_last1"] = last["pace_adj"] if last else NAN
            r["pace_adj_mean5"] = mean([p["pace_adj"] for p in last5]) if last5 else NAN
            r["bias_fit_last1"] = last["bias_fit"] if last else NAN
            r["bias_adj_last1"] = last["bias_adj"] if last else NAN
            r["bias_fit_mean5"] = mean([p["bias_fit"] for p in last5]) if last5 else NAN
            r["bias_adj_mean5"] = mean([p["bias_adj"] for p in last5]) if last5 else NAN
            # 今回のレース：同じ日のそれより前のレースからのバイアス × この馬の脚質・枠
            fb, ib = day_bias(r["race_id"], earlier_only=True)
            r["today_front_bias"], r["today_inner_bias"] = fb, ib
            style = 0.5 - r["pos_first_mean"] if not isnan(r["pos_first_mean"]) else NAN
            r["today_front_fit"] = fb * style if not isnan(fb) and not isnan(style) else NAN
            r["today_inner_fit"] = ib * (0.5 if r["waku"] <= 4 else -0.5) if not isnan(ib) and not isnan(r["waku"]) else NAN
            past.append(info.get((r["race_id"], r["horse"]), {"pace_adj": NAN, "bias_fit": NAN, "bias_adj": NAN}))

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    a = ["pace_adj_last1", "pace_adj_mean5"]
    b_past = ["bias_fit_last1", "bias_adj_last1", "bias_fit_mean5", "bias_adj_mean5"]
    b_today = ["today_front_bias", "today_inner_bias", "today_front_fit", "today_inner_fit"]
    sets = {"base": FEATURES, "+A 展開を考えた成績": FEATURES + a, "+B 過去走のバイアス補正": FEATURES + b_past,
            "+B 今回のレースのバイアス": FEATURES + b_today, "+B 両方": FEATURES + b_past + b_today,
            "+A+B 全部": FEATURES + a + b_past + b_today}
    coverage = {f: round(sum(1 for r in rows if not isnan(r[f])) / len(rows), 3) for f in a + b_past + b_today}
    print("expect cells", {str(k): round(v, 3) for k, v in sorted(expect.items())}, file=sys.stderr)
    print("coverage", coverage, file=sys.stderr, flush=True)
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = sorted({r["date"][:4] for r in rows})
    result = {"coverage": coverage, "sets": {}}
    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            bst = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = bst.predict(m(te))
            y = [r["label"] for r in te]
            per[year] = {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(te, p)}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result["sets"][name] = {"per_year": per, "mean": {k: float(np.mean([v[k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}}
    (OUT / "feature_comparison_bias.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

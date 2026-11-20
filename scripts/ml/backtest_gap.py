"""市場（オッズ）と AI の確率のズレで買うと回収率が上がるか（前年までで学習して、その年を予測する）.

    cd scripts/ml && python backtest_gap.py   # → data/ml/gap_backtest.json

- 2024・2025・2026年それぞれ、前年までで 3着以内・勝率・連対率の3モデルを学習して予測する（木の本数は metrics.json）
- AI の確率は、レースごとに合計を 1（勝率）・2（連対率）・3（3着以内）にそろえる（市場の確率と同じ合計）
- 市場の確率：確定単勝オッズの逆数を合計1にそろえたもの。連対・3着以内は、その勝率からハービルの式で出す
  （過去の馬連・3連複のオッズは、当たった組の払戻しか残っていないため）
- ズレ = AI ÷ 市場。帯ごとに、実際の率・単勝回収率・複勝回収率を出す
- 確定オッズなので、レース前に見られるオッズとは少し違う
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, matrix  # noqa: E402

YEARS = ["2024", "2025", "2026"]
BUCKETS = [(0, 0.6), (0.6, 0.8), (0.8, 1.0), (1.0, 1.2), (1.2, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 1e9)]


def race_info() -> dict:
    """race_id → {馬名: (馬番, 単勝オッズ)}, 単勝・複勝の払戻（100円あたり）."""
    out = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        if path.parent.name < "2024":
            continue
        day = json.loads(path.read_text(encoding="utf-8"))
        for race in day.get("races", []):
            win, place = {}, {}
            for p in race.get("payouts") or []:
                if p.get("type") == "単勝" and len(p.get("combo") or []) == 1:
                    win[p["combo"][0]] = p["payout"]
                if p.get("type") == "複勝" and len(p.get("combo") or []) == 1:
                    place[p["combo"][0]] = p["payout"]
            horses = {r["horse"]: (r.get("umaban"), r.get("odds")) for r in race.get("results") or []}
            out[race["race_id"]] = {"horses": horses, "win": win, "place": place}
    return out


def harville(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """勝率 p（合計1）→ 2着以内・3着以内の確率（ハービルの式）."""
    n = len(p)
    top2 = p.copy()
    top3 = p.copy()
    for i in range(n):
        for j in range(n):
            if j == i or p[j] >= 1:
                continue
            pij = p[j] * p[i] / (1 - p[j])  # j が1着、i が2着
            top2[i] += pij
            top3[i] += pij
            for k in range(n):
                if k in (i, j) or p[k] >= 1:
                    continue
                if 1 - p[j] - p[k] <= 0:
                    continue
                # j・k が1・2着（順不同は k→j も別に数える）、i が3着
                top3[i] += p[j] * p[k] / (1 - p[j]) * p[i] / (1 - p[j] - p[k])
    return top2, top3


def summarize(items: list[dict]) -> dict:
    n = len(items)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "ai": float(np.mean([x["ai"] for x in items])),
        "market": float(np.mean([x["market"] for x in items])),
        "actual": float(np.mean([x["hit"] for x in items])),
        "win_roi": float(np.mean([x["win_pay"] for x in items]) / 100),
        "place_roi": float(np.mean([x["place_pay"] for x in items]) / 100),
    }


def main() -> None:
    rows, _ = build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    metrics = json.loads((OUT / "metrics.json").read_text())
    rounds = {"top3": metrics["best_iteration"], "win": metrics["extra_targets"]["win"]["best_iteration"],
              "top2": metrics["extra_targets"]["top2"]["best_iteration"]}
    ks = {"win": 1, "top2": 2, "top3": 3}
    info = race_info()
    horses_all: list[dict] = []  # 1頭ずつ：年・各目標の AI・市場・結果
    for year in YEARS:
        tr = [r for r in rows if r["date"][:4] < year]
        te = [r for r in rows if r["date"][:4] == year]
        X_tr, X_te = matrix(tr), matrix(te)
        preds = {}
        for name, k in ks.items():
            y = [1 if r["finish"] is not None and r["finish"] <= k else 0 for r in tr]
            b = lgb.train(PARAMS, lgb.Dataset(X_tr, label=y), num_boost_round=rounds[name])
            preds[name] = b.predict(X_te)
            print(year, name, "done", file=sys.stderr, flush=True)
        by_race = defaultdict(list)
        for i, r in enumerate(te):
            by_race[r["race_id"]].append(i)
        for race_id, idx in by_race.items():
            ri = info.get(race_id)
            if not ri:
                continue
            got = [ri["horses"].get(te[i]["horse"]) for i in idx]
            if any(g is None or not g[1] or not g[0] for g in got):
                continue  # オッズがない馬がいるレースは外す
            odds = np.array([g[1] for g in got], dtype=float)
            m_win = (1 / odds) / (1 / odds).sum()
            m_top2, m_top3 = harville(m_win)
            market = {"win": m_win, "top2": m_top2, "top3": m_top3}
            for j, i in enumerate(idx):
                r = te[i]
                num = got[j][0]
                rec = {"year": year, "race_id": race_id, "odds": float(odds[j]), "finish": r["finish"],
                       "win_pay": ri["win"].get(num, 0) if r["finish"] == 1 else 0,
                       "place_pay": ri["place"].get(num, 0)}
                for name, k in ks.items():
                    p = np.array([preds[name][x] for x in idx])
                    p = p / p.sum() * k  # 合計を市場とそろえる
                    rec[name] = {"ai": float(min(p[j], 1.0)), "market": float(market[name][j]),
                                 "hit": 1 if r["finish"] is not None and r["finish"] <= k else 0}
                horses_all.append(rec)

    def flat(x: dict, name: str) -> dict:
        return {"ai": x[name]["ai"], "market": x[name]["market"], "hit": x[name]["hit"], "win_pay": x["win_pay"], "place_pay": x["place_pay"]}

    result: dict = {"years": YEARS, "horses": len(horses_all), "races": len({x["race_id"] for x in horses_all}),
                    "all_horses": summarize([flat(x, "win") for x in horses_all]), "by_ratio": {}, "strategies": {}}
    for name in ks:
        res = {}
        for lo, hi in BUCKETS:
            label = f"{lo}-{hi if hi < 1e8 else '∞'}"
            sel = [x for x in horses_all if lo <= x[name]["ai"] / x[name]["market"] < hi]
            res[label] = {"all": summarize([flat(x, name) for x in sel]),
                          **{y: summarize([flat(x, name) for x in sel if x["year"] == y]) for y in YEARS}}
        result["by_ratio"][name] = res

    # 買い方の例（1レース1頭）。ズレが一番大きい馬・AIの期待値（勝率×単勝オッズ）が一番高い馬など
    def strategy(pick) -> dict:
        out = {}
        for y in YEARS + ["all"]:
            chosen = []
            races = defaultdict(list)
            for x in horses_all:
                if y == "all" or x["year"] == y:
                    races[x["race_id"]].append(x)
            for xs in races.values():
                c = pick(xs)
                if c is not None:
                    chosen.append(flat(c, "top3") | {"hit": 1 if c["finish"] == 1 else 0})
            s = summarize(chosen)
            s["races"] = len(races)
            out[y] = s
        return out

    def best(key, cond=lambda x: True):
        return lambda xs: max((x for x in xs if cond(x)), key=key, default=None)

    for name in ks:
        ratio = lambda x, n=name: x[n]["ai"] / x[n]["market"]  # noqa: E731
        result["strategies"][f"{name}のズレ最大"] = strategy(best(ratio))
        result["strategies"][f"{name}のズレ最大（AI{name}が出走馬で上位3位以内）"] = strategy(
            lambda xs, n=name, r=ratio: max((x for x in sorted(xs, key=lambda x: -x[n]["ai"])[:3]), key=r))
    result["strategies"]["AI勝率1位"] = strategy(best(lambda x: x["win"]["ai"]))
    result["strategies"]["人気1位"] = strategy(best(lambda x: -x["odds"]))
    for th in [1.0, 1.2, 1.5, 2.0]:
        result["strategies"][f"期待値（AI勝率×単勝オッズ）最大・{th}以上"] = strategy(
            best(lambda x: x["win"]["ai"] * x["odds"], lambda x, t=th: x["win"]["ai"] * x["odds"] >= t))
    (OUT / "gap_backtest.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result["strategies"], ensure_ascii=False, indent=1), file=sys.stderr)


if __name__ == "__main__":
    main()

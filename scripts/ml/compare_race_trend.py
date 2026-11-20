"""レース別の過去の傾向（JRA の「データ分析」ページと同じ考え方）の比較（1年ずつ外す）.

    cd scripts/ml && python compare_race_trend.py   # → data/ml/feature_comparison_race_trend.json

- 対象は重賞（レース名に（ＧⅠ）〜（ＧⅢ））。同じレース（名前から「第N回」・格・冠の部分を除いて同じもの）の過去の回の出走馬から作る
  - 成績表は 2019年からなので、過去は最大7回分。予測するレースより前の回だけを使う
- 足すもの（重賞以外は欠損）
  - rt_editions：過去の回数
  - 年齢（3・4・5・6・7歳以上）が同じ馬の、過去の回での出走数と3着内率
  - 枠番が同じ馬の、過去の回での出走数と3着内率
  - 前走が同じレースだった馬の、過去の回での出走数と3着内率
- 評価：2023〜2026年を1年ずつ外す。全レースと重賞だけの logloss・AUC・上位3頭の3着内率
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402

NAN = float("nan")
GRADE = re.compile(r"（Ｇ[ⅠⅡⅢ]）")
# 冠の付け外し・名前の変更で同じレースが別名になるもの
ALIAS = {"安田記念": "安田記念", "京都大賞典": "京都大賞典", "弥生賞": "弥生賞", "府中牝馬": "府中牝馬ステークス",
         "アイルランドトロフィー": "府中牝馬ステークス"}
TREND = ["rt_editions", "rt_age_n", "rt_age_top3", "rt_waku_n", "rt_waku_top3", "rt_prev_n", "rt_prev_top3"]


def race_key(name: str) -> str:
    n = GRADE.sub("", re.sub(r"第\d+回", "", name)).strip()
    for k, v in ALIAS.items():
        if k in n:
            return v
    m = re.match(r"^.+?(賞|杯)(.+)$", n)
    if m and len(m.group(2)) >= 3 and not m.group(2).startswith("（"):
        n = m.group(2)
    return n


def age_group(age: float) -> int:
    return 7 if age >= 7 else int(age)


def main() -> None:
    names, graded = {}, set()
    for path in sorted((DATA / "json").glob("*/*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["races"]:
            if r.get("race_id"):
                n = r.get("race_name") or ""
                names[r["race_id"]] = race_key(n)
                if GRADE.search(n):
                    graded.add(r["race_id"])
    rows, _ = build_dataset(DATA)
    by_horse = defaultdict(list)
    for r in rows:
        by_horse[r["horse"]].append(r)
    for runs in by_horse.values():
        runs.sort(key=lambda r: (r["date"], r["race_id"]))
        for prev, r in zip([None] + runs[:-1], runs):
            r["_prev_key"] = names.get(prev["race_id"]) if prev and prev["date"] < r["date"] else None

    by_race = defaultdict(list)
    for r in rows:
        by_race[r["race_id"]].append(r)
    past = defaultdict(list)  # レースの名前 → 過去の回の出走馬
    for race_id in sorted(by_race, key=lambda i: (by_race[i][0]["date"], i)):
        rs = by_race[race_id]
        if race_id not in graded:
            for r in rs:
                for f in TREND:
                    r[f] = NAN
            continue
        key = names[race_id]
        hist = [p for p in past[key] if p["date"] < rs[0]["date"]]
        editions = len({p["race_id"] for p in hist})
        for r in rs:
            r["rt_editions"] = editions
            for prefix, same in (
                ("rt_age", lambda p, r=r: r["age"] == r["age"] and p["age"] == p["age"] and age_group(p["age"]) == age_group(r["age"])),
                ("rt_waku", lambda p, r=r: r["waku"] == r["waku"] and p["waku"] == r["waku"]),
                ("rt_prev", lambda p, r=r: r["_prev_key"] is not None and p["_prev_key"] == r["_prev_key"]),
            ):
                sel = [p for p in hist if same(p)]
                r[prefix + "_n"] = len(sel) if editions else NAN
                r[prefix + "_top3"] = sum(p["label"] for p in sel) / len(sel) if sel else NAN
        past[key].extend(rs)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]

    sets = {"base": FEATURES, "+レース別の傾向": FEATURES + TREND}
    rounds = json.loads((OUT / "metrics.json").read_text())["best_iteration"]
    years = ["2023", "2024", "2025", "2026"]
    result: dict = {"graded_races": len({r["race_id"] for r in rows if r["race_id"] in graded}), "sets": {}}

    def score(rs, p):
        y = [r["label"] for r in rs]
        return {"logloss": log_loss(y, p), "auc": roc_auc_score(y, p), "top3_hit": race_top3_hit(rs, p)}

    for name, feats in sets.items():
        per = {}
        for year in years:
            tr = [r for r in rows if r["date"][:4] != year]
            te = [r for r in rows if r["date"][:4] == year]
            m = lambda rs: np.array([[r[f] for f in feats] for r in rs], dtype=np.float64)  # noqa: E731
            b = lgb.train(PARAMS, lgb.Dataset(m(tr), label=[r["label"] for r in tr]), num_boost_round=rounds)
            p = b.predict(m(te))
            gi = [i for i, r in enumerate(te) if r["race_id"] in graded]
            per[year] = {"all": score(te, p), "graded": score([te[i] for i in gi], p[gi])}
            print(name, year, per[year], file=sys.stderr, flush=True)
        result["sets"][name] = {"per_year": per, "mean": {
            part: {k: float(np.mean([v[part][k] for v in per.values()])) for k in ("logloss", "auc", "top3_hit")}
            for part in ("all", "graded")}}
    base = result["sets"]["base"]["per_year"]
    new = result["sets"]["+レース別の傾向"]["per_year"]
    result["years_logloss_better"] = {part: sum(1 for y in years if new[y][part]["logloss"] < base[y][part]["logloss"]) for part in ("all", "graded")}
    (OUT / "feature_comparison_race_trend.json").write_text(json.dumps(result, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v["mean"] for k, v in result["sets"].items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

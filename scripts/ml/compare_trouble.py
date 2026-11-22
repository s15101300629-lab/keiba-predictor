"""前走などで力を出せなかった走（不利・出遅れ・競走中の疾病）を考えた特徴量の比較（前年までで学習して 2024〜2026年を予測）.

    cd scripts/ml && python compare_trouble.py   # → data/ml/feature_comparison_trouble.json

成績表の注記（JRA の成績表 PDF。全年あり。netkeiba の有料の備考は使わない）から、その走で力を出せなかったかを決める
- 不利：〔制裁〕の「(被害馬:○番)」（今も last1_victim・victim_n5 として入っている）
- 出遅れなど：〔発走状況〕の「発進不良〔出遅れ〕」「枠内駐立不良」など
- 競走中の疾病：「競走中に疾病〔鼻出血〕を発症」「競走中に異常歩様」など（完走した走）
足すもの
- B 前走の種類：last1_slow・last1_ill・last1_trouble（どれか）、trouble_n5（近5走の回数）、last1_trouble_fin（前走が不利などのときの前走の着順成分）
- C 補正：近5走から不利などの走を除いた着順成分・タイム差成分の平均、同じ芝ダの能力指数の平均と最高、不利などの走を除いた一番近い走の着順成分・能力指数
評価：全体と、人気薄（成績表の確定人気 6番人気以下）、前走で不利などがあった馬だけの logloss（合計3にそろえた後）
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset, load_races, run_components  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM, race_top3_hit  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

NAN = float("nan")
SLOW = re.compile(r"発進不良|枠内駐立不良|出遅|立ち遅")
ILL = re.compile(r"競走中に(?:疾病|異常歩様)")
B = ["last1_slow", "last1_ill", "last1_trouble", "trouble_n5", "last1_trouble_fin"]
C = ["clean_fin_mean5", "clean_time_mean5", "clean_ab_mean5", "clean_ab_best5", "clean_last_fin", "clean_last_ab"]


def note_flags() -> dict:
    """注記の「○○号は、…」から、馬ごとの出遅れなど・競走中の疾病."""
    out = {}
    for path in sorted((DATA / "json").glob("*/*.json")):
        for r in json.loads(path.read_text(encoding="utf-8"))["races"]:
            rid = r.get("race_id")
            for note in r.get("notes") or []:
                for name, text in re.findall(r"([^\s、,，。〕「」]+?)号は[、,，]([^。]*)", note):  # 2024年途中までは「,」
                    if "競走中止" in text or "取消" in text or "除外" in text:
                        continue
                    if SLOW.search(text):
                        out.setdefault((rid, name), set()).add("slow")
                    if ILL.search(text):
                        out.setdefault((rid, name), set()).add("ill")
    return out


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else NAN


def main() -> None:
    notes = note_flags()
    rows, hist = build_dataset(DATA)
    flags = {}
    for h, runs in hist.horses.items():
        for r in runs:
            f = set(notes.get((r["race_id"], h), set()))
            if r.get("victim"):
                f.add("victim")
            flags[(r["race_id"], h)] = f
    cnt = {k: sum(1 for v in flags.values() if k in v) for k in ("victim", "slow", "ill")}
    print("flag counts", cnt, len(flags), file=sys.stderr, flush=True)
    for row in rows:
        runs = [x for x in hist.horses.get(row["horse"], []) if x["date"] < row["date"]]
        last5 = runs[-5:]
        fl = [flags.get((x["race_id"], row["horse"]), set()) for x in last5]
        last, lf = (last5[-1], fl[-1]) if last5 else (None, set())
        row["last1_slow"] = (1.0 if "slow" in lf else 0.0) if last else NAN
        row["last1_ill"] = (1.0 if "ill" in lf else 0.0) if last else NAN
        row["last1_trouble"] = (1.0 if lf else 0.0) if last else NAN
        row["trouble_n5"] = float(sum(1 for f in fl if f)) if last else NAN
        row["last1_trouble_fin"] = run_components(last["finish"], last["field"], last["time_diff"])[0] if last and lf else NAN
        clean = [x for x, f in zip(last5, fl) if not f]
        comps = [run_components(x["finish"], x["field"], x["time_diff"]) for x in clean]
        row["clean_fin_mean5"] = mean([c[0] for c in comps])
        row["clean_time_mean5"] = mean([c[1] for c in comps])
        ab = [x["ab"] for x in clean if x["surface"] == row["surface"] and x["ab"] is not None]
        row["clean_ab_mean5"] = mean(ab)
        row["clean_ab_best5"] = max(ab) if ab else NAN
        cl = clean[-1] if clean else None
        row["clean_last_fin"] = run_components(cl["finish"], cl["field"], cl["time_diff"])[0] if cl else NAN
        row["clean_last_ab"] = cl["ab"] if cl and cl["ab"] is not None else NAN

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + B + C
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    pop = np.array([r["popularity"] if r["popularity"] else 99 for r in rows])
    troubled = np.array([r["last1_trouble"] == 1.0 for r in rows])
    out = {"flag_counts": cnt, "last1_trouble_rate": round(float(np.mean(troubled)), 4),
           "last1_trouble_top3_rate": round(float(y[troubled].mean()), 4), "sets": {}}
    sets = {"base": FEATURES, "+B 前走の種類": FEATURES + B, "+C 補正": FEATURES + C, "+B+C": cols}
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        res = {}
        for ty in ["2024", "2025", "2026"]:
            vy = str(int(ty) - 1)
            fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
            dfit = lgb.Dataset(X[fit][:, idx], y[fit])
            b = lgb.train(PARAMS, dfit, 6000, valid_sets=[lgb.Dataset(X[val][:, idx], y[val], reference=dfit)],
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            m = lgb.train(PARAMS, lgb.Dataset(X[tr][:, idx], y[tr]), max(b.best_iteration, 50))
            p = normalize_sum(race_id[te], m.predict(X[te][:, idx]))
            yt, lp, tt = y[te], pop[te] >= 6, troubled[te]
            res[ty] = {"logloss": log_loss(yt, p), "auc": roc_auc_score(yt, p), "top3_hit": race_top3_hit(race_id[te], yt, p),
                       "long_logloss": log_loss(yt[lp], p[lp]), "long_auc": roc_auc_score(yt[lp], p[lp]),
                       "trouble_logloss": log_loss(yt[tt], p[tt]), "trouble_auc": roc_auc_score(yt[tt], p[tt]),
                       "trouble_long_n": int((tt & lp).sum()), "trouble_long_logloss": log_loss(yt[tt & lp], p[tt & lp])}
            print(name, ty, res[ty], file=sys.stderr, flush=True)
        res["mean"] = {k: float(np.mean([res[t][k] for t in ["2024", "2025", "2026"]])) for k in res["2024"]}
        if name != "base":
            base = out["sets"]["base"]
            res["years_better"] = {k: sum(res[t][k] < base[t][k] for t in ["2024", "2025", "2026"])
                                   for k in ("logloss", "long_logloss", "trouble_logloss")}
        out["sets"][name] = res
        (OUT / "feature_comparison_trouble.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

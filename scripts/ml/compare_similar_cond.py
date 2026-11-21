"""似た条件（距離・馬場・ペース）での実績を足したときの比較（前年までで学習して 2024〜2026年を予測）.

    cd scripts/ml && python compare_similar_cond.py   # → data/ml/feature_comparison_similar_cond.json

似た条件の走 ＝ 今回と同じ芝ダで、距離 ±200m、馬場の区分（良／稍重以上）が同じ過去走。今回の馬場は発表値（学習では成績表の値）
- sc_n・sc_top3：その走の数と3着内率、sc_ab_best・sc_ab_mean：能力指数の最高と平均、sc_fin_best：着順成分の最高
- sc_strong：その走のうち3着以内だった走の「相手の強さ」（そのレースの他の馬の、そのときの調子の平均）の最高。強い相手に好走したか
- sc_strong_ab：その走での能力指数 − 相手の調子から見た見込み、の平均（相手の強さを考えた走り。見込みは調子の平均を全走で回帰した直線）
さらにペースも同じ走（今回のペースはレース前には分からないので、出走馬の顔ぶれから見た想定ペース。過去走は実際のペース）
- scp_n・scp_top3・scp_ab_best
すべてレースの日より前の走だけ。基準は今の本番（160項目）＋馬主・生産牧場・前年の成績（compare_owner.py の「全部」）
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_owner import GROUPS as OWNER_GROUPS  # noqa: E402
from compare_owner import owner_features  # noqa: E402
from features import FEATURES_BEFORE_SIMILAR as FEATURES, build_dataset, run_components  # noqa: E402  # 基準は172項目（馬主など入り）
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
SC = ["sc_n", "sc_top3", "sc_ab_best", "sc_ab_mean", "sc_fin_best", "sc_strong", "sc_strong_ab"]
SCP = ["scp_n", "scp_top3", "scp_ab_best"]


def expected_pace(nige: float, front_ratio: float) -> int:
    """出走馬の顔ぶれから見た想定ペース（app/lib/ml/features.ts の expectedPace と同じ）。0=スロー 1=ミドル 2=ハイ."""
    if nige >= 2 or front_ratio >= 0.4:
        return 2
    if nige == 0 and front_ratio < 0.25:
        return 0
    return 1


def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else NAN


def main() -> None:
    rows, hist = build_dataset(DATA)
    # 過去走ごとの「相手の強さ」（他の馬のそのときの調子の平均）
    strength = {}
    for rid, entries in hist.by_race.items():
        forms = [(h, run["form_at"]) for h, run in entries]
        for h, run in entries:
            strength[(rid, h)] = mean([f for o, f in forms if o != h])
    # 相手の強さから見た能力指数の見込み（全走で1本の直線。ラベルは使わない）
    xs, ys = [], []
    for h, runs in hist.horses.items():
        for r in runs:
            s = strength.get((r["race_id"], h), NAN)
            if r["ab"] is not None and not math.isnan(s):
                xs.append(s); ys.append(r["ab"])
    slope, intercept = np.polyfit(xs, ys, 1)
    print("ab ~ field form:", slope, intercept, file=sys.stderr, flush=True)

    for r in rows:
        runs = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"]]
        d, wet = r["distance"], (not math.isnan(r["going"]) and r["going"] >= 1)
        sel = [x for x in runs if x["surface"] == r["surface"] and x["distance"] and not math.isnan(d) and abs(x["distance"] - d) <= 200
               and x["going"] is not None and (x["going"] >= 1) == wet]
        top3 = [x for x in sel if x["finish"] is not None and x["finish"] <= 3]
        ab = [x["ab"] for x in sel if x["ab"] is not None]
        r["sc_n"] = len(sel)
        r["sc_top3"] = len(top3) / len(sel) if sel else NAN
        r["sc_ab_best"] = max(ab) if ab else NAN
        r["sc_ab_mean"] = mean(ab)
        r["sc_fin_best"] = max(run_components(x["finish"], x["field"], x["time_diff"])[0] for x in sel) if sel else NAN
        st = [strength.get((x["race_id"], r["horse"]), NAN) for x in top3]
        st = [s for s in st if not math.isnan(s)]
        r["sc_strong"] = max(st) if st else NAN
        adj = [x["ab"] - (slope * strength[(x["race_id"], r["horse"])] + intercept) for x in sel
               if x["ab"] is not None and not math.isnan(strength.get((x["race_id"], r["horse"]), NAN))]
        r["sc_strong_ab"] = mean(adj)
        pace = expected_pace(r["field_nige_cands"], r["field_front_ratio"])
        selp = [x for x in sel if x["pace"] == pace]
        abp = [x["ab"] for x in selp if x["ab"] is not None]
        r["scp_n"] = len(selp)
        r["scp_top3"] = sum(1 for x in selp if x["finish"] is not None and x["finish"] <= 3) / len(selp) if selp else NAN
        r["scp_ab_best"] = max(abp) if abp else NAN

    owner = owner_features()
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    oext = [f for g in OWNER_GROUPS.values() for f in g]
    cols = FEATURES + SC + SCP
    X = np.array([[r[f] for f in FEATURES + SC + SCP] for r in rows], dtype=np.float64)  # 馬主など・似た条件は features.py に入った
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    base = FEATURES
    sets = {"基準(160＋馬主など)": base, "+似た条件": base + SC, "+似た条件＋ペース": base + SC + SCP}
    out = {"coverage": {f: round(float(np.mean(~np.isnan(X[:, cols.index(f)]))), 3) for f in ("sc_top3", "sc_strong", "scp_top3")}, "sets": {}}
    first = None
    for name, fs in sets.items():
        idx = [cols.index(f) for f in fs]
        print(name, file=sys.stderr, flush=True)
        res = evaluate(X[:, idx], y, year, race_id, finish, PARAMS)
        if first is None:
            first = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss"] < first[t]["logloss"] for t in ["2024", "2025", "2026"])
        out["sets"][name] = res
        (OUT / "feature_comparison_similar_cond.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

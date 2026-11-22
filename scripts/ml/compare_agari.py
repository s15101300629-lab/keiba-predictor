"""末脚（上がり）の比較（前年までで学習して 2024〜2026年を予測。基準は今の本番 186項目（コース×4角の位置入り））.

    cd scripts/ml && python compare_agari.py   # → data/ml/feature_comparison_agari.json

依頼者の考え（2026-11-22）：上がりの速さは、その馬の強さ（末脚があるか）につながる。
今のモデルにも推定上がり（成績表の隊列から。本物との差 0.14秒）の偏差・順位の直近加重平均と最高は入っている。ここでは「確かさ」と「位置との組み合わせ」を足す。
G1 上がり上位の確かさ：近5走・近10走（同じ芝ダ）で、上がりがレースの上位3分の1だった割合、1位だった回数
G2 位置×上がり：近5走で「最後のコーナーが前3分の1で、上がりも上位3分の1」だった割合（前で運んで速い上がり＝強い勝ち方）、
   「後ろ3分の1から上がり1位」の割合（追い込みの末脚）
G3 上がりと着順のずれ：近5走で、上がりの順位÷頭数 − 着順÷頭数 の平均（速い上がりでも届かない＝位置取りで損している）、
   上位クラス（今回の1着賞金以上）の走での上がりの偏差の平均
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import FEATURES, build_dataset  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

NAN = float("nan")
G1 = ["agt_top_r5", "agt_top_r10", "agt_first_n10"]
G2 = ["agp_front_fast_r5", "agp_back_first_r5"]
G3 = ["agf_gap5", "agc_dev_up"]


def main() -> None:
    rows, hist = build_dataset(DATA)
    for r in rows:
        runs = [x for x in hist.horses.get(r["horse"], []) if x["date"] < r["date"] and x["surface"] == r["surface"]][::-1]
        ag = [x for x in runs if x["ag_rank"] is not None]
        a5, a10 = ag[:5], ag[:10]
        r["agt_top_r5"] = sum(1 for x in a5 if x["ag_rank"] <= 1 / 3) / len(a5) if a5 else NAN
        r["agt_top_r10"] = sum(1 for x in a10 if x["ag_rank"] <= 1 / 3) / len(a10) if a10 else NAN
        r["agt_first_n10"] = sum(1 for x in a10 if x["field"] and x["ag_rank"] <= 1 / x["field"] + 1e-9)
        p5 = [x for x in a5 if x["pos_last"] and x["field"]]
        r["agp_front_fast_r5"] = sum(1 for x in p5 if x["pos_last"] / x["field"] <= 1 / 3 and x["ag_rank"] <= 1 / 3) / len(p5) if p5 else NAN
        r["agp_back_first_r5"] = sum(1 for x in p5 if x["pos_last"] / x["field"] > 2 / 3 and x["ag_rank"] <= 1 / x["field"] + 1e-9) / len(p5) if p5 else NAN
        g = [x["ag_rank"] - x["finish"] / x["field"] for x in a5 if x["finish"] and x["field"]]
        r["agf_gap5"] = sum(g) / len(g) if g else NAN
        today = r["prize_1st"]
        up = [x["ag_dev"] for x in ag[:10] if x["ag_dev"] is not None and x["prize"] is not None and not math.isnan(today) and x["prize"] >= today]
        r["agc_dev_up"] = sum(up) / len(up) if up else NAN

    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    cols = FEATURES + G1 + G2 + G3
    X = np.array([[r[f] for f in cols] for r in rows], dtype=np.float64)
    y = np.array([r["label"] for r in rows], dtype=float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    sets = {"基準(186。c4 入り)": FEATURES, "+G1 上がり上位の確かさ": FEATURES + G1, "+G2 位置×上がり": FEATURES + G2,
            "+G3 上がりと着順のずれ・上位クラスの上がり": FEATURES + G3, "+全部": cols}
    out = {"coverage": {f: round(float(np.mean(~np.isnan(X[:, cols.index(f)]))), 3) for f in G1 + G2 + G3}, "sets": {}}
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
        (OUT / "feature_comparison_agari.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

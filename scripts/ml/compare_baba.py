"""クッション値・含水率・使用コース（A〜D）を入力に足す比較（2026-11-23、依頼者の案）.

    cd scripts/ml && python compare_baba.py   # → data/ml/feature_comparison_baba.json

- クッション値・含水率：JRA の過去の PDF（parse_baba.py → data/baba/baba.csv）。その日・その競馬場の朝の測定値。2020年〜（クッション値は 2020年秋〜）
  - baba_cushion：芝のクッション値（芝のレースだけ）
  - baba_moist：そのレースの芝ダの含水率（ゴール前と4コーナーの平均）
  - baba_moist_rel：含水率 − その競馬場・芝ダの、その年より前の平均（競馬場で砂が違い、同じ数でも意味が違うため）
- 使用コース：成績表の「(芝Ｃコース)」（その日の芝。2017年〜）。course_letter A=0 B=1 C=2 D=3（芝のレースだけ）
前年までで学習して 2024〜2026年を予測（tune_params.evaluate）。基準は今の本番（features.FEATURES）
"""

from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import evaluate  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VENUE_CODE = {"札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05", "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"}
LETTER = {"A": 0, "B": 1, "C": 2, "D": 3, "Ａ": 0, "Ｂ": 1, "Ｃ": 2, "Ｄ": 3}


def load_baba() -> dict:
    out = {}
    with (ROOT / "data/baba/baba.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out[(r["date"], VENUE_CODE[r["venue"]])] = r
    return out


def load_course_letters() -> dict:
    out = {}
    for f in (DATA / "json").glob("*/*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        m = re.search(r"[（(]芝([A-DＡ-Ｄ])コース[）)]", json.dumps(d, ensure_ascii=False))
        if m and d.get("date"):
            out[(d["date"], str(d.get("venue_code")).zfill(2))] = LETTER[m.group(1)]
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def main() -> None:
    rows, _ = features.build_dataset(DATA)
    rows = [r for r in rows if r["date"] >= TRAIN_FROM]
    baba, letters = load_baba(), load_course_letters()
    print("baba", len(baba), "letters", len(letters), file=sys.stderr, flush=True)
    # 含水率の、競馬場・芝ダ・年ごとの平均（その年より前の年の平均を引く）
    by_year = defaultdict(list)
    for (d, vc), r in baba.items():
        for surf, k in ((0, "turf"), (1, "dirt")):
            v = (num(r[f"{k}_goal"]) + num(r[f"{k}_4c"])) / 2
            if not np.isnan(v):
                by_year[(vc, surf, d[:4])].append(v)
    extra = {k: [] for k in ("baba_cushion", "baba_moist", "baba_moist_rel", "course_letter")}
    for r in rows:
        vc, surf, d = r["race_id"][4:6], int(r["surface"]), r["date"]
        b = baba.get((d, vc))
        cush = num(b["cushion"]) if b and surf == 0 else float("nan")
        k = "turf" if surf == 0 else "dirt"
        moist = (num(b[f"{k}_goal"]) + num(b[f"{k}_4c"])) / 2 if b else float("nan")
        prev = [v for y in range(2020, int(d[:4])) for v in by_year.get((vc, surf, str(y)), [])]
        rel = moist - float(np.mean(prev)) if prev and not np.isnan(moist) else float("nan")
        extra["baba_cushion"].append(cush)
        extra["baba_moist"].append(moist)
        extra["baba_moist_rel"].append(rel)
        extra["course_letter"].append(float(letters.get((d, vc), float("nan"))) if surf == 0 else float("nan"))
    X0 = np.array([[r[f] for f in features.FEATURES] for r in rows], dtype=np.float64)
    E = {k: np.array(v) for k, v in extra.items()}
    y = np.array([r["label"] for r in rows], float)
    year = np.array([r["date"][:4] for r in rows])
    race_id = np.array([r["race_id"] for r in rows])
    finish = np.array([r["finish"] if r["finish"] is not None else 99 for r in rows])
    cover = {k: {yy: round(float(np.mean(~np.isnan(v[year == yy]))), 3) for yy in ("2020", "2022", "2024", "2026")} for k, v in E.items()}
    print("cover", cover, file=sys.stderr, flush=True)
    variants = {"今": [], "+クッション値・含水率": ["baba_cushion", "baba_moist", "baba_moist_rel"], "+使用コース": ["course_letter"],
                "+全部": ["baba_cushion", "baba_moist", "baba_moist_rel", "course_letter"]}
    out = {"cover": cover, "variants": {}}
    base = None
    for name, ks in variants.items():
        X = np.column_stack([X0] + [E[k] for k in ks]) if ks else X0
        res = evaluate(X, y, year, race_id, finish, PARAMS)
        if base is None:
            base = res
        else:
            res["years_logloss_better"] = sum(res[t]["logloss_sum3"] < base[t]["logloss_sum3"] for t in ("2024", "2025", "2026"))
        out["variants"][name] = res
        (OUT / "feature_comparison_baba.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

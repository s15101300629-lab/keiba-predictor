"""コース（競馬場・芝ダ・距離）ごとの枠番別の成績を、成績表から集計する（画面の「枠順」の項目用）.

    cd scripts/ml && python course_waku.py   # → data/ml/course_waku.json

モデルの比較では枠の有利不利に効果がなかった（不採用）。これは表示用。
"""

import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROM_YEAR = "2023"


def main() -> None:
    agg: dict[str, dict[str, list[int]]] = {}
    years = set()
    for path in sorted(glob.glob(str(ROOT / "data/jra-seiseki/csv/*/results.csv"))):
        year = Path(path).parent.name
        if year < FROM_YEAR:
            continue
        years.add(year)
        for r in csv.DictReader(open(path, encoding="utf-8")):
            if r["surface"] not in ("芝", "ダート") or not r["waku"] or not r["finish"].isdigit():
                continue
            course = f"{r['venue']}-{r['surface']}-{r['distance']}"
            cell = agg.setdefault(course, {}).setdefault(r["waku"], [0, 0])
            cell[0] += 1
            cell[1] += int(r["finish"]) <= 3
    out = {"years": sorted(years), "fields": ["starts", "top3"], "courses": agg}
    (ROOT / "data/ml/course_waku.json").write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(len(agg), "courses")


if __name__ == "__main__":
    main()

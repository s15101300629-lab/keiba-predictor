"""JRA の馬場情報（クッション値・含水率）の過去の PDF を読んで data/baba/baba.csv にする（2026-11-23）.

    python scripts/ml/parse_baba.py   # data/baba/pdf/{年}pdf_{競馬場}{回}.pdf → data/baba/baba.csv

PDF は https://www.jra.go.jp/keiba/baba/archive/{年}.html から（2020〜2026年、開催ごとに1つ）。pdftotext -layout で文字にして読む
- 2020〜2024年：何日分かずつの表（日付の範囲と曜日の列。芝のクッション値・芝とダートの含水率（ゴール前・4コーナー））
- 2025年〜：1日1行の表（使用コース A〜D・クッション値・含水率）
2020年の初めはクッション値がない。2021年中京6回の PDF は文字を読めない
"""

from __future__ import annotations

import csv
import re
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "baba" / "pdf"
OUT = ROOT / "data" / "baba" / "baba.csv"
VENUES = {"sapporo": "札幌", "hakodate": "函館", "fukushima": "福島", "niigata": "新潟", "tokyo": "東京", "nakayama": "中山",
          "chukyo": "中京", "kyoto": "京都", "hanshin": "阪神", "kokura": "小倉"}
WD = "月火水木金土日"
NUM = r"(\d+(?:\.\d+)?)"


def nums(line: str) -> list[float]:
    return [float(x) for x in re.findall(r"(?<![\d.])\d+\.\d+|(?<![\d.])\d+(?![\d.])", line)]


def parse_blocks(text: str, year: int) -> list[dict]:
    rows: dict[str, dict] = {}
    lines = text.splitlines()
    start = None
    weekdays: list[str] = []
    mode = None
    for line in lines:
        z = line.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        m = re.search(r"(\d{4})年(\d+)月(\d+)日\s*(?:～|から|~)", z)
        if m:
            start = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        if re.search(r"[月火水木金土日]曜日", z) and not re.search(r"\d+\.\d", z):
            weekdays = re.findall(r"([月火水木金土日])曜日", z)
            continue
        if start is None or not weekdays:
            continue
        # 曜日の列 → 日付（範囲の初日から、曜日が合う日を順に）
        def dates_for(n_vals: int) -> list[str]:
            ws = weekdays[-n_vals:] if n_vals <= len(weekdays) else weekdays
            out, d = [], start
            for w in weekdays:
                while WD[d.weekday()] != w:
                    d += timedelta(days=1)
                out.append(d.isoformat())
                d += timedelta(days=1)
            return out[len(weekdays) - len(ws):]
        if "クッション値" in z:
            key = "cushion"
        elif "芝コース含水率" in z:
            mode, key = "turf", "turf_goal"
        elif "ダートコース含水率" in z:
            mode, key = "dirt", "dirt_goal"
        elif "コーナー" in z and mode:
            key = f"{mode}_4c"
        else:
            continue
        vals: list[float] = []
        if "コーナー" in z:
            vals = nums(re.sub(r"^.*?コーナー", "", z))
        elif "ゴール前" in z:
            vals = nums(re.sub(r"^.*?ゴール前", "", z))
        elif "クッション値" in z:
            vals = nums(re.sub(r"^.*?クッション値", "", z))
        if not vals:
            continue
        for d, v in zip(dates_for(len(vals)), vals):
            rows.setdefault(d, {})[key] = v
    return [{"date": d, **v} for d, v in rows.items()]


def parse_rows(text: str, year: int) -> list[dict]:
    out = []
    for line in text.splitlines():
        z = line.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
        m = re.search(r"(\d+)月\s*(\d+)日\s+[月火水木金土日]曜日\s+([A-D])\s+\d+:\d+\s+" + NUM + r"\s+\d+:\d+\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM + r"\s+" + NUM, z)
        if not m:
            continue
        out.append({"date": date(year, int(m.group(1)), int(m.group(2))).isoformat(), "course": m.group(3), "cushion": float(m.group(4)),
                    "turf_goal": float(m.group(5)), "turf_4c": float(m.group(6)), "dirt_goal": float(m.group(7)), "dirt_4c": float(m.group(8))})
    return out


def main() -> None:
    rows = []
    for f in sorted(SRC.glob("*.pdf")):
        m = re.match(r"(\d{4})pdf_([a-z]+)(\d+)\.pdf", f.name)
        if not m:
            continue
        year, venue = int(m.group(1)), VENUES[m.group(2)]
        text = subprocess.run(["pdftotext", "-layout", str(f), "-"], capture_output=True, text=True).stdout
        got = parse_rows(text, year) if "使用コース" in text else parse_blocks(text, year)
        if not got:
            print("読めない:", f.name, file=sys.stderr)
        for r in got:
            rows.append({"venue": venue, "kai": int(m.group(3)), **r})
    keys = ["date", "venue", "kai", "course", "cushion", "turf_goal", "turf_4c", "dirt_goal", "dirt_4c"]
    rows.sort(key=lambda r: (r["date"], r["venue"]))
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})
    print(f"{len(rows)} 日・競馬場", file=sys.stderr)


if __name__ == "__main__":
    main()

"""JRA の馬場情報のページから、最近のクッション値・含水率を data/baba/baba.csv に足す（2026-11-24）.

    python scripts/ml/collect_baba_live.py

過去の PDF（parse_baba.py）は開催が終わってから出るので、開催中・直近の分はこちらで取る。毎日の答え合わせ（day-check.yml）と
再学習（ml-train.yml）の前に動かす。PDF の行がある日・競馬場はそのまま（PDF を優先）
- https://www.jra.go.jp/keiba/baba/_data_cushion.html：競馬場ごと（<div id="rcA" title="東京">）の、測定の日時とクッション値
- https://www.jra.go.jp/keiba/baba/_data_moist.html：同じく、芝・ダートの含水率（ゴール前 mg・4コーナー m4c）
同じ日に何回か測ったときは、一番早い測定（PDF と同じ朝の値）を使う。ページに年がないので、今日（日本時間）から決める
使用コースは学習では成績表から読む。答え合わせ（結果ページには使用コースがない）のため、今週の分を競馬場ごとのページから取って入れる
"""

from __future__ import annotations

import csv
import re
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "baba" / "baba.csv"
BASE = "https://www.jra.go.jp/keiba/baba/"
UA = "keiba-predictor (personal research)"
KEYS = ["date", "venue", "kai", "course", "cushion", "turf_goal", "turf_4c", "dirt_goal", "dirt_4c"]


def get(name: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(BASE + name, headers={"User-Agent": UA}), timeout=30) as r:
        return r.read().decode("cp932", errors="replace")


def when(text: str, today) -> tuple[str, str] | None:
    m = re.search(r"(\d+)月(\d+)日.*?(\d+)時(\d+)分", text)
    if not m:
        return None
    mo, d = int(m.group(1)), int(m.group(2))
    y = today.year - 1 if mo > today.month + 1 else today.year  # 1月に前の年の12月の値を見るとき
    return f"{y}-{mo:02d}-{d:02d}", f"{int(m.group(3)):02d}:{m.group(4)}"


def units(page: str):
    for block in re.split(r'<div id="rc[A-Z]"', page)[1:]:
        venue = (re.search(r'title="([^"]+)"', block) or [None, ""])[1]
        for u in block.split('<div class="unit">')[1:]:
            yield venue, u


def main() -> None:
    today = datetime.now(timezone(timedelta(hours=9))).date()
    got: dict[tuple[str, str], dict] = {}
    for venue, u in units(get("_data_cushion.html")):
        t = when(re.sub(r"<[^>]+>", " ", u), today)
        v = re.search(r'class="cushion">\s*([\d.]+)', u)
        if t and v:
            r = got.setdefault((t[0], venue), {"_tc": "99:99", "_tm": "99:99"})
            if t[1] < r["_tc"]:
                r.update({"_tc": t[1], "cushion": v.group(1)})
    page = get("_data_moist.html")
    for block in re.split(r'<div id="rc[A-Z]"', page)[1:]:
        venue = (re.search(r'title="([^"]+)"', block) or [None, ""])[1]
        for u in block.split('<div class="unit">')[1:]:
            t = when(re.sub(r"<[^>]+>", " ", u), today)
            tg = re.search(r'class="turf">.*?class="mg"[^>]*>\s*([\d.]+).*?class="m4c"[^>]*>\s*([\d.]+)', u, re.S)
            dg = re.search(r'class="dirt">.*?class="mg"[^>]*>\s*([\d.]+).*?class="m4c"[^>]*>\s*([\d.]+)', u, re.S)
            if t and (tg or dg):
                r = got.setdefault((t[0], venue), {"_tc": "99:99", "_tm": "99:99"})
                if t[1] < r["_tm"]:
                    r.update({"_tm": t[1], "turf_goal": tg.group(1) if tg else "", "turf_4c": tg.group(2) if tg else "",
                              "dirt_goal": dg.group(1) if dg else "", "dirt_4c": dg.group(2) if dg else ""})
    # 使用コース（競馬場ごとのページ。今週の分）：その競馬場の今日以降の行に入れる
    course = {}
    for name in ("index.html", "index2.html", "index3.html"):
        try:
            page = get(name)
        except Exception:  # noqa: BLE001
            continue
        v = re.search(r"第\d+回([^\s<（(]+?)競馬", page)
        c = re.search(r"<h3>使用コース</h3>.*?<strong>\s*([A-DＡ-Ｄ])\s*</strong>", page, re.S)
        if v and c:
            course[v.group(1)] = c.group(1).translate(str.maketrans("ＡＢＣＤ", "ABCD"))
    for (d, venue), r in got.items():
        if venue in course and d >= (today - timedelta(days=3)).isoformat():
            r["course"] = course[venue]
    rows = list(csv.DictReader(OUT.open(encoding="utf-8"))) if OUT.exists() else []
    have = {(r["date"], r["venue"]) for r in rows}
    added = 0
    for (d, venue), r in sorted(got.items()):
        if (d, venue) in have or not venue:
            continue
        rows.append({"date": d, "venue": venue, "kai": "", "course": r.get("course", ""), **{k: r.get(k, "") for k in KEYS[4:]}})
        added += 1
    rows.sort(key=lambda r: (r["date"], r["venue"]))
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=KEYS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in KEYS})
    print(f"馬場情報のページ：{len(got)} 日・競馬場、足した {added}", file=sys.stderr)


if __name__ == "__main__":
    main()

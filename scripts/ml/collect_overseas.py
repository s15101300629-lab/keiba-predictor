"""海外の走の検証用に、netkeiba の戦績（全列）を集める（2026-11-23、ジャパンカップの外国馬のため）.

    cd scripts/ml && python collect_overseas.py [追加の馬ID ...]   # → data/overseas/histories.json（再開できる）

対象（どれも重賞の出走馬か、予測する出走馬）
- 成績表 2017年〜で [外]（外国馬）の走がある馬（race_horses.json で馬番 → netkeiba の馬ID）
- 重賞の出走馬の戦績（data/pedigree/histories.json）に海外の走がある馬（遠征帰りの日本馬）
- 引数で渡した馬ID（ジャパンカップの外国馬など）
取得は db.netkeiba.com/horse/result/{id}/ を5秒に1ページ。User-Agent はこのアプリの名前
保存：馬ID → {"names": 成績表での馬名, "rows": [列名 → 値]}
"""

from __future__ import annotations

import csv
import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "overseas" / "histories.json"
UA = "keiba-predictor (personal research)"
WAIT = 5.0
JRA = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
NAR = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広", "ばんえい"]


def is_overseas(venue: str) -> bool:
    return bool(venue) and not any(v in venue for v in JRA + NAR)


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def parse(page: str) -> list[dict]:
    tb = re.search(r"<table[^>]*db_h_race_results.*?</table>", page, re.S)
    if not tb:
        return []
    rows = re.findall(r"<tr.*?</tr>", tb.group(0), re.S)
    head = [clean(c).replace(" ", "") for c in re.findall(r"<th.*?</th>", rows[0], re.S)]
    out = []
    for r in rows[1:]:
        cells = [clean(c) for c in re.findall(r"<td.*?</td>", r, re.S)]
        out.append({h: c for h, c in zip(head, cells) if h and h not in ("映像", "厩舎コメント", "備考")})
    return out


def fetch(url: str) -> str | None:
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=30) as res:
                return res.read().decode("utf-8", errors="replace")
        except Exception as e:  # noqa: BLE001
            print("  !", url, e, file=sys.stderr)
            time.sleep(10 * (attempt + 1))
    return None


def targets(extra: list[str]) -> dict[str, set[str]]:
    rh = json.loads((ROOT / "data/pedigree/all/race_horses.json").read_text(encoding="utf-8"))
    names: dict[str, set[str]] = {}
    want_foreign: set[str] = set()
    for f in sorted((ROOT / "data/jra-seiseki/csv").glob("*/results.csv")):
        with f.open(encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                hid = rh.get(r["race_id"], {}).get(r["umaban"])
                if not hid:
                    continue
                names.setdefault(hid, set()).add(r["horse"])
                if r["horse_mark"] == "[外]":
                    want_foreign.add(hid)
    hist = json.loads((ROOT / "data/pedigree/histories.json").read_text(encoding="utf-8"))
    returnees = {k for k, v in hist.items() if any(is_overseas(x.get("開催", "")) for x in v)}
    ids = want_foreign | returnees | set(extra)
    print(f"外国馬 {len(want_foreign)}・遠征帰り {len(returnees)}・指定 {len(extra)} → {len(ids)}頭", file=sys.stderr)
    return {i: names.get(i, set()) for i in sorted(ids)}


def main() -> None:
    extra = [a for a in sys.argv[1:] if re.fullmatch(r"\d{10}", a)]
    todo = targets(extra)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    n = 0
    for hid, nm in todo.items():
        if hid in data and data[hid].get("rows"):
            data[hid]["names"] = sorted(nm)
            continue
        base = f"https://db.netkeiba.com/horse/result/{hid}/"
        page = fetch(base)
        time.sleep(WAIT)
        if page is None:
            continue
        rows = parse(page)
        data[hid] = {"names": sorted(nm), "title": clean((re.search(r"<title>(.*?)</title>", page) or [None, ""])[1]), "rows": rows}
        n += 1
        if n % 20 == 0:
            OUT.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
            print(f"  {n}", file=sys.stderr, flush=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    ov = sum(1 for v in data.values() for r in v["rows"] if is_overseas(r.get("開催", "")))
    print(f"done: {len(data)}頭・海外の走 {ov}", file=sys.stderr)


if __name__ == "__main__":
    main()

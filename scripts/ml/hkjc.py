"""香港の走の上がり・位置取り・タイム差（HKJC の公式サイトの区間タイム）。2026-11-26、依頼者の指示.

    cd scripts/ml && python hkjc.py build   # data/overseas/histories.json の馬の香港の走 → data/overseas/hkjc.json（取ったページの中身をためる）

- ページ：https://racing.hkjc.com/racing/information/English/Racing/DisplaySectionalTime.aspx?RaceDate=日/月/年&RaceNo=N
  （全レース・全馬の 400m ごとの区間タイムと、最後の2区間は 200m ごと。各区間の終わりの位置）
- netkeiba の戦績の日付・R（レース番号。空のときは 1〜11R を順に）・馬番・着順・距離で、同じ馬の行を決める
- 計算（features.py の add_ability の推定上がりと同じ意味）：上がり＝最後から2つ目の区間の後半 200m＋最後の区間 400m、
  ag_dev＝完走馬の上がりの平均−自分、ag_rank＝上がりの順位÷完走頭数、early_gap＝600m 地点（走破タイム−上がり）の先頭との差、
  fade＝自分の上がり−レースの上がり（先頭の区間タイム）、pos_first＝最初の区間の位置、pos_last＝最後から2つ目の区間の位置、time_diff＝勝ち馬との差
app/api/races/[raceId]/route.ts の hkRun が同じ計算（ファイルにないレースはその場で取る）
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "overseas" / "hkjc.json"
UA = "keiba-predictor (personal research)"
HK_VENUES = ("シャティン", "ハッピーヴァレー", "ハッピーバレー", "香港")
_cache: dict | None = None


def _num(s: str) -> float | None:
    m = re.match(r"^(?:(\d+):)?(\d+\.\d+)$", s.strip())
    return None if not m else int(m.group(1) or 0) * 60 + float(m.group(2))


def parse(page: str) -> dict | None:
    """区間タイムのページ → {distance, race_last600, horses: [{no, finish, pos: [...], last600, time}]}."""
    dm = re.search(r"Race\s+\d+\s.{0,80}?-\s*(\d{3,4})M\s*-", re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", page)))
    if not dm:
        return None
    rs = page.find("Sectional Time:")
    race = None
    if rs >= 0:
        cells = re.findall(r"<td[^>]*>(.*?)</td>", page[rs:rs + 4000], re.S)[1:]
        secs = []
        for c in cells:
            t = re.sub(r"<[^>]+>", " ", html.unescape(c)).split()
            if not t or _num(t[0]) is None:
                break
            secs.append([_num(x) for x in t if _num(x) is not None])
        if len(secs) >= 2 and len(secs[-2]) >= 3:
            race = secs[-2][2] + secs[-1][0]
    horses = []
    for tr in re.findall(r"<tr>(.*?)</tr>", page, re.S):
        if "horseid=" not in tr:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
        fin = re.sub(r"<[^>]+>|\s", "", cells[0])
        no = re.sub(r"<[^>]+>|\s", "", cells[1])
        if not fin.isdigit() or not no.isdigit():
            continue
        pos, secs = [], []
        for c in cells[3:]:
            p = re.search(r'class="f_fl">\s*(\d+)\s*<', c)
            if not p:
                continue
            pos.append(int(p.group(1)))
            body = re.sub(r"<[^>]+>", " ", html.unescape(c.split("</p>", 1)[1] if "</p>" in c else c)).split()
            secs.append([_num(x) for x in body if _num(x) is not None])
        total = _num(re.sub(r"<[^>]+>|\s", "", cells[-1]))
        last600 = secs[-2][2] + secs[-1][0] if len(secs) >= 2 and len(secs[-2]) >= 3 and secs[-1] else None
        horses.append({"no": int(no), "finish": int(fin), "pos": pos, "last600": last600, "time": total})
    return {"distance": int(dm.group(1)), "race_last600": race, "horses": horses} if horses else None


def fetch(date: str, no: int) -> dict | None:
    """date は YYYY-MM-DD。ためたものがあればそれを使う."""
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = f"{date}|{no}"
    if key not in _cache:
        y, m, d = date.split("-")
        url = f"https://racing.hkjc.com/racing/information/English/Racing/DisplaySectionalTime.aspx?RaceDate={d}/{m}/{y}&RaceNo={no}"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                page = r.read().decode("utf-8", "replace")
            _cache[key] = parse(page)
        except Exception as ex:  # noqa: BLE001
            print("  ! hkjc", key, ex, file=sys.stderr)
            return None
        time.sleep(1.5)
    return _cache[key]


def save() -> None:
    if _cache is not None:
        CACHE.write_text(json.dumps(_cache, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def metrics(race: dict, umaban: int) -> dict | None:
    """そのレースの、馬番 umaban の馬の上がり・位置取り・タイム差."""
    if race["race_last600"] is None:  # レース全体の上がりがないと fade が出せないので、このレースの区間タイムは使わない
        return None
    fin = [h for h in race["horses"] if h["last600"] is not None and h["time"] is not None]
    me = next((h for h in race["horses"] if h["no"] == umaban), None)
    if not me or len(fin) < 3 or me not in fin:
        return None
    cum = [h["time"] - h["last600"] for h in fin]
    a = me["last600"]
    m = sum(h["last600"] for h in fin) / len(fin)
    srt = sorted(h["last600"] for h in fin)
    win = min(h["time"] for h in fin)
    return {"ag_dev": round(m - a, 4), "ag_rank": round((1 + srt.index(a)) / len(fin), 4), "early_gap": round(me["time"] - a - min(cum), 4),
            "fade": round(a - race["race_last600"], 4),
            "pos_first": me["pos"][0] if me["pos"] else None, "pos_last": me["pos"][-2] if len(me["pos"]) >= 2 else None,
            "time_diff": round(me["time"] - win, 1)}


def hk_run(venue: str, date: str, race_no: str, distance: int, umaban: str, finish: str) -> dict | None:
    """netkeiba の香港の走1つ → metrics（見つからなければ None）."""
    if not any(v in venue for v in HK_VENUES) or not str(umaban).isdigit() or not str(finish).isdigit():
        return None
    nos = [int(race_no)] if str(race_no).isdigit() else list(range(1, 12))
    for no in nos:
        race = fetch(date, no)
        if not race or abs(race["distance"] - distance) > 10:
            continue
        me = next((h for h in race["horses"] if h["no"] == int(umaban)), None)
        if me and me["finish"] == int(finish):
            return metrics(race, int(umaban))
    return None


def build() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    hist = json.loads((ROOT / "data/overseas/histories.json").read_text(encoding="utf-8"))
    n = ok = 0
    for v in hist.values():
        for r in v["rows"]:
            if any(x in r.get("開催", "") for x in HK_VENUES) and (r.get("距離") or "").startswith(("芝", "ダ")):
                n += 1
                d = re.match(r"^[芝ダ](\d+)", r["距離"])
                ok += hk_run(r["開催"], r["日付"].replace("/", "-"), r.get("R", ""), int(d.group(1)), r.get("馬番", ""), r.get("着順", "")) is not None
                if n % 20 == 0:
                    save()
                    print(n, ok, file=sys.stderr, flush=True)
    save()
    print(f"香港の走 {n}、取れた {ok}", file=sys.stderr)


if __name__ == "__main__":
    if sys.argv[1:] == ["build"]:
        build()

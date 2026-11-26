"""フランスの走の上がり・位置取り・タイム差（France Galop の公式の区間タイム。2026-11-26、依頼者の指示）.

    cd scripts/ml && python francegalop.py build      # data/overseas/histories.json の馬のフランスの走
    cd scripts/ml && python francegalop.py upcoming   # data/overseas/upcoming_overseas.json の馬（出走予定の重賞の馬・外国馬）の戦績から
    → data/overseas/francegalop.json（日付|競馬場|R → レースの中身。null は PDF がない）

- PDF：https://www7.france-galop.com/Casaques/Tracking//{YYYYMMDD}{LON|CHA|DEA}{RR}_last_times_fr.pdf
  （パリロンシャン・シャンティイ・ドーヴィル、2020年〜。最後の 1000m を 200m ごとに、各馬の通過タイムと位置）
- 計算は hkjc.py の metrics と同じ（上がり＝到着の通過タイム − 600m 地点の通過タイム。pos_first はない（最後の 1000m しかない）、
  pos_last＝400m 地点の位置、time_diff＝到着の通過タイムの勝ち馬との差）
- アプリ（route.ts）と答え合わせは、このファイルを読む（PDF の読み方を Python 1か所にするため。アプリはその場では取らない）
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "overseas" / "francegalop.json"
UA = "keiba-predictor (personal research)"
TRACKS = {"パリロンシャン": "LON", "ロンシャン": "LON", "シャンティイ": "CHA", "ドーヴィル": "DEA"}
_cache: dict | None = None
logging.getLogger("pypdf").setLevel(logging.ERROR)


def _sec(s: str) -> float:
    m, x = s.split(":")
    return int(m) * 60 + float(x)


def parse(text: str) -> dict | None:
    dm = re.search(r"-\s*(\d{3,4})m\s*\n", text)
    if not dm:
        return None
    lead = re.findall(r"(\d\d:\d\d\.\d\d)", text[text.find("Temps passage en tête"):text.find("Partant")])
    race = _sec(lead[-2]) - _sec(lead[-8]) if len(lead) >= 12 else None  # 先頭の通過：[通過, 区間] の組（6組）。到着 − 600m 地点
    body = text[text.find("Temps\nofficiel"):]
    horses = []
    for blk in re.split(r"\n(?=\d{1,2}\n\(\d{1,2}\)\n)", "\n" + body)[1:]:
        lines = blk.strip().split("\n")
        no = int(lines[0])
        fin = re.match(r"^(\d{1,2})\s+\d", lines[4]) if len(lines) > 4 else None
        pairs = [(int(a), _sec(b)) for a, b in re.findall(r"^(\d{1,2}) (\d\d:\d\d\.\d\d)$", blk, re.M)]
        if len(pairs) < 4:
            continue
        horses.append({"no": no, "finish": int(fin.group(1)) if fin else None, "pos400": pairs[-3][0],
                       "c600": pairs[-4][1], "arr": pairs[-1][1]})
    return {"distance": int(dm.group(1)), "race_last600": None if race is None else round(race, 2), "horses": horses} if horses else None


def fetch(date: str, track: str, no: int) -> dict | None:
    global _cache
    if _cache is None:
        _cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}
    key = f"{date}|{track}|{no}"
    if key not in _cache:
        url = f"https://www7.france-galop.com/Casaques/Tracking//{date.replace('-', '')}{track}{no:02d}_last_times_fr.pdf"
        try:
            import pypdf  # noqa: PLC0415
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
                b = r.read()
            text = pypdf.PdfReader(io.BytesIO(b)).pages[0].extract_text() or ""
            _cache[key] = parse(text)
        except urllib.error.HTTPError as ex:
            _cache[key] = None if ex.code == 404 else _cache.get(key)
            if ex.code != 404:
                print("  ! fg", key, ex, file=sys.stderr)
        except Exception as ex:  # noqa: BLE001
            print("  ! fg", key, ex, file=sys.stderr)
            return None
        time.sleep(1.5)
    return _cache.get(key)


def save() -> None:
    if _cache is not None:
        CACHE.write_text(json.dumps(_cache, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def metrics(race: dict, umaban: int) -> dict | None:
    if race["race_last600"] is None:  # レース全体の上がりがないと fade が出せないので、このレースの区間タイムは使わない
        return None
    fin = [h for h in race["horses"] if h["finish"]]
    me = next((h for h in fin if h["no"] == umaban), None)
    if not me or len(fin) < 3:
        return None
    last = {h["no"]: h["arr"] - h["c600"] for h in fin}
    a = last[me["no"]]
    srt = sorted(last.values())
    return {"ag_dev": round(sum(last.values()) / len(last) - a, 4), "ag_rank": round((1 + srt.index(a)) / len(last), 4),
            "early_gap": round(me["c600"] - min(h["c600"] for h in fin), 4),
            "fade": round(a - race["race_last600"], 4),
            "pos_first": None, "pos_last": me["pos400"], "time_diff": round(me["arr"] - min(h["arr"] for h in fin), 1)}


def fg_run(venue: str, date: str, race_no: str, distance: int, umaban: str, finish: str) -> dict | None:
    track = next((c for k, c in TRACKS.items() if k in venue), None)
    if not track or date < "2020-01-01" or not str(umaban).isdigit() or not str(finish).isdigit():
        return None
    nos = [int(race_no)] if str(race_no).isdigit() else list(range(1, 11))
    for no in nos:
        race = fetch(date, track, no)
        if not race or abs(race["distance"] - distance) > 10:
            continue
        me = next((h for h in race["horses"] if h["no"] == int(umaban)), None)
        if me and me["finish"] == int(finish):
            return metrics(race, int(umaban))
    return None


def _runs(rows: list[dict]) -> int:
    ok = 0
    for r in rows:
        d = re.match(r"^[芝ダ](\d+)", r.get("距離") or "")
        if d and any(k in r.get("開催", "") for k in TRACKS):
            ok += fg_run(r["開催"], r["日付"].replace("/", "-"), r.get("R", ""), int(d.group(1)), r.get("馬番", ""), r.get("着順", "")) is not None
    return ok


def main() -> None:
    if sys.argv[1:] == ["build"]:
        hist = json.loads((ROOT / "data/overseas/histories.json").read_text(encoding="utf-8"))
        ok = sum(_runs(v["rows"]) for v in hist.values())
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from check_jra_day import netkeiba_performances  # noqa: PLC0415
        ids = json.loads((ROOT / "data/overseas/upcoming_overseas.json").read_text(encoding="utf-8"))["horses"]
        ok = 0
        for hid in ids:
            ok += _runs(netkeiba_performances(hid))
            save()
    save()
    print(f"フランスの走で区間タイムが取れた：{ok}", file=sys.stderr)


if __name__ == "__main__":
    main()

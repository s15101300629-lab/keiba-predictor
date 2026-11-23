"""racedb.com（世界の競馬）から、海外の走のレース結果（着差・単勝倍率・1着賞金）を集める（2026-11-23、依頼者の案）.

    python scripts/ml/collect_racedb.py   # → data/overseas/racedb.json（再開できる）

- 対象：data/overseas/histories.json（collect_overseas.py）の馬の、海外の走
- 名前の一覧（/name/*.html）で馬のページを探す。外国馬は英語名（netkeiba のページ題名のカッコ内）、日本馬はカタカナ名
- 馬のページの「主な競走結果」から、netkeiba の海外の走と同じ日付のレースのページ（/k/*.html）を取る
- レースのページは、表の行をセルの文字の並びのまま保存する（読み方は後で決める）。見出しの文字も保存する
- 3秒に1ページ。User-Agent はこのアプリの名前。このセッションからは接続できないので GitHub Actions で動かす
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
SRC = ROOT / "data" / "overseas" / "histories.json"
OUT = ROOT / "data" / "overseas" / "racedb.json"
BASE = "https://racedb.com"
UA = "keiba-predictor (personal research)"
WAIT = 3.0
LETTERS = list("abcdefghijklmnopqrstuvwxyz") + ["aa", "ai", "au", "ae", "ao", "ka", "ki", "ku", "ke", "ko", "sa", "si", "su", "se", "so",
                                                "ta", "ti", "tu", "te", "to", "na", "ni", "nu", "ne", "no", "ha", "hi", "hu", "he", "ho",
                                                "ma", "mi", "mu", "me", "mo", "ya", "yu", "yo", "ra", "ri", "ru", "re", "ro", "wa", "wo", "other"]
JRA = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
NAR = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広", "ばんえい"]


def is_overseas(venue: str) -> bool:
    return bool(venue) and not any(v in venue for v in JRA + NAR)


def get(url: str) -> str | None:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
                b = r.read()
            time.sleep(WAIT)
            try:
                return b.decode("utf-8")
            except UnicodeDecodeError:
                return b.decode("cp932", errors="replace")
        except Exception as e:  # noqa: BLE001
            if "404" in str(e):
                time.sleep(WAIT)
                return None
            print("  !", url, e, file=sys.stderr, flush=True)
            time.sleep(10 * (attempt + 1))
    return None


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def norm_name(s: str) -> str:
    s = s.split("\u3000")[0]
    s = re.sub(r"\(.*?\)|（.*?）", "", s).replace("*", "").replace("＊", "")
    return re.sub(r"[\s'’.\-・]", "", s).lower()


def tables(page: str) -> list[list[list[str]]]:
    out = []
    for t in re.findall(r"<table.*?</table>", page, re.S):
        rows = [[clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)] for r in re.findall(r"<tr.*?</tr>", t, re.S)]
        out.append([r for r in rows if r])
    return out


def main() -> None:
    src = json.loads(SRC.read_text(encoding="utf-8"))
    data = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"index": {}, "horses": {}, "races": {}}
    save = lambda: OUT.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")  # noqa: E731
    # 1. 名前の一覧
    # 一覧の読み直し（名前の後ろの「　牡（2021）」を除く形に、保存済みの一覧からそろえ直す）
    if data["index"] and not data.get("index_v2"):
        flat = [x for v in data["index"].values() for x in v]
        data["index"] = {}
        for href, name in flat:
            data["index"].setdefault(norm_name(name), []).append([href, name])
        data["index_v2"] = True
    if not data["index"]:
        for l in LETTERS:
            page = get(f"{BASE}/name/{l}.html")
            if not page:
                continue
            for href, name in re.findall(r'href="(/p/\d+\.html)"[^>]*>([^<]+)<', page):
                data["index"].setdefault(norm_name(name), []).append([href, name.strip()])
        save()
        print("index:", len(data["index"]), file=sys.stderr, flush=True)
    # 2. 馬のページ → 3. 海外の走と同じ日付のレースのページ
    n = 0
    for hid, v in src.items():
        dates = sorted({r["日付"].replace("/", ".") for r in v["rows"] if is_overseas(r.get("開催", "")) and r.get("日付")})
        # 着差（馬身）→ 秒の換算を成績表で決めるため、JRA の走も1つ（JRA の重賞で、racedb にあれば）
        jra_dates = sorted({r["日付"].replace("/", ".") for r in v["rows"] if any(x in r.get("開催", "") for x in JRA)
                            and re.search(r"\(G[123I]+\)", r.get("レース名", "")) and r["日付"] >= "2017"})
        if not dates or (hid in data["horses"] and data["horses"][hid].get("page")):
            continue
        title = v.get("title", "").split("の競走成績")[0]
        outer = title.split("(")[0].strip()
        inner = (re.search(r"\(([^)]+)\)", title) or [None, ""])[1]
        cands = []
        for nm in (outer, inner):  # 外国馬は「英語名 (カナ)」、日本馬は「カナ (英語名)」
            if nm and not cands:
                cands = data["index"].get(norm_name(nm), [])
        entry = {"title": title, "candidates": cands, "page": None, "races": []}
        for href, _ in cands[:3]:  # 同名の馬がいるときは、海外の走の日付が一番多く重なるページ
            page = get(BASE + href)
            if not page:
                continue
            runs = []
            for row in re.findall(r"<tr.*?</tr>", page, re.S):
                d = re.search(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", clean(row))
                k = re.search(r'href="(/k/\d+\.html)"', row)
                if d and k:
                    runs.append((f"{d.group(1)}.{int(d.group(2)):02d}.{int(d.group(3)):02d}", k.group(1)))
            hit = [k for d, k in runs if d in dates]
            if len(hit) > len(entry["races"]):
                jra = [k for d, k in runs if d in jra_dates][:1]
                entry.update({"page": href, "races": hit, "jra_races": jra})
        data["horses"][hid] = entry
        for k in entry["races"] + entry.get("jra_races", []):
            if k in data["races"] and data["races"][k].get("text"):
                continue
            page = get(BASE + k)
            if page:
                head = clean(re.sub(r"<table.*?</table>", " ", re.sub(r"<(script|style).*?</\1>", "", page, flags=re.S), flags=re.S))[:1500]
                text = clean(re.sub(r"<(script|style).*?</\1>", "", page, flags=re.S))[:4000]
                data["races"][k] = {"head": head, "text": text, "tables": tables(page)}
        n += 1
        if n % 10 == 0:
            save()
            print(f"  {n} horses, {len(data['races'])} races", file=sys.stderr, flush=True)
    save()
    found = sum(1 for h in data["horses"].values() if h["page"])
    print(f"done: horses {len(data['horses'])}（見つかった {found}）, races {len(data['races'])}", file=sys.stderr)


if __name__ == "__main__":
    main()

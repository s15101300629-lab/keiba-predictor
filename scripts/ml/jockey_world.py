"""外国人騎手の海外での成績（JRA-VAN Ver.World の「海外主要G1レースの成績」）。2026-11-26、依頼者の指示.

    cd scripts/ml && python jockey_world.py   # → data/overseas/jockey_world.json

- 騎手の検索：https://world.jra-van.jp/db/search?category=2&keywords=名前（頭文字を外したカナ）。頭文字があれば合わせる
- 騎手のページ（/db/jockey/ID/）の年ごとの 1着・2着・3着・着外
- 式：短期免許の騎手（jockey_career.json）のうち JRA で 50騎乗以上の騎手で、JRA での3着内率 ＝ 切片 ＋ 傾き × 海外主要G1の3着内率（縮めた値）
- 使い方（features.py の fill_foreign_people・features.ts の fillForeignPeople）：外国馬の騎手が JRA で乗ったことがなく、ここに載っていれば、
  式で出した3着内率が同じレースの日本馬の騎手の中で下から何割かの位置の値で、騎手の項目を埋める（載っていなければ中央値）
アプリ（route.ts）と答え合わせ（check_jra_day.py）は、ファイルにない騎手をその場で検索する（lookup）
"""

from __future__ import annotations

import html
import json
import re
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "overseas" / "jockey_world.json"
UA = "keiba-predictor (personal research)"
BASE = "https://world.jra-van.jp"
SHRINK = 20  # 騎乗数が少ない騎手は全体の3着内率に寄せる（騎乗数 + 20 で割る）


def get(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
        b = r.read()
    time.sleep(1.5)
    return b.decode("utf-8", "replace")


def split_name(name: str) -> tuple[str, str]:
    """「Ｌトーレス」「L.トーレス」「ムーア」→ (頭文字, カナ)."""
    s = unicodedata.normalize("NFKC", name or "").split("(")[0].replace(".", "").replace("・", "").replace(" ", "")
    m = re.match(r"^([A-Za-z])(.*)$", s)
    return (m.group(1).upper(), m.group(2)) if m else ("", s)


def lookup(name: str) -> dict | None:
    """騎手を検索して、年ごとの [1着, 2着, 3着, 着外] を返す。見つからなければ None."""
    ini, kana = split_name(name)
    if not kana:
        return None
    t = get(f"{BASE}/db/search?category=2&keywords={urllib.parse.quote(kana)}")
    cands = [(h, re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", x)))) for h, x in re.findall(r'href="(/db/jockey/[^"]+/)"[^>]*>(.*?)</a>', t, re.S)]
    cands = [(h, unicodedata.normalize("NFKC", x).replace(".", "")) for h, x in cands]
    cands = [(h, x) for h, x in cands if kana in x and (not ini or x.startswith(ini))]
    if len({h for h, _ in cands}) != 1:
        return None
    href, full = cands[0]
    page = get(BASE + href)
    i = page.find("海外主要G1レースの成績")
    years = {}
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page[i:i + 20000], re.S):
        c = [re.sub(r"\s+", "", html.unescape(re.sub(r"<[^>]+>", "", x))) for x in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S)]
        if len(c) >= 5 and re.fullmatch(r"\d{4}", c[0]) and all(x.isdigit() for x in c[1:5]):
            years[c[0]] = [int(x) for x in c[1:5]]
    return {"id": href.split("/")[-2], "name": full, "years": years}


def world_rate(entry: dict, year: str, prior: float) -> float | None:
    """その年まで（その年を含む）の海外主要G1の3着内率を、全体の率に縮めた値."""
    n = t = 0
    for y, (a, b, c, d) in entry["years"].items():
        if y <= year:
            n += a + b + c + d
            t += a + b + c
    return None if n == 0 else (t + SHRINK * prior) / (n + SHRINK)


def main() -> None:
    career = json.loads((ROOT / "data/overseas/jockey_career.json").read_text(encoding="utf-8"))["jockeys"]
    old = json.loads(OUT.read_text(encoding="utf-8"))["jockeys"] if OUT.exists() else {}
    names = list(career) + [a for a in sys.argv[1:]]
    out = {}
    for k in names:
        try:
            e = old.get(k) or lookup(k)
        except Exception as ex:  # noqa: BLE001
            print("  !", k, ex, file=sys.stderr)
            e = None
        if e:
            out[k] = e
    tot = [sum(v[:3]) for e in out.values() for v in e["years"].values()]
    allr = [sum(v) for e in out.values() for v in e["years"].values()]
    prior = sum(tot) / max(sum(allr), 1)
    xs, ys = [], []
    for k, e in out.items():
        rides = career.get(k, [])
        if len(rides) >= 50:
            r = world_rate(e, "9999", prior)
            if r is not None:
                xs.append(r); ys.append(sum(x[4] for x in rides) / len(rides))
    b, a = np.polyfit(xs, ys, 1)
    OUT.write_text(json.dumps({"note": "JRA-VAN Ver.World の海外主要G1の成績（scripts/ml/jockey_world.py）。fit：JRA の3着内率＝切片＋傾き×海外主要G1の3着内率（縮めた値）",
                               "prior": round(prior, 5), "shrink": SHRINK,
                               "fit": {"intercept": round(float(a), 5), "slope": round(float(b), 5), "n": len(xs), "r": round(float(np.corrcoef(xs, ys)[0, 1]), 3)},
                               "jockeys": out}, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"騎手 {len(out)}/{len(names)}、式 n={len(xs)} 切片 {a:.3f} 傾き {b:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()

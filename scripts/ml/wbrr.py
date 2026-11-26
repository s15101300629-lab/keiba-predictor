"""ロンジンワールドベストレースホースランキング（JRA の PDF）で外国馬のレーティングを補う（2026-11-26、依頼者の指示）.

    cd scripts/ml && python wbrr.py [YYYYMMDD]   # → data/overseas/wbrr.json（日付を省くと、今日から60日さかのぼって一番新しい版）

- PDF：https://www.jra.go.jp/datafile/ranking/wrank/pdf/{YYYYMMDD}-ranking.pdf（上位30〜40頭。数字・英語名・国・生年）
- 馬は netkeiba の英語名の検索（db.netkeiba.com/?pid=horse_list&word=）で馬ID とカナ名に。生年が合うものだけ
- 式は作らない（日本馬が5頭ほどで数字の幅が狭く、式にならないため）。外国馬（調教国が JPN 以外）のレーティングは、
  同じ版の日本馬のうち数字が一番近い馬（同じなら平均。日本馬より上・下なら端の馬）の AI のレーティング（store.json の最新）にする
- アプリ（route.ts）と答え合わせ（features.py の FOREIGN_FIX）は、外国馬の馬ID でこのファイルを引く
"""

from __future__ import annotations

import io
import json
import logging
import re
import sys
import time
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pypdf

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "overseas" / "wbrr.json"
logging.getLogger("pypdf").setLevel(logging.ERROR)


def get(url: str) -> tuple[bytes, str] | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "keiba-predictor (personal research)"}), timeout=30) as r:
            b, u = r.read(), r.geturl()
        time.sleep(1)
        return b, u
    except Exception:  # noqa: BLE001
        return None


def netkeiba_horse(name: str, yof: str) -> tuple[str, str] | None:
    r = get("https://db.netkeiba.com/?pid=horse_list&word=" + urllib.parse.quote(name))
    if not r:
        return None
    t = r[0].decode("utf-8", "replace")
    m = re.search(r"/horse/(\d{10})/", r[1])
    if m and m.group(1)[:4] == yof:
        kana = re.search(r"<title>\s*([^ (]+)", t)
        return m.group(1), kana.group(1) if kana else ""
    for hid, kana in re.findall(r'href="/horse/(\d{10})/"[^>]*title="([^"]+)"', t):
        if hid[:4] == yof:
            return hid, kana
    return None


def main() -> None:
    import urllib.parse  # noqa: F401,PLC0415
    days = [sys.argv[1]] if len(sys.argv) > 1 else [(date.today() - timedelta(days=k)).strftime("%Y%m%d") for k in range(60)]
    pdf = None
    for d in days:
        r = get(f"https://www.jra.go.jp/datafile/ranking/wrank/pdf/{d}-ranking.pdf")
        if r and r[0][:4] == b"%PDF":
            pdf, day = r[0], d
            break
    if not pdf:
        print("ランキングの PDF が見つからない", file=sys.stderr)
        return
    text = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(io.BytesIO(pdf)).pages)
    rows = re.findall(r"^\d+ (\d{3}) [A-Z,]+ [TDA,]+ (.+?) \(([A-Z]{2,3})\) (\d{4}) [A-Z] .*? ([A-Z]{2,3})$", text, re.M)
    store = json.loads((ROOT / "data/ml/store.json").read_text(encoding="utf-8"))
    ratings = store.get("ratings", {})
    alias = {features.clean_horse_name(k): k for k in ratings}
    horses = []
    for rating, name, _bred, yof, trained in rows:
        hit = netkeiba_horse(name, yof)
        horses.append({"name": name, "yof": yof, "trained": trained, "rating": int(rating), "horse_id": hit[0] if hit else None, "kana": hit[1] if hit else None})
    anchors = []
    for h in horses:
        if h["trained"] == "JPN" and h["kana"]:
            rt = ratings.get(alias.get(h["kana"], h["kana"]))
            if rt:
                anchors.append({"kana": h["kana"], "rating": h["rating"], "elo": rt})
    foreign = {}
    if anchors:
        for h in horses:
            if h["trained"] == "JPN" or not h["horse_id"]:
                continue
            dist = min(abs(a["rating"] - h["rating"]) for a in anchors)
            near = [a for a in anchors if abs(a["rating"] - h["rating"]) == dist]
            foreign[h["horse_id"]] = {"name": h["name"], "rating": h["rating"], "anchors": [a["kana"] for a in near],
                                      "elom": round(sum(a["elo"][0] for a in near) / len(near), 4),
                                      "elos": [round(sum(a["elo"][2] for a in near) / len(near), 4), round(sum(a["elo"][4] for a in near) / len(near), 4)]}
    OUT.write_text(json.dumps({"note": "ロンジンワールドベストレースホースランキング（scripts/ml/wbrr.py）。foreign＝外国馬の馬ID → 数字が一番近い日本馬の AI のレーティング",
                               "date": day, "store_last_date": store["lastDate"], "anchors": anchors, "foreign": foreign, "horses": horses},
                              ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{day}: {len(horses)} 頭、日本馬（基準） {len(anchors)}、外国馬 {len(foreign)}", file=sys.stderr)


if __name__ == "__main__":
    import urllib.parse  # noqa: F401
    main()

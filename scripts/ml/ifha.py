"""IFHA のレースの格（2026-11-26、依頼者の指示「IFHA にレースの格が記載されているから、それを元に」）→ data/overseas/ifha.json.

    cd scripts/ml && python ifha.py

- 国の格（A）：国際格付け基準（ICS、IFHA の Blue Book 2026。https://www.tjcis.com/pdf/icsc26/FrontMaterial2026.pdf の目次）の Part I・II・III。
  Part I の重賞は国際的な格（G1〜G3・L）、Part II の重賞は「リステッド相当」、Part III は格の扱いなし
- レースの強さ（B）：IFHA の「世界のトップ100 G1」の年ごとのレースレーティング（上位4頭のレーティングの平均。
  https://www.ifhaonline.org/resources/WTGradedRanking/LWGRank.asp?batch=N。2017〜2025年）
- レーティング → 1着賞金：トップ100に入った JRA の G1（成績表で競馬場・距離・芝ダ・年が1レースに決まるもの）の
  log 1着賞金 ＝ 切片 ＋ 傾き × レーティング。トップ100に入らない JRA の G1 の1着賞金の中央値（unrated_g1）も出す
海外の走の1着賞金の決め方は overseas.ifha_prize（Python）と app/api/races/[raceId]/route.ts の ifhaPrize（TS）が同じ
"""

from __future__ import annotations

import html
import json
import math
import re
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "overseas" / "ifha.json"
UA = "keiba-predictor (personal research)"

# ICS 2026 の目次（Part I・II。Part III とどこにも載っていない国は 3）
PARTS = {
    **{c: 1 for c in ("ARG", "AUS", "BRZ", "CAN", "CHI", "FR", "GER", "GB", "HK", "IRE", "JPN", "NZ", "PER", "SAF", "UAE", "USA", "URU")},
    **{c: 2 for c in ("BAH", "IND", "ITY", "KOR", "MAL", "PAN", "PR", "KSA", "SCA", "TUR", "VEN", "ZIM")},
}
# netkeiba の開催の文字 → [国, IFHA の競馬場名（なければ国の全部の競馬場）]。上から順に最初に合ったもの
VENUES = [
    ["パリロンシャン", "FR", ["PARIS-LONGCHAMP", "PARIS LONGCHAMP", "PARISLONGCHAMP"]], ["ロンシャン", "FR", ["PARIS-LONGCHAMP", "PARIS LONGCHAMP", "PARISLONGCHAMP"]],
    ["シャンティイ", "FR", ["CHANTILLY"]], ["ドーヴィル", "FR", ["DEAUVILLE"]], ["サンクルー", "FR", ["SAINT-CLOUD", "SAINT CLOUD"]],
    ["クレールフォンテ", "FR", ["CLAIREFONTAINE"]], ["フランス", "FR", []],
    ["アスコット", "GB", ["ASCOT"]], ["ヨーク", "GB", ["YORK"]], ["グッドウッド", "GB", ["GOODWOOD"]], ["ニューマーケット", "GB", ["NEWMARKET"]],
    ["エプソム", "GB", ["EPSOM DOWNS"]], ["サンダウン", "GB", ["SANDOWN PARK"]], ["ドンカスター", "GB", ["DONCASTER"]], ["ニューベリー", "GB", ["NEWBURY"]],
    ["ヘイドック", "GB", ["HAYDOCK PARK"]], ["チェスター", "GB", ["CHESTER"]], ["イギリス", "GB", []],
    ["レパーズタウン", "IRE", ["LEOPARDSTOWN"]], ["カラ", "IRE", ["CURRAGH"]], ["アイルランド", "IRE", []],
    ["ケルン", "GER", ["COLOGNE"]], ["ミュンヘン", "GER", ["MUNICH"]], ["バーデンバーデン", "GER", ["BADEN-BADEN", "BADEN BADEN"]],
    ["ホッペガルテン", "GER", ["HOPPEGARTEN"]], ["ハンブルク", "GER", ["HAMBURG"]], ["ドイツ", "GER", []],
    ["シャティン", "HK", ["SHA TIN"]], ["ハッピーヴァレー", "HK", ["HAPPY VALLEY"]], ["ハッピーバレー", "HK", ["HAPPY VALLEY"]], ["香港", "HK", []],
    ["メイダン", "UAE", ["MEYDAN"]], ["アラブ首長国連邦", "UAE", []], ["ドバイ", "UAE", []],
    ["キングアブドゥル", "KSA", ["RIYADH"]], ["サウジアラビア", "KSA", []],
    ["フレミントン", "AUS", ["FLEMINGTON"]], ["コーフィールド", "AUS", ["CAULFIELD"]], ["ムーニーバレー", "AUS", ["MOONEE VALLEY"]],
    ["ランドウィック", "AUS", ["RANDWICK"]], ["ローズヒル", "AUS", ["ROSEHILL"]], ["オーストラリア", "AUS", []],
    ["ニュージーランド", "NZ", []], ["エラズリー", "NZ", ["ELLERSLIE"]],
    ["デルマー", "USA", ["DEL MAR"]], ["サンタアニタ", "USA", ["SANTA ANITA"]], ["チャーチルダウン", "USA", ["CHURCHILL DOWNS"]],
    ["キーンランド", "USA", ["KEENELAND"]], ["ベルモント", "USA", ["BELMONT PARK", "BELMONT AT THE BIG A"]], ["サラトガ", "USA", ["SARATOGA"]],
    ["ガルフストリーム", "USA", ["GULFSTREAM PARK"]], ["モンマス", "USA", ["MONMOUTH PARK"]], ["アメリカ", "USA", []],
    ["ウッドバイン", "CAN", ["WOODBINE"]], ["カナダ", "CAN", []],
    ["チリ", "CHI", []], ["ペルー", "PER", []], ["アルゼンチン", "ARG", []], ["ブラジル", "BRZ", []], ["ウルグアイ", "URU", []], ["南アフリカ", "SAF", []],
    ["イタリア", "ITY", []], ["ソウル", "KOR", []], ["韓国", "KOR", []], ["バーレーン", "BAH", []], ["スウェーデン", "SCA", []], ["トルコ", "TUR", []],
    ["カタール", "QAT", []], ["アルライヤーン", "QAT", []],
]
JRA_TRACK = {"SAPPORO": 1, "HAKODATE": 2, "FUKUSHIMA": 3, "NIIGATA": 4, "TOKYO": 5, "NAKAYAMA": 6, "CHUKYO": 7, "KYOTO": 8, "HANSHIN": 9, "KOKURA": 10}


def fetch(url: str) -> str:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=60) as r:
        b = r.read()
    time.sleep(2)
    return b.decode("latin-1")


def top100() -> list[dict]:
    by_year: dict[str, list[dict]] = {}
    for batch in range(1, 30):
        try:
            t = fetch(f"https://www.ifhaonline.org/resources/WTGradedRanking/LWGRank.asp?batch={batch}")
        except Exception:  # noqa: BLE001
            break
        s = re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t)))
        y = re.search(r"(\d{4}) Ranking", s)
        if not y:
            break
        rows = []
        for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
            c = [re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", x))).strip() for x in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            if len(c) >= 10 and re.match(r"^\d", c[0]) and re.match(r"^\d+(\.\d+)?$", c[-1]):
                rows.append({"year": y.group(1), "race": c[1], "distance": int(c[2]), "surface": 0 if c[4] == "T" else 1,
                             "track": c[-5], "country": c[-4], "age": c[-3], "sex": c[-2], "rating": float(c[-1])})
        if not rows:
            break
        by_year[y.group(1)] = rows  # 同じ年が2回あるときは後のもの（確定版）
        print(batch, y.group(1), len(rows), file=sys.stderr, flush=True)
    return [r for y in sorted(by_year) for r in by_year[y]]


def main() -> None:
    races = top100()
    # JRA の G1（成績表）：年・競馬場・距離・芝ダ → 1着賞金（万円）
    jra: dict[tuple, list[float]] = {}
    all_g1: list[tuple] = []
    for fp in sorted((ROOT / "data/jra-seiseki/json").glob("*/*.json")):
        day = json.loads(fp.read_text(encoding="utf-8"))
        for r in day["races"]:
            if "ＧⅠ" not in (r.get("race_name") or "") or r.get("surface") not in ("芝", "ダート") or not r.get("prize"):
                continue
            k = (r["date"][:4], int(day["venue_code"]), r.get("distance"), 0 if r["surface"] == "芝" else 1)
            jra.setdefault(k, []).append(r["prize"][0] / 10000)
            all_g1.append(k)
    xs, ys, used = [], [], set()
    for r in races:
        if r["country"] != "JPN" or r["track"] not in JRA_TRACK:
            continue
        k = (r["year"], JRA_TRACK[r["track"]], r["distance"], r["surface"])
        if len(jra.get(k, [])) == 1:
            xs.append(r["rating"]); ys.append(math.log(jra[k][0])); used.add(k)
    b, a = (lambda p: (p[0], p[1]))(__import__("numpy").polyfit(xs, ys, 1))
    rated = {(r["year"], JRA_TRACK.get(r["track"]), r["distance"], r["surface"]) for r in races if r["country"] == "JPN"}
    unrated = [jra[k][i] for k in set(all_g1) if k not in rated for i in range(len(jra[k]))]
    unrated_g1 = round(statistics.median(unrated), 1)
    for r in races:
        r["prize"] = round(min(math.exp(a + b * r["rating"]), 50000.0), 1)
    out = {"note": "IFHA のレースの格（scripts/ml/ifha.py）。parts＝ICS 2026 の Part（ないものは 3）、races＝世界のトップ100 G1 のレースレーティングと、"
                   "JRA の G1 の関係の式で直した1着賞金（万円）、unrated_g1＝トップ100に入らない JRA の G1 の1着賞金の中央値",
           "fit": {"intercept": round(float(a), 5), "slope": round(float(b), 5), "n": len(xs)}, "unrated_g1": unrated_g1,
           "parts": PARTS, "venues": VENUES, "races": races}
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"トップ100 {len(races)} 件、式 n={len(xs)} 傾き {b:.4f}、トップ100外の JRA G1 の中央値 {unrated_g1}（{len(unrated)} レース）", file=sys.stderr)


if __name__ == "__main__":
    main()

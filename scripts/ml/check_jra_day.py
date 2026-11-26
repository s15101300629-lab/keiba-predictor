"""JRA のレース結果ページ（https://www.jra.go.jp/JRADB/accessS.html）から、その日の結果を取って AI の予測と突き合わせる.

    cd scripts/ml && python check_jra_day.py 2026-11-21            # → data/ml/day_check/2026-11-21.json
    cd scripts/ml && python check_jra_day.py 2026-11-21 --training # 調教評価（netkeiba の調教ページ）も取って入れる（Actions で使う）
    --saved --inputs-only --model=../../data/ml/prev/model.json        # 過去の日の買い方の成績の入力だけ（prev_models.py の一式で）
買い方の成績（day_bets.ts）の入力 data/ml/day_check/inputs/日付.json.gz と、結果ページの払戻（payouts）も保存する
    日付を省くと今日（JST）。その日の結果ページがなければ何もしない
画面の「答え合わせ」（app/results）がこの JSON を読む。毎日 17:30 JST ごろに day-check.yml が動かして main に保存する
- 確定の単勝・複勝オッズはオッズのページ（accessO、pw151ou…）から取る

成績表 PDF は翌日に出るので、それまでの「当日の答え合わせ」用。学習には使わない（学習は翌日の PDF で ml-train.yml が行う）。
- 結果ページは POST（cname）で開く。開催中の競馬場の「全レース」ページ（pw01ses01…）を読む
- 成績表と同じ形（data/jra-seiseki/json の races）に直し、一時フォルダで既存の成績表に足して build_dataset する。予測は今の data/ml/model.json
- 結果ページにないもの：単勝オッズ（人気だけ）、2着以下の馬主・生産牧場、不利（被害馬）。今のモデルは馬主などを使っていない
- 血統（ped_*）は、同じ馬の前の走の値を使う（同じ芝ダで距離が一番近い走。父などの成績はレース前年までなので同じ年なら同じ）。初出走の馬は欠損
"""

from __future__ import annotations

import copy
import os
import html as htmlmod
import json
import math
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import overseas  # noqa: E402
import jra_odds  # noqa: E402

EXOTIC_ODDS: dict[str, dict] = {}  # race_id → 券種 → 組み合わせ → 確定オッズ（fetch_day が埋める）
from features import build_dataset  # noqa: E402
from train import DATA, OUT  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

BASE = "https://www.jra.go.jp"
VENUES = {"01": ("sapporo", "札幌"), "02": ("hakodate", "函館"), "03": ("fukushima", "福島"), "04": ("niigata", "新潟"),
          "05": ("tokyo", "東京"), "06": ("nakayama", "中山"), "07": ("chukyo", "中京"), "08": ("kyoto", "京都"),
          "09": ("hanshin", "阪神"), "10": ("kokura", "小倉")}


def post(path: str, cname: str) -> str:
    for attempt in range(4):
        try:
            res = requests.post(BASE + path, data={"cname": cname}, timeout=30)
            if res.ok:
                return res.content.decode("cp932", errors="replace")
        except requests.RequestException as e:
            print("error", e, file=sys.stderr)
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"failed {path} {cname}")


def text(s: str) -> str:
    return re.sub(r"\s+", " ", htmlmod.unescape(re.sub(r"<[^>]+>", " ", s))).strip()


def num(s: str | None):
    s = (s or "").replace(",", "").strip()
    try:
        return float(s) if "." in s else int(s)
    except ValueError:
        return None


def time_sec(s: str) -> float | None:
    m = re.match(r"(?:(\d+):)?(\d+\.\d)", s.strip())
    return None if not m else int(m[1] or 0) * 60 + float(m[2])


def sex_age(sa: str) -> tuple[str | None, int | None]:
    """「牡3」「牝4」「せん5」→ 成績表と同じ「牡・牝・セ」と年齢（2026-11-22 まで、せん馬は性別「せ」・年齢なしになっていた）."""
    m = re.match(r"(牡|牝|せん|セ)\s*(\d+)", sa or "")
    if not m:
        return None, None
    return ("セ" if m[1] in ("せん", "セ") else m[1]), int(m[2])


def repair_results(days: list[dict]) -> None:
    """保存済みの結果の直し：せん馬の性別「せ」→「セ」、年齢がなければ馬ID の先頭4けた（生まれた年）から（JRA の年齢＝年−生まれた年）."""
    for day in days:
        for r in day["races"]:
            for h in r["results"]:
                if h.get("sex") == "せ":
                    h["sex"] = "セ"
                if h.get("age") is None and re.match(r"\d{4}", h.get("horse_id") or ""):
                    h["age"] = int(day["date"][:4]) - int(h["horse_id"][:4])


def parse_race(block: str, date: str, venue_code: str) -> dict | None:
    m = re.search(r"(\d+)回[^<\d]*(\d+)日", text(re.search(r'<div class="cell date">(.*?)</div>', block, re.S)[1]))
    kai, day = int(m[1]), int(m[2])
    race_no = int(re.search(r'alt="(\d+)レース"', block)[1])
    name = text(re.search(r'<span class="race_name">(.*?)</span>\s*</span>', block, re.S)[1]) if 'class="race_name"' in block else ""
    cm = re.search(r'<div class="cell course">(.*?)</div>', block, re.S)
    ct = text(cm[1]) if cm else ""
    dist = num(re.search(r"([\d,]+)\s*メートル", ct)[1])
    course = re.sub(r"([右左])外", r"\1・外", re.search(r"（(.*?)）", ct)[1].replace(" ", ""))
    surface = "障害" if "障害" in course or "障害" in name else "ダート" if course.startswith("ダート") else "芝" if course.startswith("芝") else course
    baba = dict(re.findall(r'<span class="cap">(.*?)</span><span class="txt">(.*?)</span>', block))
    going = baba.get("ダート") if surface == "ダート" else baba.get("芝")
    prize1 = re.search(r'<span class="cap">1着</span><span class="num">([\d,]+)</span>', block)
    body = block[block.find("<tbody>"):block.find("</tbody>")]
    results = []
    for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        cell = {c: v for c, v in re.findall(r'<td class="(\w+)">(.*?)</td>', tr, re.S)}
        place = text(cell.get("place", ""))
        hid = re.search(r"pw01dud00(\d{10})", cell.get("horse", ""))
        sa = text(cell.get("age", ""))
        jt = cell.get("jockey", "")
        jmark = re.search(r'class="mark jockey">(.*?)<', jt)
        jockey = text(re.sub(r"<span[^>]*>.*?</span>", "", jt)).replace(" ", "")
        # 「480(+2)」。初出走は「464(初出走)」など増減が数でない（2026-11-22 まで新馬は馬体重ごと欠けていた）。成績表と同じく増減は空
        hw = re.match(r"(\d+)\s*(?:\(([+\-]?\d+)\))?", text(cell.get("h_weight", "")))
        corners = [int(x) for x in re.findall(r"<li[^>]*>(\d+)</li>", tr)]
        status = place if place in ("取消", "除外", "中止", "失格") else None
        results.append({
            "waku": num(re.search(r"waku/(\d)\.png", cell.get("waku", ""))[1]) if "waku/" in cell.get("waku", "") else None,
            "umaban": num(text(cell.get("num", ""))), "horse": text(cell.get("horse", "")), "horse_id": hid[1] if hid else None,
            "sex": sex_age(sa)[0], "age": sex_age(sa)[1], "weight": num(text(cell.get("weight", ""))),
            "jockey": jockey, "jockey_mark": jmark[1] if jmark else None, "owner": None, "breeder": None,
            "trainer": text(cell.get("trainer", "")).replace(" ", ""), "status": status,
            "body_weight": int(hw[1]) if hw else None, "body_weight_diff": int(hw[2]) if hw and hw[2] is not None else None,
            "popularity": num(text(cell.get("pop", ""))), "time": text(cell.get("time", "")),
            "time_sec": time_sec(text(cell.get("time", ""))) if place.isdigit() else None,
            "finish": int(place) if place.isdigit() else None, "corner_positions": corners or None,
            "agari": num(text(cell.get("f_time", ""))),
        })
    foot = text(block[block.find("</tbody>"):])
    lm = re.search(r"ハロンタイム ([\d. \-]+?) 上り", foot)
    laps = [float(x) for x in re.findall(r"\d+\.\d", lm[1])] if lm else None
    l3 = re.search(r"3F ([\d.]+)", foot)
    corners = []
    cs = re.search(r"コーナー通過順位 (.*?) 払戻金", foot)
    if cs:
        for c, order in re.findall(r"(\d)コーナー (.*?)(?= \dコーナー |$)", cs[1]):
            corners.append({"corner": c, "order": order.replace(" ", "").replace("*", "").replace("=", "＝")})
    winner = re.search(r"馬主：(.*?) 生産牧場：(.*?) (?:取引市場|競走中|$)", foot)
    if winner:
        for h in results:
            if h["finish"] == 1:
                h["owner"], h["breeder"] = winner[1].strip(), winner[2].strip()
    ev = re.search(r"競走中の出来事等(.*?)(?:ページトップへ戻る|$)", foot)
    notes = [x.strip() for x in ev[1].split("・") if x.strip()] if ev else []  # 出遅れ・不利・制裁などの文
    starters = [h for h in results if h["status"] not in ("取消", "除外")]
    payouts = parse_payouts(block)
    oc = re.search(r"'(pw151ou[^']+)'", block)
    return {"odds_cname": oc[1] if oc else None, "race_no": race_no, "race_name": name, "distance": dist, "course": course, "surface": surface, "going": going,
            "weather": baba.get("天候"), "prize": [int(prize1[1].replace(",", "")) * 10000] if prize1 else [],
            "race_id": f"{date[:4]}{venue_code}{kai:02d}{day:02d}{race_no:02d}", "date": date, "kai": kai, "day": day,
            "starters": len(starters), "results": results, "laps": laps, "corners": corners, "notes": notes,
            "last_3f": float(l3[1]) if l3 else None, "payouts": payouts}


PAYOUT_TYPES = {"win": "単勝", "place": "複勝", "wakuren": "枠連", "wide": "ワイド", "umaren": "馬連", "umatan": "馬単", "trio": "3連複", "tierce": "3連単"}


def parse_payouts(block: str) -> dict:
    """払戻金（券種 → 組 → 円）。馬連・ワイド・3連複・枠連は小さい順「1-2」、馬単・3連単は着順「2-1」（2026-11-24）."""
    out: dict = {}
    for cls, body in re.findall(r'<li class="(win|place|wakuren|wide|umaren|umatan|trio|tierce)">(.*?)</li>', block, re.S):
        t = PAYOUT_TYPES[cls]
        for num, yen in re.findall(r'<div class="num">(.*?)</div>\s*<div class="yen">([\d,]+)<', body, re.S):
            ns = [int(x) for x in re.findall(r"\d+", text(num))]
            if not ns:
                continue
            if t not in ("馬単", "3連単"):
                ns = sorted(ns)
            out.setdefault(t, {})["-".join(map(str, ns))] = int(yen.replace(",", ""))
    return out


def fetch_day(date: str) -> list[dict]:
    ymd = date.replace("-", "")
    top = post("/JRADB/accessS.html", "pw01sli00/AF")
    days = []
    for cname in sorted(set(re.findall(r"'(pw01srl\d{2}(\d{2})\d{8}" + ymd + r"/[0-9A-F]{2})'", top))):
        page = post("/JRADB/accessS.html", cname[0])
        allc = re.search(r"'(pw01ses\d{2}" + cname[1] + r"\d{8}" + ymd + r"/[0-9A-F]{2})'", page)
        if not allc:
            continue
        full = post("/JRADB/accessS.html", allc[1])
        blocks = full.split('<table class="basic narrow-xy striped">')[1:]
        races = [parse_race(b, date, cname[1]) for b in blocks]
        races = [r for r in races if r]
        for r in races:
            if not r["odds_cname"] or r["surface"] not in ("芝", "ダート"):
                continue
            page = post("/JRADB/accessO.html", r["odds_cname"])
            odds = {}
            for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", page, re.S):
                n = re.search(r'<td class="num">(\d+)</td>', tr)
                t = re.search(r'<td class="odds_tan">(?:<strong[^>]*>)?([\d.]+)<', tr)
                fm = re.search(r'class="min">([\d.]+)<.*?class="max">([\d.]+)<', tr, re.S)
                if n and t:
                    odds[int(n[1])] = (float(t[1]), (float(fm[1]), float(fm[2])) if fm else None)
            for h in r["results"]:
                if h["umaban"] in odds:
                    h["odds"], h["place_odds"] = odds[h["umaban"]]
            time.sleep(1)
            # 馬連・ワイド・馬単・3連複・3連単の全部の組み合わせの確定オッズ（jra_odds.py。依頼者の案 8）。data/jra-odds/日付.json に保存する
            key = r["odds_cname"][7:].split("Z")[0]
            exotic = {}
            for kind, cn in jra_odds.links(page, key).items():
                try:
                    exotic[kind] = jra_odds.parse(kind, post("/JRADB/accessO.html", cn))
                except SystemExit:
                    pass
                time.sleep(1)
            EXOTIC_ODDS[r["race_id"]] = exotic
        code, (slug, vname) = cname[1], VENUES[cname[1]]
        days.append({"file": f"jra-html-{date}-{slug}", "year": int(date[:4]), "kai": races[0]["kai"], "venue": slug,
                     "venue_code": code, "venue_name": vname, "day": races[0]["day"], "date": date, "races": races})
        time.sleep(1)
    return days


def fetch_training(race_ids: list[str]) -> list[dict]:
    from collect_training import URL, WAIT_SECONDS, parse
    out = []
    for rid in race_ids:
        res = requests.get(URL.format(rid), headers={"User-Agent": "Mozilla/5.0", "Connection": "close"}, timeout=20)
        res.encoding = "utf-8"
        if res.ok:
            out.append({"race_id": rid, "horses": parse(res.text)})
        else:
            print("training HTTP", res.status_code, rid, file=sys.stderr)
        time.sleep(WAIT_SECONDS)
    return out


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """model.ts の predictRaw と同じ."""
    out = np.zeros(len(X))
    for i, row in enumerate(X):
        s = 0.0
        for tree in model["trees"]:
            node = tree[0]
            while "v" not in node:
                v = row[node["f"]]
                if math.isnan(v):
                    if node["m"] == 0:
                        left = 0 <= node["t"]
                    else:
                        left = node["d"] == 1
                elif node["m"] == 1 and v == 0:
                    left = node["d"] == 1
                else:
                    left = v <= node["t"]
                node = tree[node["l"] if left else node["r"]]
            s += node["v"]
        out[i] = s
    return 1 / (1 + np.exp(-out))


RAW = Path(__file__).resolve().parents[2] / "data" / "jra-results"


# --- 海外の走（成績表にない）を過去走に足す。アプリ（app/api/races/[raceId]/route.ts の overseasRuns）と同じ形 ---
JRA_VENUES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
NAR_VENUES = ["門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀", "帯広", "ばんえい"]
OVERSEAS_PRIZE = [(r"\((?:GI|G1)\)", 17000), (r"\((?:GII|G2)\)", 5700), (r"\((?:GIII|G3)\)", 4100), (r"\(L\)", 2400), (r"\(OP\)", 2200),
                  (r"新馬", 750), (r"未勝利", 560)]
TIME_DIFF_BY_FINISH = [0, 0, 0.2, 0.4, 0.6, 0.7, 0.9, 1.1, 1.2, 1.4, 1.6, 1.7, 1.9, 2.2, 2.4, 2.7, 3.1]
OVERSEAS_GOING = {"良": 0, "稍": 1, "重": 2, "不": 3}
DEBUT_BODY_WEIGHT = {"牡": 476, "牝": 448, "セ": 468}  # アプリ（route.ts）と同じ
GRADED_PRIZE = 38_000_000  # 1着賞金がこれ以上（G3 以上）のレースの出走馬は、日本馬でも海外の走があるか調べる


def is_overseas_venue(venue: str) -> bool:
    return bool(venue) and not any(v in venue for v in JRA_VENUES) and not any(v in venue for v in NAR_VENUES)


def netkeiba_performances(horse_id: str) -> list[dict]:
    """netkeiba の戦績（db.netkeiba.com/horse/result/ID/）。1ページ目と、ページの番号のリンクがあればその先も."""
    rows: list[dict] = []
    pages, seen = [1], set()
    while pages:
        page = pages.pop(0)
        url = f"https://db.netkeiba.com/horse/result/{horse_id}/" + (f"?page={page}" if page > 1 else "")
        res = None
        for attempt in range(3):
            try:
                res = requests.get(url, timeout=30)
                if res.ok:
                    break
            except requests.RequestException:
                res = None
            time.sleep(5 * (attempt + 1))
        time.sleep(1)
        if res is None or not res.ok:
            print(f"netkeiba HTTP {res.status_code if res is not None else '-'} {horse_id}", file=sys.stderr, flush=True)
            break
        t = res.content.decode("utf-8", "replace")
        if page == 1:
            pages += sorted({int(x) for x in re.findall(r"[?&]page=(\d+)", t) if 1 < int(x) <= 50})
        for tb in re.findall(r"<table[^>]*>(.*?)</table>", t, re.S):
            trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tb, re.S)
            cells = [[htmlmod.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, re.S)] for tr in trs]
            if not cells or not cells[0] or cells[0][0] != "日付":
                continue
            head = [h.split("\n")[0] for h in cells[0]]
            for c in cells[1:]:
                v = dict(zip(head, c))
                key = json.dumps(v, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    rows.append(v)
    return rows


VENUE_MAP = json.loads((Path(__file__).resolve().parents[2] / "data/overseas/venue_map.json").read_text(encoding="utf-8"))


def overseas_course(venue_text: str, surface: int) -> dict:
    """海外の競馬場 → 近い JRA のコース（data/overseas/venue_map.json）。競馬場・回り・直線の区分・直線・高低差."""
    v, layout = next(((c, o) for pat, c, o in VENUE_MAP["map"] if pat in venue_text), VENUE_MAP["default"])
    c = features.COURSE.get((v, surface, layout if surface == 0 else 0)) or features.COURSE.get((v, surface, 0))
    return {"venue": v, "turn": 1 if v in (4, 5, 7) else 0, "sband": features.straight_band(c[0]), "c_straight": c[0], "c_elev": c[1]}


def overseas_runs(perf: list[dict], venue_map: bool = True) -> list[dict]:
    """海外のレースを、features.History.add_race の過去走と同じ形にする（アプリの overseasRuns と同じ値）.
    venue_map：競馬場を近い JRA のコースに読み替える（2026-11-26。False は前の形＝競馬場不明。overseas_eval.py の比べる基準）."""
    out = []
    for v in perf:
        if not is_overseas_venue(v.get("開催", "")):
            continue
        date = (v.get("日付") or "").replace("/", "-")
        course = re.match(r"^(芝|ダ)(\d+)", v.get("距離") or "")
        try:
            field = int(v.get("頭数") or "")
        except ValueError:
            field = 0
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", date) or not course or field <= 1:
            continue
        fin = v.get("着順") or ""
        finish = int(fin) if fin.isdigit() and int(fin) > 0 else None
        name = v.get("レース名") or ""
        prize = next((p for pat, p in OVERSEAS_PRIZE if re.search(pat, name)), 800)
        try:
            weight = float(v.get("斤量") or "")
        except ValueError:
            weight = None
        out.append({
            "date": date, "venue": 0, "surface": 0 if course[1] == "芝" else 1, "distance": int(course[2]),
            "going": OVERSEAS_GOING.get((v.get("馬場") or "")[:1]), "field": field, "finish": finish,
            "time_diff": TIME_DIFF_BY_FINISH[min(finish, len(TIME_DIFF_BY_FINISH) - 1)] if finish else None,
            "pace": None, "pos_first": None, "pos_last": None, "prize": prize, "weight": weight if weight and weight > 0 else None,
            "jockey": None, "tidx": None, "trainer": None, "victim": 0, "ab": None, **{k: None for k in features.AGARI_KEYS},
            "turn": None, "sband": None, "race_id": None, "c_straight": None, "c_elev": None, "form_at": None, "tm5_at": None,
            "ov_venue": v.get("開催", ""), "ov_name": name,  # 見積もり（overseas.improve）で使う
            "ov_r": v.get("R", ""), "ov_umaban": v.get("馬番", ""), "ov_finish": fin,  # 香港の区間タイム（hkjc.py）で使う
        })
        if venue_map:
            out[-1].update(overseas_course(v.get("開催", ""), out[-1]["surface"]))
    return out


def overseas_runs_for(days: list[dict], date: str) -> dict[str, list[dict]]:
    """海外の走がありそうな馬だけ netkeiba の戦績を取る：成績表にその日より前の走がない馬（新馬戦を除く。外国馬など）と、G3 以上のレースの出走馬."""
    known = set()
    for fp in (DATA / "json").glob("*/*.json"):
        for r in json.loads(fp.read_text(encoding="utf-8"))["races"]:
            if (r.get("date") or "") < date:
                known.update(features.clean_horse_name(h["horse"]) for h in r["results"])
    out: dict[str, list[dict]] = {}
    for day in days:
        for r in day["races"]:
            if r["surface"] not in ("芝", "ダート"):
                continue
            graded = ((r.get("prize") or [0])[0] or 0) >= GRADED_PRIZE
            debut = "新馬" in (r.get("race_name") or "") or "デビュー" in (r.get("race_name") or "")
            for h in r["results"]:
                if not h.get("horse_id") or not (graded or (h["horse"] not in known and not debut)):
                    continue
                perf = netkeiba_performances(h["horse_id"])
                # 見積もり（overseas.py。racedb の1着賞金・着差、主要国以外は格を1つ下げる、能力指数・タイム指数の式）
                runs = [overseas.improve(r, r["ov_venue"], r["ov_name"], h["horse_id"], h.get("sex")) for r in overseas_runs(perf)]
                if runs:
                    out[h["horse"]] = runs
    print(f"海外の走を足した馬: {len(out)} 頭（{sum(len(v) for v in out.values())} 走）", file=sys.stderr, flush=True)
    return out


def fill_owner_breeder(days: list[dict], date: str) -> None:
    """結果ページには2着以下の馬主・生産牧場がない。成績表の同じ馬名の、その日より前の一番新しい走から補う
    （空のままだと「初めて見る馬主・生産牧場」として計算され、学習データとずれる）."""
    last: dict[str, tuple[str, str | None, str | None]] = {}
    last_bw: dict[str, tuple[str, int]] = {}
    for fp in (DATA / "json").glob("*/*.json"):
        for r in json.loads(fp.read_text(encoding="utf-8"))["races"]:
            d = r.get("date") or ""
            for h in r["results"]:
                name = features.clean_horse_name(h["horse"])  # 成績表の「（12頭）」付きの馬名（2026-11-25）
                if h.get("owner") and d < date and (name not in last or d >= last[name][0]):
                    last[name] = (d, h.get("owner"), h.get("breeder"))
                if h.get("body_weight") and d < date and (name not in last_bw or d >= last_bw[name][0]):
                    last_bw[name] = (d, h["body_weight"])
    n = m = nb = 0
    for day in days:
        for r in day["races"]:
            for h in r["results"]:
                # 馬体重が「計不」（外国馬など）：アプリと同じく、前走の馬体重・増減0、なければ初出走馬の性別ごとの中央値
                # （空のままだと、学習データにない値として確率が大きく下がる）
                if h.get("body_weight") is None and h.get("status") not in ("取消", "除外"):
                    debut = "新馬" in (r.get("race_name") or "") or "デビュー" in (r.get("race_name") or "")
                    h["body_weight"] = last_bw[h["horse"]][1] if h["horse"] in last_bw else DEBUT_BODY_WEIGHT.get(h.get("sex") or "", 476)
                    h["body_weight_diff"] = None if debut else 0  # 新馬は成績表と同じく増減なし（11/22 までに保存した日は新馬の馬体重が欠けている）
                    nb += 1
                if h.get("owner"):
                    continue
                m += 1
                if h["horse"] in last:
                    _, h["owner"], h["breeder"] = last[h["horse"]]
                    n += 1
    print(f"馬主・生産牧場を前の走から補った: {n} / {m} 頭（残りは初出走など）。馬体重を補った: {nb} 頭", file=sys.stderr, flush=True)


def save_raw(date: str, days: list[dict]) -> None:
    """結果ページの中身（全レース。推定でない上がり3F・通過・ラップ・確定オッズ・競走中の出来事）をためる（後で特徴量に使う）."""
    RAW.mkdir(parents=True, exist_ok=True)
    keep = [{**d, "races": [{k: v for k, v in r.items() if k != "odds_cname"} for r in d["races"]]} for d in days]
    (RAW / f"{date}.json").write_text(json.dumps({"date": date, "source": "https://www.jra.go.jp/JRADB/accessS.html", "days": keep},
                                                 ensure_ascii=False) + "\n", encoding="utf-8")
    mine = {r["race_id"]: EXOTIC_ODDS[r["race_id"]] for d in days for r in d["races"] if EXOTIC_ODDS.get(r["race_id"])}
    if mine:
        od = DATA.parent / "jra-odds"
        od.mkdir(parents=True, exist_ok=True)
        import gzip  # 1日 約1.4MB（3連単が多い）なので gzip で保存する（約0.3MB）
        with gzip.open(od / f"{date}.json.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"date": date, "note": "確定オッズ（jra_odds.py）。ワイドは [下限, 上限]", "races": mine},
                                ensure_ascii=False, separators=(",", ":")) + "\n")


def store_run(r: dict) -> list:
    """海外の走（overseas_runs・overseas.improve の形）→ store.horses の1走（アプリの overseasRuns と同じ並び）."""
    return [r["date"], r["venue"], r["surface"], r["distance"], -1 if r["going"] is None else r["going"], r["field"],
            -1 if r["finish"] is None else r["finish"], -1 if r["time_diff"] is None else r["time_diff"],
            -1, r["pos_first"] or -1, r["pos_last"] or -1, r["prize"], r["weight"], -1, -1, r["tidx"], 0, r["ab"],
            r["ag_dev"], r["ag_rank"], r["early_gap"], r["fade"],
            r["turn"], r["sband"], None, r["c_straight"], r["c_elev"]]


def save_bet_inputs(date, days, rows, rid, raw3, extra_raw, model, mpath) -> None:
    """data/ml/day_check/inputs/日付.json.gz：レースごとに、各馬の着順・確定オッズ・モデルの生の確率・本体の入力と、払戻."""
    import gzip
    flat = {r["race_id"]: (d, r) for d in days for r in d["races"]}
    races = []
    for race_id in sorted(set(rid)):
        d, info = flat[race_id]
        res = {h["umaban"]: h for h in info["results"]}
        horses = []
        for i in np.flatnonzero(rid == race_id):
            h = res[rows[i]["umaban"]]
            x = [rows[i].get(f, float("nan")) for f in model["features"]]
            horses.append({"n": rows[i]["umaban"], "horse": rows[i]["horse"], "finish": h["finish"], "odds": h.get("odds"),
                           "place": list(h["place_odds"]) if h.get("place_odds") else None,
                           "top3": float(raw3[i]), "top2": float(extra_raw["top2"][i]) if "top2" in extra_raw else None,
                           "win": float(extra_raw["win"][i]) if "win" in extra_raw else None,
                           "x": [None if (v is None or (isinstance(v, float) and math.isnan(v))) else float(v) for v in x],
                           # 一致テスト（parity_day.ts）用：アプリと同じ入力から TS で計算し直して、項目ごとに比べる
                           "meta": {k: rows[i].get(k) for k in ("jockey", "trainer", "owner", "breeder")},
                           # 外国馬（海外の厩舎）と、足した海外の走（store.horses と同じ形）。アプリと同じ扱いを TS で確かめるため
                           **({"foreign": True} if features.is_foreign_trainer(rows[i].get("trainer")) else {}),
                           **({"extra": [store_run(x) for x in features.EXTRA_RUNS[rows[i]["horse"]] if x["date"] < date]}
                              if features.EXTRA_RUNS.get(rows[i]["horse"]) else {})})
        r0 = rows[int(np.flatnonzero(rid == race_id)[0])]
        races.append({"race_id": race_id, "date": date, "turn": r0.get("turn"), "sband": r0.get("sband"), "venue": d["venue_name"], "race_no": info["race_no"], "race_name": info["race_name"],
                      "course": f'{info["surface"]}{info["distance"]}', "payouts": info.get("payouts") or {}, "horses": horses})
    path = OUT / "day_check" / "inputs" / f"{date}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        # model_dir：モデルのフォルダ（data/ml からの相対。"" は本番、"prev" は 2025年までの一式）。day_bets.ts が同じフォルダの係数を読む
        mdir = Path(mpath).resolve().parent
        rel = "" if mdir == OUT.resolve() else str(mdir.relative_to(OUT.resolve()))
        fh.write(json.dumps({"date": date, "model": Path(mpath).name, "model_dir": rel, "features": model["features"], "races": races},
                            ensure_ascii=False, separators=(",", ":")) + "\n")


def backfill() -> None:
    """結果ページで見られる過去の開催日（約2か月分）のうち、まだ保存していない日を取る（予測はしない）."""
    top = post("/JRADB/accessS.html", "pw01sli00/AF")
    dates = sorted({f"{m[:4]}-{m[4:6]}-{m[6:]}" for m in re.findall(r"'pw01srl\d{12}(\d{8})/", top)})
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    for date in dates:
        if date >= today or (RAW / f"{date}.json").exists():  # 当日は途中までしかないので取らない（夕方の毎日の実行で取る）
            continue
        days = fetch_day(date)
        if days:
            save_raw(date, days)
            print(date, sum(len(d["races"]) for d in days), "races", file=sys.stderr, flush=True)


def odds_backfill() -> None:
    """結果ページで見られる過去の開催日のうち、全部の組み合わせの確定オッズ（data/jra-odds）がまだない日を取る（依頼者の案 8・2 の検証用）."""
    top = post("/JRADB/accessS.html", "pw01sli00/AF")
    dates = sorted({f"{m[:4]}-{m[4:6]}-{m[6:]}" for m in re.findall(r"'pw01srl\d{12}(\d{8})/", top)})
    from datetime import datetime, timedelta, timezone
    today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    for date in dates:
        if date >= today or (DATA.parent / "jra-odds" / f"{date}.json.gz").exists():
            continue
        EXOTIC_ODDS.clear()
        days = fetch_day(date)
        if days:
            save_raw(date, days)
            print(date, len(EXOTIC_ODDS), "races（オッズ）", file=sys.stderr, flush=True)


def fill_course_letter(days: list[dict], date: str) -> None:
    """芝の使用コース（A〜D）：結果ページにないので、その日の成績表（後から予測し直すとき。使用コースはレース前に分かる情報）から入れる.
    成績表がまだない当日は、features.load_baba が data/baba/baba.csv の使用コース（collect_baba_live.py が馬場情報のページから足す）を使う"""
    pat = re.compile(r"[（(]芝([A-DＡ-Ｄ])コース[）)]")
    letters = {}
    for fp in (DATA / "json" / date[:4]).glob("*.json"):
        d = json.loads(fp.read_text(encoding="utf-8"))
        if d.get("date") == date or any(r.get("date") == date for r in d.get("races", [])):
            m = pat.search(json.dumps(d, ensure_ascii=False))
            if m:
                letters[str(d.get("venue_code"))] = m.group(1)
    n = 0
    for d in days:
        if not pat.search(json.dumps(d, ensure_ascii=False)) and str(d.get("venue_code")) in letters:
            d["course_note"] = f"(芝{letters[str(d['venue_code'])]}コース)"
            n += 1
    print(f"使用コースを成績表から入れた: {n} / {len(days)} 競馬場", file=sys.stderr)


def main() -> None:
    if "--backfill" in sys.argv:
        backfill()
        return
    if "--odds-backfill" in sys.argv:
        odds_backfill()
        return
    from datetime import datetime, timedelta, timezone
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    if "--resave" in sys.argv:  # 結果ページを取り直して保存し直すだけ（解析を直したとき。予測はしない）
        days = fetch_day(date)
        if days:
            save_raw(date, days)
            print(date, sum(len(d["races"]) for d in days), "races resaved", file=sys.stderr)
        return
    if "--saved" in sys.argv:  # 保存済みの結果ページの中身から予測し直す（取り直さない）
        days = json.loads((RAW / f"{date}.json").read_text(encoding="utf-8"))["days"]
    else:
        days = fetch_day(date)
        if not days:
            print(f"{date}: 結果ページなし", file=sys.stderr)
            return
        save_raw(date, days)
    days = copy.deepcopy(days)
    repair_results(days)
    if not os.environ.get("NO_OWNER_FILL"):  # NO_OWNER_FILL=1：直す前（2026-11-22 より前）の動きを再現する（入力の見張りの確かめ用）
        fill_owner_breeder(days, date)
    fill_course_letter(days, date)
    features.EXTRA_RUNS = overseas_runs_for(days, date)
    features.EXTRA_RUNS_RACES = {r["race_id"] for d in days for r in d["races"]}  # 足すのはその日のレースだけ（アプリと同じ）
    # 外国馬の扱い（2026-11-26。アプリと同じ）：レーティングの初めの値、騎手（短期免許は通算）・調教師・馬主・生産者（日本馬の中央値）
    features.FOREIGN_FIX = {"rating": True, "people": True}
    ri = DATA.parent / "overseas" / "rating_init.json"
    features.RATING_INIT = json.loads(ri.read_text(encoding="utf-8"))["coef"] if ri.exists() else None
    wb = DATA.parent / "overseas" / "wbrr.json"
    features.WBRR = json.loads(wb.read_text(encoding="utf-8"))["foreign"] if wb.exists() else None
    # 外国人騎手の海外主要G1の成績（ファイルにない外国馬の騎手は JRA-VAN Ver.World をその場で検索する）
    jw = DATA.parent / "overseas" / "jockey_world.json"
    features.JOCKEY_WORLD = json.loads(jw.read_text(encoding="utf-8")) if jw.exists() else None
    if features.JOCKEY_WORLD:
        import jockey_world  # noqa: PLC0415
        for d in days:
            for r in d["races"]:
                for h in r["results"]:
                    k = features.norm_jockey(h.get("jockey"))
                    if features.is_foreign_trainer(h.get("trainer")) and k and k not in features.JOCKEY_WORLD["jockeys"]:
                        try:
                            e = jockey_world.lookup(h["jockey"])
                        except Exception:  # noqa: BLE001
                            e = None
                        if e:
                            features.JOCKEY_WORLD["jockeys"][k] = e
    flat = [r for d in days for r in d["races"] if r["surface"] in ("芝", "ダート")]
    print(f"{date}: {len(flat)} races", file=sys.stderr, flush=True)
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "jra-seiseki"
    (root / "json").mkdir(parents=True)
    # 成績表がもう出ている日（--saved で後から予測し直すとき）は、その日のファイルを除く（結果ページの分と重なるため）
    for y in (DATA / "json").iterdir():
        (root / "json" / y.name).mkdir()
        for fp in y.glob("*.json"):
            if date[:4] == y.name and any(r.get("date") == date for r in json.loads(fp.read_text(encoding="utf-8"))["races"]):
                continue
            (root / "json" / y.name / fp.name).symlink_to(fp.resolve())
    (root / "json" / "extra").mkdir()
    for d in days:
        (root / "json" / "extra" / (d["file"] + ".json")).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    # 調教評価：既存＋（--training なら）当日分。血統：同じ馬の前の走の値
    (tmp / "ml").symlink_to(OUT.resolve())
    if (DATA.parent / "baba").exists():  # クッション値・含水率・使用コース（features.load_baba）
        (tmp / "baba").symlink_to((DATA.parent / "baba").resolve())
    (tmp / "training").mkdir()
    for p in (DATA.parent / "training").glob("*.jsonl"):
        (tmp / "training" / p.name).symlink_to(p.resolve())
    if "--training" in sys.argv:
        tr = fetch_training([r["race_id"] for r in flat])
        (tmp / "training" / "extra.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tr), encoding="utf-8")
    ped_src = DATA.parent / "pedigree" / "all" / "runner_features.csv"
    (tmp / "pedigree" / "all").mkdir(parents=True)
    lines = ped_src.read_text(encoding="utf-8").splitlines()
    head = lines[0].split(",")
    seen: dict[str, list] = {}
    meta = {}
    for d in (DATA / "json").glob("*/*.json"):
        for r in json.loads(d.read_text(encoding="utf-8"))["races"]:
            meta[r.get("race_id")] = (r.get("surface"), r.get("distance") or 0, r.get("date") or "")
    for ln in lines[1:]:
        v = ln.split(",")
        seen.setdefault(v[2], []).append(v)
    add = []
    for r in flat:
        for h in r["results"]:
            cands = [v for v in seen.get(h["horse_id"] or "", []) if meta.get(v[0], (None,))[0] == r["surface"]]
            if cands:
                best = min(cands, key=lambda v: (abs(meta[v[0]][1] - r["distance"]), "".join(reversed(meta[v[0]][2]))))
                add.append(",".join([r["race_id"], str(h["umaban"])] + best[2:]))
    (tmp / "pedigree" / "all" / "runner_features.csv").write_text("\n".join(lines + add) + "\n", encoding="utf-8")

    rows, _ = build_dataset(root)
    ids = {r["race_id"] for r in flat}
    rows = [r for r in rows if r["race_id"] in ids]
    # --model=パス：その日より前のデータで学習したモデル（後から予測し直すときは、その日を学習したモデルを使わない）
    mpath = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--model=")), str(OUT / "model.json"))
    model = json.loads(Path(mpath).read_text(encoding="utf-8"))
    print("model", mpath, len(model["features"]), "features", file=sys.stderr, flush=True)
    X = np.array([[r.get(f, float("nan")) for f in model["features"]] for r in rows], dtype=np.float64)
    rid = np.array([r["race_id"] for r in rows])
    raw3 = predict(model, X)
    p = normalize_sum(rid, raw3)
    # 買い方の成績（day_bets.ts）用の入力：モデルの生の確率（3着以内・連対・勝率）と本体の入力（☆の2段目）。画面と同じ計算をするため
    extra_raw = {}
    for kind in ("win", "top2"):
        mp = Path(mpath).with_name(f"model_{kind}.json")
        if mp.exists():
            mk = json.loads(mp.read_text(encoding="utf-8"))
            extra_raw[kind] = predict(mk, np.array([[r.get(f, float("nan")) for f in mk["features"]] for r in rows], dtype=np.float64))
    y = np.array([r["label"] for r in rows])
    shutil.rmtree(tmp)

    from sklearn.metrics import log_loss, roc_auc_score
    byid = {r["race_id"]: r for r in flat}
    races_out, ai_hits, pop_hits, n3 = [], 0, 0, 0
    top1 = {"win": 0, "top3": 0}
    for race_id in sorted(ids):
        idx = [i for i in range(len(rows)) if rid[i] == race_id]
        if not idx:
            continue
        info = byid[race_id]
        res = {h["umaban"]: h for h in info["results"]}
        order = sorted(idx, key=lambda i: -p[i])
        ai3 = [rows[i]["umaban"] for i in order[:3]]
        pop3 = [h["umaban"] for h in sorted((h for h in info["results"] if h["popularity"]), key=lambda h: h["popularity"])[:3]]
        top = {h["umaban"] for h in info["results"] if h["finish"] and h["finish"] <= 3}
        ai_hits += len(top & set(ai3)); pop_hits += len(top & set(pop3)); n3 += 3
        f1 = res[ai3[0]]["finish"]
        top1["win"] += f1 == 1; top1["top3"] += bool(f1 and f1 <= 3)
        races_out.append({
            "race_id": race_id, "venue": next(d["venue_name"] for d in days if race_id[4:6] == d["venue_code"]), "race_no": info["race_no"],
            "race_name": info["race_name"], "course": f'{info["surface"]}{info["distance"]}', "going": info["going"],
            "ai_top3": [{"umaban": rows[i]["umaban"], "horse": rows[i]["horse"], "p": round(float(p[i]), 3),
                         "finish": res[rows[i]["umaban"]]["finish"], "pop": res[rows[i]["umaban"]]["popularity"],
                         "odds": res[rows[i]["umaban"]].get("odds")} for i in order[:3]],
            "result_top3": [{"umaban": h["umaban"], "horse": h["horse"], "pop": h["popularity"], "odds": h.get("odds"),
                             "ai_rank": 1 + [rows[i]["umaban"] for i in order].index(h["umaban"])}
                            for h in sorted((h for h in info["results"] if h["finish"] and h["finish"] <= 3), key=lambda h: h["finish"])],
            "ai_hits": len(top & set(ai3)), "pop_hits": len(top & set(pop3)),
        })
    # 本番の入力の見張り（依頼者の案 4）：学習データの直近1年と比べて、欠けの割合・平均のずれが大きい項目
    drift_flags = None
    if (OUT / "feature_stats.json").exists():
        import drift
        drift_flags = drift.check(X, model["features"])
        print("入力のずれ:", json.dumps(drift_flags[:20], ensure_ascii=False), file=sys.stderr)
        if os.environ.get("GITHUB_STEP_SUMMARY"):
            with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
                fh.write(f"### {date} 入力の見張り：ずれが大きい項目 {len(drift_flags)}\n\n")
                for x in drift_flags[:30]:
                    fh.write(f"- {x['feature']}（{'・'.join(x['flags'])}）欠け {x['missing_day']:.0%}（学習 {x['missing_train']:.0%}）平均のずれ {x['mean_z']}\n")
    n = len(races_out)
    out = {"date": date, "races": n, "runners": len(rows), "training": "--training" in sys.argv,
           "train_rank_coverage": round(float(np.mean([r.get("train_rank") is not None and not (isinstance(r.get("train_rank"), float) and math.isnan(r["train_rank"])) for r in rows])), 3),
           "ped_coverage": round(float(np.mean([not math.isnan(r.get("ped_top3", float("nan")) or float("nan")) for r in rows])), 3),
           "logloss": round(float(log_loss(y, p)), 4), "auc": round(float(roc_auc_score(y, p)), 4),
           "ai_top3_hit": round(ai_hits / n3, 4), "pop_top3_hit": round(pop_hits / n3, 4),
           "ai_top1_win": round(top1["win"] / n, 4), "ai_top1_top3": round(top1["top3"] / n, 4), "drift": drift_flags, "detail": races_out}
    save_bet_inputs(date, days, rows, rid, raw3, extra_raw, model, mpath)
    if "--inputs-only" in sys.argv:  # 買い方の成績の入力だけ作る（過去の日を別のモデルで。AI成績の day_check/日付.json は変えない）
        return
    path = OUT / "day_check" / f"{date}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

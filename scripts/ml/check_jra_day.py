"""JRA のレース結果ページ（https://www.jra.go.jp/JRADB/accessS.html）から、その日の結果を取って AI の予測と突き合わせる.

    cd scripts/ml && python check_jra_day.py 2026-11-21            # → data/ml/day_check/2026-11-21.json
    cd scripts/ml && python check_jra_day.py 2026-11-21 --training # 調教評価（netkeiba の調教ページ）も取って入れる（Actions で使う）
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
        hw = re.match(r"(\d+)\s*\(([+\-]?\d+)\)", text(cell.get("h_weight", "")))
        corners = [int(x) for x in re.findall(r"<li[^>]*>(\d+)</li>", tr)]
        status = place if place in ("取消", "除外", "中止", "失格") else None
        results.append({
            "waku": num(re.search(r"waku/(\d)\.png", cell.get("waku", ""))[1]) if "waku/" in cell.get("waku", "") else None,
            "umaban": num(text(cell.get("num", ""))), "horse": text(cell.get("horse", "")), "horse_id": hid[1] if hid else None,
            "sex": sa[:1] if sa else None, "age": num(sa[1:]) if sa else None, "weight": num(text(cell.get("weight", ""))),
            "jockey": jockey, "jockey_mark": jmark[1] if jmark else None, "owner": None, "breeder": None,
            "trainer": text(cell.get("trainer", "")).replace(" ", ""), "status": status,
            "body_weight": int(hw[1]) if hw else None, "body_weight_diff": int(hw[2]) if hw else None,
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
    oc = re.search(r"'(pw151ou[^']+)'", block)
    return {"odds_cname": oc[1] if oc else None, "race_no": race_no, "race_name": name, "distance": dist, "course": course, "surface": surface, "going": going,
            "weather": baba.get("天候"), "prize": [int(prize1[1].replace(",", "")) * 10000] if prize1 else [],
            "race_id": f"{date[:4]}{venue_code}{kai:02d}{day:02d}{race_no:02d}", "date": date, "kai": kai, "day": day,
            "starters": len(starters), "results": results, "laps": laps, "corners": corners, "notes": notes,
            "last_3f": float(l3[1]) if l3 else None}


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


def save_raw(date: str, days: list[dict]) -> None:
    """結果ページの中身（全レース。推定でない上がり3F・通過・ラップ・確定オッズ・競走中の出来事）をためる（後で特徴量に使う）."""
    RAW.mkdir(parents=True, exist_ok=True)
    keep = [{**d, "races": [{k: v for k, v in r.items() if k != "odds_cname"} for r in d["races"]]} for d in days]
    (RAW / f"{date}.json").write_text(json.dumps({"date": date, "source": "https://www.jra.go.jp/JRADB/accessS.html", "days": keep},
                                                 ensure_ascii=False) + "\n", encoding="utf-8")


def backfill() -> None:
    """結果ページで見られる過去の開催日（約2か月分）のうち、まだ保存していない日を取る（予測はしない）."""
    top = post("/JRADB/accessS.html", "pw01sli00/AF")
    dates = sorted({f"{m[:4]}-{m[4:6]}-{m[6:]}" for m in re.findall(r"'pw01srl\d{12}(\d{8})/", top)})
    for date in dates:
        if (RAW / f"{date}.json").exists():
            continue
        days = fetch_day(date)
        if days:
            save_raw(date, days)
            print(date, sum(len(d["races"]) for d in days), "races", file=sys.stderr, flush=True)


def main() -> None:
    if "--backfill" in sys.argv:
        backfill()
        return
    from datetime import datetime, timedelta, timezone
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    date = args[0] if args else datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")
    days = fetch_day(date)
    if not days:
        print(f"{date}: 結果ページなし", file=sys.stderr)
        return
    save_raw(date, days)
    flat = [r for d in days for r in d["races"] if r["surface"] in ("芝", "ダート")]
    print(f"{date}: {len(flat)} races", file=sys.stderr, flush=True)
    tmp = Path(tempfile.mkdtemp())
    root = tmp / "jra-seiseki"
    (root / "json").mkdir(parents=True)
    for y in (DATA / "json").iterdir():
        (root / "json" / y.name).symlink_to(y.resolve())
    (root / "json" / "extra").mkdir()
    for d in days:
        (root / "json" / "extra" / (d["file"] + ".json")).write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    # 調教評価：既存＋（--training なら）当日分。血統：同じ馬の前の走の値
    (tmp / "ml").symlink_to(OUT.resolve())
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
    model = json.loads((OUT / "model.json").read_text(encoding="utf-8"))
    X = np.array([[r.get(f, float("nan")) for f in model["features"]] for r in rows], dtype=np.float64)
    rid = np.array([r["race_id"] for r in rows])
    p = normalize_sum(rid, predict(model, X))
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
    n = len(races_out)
    out = {"date": date, "races": n, "runners": len(rows), "training": "--training" in sys.argv,
           "train_rank_coverage": round(float(np.mean([r.get("train_rank") is not None and not (isinstance(r.get("train_rank"), float) and math.isnan(r["train_rank"])) for r in rows])), 3),
           "ped_coverage": round(float(np.mean([not math.isnan(r.get("ped_top3", float("nan")) or float("nan")) for r in rows])), 3),
           "logloss": round(float(log_loss(y, p)), 4), "auc": round(float(roc_auc_score(y, p)), 4),
           "ai_top3_hit": round(ai_hits / n3, 4), "pop_top3_hit": round(pop_hits / n3, 4),
           "ai_top1_win": round(top1["win"] / n, 4), "ai_top1_top3": round(top1["top3"] / n, 4), "detail": races_out}
    path = OUT / "day_check" / f"{date}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "detail"}, ensure_ascii=False))


if __name__ == "__main__":
    main()

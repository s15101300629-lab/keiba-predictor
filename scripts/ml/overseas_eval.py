"""海外の走の扱いの比較（2026-11-23、ジャパンカップの外国馬のため）.

    cd scripts/ml && python overseas_eval.py   # → data/ml/overseas_eval.json

比べる3通り（海外の走は学習データ＝成績表にないので、予測するときだけ過去走に足す。アプリと同じ）
- なし：海外の走を足さない
- 今：アプリ・答え合わせの今の形（check_jra_day.overseas_runs。格で決めた賞金・着順ごとのタイム差・能力指数なし）
- 見積もり：racedb.com のレース結果で、1着賞金（円に直す）と着差（馬身→秒）を本物の値にし、能力指数・タイム指数を成績表の
  JRA の走から見積もる（芝ダごとに、log 1着賞金・タイム差・頭数で当てる最小二乗）。racedb にない走は今の形で、
  欧州・香港・豪州・NZ・UAE・米国・カナダ・サウジ以外の国の格は1つ下げる（依頼者の指示「区別する」）
対象：成績表 2017年〜の、海外の走がある馬（外国馬・遠征帰りの日本馬。data/overseas/histories.json）の、海外の走のあとの JRA の走
評価：年ごとに前の年までで学習（海外の走なしのデータ。アプリの本体と同じ作り）。試す年 2021〜2026
- 対象の馬の logloss（レースごとに合計3にそろえた後）、AI の順位と着順、3着以内に来た馬の AI の順位
- 外国馬は1走ずつ並べる
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parent))
import features  # noqa: E402
import overseas  # noqa: E402
from check_jra_day import OVERSEAS_PRIZE, overseas_runs  # noqa: E402
from train import DATA, OUT, PARAMS, TRAIN_FROM  # noqa: E402
from tune_params import normalize_sum  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
TEST_YEARS = ["2021", "2022", "2023", "2024", "2025", "2026"]
# 通貨 → 円（おおよその値。年による違いは見ない）
FX = {"ポンド": 190.0, "ユーロ": 160.0, "ドル": 150.0, "オーストラリアドル": 100.0, "ニュージーランドドル": 88.0, "香港ドル": 19.0,
      "ディルハム": 41.0, "リヤル": 40.0, "カナダドル": 108.0, "ウォン": 0.11, "円": 1.0}
MAJOR = ["アイルランド", "イギリス", "フランス", "アメリカ", "アラブ首長国連邦", "オーストラリア", "ニュージーランド", "香港", "カナダ", "ドイツ", "サウジアラビア"]
# netkeiba の開催（競馬場名）→ 国が主要国か。主要国の競馬場名の一部（racedb にない走の格を下げるかどうかにだけ使う）
MAJOR_VENUES = ["アスコット", "ヨーク", "サンダウン", "ニューマーケット", "グッドウッド", "エプソム", "ドンカスター", "ヘイドック", "レスター", "ケンプトン",
                "リングフィールド", "ニューベリー", "チェスター", "カラ", "レパーズタウン", "パリロンシャン", "ロンシャン", "シャンティイ", "ドーヴィル",
                "サンクルー", "メゾンラフィット", "メイダン", "キングアブドゥル", "シャティン", "ハッピーバレー", "フレミントン", "コーフィールド", "ムーニーバレー",
                "ランドウィック", "ローズヒル", "エラズリー", "トレンサム", "サンタアニタ", "デルマー", "チャーチルダウンズ", "キーンランド", "ベルモント",
                "サラトガ", "アケダクト", "ガルフストリーム", "ウッドバイン", "バーデンバーデン", "ケルン", "ホッペガルテン", "ミュンヘン", "ハンブルク",
                "イギリス", "アイルランド", "フランス", "アメリカ", "ドバイ", "香港", "豪州", "オーストラリア", "ドイツ"]
SEC_PER_LENGTH = 0.17  # racedb の JRA のレースで決め直す（calibrate_length）
SHORT = {"ハナ": 0.05, "短頭": 0.05, "アタマ": 0.1, "頭": 0.1, "クビ": 0.15, "首": 0.15, "大差": 12.0, "同着": 0.0}


def parse_margin(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    if s in SHORT:
        return SHORT[s]
    m = re.fullmatch(r"(\d+(?:\.\d+)?)", s)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(?:(\d+)\s*)?(\d)/(\d)", s)  # 1 1/2 など
    if m:
        return float(m.group(1) or 0) + int(m.group(2)) / int(m.group(3))
    return None


def jp_number(t: str) -> float | None:
    """「737,260」「1億8,000万」「130万」→ 数."""
    t = t.replace(",", "")
    total = 0.0
    m = re.fullmatch(r"(?:([\d\.]+)億)?(?:([\d\.]+)万)?([\d\.]+)?", t)
    if not m or not any(m.groups()):
        return None
    if m.group(1):
        total += float(m.group(1)) * 1e8
    if m.group(2):
        total += float(m.group(2)) * 1e4
    if m.group(3):
        total += float(m.group(3)) * (1e4 if (m.group(1) and not m.group(2)) else 1)
    return total


def parse_race(r: dict) -> dict | None:
    """racedb のレースのページ → 1着賞金（円）と、馬名（英語・カナ）ごとの着順・勝ち馬からの着差（馬身）・単勝倍率."""
    head = r.get("text") or r["head"]
    dm = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", head)
    date = f"{dm.group(1)}-{int(dm.group(2)):02d}-{int(dm.group(3)):02d}" if dm else None
    m = re.search(r"１着賞金\s*([\d,\.億万]+)\s*([^\s\d,\.億万][^\s]*)", head)
    prize = None
    if m:
        cur = m.group(2)
        rate = FX.get(cur) or next((v for k, v in sorted(FX.items(), key=lambda kv: -len(kv[0])) if cur.startswith(k)), None)
        amount = jp_number(m.group(1))
        if rate and amount:
            prize = amount * rate
    table = next((t for t in r["tables"] if t and any("着差" in c for c in t[0])), None)
    if not table:
        return {"date": date, "prize": prize, "horses": {}}
    head_row = table[0]
    col = {k: next((i for i, c in enumerate(head_row) if k in c.replace(" ", "")), None) for k in ("馬名", "着差", "単勝")}
    horses = {}
    cum = 0.0
    for pos, row in enumerate(table[1:], start=1):
        if col["馬名"] is None or col["馬名"] >= len(row):
            continue
        name = row[col["馬名"]]
        mg = row[col["着差"]] if col["着差"] is not None and col["着差"] < len(row) else ""
        if pos > 1:
            x = parse_margin(mg)
            cum = cum + x if x is not None and cum is not None else None
        odds = None
        if col["単勝"] is not None and col["単勝"] < len(row):
            try:
                odds = float(row[col["単勝"]])
            except ValueError:
                pass
        en = re.split(r"[（(]", name)[0].strip()
        kana = (re.search(r"[（(]([^）)]+)[）)]", name) or [None, ""])[1]
        x = {"finish": pos, "lengths": cum, "odds": odds}
        for key in {norm(en), norm(kana)} - {""}:
            horses[key] = x
    return {"date": date, "prize": prize, "horses": horses}


def norm(s: str) -> str:
    return re.sub(r"[\s'’.\-・*＊]", "", s or "").lower()


def load_sources():
    hist = json.loads((ROOT / "data/overseas/histories.json").read_text(encoding="utf-8"))
    rdb_path = ROOT / "data/overseas/racedb.json"
    rdb = json.loads(rdb_path.read_text(encoding="utf-8")) if rdb_path.exists() else {"horses": {}, "races": {}}
    races = {k: parse_race(v) for k, v in rdb["races"].items()}
    return hist, rdb, races


def racedb_for_run(hid: str, title: str, date: str, rdb: dict, races: dict) -> dict | None:
    """その馬のその日の racedb の結果（着順・着差・1着賞金）."""
    h = rdb["horses"].get(hid)
    if not h:
        return None
    jp = norm(title.split("(")[0].split(" ")[0])
    en = norm((re.search(r"\(([^)]+)\)", title) or [None, ""])[1])
    for k in h.get("races", []):
        r = races.get(k)
        if not r or r["date"] != date:
            continue
        x = r["horses"].get(en) or r["horses"].get(jp)
        if x:
            return {**x, "prize": r["prize"]}
    return None


def calibrate_length(rdb: dict, races: dict, td_jra: dict) -> tuple[float, int]:
    """racedb の JRA のレースの着差（馬身）と、成績表のタイム差（秒。馬名と日付で合わせる）から、1馬身あたりの秒（原点を通る最小二乗）."""
    jra = {k for h in rdb["horses"].values() for k in h.get("jra_races", [])}
    xs, ys = [], []
    for k in jra:
        r = races.get(k)
        if not r or not r["date"]:
            continue
        seen = set()
        for name_key, x in r["horses"].items():
            t = td_jra.get((name_key, r["date"]))
            if t is None or x["lengths"] is None or x["finish"] == 1 or id(x) in seen:
                continue
            seen.add(id(x))
            xs.append(x["lengths"]); ys.append(t)
    xs, ys = np.array(xs), np.array(ys)
    if len(xs) < 30:
        return SEC_PER_LENGTH, len(xs)
    ok = (xs < 15) & (ys < 3)
    return float((xs[ok] * ys[ok]).sum() / (xs[ok] * xs[ok]).sum()), int(ok.sum())


VARIANTS = {  # 名前 → (1着賞金の決め方, 競馬場の読み替え, 1 レーティングの初めの値, 2 騎手・調教師・馬主・生産者)
    # 1着賞金 old：racedb の値、なければ格（主要国以外は1つ下げる）。ifha：IFHA の格（overseas.ifha_prize。2026-11-26、依頼者の指示）
    "今": ("old", False, False, False),
    "新": ("ifha", True, True, True),
}
RATING_INIT_PATH = ROOT / "data" / "overseas" / "rating_init.json"


def fit_rating_init(rows: list[dict]) -> dict:
    """日本馬の「能力指数（今回の芝ダの近走の直近加重平均 ab_wmean）とレーティング」の関係（芝ダ別、最小二乗）.
    レーティングが落ち着いた馬（更新5日以上）の、学習の期間（TRAIN_FROM〜）の行で当てる."""
    coef = {}
    for surf in (0, 1):
        out = {}
        for name in ("elom", "elos"):
            sel = [(r["ab_wmean"], r[name]) for r in rows if r["date"] >= TRAIN_FROM and r["surface"] == surf
                   and not math.isnan(r["ab_wmean"]) and r[name + "_n"] >= 5 and not features.is_foreign_trainer(r.get("trainer"))]
            x = np.array([a for a, _ in sel]); yv = np.array([b for _, b in sel])
            b, a = np.polyfit(x, yv, 1)
            out[name] = [round(float(a), 4), round(float(b), 4)]
            out[name + "_n"] = len(sel)
            out[name + "_r"] = round(float(np.corrcoef(x, yv)[0, 1]), 4)
        coef[str(surf)] = out
    return coef


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = args or list(VARIANTS)
    hist, rdb, races = load_sources()
    # 馬名（成績表。clean_horse_name でそろえた名前）→ netkeiba の馬ID
    by_name = {features.clean_horse_name(nm): hid for hid, v in hist.items() for nm in v["names"]}
    perf = {hid: v["rows"] for hid, v in hist.items()}
    print("馬:", len(hist), "racedb のレース:", len(races), "比べる:", names, file=sys.stderr, flush=True)

    # 1. 海外の走なしで作る（学習・比べる基準）
    features.EXTRA_RUNS = {}
    features.FOREIGN_FIX = {"rating": False, "people": False}
    rows0, h0 = features.build_dataset(DATA)
    rating_init = fit_rating_init(rows0)
    RATING_INIT_PATH.write_text(json.dumps({"note": "外国馬のレーティングの初めの値の式：レーティング＝切片＋傾き×能力指数（ab_wmean）。芝ダ別（scripts/ml/overseas_eval.py の fit_rating_init）",
                                            "coef": rating_init}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("レーティングの初めの値の式:", rating_init, file=sys.stderr, flush=True)
    # 見積もりの式：JRA の走（成績表）の能力指数・タイム指数を、芝ダごとに log 1着賞金・タイム差・頭数で当てる
    coef = {}
    for surf in (0, 1):
        X, ya, yt = [], [], []
        for runs in h0.horses.values():
            for r in runs:
                if r["surface"] != surf or r["ab"] is None or r["prize"] is None or r["time_diff"] is None or r["tidx"] is None:
                    continue
                X.append([1.0, math.log(max(r["prize"], 100)), min(r["time_diff"], 3.0), math.log(r["field"])])
                ya.append(r["ab"]); yt.append(r["tidx"])
        X = np.array(X)
        coef[surf] = {"ab": np.linalg.lstsq(X, np.array(ya), rcond=None)[0].tolist(), "tidx": np.linalg.lstsq(X, np.array(yt), rcond=None)[0].tolist(), "n": len(X)}
    # 着差の換算：racedb の JRA のレースの出走馬（カナ名）の成績表のタイム差
    td_jra: dict[tuple, float] = {}
    for runs_name, runs in h0.horses.items():
        for r in runs:
            if r["time_diff"] is not None:
                td_jra[(norm(re.split(r"[（(]", runs_name)[0]), r["date"])] = r["time_diff"]
    spl, n_cal = calibrate_length(rdb, races, td_jra)
    print("見積もりの式:", coef, "1馬身:", spl, file=sys.stderr, flush=True)

    def improved_runs(hid: str, mode: str, venue_map: bool) -> list[dict]:
        """overseas.improve と同じ見積もり（mode="major" が今のアプリ）。"old" は 2026-11-25 までの1着賞金の決め方."""
        out = []
        for run in overseas_runs(perf[hid], venue_map=venue_map):
            x = racedb_for_run(hid, hist[hid].get("title", ""), run["date"], rdb, races)
            rp = min(x["prize"] / 10000, 50000) if x and x.get("prize") else None  # JRA の一番高い1着賞金（5億円）で頭打ち
            major = any(m in run["ov_venue"] for m in MAJOR_VENUES)
            graded = next((p for pat, p in OVERSEAS_PRIZE[:5] if re.search(pat, run["ov_name"])), None)  # G1〜OP
            lower = {17000: 5700, 5700: 4100, 4100: 2400}
            if mode == "ifha":
                prize = round(overseas.ifha_prize(run["ov_venue"], run["ov_name"], run["date"], run["surface"], run["distance"], hid, run["prize"]), 1)
            elif rp is not None:
                prize = rp
            elif not major and graded is not None:
                prize = lower.get(graded, graded)
            else:
                prize = run["prize"]
            td = run["time_diff"]
            if x and x.get("lengths") is not None and run["finish"]:
                td = round(min(x["lengths"] * spl, 9.9), 1)
            run = {**run, "prize": prize, "time_diff": td}
            if mode == "ifha":  # 香港の走は公式の区間タイムから（hkjc.py）
                hk = overseas.hk_fields(run, run["ov_venue"])
                if hk:
                    run.update({k: hk[k] for k in ("ag_dev", "ag_rank", "early_gap", "fade", "pos_first", "pos_last")})
                    if run["finish"]:
                        run["time_diff"] = td = hk["time_diff"]
            if run["finish"] and td is not None:
                c = coef[run["surface"]]
                z = [1.0, math.log(max(prize, 100)), min(td, 3.0), math.log(run["field"])]
                run["ab"] = round(float(np.dot(c["ab"], z)), 4)
                run["tidx"] = round(float(np.dot(c["tidx"], z)), 4)
            out.append(run)
        return out

    keep = [i for i, r in enumerate(rows0) if r["date"] >= TRAIN_FROM]
    meta = [{k: rows0[i].get(k) for k in ("date", "race_id", "horse", "finish", "popularity", "label", "trainer")} for i in keep]
    Xs = {"なし": np.array([[rows0[i][f] for f in features.FEATURES] for i in keep], dtype=np.float64)}
    est_rows: dict[str, dict] = {}
    del rows0, h0
    runs_by_mode: dict[str, dict] = {}
    for name in names:
        mode, vmap, fix_r, fix_p = VARIANTS[name]
        if (mode, vmap) not in runs_by_mode:
            runs_by_mode[(mode, vmap)] = {nm: improved_runs(hid, mode, vmap) for nm, hid in by_name.items()}
        features.EXTRA_RUNS = {k: v for k, v in runs_by_mode[(mode, vmap)].items() if v}
        features.FOREIGN_FIX = {"rating": bool(fix_r), "people": fix_p, "rating_all": fix_r == "all"}
        features.RATING_INIT = rating_init
        rows, _ = features.build_dataset(DATA)
        Xs[name] = np.array([[rows[i][f] for f in features.FEATURES] for i in keep], dtype=np.float64)
        est_rows[name] = {j: rows[i]["_estimated"] for j, i in enumerate(keep) if rows[i].get("_estimated")}
        del rows
        print("作った:", name, file=sys.stderr, flush=True)
    features.EXTRA_RUNS = {}
    features.FOREIGN_FIX = {"rating": False, "people": False}

    # 対象：海外の走のあとの JRA の走
    any_runs = next(iter(runs_by_mode.values()))
    first_ov = {nm: min(r["date"] for r in runs) for nm, runs in any_runs.items() if runs}
    year = np.array([r["date"][:4] for r in meta])
    y = np.array([r["label"] for r in meta], float)
    race_id = np.array([r["race_id"] for r in meta])
    target = np.array([r["horse"] in first_ov and r["date"] > first_ov[r["horse"]] for r in meta])
    foreign = np.array([features.is_foreign_trainer(r["trainer"]) for r in meta])
    out = {"note": "海外の走の扱いの比較。学習は海外の走なし、予測のときだけ足す。logloss は合計3にそろえた後。"
                   "1＝JRA の走がない外国馬のレーティングの初めの値、2＝外国馬の騎手（短期免許は通算）・調教師・馬主・生産者（日本馬の中央値）",
           "sec_per_length": spl, "length_pairs": n_cal, "coef": coef, "rating_init": rating_init, "racedb_races": len(races), "years": {}, "foreign_runs": []}
    # 日本馬（外国馬でない馬）の値が変わらないかの確かめ：「今」と比べて、値が変わった行の数（項目ごと）。
    # 出走馬の中の比較（レーティングの順位など）は、同じレースの外国馬の値が変わると変わるので分けて数える
    base = Xs.get("今")
    if base is not None:
        out["japanese_changed"] = {}
        for name in names:
            if name == "今":
                continue
            d = ~((Xs[name] == base) | (np.isnan(Xs[name]) & np.isnan(base)))
            jp = d[~foreign]
            out["japanese_changed"][name] = {f: int(jp[:, k].sum()) for k, f in enumerate(features.FEATURES) if jp[:, k].any()}
    preds = {k: np.full(len(y), np.nan) for k in Xs}
    for ty in TEST_YEARS:
        vy = str(int(ty) - 1)
        fit, val, tr, te = year < vy, year == vy, year < ty, year == ty
        if not (te & target).any():
            continue
        X0 = Xs["なし"]
        b = lgb.train(PARAMS, lgb.Dataset(X0[fit], y[fit]), 6000, valid_sets=[lgb.Dataset(X0[val], y[val])],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        m = lgb.train(PARAMS, lgb.Dataset(X0[tr], y[tr]), max(b.best_iteration, 50))
        races_t = np.isin(race_id, np.unique(race_id[te & target]))
        for k, X in Xs.items():
            p = m.predict(X[races_t])
            preds[k][races_t] = normalize_sum(race_id[races_t], p)
        print(ty, int((te & target).sum()), file=sys.stderr, flush=True)
    done = target & ~np.isnan(preds["なし"])

    def summary(mask) -> dict:
        res = {"runs": int(mask.sum()), "top3": int(y[mask].sum())}
        for k, p in preds.items():
            rank = np.array([1 + int((preds[k][race_id == race_id[i]] > p[i]).sum()) for i in np.flatnonzero(mask)])
            res[k] = {"logloss": round(float(log_loss(y[mask], np.clip(p[mask], 1e-6, 1 - 1e-6), labels=[0, 1])), 5),
                      "mean_prob": round(float(p[mask].mean()), 4), "mean_rank": round(float(rank.mean()), 2),
                      "top3_rank_mean": round(float(rank[y[mask] == 1].mean()), 2) if y[mask].sum() else None}
        return res

    out["all"] = summary(done)
    out["foreign"] = summary(done & foreign)
    out["japanese"] = summary(done & ~foreign)
    # 外国馬のいるレースの日本馬（予測が変わるのは、合計3にそろえる分と、出走馬の中の比較の分だけのはず）
    same_race = np.isin(race_id, np.unique(race_id[done & foreign])) & ~foreign & ~np.isnan(preds["なし"])
    out["japanese_in_foreign_races"] = summary(same_race)
    for i in np.flatnonzero(done & foreign):
        r = meta[i]
        out["foreign_runs"].append({"date": r["date"], "race_id": r["race_id"], "horse": r["horse"], "finish": r["finish"], "popularity": r["popularity"],
                                    "estimated": {k: est_rows[k].get(int(i)) for k in est_rows if est_rows[k].get(int(i))},
                                    **{k: {"prob": round(float(preds[k][i]), 4), "rank": 1 + int((preds[k][race_id == race_id[i]] > preds[k][i]).sum())} for k in preds}})
    tag = "" if not args else "_" + "_".join(VARIANTS_TAG.get(n, n) for n in names)
    (OUT / f"overseas_eval{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("all", "foreign", "japanese", "japanese_in_foreign_races")}, ensure_ascii=False), file=sys.stderr)


VARIANTS_TAG = {"今": "now", "新": "ifha"}


if __name__ == "__main__":
    main()

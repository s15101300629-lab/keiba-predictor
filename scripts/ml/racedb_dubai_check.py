"""ドバイ（メイダン）の走が racedb の結果とつながらない理由を調べる（2026-11-26、依頼者の案 6）.

    cd scripts/ml && python racedb_dubai_check.py   # → data/ml/racedb_dubai_check.json

netkeiba の海外の走のうちメイダンの走で racedb とつながっていない馬について、racedb の馬のページを取り直し、
- ページが取れるか、ページの行にその日付（前後1日も）があるか、/k/ のリンクがあるか
を記録する。3秒に1ページ・最大40頭
"""

from __future__ import annotations

import json
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_racedb import BASE, clean, get  # noqa: E402
from check_jra_day import overseas_runs  # noqa: E402
from overseas_eval import load_sources, racedb_for_run  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    hist, rdb, races = load_sources()
    out = []
    for hid, v in hist.items():
        miss = [r for r in overseas_runs(v["rows"]) if "メイダン" in r["ov_venue"] and not racedb_for_run(hid, v.get("title", ""), r["date"], rdb, races)]
        if not miss or len(out) >= 40:
            continue
        h = rdb["horses"].get(hid) or {}
        cands = h.get("candidates") or []
        rec = {"hid": hid, "title": v.get("title", "")[:40], "missing": [(r["date"], r["ov_name"]) for r in miss], "candidates": cands[:3], "pages": []}
        for href, _ in cands[:2]:
            page = get(BASE + href)
            if not page:
                rec["pages"].append({"href": href, "loaded": False})
                continue
            rows = []
            for row in re.findall(r"<tr.*?</tr>", page, re.S):
                t = clean(row)
                for d, _ in rec["missing"]:
                    dd = date.fromisoformat(d)
                    pats = {f"{x.year}. {x.month}. {x.day}" for x in (dd - timedelta(days=1), dd, dd + timedelta(days=1))} | \
                           {f"{x.year}.{x.month:02d}.{x.day:02d}" for x in (dd - timedelta(days=1), dd, dd + timedelta(days=1))}
                    if any(p in re.sub(r"\s+", " ", t) or p.replace(" ", "") in t.replace(" ", "") for p in pats) or "メイダン" in t or "ドバイ" in t:
                        rows.append({"text": t[:200], "k": re.findall(r'href="(/k/\d+\.html)"', row)})
                        break
            dates_on_page = re.findall(r"(\d{4})\.\s*(\d{1,2})\.\s*(\d{1,2})", clean(page))
            rec["pages"].append({"href": href, "loaded": True, "rows": rows[:10], "n_dates": len(dates_on_page),
                                 "first_dates": ["-".join(x) for x in dates_on_page[:5]], "has_meydan_word": "メイダン" in page})
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False)[:400], file=sys.stderr, flush=True)
    (ROOT / "data/ml/racedb_dubai_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

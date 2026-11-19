"""netkeiba スマホ版のレース結果ページ（db.sp.netkeiba.com/race/<race_id>/）から、各馬の「備考」（出遅れなど）を取得する.

    cd scripts/ml && python collect_remarks.py 2026

- PC 版の備考列は空（有料会員向け）だが、スマホ版はログインなしで見える
- 対象：data/jra-seiseki/csv/<年>/races.csv の平地レース
- 保存先：data/remarks/<年>.jsonl（1行1レース）。続きから再開できる
"""

from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
URL = "https://db.sp.netkeiba.com/race/{}/"
WAIT_SECONDS = 2.0


def parse(html: str) -> list[dict]:
    body = html.split("<tbody>", 1)[1].split("</tbody>", 1)[0] if "<tbody>" in html else ""
    horses = []
    for row in re.findall(r"<tr>(.*?)</tr>", body, re.S):
        umaban = re.search(r"<td[^>]*>\s*(\d+)\s*</td>\s*<!--馬番-->", row)
        remark = re.search(r"<td[^>]*>\s*([^<]*?)\s*</td>\s*<!--備考-->", row)
        horse = re.search(r'href="https://db\.sp\.netkeiba\.com/horse/(\w+)/">([^<]+)</a>', row)
        if not umaban:
            continue
        horses.append({
            "umaban": int(umaban.group(1)),
            "horse_id": horse.group(1) if horse else None,
            "horse": horse.group(2).strip() if horse else None,
            "remark": remark.group(1) if remark else None,
        })
    return horses


def main() -> None:
    year = sys.argv[1]
    races = [r["race_id"] for r in csv.DictReader(open(ROOT / f"data/jra-seiseki/csv/{year}/races.csv", encoding="utf-8"))
             if r["surface"] in ("芝", "ダート")]
    out = ROOT / f"data/remarks/{year}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["race_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [r for r in races if r not in done]
    print(f"{year}: {len(races)} races, done {len(done)}, todo {len(todo)}", file=sys.stderr, flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)"
    failures = 0
    with out.open("a", encoding="utf-8") as f:
        for i, race_id in enumerate(todo, 1):
            res = None
            for attempt in range(4):
                try:
                    res = session.get(URL.format(race_id), timeout=20)
                    res.encoding = "utf-8"
                    if res.ok:
                        break
                    print(f"HTTP {res.status_code} {race_id}", file=sys.stderr, flush=True)
                except requests.RequestException as e:
                    print(f"error {race_id}: {e}", file=sys.stderr, flush=True)
                time.sleep(30 * (attempt + 1))
            if res is None or not res.ok:
                failures += 1
                if failures >= 10:
                    sys.exit("too many failures")
                continue
            failures = 0
            f.write(json.dumps({"race_id": race_id, "horses": parse(res.text)}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 100 == 0:
                print(f"{i}/{len(todo)}", file=sys.stderr, flush=True)
            time.sleep(WAIT_SECONDS)


if __name__ == "__main__":
    main()

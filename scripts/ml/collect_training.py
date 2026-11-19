"""netkeiba の調教ページ（追い切りの短評と A〜D 評価）を、成績表のレースについて取得する.

    cd scripts/ml && python collect_training.py 2026

- 対象：data/jra-seiseki/csv/<年>/races.csv の平地レース（障害は除く）
- 保存先：data/training/<年>.jsonl（1行1レース）。続きから再開できる
- netkeiba への負担を抑えるため、1ページごとに間をあける
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
URL = "https://race.netkeiba.com/race/oikiri.html?race_id={}"
WAIT_SECONDS = 2.0
ROW = re.compile(r'<tr class\s*=\s*"OikiriDataHead\d+ HorseList"[^>]*>(.*?)</tr>', re.S)


def text(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def parse(html: str) -> list[dict]:
    horses = []
    for row in ROW.findall(html):
        umaban = re.search(r'class="Umaban">\s*(\d+)', row)
        horse = re.search(r'href="https://db\.netkeiba\.com/horse/(\w+)"[^>]*>([^<]+)</a>', row)
        critic = re.search(r'class="Training_Critic">(.*?)</td>', row, re.S)
        rank = re.search(r'class="Rank_[^"]*">(.*?)</td>', row, re.S)
        horses.append({
            "umaban": int(umaban.group(1)) if umaban else None,
            "horse_id": horse.group(1) if horse else None,
            "horse": horse.group(2).strip() if horse else None,
            "critic": text(critic.group(1)) if critic else None,
            "rank": text(rank.group(1)) if rank else None,
        })
    return horses


def main() -> None:
    year = sys.argv[1]
    races = [r["race_id"] for r in csv.DictReader(open(ROOT / f"data/jra-seiseki/csv/{year}/races.csv", encoding="utf-8"))
             if r["surface"] in ("芝", "ダート")]
    out = ROOT / f"data/training/{year}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["race_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    todo = [r for r in races if r not in done]
    print(f"{year}: {len(races)} races, done {len(done)}, todo {len(todo)}", file=sys.stderr, flush=True)
    session = requests.Session()
    session.headers["User-Agent"] = "Mozilla/5.0"
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
                time.sleep(30 * (attempt + 1))  # 失敗したら長めに待つ
            if res is None or not res.ok:
                failures += 1
                if failures >= 10:  # 続けて失敗するなら止める（保存しないので、再実行で続きから取れる）
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

from __future__ import annotations

import concurrent.futures
import gzip
import json
import os
import re
import urllib.request
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path.cwd()
MANIFEST_PATH = Path("jra-pdf-index/manifest.json")
OUT = Path("jra-pdf-data")
OUT.mkdir(exist_ok=True)
YEAR = int(os.environ["JRA_YEAR"])
WORKERS = int(os.environ.get("JRA_WORKERS", "8"))
PARSER = ROOT / "scripts/jra_pdf_probe.py"

manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
entries = [item for item in manifest["pdfs"] if item["year"] == YEAR]
if not entries:
    raise RuntimeError(f"No official JRA PDFs listed for {YEAR}")

def fetch_official_cancelled_races(date: str, venue: str) -> tuple[list[int], str]:
    year, month, day = (int(value) for value in date.split("-"))
    calendar_url = (
        f"https://www.jra.go.jp/keiba/calendar{year}/{year}/{month}/{month:02d}{day:02d}.html"
    )
    request = urllib.request.Request(
        calendar_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; JRAResultResearch/1.0)"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        raw = response.read()
        charset = response.headers.get_content_charset()
    if not charset:
        meta = re.search(rb"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", raw[:4096], re.I)
        charset = meta.group(1).decode("ascii") if meta else "utf-8"
    try:
        html = raw.decode(charset)
    except (LookupError, UnicodeDecodeError):
        html = raw.decode("utf-8", errors="replace")
    if "\ufffd" in html:
        for fallback in ("cp932", "shift_jis"):
            try:
                candidate = raw.decode(fallback)
            except (LookupError, UnicodeDecodeError):
                continue
            if "\ufffd" not in candidate and "競馬" in candidate:
                html = candidate
                break
    text_only = re.sub(r"<[^>]+>", " ", html)
    text_only = re.sub(r"\s+", "", text_only)
    cancelled = [
        int(match.group(1))
        for match in re.finditer(
            rf"{re.escape(venue)}競馬は第(\d+)レースを取りやめ", text_only
        )
    ]
    return sorted(set(cancelled)), calendar_url


def parse_one(entry: dict) -> dict:
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix=f"jra-{YEAR}-") as temp_dir:
        env = os.environ.copy()
        env.update({
            "JRA_PDF_URL": entry["sourceUrl"],
            "JRA_OUTPUT_DIR": temp_dir,
            "JRA_DIAGNOSTICS": "0",
            "JRA_EXPECTED_RACES": "0",
            "JRA_JOCKEY_NAMES_FILE": str(OUT / "jockey-names.json"),
        })
        result = subprocess.run(
            [sys.executable, str(PARSER)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=150,
        )
        summary_path = Path(temp_dir) / "summary.json"
        races_path = Path(temp_dir) / "parsed-races.json"
        if result.returncode != 0 or not summary_path.exists() or not races_path.exists():
            raise RuntimeError(
                f"PDF parse failed: {entry['sourceUrl']}\n"
                f"exit={result.returncode}\nstdout={result.stdout[-2000:]}\n"
                f"stderr={result.stderr[-4000:]}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        races = json.loads(races_path.read_text(encoding="utf-8"))

    if summary["validationErrorCount"] != 0:
        raise RuntimeError(
            f"Validation errors in {entry['sourceUrl']}: {summary['validationErrors']}"
        )
    if not races:
        raise RuntimeError(f"No race rows parsed: {entry['sourceUrl']}")

    cancelled_races = []
    race_numbers = sorted(race["raceNumber"] for race in races)
    if len(races) == 12:
        pass
    elif 1 <= len(races) < 12:
        first = races[0]
        missing = sorted(set(range(1, 13)) - set(race_numbers))
        cancelled, calendar_url = fetch_official_cancelled_races(first["date"], first["venue"])
        if missing != cancelled:
            raise RuntimeError(
                f"JRA official calendar does not confirm omitted races: "
                f"parsed={len(races)}, missing={missing}, cancelled={cancelled}, "
                f"calendar={calendar_url}"
            )
        cancelled_races = cancelled
    else:
        raise RuntimeError(f"Unexpected race count {len(races)} in {entry['sourceUrl']}")
    for race in races:
        if not race.get("raceKey"):
            raise RuntimeError(f"Race key missing: {entry['sourceUrl']} R{race['raceNumber']}")
        if len(race["horses"]) != race["declaredRunnerCount"]:
            raise RuntimeError(
                f"Runner count mismatch: {entry['sourceUrl']} R{race['raceNumber']}"
            )
        numbers = [horse["horseNumber"] for horse in race["horses"]]
        if len(numbers) != len(set(numbers)):
            raise RuntimeError(
                f"Duplicate horse number: {entry['sourceUrl']} R{race['raceNumber']}"
            )
        for horse in race["horses"]:
            if not horse.get("horseName") or not horse.get("jockeyName"):
                raise RuntimeError(
                    f"Required horse/jockey field missing: {entry['sourceUrl']} "
                    f"R{race['raceNumber']} H{horse.get('horseNumber')}"
                )
            horse["sourceUrl"] = entry["sourceUrl"]
            horse["sourcePdfSha256"] = summary["sourceSha256"]
        race["sourceUrl"] = entry["sourceUrl"]
        race["sourcePdfSha256"] = summary["sourceSha256"]
        race["sourcePdfBytes"] = summary["pdfBytes"]
        race["sourcePdfPageCount"] = summary["pageCount"]

    return {
        "year": YEAR,
        "sourceUrl": entry["sourceUrl"],
        "sourcePdfSha256": summary["sourceSha256"],
        "pdfBytes": summary["pdfBytes"],
        "pageCount": summary["pageCount"],
        "raceCount": len(races),
        "cancelledRaceNumbers": cancelled_races,
        "horseCount": sum(len(race["horses"]) for race in races),
        "missingBodyWeightCount": summary["missingBodyWeightCount"],
        "missingSexCount": summary["missingSexCount"],
        "missingMarginCount": summary["missingMarginCount"],
        "races": races,
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }

results = []
failures = []
with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {executor.submit(parse_one, entry): entry for entry in entries}
    for future in concurrent.futures.as_completed(futures):
        entry = futures[future]
        try:
            results.append(future.result())
        except Exception as exc:
            failures.append({"sourceUrl": entry["sourceUrl"], "error": str(exc)})
            print(f"FAILED {entry['sourceUrl']}: {exc}", flush=True)

if failures:
    report = {
        "year": YEAR,
        "expectedPdfCount": len(entries),
        "successPdfCount": len(results),
        "failurePdfCount": len(failures),
        "failures": failures,
    }
    (OUT / f"validation-{YEAR}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raise RuntimeError(
        f"{YEAR}: {len(failures)} of {len(entries)} official PDFs failed validation"
    )

races = [race for result in results for race in result["races"]]
race_keys = [race["raceKey"] for race in races]
if len(race_keys) != len(set(race_keys)):
    duplicates = sorted({key for key in race_keys if race_keys.count(key) > 1})
    raise RuntimeError(f"Duplicate race keys in {YEAR}: {duplicates[:20]}")

races.sort(key=lambda race: (race["date"], race["venue"], race["raceNumber"]))
year_file = OUT / f"jra-results-{YEAR}.jsonl.gz"
with gzip.open(year_file, "wt", encoding="utf-8", newline="\n") as output:
    for race in races:
        output.write(json.dumps(race, ensure_ascii=False, separators=(",", ":")) + "\n")

report = {
    "year": YEAR,
    "sourceIndex": manifest["sourceIndexUrl"],
    "expectedPdfCount": len(entries),
    "successPdfCount": len(results),
    "failurePdfCount": 0,
    "raceCount": len(races),
    "horseCount": sum(len(race["horses"]) for race in races),
    "uniqueRaceKeyCount": len(set(race_keys)),
    "duplicateRaceKeyCount": 0,
    "missingBodyWeightCount": sum(item["missingBodyWeightCount"] for item in results),
    "missingSexCount": sum(item["missingSexCount"] for item in results),
    "missingMarginCount": sum(item["missingMarginCount"] for item in results),
    "sourcePdfSha256Count": len({item["sourcePdfSha256"] for item in results}),
    "totalSourcePdfBytes": sum(item["pdfBytes"] for item in results),
    "outputFile": str(year_file),
    "outputBytes": year_file.stat().st_size,
    "elapsedSeconds": round(sum(item["elapsedSeconds"] for item in results), 2),
    "parallelWorkers": WORKERS,
    "cancelledRaceCount": sum(len(item.get("cancelledRaceNumbers", [])) for item in results),
}
(OUT / f"validation-{YEAR}.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(report, ensure_ascii=False))

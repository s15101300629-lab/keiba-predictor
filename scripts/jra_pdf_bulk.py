from __future__ import annotations

import contextlib
import io
import json
import os
import runpy
import shutil
import tempfile
from collections import Counter
from pathlib import Path

YEAR = int(os.environ.get("JRA_YEAR", "2025"))
ROOT = Path("data/jra-results")
ROOT.mkdir(parents=True, exist_ok=True)

# Rebuild the manifest directly from JRA's current official index.
runpy.run_path("scripts/jra_pdf_index.py")
manifest = json.loads(Path("jra-pdf-index/manifest.json").read_text(encoding="utf-8"))
entries = [item for item in manifest["pdfs"] if item["year"] == YEAR]
if not entries:
    raise RuntimeError(f"No official JRA PDFs found for {YEAR}")
if len({item["sourceUrl"] for item in entries}) != len(entries):
    raise RuntimeError(f"Duplicate PDF URLs in {YEAR} manifest")

all_races = []
failures = []
pdf_summaries = []
race_keys = set()
horse_rows = 0

for index, entry in enumerate(entries, 1):
    temp_dir = Path(tempfile.mkdtemp(prefix=f"jra-{YEAR}-"))
    os.environ["JRA_PDF_URL"] = entry["sourceUrl"]
    os.environ["JRA_OUTPUT_DIR"] = str(temp_dir)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            runpy.run_path("scripts/jra_pdf_probe.py")
        summary_path = temp_dir / "summary.json"
        races_path = temp_dir / "parsed-races.json"
        if not summary_path.exists() or not races_path.exists():
            raise RuntimeError("Parser did not create summary and race data")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        races = json.loads(races_path.read_text(encoding="utf-8"))
        if summary.get("validationErrorCount") != 0:
            raise RuntimeError(f"Parser validation errors: {summary.get('validationErrors')}")
        if not races:
            raise RuntimeError("No races parsed")
        for race in races:
            key = race.get("raceKey")
            if not key or key in race_keys:
                raise RuntimeError(f"Missing or duplicate race key: {key}")
            race_keys.add(key)
            if race.get("parsedRunnerCount") != race.get("declaredRunnerCount"):
                raise RuntimeError(f"Runner count mismatch: {key}")
            for horse in race["horses"]:
                if not horse.get("horseKey") or not horse.get("horseName"):
                    raise RuntimeError(f"Missing normalized horse identity: {key}")
                horse_rows += 1
            race["sourcePdf"] = entry["sourceUrl"]
        all_races.extend(races)
        pdf_summaries.append({
            "sourcePdf": entry["sourceUrl"],
            "raceCount": len(races),
            "horseCount": sum(len(race["horses"]) for race in races),
            "pageCount": summary["pageCount"],
        })
    except Exception as exc:
        failures.append({"sourcePdf": entry["sourceUrl"], "error": str(exc)})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    if index % 12 == 0 or index == len(entries):
        print(json.dumps({
            "year": YEAR,
            "processedPdfCount": index,
            "expectedPdfCount": len(entries),
            "parsedRaceCount": len(all_races),
            "parsedHorseCount": horse_rows,
            "failureCount": len(failures),
        }, ensure_ascii=False))

if failures:
    report = {
        "year": YEAR,
        "expectedPdfCount": len(entries),
        "successfulPdfCount": len(pdf_summaries),
        "failureCount": len(failures),
        "failures": failures,
    }
    Path(f"jra-bulk-failures-{YEAR}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    raise RuntimeError(f"{YEAR}: {len(failures)} of {len(entries)} PDFs failed; no annual dataset written")

all_races.sort(key=lambda race: (race["date"], race["venue"], race["raceNumber"]))
data = {
    "schemaVersion": 1,
    "sourceIndexUrl": manifest["sourceIndexUrl"],
    "year": YEAR,
    "pdfCount": len(pdf_summaries),
    "raceCount": len(all_races),
    "horseRowCount": horse_rows,
    "pdfSummaries": pdf_summaries,
    "races": all_races,
}
target = ROOT / f"{YEAR}.json"
target.write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
validation = {
    "year": YEAR,
    "pdfCount": len(pdf_summaries),
    "raceCount": len(all_races),
    "horseRowCount": horse_rows,
    "uniqueRaceKeys": len(race_keys),
    "duplicateRaceKeys": len(all_races) - len(race_keys),
    "fileBytes": target.stat().st_size,
    "failureCount": 0,
}
(ROOT / f"{YEAR}.validation.json").write_text(
    json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(validation, ensure_ascii=False))

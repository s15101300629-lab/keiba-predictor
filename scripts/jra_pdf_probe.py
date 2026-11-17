from __future__ import annotations

import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path

PDF_URL = "https://jra.jp/datafile/seiseki/report/2025/2025-1kyoto1.pdf"
OUT = Path("jra-pdf-probe")
OUT.mkdir(exist_ok=True)

request = urllib.request.Request(
    PDF_URL,
    headers={
        "User-Agent": "Mozilla/5.0 (compatible; JRAResultResearch/1.0)",
        "Accept": "application/pdf",
    },
)
with urllib.request.urlopen(request, timeout=45) as response:
    pdf_bytes = response.read()
    content_type = response.headers.get("Content-Type", "")

if not pdf_bytes.startswith(b"%PDF"):
    raise RuntimeError(
        f"JRA response is not a PDF: content-type={content_type}, bytes={len(pdf_bytes)}"
    )

try:
    import fitz
except ImportError as exc:
    raise RuntimeError("PyMuPDF is required to inspect the JRA PDF text layer") from exc

def clean(value: str) -> str:
    value = re.sub(r"[\x00-\x1f]", "", value)
    return re.sub(r"\s+", "", value)

document = fitz.open(stream=pdf_bytes, filetype="pdf")
pages = []
layout_blocks = []
layout_lines = []

for page_number, page in enumerate(document, start=1):
    text = page.get_text("text", sort=True)
    pages.append({"page": page_number, "characters": len(text), "text": text})
    page_dict = page.get_text("dict")
    for block in page_dict["blocks"]:
        if block.get("type") != 0:
            continue
        block_text = "\n".join(
            span["text"]
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).strip()
        if block_text:
            layout_blocks.append({
                "page": page_number,
                "bbox": [round(value, 2) for value in block["bbox"]],
                "text": block_text,
            })
        for line in block.get("lines", []):
            line_text = "".join(span["text"] for span in line.get("spans", [])).strip()
            if line_text:
                layout_lines.append({
                    "page": page_number,
                    "bbox": [round(value, 2) for value in line["bbox"]],
                    "text": line_text,
                })

combined = "\n\n".join(
    f"===== PAGE {item['page']} =====\n{item['text']}" for item in pages
)
(OUT / "sample-extracted.txt").write_text(combined, encoding="utf-8")
(OUT / "layout-blocks.json").write_text(
    json.dumps(layout_blocks, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "layout-lines.json").write_text(
    json.dumps(layout_lines, ensure_ascii=False, indent=2), encoding="utf-8"
)

# A race card is one column on one page. The PDF text blocks can contain
# multiple horse rows, so runner extraction is based on individual text lines.
headers = []
for block in layout_blocks:
    match = re.search(r"第\s*(\d+)\s*競走", block["text"])
    if match:
        headers.append({
            "page": block["page"],
            "x": block["bbox"][0],
            "y0": block["bbox"][1],
            "y1": block["bbox"][3],
            "raceNumber": int(match.group(1)),
            "text": block["text"],
        })

parsed_races = []
validation_errors = []
coat_pattern = r"黒鹿|青鹿|青毛|鹿|栗|芦|白毛|白|青|栃栗|鹿毛"

for header in headers:
    page_number = header["page"]
    left_column = header["x"] < 420
    column_blocks = [
        block for block in layout_blocks
        if block["page"] == page_number
        and ((block["bbox"][0] < 420) == left_column)
    ]
    column_lines = [
        line for line in layout_lines
        if line["page"] == page_number
        and ((line["bbox"][0] < 420) == left_column)
    ]

    headcount = None
    headcount_y = None
    for block in column_blocks:
        match = re.search(r"（\s*(\d+)\s*頭\s*）", block["text"])
        if match:
            headcount = int(match.group(1))
            headcount_y = block["bbox"][1]
            break

    if headcount is None:
        validation_errors.append(
            f"race {header['raceNumber']}: declared horse count not found"
        )
        continue

    # Excluded runners appear in the table but are not part of the final
    # declared field count. Keep their records and mark them explicitly.
    excluded_numbers = set()
    for block in column_blocks:
        if "競走除外" not in block["text"]:
            continue
        match = re.match(r"\s*([1-8])\s*(\d{1,2})", block["text"])
        if match:
            excluded_numbers.add(int(match.group(2)))

    # Group line fragments by vertical position. A 3.0pt tolerance joins
    # columns belonging to one horse row without joining adjacent runners.
    candidates = [
        line for line in column_lines
        if line["bbox"][1] >= header["y1"] + 25
        and line["bbox"][1] < headcount_y + 1
    ]
    candidates.sort(key=lambda item: (
        (item["bbox"][1] + item["bbox"][3]) / 2,
        item["bbox"][0],
    ))

    clusters = []
    for line in candidates:
        center_y = (line["bbox"][1] + line["bbox"][3]) / 2
        if not clusters or abs(center_y - clusters[-1]["centerY"]) > 3.0:
            clusters.append({"centerY": center_y, "lines": [line]})
        else:
            clusters[-1]["lines"].append(line)
            clusters[-1]["centerY"] = sum(
                (part["bbox"][1] + part["bbox"][3]) / 2
                for part in clusters[-1]["lines"]
            ) / len(clusters[-1]["lines"])

    horses = []
    for cluster in clusters:
        row_lines = sorted(cluster["lines"], key=lambda item: item["bbox"][0])
        number_line = next(
            (
                clean(line["text"])
                for line in row_lines
                if line["bbox"][0] < (25 if left_column else 435)
            ),
            "",
        )
        number_match = re.match(r"^([1-8])(\d{1,2})", number_line)
        if not number_match:
            continue

        horse_info = next(
            (
                clean(line["text"])
                for line in row_lines
                if (25 if left_column else 455)
                <= line["bbox"][0]
                < (155 if left_column else 585)
            ),
            "",
        )
        info_match = re.match(
            rf"^(.*?)((?:牡|牝|せん|セン)?)(\d)({coat_pattern})(\d{{2}})$",
            horse_info,
        )
        if not info_match:
            validation_errors.append(
                f"race {header['raceNumber']}: horse {number_match.group(2)} "
                f"horse-info parse failed: {horse_info!r}"
            )
            continue

        horse_number = int(number_match.group(2))
        horses.append({
            "finishOrder": len(horses) + 1,
            "frameNumber": int(number_match.group(1)),
            "horseNumber": horse_number,
            "horseName": info_match.group(1),
            "sex": info_match.group(2) or None,
            "age": int(info_match.group(3)),
            "coatColor": info_match.group(4),
            "carriedWeightKg": int(info_match.group(5)),
            "status": "excluded" if horse_number in excluded_numbers else "finished_or_ran",
            "sourcePage": page_number,
            "sourceY": round(cluster["centerY"], 2),
        })

    horse_numbers = [horse["horseNumber"] for horse in horses]
    duplicate_numbers = sorted({
        number for number in horse_numbers if horse_numbers.count(number) > 1
    })
    active_count = sum(horse["status"] != "excluded" for horse in horses)
    if duplicate_numbers:
        validation_errors.append(
            f"race {header['raceNumber']}: duplicate horse numbers {duplicate_numbers}"
        )
    if active_count != headcount:
        validation_errors.append(
            f"race {header['raceNumber']}: parsed active runners {active_count} "
            f"!= declared count {headcount} (all parsed rows={len(horses)}, "
            f"excluded={len(excluded_numbers)})"
        )

    header_text = clean(header["text"])
    date_match = re.search(r"(\d{4})年(\d+)月(\d+)日", header_text)
    venue_match = re.search(r"\d{4}年\d+([^）]+)", header_text)
    distance_match = re.search(r"(\d)[，,](\d{3})", header_text)
    surface_match = re.search(r"（(芝|ダート|障害)[^）]*）", header_text)
    parsed_races.append({
        "raceNumber": header["raceNumber"],
        "date": (
            f"{int(date_match.group(1)):04d}-{int(date_match.group(2)):02d}-"
            f"{int(date_match.group(3)):02d}"
            if date_match else None
        ),
        "venue": venue_match.group(1) if venue_match else None,
        "distanceMeters": (
            int(distance_match.group(1) + distance_match.group(2))
            if distance_match else None
        ),
        "surface": surface_match.group(1) if surface_match else None,
        "declaredRunnerCount": headcount,
        "parsedRowCountIncludingExclusions": len(horses),
        "excludedRunnerCount": len(excluded_numbers),
        "validatedActiveRunnerCount": active_count,
        "horses": horses,
    })

summary = {
    "source": PDF_URL,
    "httpContentType": content_type,
    "pdfBytes": len(pdf_bytes),
    "pageCount": len(document),
    "pageCharacters": [item["characters"] for item in pages],
    "totalExtractedCharacters": len(combined),
    "layoutBlockCount": len(layout_blocks),
    "layoutLineCount": len(layout_lines),
    "raceCount": len(headers),
    "parsedRaceCount": len(parsed_races),
    "parsedHorseRows": sum(len(race["horses"]) for race in parsed_races),
    "validationErrorCount": len(validation_errors),
    "validationErrors": validation_errors,
    "textExtractionSucceeded": bool(combined.strip()),
}
(OUT / "parsed-races.json").write_text(
    json.dumps(parsed_races, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))

if len(document) == 0 or len(combined.strip()) < 1000:
    raise RuntimeError("PDF text extraction did not produce enough usable text")
if len(parsed_races) != 12 or validation_errors:
    raise RuntimeError(
        f"Structured extraction validation failed: {len(validation_errors)} errors"
    )

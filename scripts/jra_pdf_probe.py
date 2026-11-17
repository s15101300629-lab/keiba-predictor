from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

PDF_URL = "https://jra.jp/datafile/seiseki/report/2025/2025-1kyoto1.pdf"
OUT = Path("jra-pdf-probe")
OUT.mkdir(exist_ok=True)

request = urllib.request.Request(
    PDF_URL,
    headers={"User-Agent": "Mozilla/5.0 (compatible; JRAResultResearch/1.0)", "Accept": "application/pdf"},
)
with urllib.request.urlopen(request, timeout=45) as response:
    pdf_bytes = response.read()
    content_type = response.headers.get("Content-Type", "")

if not pdf_bytes.startswith(b"%PDF"):
    raise RuntimeError(f"JRA response is not a PDF: content-type={content_type}, bytes={len(pdf_bytes)}")

try:
    import fitz
except ImportError as exc:
    raise RuntimeError("PyMuPDF is required to inspect the JRA PDF text layer") from exc

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

combined = "\n\n".join(f"===== PAGE {item['page']} =====\n{item['text']}" for item in pages)
(OUT / "sample-extracted.txt").write_text(combined, encoding="utf-8")
(OUT / "layout-blocks.json").write_text(
    json.dumps(layout_blocks, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
(OUT / "layout-lines.json").write_text(
    json.dumps(layout_lines, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

race_headers = re.findall(r"第\s*\d+\s*競走", combined)
horse_like_lines = [
    line.strip() for line in combined.splitlines()
    if re.match(r"^\s*[1-8]\s*\d{1,2}\s+\S+", line)
]
summary = {
    "source": PDF_URL,
    "httpContentType": content_type,
    "pdfBytes": len(pdf_bytes),
    "pageCount": len(document),
    "pageCharacters": [item["characters"] for item in pages],
    "totalExtractedCharacters": len(combined),
    "layoutBlockCount": len(layout_blocks),
    "layoutLineCount": len(layout_lines),
    "raceHeaderMatches": len(race_headers),
    "horseLikeLineCandidates": len(horse_like_lines),
    "sampleHorseLikeLines": horse_like_lines[:20],
    "textExtractionSucceeded": bool(combined.strip()),
}
(OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False))
if len(document) == 0 or len(combined.strip()) < 1000:
    raise RuntimeError("PDF text extraction did not produce enough usable text")

from __future__ import annotations

import hashlib
import json
import re
import urllib.parse
import urllib.request
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path

INDEX_URL = "https://www.jra.go.jp/datafile/seiseki/index.html"
OUT = Path("jra-pdf-index")
OUT.mkdir(exist_ok=True)
USER_AGENT = "Mozilla/5.0 (compatible; JRAResultResearch/1.0)"


class AnchorParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.links = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.current = {"href": href, "text": []}

    def handle_data(self, data):
        if self.current is not None:
            self.current["text"].append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.current is not None:
            self.links.append({
                "href": self.current["href"],
                "text": re.sub(r"\s+", "", "".join(self.current["text"])),
            })
            self.current = None


def fetch_html(url: str) -> tuple[str, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
    )
    with urllib.request.urlopen(request, timeout=45) as response:
        body = response.read()
        content_type = response.headers.get("Content-Type", "")
        charset = response.headers.get_content_charset()
    if not body:
        raise RuntimeError(f"Empty HTML response: {url}")
    if not charset:
        charset_match = re.search(rb"charset=([A-Za-z0-9._-]+)", body[:4096], re.I)
        charset = charset_match.group(1).decode("ascii") if charset_match else "utf-8"
    try:
        html = body.decode(charset)
    except (LookupError, UnicodeDecodeError):
        html = body.decode("utf-8", errors="replace")
    if "\ufffd" in html:
        for fallback_charset in ("cp932", "shift_jis", "euc_jp"):
            try:
                candidate = body.decode(fallback_charset)
            except (LookupError, UnicodeDecodeError):
                continue
            if "\ufffd" not in candidate and "年" in candidate:
                html = candidate
                break
    return html, content_type


def parse_links(html: str) -> list[dict]:
    parser = AnchorParser()
    parser.feed(html)
    return parser.links


index_html, index_content_type = fetch_html(INDEX_URL)
index_links = parse_links(index_html)
year_pages = {}
for link in index_links:
    match = re.fullmatch(r"(20\d{2})年", link["text"])
    if not match:
        continue
    year = int(match.group(1))
    page_url = urllib.parse.urljoin(INDEX_URL, link["href"])
    if "/report/" in urllib.parse.urlparse(page_url).path:
        year_pages[year] = page_url

if not year_pages:
    candidates=[{"text":link["text"],"href":link["href"],"resolved":urllib.parse.urljoin(INDEX_URL,link["href"])} for link in index_links if "20" in link["text"]]
    print(json.dumps({"linkCount":len(index_links),"yearCandidates":candidates},ensure_ascii=False))
    raise RuntimeError("No annual JRA PDF index pages found on the official index")

years = sorted(year_pages)
if years != list(range(years[0], years[-1] + 1)):
    raise RuntimeError(f"Annual JRA PDF pages are not contiguous: {years}")
if len(years) < 10:
    raise RuntimeError(f"Expected at least 10 annual pages, found {years}")

pdf_entries = []
year_summaries = []
html_hashes = {}
for year in years:
    page_url = year_pages[year]
    html, content_type = fetch_html(page_url)
    html_hashes[str(year)] = hashlib.sha256(html.encode("utf-8")).hexdigest()
    links = parse_links(html)
    seen = set()
    year_entries = []
    for link in links:
        absolute_url = urllib.parse.urljoin(page_url, link["href"])
        parsed = urllib.parse.urlparse(absolute_url)
        if not parsed.path.lower().endswith(".pdf"):
            continue
        if parsed.netloc.lower() not in ("jra.go.jp", "www.jra.go.jp"):
            continue
        if f"/report/{year}/" not in parsed.path:
            raise RuntimeError(f"PDF URL year/path mismatch: expected {year}: {absolute_url}")
        if absolute_url in seen:
            continue
        seen.add(absolute_url)
        entry = {
            "year": year,
            "yearPageUrl": page_url,
            "sourceUrl": absolute_url,
            "linkText": link["text"],
        }
        year_entries.append(entry)
    if not year_entries:
        raise RuntimeError(f"No PDF links found on annual page {year}: {page_url}")
    year_entries.sort(key=lambda x: x["sourceUrl"])
    pdf_entries.extend(year_entries)
    year_summaries.append({
        "year": year,
        "yearPageUrl": page_url,
        "pdfCount": len(year_entries),
        "htmlContentType": content_type,
    })

all_urls = [entry["sourceUrl"] for entry in pdf_entries]
if len(all_urls) != len(set(all_urls)):
    raise RuntimeError("Duplicate PDF URLs found across annual pages")

manifest = {
    "schemaVersion": 1,
    "sourceIndexUrl": INDEX_URL,
    "indexContentType": index_content_type,
    "years": years,
    "yearPages": year_summaries,
    "pdfCount": len(pdf_entries),
    "pdfs": pdf_entries,
}
summary = {
    "sourceIndexUrl": INDEX_URL,
    "years": years,
    "yearPageCount": len(year_pages),
    "pdfCount": len(pdf_entries),
    "pdfCountByYear": {str(item["year"]): item["pdfCount"] for item in year_summaries},
    "indexHtmlSha256": hashlib.sha256(index_html.encode("utf-8")).hexdigest(),
    "yearPageHtmlSha256": html_hashes,
}
(OUT / "manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
)
(OUT / "summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False))

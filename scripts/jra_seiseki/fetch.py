"""Discover and download 成績表 PDFs from https://www.jra.go.jp/datafile/seiseki/."""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin

BASE = "https://www.jra.go.jp/datafile/seiseki/"
YEAR_INDEX = BASE + "report/{year}.html"
USER_AGENT = "Mozilla/5.0 (compatible; keiba-predictor-seiseki/1.0)"

# 2026-5tokyo4.pdf = 2026年 第5回 東京 4日目.  発売票数 (…-hyo-…) PDFs are skipped.
RESULT_PDF_RE = re.compile(r"(\d{4})-(\d{1,2})([a-z]+)(\d{1,2})\.pdf$")


@dataclass(frozen=True)
class PdfLink:
    url: str
    name: str  # file stem, e.g. 2026-5tokyo4
    date: str | None  # YYYY-MM-DD when the index page shows it


def http_get(url: str, retries: int = 3) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as res:
                return res.read()
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def list_year(year: int) -> list[PdfLink]:
    url = YEAR_INDEX.format(year=year)
    html = http_get(url).decode("shift_jis", errors="replace")
    links: dict[str, PdfLink] = {}
    current_date = None
    # Walk headings ("11月15日（日曜）") and links in document order.
    # Recent days are grouped under date headings; older ones under venue headings.
    for m in re.finditer(r"<h[2-4][^>]*>([^<]*)|href=\"([^\"]+\.pdf)\"", html):
        if m.group(1) is not None:
            dm = re.match(r"\s*(\d{1,2})月(\d{1,2})日", m.group(1))
            current_date = f"{year}-{int(dm.group(1)):02d}-{int(dm.group(2)):02d}" if dm else None
            continue
        href = m.group(2)
        fm = RESULT_PDF_RE.search(href)
        if not fm or int(fm.group(1)) != year:
            continue
        name = href.rsplit("/", 1)[-1][:-4]
        links.setdefault(name, PdfLink(urljoin(url, href), name, current_date))
    return list(links.values())


def download(link: PdfLink, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = http_get(link.url)
    if not data.startswith(b"%PDF"):
        raise RuntimeError(f"{link.url} did not return a PDF")
    dest.write_bytes(data)
    return dest

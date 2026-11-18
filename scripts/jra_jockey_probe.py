from __future__ import annotations

import json
import re
import fitz
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

BASE = "https://www.jra.go.jp/datafile/meikan/"
OUT = Path("jra-jockey-probe")
OUT.mkdir(exist_ok=True)
UA = "Mozilla/5.0 (compatible; JRAResultResearch/1.0)"


class Links(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.items = []
        self.current = None
        self.all_text = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            href = dict(attrs).get("href")
            if href:
                self.current = {"href": href, "text": []}

    def handle_data(self, data):
        self.all_text.append(data)
        if self.current is not None:
            self.current["text"].append(data)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.current is not None:
            self.items.append({
                "href": self.current["href"],
                "text": re.sub(r"\s+", " ", "".join(self.current["text"])).strip(),
            })
            self.current = None


def get(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as response:
        raw = response.read()
        charset = response.headers.get_content_charset()
    if not charset:
        meta = re.search(rb"charset\s*=\s*['\"]?([A-Za-z0-9._-]+)", raw[:4096], re.I)
        charset = meta.group(1).decode("ascii") if meta else "utf-8"
    try:
        decoded = raw.decode(charset)
    except (LookupError, UnicodeDecodeError):
        decoded = raw.decode("utf-8", errors="replace")
    if "\ufffd" in decoded:
        for fallback in ("cp932", "shift_jis", "euc_jp"):
            try:
                candidate = raw.decode(fallback)
            except (LookupError, UnicodeDecodeError):
                continue
            if "\ufffd" not in candidate and "騎手" in candidate:
                decoded = candidate
                break
    return decoded


index_url = urllib.parse.urljoin(BASE, "jockey.html")
index_html = get(index_url)
parser = Links()
parser.feed(index_html)
groups = [x for x in parser.items if "行" in x["text"] and len(x["text"]) <= 4]
group_results = []
active_names = set()
for item in groups:
    url = urllib.parse.urljoin(index_url, item["href"])
    html = get(url)
    lp = Links()
    lp.feed(html)
    names = []
    for link in lp.items:
        text = link["text"].replace("★", "").strip()
        compact = re.sub(r"\s+", "", text)
        if 2 <= len(compact) <= 8 and re.search(r"[一-龯ぁ-んァ-ヶ]", compact):
            if not re.search(r"(プロフィール|騎手|一覧|戻る|次へ|前へ)", compact):
                names.append(text)
                active_names.add(compact)
    group_results.append({"group": item["text"], "url": url, "candidateNames": names})

retired_url = urllib.parse.urljoin(BASE, "jretirement.html")
retired_html = get(retired_url)
rp = Links()
rp.feed(retired_html)
retired_names = set()
retired_samples = []
retired_full_text = "".join(rp.all_text)
for match in re.finditer(r"([^\n（）()]{2,20})[（(](20\d{2})年\d+月\d+日引退[）)]", retired_full_text):
    name = re.sub(r"\s+", "", match.group(1).replace("★", ""))
    name = re.sub(r"^(?:美浦|栗東|年度)$", "", name)
    if name:
        retired_names.add(name)
        retired_samples.append({"name": name, "year": int(match.group(2))})

result = {
    "indexUrl": index_url,
    "groupLinks": [{"text": x["text"], "href": x["href"]} for x in groups],
    "allIndexLinks": parser.items,
    "indexText": "".join(parser.all_text),
    "groupResults": group_results,
    "activeCandidateCount": len(active_names),
    "activeCandidates": sorted(active_names),
    "retiredUrl": retired_url,
    "retiredCandidateCount": len(retired_names),
    "retiredCandidates": sorted(retired_names),
    "combinedUniqueCount": len(active_names | retired_names),
}
(OUT / "jockey-index.html").write_text(index_html, encoding="utf-8")
(OUT / "jockey-retired.html").write_text(retired_html, encoding="utf-8")
agent_results = {}
for region in ("miho", "rittou"):
    agent_url = f"https://jra.jp/datafile/meikan/pdf/agent_{region}.pdf"
    req = urllib.request.Request(agent_url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=45) as response:
        pdf_bytes = response.read()
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_data = []
    for page_number, page in enumerate(pdf, start=1):
        page_text = page.get_text("text", sort=True)
        (OUT / f"agent_{region}.txt").write_text(page_text, encoding="utf-8")
        page_data.append({
            "page": page_number,
            "width": page.rect.width,
            "height": page.rect.height,
            "words": [
                {"x0": round(w[0], 1), "y0": round(w[1], 1), "x1": round(w[2], 1), "y1": round(w[3], 1), "text": w[4]}
                for w in page.get_text("words")
            ],
        })
    agent_results[region] = {"url": agent_url, "bytes": len(pdf_bytes), "pages": page_data}
(OUT / "agent-layout.json").write_text(json.dumps(agent_results, ensure_ascii=False, indent=2), encoding="utf-8")
active_pdf_names = set()
for region, data in agent_results.items():
    for page in data["pages"]:
        for word in page["words"]:
            x = word["x0"]
            y = word["y0"]
            raw_name = word["text"]
            if y < 135 or raw_name == "騎手名":
                continue
            if region == "miho":
                is_jockey_column = (0 < x < 150) or (290 < x < 400)
            else:
                is_jockey_column = (0 < x < 180) or (380 < x < 510)
            if not is_jockey_column:
                continue
            name = re.sub(r"\s+", "", raw_name.replace("★", "").replace("☆", "")).replace("．", ".")
            if len(name) >= 2 and "若手騎手" not in name and re.search(r"[一-龯ぁ-んァ-ヶ]", name):
                active_pdf_names.add(name)

jockey_names = active_pdf_names | retired_names
if len(active_pdf_names) < 100 or len(retired_names) < 100 or len(jockey_names) < 300:
    raise RuntimeError(
        f"Official JRA jockey dictionary unexpectedly small: active={len(active_pdf_names)}, "
        f"retired={len(retired_names)}, combined={len(jockey_names)}"
    )
data_dir = Path("jra-pdf-data")
data_dir.mkdir(exist_ok=True)
dictionary = {
    "schemaVersion": 1,
    "source": "JRA official jockey agent PDFs and retired jockey index",
    "sources": [
        "https://jra.jp/datafile/meikan/pdf/agent_miho.pdf",
        "https://jra.jp/datafile/meikan/pdf/agent_rittou.pdf",
        retired_url,
    ],
    "activeNames": sorted(active_pdf_names),
    "retiredNames": sorted(retired_names),
    "names": sorted(jockey_names),
}
(data_dir / "jockey-names.json").write_text(
    json.dumps(dictionary, ensure_ascii=False, indent=2), encoding="utf-8"
)
result["activeCandidateCount"] = len(active_pdf_names)
result["activeCandidates"] = sorted(active_pdf_names)
result["combinedUniqueCount"] = len(jockey_names)
result["dictionaryPath"] = str(data_dir / "jockey-names.json")
result["agentPdfs"] = {
    key: {"url": value["url"], "bytes": value["bytes"], "pageCount": len(value["pages"])}
    for key, value in agent_results.items()
}
(OUT / "jockey-dictionary-probe.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps({
    "groupLinks": result["groupLinks"],
    "activeCandidateCount": len(active_names),
    "retiredCandidateCount": len(retired_names),
    "combinedUniqueCount": result["combinedUniqueCount"],
    "activeSample": sorted(active_names)[:20],
    "retiredSample": sorted(retired_names)[:20],
}, ensure_ascii=False))

from __future__ import annotations

import json
import re
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

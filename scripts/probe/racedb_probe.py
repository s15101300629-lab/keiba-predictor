"""racedb.com の中身を確かめる（依頼者の案。このセッションからは接続できないので GitHub Actions で数ページだけ取る）.
頭文字 O の一覧から Ombudsman の馬のページ、そこから最初のレースのページを取り、文字だけにして data/ml/racedb_probe.txt に書く."""
import html, re, time, urllib.request
from pathlib import Path
UA = "keiba-predictor (personal research)"
OUT = Path(__file__).resolve().parents[2] / "data/ml/racedb_probe.txt"
def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
            b = r.read()
        for enc in ("utf-8", "cp932", "euc-jp"):
            try: return b.decode(enc)
            except UnicodeDecodeError: pass
        return b.decode("utf-8", "replace")
    except Exception as e:
        return f"ERROR {e}"
def text(s): return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style).*?</\1>", "", s, flags=re.S))))
def absu(l): return l if l.startswith("http") else "https://racedb.com" + (l if l.startswith("/") else "/" + l)
out = []
o = get("https://racedb.com/name/o.html"); time.sleep(3)
m = re.search(r'href="([^"]+)"[^>]*>\s*Ombudsman', o)
out.append("OMBUDSMAN LINK: " + (m.group(1) if m else "なし"))
if m:
    h = get(absu(m.group(1))); time.sleep(3)
    out += ["===== " + absu(m.group(1)), text(h)[:6000], "LINKS: " + " ".join(re.findall(r'href="([^"]+)"', h)[:150])]
    races = [l for l in re.findall(r'href="([^"]+)"', h) if not l.startswith(("http", "#", "/name", "/p/", "/a/")) or "/r" in l]
    out.append("RACE LINK CANDIDATES: " + " ".join(races[:20]))
    for l in races[:2]:
        r = get(absu(l)); time.sleep(3)
        out += ["===== " + absu(l), text(r)[:5000]]
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

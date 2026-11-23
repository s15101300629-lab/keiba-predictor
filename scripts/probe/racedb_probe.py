"""racedb.com の中身を確かめる（依頼者の案。このセッションからは接続できないので GitHub Actions で数ページだけ取る）.
robots.txt・名前の一覧ページ・そこからたどれる馬のページとレースのページを1つずつ取り、文字だけにして data/ml/racedb_probe.txt に書く."""
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
out = []
for url in ["https://racedb.com/robots.txt", "https://racedb.com/name/index.html"]:
    s = get(url); time.sleep(3)
    out += [f"===== {url}", text(s)[:3000], "LINKS: " + " ".join(re.findall(r'href="([^"]+)"', s)[:120])]
    if url.endswith("index.html"):
        links = re.findall(r'href="([^"]+)"', s)
        horse = next((l for l in links if re.search(r"horse|uma|/name/[^i]", l)), None)
        if horse:
            u = horse if horse.startswith("http") else "https://racedb.com" + ("" if horse.startswith("/") else "/name/") + horse
            h = get(u); time.sleep(3)
            out += [f"===== {u}", text(h)[:4000], "LINKS: " + " ".join(re.findall(r'href="([^"]+)"', h)[:120])]
            race = next((l for l in re.findall(r'href="([^"]+)"', h) if re.search(r"race", l)), None)
            if race:
                u2 = race if race.startswith("http") else "https://racedb.com" + ("" if race.startswith("/") else "/") + race
                r2 = get(u2)
                out += [f"===== {u2}", text(r2)[:4000]]
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

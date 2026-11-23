"""racedb.com のレースのページを確かめる（3回目。英チャンピオンS 2026 と ジャパンC 2025）."""
import html, re, time, urllib.request
from pathlib import Path
UA = "keiba-predictor (personal research)"
OUT = Path(__file__).resolve().parents[2] / "data/ml/racedb_probe.txt"
def get(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": UA}), timeout=30) as r:
            return r.read().decode("utf-8", "replace")
    except Exception as e:
        return f"ERROR {e}"
def text(s): return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", re.sub(r"<(script|style).*?</\1>", "", s, flags=re.S))))
out = []
for u in ["https://racedb.com/k/30032026101706.html", "https://racedb.com/p/063999.html"]:
    s = get(u); time.sleep(3)
    out += ["===== " + u, text(s)[:7000]]
    if u.endswith("063999.html"):
        out.append("TABLE HTML: " + (re.search(r"主な競走結果.*?</table>", s, re.S) or [""])[0][:3000])
OUT.write_text("\n".join(out) + "\n", encoding="utf-8")

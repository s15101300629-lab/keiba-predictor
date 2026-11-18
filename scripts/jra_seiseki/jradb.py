"""JRA「過去のレース結果」「競走馬情報」(https://www.jra.go.jp/JRADB/) のクライアント.

JRADB の各ページは ``cname=<種別+キー>/<2桁チェックサム>`` を POST して開く。
チェックサムはサイト側が発行したものしか使えないため、ページ内のリンクを辿って取得する:

    過去レース検索トップ (月ごとのチェックサム表 objParam を含む)
      -> 月の開催一覧  pw01skl10{YYYYMM}
      -> 開催日        pw01srl10{場}{年}{回}{日}{YYYYMMDD}
      -> レース結果    pw01sde10{場}{年}{回}{日}{R}{YYYYMMDD}
      -> 競走馬情報    pw01dud10{馬ID 10桁}   (馬IDは netkeiba の horse_id と同じ)
"""

from __future__ import annotations

import html as htmllib
import re
import threading
import time
import urllib.parse
import urllib.request

BASE = "https://www.jra.go.jp/JRADB/"
USER_AGENT = "Mozilla/5.0 (compatible; keiba-predictor-seiseki/1.0)"
INDEX_CNAME = "pw01skl00999999/B3"

DAY_RE = re.compile(r"pw01srl10(\d{2})(\d{4})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}")
RACE_RE = re.compile(r"pw01sde10(\d{2})(\d{4})(\d{2})(\d{2})(\d{2})(\d{8})/[0-9A-F]{2}")
HORSE_LINK_RE = re.compile(r"pw01dud10(\d{10})/[0-9A-F]{2}")


class JraDb:
    """Polite sequential-ish HTTP client (shared rate limit across threads)."""

    def __init__(self, wait: float = 0.5, retries: int = 3, timeout: float = 30):
        self.wait = wait
        self.retries = retries
        self.timeout = timeout
        self.requests = 0
        self._lock = threading.Lock()
        self._next_at = 0.0
        self._month_params: dict[str, str] | None = None

    def _throttle(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next_at - now
            self._next_at = max(now, self._next_at) + self.wait
        if delay > 0:
            time.sleep(delay)

    def get(self, page: str, cname: str) -> str:
        """POST ``cname`` to ``/JRADB/{page}`` and return the decoded HTML."""
        data = urllib.parse.urlencode({"cname": cname}).encode()
        last: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            try:
                req = urllib.request.Request(BASE + page, data=data, headers={"User-Agent": USER_AGENT})
                with urllib.request.urlopen(req, timeout=self.timeout) as res:
                    body = res.read()
                self.requests += 1
                text = body.decode("cp932", errors="replace")
                if "error013" in text or len(body) < 2000:
                    raise RuntimeError(f"JRADB returned an error page for {cname}")
                return text
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"JRADB {page} {cname} failed: {last}")

    # --- navigation -------------------------------------------------------------

    def month_cname(self, year: int, month: int) -> str:
        if self._month_params is None:
            top = self.get("accessS.html", INDEX_CNAME)
            self._month_params = dict(re.findall(r'objParam\["(\d{4})"\]="([0-9A-F]{2})"', top))
            cur = re.search(r'var yearMonth = "(\d{6})"', top)
            self._current_month = cur.group(1) if cur else "999999"
        ym = f"{year:04d}{month:02d}"
        chk = self._month_params.get(ym[2:])
        if not chk:
            raise RuntimeError(f"no JRADB month link for {ym}")
        kind = "pw01skl00" if ym >= self._current_month else "pw01skl10"
        return f"{kind}{ym}/{chk}"

    def month_days(self, year: int, month: int) -> dict[tuple[str, int, int], str]:
        """{(venue_code, kai, day): cname} for every race day listed in the month."""
        html = self.get("accessS.html", self.month_cname(year, month))
        out = {}
        for m in DAY_RE.finditer(html):
            venue, y, kai, day, _date = m.groups()
            if int(y) == year:
                out[(venue, int(kai), int(day))] = m.group(0)
        return out

    def day_races(self, day_cname: str) -> dict[str, str]:
        """{race_id: cname} for the races of one race day."""
        html = self.get("accessS.html", day_cname)
        out = {}
        for m in RACE_RE.finditer(html):
            venue, y, kai, day, race, _date = m.groups()
            out[f"{y}{venue}{kai}{day}{race}"] = m.group(0)
        return out

    def race_horses(self, race_cname: str) -> list[dict]:
        return parse_race_horses(self.get("accessS.html", race_cname))

    def horse(self, horse_id: str, cname: str) -> dict:
        return parse_horse(self.get("accessU.html", cname), horse_id)


# --- parsers ---------------------------------------------------------------------


def _text(fragment: str) -> str:
    fragment = re.sub(r"<span class=\"sanku\">.*?</span>", "", fragment, flags=re.S)
    t = htmllib.unescape(re.sub(r"<[^>]+>", "", fragment))
    return re.sub(r"\s+", " ", t.replace("　", " ")).strip()


def parse_race_horses(html: str) -> list[dict]:
    """Rows of a race result page: umaban, horse name, horse_id and its page cname."""
    rows = []
    for tr in re.findall(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.S):
        num = re.search(r'<td class="num">\s*(\d+)\s*</td>', tr)
        link = re.search(r'<a href="/JRADB/accessU\.html\?CNAME=(pw01dud10(\d{10})/[0-9A-F]{2})">(.*?)</a>', tr, flags=re.S)
        if not num or not link:
            continue
        rows.append({
            "umaban": int(num.group(1)),
            "horse": _text(link.group(3)),
            "horse_id": link.group(2),
            "cname": link.group(1),
        })
    return rows


def parse_horse(html: str, horse_id: str) -> dict:
    """競走馬情報ページのプロフィール（父・母・母の父・母の母 など）."""
    fields: dict[str, str] = {}
    links: dict[str, str] = {}
    prof = html[html.find('<div class="profile'):]
    prof = prof[: prof.find('<div class="prize')] if '<div class="prize' in prof else prof
    for dt, dd in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", prof, flags=re.S):
        key = _text(dt)
        fields[key] = _text(dd)
        m = HORSE_LINK_RE.search(dd)
        if m:
            links[key] = m.group(1)

    name = name_en = country = retired = None
    h1 = re.search(r"<h1>((?:(?!</h1>).)*?<span class=\"opt\">競走馬情報</span>.*?)</h1>", html, flags=re.S)
    if h1:
        inner = h1.group(1)
        m = re.search(r'<span class="opt">競走馬情報</span>(.*?)<span class="name_en">(.*?)</span>', inner, flags=re.S)
        if m:
            name = _text(m.group(1))
            en = _text(m.group(2))
            cm = re.match(r"(.*?)[（(]([^）)]+)[）)]\s*$", en)
            name_en, country = (cm.group(1).strip(), cm.group(2)) if cm else (en or None, None)
        rm = re.search(r"抹消年月日\s*(\d{4})年(\d{1,2})月(\d{1,2})日", _text(inner))
        if rm:
            retired = f"{rm.group(1)}-{int(rm.group(2)):02d}-{int(rm.group(3)):02d}"

    birth = None
    bm = re.match(r"(\d{4})年(\d{1,2})月(\d{1,2})日", fields.get("生年月日", ""))
    if bm:
        birth = f"{bm.group(1)}-{int(bm.group(2)):02d}-{int(bm.group(3)):02d}"

    def f(key: str) -> str | None:
        v = fields.get(key)
        return v or None

    if not (f("父") or f("母")):
        raise RuntimeError(f"no pedigree found on the page of horse {horse_id}")
    return {
        "horse_id": horse_id,
        "name": name,
        "name_en": name_en,
        "country": country,
        "sex": f("性別"),
        "color": f("毛色"),
        "birth_date": birth,
        "sire": f("父"),
        "sire_id": links.get("父"),
        "dam": f("母"),
        "dam_id": links.get("母"),
        "damsire": f("母の父"),
        "damdam": f("母の母"),
        "breeder": f("生産牧場"),
        "region": f("産地"),
        "market": f("取引市場"),
        "retired_date": retired,
    }
